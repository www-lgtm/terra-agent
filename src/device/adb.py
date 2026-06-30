"""Low-level ADB operations: tap, swipe, screenshot, shell commands.

Supports multiple devices via a serial-keyed device pool + thread-local binding.
"""

from __future__ import annotations

import logging
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from io import BytesIO
from pathlib import Path

from PIL import Image

from config.settings import config

logger = logging.getLogger(__name__)

# Persistent single-worker executor for screencap — avoids spawning a new
# daemon thread on every screenshot (~2000 threads in a long task).
_screencap_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="adb-screencap")

SCREENSHOTS_DIR = Path(config.DATA_DIR) / "screenshots"

def _get_screenshot_path(serial: str) -> Path:
    """Return per-device screenshot path to avoid cross-device contamination."""
    return SCREENSHOTS_DIR / serial.replace(":", "_") / "current.png"


class ADBDevice:
    """Wraps ADB commands for a single device."""

    def __init__(self, serial: str, adb_path: str = "adb") -> None:
        self.serial = serial
        self.adb_path = adb_path
        self._heartbeat_ok = True
        self._consecutive_screencap_failures = 0
        self._lock = threading.Lock()

    def _run(self, *args: str, timeout: float = 10.0) -> tuple[int, str, str]:
        cmd = [self.adb_path, "-s", self.serial, *args]
        logger.debug("ADB: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr

    def shell(self, *args: str) -> str:
        code, out, err = self._run("shell", *args)
        if code != 0:
            raise RuntimeError(f"ADB shell failed: {err.strip()}")
        return out.strip()

    def tap(self, x: int, y: int) -> None:
        from src.utils.circuit_breaker import adb_breaker, CircuitOpenError
        noise_x = random.randint(-config.adb.tap_noise_px, config.adb.tap_noise_px)
        noise_y = random.randint(-config.adb.tap_noise_px, config.adb.tap_noise_px)
        try:
            adb_breaker.call(lambda: self._run("shell", "input", "tap", str(x + noise_x), str(y + noise_y)))
        except CircuitOpenError:
            # Let the ModuleNotFoundError propagate so the agent loop sees it
            raise RuntimeError("ADB circuit breaker open — device is likely offline") from None

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self._run("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms))

    def screencap(self, timeout: float = 15.0) -> bytes:
        """Capture screenshot via exec-out screencap -p. Returns PNG bytes.

        Uses a persistent single-worker thread pool instead of spawning a new
        daemon thread per call — saving ~2000 thread creations in a long task.
        The pool's single worker serializes screencap calls, which is correct:
        ADB devices can't handle concurrent screencap anyway.

        Raises RuntimeError on any failure.
        """
        cmd = [self.adb_path, "-s", self.serial, "exec-out", "screencap", "-p"]

        def _capture() -> bytes:
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
                if proc.returncode != 0:
                    raise RuntimeError(f"screencap failed: {proc.stderr.decode()}")
                return proc.stdout
            except subprocess.TimeoutExpired:
                logger.warning("screencap timed out after %.0fs — killing ADB subprocess", timeout)
                try:
                    subprocess.run(
                        [self.adb_path, "-s", self.serial, "exec-out", "kill", "-9", "screencap"],
                        capture_output=True, timeout=3.0,
                    )
                except Exception:
                    pass
                raise RuntimeError(
                    f"screencap timed out after {timeout:.0f}s — emulator may be frozen"
                )

        future = _screencap_executor.submit(_capture)
        try:
            return future.result(timeout=timeout + 5.0)
        except FutureTimeout:
            logger.error(
                "screencap watchdog timeout after %.0fs — ADB may be dead", timeout + 5.0)
            self._heartbeat_ok = False
            # Don't cancel the future — it may still complete but we've
            # already decided the device is unhealthy.
            raise RuntimeError(
                f"screencap hung after {timeout + 5.0:.0f}s (watchdog). "
                "ADB device marked unhealthy. Restart emulator or bot."
            )

    def save_screenshot(self) -> Path:
        data = self.screencap()
        path = _get_screenshot_path(self.serial)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def get_screenshot_image(self) -> Image.Image:
        """Capture screenshot, save to per-device path, and return PIL Image.

        On consecutive failures, marks the device as unhealthy so callers
        can skip further attempts instead of hammering a dead connection.
        """
        try:
            data = self.screencap()
            self._consecutive_screencap_failures = 0
            path = _get_screenshot_path(self.serial)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return Image.open(BytesIO(data))
        except Exception:
            self._consecutive_screencap_failures += 1
            if self._consecutive_screencap_failures >= 2:
                self._heartbeat_ok = False
                logger.error(
                    "ADB device %s marked unhealthy after %d consecutive screencap failures",
                    self.serial, self._consecutive_screencap_failures,
                )
            raise

    def press_back(self) -> None:
        self._run("shell", "input", "keyevent", "4")

    def press_home(self) -> None:
        self._run("shell", "input", "keyevent", "3")

    def get_screen_size(self) -> tuple[int, int]:
        """Return the actual rendering screen size (width, height).

        `adb shell wm size` reports the physical display's native orientation
        (e.g. 1080x1920 on a portrait phone). When the emulator/app renders in
        landscape (1920x1080), the physical dimensions are swapped relative to
        the actual screenshot pixels. Cross-validate against a real screenshot
        and return the screenshot dimensions when they differ.
        """
        out = self.shell("wm", "size")
        parts = out.strip().split()[-1].split("x")
        wm_w, wm_h = int(parts[0]), int(parts[1])
        try:
            img = self.get_screenshot_image()
            img_w, img_h = img.size
            wm_is_landscape = wm_w > wm_h
            img_is_landscape = img_w > img_h
            if wm_is_landscape != img_is_landscape:
                logger.debug(
                    "get_screen_size: wm=%dx%d screenshot=%dx%d — "
                    "using screenshot (orientation mismatch)",
                    wm_w, wm_h, img_w, img_h,
                )
                return img_w, img_h
        except Exception:
            pass  # Fall through to wm size on screenshot failure
        return wm_w, wm_h

    def heartbeat(self) -> bool:
        """Check if device is connected and responsive.

        P3: also tests screenshot capture — catches GPU/SurfaceFlinger hangs
        that pass shell echo but prevent the agent from working.
        """
        try:
            code, _, _ = self._run("shell", "echo", "ping", timeout=3.0)
            if code != 0:
                self._heartbeat_ok = False
                return False
            # Also verify screenshot works (binary subprocess — _run is text-mode)
            import subprocess as _sp
            proc = _sp.run(
                [self.adb_path, "-s", self.serial, "exec-out", "screencap", "-p"],
                capture_output=True, timeout=5.0,
            )
            screenshot_ok = (
                proc.returncode == 0
                and len(proc.stdout) > 8
                and proc.stdout[:4] == b'\x89PNG'
            )
            self._heartbeat_ok = screenshot_ok
            return screenshot_ok
        except Exception:
            self._heartbeat_ok = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._heartbeat_ok


# ---- Device pool + thread-local binding ----

_adb_devices: dict[str, ADBDevice] = {}
_thread_device = threading.local()


def _get_thread_serial() -> str | None:
    return getattr(_thread_device, "serial", None)


def bind_device_to_thread(serial: str) -> None:
    """Bind a device serial to the current thread.

    Called by TerraAgent.__init__ so all tool calls from this agent's thread
    automatically use the correct device via get_adb().
    """
    _thread_device.serial = serial


def get_adb(serial: str | None = None) -> ADBDevice:
    """Get the ADB device for the current context.

    Resolution order:
    1. Explicit `serial` argument
    2. Thread-local binding (set by TerraAgent on init)
    3. Raises RuntimeError if neither is available — NEVER silently
       falls back to an arbitrary device.  Silent fallback caused
       cross-device contamination when two agents ran simultaneously.

    For concierge/startup code that genuinely needs "any available device",
    use get_any_adb() instead — it explicitly opts into fallback.
    """
    key = serial or _get_thread_serial()
    if key and key in _adb_devices:
        return _adb_devices[key]

    # No thread-local binding AND no explicit serial in multi-device setup.
    # This is a bug — every agent thread must bind before using ADB.
    if _adb_devices:
        raise RuntimeError(
            "ADB device not bound to current thread. "
            "Call bind_device_to_thread(serial) before using ADB from this thread. "
            f"Available devices: {list(_adb_devices.keys())}"
        )
    raise RuntimeError("No ADB device available. Call init_adb(serial) first.")


def get_any_adb() -> ADBDevice:
    """Get any available ADB device — only for concierge/startup code.

    Agent threads MUST use get_adb() with proper thread-local binding.
    This function exists for code paths that genuinely don't know which
    device to use (health checks, emulator inventory scans).
    """
    if _adb_devices:
        return next(iter(_adb_devices.values()))
    raise RuntimeError("No ADB device available.")


def init_adb(serial: str) -> ADBDevice:
    """Initialize and register an ADB device. Binds to current thread.

    If a device with the same serial is already registered, reuses it.
    """
    if serial not in _adb_devices:
        adb_path = config.adb.path
        device = ADBDevice(serial, adb_path=adb_path)
        if not device.heartbeat():
            raise ConnectionError(f"ADB device {serial} is not reachable")
        _adb_devices[serial] = device
        logger.info("ADB device registered: %s", serial)

    # Bind to current thread
    bind_device_to_thread(serial)

    # Start health monitor (one per process, monitors first-registered device)
    from src.device.emulator import emulator_manager
    if not emulator_manager.is_monitoring:
        emulator_manager.start_health_monitor(serial)

    return _adb_devices[serial]


def list_devices() -> dict[str, ADBDevice]:
    """Return all registered devices. {serial: ADBDevice}."""
    return dict(_adb_devices)


def remove_device(serial: str) -> None:
    """Remove a device from the pool (e.g. on disconnect)."""
    _adb_devices.pop(serial, None)


# Backward-compat: the module-level alias returns the first device or raises.
# Prefer get_adb() instead.
adb_device: ADBDevice | None = None  # stale, kept for compatibility
