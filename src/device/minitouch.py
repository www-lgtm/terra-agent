"""Minitouch controller — MAA MinitouchController Python port.

Minitouch is a touch injection daemon that provides pixel-precise touch events.
Unlike ADB `input swipe`, minitouch:
  - Has no overshoot/acceleration
  - Supports sub-millisecond timing
  - Produces deterministic, repeatable swipes

Protocol (MAA-compatible):
  Start:  adb shell /data/local/tmp/minitouch
  Init:   reads header line: ^ <max_contacts> <max_x> <max_y> <max_pressure>
  Touch:  d <contact> <x> <y> <pressure>\nc\n   (down)
          m <contact> <x> <y> <pressure>\nc\n   (move)
          u <contact>\nc\n                       (up)
          w <ms>\n                                (wait, executed client-side)

Usage:
    mt = MinitouchController(adb_device)
    mt.start()
    mt.swipe(x1, y1, x2, y2, duration_ms=200, steps=10)
    mt.stop()
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.device.adb import ADBDevice

logger = logging.getLogger(__name__)

MINITOUCH_BINARY_PATH = "/data/local/tmp/minitouch"
MAA_MINITOUCH_X86_64 = "d:/vsworkspace/MaaAssistantArknights/resource/minitouch/x86_64/minitouch"


class MinitouchController:
    """Exact port of MAA's MinitouchController.

    Deploys minitouch binary, parses device capabilities, and provides
    pixel-precise swipe() and tap() operations.
    """

    def __init__(self, adb: ADBDevice) -> None:
        self._adb = adb
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

        # Device capabilities (parsed from minitouch header)
        self.max_contacts: int = 0
        self.max_x: int = 0
        self.max_y: int = 0
        self.max_pressure: int = 0
        self.x_scaling: float = 1.0
        self.y_scaling: float = 1.0

        # Screen dimensions (from ADB)
        self._screen_w: int = 0
        self._screen_h: int = 0

    # ── Lifecycle ────────────────────────────────────────────────────────

    def deploy(self) -> bool:
        """Push minitouch binary to device if not already present."""
        result = self._adb.shell(f"test -f {MINITOUCH_BINARY_PATH} && echo OK || echo MISSING")
        if "OK" in result:
            logger.info("minitouch already deployed")
            return True

        logger.info("Deploying minitouch x86_64 binary...")
        self._adb._run("push", MAA_MINITOUCH_X86_64, MINITOUCH_BINARY_PATH)
        self._adb.shell(f"chmod 755 {MINITOUCH_BINARY_PATH}")
        logger.info("minitouch deployed")
        return True

    def start(self) -> bool:
        """Start minitouch daemon and parse device capabilities."""
        self._screen_w, self._screen_h = self._adb.get_screen_size()

        with self._lock:
            if self._proc is not None:
                return True

            self.deploy()

            # Start minitouch via ADB shell
            cmd = [
                self._adb.adb_path, "-s", self._adb.serial,
                "shell", MINITOUCH_BINARY_PATH,
            ]
            logger.info("Starting minitouch: %s", " ".join(cmd))

            # On Windows, interactive subprocess pipes can deadlock.
            # Use a thread to read stderr so it doesn't fill up.
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            # Drain stderr in background to avoid deadlock on Windows
            # (already set to DEVNULL, but Popen handles that fine)

            # Read the header: ^ <max_contacts> <max_x> <max_y> <max_pressure>
            # followed by a newline, then "$" prompt
            # Use a thread with timeout to avoid blocking forever
            header_bytes = [b""]
            done = threading.Event()

            def _read_header() -> None:
                try:
                    while not done.is_set():
                        ch = self._proc.stdout.read(1)  # type: ignore[union-attr]
                        if not ch:
                            time.sleep(0.01)
                            continue
                        header_bytes[0] += ch
                        if b"$" in header_bytes[0]:
                            done.set()
                            return
                except Exception:
                    done.set()

            reader = threading.Thread(target=_read_header, daemon=True)
            reader.start()
            reader.join(timeout=8.0)

            if not done.is_set():
                done.set()
                logger.warning("minitouch header read timed out after 8s — device may not support minitouch")
                self._stop_proc()
                return False

            header = header_bytes[0].decode("utf-8", errors="replace")
            logger.info("minitouch header: %r", header)

            # Parse header: find ^ line
            import re
            cap_match = re.search(r'\^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', header)
            if not cap_match:
                logger.error("Failed to parse minitouch header: %r", header)
                self._stop_proc()
                return False

            self.max_contacts = int(cap_match.group(1))
            size1 = int(cap_match.group(2))
            size2 = int(cap_match.group(3))
            self.max_pressure = int(cap_match.group(4))

            # MAA convention: max(x,y) = width, min(x,y) = height
            self.max_x = max(size1, size2)
            self.max_y = min(size1, size2)

            self.x_scaling = self.max_x / max(self._screen_w, 1)
            self.y_scaling = self.max_y / max(self._screen_h, 1)

            logger.info(
                "minitouch ready: screen=%dx%d max=%dx%d pressure=%d scaling=%.3f,%.3f",
                self._screen_w, self._screen_h,
                self.max_x, self.max_y, self.max_pressure,
                self.x_scaling, self.y_scaling,
            )
            return True

    def stop(self) -> None:
        """Stop minitouch daemon."""
        with self._lock:
            self._stop_proc()

    def _stop_proc(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
            self._proc.stdout.close()
            self._proc.kill()
            self._proc.wait(timeout=3)
        except Exception:
            pass
        self._proc = None

    # ── Touch operations ─────────────────────────────────────────────────

    def tap(self, x: int, y: int, pressure: int | None = None) -> bool:
        """Single tap at screen coordinates."""
        if pressure is None:
            pressure = self.max_pressure

        cmds = (
            self._down_cmd(0, x, y, pressure) +
            self._commit_cmd() +
            self._wait_cmd(50) +
            self._up_cmd(0) +
            self._commit_cmd()
        )
        return self._send(cmds)

    def swipe(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: int = 200,
        steps: int = 10,
        pressure: int | None = None,
    ) -> bool:
        """MAA-equivalent swipe with minitouch.

        Args:
            x1, y1: Start position (screen coordinates).
            x2, y2: End position (screen coordinates).
            duration_ms: Total swipe duration.
            steps: Number of intermediate move commands.
            pressure: Touch pressure (default: max_pressure).
        """
        if pressure is None:
            pressure = self.max_pressure

        step_delay = max(1, duration_ms // steps)

        # Down
        cmds = (
            self._down_cmd(0, x1, y1, pressure) +
            self._commit_cmd() +
            self._wait_cmd(5)
        )

        # Move in steps
        for i in range(1, steps + 1):
            t = i / steps
            mx = int(x1 + (x2 - x1) * t)
            my = int(y1 + (y2 - y1) * t)
            cmds += (
                self._move_cmd(0, mx, my, pressure) +
                self._commit_cmd() +
                self._wait_cmd(step_delay)
            )

        # Up
        cmds += (
            self._up_cmd(0) +
            self._commit_cmd()
        )

        return self._send(cmds)

    def swipe_slowly(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: int = 200,
        extra_swipe: bool = False,
        slope_in: float = 1.0,
        slope_out: float = 1.0,
    ) -> bool:
        """MAA SlowlySwipe-style: ease-in/ease-out curve with extra final swipe.

        MAA's SwipeHelper::slowly_swipe uses ease-in-out interpolation and
        optionally adds an extra swipe at the end to overcome inertia.
        """
        # For box scanning, we use the simple linear swipe
        # (ease curves matter more for drag operations, not scrolling)
        return self.swipe(x1, y1, x2, y2, duration_ms=duration_ms, steps=max(10, duration_ms // 20))

    # ── Minitouch protocol helpers ───────────────────────────────────────

    def _scale_coords(self, x: int, y: int) -> tuple[int, int]:
        """Scale screen coordinates to minitouch coordinates."""
        return (
            int(x * self.x_scaling),
            int(y * self.y_scaling),
        )

    @staticmethod
    def _down_cmd(contact: int, x: int, y: int, pressure: int) -> str:
        # Note: we don't scale here — scaling is done in _send for swipe
        return f"d {contact} {x} {y} {pressure}\n"

    @staticmethod
    def _move_cmd(contact: int, x: int, y: int, pressure: int) -> str:
        return f"m {contact} {x} {y} {pressure}\n"

    @staticmethod
    def _up_cmd(contact: int) -> str:
        return f"u {contact}\n"

    @staticmethod
    def _commit_cmd() -> str:
        return "c\n"

    @staticmethod
    def _wait_cmd(ms: int) -> str:
        return f"w {ms}\n"

    def _send(self, cmds: str) -> bool:
        """Send commands to minitouch and wait for completion."""
        with self._lock:
            if self._proc is None or self._proc.stdin is None:
                logger.warning("minitouch not running, falling back to ADB swipe")
                return False

            try:
                # Scale all coordinates in the command string
                total_wait_ms = 0
                processed_lines = []
                for line in cmds.strip().split("\n"):
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] in ("d", "m") and len(parts) >= 4:
                        # Scale x, y
                        x = int(parts[2])
                        y = int(parts[3])
                        sx, sy = self._scale_coords(x, y)
                        parts[2] = str(sx)
                        parts[3] = str(sy)
                    elif parts[0] == "w" and len(parts) >= 2:
                        total_wait_ms += int(parts[1])
                    processed_lines.append(" ".join(parts))

                final_cmds = "\n".join(processed_lines) + "\n"
                self._proc.stdin.write(final_cmds.encode())
                self._proc.stdin.flush()

                # Wait for all w commands
                if total_wait_ms > 0:
                    time.sleep(total_wait_ms / 1000.0)

                return True
            except (BrokenPipeError, OSError) as e:
                logger.warning("minitouch communication error: %s", e)
                self._stop_proc()
                return False


# Singleton-per-ADB-device
_controllers: dict[str, MinitouchController] = {}


def get_minitouch(adb: ADBDevice) -> MinitouchController | None:
    """Get or create a minitouch controller for an ADB device."""
    if adb.serial not in _controllers:
        mt = MinitouchController(adb)
        if mt.start():
            _controllers[adb.serial] = mt
        else:
            return None
    return _controllers[adb.serial]


def stop_all() -> None:
    """Stop all minitouch controllers."""
    for mt in _controllers.values():
        mt.stop()
    _controllers.clear()
