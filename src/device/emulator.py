"""Emulator discovery and connection management.

Includes ADB health monitoring with auto-reconnect, process memory
monitoring, and full lifecycle management (restart, scheduled reboot).
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from config.settings import config

logger = logging.getLogger(__name__)

# Lazy import — psutil may not be installed on all systems
_psutil: object | None = None


def _get_psutil():
    global _psutil
    if _psutil is None:
        try:
            import psutil as _p
            _psutil = _p
        except ImportError:
            logger.warning("psutil not installed — memory monitoring disabled. "
                          "Install with: pip install psutil")
            _psutil = False
    return _psutil if _psutil is not False else None


# ---- Emulator process name patterns ----

_EMULATOR_PROCESS_PATTERNS: dict[str, list[str]] = {
    "ldplayer": [
        "Ld9BoxHeadless.exe", "LdBoxHeadless.exe",
        "dnplayer.exe", "ldplayer.exe",
    ],
    "mumu": [
        "MuMuPlayer.exe", "MuMuVMM.exe",
        "NemuHeadless.exe", "NemuPlayer.exe",
    ],
    "bluestacks": [
        "BlueStacks.exe", "HD-Player.exe",
        "BlueStacksAppPlayer.exe",
    ],
    # "generic" → fallback: find any process with adb in its command line
}


# ---- Emulator console helpers ----

def _ldconsole(console_path: str, *args: str, timeout: float = 30.0) -> tuple[int, str, str]:
    """Run an ldconsole command. Returns (returncode, stdout, stderr)."""
    cmd = [console_path, *args]
    logger.debug("ldconsole: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def _emu_ensure_dpi_awareness() -> None:
    """Ensure the process is DPI-aware before querying window rects.

    On scaled displays, a non-DPI-aware process gets virtualized window
    coordinates from GetWindowRect that don't match the physical mouse
    hook coordinates from MSLLHOOKSTRUCT.pt, causing ALL clicks to be
    silently filtered.

    Safe to call multiple times — only the first call takes effect.
    """
    try:
        from src.agent.mouse_hook import MouseClickMonitor
        MouseClickMonitor._ensure_dpi_awareness()
    except Exception:
        pass


class EmulatorManager:
    """Discover and manage ADB-connected devices (emulators or physical).

    Lifecycle management (restart, memory monitoring) is included in the
    health monitor loop, so enabling health monitoring gives you full
    emulator lifecycle management for free.
    """

    def __init__(self, adb_path: str | None = None) -> None:
        self.adb_path = adb_path or config.adb.path
        self._devices: dict[str, str] = {}
        # Emulator inventory — lazily loaded
        self._inventory: Any = None
        # Per-device health monitoring
        self._monitor_threads: dict[str, threading.Thread] = {}
        self._monitor_stop_events: dict[str, threading.Event] = {}
        self._consecutive_failures: dict[str, int] = {}
        self._monitor_callbacks: list[Callable[[str, str], None]] = []
        self._max_callbacks = 20
        # Restart cooldown tracking: serial → timestamp of last restart
        self._last_restart: dict[str, float] = {}
        # Restart-in-progress flag per serial
        self._restarting: dict[str, bool] = {}
        self._restart_lock = threading.Lock()
        self._known_serials: set[str] = set()

    @property
    def inventory(self):
        """Lazily load the emulator inventory singleton."""
        if self._inventory is None:
            from src.device.emulator_inventory import get_emulator_inventory
            self._inventory = get_emulator_inventory()
        return self._inventory

    # ---- Instance enumeration (OS-level, multi-instance aware) ----

    def list_emulator_instances(self) -> list[dict[str, Any]]:
        """Return all emulator instances — running or not — with OS-level detection.

        Each dict: {serial, emulator_type, name, status, inventory_id, games}.
        Uses tasklist (Windows) or ps (Linux) to find running emulator processes,
        then maps them to ADB serials and inventory entries.

        Powered-off instances (in inventory but no running process) are also returned
        with status='offline'.
        """
        running_processes = self._detect_emu_processes()
        self.discover()

        # Build result: merge process list + ADB devices + inventory
        seen_serials: set[str] = set()
        results: list[dict] = []

        # 1. Running processes first
        for proc in running_processes:
            serial = self._match_process_to_serial(proc)
            entry = self.inventory.find_by_serial(serial) if serial else None
            if not entry and serial:
                # Try port match
                port = serial.split(":")[-1] if ":" in serial else serial
                for e in self.inventory.list_all():
                    if port in e.adb_ports:
                        entry = e
                        break

            results.append({
                "serial": serial or "",
                "emulator_type": proc["emu_type"],
                "name": entry.name if entry else proc.get("name", "未知"),
                "status": "online" if serial else "running_no_adb",
                "inventory_id": entry.id if entry else "",
                "games": entry.installed_games if entry else [],
                "processes": proc["pids"],
            })
            if serial:
                seen_serials.add(serial)

        # 2. Online ADB devices not matched to any process (edge case)
        for serial in self.list_online:
            if serial in seen_serials:
                continue
            entry = self.inventory.find_by_serial(serial)
            if not entry:
                port = serial.split(":")[-1] if ":" in serial else serial
                for e in self.inventory.list_all():
                    if port in e.adb_ports:
                        entry = e
                        break
            results.append({
                "serial": serial,
                "emulator_type": entry.emulator_type if entry else "unknown",
                "name": entry.name if entry else f"未知设备 ({serial[:16]})",
                "status": "online",
                "inventory_id": entry.id if entry else "",
                "games": entry.installed_games if entry else [],
                "processes": 0,
            })
            seen_serials.add(serial)

        # 3. Inventory entries not currently online
        for e in self.inventory.list_all():
            if e.current_serial in seen_serials:
                continue
            # Check if any of its ports is online
            online = False
            for port in e.adb_ports:
                for s in self.list_online:
                    if f":{port}" in s:
                        online = True
                        e.current_serial = s
                        break
                if online:
                    break
            if online:
                continue  # Already covered above

            results.append({
                "serial": "",
                "emulator_type": e.emulator_type,
                "name": e.name,
                "status": "offline",
                "inventory_id": e.id,
                "games": e.installed_games,
                "processes": 0,
            })

        return results

    def _detect_emu_processes(self) -> list[dict[str, Any]]:
        """Use OS-level tools to find running emulator processes.

        Returns list of {emu_type, name, pids}.
        For MuMu: each MuMuPlayerHeadless.exe → one instance.
        For LDPlayer: each LdBoxHeadless.exe → one instance.
        """
        import subprocess as _sp
        import platform as _pf

        results: list[dict] = []
        is_win = _pf.system() == "Windows"

        if is_win:
            try:
                out = _sp.run(["tasklist", "/FO", "CSV", "/NH"],
                            capture_output=True, text=True, timeout=10).stdout
            except Exception:
                return results

            mu_pids: list[int] = []
            ld_pids: list[int] = []
            for line in out.strip().split("\n"):
                line = line.strip().strip('"')
                if not line:
                    continue
                # tasklist CSV: "imagename.exe","pid","session","session#","mem"
                parts = line.split('","')
                if len(parts) < 2:
                    continue
                name = parts[0].lower()
                try:
                    pid = int(parts[1])
                except ValueError:
                    continue

                if any(p in name for p in ["mumuplayer", "mumuvmm", "nemuheadless", "nemuplayer"]):
                    mu_pids.append(pid)
                elif any(p in name for p in ["ldboxheadless", "ld9boxheadless", "dnplayer", "ldplayer"]):
                    ld_pids.append(pid)

            if mu_pids:
                results.append({
                    "emu_type": "mumu",
                    "name": "MuMu 12",
                    "pids": len(mu_pids),
                })
            if ld_pids:
                results.append({
                    "emu_type": "ldplayer",
                    "name": "雷电模拟器",
                    "pids": len(ld_pids),
                })
        else:
            # Linux: try ps aux | grep
            try:
                out = _sp.run(["ps", "aux"], capture_output=True, text=True, timeout=10).stdout
            except Exception:
                return results

            mu_count = len([l for l in out.split("\n") if "mumu" in l.lower()])
            ld_count = len([l for l in out.split("\n") if "ldplayer" in l.lower() or "ldbox" in l.lower()])
            if mu_count:
                results.append({"emu_type": "mumu", "name": "MuMu", "pids": mu_count})
            if ld_count:
                results.append({"emu_type": "ldplayer", "name": "雷电模拟器", "pids": ld_count})

        return results

    def _match_process_to_serial(self, proc: dict) -> str | None:
        """Match a process entry to an online ADB serial.

        For MuMu: iterate online serials, pick the one not yet matched.
        For LDPlayer: try emulator-5554 then 127.0.0.1:5555.
        """
        online = self.list_online
        emu_type = proc.get("emu_type", "")

        if emu_type == "mumu":
            # Network ADB — return first matching
            for s in online:
                if s.startswith("127.0.0.1:") or s.startswith("localhost:"):
                    return s
        elif emu_type == "ldplayer":
            for s in online:
                if s.startswith("emulator-") or s == "127.0.0.1:5555":
                    return s
        return online[0] if online else None

    def deduplicate_devices(self, serials: list[str]) -> list[str]:
        """Remove ghost ADB connections pointing to the same physical VM.

        For MuMu 12, each VM exposes multiple network ADB ports, but
        `adb shell getprop ro.serialno` returns the same value for all
        ports of the same VM.  Two truly separate VMs (multi-instance
        clones) return different values.
        """
        if len(serials) <= 1:
            return serials

        import subprocess as _sp
        seen: dict[str, str] = {}  # serialno → first serial
        result: list[str] = []

        for s in serials:
            try:
                proc = _sp.run(
                    [config.adb.path, "-s", s, "shell", "getprop", "ro.serialno"],
                    capture_output=True, text=True, timeout=5.0,
                )
                sno = proc.stdout.strip()
                if not sno:
                    # Can't get serialno.  If another device on the same host
                    # DID return one, this is almost certainly a ghost connection
                    # to that same VM (e.g. stale port 7555 → same MuMu VM as
                    # port 16384).  Drop it rather than creating a phantom slot.
                    if seen:
                        logger.info(
                            "Ghost ADB device: %s has no serialno while %s does — dropping",
                            s, list(seen.values())[0],
                        )
                        try:
                            _sp.run([config.adb.path, "disconnect", s],
                                   capture_output=True, timeout=3)
                        except Exception:
                            pass
                        continue
                    # No other device returned a serialno yet — keep it for now
                    result.append(s)
                    continue
                if sno in seen:
                    logger.info("Ghost ADB device: %s is same VM as %s (serialno=%s) — dropping",
                               s, seen[sno], sno)
                    # Disconnect ghost to clean up ADB state
                    try:
                        _sp.run([config.adb.path, "disconnect", s],
                               capture_output=True, timeout=3)
                    except Exception:
                        pass
                else:
                    seen[sno] = sno
                    result.append(s)
            except Exception:
                # Can't reach — keep it (maybe it's a legit separate device)
                result.append(s)

        if len(result) < len(serials):
            logger.info("Dedup: %d devices → %d unique", len(serials), len(result))
        return result

    # ---- Discovery ----

    def discover(self) -> list[tuple[str, str]]:
        """Scan adb devices. Returns list of (serial, state) tuples."""
        proc = subprocess.run(
            [self.adb_path, "devices"],
            capture_output=True, text=True, timeout=10.0,
        )
        devices: list[tuple[str, str]] = []
        for line in proc.stdout.strip().split("\n")[1:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                devices.append((parts[0], parts[1]))
        self._devices = {serial: state for serial, state in devices}
        return devices

    @property
    def first_online(self) -> str | None:
        """Return the serial of the first online device matching our emulator type."""
        online = self.list_online
        return online[0] if online else None

    def _mumu_get_current_ports(self) -> dict[int, str]:
        """Query MuMuManager for current ADB port of each VM.

        Returns {vm_index: adb_port, ...}.  Empty dict if MuMuManager fails.
        Runs once per list_online call — no caching, always fresh.
        """
        exe_dir = Path(config.emulator.console_path).parent  # .../shell/
        mgr = exe_dir / "MuMuManager.exe"
        if not mgr.exists():
            return {}

        import json as _json
        import subprocess as _sp
        result: dict[int, str] = {}
        for vm_idx in (0, 1, 2):  # Scan up to 3 instances
            try:
                proc = _sp.run([str(mgr), "info", "--vmindex", str(vm_idx)],
                              capture_output=True, timeout=10,
                              encoding="utf-8", errors="replace")
                info = _json.loads(proc.stdout)
                if info.get("is_android_started"):
                    port = str(info.get("adb_port", ""))
                    if port:
                        result[vm_idx] = port
            except Exception:
                pass
        return result

    def _probe_mumu_ports(self) -> None:
        """Connect to current MuMu ADB ports (dynamically queried from MuMuManager).

        Also disconnects stale connections to known MuMu ports that are NOT
        in the live set returned by MuMuManager.  Without this, a lingering
        ``adb connect 127.0.0.1:7555`` from a previous session survives
        across bot restarts and looks like a second device, but it points to
        the same physical VM as the live port — causing two agent slots to
        fight over one emulator.
        """
        live_ports = self._mumu_get_current_ports()
        import subprocess as _sp

        # Disconnect stale connections to known MuMu ports that aren't live.
        # Live ports are re-connected below so they stay fresh.
        all_known = set(self.inventory.all_known_ports())
        live_set = set(live_ports.values())
        stale = all_known - live_set
        for port in stale:
            try:
                _sp.run([config.adb.path, "disconnect", f"127.0.0.1:{port}"],
                       capture_output=True, timeout=3)
            except Exception:
                pass

        if not live_ports:
            return
        for vm_idx, port in live_ports.items():
            _sp.run([config.adb.path, "connect", f"127.0.0.1:{port}"],
                   capture_output=True, timeout=5)

    def _probe_emu_ports(self) -> None:
        """Try adb connect to known emulator ports.

        For MuMu 12: uses MuMuManager to get live port assignments.
        For other types: falls back to inventory port lists.
        """
        import subprocess as _sp
        if config.emulator.type == "mumu":
            self._probe_mumu_ports()
            return
        ports = self.inventory.all_known_ports()
        for port in ports:
            try:
                _sp.run([config.adb.path, "connect", f"127.0.0.1:{port}"],
                       capture_output=True, timeout=5)
            except Exception:
                pass

    @property
    def list_online(self) -> list[str]:
        """Return serials of online devices matching our emulator type.

        For MuMu 12: queries MuMuManager for live port assignments,
        connects to them, and returns only those. No hardcoded ports.
        """
        self._probe_emu_ports()
        self.discover()
        all_online = [s for s, st in self._devices.items() if st == "device"]

        # ── Filter by emulator type ──
        emu_type = config.emulator.type
        if emu_type == "mumu":
            # Accept ALL 127.0.0.1:* devices. Ghost dedup via deduplicate_devices.
            matching = [s for s in all_online if s.startswith("127.0.0.1:")]
        elif emu_type == "ldplayer":
            matching = [s for s in all_online
                       if s.startswith("emulator-") or s.startswith("127.0.0.1:")]
        else:
            matching = all_online

        # Deduplicate: MuMu 12 exposes multiple ports for the same VM
        matching = self.deduplicate_devices(matching)
        return matching

    def is_online(self, serial: str) -> bool:
        """Check if a specific device is online."""
        self.discover()
        return self._devices.get(serial) == "device"

    def get_emulator_window(self, serial: str = "") -> dict | None:
        """Find the Windows window handle and rect for an emulator instance.

        Uses the configured emulator type to search for the correct window.
        Returns dict with hwnd + title + rect, or None if not found.
        Used by the observation learning system to map mouse clicks to
        device coordinates.

        Args:
            serial: ADB device serial (optional, used for multi-instance matching).
        """
        try:
            import win32gui
        except ImportError:
            logger.debug("pywin32 not available — cannot find emulator window")
            return None

        emu_type = config.emulator.type
        if emu_type == "mumu":
            return self._find_mumu_window(serial)
        elif emu_type == "ldplayer":
            return self._find_ldplayer_window(serial)
        else:
            logger.debug("Window detection not supported for emulator type '%s'", emu_type)
            return None

    @staticmethod
    def _find_mumu_window(serial: str = "") -> dict | None:
        """Find MuMu emulator main window by enumerating visible windows.

        Uses a layered search: exact title match first, then Qt class name,
        then largest-visible-window heuristic.  This avoids the fragile
        'mumuplayer' substring check that breaks when window titles vary
        across MuMu versions or when a game name takes over the title bar.
        """
        # Ensure DPI awareness BEFORE GetWindowRect queries.
        # Without this, GetWindowRect returns virtualized coords that don't
        # match mouse hook physical coords, causing all clicks to be filtered.
        _emu_ensure_dpi_awareness()

        try:
            import win32gui
        except ImportError:
            return None

        # ── Strategy 1: known title patterns for MuMu + other emulators ──
        _patterns = [
            "mumuplayer", "mumu", "雷电", "ldplayer", "bluestacks",
            "nox", "noxplayer", "memu", "gameloop",
        ]
        result: dict | None = None

        def _title_cb(hwnd: int, _ctx) -> bool:
            nonlocal result
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title or not title.strip():
                return True
            title_lower = title.lower()
            for pat in _patterns:
                if pat in title_lower:
                    try:
                        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                        if (right - left) < 200:
                            return True
                        result = {
                            "hwnd": hwnd,
                            "title": title,
                            "rect": (left, top, right, bottom),
                        }
                        return False  # Stop — found
                    except Exception:
                        pass
                    break
            return True

        try:
            win32gui.EnumWindows(_title_cb, None)
        except Exception:
            pass

        if result:
            logger.debug("Found emulator window: '%s' hwnd=%d rect=%s",
                        result["title"], result["hwnd"], result["rect"])
            return result

        # ── Strategy 2: Qt-based windows (MuMu 12) ──
        best_area = 0
        best_qt: dict | None = None

        def _qt_cb(hwnd: int, _ctx) -> bool:
            nonlocal best_qt, best_area
            if not win32gui.IsWindowVisible(hwnd):
                return True
            try:
                cls = win32gui.GetClassName(hwnd)
                if "QWindowIcon" not in cls:
                    return True
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                w, h = right - left, bottom - top
                if w < 200 or h < 200:
                    return True
                area = w * h
                if area > best_area:
                    best_area = area
                    best_qt = {
                        "hwnd": hwnd,
                        "title": win32gui.GetWindowText(hwnd),
                        "rect": (left, top, right, bottom),
                    }
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(_qt_cb, None)
        except Exception:
            pass

        if best_qt:
            logger.debug("Found emulator by Qt class: '%s' hwnd=%d",
                        best_qt["title"], best_qt["hwnd"])
            return best_qt

    @staticmethod
    def _find_ldplayer_window(serial: str = "") -> dict | None:
        """Find LDPlayer main window."""
        try:
            import win32gui
        except ImportError:
            return None

        result: dict | None = None

        def _callback(hwnd: int, _ctx) -> bool:
            nonlocal result
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if "雷电" in title or "ldplayer" in title.lower():
                try:
                    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                    if (right - left) < 200:
                        return True
                    result = {
                        "hwnd": hwnd,
                        "title": title,
                        "rect": (left, top, right, bottom),
                    }
                    return False
                except Exception:
                    pass
            return True

        try:
            win32gui.EnumWindows(_callback, None)
        except Exception as e:
            logger.debug("EnumWindows failed: %s", e)
            return None

        return result

    def connect_tcp(self, host: str, port: int = 5555) -> str | None:
        """Connect to a device over TCP/IP. Returns serial on success."""
        addr = f"{host}:{port}"
        proc = subprocess.run(
            [self.adb_path, "connect", addr],
            capture_output=True, text=True, timeout=10.0,
        )
        if "connected" in proc.stdout.lower():
            logger.info("Connected to %s", addr)
            return addr
        logger.warning("Failed to connect to %s: %s", addr, proc.stdout.strip())
        return None

    # ---- ADB-level operations ----

    def _ping_device(self, serial: str) -> bool:
        """Send a lightweight ADB ping to check device responsiveness.

        P3: also tests screenshot capture — catches GPU/SurfaceFlinger hangs
        that would pass a plain shell echo but prevent the agent from working.
        """
        try:
            proc = subprocess.run(
                [self.adb_path, "-s", serial, "shell", "echo", "ping"],
                capture_output=True, text=True, timeout=5.0,
            )
            echo_ok = "ping" in proc.stdout
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("Heartbeat ping failed: %s", e)
            return False

        # P3: also verify screenshot works (catches GPU freeze / SurfaceFlinger hang)
        if echo_ok:
            try:
                proc2 = subprocess.run(
                    [self.adb_path, "-s", serial, "exec-out", "screencap", "-p"],
                    capture_output=True, timeout=5.0,
                )
                # Valid PNG starts with \x89PNG\r\n\x1a\n
                screenshot_ok = (
                    proc2.returncode == 0
                    and len(proc2.stdout) > 8
                    and proc2.stdout[:4] == b'\x89PNG'
                )
                if not screenshot_ok:
                    logger.debug("Heartbeat: screenshot failed (len=%d, magic=%s)",
                                len(proc2.stdout), proc2.stdout[:8].hex() if proc2.stdout else "none")
                return screenshot_ok
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.debug("Heartbeat screenshot test failed: %s", e)
                return False
        return False

    def _reconnect(self, serial: str) -> bool:
        """Attempt to reconnect to a device. Returns True on success."""
        for attempt in range(config.adb.max_reconnect_attempts):
            wait = min(2 ** attempt, 30)
            logger.info("Reconnect attempt %d/%d for %s (waiting %ds)...",
                         attempt + 1, config.adb.max_reconnect_attempts, serial, wait)

            self.discover()
            if self.is_online(serial):
                logger.info("Device %s re-discovered", serial)
                return True

            if ":" in serial:
                host, port = serial.split(":", 1)
                result = self.connect_tcp(host, int(port))
                if result and self.is_online(serial):
                    return True

            time.sleep(wait)

        return False

    def adb_reboot(self, serial: str) -> bool:
        """Soft restart: reboot the Android system inside the emulator via ADB.

        This restarts the Android OS without restarting the emulator VM.
        Much faster than a full emulator restart (~30-60s vs ~90-120s).
        Often sufficient to reclaim leaked memory.

        Returns True if the reboot command was accepted.
        """
        try:
            proc = subprocess.run(
                [self.adb_path, "-s", serial, "reboot"],
                capture_output=True, text=True, timeout=10.0,
            )
            # adb reboot often returns non-zero even on success (device disconnects)
            logger.info("ADB reboot sent to %s (rc=%d)", serial, proc.returncode)
            return True
        except Exception as e:
            logger.warning("ADB reboot failed for %s: %s", serial, e)
            return False

    def wait_for_device(self, serial: str, timeout: float | None = None) -> bool:
        """Block until the device is back online after a restart.

        Args:
            serial: Device serial.
            timeout: Max wait in seconds. Defaults to config.emulator.boot_wait_seconds.

        Returns:
            True if the device came back, False if timed out.
        """
        timeout = timeout or config.emulator.boot_wait_seconds
        deadline = time.monotonic() + timeout
        adb_poll = config.emulator.adb_poll_interval

        # First wait for adb to acknowledge the device at all
        logger.info("Waiting for %s to come online (timeout=%.0fs)...", serial, timeout)
        while time.monotonic() < deadline:
            self.discover()
            state = self._devices.get(serial, "offline")
            if state == "device":
                # Now do a functional check: can we run shell commands?
                if self._ping_device(serial):
                    logger.info("Device %s is back online after restart", serial)
                    return True
                else:
                    logger.debug("Device %s visible but not responsive yet...", serial)
            elif state == "offline":
                logger.debug("Device %s is offline (booting)...", serial)
            time.sleep(adb_poll)

        logger.warning("Device %s did not come back within %.0fs", serial, timeout)
        return False

    # ---- Process memory monitoring ----

    def get_emulator_memory_mb(self) -> dict[str, int]:
        """Return total memory usage of emulator processes in MB.

        Returns:
            {process_name: memory_mb} dict. Empty if psutil unavailable or no
            emulator processes found.
        """
        psutil = _get_psutil()
        if psutil is None:
            return {}

        patterns = self._get_process_patterns()
        if not patterns:
            return {}

        result: dict[str, int] = {}
        pattern_set = {p.lower() for p in patterns}
        try:
            for proc in psutil.process_iter(["name", "memory_info"]):
                try:
                    name = proc.info["name"] or ""
                    if name.lower() in pattern_set:
                        mem_mb = int(proc.info["memory_info"].wset / (1024 * 1024))
                        # Sum if multiple processes with same name (unlikely)
                        result[name] = result.get(name, 0) + mem_mb
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.warning("Process scan failed: %s", e)

        return result

    def get_total_emulator_memory_mb(self) -> int:
        """Return sum of all emulator process memory in MB."""
        mem = self.get_emulator_memory_mb()
        return sum(mem.values())

    # ---- Full emulator restart ----

    def restart_emulator(self, serial: str) -> str:
        """Full restart: quit the emulator process, then launch it again.

        Steps:
        1. Fire "pre_restart" callbacks so the scheduler can pause
        2. Try ADB reboot first (fast path)
        3. If that fails, use console to quit + launch (slow path)
        4. Wait for device to come back online
        5. Fire "post_restart" callbacks so the scheduler can resume
        6. Re-init ADB device pool entry

        Returns:
            "ok" — restart completed successfully
            "failed" — restart attempt failed
            "already_running" — a restart is already in progress for this device
        """
        # Fast rejection without touching locks
        if self._restarting.get(serial, False):
            logger.info("Restart already in progress for %s — skipping", serial)
            return "already_running"

        # Thread-safe commit to restart
        with self._restart_lock:
            if self._restarting.get(serial, False):
                return "already_running"
            self._restarting[serial] = True

        try:
            logger.warning("=== Restarting emulator for %s ===", serial)
            self._fire_callbacks("pre_restart", serial)

            # ---- Phase 1: Shutdown ----
            self._fire_callbacks("shutting_down", serial)

            # Check if device is actually connected — if not, skip ADB reboot
            # entirely and go straight to cold launch. ADB reboot on a
            # disconnected device appears to "succeed" (rc=0 or 1) but the
            # device never comes back.
            device_connected = False
            try:
                from src.device.adb import get_adb as _check_adb
                dev = _check_adb(serial=serial)
                device_connected = dev._heartbeat_ok
            except Exception:
                pass

            if device_connected:
                # Try ADB reboot first — fast, often enough for memory cleanup
                adb_ok = self.adb_reboot(serial)
                if adb_ok:
                    logger.info("ADB reboot sent, waiting for device to disconnect...")
                    time.sleep(5.0)

                    if self.wait_for_device(serial):
                        self._on_restart_complete(serial)
                        return "ok"

                    logger.warning("ADB reboot didn't bring device back — trying console restart")
            else:
                logger.info("Device not connected — skipping ADB reboot, going to cold launch")

            # ---- Phase 2: Console-based hard restart ----
            # _mumu_restart handles _on_restart_complete internally (the serial
            # changes for network ADB), so return directly on success.
            if self._console_restart(serial):
                return "ok"

            logger.error("Console restart failed for %s", serial)
            self._fire_callbacks("restart_failed", serial)
            return "failed"

        except Exception as e:
            logger.exception("Restart crashed for %s: %s", serial, e)
            self._fire_callbacks("restart_failed", serial)
            return "failed"
        finally:
            with self._restart_lock:
                self._restarting[serial] = False

    def _console_restart(self, serial: str) -> bool:
        """Quit and re-launch the emulator via its console tool.

        Returns True if the launch command succeeded. The caller is responsible
        for waiting for ADB to come back.
        """
        emu_type = config.emulator.type

        if emu_type == "ldplayer":
            return self._ldplayer_restart(serial)
        elif emu_type == "mumu":
            return self._mumu_restart(serial)
        else:
            logger.warning(
                "Console restart not supported for emulator type '%s'. "
                "Install psutil and use ADB reboot instead, or set EMULATOR_TYPE "
                "to 'ldplayer'/'mumu' and configure EMULATOR_CONSOLE.", emu_type)
            return False

    def _ldplayer_restart(self, serial: str) -> bool:
        """LDPlayer-specific restart via ldconsole."""
        console = config.emulator.console_path
        instance = config.emulator.instance_name

        if not Path(console).exists():
            logger.error("ldconsole not found at %s", console)
            return False

        logger.info("Quitting LDPlayer instance '%s'...", instance)
        rc, out, err = _ldconsole(console, "quit", "--name", instance,
                                  timeout=config.emulator.shutdown_wait_seconds)
        logger.info("ldconsole quit: rc=%d out=%s err=%s", rc, out.strip(), err.strip())

        # Wait for process to fully exit
        time.sleep(config.emulator.shutdown_wait_seconds)

        logger.info("Launching LDPlayer instance '%s'...", instance)
        rc, out, err = _ldconsole(console, "launch", "--name", instance,
                                  timeout=config.emulator.shutdown_wait_seconds)
        logger.info("ldconsole launch: rc=%d out=%s err=%s", rc, out.strip(), err.strip())
        if rc == 0 and self.wait_for_device(serial):
            self._on_restart_complete(serial)
            return True
        return False

    def _mumu_restart(self, serial: str) -> bool:
        """MuMu 12 restart: gracefully shutdown the VM, then relaunch.

        Tries per-instance MuMuManager shutdown first (safe for multi-instance).
        Falls back to taskkill if MuMuManager is unavailable.

        Returns True on success. On success, calls _on_restart_complete with
        the NEW device serial (MuMu 12 uses network ADB so the serial changes).
        """
        import subprocess as _sp
        import time as _time

        logger.info("MuMu restart for %s", serial)

        # ── Find the VM index for this serial ──
        vm_idx = None
        live_ports = self._mumu_get_current_ports()
        for idx, port in live_ports.items():
            if f"127.0.0.1:{port}" == serial:
                vm_idx = idx
                break

        # ── Step 1: Graceful per-instance shutdown ──
        killed = False
        if vm_idx is not None:
            try:
                console = config.emulator.console_path
                _sp.run(
                    [console, "control", "-v", str(vm_idx), "shutdown"],
                    capture_output=True, timeout=30,
                )
                logger.info("MuMu VM %d shutdown via MuMuManager", vm_idx)
                killed = True
            except Exception as e:
                logger.warning("MuMuManager shutdown VM %d failed: %s — falling back to taskkill", vm_idx, e)

        if not killed:
            # Step 1 (fallback): Kill only this specific MuMu instance.
            # WARNING: taskkill /IM kills ALL MuMuPlayer.exe processes system-wide.
            # In multi-instance environments, this should only be used as a
            # last resort when MuMuManager shutdown fails.
            logger.warning(
                "MuMu restart: using taskkill /IM MuMuPlayer.exe (kills ALL MuMu instances)"
            )
            try:
                _sp.run(["taskkill", "/F", "/IM", "MuMuPlayer.exe"],
                       capture_output=True, timeout=15)
                killed = True
                logger.info("MuMuPlayer.exe killed (all instances)")
            except Exception as e:
                logger.warning("taskkill MuMuPlayer.exe failed: %s", e)

        _time.sleep(3.0)

        # Step 1.5: Disconnect lingering ADB connections.
        # On Windows, TCP connections can survive the process for seconds
        # after taskkill. If list_online still sees 127.0.0.1:16384 when
        # _mumu_launch runs, it short-circuits and returns without actually
        # launching MuMuPlayer.exe — the user never sees the emulator window.
        adb_path = config.adb.path
        for port in ("16384", "7555", "7556", "7557"):
            try:
                _sp.run([adb_path, "disconnect", f"127.0.0.1:{port}"],
                       capture_output=True, timeout=5)
            except Exception:
                pass
        # Also disconnect the old USB-style serial if it's a network alias
        if serial and serial.startswith("emulator-"):
            try:
                _sp.run([adb_path, "disconnect", serial],
                       capture_output=True, timeout=5)
            except Exception:
                pass

        # Step 2: Launch — returns the new ADB serial (127.0.0.1:xxxxx)
        new_serial = self._mumu_launch(serial)
        if new_serial is not None:
            self._on_restart_complete(new_serial, old_serial=serial)
            return True
        return False

    def _on_restart_complete(self, serial: str, old_serial: str = "") -> None:
        """Called after a successful restart. Re-initializes ADB and fires callbacks.

        If the serial changed (e.g. MuMu 12 network ADB), stops monitoring the
        old serial and starts monitoring the new one.
        """
        self._last_restart[serial] = time.time()

        logger.info("=== Emulator %s restarted successfully ===", serial)

        # Stop old health monitor if serial changed
        if old_serial and old_serial != serial:
            logger.info("Stopping health monitor for old serial %s", old_serial)
            self.stop_health_monitor(old_serial)

        # Re-init ADB device pool
        try:
            from src.device.adb import init_adb, remove_device
            if old_serial and old_serial != serial:
                remove_device(old_serial)
            remove_device(serial)
            init_adb(serial)
            logger.info("ADB device pool re-initialized for %s", serial)
        except Exception as e:
            logger.warning("ADB re-init after restart failed: %s", e)

        # Start health monitor for new serial if not already monitoring
        if serial not in self._monitor_threads or not self._monitor_threads[serial].is_alive():
            self.start_health_monitor(serial)

        # Register the new serial with the scheduler so timed tasks can use it
        try:
            from src.scheduler.cron_scheduler import get_engine as _get_sched
            _sched = _get_sched()
            if serial not in _sched.device_serials:
                _sched.add_device(serial)
                if old_serial and old_serial in _sched.device_serials:
                    _sched.remove_device(old_serial)
        except Exception:
            pass

        self._fire_callbacks("post_restart", serial)

    def launch_emulator(self, force_new: bool = False) -> str | None:
        """Cold-start the emulator when it is not running.

        Does NOT require a device serial — uses the configured instance name
        and console path from settings. Returns the serial of the newly
        launched device on success, or None on failure.

        When force_new=True, launches a NEW instance even if one is already
        running (multi-instance/多开). The existing instance is left untouched.

        Supported emulator types: ldplayer, mumu.
        """
        emu_type = config.emulator.type

        if emu_type == "ldplayer":
            return self._ldplayer_launch(force_new=force_new)
        elif emu_type == "mumu":
            return self._mumu_launch(force_new=force_new)
        else:
            logger.warning(
                "Cold launch not supported for emulator type '%s'. "
                "Set EMULATOR_TYPE to 'ldplayer' or 'mumu'.", emu_type)
            return None

    def _ldplayer_launch(self, force_new: bool = False) -> str | None:
        """Launch LDPlayer instance via ldconsole. Returns serial on success."""
        console = config.emulator.console_path
        instance = config.emulator.instance_name

        if not Path(console).exists():
            logger.error("ldconsole not found at %s", console)
            return None

        # Check if already running — skip if not forcing new
        if not force_new:
            online = self.list_online
            if online:
                logger.info("LDPlayer already running: %s", online)
                return online[0]

        logger.info("Launching LDPlayer instance '%s'...", instance)
        rc, out, err = _ldconsole(console, "launch", "--name", instance,
                                  timeout=config.emulator.boot_wait_seconds)
        logger.info("ldconsole launch: rc=%d out=%s err=%s", rc, out.strip(), err.strip())

        if rc != 0:
            logger.error("LDPlayer launch failed: rc=%d err=%s", rc, err.strip())
            return None

        # Wait for device to appear in ADB
        deadline = time.monotonic() + config.emulator.boot_wait_seconds
        while time.monotonic() < deadline:
            online_devices = self.list_online
            if online_devices:
                serial = online_devices[0]
                logger.info("LDPlayer launched: %s", serial)
                return serial
            time.sleep(config.emulator.adb_poll_interval)

        logger.error("LDPlayer did not appear in ADB after %.0fs",
                     config.emulator.boot_wait_seconds)
        return None

    def _mumu_launch(self, serial: str = "", force_new: bool = False) -> str | None:
        """Launch MuMu 12. Returns ADB serial on success.

        force_new=True: launches cloned VM #1 via MuMuManager.exe without
                        killing the running main VM.  Parses JSON from
                        `MuMuManager info` to get the dynamic ADB port.
        force_new=False: starts MuMuPlayer.exe (main VM).
        """
        import subprocess as _sp
        import os as _os

        console = config.emulator.console_path
        exe_dir = Path(console).parent  # .../shell/

        online_before = set(self.list_online)

        if force_new:
            mgr = exe_dir / "MuMuManager.exe"
            if not mgr.exists():
                logger.error("MuMuManager.exe not found at %s", mgr)
                return None

            # Step 1: Launch VM #1
            logger.info("Launching MuMu VM #1 via MuMuManager.exe")
            try:
                _sp.run([str(mgr), "control", "--vmindex", "1", "launch"],
                       capture_output=True, timeout=15, encoding="utf-8",
                       errors="replace")
            except Exception as e:
                logger.error("MuMuManager launch failed: %s", e)
                return None

            # Step 2: Poll until VM starts, then get its ADB port
            import json as _json
            import time as _t
            deadline = _t.monotonic() + config.emulator.boot_wait_seconds
            adb_port: str = ""
            while _t.monotonic() < deadline:
                try:
                    proc = _sp.run([str(mgr), "info", "--vmindex", "1"],
                                  capture_output=True, timeout=10,
                                  encoding="utf-8", errors="replace")
                    info = _json.loads(proc.stdout)
                    if info.get("is_android_started"):
                        adb_port = str(info.get("adb_port", ""))
                        if adb_port:
                            break
                except (_json.JSONDecodeError, Exception):
                    pass
                _t.sleep(config.emulator.adb_poll_interval)

            if not adb_port:
                logger.error("MuMu VM #1 started but could not determine ADB port")
                return None

            # Step 3: Connect ADB to the new device
            serial_new = f"127.0.0.1:{adb_port}"
            logger.info("MuMu VM #1 ADB port: %s", serial_new)
            _sp.run([config.adb.path, "connect", serial_new],
                   capture_output=True, timeout=5)

            # Verify directly — do NOT use list_online filtering which
            # rejects dynamic ports not in the inventory.
            import time as _t
            deadline2 = _t.monotonic() + 30
            while _t.monotonic() < deadline2:
                try:
                    proc = _sp.run([config.adb.path, "-s", serial_new, "shell", "echo", "ping"],
                                  capture_output=True, timeout=5)
                    if proc.returncode == 0:
                        logger.info("MuMu VM #1 reachable: %s", serial_new)
                        self._known_serials.add(serial_new)
                        try:
                            from src.device.adb import init_adb
                            init_adb(serial_new)
                        except Exception:
                            pass
                        return serial_new
                except Exception:
                    pass
                _t.sleep(2)

            logger.error("MuMu VM #1 not reachable via ADB at %s", serial_new)
            return None

        # ── Normal launch (force_new=False) ──
        if online_before:
            logger.info("MuMu already running: %s", online_before)
            return list(online_before)[0]

        logger.info("Launching MuMu 12: %s", console)
        try:
            _sp.Popen([console], shell=True)
            logger.info("MuMu 12 launch command sent")
        except Exception as e:
            logger.error("MuMu 12 launch failed: %s", e)
            return None

        # list_online internally calls _probe_mumu_ports → MuMuManager
        # → adb connect to the live dynamic port → discover → dedup.
        # No hardcoded port list needed.
        import time as _time
        deadline = _time.monotonic() + config.emulator.boot_wait_seconds
        while _time.monotonic() < deadline:
            online = self.list_online
            for dev in online:
                if dev.startswith("127.0.0.1:") or dev.startswith("localhost:"):
                    logger.info("MuMu 12 device appeared: %s", dev)
                    self._known_serials.add(dev)
                    try:
                        from src.device.adb import init_adb, remove_device
                        if serial and serial != dev:
                            remove_device(serial)
                        init_adb(dev)
                    except Exception:
                        pass
                    return dev
            _time.sleep(config.emulator.adb_poll_interval)

        logger.error("MuMu 12 did not appear in ADB after %.0fs",
                     config.emulator.boot_wait_seconds)
        return None

    def is_restart_in_progress(self, serial: str) -> bool:
        """Check if a restart is currently in progress for a device."""
        return self._restarting.get(serial, False)

    def time_since_last_restart(self, serial: str) -> float | None:
        """Return seconds since last restart, or None if never restarted."""
        last = self._last_restart.get(serial)
        if last is None:
            return None
        return time.time() - last

    # ---- Health monitor (with memory + scheduled restart integrated) ----

    def start_health_monitor(self, serial: str | None = None) -> None:
        """Start background monitoring for one or all devices.

        Monitors: ADB connectivity, memory usage, scheduled restart time.
        Fires callbacks on disconnect/reconnect/restart events.
        """
        serials = [serial] if serial else self.list_online
        started = 0
        for s in serials:
            self._known_serials.add(s)
            if s in self._monitor_threads and self._monitor_threads[s].is_alive():
                continue
            self._consecutive_failures[s] = 0
            self._monitor_stop_events[s] = threading.Event()
            t = threading.Thread(
                target=self._monitor_loop,
                args=(s, self._monitor_stop_events[s]),
                daemon=True,
                name=f"health-monitor-{s}",
            )
            t.start()
            self._monitor_threads[s] = t
            started += 1
        if started:
            logger.info("Health monitor started for %d device(s)", started)

    def stop_health_monitor(self, serial: str | None = None) -> None:
        """Stop health monitoring for one or all devices."""
        serials = [serial] if serial else list(self._monitor_threads.keys())
        for s in serials:
            if s in self._monitor_stop_events:
                self._monitor_stop_events[s].set()
            thread = self._monitor_threads.pop(s, None)
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
            self._consecutive_failures.pop(s, None)
            self._monitor_stop_events.pop(s, None)

    def _monitor_loop(self, serial: str, stop_event: threading.Event) -> None:
        """Per-device health monitoring loop.

        Checks (in this order, each tick):
        1. Memory usage → auto-restart if above limit (with cooldown)
        2. Scheduled restart time → restart if due
        3. ADB heartbeat → reconnect on failure
        """
        interval = config.adb.heartbeat_interval
        memory_check_interval = max(interval * 6, 180.0)  # ~3 min for memory
        _last_memory_check = 0.0

        logger.info("Health monitor started for %s (interval=%ds, memory_limit=%dMB, restart_cron=%s)",
                     serial, config.adb.heartbeat_interval,
                     config.emulator.memory_limit_mb,
                     config.emulator.restart_cron or "disabled")

        while not stop_event.is_set():
            stop_event.wait(interval)
            if stop_event.is_set():
                break

            # Skip all checks if a restart is in progress
            if self.is_restart_in_progress(serial):
                continue

            now = time.time()

            # --- Check 1: Memory watchdog ---
            if config.emulator.memory_limit_mb > 0 and (now - _last_memory_check) >= memory_check_interval:
                _last_memory_check = now
                self._check_memory_and_restart(serial)

            # --- Check 2: Scheduled restart ---
            if config.emulator.restart_cron:
                self._check_scheduled_restart(serial, now)

            # --- Check 3: ADB heartbeat ---
            self._check_heartbeat(serial)

        logger.info("Health monitor stopped for %s", serial)

    def _check_memory_and_restart(self, serial: str) -> None:
        """Check emulator process memory and trigger restart if above limit."""
        mem = self.get_total_emulator_memory_mb()
        limit = config.emulator.memory_limit_mb

        if mem == 0:
            return  # psutil unavailable or no processes found

        mem_gb = mem / 1024.0
        limit_gb = limit / 1024.0

        if mem >= limit:
            cooldown = config.emulator.restart_cooldown_seconds
            last = self._last_restart.get(serial, 0)
            if time.time() - last < cooldown:
                logger.warning(
                    "Emulator memory %dMB (%.1fGB) exceeds limit %dMB (%.1fGB) "
                    "but restart cooldown active (last restart %.0fs ago)",
                    mem, mem_gb, limit, limit_gb, time.time() - last)
                return

            logger.warning(
                "=== MEMORY LIMIT EXCEEDED: %dMB / %.1fGB (limit=%dMB / %.1fGB) — restarting ===",
                mem, mem_gb, limit, limit_gb)

            # Run restart in a separate thread so we don't block the monitor
            t = threading.Thread(
                target=self.restart_emulator,
                args=(serial,),
                daemon=True,
                name=f"emu-restart-{serial}",
            )
            t.start()
        else:
            logger.debug("Emulator memory: %dMB / %.1fGB (limit=%.1fGB)",
                        mem, mem_gb, limit_gb)

    def _check_scheduled_restart(self, serial: str, now: float) -> None:
        """Check if the daily scheduled restart time has arrived."""
        try:
            from croniter import croniter as _croniter
        except ImportError:
            logger.debug("croniter not installed — scheduled restart check skipped")
            return

        try:
            cron = _croniter(config.emulator.restart_cron, datetime.now())
            next_time = cron.get_next(datetime)
            # If the next scheduled time is within this poll window (±interval), trigger
            next_ts = next_time.timestamp()
            if 0 <= (next_ts - now) < config.adb.heartbeat_interval * 2:
                # Skip if a restart is already in progress (e.g. memory-triggered)
                if self.is_restart_in_progress(serial):
                    logger.debug("Scheduled restart skipped: restart already in progress for %s", serial)
                    return
                last = self._last_restart.get(serial, 0)
                # Don't restart if we already restarted in the last hour
                if now - last > 3600:
                    logger.info("=== SCHEDULED RESTART: %s (cron=%s) ===",
                               time.strftime("%H:%M"), config.emulator.restart_cron)
                    t = threading.Thread(
                        target=self.restart_emulator,
                        args=(serial,),
                        daemon=True,
                        name=f"emu-sched-restart-{serial}",
                    )
                    t.start()
                else:
                    logger.debug("Scheduled restart skipped for %s (restarted %.0fs ago)",
                               serial, now - last)
        except Exception as e:
            logger.debug("Scheduled restart check failed: %s", e)

    def _check_heartbeat(self, serial: str) -> None:
        """Standard ADB heartbeat check with auto-reconnect."""
        ok = self._ping_device(serial)
        failures = self._consecutive_failures.get(serial, 0)

        if ok:
            if failures > 0:
                logger.info("Device %s recovered after %d failures", serial, failures)
                self._fire_callbacks("reconnected", serial)
            self._consecutive_failures[serial] = 0
        else:
            self._consecutive_failures[serial] = failures + 1
            logger.warning("Heartbeat failed for %s (%d/%d consecutive)",
                           serial, self._consecutive_failures[serial], 3)

            if self._consecutive_failures[serial] >= 3:
                logger.error("Device %s lost after %d failures — alerting",
                             serial, self._consecutive_failures[serial])
                self._fire_callbacks("disconnected", serial)

            if self._consecutive_failures[serial] >= 2:
                recovered = self._reconnect(serial)
                if recovered:
                    self._consecutive_failures[serial] = 0

    # ---- Callbacks / events ----

    def _fire_callbacks(self, event_type: str, serial: str) -> None:
        """Notify all registered health event callbacks."""
        for cb in self._monitor_callbacks:
            try:
                cb(event_type, serial)
            except Exception:
                logger.exception(
                    "Callback %s failed for event=%s serial=%s — "
                    "this may prevent notifications from being sent",
                    getattr(cb, '__name__', str(cb)[:80]), event_type, serial)

    def on_health_event(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for health events: callback(event_type, serial).

        Event types:
          - "reconnected": device came back after being down
          - "disconnected": device unreachable after 3 consecutive failures
          - "pre_restart": about to restart the emulator
          - "shutting_down": emulator process is being terminated
          - "post_restart": emulator restart completed, device is back online
          - "restart_failed": restart attempt failed completely
        """
        if callback not in self._monitor_callbacks:
            self._monitor_callbacks.append(callback)
            if len(self._monitor_callbacks) > self._max_callbacks:
                self._monitor_callbacks.pop(0)

    def remove_health_callback(self, callback: Callable[[str, str], None]) -> None:
        """Remove a previously registered health event callback."""
        try:
            self._monitor_callbacks.remove(callback)
        except ValueError:
            pass

    # ---- Internal helpers ----

    def _get_process_patterns(self) -> list[str]:
        """Return the list of process names to watch for memory monitoring."""
        if config.emulator.watch_process_names:
            return list(config.emulator.watch_process_names)
        return _EMULATOR_PROCESS_PATTERNS.get(config.emulator.type, [])

    # ---- Status ----

    @property
    def is_monitoring(self) -> bool:
        return bool(self._monitor_threads and
                    any(t.is_alive() for t in self._monitor_threads.values()))

    @property
    def health_status(self) -> dict:
        """Return current health + memory status for all monitored devices."""
        mem = self.get_emulator_memory_mb()
        total_mem_mb = sum(mem.values())

        devices: dict[str, dict] = {}
        for serial in self._monitor_threads:
            last_restart = self._last_restart.get(serial)
            devices[serial] = {
                "monitoring": self._monitor_threads[serial].is_alive(),
                "consecutive_failures": self._consecutive_failures.get(serial, 0),
                "online": self.is_online(serial),
                "restarting": self._restarting.get(serial, False),
                "last_restart": (datetime.fromtimestamp(last_restart).isoformat()
                                if last_restart else None),
            }

        return {
            "devices": devices,
            "emulator_memory": {
                "processes": mem,
                "total_mb": total_mem_mb,
                "total_gb": round(total_mem_mb / 1024.0, 2),
                "limit_mb": config.emulator.memory_limit_mb,
            },
        }


emulator_manager = EmulatorManager()
