"""Windows global low-level mouse hook for observation recording.

Captures left-button clicks on the emulator window and maps desktop coordinates
to device pixel coordinates.  Uses ctypes to install a WH_MOUSE_LL hook —
the lowest-level mouse hook that runs in the installing thread's message loop.

Architecture:
  MouseClickMonitor (public API)
    └─ _MouseHookThread (dedicated thread with Windows message pump)
         └─ LowLevelMouseProc (ctypes callback, called by Windows on each click)

IMPORTANT: WH_MOUSE_LL runs in the installing thread's message loop context.
The monitor thread MUST pump Windows messages (GetMessage + DispatchMessage).
Without the pump, the hook never fires.

Usage:
    monitor = MouseClickMonitor()
    monitor.start(emu_hwnd=12345, device_width=1600, device_height=900)
    # ... user clicks on emulator window ...
    clicks = monitor.stop()  # returns list[ClickEvent]
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Windows API constants ────────────────────────────────────────────

WH_MOUSE_LL = 14
WM_LBUTTONDOWN = 0x0201

# Load Windows DLLs
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


# ── Windows API structures ───────────────────────────────────────────

class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", POINT),
    ]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


# ── Windows API function prototypes ──────────────────────────────────

HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM,
)

_user32.SetWindowsHookExW.argtypes = [
    ctypes.c_int,            # idHook
    HOOKPROC,                # lpfn
    wintypes.HINSTANCE,      # hmod
    wintypes.DWORD,          # dwThreadId
]
_user32.SetWindowsHookExW.restype = wintypes.HHOOK

_user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
_user32.UnhookWindowsHookEx.restype = wintypes.BOOL

_user32.CallNextHookEx.argtypes = [
    wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM,
]
_user32.CallNextHookEx.restype = ctypes.c_long

_user32.GetMessageW.argtypes = [
    ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT,
]
_user32.GetMessageW.restype = wintypes.BOOL

_user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
_user32.TranslateMessage.restype = wintypes.BOOL

_user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
_user32.DispatchMessageW.restype = ctypes.c_long

_user32.PostThreadMessageW.argtypes = [
    wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
]
_user32.PostThreadMessageW.restype = wintypes.BOOL

WM_QUIT = 0x0012


# ── Click event ──────────────────────────────────────────────────────

@dataclass
class ClickEvent:
    """A single mouse click captured on the emulator window."""
    timestamp: float          # monotonic timestamp (time.monotonic())
    desktop_x: int            # Windows desktop absolute X
    desktop_y: int            # Windows desktop absolute Y
    device_x: int             # Mapped device pixel X
    device_y: int             # Mapped device pixel Y


# ── Emulator window finder (pywin32 helpers) ─────────────────────────

# Known emulator window title patterns.  Lowercased substring matches.
# Order matters: more specific patterns first to avoid false positives.
_EMULATOR_TITLE_PATTERNS: list[str] = [
    "mumuplayer",      # MuMu 12
    "mumu模拟器",       # MuMu Chinese title
    "mumu",            # MuMu (any version)
    "雷电",             # LDPlayer Chinese
    "ldplayer",        # LDPlayer English
    "ldconsole",       # LDPlayer console
    "bluestacks",      # BlueStacks
    "nox", "noxplayer",# Nox
    "memu",            # MEmu
    "gameloop",        # Tencent GameLoop
    "腾讯手游助手",      # Tencent GameLoop Chinese
    "android emulator",# Generic AVD
    "subsystem",       # Windows Subsystem for Android
    "模拟器",           # Generic Chinese emulator
    "逍遥",             # XiaoYao emulator
    "夜神",             # Ye Shen / Nox Chinese
]


def _find_window_by_class(class_name_substring: str) -> dict | None:
    """Find a visible window whose class name contains the given substring.

    MuMu 12 uses Qt-based class names like 'Qt5152QWindowIcon' — the Qt
    version number changes between releases, so an exact class match is
    fragile.  This searches by substring and picks the largest visible match.
    """
    try:
        import win32gui
    except ImportError:
        return None

    best: dict | None = None
    best_area = 0

    def _cb(hwnd: int, _ctx) -> bool:
        nonlocal best, best_area
        if not win32gui.IsWindowVisible(hwnd):
            return True
        cls = win32gui.GetClassName(hwnd)
        if class_name_substring not in cls:
            return True
        try:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            w, h = right - left, bottom - top
            if w < 200 or h < 200:
                return True
            area = w * h
            if area > best_area:
                best_area = area
                best = {
                    "hwnd": hwnd,
                    "title": win32gui.GetWindowText(hwnd),
                    "rect": (left, top, right, bottom),
                    "class": cls,
                }
        except Exception:
            pass
        return True

    try:
        win32gui.EnumWindows(_cb, None)
    except Exception:
        pass
    return best


def _list_all_visible_windows() -> list[str]:
    """Debug helper: list all visible windows with non-empty titles."""
    try:
        import win32gui
    except ImportError:
        return ["<pywin32 not available>"]
    titles: list[str] = []

    def _cb(hwnd: int, _ctx) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        try:
            t = win32gui.GetWindowText(hwnd)
            if t and len(t.strip()) > 1:
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                w, h = right - left, bottom - top
                cls = win32gui.GetClassName(hwnd)
                titles.append(f"  '{t}' cls={cls} size={w}x{h}")
        except Exception:
            pass
        return True

    win32gui.EnumWindows(_cb, None)
    return titles


def find_mumu_window() -> dict | None:
    """Find the emulator window for mouse click coordinate mapping.

    Tries multiple strategies in order:
      1. Known emulator title patterns (MuMu, LDPlayer, BlueStacks, etc.)
      2. Qt-based windows (MuMu 12 uses Qt5xxQWindowIcon classes)
      3. Largest visible window that isn't a desktop/browser/IDE/tool window

    Returns dict with hwnd, title, rect, or None if no plausible window found.
    """
    # Ensure DPI awareness BEFORE GetWindowRect queries.
    # On scaled displays, a non-DPI-aware process gets virtualized window
    # coordinates that don't match the physical mouse hook coordinates,
    # causing ALL clicks to be silently filtered.
    try:
        MouseClickMonitor._ensure_dpi_awareness()
    except Exception:
        pass

    try:
        import win32gui
    except ImportError:
        logger.warning("pywin32 not available — cannot find emulator window")
        return None

    # ── Strategy 1: Known emulator titles ──
    result: dict | None = None
    best_area = 0

    def _title_cb(hwnd: int, _ctx) -> bool:
        nonlocal result, best_area
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title or not title.strip():
            return True
        title_lower = title.lower()
        for pattern in _EMULATOR_TITLE_PATTERNS:
            if pattern in title_lower:
                try:
                    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                    w, h = right - left, bottom - top
                    if w < 200 or h < 200:
                        return True
                    area = w * h
                    if area > best_area:
                        best_area = area
                        result = {
                            "hwnd": hwnd,
                            "title": title,
                            "rect": (left, top, right, bottom),
                        }
                except Exception:
                    pass
                break  # Found a match for this window, no need to try other patterns
        return True

    try:
        win32gui.EnumWindows(_title_cb, None)
    except Exception as e:
        logger.debug("EnumWindows (titles) failed: %s", e)

    if result:
        logger.debug("Found emulator by title: '%s' hwnd=%d rect=%s",
                    result["title"], result["hwnd"], result["rect"])
        return result

    # ── Strategy 2: Qt-based windows (MuMu 12 uses Qt5xxQWindowIcon) ──
    result = _find_window_by_class("QWindowIcon")
    if result and result.get("title", "").strip():
        logger.debug("Found emulator by Qt class: '%s' hwnd=%d",
                    result["title"], result["hwnd"])
        return result

    # ── Strategy 3: Largest visible window, excluding known non-emulator ──
    excluded_classes = {
        "progman", "workerw", "shell_traywnd",            # Windows desktop/shell
        "taskbar", "taskbartraywnd", "rebarwindow32",
        "chrome_widgetwin_1", "mozillawindowclass",       # Browsers
        "notepad", "cabinetwndclass", "explorer",          # System tools
        "consolewindowclass", "conemu",                    # Terminals
        "afx:", "wpf:", "windowsforms10",                 # .NET/VS windows
        "barbuttonmessagewindow",
    }

    largest: dict | None = None
    largest_area = 0

    def _largest_cb(hwnd: int, _ctx) -> bool:
        nonlocal largest, largest_area
        if not win32gui.IsWindowVisible(hwnd):
            return True
        try:
            cls = win32gui.GetClassName(hwnd).lower()
            for excl in excluded_classes:
                if excl in cls:
                    return True
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            w, h = right - left, bottom - top
            if w < 400 or h < 400:
                return True
            area = w * h
            if area > largest_area:
                largest_area = area
                largest = {
                    "hwnd": hwnd,
                    "title": win32gui.GetWindowText(hwnd),
                    "rect": (left, top, right, bottom),
                    "class": cls,
                }
        except Exception:
            pass
        return True

    try:
        win32gui.EnumWindows(_largest_cb, None)
    except Exception:
        pass

    if largest:
        logger.debug("Found emulator by largest-window heuristic: '%s' cls=%s hwnd=%d size=%d",
                    largest["title"], largest.get("class", "?"),
                    largest["hwnd"], largest_area)
        return largest

    # ── Nothing found — log all windows for debugging ──
    all_windows = _list_all_visible_windows()
    # Log ALL windows (not just 30) when detection fails — the emulator
    # title might be something unexpected and we need the full picture.
    logger.warning(
        "No emulator window detected among %d visible windows. "
        "Full window list:\n%s",
        len(all_windows), "\n".join(all_windows),
    )
    # Also try: log desktop screen size & DPI awareness for diagnostics
    try:
        import ctypes as _ct
        user32 = _ct.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        logger.info(
            "Desktop: %dx%d. DPI awareness: try setting "
            "SetProcessDpiAwareness(2) if coords seem off.",
            screen_w, screen_h,
        )
    except Exception:
        pass
    return None


# ── MouseClickMonitor ────────────────────────────────────────────────

class MouseClickMonitor:
    """Thread-safe Windows mouse click monitor.

    Installs a WH_MOUSE_LL hook on a dedicated thread.  Captures left-click
    coordinates, filters to emulator window only, and maps to device pixels.

    IMPORTANT — DPI awareness:
      On Windows 10/11 with display scaling ≠ 100%, a non-DPI-aware process
      gets DPI virtualization: GetWindowRect returns scaled coords while
      MSLLHOOKSTRUCT.pt may report physical coords. The mismatch causes ALL
      clicks to be silently filtered.  We call SetProcessDpiAwareness(2)
      (Per-Monitor DPI Aware v2) to align both coordinate spaces so Window
      rects and mouse hook coordinates are consistent.

    Usage:
        monitor = MouseClickMonitor()
        monitor.start(emu_window=emu_info, device_width=1600, device_height=900)
        # ... clicks happen ...
        clicks = monitor.stop()  # returns list[ClickEvent]
    """

    # DPI awareness flag — set once per process.
    _dpi_awareness_set: bool = False

    @staticmethod
    def _ensure_dpi_awareness() -> None:
        """Make the process DPI-aware so GetWindowRect and mouse hook coords match.

        Must be called BEFORE any window rect queries or mouse hook installation.
        Safe to call multiple times — only the first call takes effect.
        """
        if MouseClickMonitor._dpi_awareness_set:
            return
        MouseClickMonitor._dpi_awareness_set = True
        try:
            # Per-Monitor DPI Aware v2 (Windows 10 1703+)
            # Value 2 = PROCESS_PER_MONITOR_DPI_AWARE
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            logger.info("DPI awareness set: Per-Monitor DPI Aware v2")
        except AttributeError:
            # Fallback for older systems without shcore.dll
            try:
                ctypes.windll.user32.SetProcessDPIAware()
                logger.info("DPI awareness set: System DPI Aware (legacy)")
            except Exception:
                logger.debug("SetProcessDPIAware also unavailable — "
                           "coordinate mismatch possible on scaled displays")
        except Exception as e:
            logger.warning("Failed to set DPI awareness: %s — "
                         "mouse click capture may be unreliable on scaled displays", e)

    def __init__(self) -> None:
        self._clicks: list[ClickEvent] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._hook: int | None = None  # HHOOK value
        self._thread_id: int | None = None
        self._t0: float = 0.0  # monotonic start time
        # Emulator window tracking
        self._emu_hwnd: int = 0
        self._emu_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._device_w: int = 1600
        self._device_h: int = 900

    def start(self, emu_window: dict, device_width: int, device_height: int) -> None:
        """Install the mouse hook and start the message pump thread.

        Args:
            emu_window: Result from find_mumu_window() — dict with hwnd + rect.
            device_width: Emulator device screen width in pixels (from ADB).
            device_height: Emulator device screen height in pixels.
        """
        if self._thread is not None:
            raise RuntimeError("MouseClickMonitor already started")

        # Ensure DPI awareness BEFORE any coordinate operations.
        # Without this, GetWindowRect and MSLLHOOKSTRUCT coords diverge
        # on scaled displays, causing ALL clicks to be silently filtered.
        self._ensure_dpi_awareness()

        self._emu_hwnd = emu_window.get("hwnd", 0)
        self._emu_rect = emu_window.get("rect", (0, 0, 0, 0))
        self._device_w = device_width
        self._device_h = device_height

        self._t0 = time.monotonic()
        self._clicks.clear()

        self._thread = threading.Thread(
            target=self._hook_thread,
            daemon=True,
            name="mouse-hook",
        )
        self._thread.start()

        # Wait for the hook to be installed (max 2 seconds)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._hook is not None:
                logger.info("Mouse hook installed (emu=%d, dev=%dx%d)",
                           self._emu_hwnd, device_width, device_height)
                return
            time.sleep(0.05)
        logger.warning("Mouse hook installation timed out — clicks may not be captured")

    def stop(self) -> list[ClickEvent]:
        """Uninstall the hook, stop the thread, and return captured clicks."""
        if self._thread is None:
            return []

        # Post WM_QUIT to the hook thread's message queue
        if self._thread_id is not None:
            _user32.PostThreadMessageW(
                wintypes.DWORD(self._thread_id),
                wintypes.UINT(WM_QUIT),
                wintypes.WPARAM(0),
                wintypes.LPARAM(0),
            )

        self._thread.join(timeout=3.0)
        self._thread = None
        self._hook = None
        self._thread_id = None

        with self._lock:
            clicks = list(self._clicks)
        logger.info("Mouse hook stopped: %d clicks captured", len(clicks))
        return clicks

    @property
    def click_count(self) -> int:
        with self._lock:
            return len(self._clicks)

    def drain_clicks(self) -> list[ClickEvent]:
        """Atomically retrieve and clear the current click list.

        Called by the observation recorder thread to associate clicks
        with the most recently captured frame.
        """
        with self._lock:
            clicks = self._clicks[:]
            self._clicks.clear()
        return clicks

    def update_window_rect(self, emu_window: dict) -> None:
        """Update the emulator window rect for coordinate mapping.

        Call periodically during recording to handle the user moving
        or resizing the emulator window.  Thread-safe: the hook callback
        reads these fields without a lock, so we assign atomically.
        """
        self._emu_hwnd = emu_window.get("hwnd", self._emu_hwnd)
        self._emu_rect = emu_window.get("rect", self._emu_rect)

    # ── Internal ─────────────────────────────────────────────────

    def _hook_thread(self) -> None:
        """Dedicated thread: install hook and pump Windows messages."""
        self._thread_id = _kernel32.GetCurrentThreadId()

        # Install the low-level mouse hook
        hook_proc = HOOKPROC(self._low_level_mouse_proc)
        self._hook = _user32.SetWindowsHookExW(
            WH_MOUSE_LL,
            hook_proc,
            wintypes.HINSTANCE(0),  # No DLL, this is a low-level hook
            wintypes.DWORD(0),      # 0 = global (all threads)
        )

        if not self._hook:
            err = ctypes.get_last_error()
            logger.error("SetWindowsHookEx failed: error=%d", err)
            # Keep reference to hook_proc to prevent GC
            self._hook_proc = hook_proc
            return

        # Pump messages — this is REQUIRED for WH_MOUSE_LL to work.
        # Without GetMessage, the hook never fires.
        self._hook_proc = hook_proc  # Keep alive
        logger.debug("Mouse hook thread running (tid=%d, hook=%d)",
                    self._thread_id, self._hook)

        msg = MSG()
        try:
            while True:
                ret = _user32.GetMessageW(
                    ctypes.byref(msg),
                    wintypes.HWND(0),
                    wintypes.UINT(0),
                    wintypes.UINT(0),
                )
                if ret <= 0:
                    # GetMessage returns 0 for WM_QUIT, -1 on error
                    break
                _user32.TranslateMessage(ctypes.byref(msg))
                _user32.DispatchMessageW(ctypes.byref(msg))
        except Exception as e:
            logger.error("Mouse hook message pump crashed: %s", e)
        finally:
            if self._hook:
                _user32.UnhookWindowsHookEx(self._hook)
                self._hook = None
            logger.debug("Mouse hook thread exited")

    def _low_level_mouse_proc(self, nCode: int, wParam: int, lParam: int) -> int:
        """Low-level mouse hook callback — called by Windows on each mouse event.

        We only capture WM_LBUTTONDOWN (left button press) events that occur
        within the emulator window bounds.

        IMPORTANT: This runs on the hook thread.  Do NOT call blocking
        functions or allocate large objects here.
        """
        if nCode < 0:
            return _user32.CallNextHookEx(None, nCode, wParam, lParam)

        if wParam == WM_LBUTTONDOWN:
            try:
                hook_struct = ctypes.cast(
                    lParam, ctypes.POINTER(MSLLHOOKSTRUCT),
                ).contents
                desktop_x = hook_struct.pt.x
                desktop_y = hook_struct.pt.y

                # Snapshot rect atomically — update_window_rect() may be
                # called from another thread while we're unpacking.
                rect = self._emu_rect
                left, top, right, bottom = rect
                if left <= desktop_x <= right and top <= desktop_y <= bottom:
                    # Map to device coordinates
                    window_w = right - left
                    window_h = bottom - top
                    if window_w > 0 and window_h > 0:
                        rel_x = desktop_x - left
                        rel_y = desktop_y - top
                        device_x = int(rel_x * self._device_w / window_w)
                        device_y = int(rel_y * self._device_h / window_h)

                        event = ClickEvent(
                            timestamp=time.monotonic() - self._t0,
                            desktop_x=desktop_x,
                            desktop_y=desktop_y,
                            device_x=device_x,
                            device_y=device_y,
                        )
                        with self._lock:
                            self._clicks.append(event)
            except Exception:
                # Silently swallow errors in the hook callback —
                # an exception here crashes the hook chain.
                pass

        return _user32.CallNextHookEx(None, nCode, wParam, lParam)
