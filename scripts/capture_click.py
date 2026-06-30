"""Capture mouse position as game coordinates — countdown mode.
Uses the emulator's CLIENT AREA (game render region), not the full window.
"""

import sys
import time
import ctypes
from ctypes import wintypes
import msvcrt

_user32 = ctypes.WinDLL("user32", use_last_error=True)

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Find emulator window
def find_emulator():
    try:
        import win32gui, win32con
    except ImportError:
        print("ERROR: pywin32 not installed")
        return None
    best = None
    best_area = 0
    def cb(hwnd, _):
        nonlocal best, best_area
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd).lower()
        for pattern in ["mumuplayer", "mumu", "雷电", "ldplayer", "bluestacks"]:
            if pattern in title:
                rect = win32gui.GetWindowRect(hwnd)
                left, top, right, bottom = rect
                area = (right - left) * (bottom - top)
                if area > best_area:
                    best_area = area
                    # Get CLIENT area in screen coords (excludes title bar/borders)
                    cr = win32gui.GetClientRect(hwnd)
                    ctl = win32gui.ClientToScreen(hwnd, (cr[0], cr[1]))
                    cbr = win32gui.ClientToScreen(hwnd, (cr[2], cr[3]))
                    best = {
                        "hwnd": hwnd,
                        "window_rect": rect,
                        "client_left": ctl[0],
                        "client_top": ctl[1],
                        "client_right": cbr[0],
                        "client_bottom": cbr[1],
                        "title": win32gui.GetWindowText(hwnd),
                    }
                return True
        return True
    win32gui.EnumWindows(cb, None)
    return best

emu = find_emulator()
if not emu:
    print("ERROR: No emulator window found.")
    sys.exit(1)

cl = emu["client_left"]
ct = emu["client_top"]
cr = emu["client_right"]
cb = emu["client_bottom"]
cw = cr - cl
ch = cb - ct

wl, wt, wr, wb = emu["window_rect"]
ww = wr - wl
wh = wb - wt
title_bar = ct - wt

print(f"Emulator: '{emu['title']}'")
print(f"Window:  {ww}x{wh} (title bar: {title_bar}px)")
print(f"Client:  {cw}x{ch} at desktop ({cl},{ct})-({cr},{cb})")

try:
    from src.device.adb import get_adb
    dev_w, dev_h = get_adb().get_screen_size()
except Exception:
    dev_w, dev_h = 1920, 1080

print(f"Device:  {dev_w}x{dev_h}")
print(f"Mapping: client({cw}x{ch}) -> device({dev_w}x{dev_h})")
print()
print("Press Enter, then move mouse to the bell before countdown...")
msvcrt.getch()

for i in range(5, 0, -1):
    print(f"  {i}...", flush=True)
    time.sleep(1)

print("  CAPTURE!", flush=True)
pt = wintypes.POINT()
_user32.GetCursorPos(ctypes.byref(pt))
dx, dy = pt.x, pt.y

# Map using CLIENT area, not window rect
rel_x = dx - cl
rel_y = dy - ct
device_x = int(rel_x * dev_w / cw)
device_y = int(rel_y * dev_h / ch)
x_pct = round(device_x / dev_w, 4)
y_pct = round(device_y / dev_h, 4)

# Warn if mouse wasn't in client area
in_client = (0 <= rel_x <= cw and 0 <= rel_y <= ch)

print()
print(f"Desktop:  ({dx}, {dy})")
print(f"Client:   ({rel_x}, {rel_y}) / {cw}x{ch}")
if not in_client:
    print("WARNING: mouse was NOT in the game client area!")
print(f"Device:   ({device_x}, {device_y}) / {dev_w}x{dev_h}")
print(f"\n  >>> adb_tap_position({x_pct}, {y_pct}) <<<")
