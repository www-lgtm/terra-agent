"""ADB connection test: list devices, screenshot, get screen size.

Usage:
    python scripts/test_adb.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.device.emulator import emulator_manager
from src.device.adb import ADBDevice


def main() -> None:
    print("ADB Connection Test")
    print("=" * 50)

    # Discover devices
    print("\n1. Scanning for devices...")
    devices = emulator_manager.discover()
    if not devices:
        print("   No devices found. Make sure an emulator is running.")
        return

    for serial, state in devices:
        status = "ONLINE" if state == "device" else state
        print(f"   {serial}: {status}")

    # Pick first online
    serial = emulator_manager.first_online
    if not serial:
        print("   No online device found.")
        return

    print(f"\n2. Testing device: {serial}")

    device = ADBDevice(serial)

    # Heartbeat
    print("   Heartbeat...", end=" ")
    if device.heartbeat():
        print("OK")
    else:
        print("FAILED")
        return

    # Screen size
    try:
        w, h = device.get_screen_size()
        print(f"   Screen size: {w}x{h}")
    except Exception as e:
        print(f"   Screen size: FAILED ({e})")

    # Screenshot
    try:
        path = device.save_screenshot()
        size_kb = path.stat().st_size / 1024
        print(f"   Screenshot: {path} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"   Screenshot: FAILED ({e})")

    print("\nADB test complete.")


if __name__ == "__main__":
    main()
