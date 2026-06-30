"""Capture screenshot + OCR debug tool.

Usage:
    python scripts/capture_ui.py              # One screenshot + OCR
    python scripts/capture_ui.py --watch       # Continuous monitoring
    python scripts/capture_ui.py --save-templates  # Capture regions as templates
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.device.emulator import emulator_manager
from src.device.adb import ADBDevice
from src.vision.ocr import ocr_engine


def capture_once(device: ADBDevice) -> None:
    path = device.save_screenshot()
    print(f"Screenshot: {path}")

    ocr_engine.load()
    detections = ocr_engine.read_text(path)

    print(f"\nOCR detected {len(detections)} text regions:")
    print("-" * 60)
    for d in sorted(detections, key=lambda x: (x["bbox"][1], x["bbox"][0])):
        conf_bar = "█" * int(d["confidence"] * 10)
        print(f"  [{conf_bar:<10}] {d['text']:<20} @ ({d['center'][0]}, {d['center'][1]})")


def watch_continuous(device: ADBDevice) -> None:
    print("Continuous monitoring (Ctrl+C to stop)...")
    last_hash = ""
    ocr_engine.load()

    try:
        while True:
            path = device.save_screenshot()

            import hashlib
            from PIL import Image
            img = Image.open(path)
            small = img.resize((64, 64)).convert("L")
            screen_hash = hashlib.sha256(bytes(small.tobytes())).hexdigest()[:8]

            if screen_hash != last_hash:
                last_hash = screen_hash
                detections = ocr_engine.read_text(path)
                texts = [d["text"] for d in detections[:15]]
                print(f"[{screen_hash}] {' | '.join(texts)}")

            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nStopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture UI for debugging")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--output", type=str, help="Save OCR result to JSON file")
    args = parser.parse_args()

    serial = emulator_manager.first_online
    if not serial:
        print("No ADB device found.")
        sys.exit(1)

    device = ADBDevice(serial)
    if not device.heartbeat():
        print(f"Device {serial} is not responding.")
        sys.exit(1)

    print(f"Connected to: {serial}")
    print(f"Screen size: {device.get_screen_size()}")

    if args.watch:
        watch_continuous(device)
    else:
        capture_once(device)


if __name__ == "__main__":
    main()
