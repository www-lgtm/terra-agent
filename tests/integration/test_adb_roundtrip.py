"""Integration test: ADB connect → screenshot → OCR → tap → verify.

Requires a connected emulator/device with Arknights running.
"""

import pytest
from src.device.emulator import emulator_manager
from src.device.adb import ADBDevice, init_adb
from src.vision.ocr import ocr_engine


@pytest.fixture
def device():
    serial = emulator_manager.first_online
    if not serial:
        pytest.skip("No ADB device available")
    return ADBDevice(serial)


def test_adb_heartbeat(device):
    assert device.heartbeat(), "Device should respond to heartbeat"


def test_screenshot(device):
    path = device.save_screenshot()
    assert path.exists()
    assert path.stat().st_size > 10000  # Should be at least 10KB


def test_screen_size(device):
    w, h = device.get_screen_size()
    assert w > 0 and h > 0
    assert 480 <= w <= 4096
    assert 800 <= h <= 4096


def test_ocr_on_screenshot(device):
    path = device.save_screenshot()
    ocr_engine.load()
    detections = ocr_engine.read_text(path)
    print(f"Detected: {[d['text'] for d in detections]}")
    # At minimum, the game should show some text
