"""Offline OCR tests using screenshots in data/test_fixtures/."""

from pathlib import Path

import pytest
from PIL import Image

from src.vision.ocr import ocr_engine

FIXTURES_DIR = Path("data/test_fixtures/arknights")


@pytest.mark.skipif(not FIXTURES_DIR.exists(), reason="No test fixtures yet")
def test_ocr_on_fixtures():
    pngs = list(FIXTURES_DIR.glob("*.png"))
    if not pngs:
        pytest.skip("No PNG fixtures found")

    ocr_engine.load()
    for png in pngs:
        detections = ocr_engine.read_text(png)
        print(f"{png.name}: {len(detections)} text regions detected")
        for d in detections:
            print(f"  [{d['confidence']:.2f}] {d['text']}")

        # At minimum, we should detect some text on game screens
        if "main" in png.name.lower():
            assert len(detections) > 0, f"No text found on {png.name}"
