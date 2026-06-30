"""OCR wrapper for game text detection — PaddleOCR primary, RapidOCR fallback.

Uses PaddleOCR (same PP-OCR engine as MAA) for highest Chinese text accuracy.
Falls back to RapidOCR (ONNX Runtime) when PaddleOCR is not installed or
crashes at runtime (e.g. oneDNN incompatibility on some CPUs).

Both engines share the same model architecture (PP-OCR). The difference is
the inference backend:
  - PaddleOCR: native PaddlePaddle C++ inference (same as MAA)
  - RapidOCR: ONNX Runtime (cross-platform, ~50MB, no PaddlePaddle dependency)
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Disable oneDNN/MKLDNN for PaddlePaddle 3.x on CPUs where it crashes
# (NotImplementedError in ConvertPirAttribute2RuntimeAttribute)
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("KMP_AFFINITY", "disabled")

logger = logging.getLogger(__name__)


class OCREngine:
    """Wraps PaddleOCR (primary) or RapidOCR (fallback) for game text detection.

    Both backends expose the same API:
      - read_text(image) → list[{"text", "confidence", "bbox", "center"}]
      - find_text(image, target) → best match dict or None

    PaddleOCR inference errors (e.g. ONEDNN crashes) trigger automatic fallback
    to RapidOCR at the next read_text() call.
    """

    def __init__(self) -> None:
        self._reader: Any = None
        self._backend: str = ""  # "paddle" or "rapid"
        self._load_lock = threading.Lock()
        self._read_lock = threading.Lock()  # Protects concurrent read_text calls
        self._loaded: bool = False
        self._paddle_broken: bool = False  # True if PaddleOCR crashed at inference

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend(self) -> str:
        return self._backend

    def preload(self) -> None:
        """Preload OCR model in background. Call at agent init."""
        def _load() -> None:
            self.load()
        t = threading.Thread(target=_load, daemon=True)
        t.start()

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self._load_backend()

    def _load_backend(self) -> None:
        """Load OCR backend: RapidOCR first, PaddleOCR as fallback."""
        # Primary: RapidOCR — no oneDNN, always works on Windows
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._reader = RapidOCR()
            self._reader.min_height = 10
            self._backend = "rapid"
            self._loaded = True
            logger.info("OCR: RapidOCR loaded (PP-OCRv4 via ONNX Runtime)")
            return
        except ImportError:
            logger.debug("RapidOCR not available, trying PaddleOCR...")

        # Fallback: PaddleOCR — may crash on some CPUs (oneDNN bug)
        try:
            from paddleocr import PaddleOCR
            self._reader = PaddleOCR(lang="ch")
            self._backend = "paddle"
            self._loaded = True
            logger.info("OCR: PaddleOCR loaded (PP-OCR via PaddlePaddle)")
            return
        except ImportError:
            logger.debug("PaddleOCR not available.")
        except Exception as e:
            logger.warning("PaddleOCR init failed: %s", e)

        logger.warning("No OCR engine. Install: pip install rapidocr-onnxruntime")
        self._reader = None

    def _switch_to_rapid(self) -> bool:
        """Hot-switch to RapidOCR after a PaddleOCR crash. Returns True on success.

        Must be called while holding self._read_lock to prevent concurrent
        reads from seeing a partially-switched backend.
        """
        if self._backend == "rapid":
            return True
        logger.warning("PaddleOCR inference crashed — switching to RapidOCR fallback")
        self._paddle_broken = True
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._reader = RapidOCR()
            self._backend = "rapid"
            logger.info("OCR: Switched to RapidOCR fallback")
            return True
        except ImportError:
            logger.error("RapidOCR not available for fallback — OCR disabled")
            self._reader = None
            return False

    def read_text(self, image: Image.Image | str | Path) -> list[dict[str, Any]]:
        """Read all text from an image. Thread-safe via _read_lock.

        Returns:
            List of {"text", "confidence", "bbox": (x1,y1,x2,y2), "center": (cx,cy)}.
        """
        if not self._loaded:
            self.load()
        with self._read_lock:
            return self._read_text_locked(image)

    def _read_text_locked(self, image: Image.Image | str | Path) -> list[dict[str, Any]]:
        """Internal read_text — caller must hold self._read_lock."""
        if self._reader is None:
            return []

        if self._backend == "paddle" and not self._paddle_broken:
            try:
                return self._read_text_paddle(image)
            except Exception as e:
                logger.warning("PaddleOCR read_text crashed: %s", e)
                if not self._switch_to_rapid():
                    return []
                # Fall through to RapidOCR path
        if self._backend == "rapid":
            return self._read_text_rapid(image)
        return []

    def _read_text_paddle(self, image: Image.Image | str | Path) -> list[dict[str, Any]]:
        """PaddleOCR: returns [{"text", "confidence", "bbox", "center"}, ...]."""
        # PaddleOCR 3.x only accepts numpy.ndarray or file path (str).
        # numpy.ndarray triggers intermittent oneDNN crashes on some CPUs.
        # Writing to a temp file and passing the path avoids the crash.
        import tempfile
        if isinstance(image, Image.Image):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f, format="PNG")
                tmp_path = f.name
            try:
                results = self._reader.ocr(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        elif isinstance(image, Path):
            results = self._reader.ocr(str(image))
        else:
            results = self._reader.ocr(image)
        if not results or not results[0]:
            return []

        detections: list[dict[str, Any]] = []
        for item in results[0]:
            # PaddleOCR 3.x item: (bbox_4pts, (text, confidence))
            bbox, text_info = item[0], item[1]
            text, confidence = text_info[0], float(text_info[1])

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x_min = int(min(xs))
            y_min = int(min(ys))
            x_max = int(max(xs))
            y_max = int(max(ys))

            detections.append({
                "text": text,
                "confidence": confidence,
                "bbox": (x_min, y_min, x_max, y_max),
                "center": ((x_min + x_max) // 2, (y_min + y_max) // 2),
            })
        return detections

    def _read_text_rapid(self, image: Image.Image | str | Path) -> list[dict[str, Any]]:
        """RapidOCR: returns [{"text", "confidence", "bbox", "center"}, ...]."""
        result, _ = self._reader(image)
        if result is None:
            return []

        detections: list[dict[str, Any]] = []
        for item in result:
            bbox, text, confidence = item[0], item[1], float(item[2])
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x_min = int(min(xs))
            y_min = int(min(ys))
            x_max = int(max(xs))
            y_max = int(max(ys))
            detections.append({
                "text": text,
                "confidence": confidence,
                "bbox": (x_min, y_min, x_max, y_max),
                "center": ((x_min + x_max) // 2, (y_min + y_max) // 2),
            })
        return detections

    def find_text(
        self,
        image: Image.Image | str | Path,
        target: str,
        min_confidence: float = 0.5,
    ) -> dict[str, Any] | None:
        """Find a specific text string in an image. Returns best match or None."""
        detections = self.read_text(image)
        best = None
        best_conf = 0.0
        for d in detections:
            if target in d["text"] and d["confidence"] >= min_confidence:
                if d["confidence"] > best_conf:
                    best = d
                    best_conf = d["confidence"]
        return best

    def read_region(
        self,
        image: Image.Image | str | Path,
        x: int, y: int, w: int, h: int,
    ) -> str:
        """Read text from a specific region of an image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        crop = image.crop((x, y, x + w, y + h))
        detections = self.read_text(crop)
        return " ".join(d["text"] for d in detections)


ocr_engine = OCREngine()
