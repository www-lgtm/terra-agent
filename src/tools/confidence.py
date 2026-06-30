"""Confidence thresholds and adaptive learning.

Three signal sources:
- OCR confidence (PaddleOCR native score 0-1)
- Template match score (OpenCV matchTemplate 0-1)
- VLM semantic consistency (LLM comparison)

Two intervention points:
1. Pre-execution: confidence check before tap
2. Post-execution: screen change verification

Thresholds are adjusted over time based on user feedback.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIDENCE_FILE = Path("data/memory/confidence.json")


@dataclass
class Thresholds:
    ocr_auto: float = 0.8       # Auto-execute when OCR confidence >= this
    ocr_confirm: float = 0.5    # Ask user when between ocr_confirm and ocr_auto
                                # Below ocr_confirm: abandon, ask user directly


@dataclass
class Stats:
    ocr_confirmations: int = 0      # Times user was asked about OCR result
    ocr_confirmations_ok: int = 0   # Times user confirmed OCR was correct

    @property
    def ocr_confirm_rate(self) -> float:
        if self.ocr_confirmations == 0:
            return 1.0
        return self.ocr_confirmations_ok / self.ocr_confirmations


class ConfidenceManager:
    """Manages confidence thresholds with adaptive learning."""

    def __init__(self) -> None:
        self.thresholds = Thresholds()
        self.stats = Stats()

    def evaluate_ocr(self, confidence: float) -> str:
        """Evaluate OCR confidence. Returns: 'auto', 'confirm', or 'ask'."""
        if confidence >= self.thresholds.ocr_auto:
            return "auto"
        elif confidence >= self.thresholds.ocr_confirm:
            return "confirm"
        return "ask"

    def record_feedback(self, source: str, was_correct: bool) -> None:
        """Record user feedback to adjust thresholds over time."""
        if source == "ocr":
            self.stats.ocr_confirmations += 1
            if was_correct:
                self.stats.ocr_confirmations_ok += 1

        self._adapt()

    def _adapt(self) -> None:
        """Adjust thresholds based on accumulated feedback. Requires 10+ samples."""
        if self.stats.ocr_confirmations >= 10:
            rate = self.stats.ocr_confirm_rate
            if rate >= 0.8:
                self.thresholds.ocr_auto = max(0.5, self.thresholds.ocr_auto - 0.05)
                self.thresholds.ocr_confirm = max(0.3, self.thresholds.ocr_confirm - 0.05)
            elif rate <= 0.4:
                self.thresholds.ocr_auto = min(0.95, self.thresholds.ocr_auto + 0.05)

    def save(self) -> None:
        CONFIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "thresholds": {
                "ocr_auto": self.thresholds.ocr_auto,
                "ocr_confirm": self.thresholds.ocr_confirm,
            },
            "stats": {
                "ocr_confirmations": self.stats.ocr_confirmations,
                "ocr_confirmations_ok": self.stats.ocr_confirmations_ok,
            },
        }
        CONFIDENCE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not CONFIDENCE_FILE.exists():
            return
        data = json.loads(CONFIDENCE_FILE.read_text())
        t = data.get("thresholds", {})
        self.thresholds = Thresholds(
            ocr_auto=t.get("ocr_auto", 0.8),
            ocr_confirm=t.get("ocr_confirm", 0.5),
        )
        s = data.get("stats", {})
        self.stats = Stats(
            ocr_confirmations=s.get("ocr_confirmations", 0),
            ocr_confirmations_ok=s.get("ocr_confirmations_ok", 0),
        )


confidence_mgr = ConfidenceManager()
