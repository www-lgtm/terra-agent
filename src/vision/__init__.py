"""Vision layer: template matching (Box scan), OCR, and VLM description."""

from src.vision.ocr import OCREngine, ocr_engine
from src.vision.template_match import TemplateMatcher, template_matcher
from src.vision.vlm import VLMDescriptor, vlm_descriptor

__all__ = [
    "OCREngine", "ocr_engine",
    "TemplateMatcher", "template_matcher",
    "VLMDescriptor", "vlm_descriptor",
]
