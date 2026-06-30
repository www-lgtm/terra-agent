"""Operator card locator — MAA OperBoxImageAnalyzer exact port.

Exact replica of MAA's algorithm:
  1. Match 9 profession flag templates within precise row ROIs
  2. NMS on flag rects (MAA-style intersection/area threshold)
  3. Sort by row then horizontal (MAA Y-tolerance clustering)
  4. Each field (name, elite, level, potential) extracted from precise
     pixel offsets relative to the flag rect (from MAA tasks.json)
  5. Name OCR: MAA OperNameAnalyzer boundary detection (binarize →
     find text edges → trim → OCR on COLOR image)
  6. OCR results post-processed with CharsNameOcrReplace rules (MAA)
  7. (profession, name) joint database lookup (MAA find_oper) — no fuzzy fallback
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from config.settings import config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# MAA constants — all values from resource/tasks/tasks.json (1280×720 base)
# All pixel values are SCALED to device screen width at runtime.
# ═══════════════════════════════════════════════════════════════════════════

MAA_BASE_WIDTH = 1280
MAA_BASE_HEIGHT = 720

# Profession icon search ROIs (x, y, w, h)
#   OperBoxFlagRoleTopROI:    roi: [0, 78, 1145, 50]
#   OperBoxFlagRoleBottomROI: roi: [0, 394, 1145, 50]
ROLE_TOP_ROI = (0, 78, 1145, 50)
ROLE_BOTTOM_ROI = (0, 394, 1145, 50)

# Field ROIs relative to profession flag rect top-left corner
#   OperBoxFlagElite: roi: [11, 171, 24, 35]   templThreshold: 0.9
#   OperBoxLevelOCR:  roi: [5, 230, 37, 23]
#   OperBoxNameOCR:   rectMove: [0, 265, 128, 22]
#   OperBoxPotential: roi: [99, 194, 24, 24]   templThreshold: 0.85
ELITE_ROI_OFFSET = (11, 171, 24, 35)
LEVEL_ROI_OFFSET = (0, 222, 50, 33)  # Expanded MAA [5,230,37,23] for higher res tolerance
NAME_ROI_OFFSET = (0, 265, 128, 22)
POTENTIAL_ROI_OFFSET = (99, 194, 24, 24)

# Name OCR binarization params (MAA OperBoxNameOCR specialParams)
#   [160, 4, 0, 0, 3, 10] →
#     bin_threshold=160, expansion=4, trim(0,0), bottom_line=3, width_threshold=10
NAME_BIN_LOWER = 160
NAME_BIN_UPPER = 255
NAME_BIN_EXPANSION = 4
NAME_BOTTOM_LINE = 3
NAME_WIDTH_THRESHOLD = 10

# Template match thresholds — EXACT MAA values from tasks.json
ELITE_TEMPLATE_THRESHOLD = 0.9    # MAA tasks.json: "templThreshold": 0.9
POTENTIAL_TEMPLATE_THRESHOLD = 0.85  # MAA tasks.json: "templThreshold": 0.85
FLAG_TEMPLATE_THRESHOLD = 0.65    # balanced: enough recall for all professions, NMS handles duplicates

# NMS: MAA uses intersect_area > 0.7 * own_area (VisionHelper.h:140)
NMS_AREA_RATIO = 0.7


# ── Scale helpers ──────────────────────────────────────────────────────────

def _scale_factor(screen_w: int) -> float:
    return screen_w / MAA_BASE_WIDTH


def _scale_rect(rect: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    """Scale a (x, y, w, h) rect from 1280-base to device pixels."""
    return tuple(int(v * scale) for v in rect)


def _offset_rect(base_rect: tuple[int, int, int, int],
                 offset: tuple[int, int, int, int],
                 scale: float) -> tuple[int, int, int, int]:
    """MAA rect.move() equivalent: (base.x + off.x, base.y + off.y, off.w, off.h)."""
    bx, by, _bw, _bh = base_rect
    dx, dy, dw, dh = offset
    return (
        bx + int(dx * scale),
        by + int(dy * scale),
        bx + int((dx + dw) * scale),
        by + int((dy + dh) * scale),
    )


# ── Profession mapping (matches MAA's OperBoxFlagRole1~9 templates) ──

PROFESSION_MAP: dict[str, str] = {
    "role_1": "术师",     # Caster
    "role_2": "医疗",     # Medic
    "role_3": "先锋",     # Pioneer
    "role_4": "狙击",     # Sniper
    "role_5": "特种",     # Special
    "role_6": "辅助",     # Support
    "role_7": "重装",     # Tank
    "role_8": "近卫",     # Warrior
    "role_9": "近卫",     # Warrior (duplicate, MAA has two warrior roles)
}

PROFESSION_TEMPLATE_NAMES = list(PROFESSION_MAP.keys())

# Map Chinese profession name → battle_data.json profession enum string
_PROFESSION_TO_ENUM: dict[str, str] = {
    "术师": "CASTER",
    "医疗": "MEDIC",
    "先锋": "PIONEER",
    "狙击": "SNIPER",
    "特种": "SPECIAL",
    "辅助": "SUPPORT",
    "重装": "TANK",
    "近卫": "WARRIOR",
}

# ═══════════════════════════════════════════════════════════════════════════
# MAA CharsNameOcrReplace — OCR correction rules for operator names
# Exact port from MAA tasks.json CharsNameOcrReplace + NumberOcrReplace
# ═══════════════════════════════════════════════════════════════════════════

# Number OCR corrections (MAA NumberOcrReplace)
NUMBER_OCR_REPLACE: list[tuple[str, str]] = [
    (r"[Oo]", "0"),
    (r"[Ii]", "1"),
    (r"[Ll]", "1"),
    (r"[\{\}\[\]]", "1"),
    ("B", "8"),
    ("台", "8"),
    ("十", "+"),
    ("萬", "万"),
    ("만", "万"),
    ("億", "亿"),
    ("억", "亿"),
    (r"^\.", ""),
    (" ", ""),
    ("S", "5"),
    ("s", "5"),
    ("g", "9"),
    ("q", "9"),
    ("T", "7"),
    ("A", "4"),
    ("Z", "2"),
    ("z", "2"),
]

# Character name OCR corrections (MAA CharsNameOcrReplace)
CHARS_NAME_OCR_REPLACE: list[tuple[str, str]] = [
    # Strip leading garbage
    (r"^(?:[<4~]|《|。ぐ)+", ""),
    (r"^c(?!ast)", ""),
    (r"^\^一", ""),
    (r"^[_<>]*(?=.)", ""),
    (r"[_<>]*$", ""),
    # Specific operator name fixes
    (r".*弦惊.*", "“弦惊”"),
    (r".*亚梅塔", "菲亚梅塔"),
    (r".*逍遥.*", "“逍遥”"),
    (r".*清平.*", "“清平”"),
    (r".*打字机.*", "“打字机”"),
    (r".*耀阳.*", "“耀阳”"),
    ("玛恩.*", "玛恩纳"),
    (r".*默德克萨.*", "缄默德克萨斯"),
    (r".*芜拉普兰.*", "荒芜拉普兰德"),
    (r"^重岳.+", "重岳"),
    (r"(麒.+夜刀|.*麟.*夜刀)$", "麒麟R夜刀"),
    (r".*龙.*黑角$", "火龙S黑角"),
    ("[Uu]-[O0o]f{1,2}icial", "U-Official"),
    (r".*威龙陈", "假日威龙陈"),
    ("青积", "青枳"),
    (r".*烬艾雅法.*", "纯烬艾雅法拉"),
    (r"^[委泰黍]$", "黍"),
    (r"^[H夕]$", "夕"),
    ("12E", "12F"),
    (r"归.*幽灵鲨", "归溟幽灵鲨"),
]


def _apply_ocr_replace(text: str, rules: list[tuple[str, str]]) -> str:
    """Apply regex-based OCR replacement rules sequentially (MAA ocrReplace)."""
    result = text.strip()
    for pattern, replacement in rules:
        result = re.sub(pattern, replacement, result)
    return result


def _correct_ocr_number(text: str) -> str:
    """Apply NumberOcrReplace corrections to OCR text."""
    return _apply_ocr_replace(text, NUMBER_OCR_REPLACE)


# ═══════════════════════════════════════════════════════════════════════════
# Dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OperatorCard:
    """A detected operator card on the OperBox screen.

    profession_icon_bbox is the MAA-style flag rect that anchors all field ROIs.
    """
    profession: str = ""          # Chinese profession name
    profession_icon_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    profession_icon_center: tuple[int, int] = (0, 0)

    # Field detection results
    name: str = ""
    level: int = 0
    elite: int = 0     # 0, 1, 2
    potential: int = 1
    rarity: int = 0

    # Confidence scores
    flag_score: float = 0.0
    name_score: float = 0.0

    # Screen dimensions (for scaling)
    screen_w: int = 0
    screen_h: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Card detection — MAA port
# ═══════════════════════════════════════════════════════════════════════════

def _nms_flag_rects(cards: list[OperatorCard], area_ratio: float = NMS_AREA_RATIO) -> list[OperatorCard]:
    """MAA-style NMS (VisionHelper.h:123). Sort by score desc, suppress overlapping.

    MAA NMS algorithm:
      sort by score desc → for each kept box, suppress any remaining box
      where intersect_area > threshold * own_area.
    """
    if not cards:
        return []

    # Sort by flag_score descending
    cards_sorted = sorted(cards, key=lambda c: -c.flag_score)
    kept: list[OperatorCard] = []

    for i, card in enumerate(cards_sorted):
        if card.flag_score < 0.1:
            continue
        kept.append(card)

        fx1, fy1, fx2, fy2 = card.profession_icon_bbox
        # Suppress all remaining cards that overlap too much
        for j in range(i + 1, len(cards_sorted)):
            other = cards_sorted[j]
            if other.flag_score <= 0:
                continue
            ox1, oy1, ox2, oy2 = other.profession_icon_bbox

            # Intersection area
            ix1 = max(fx1, ox1)
            iy1 = max(fy1, oy1)
            ix2 = min(fx2, ox2)
            iy2 = min(fy2, oy2)

            if ix1 >= ix2 or iy1 >= iy2:
                continue  # no intersection

            inter_area = (ix2 - ix1) * (iy2 - iy1)
            other_area = (ox2 - ox1) * (oy2 - oy1)

            if other_area > 0 and inter_area > area_ratio * other_area:
                other.flag_score = -1  # suppress

    logger.debug("NMS: %d → %d cards (ratio=%.1f)", len(cards), len(kept), area_ratio)
    return kept


def _sort_by_horizontal_(cards: list[OperatorCard]) -> list[OperatorCard]:
    """MAA sort_by_horizontal_ (VisionHelper.h:74-79).

    | 1 2 3 4 |
    | 5 6 7 8 |

    Y-tolerance clustering: if |y1 - y2| < 5px → same row, sort by X.
    Otherwise sort by Y. Uses flag rect top-left corner (MAA convention).

    NOTE: MAA's comparator violates transitivity (e.g. A≈B, B≈C but A≠C
    when Y differences straddle 5px). We implement a stable two-pass:
      1. Sort by Y → cluster into rows (adjacent Y diff < 5px)
      2. Sort each row by X → concatenate.
    This produces identical results for all well-formed card layouts.
    """
    if not cards:
        return cards

    # Pass 1: Sort by flag rect Y, then cluster into rows
    cards.sort(key=lambda c: c.profession_icon_bbox[1])

    rows: list[list[OperatorCard]] = []
    current_row: list[OperatorCard] = [cards[0]]
    prev_y = cards[0].profession_icon_bbox[1]

    for card in cards[1:]:
        cy = card.profession_icon_bbox[1]
        if abs(cy - prev_y) < 5:
            current_row.append(card)
        else:
            rows.append(current_row)
            current_row = [card]
        prev_y = cy
    rows.append(current_row)

    # Pass 2: Sort each row by X, then flatten
    for row in rows:
        row.sort(key=lambda c: c.profession_icon_bbox[0])

    return [card for row in rows for card in row]


def locate_operator_cards(
    screenshot: Image.Image,
    template_match_threshold: float = FLAG_TEMPLATE_THRESHOLD,
) -> list[OperatorCard]:
    """Detect operator cards by matching profession flag icons.

    MAA algorithm port:
      1. Scale precise row ROIs from 1280-base to device resolution
      2. For each of 9 profession templates, find ALL matches in top AND bottom row ROIs
      3. NMS on flag rects (MAA intersection/area threshold)
      4. Sort by row then horizontal (MAA Y-tolerance clustering)

    Args:
        screenshot: Full-screen PIL Image (RGB).
        template_match_threshold: Min template match score (0-1).

    Returns:
        List of OperatorCard sorted top-row-first, left-to-right within each row.
    """
    from src.vision.template_match import template_matcher

    _ensure_profession_templates()

    w, h = screenshot.width, screenshot.height
    scale = _scale_factor(w)

    # Scale MAA's precise row ROIs
    top_rx, top_ry, top_rw, top_rh = _scale_rect(ROLE_TOP_ROI, scale)
    bot_rx, bot_ry, bot_rw, bot_rh = _scale_rect(ROLE_BOTTOM_ROI, scale)

    # Clip to screen bounds
    top_rw = min(top_rw, w - top_rx)
    top_rh = min(top_rh, h - top_ry)
    bot_rw = min(bot_rw, w - bot_rx)
    bot_rh = min(bot_rh, h - bot_ry)

    if top_rh <= 0 or bot_rh <= 0:
        logger.warning("OperBox row ROIs out of screen bounds (screen=%dx%d)", w, h)
        return []

    all_matches: list[dict] = []

    # Scale min_distance to device resolution (MAA: min(cols, rows) / 2)
    min_dist = max(20, int(20 * scale))

    for tname in PROFESSION_TEMPLATE_NAMES:
        for row_name, (rx, ry, rw, rh) in [
            ("top", (top_rx, top_ry, top_rw, top_rh)),
            ("bottom", (bot_rx, bot_ry, bot_rw, bot_rh)),
        ]:
            row_img = screenshot.crop((rx, ry, rx + rw, ry + rh))
            matches = template_matcher.find_all_matches(
                row_img, tname,
                threshold=template_match_threshold,
                min_distance=min_dist,
                grayscale=True,
            )
            # Adjust coordinates back to full-screen
            for m in matches:
                cx, cy = m["center"]
                m["center"] = (cx + rx, cy + ry)
                bx1, by1, bx2, by2 = m["bbox"]
                m["bbox"] = (bx1 + rx, by1 + ry, bx2 + rx, by2 + ry)
                m["_profession"] = PROFESSION_MAP[tname]
                m["_template"] = tname
            all_matches.extend(matches)

    if not all_matches:
        logger.debug("locate_operator_cards: no profession icons matched")
        return []

    # Build OperatorCard objects
    cards = []
    for m in all_matches:
        card = OperatorCard(
            profession=m["_profession"],
            profession_icon_bbox=m["bbox"],
            profession_icon_center=m["center"],
            flag_score=m["score"],
            screen_w=w,
            screen_h=h,
        )
        cards.append(card)

    # MAA NMS
    cards = _nms_flag_rects(cards)

    # MAA sort_by_horizontal_
    cards = _sort_by_horizontal_(cards)

    logger.debug("locate_operator_cards: %d cards detected (after NMS, %d raw)",
                 len(cards), len(all_matches))
    return cards


# ═══════════════════════════════════════════════════════════════════════════
# Name ROI preprocessing — MAA OperNameAnalyzer exact port
# ═══════════════════════════════════════════════════════════════════════════

def _prep_name_for_ocr(
    crop_rgb: Image.Image,
    upscale: int = 3,
) -> Image.Image | None:
    """Prepare name ROI for OCR by upscaling.

    Game operator names in the name ROI are typically 12-18px tall.
    PP-OCR's rec model resizes to 48px — upscaling 3x brings text to
    36-54px, closest to the model's training distribution.

    Higher upscale (5x=60-90px) was tested and REDUCED accuracy — the extra
    pixelation introduces artifacts that the detection model misreads as
    extra characters.

    Returns:
        Upscaled PIL Image, or None if input is too small.
    """
    if crop_rgb.width < 5 or crop_rgb.height < 5:
        return None
    w, h = crop_rgb.size
    return crop_rgb.resize((w * upscale, h * upscale), Image.LANCZOS)


# ═══════════════════════════════════════════════════════════════════════════
# Field-level recognizers
# ═══════════════════════════════════════════════════════════════════════════

def classify_elite_in_roi(
    screenshot: Image.Image,
    card: OperatorCard,
    threshold: float = ELITE_TEMPLATE_THRESHOLD,
) -> int:
    """Template-match E1/E2 badge in precise ROI offset from flag rect.

    Uses MAA's BestMatcher approach: match against both E1 and E2 templates,
    return the best match (E2 checked first, as in MAA).

    Returns 0, 1, or 2.
    """
    from src.vision.template_match import template_matcher

    scale = _scale_factor(card.screen_w)
    elite_x1, elite_y1, elite_x2, elite_y2 = _offset_rect(
        card.profession_icon_bbox, ELITE_ROI_OFFSET, scale,
    )

    if elite_x2 <= elite_x1 or elite_y2 <= elite_y1:
        return 0

    roi_img = screenshot.crop((elite_x1, elite_y1, elite_x2, elite_y2))

    # Try E2 first, then E1 (MAA order: OperBoxFlagElite1.png, OperBoxFlagElite2.png)
    for elite_val, tname in [(2, "operbox_elite_2"), (1, "operbox_elite_1")]:
        if tname not in template_matcher.loaded_templates:
            continue
        result = template_matcher.match(roi_img, tname, threshold=threshold, grayscale=True)
        if result:
            return elite_val

    return 0


def ocr_level_in_roi(
    screenshot: Image.Image,
    card: OperatorCard,
) -> int:
    """OCR the level number from precise ROI offset from flag rect.

    Uses NumberOcrReplace cleanup.
    Returns level as int (1-90, defaults to 1 on failure).
    """
    from src.vision.ocr import ocr_engine

    scale = _scale_factor(card.screen_w)
    lv_x1, lv_y1, lv_x2, lv_y2 = _offset_rect(
        card.profession_icon_bbox, LEVEL_ROI_OFFSET, scale,
    )

    if lv_x2 <= lv_x1 or lv_y2 <= lv_y1:
        return 1

    roi_img = screenshot.crop((lv_x1, lv_y1, lv_x2, lv_y2))

    # Try upscaled first (better for tiny text), fallback to 1x
    all_raw_texts: list[str] = []
    for img_to_ocr in [
        roi_img.resize((roi_img.width * 3, roi_img.height * 3), Image.LANCZOS),
        roi_img,
    ]:
        detections = ocr_engine.read_text(img_to_ocr)
        for d in detections:
            text = _correct_ocr_number(d["text"].strip())
            # Remove "LV", "Lv", "lv" prefix
            text = re.sub(r'^[Ll][Vv]\s*', '', text)
            all_raw_texts.append(text)
        if all_raw_texts:
            break

    # Try each detection individually first
    for text in all_raw_texts:
        match = re.search(r"(\d{1,2})", text)
        if match:
            level = int(match.group(1))
            if 1 <= level <= 90:
                return level

    # Fallback: concatenate all text (handles split digits like "9" + "0")
    combined = "".join(all_raw_texts)
    combined = re.sub(r'[^0-9]', '', combined)
    if combined and 1 <= int(combined) <= 90:
        return int(combined)

    return 1


def ocr_name_in_roi(
    screenshot: Image.Image,
    card: OperatorCard,
    all_oper_names: list[str] | None = None,
) -> tuple[str, float]:
    """OCR operator name using MAA OperNameAnalyzer preprocessing + find_oper() match.

    MAA pipeline (exact port):
      1. Crop name ROI at precise offset from flag rect
      2. OperNameAnalyzer: binarize → boundary detection → trim → COLOR OCR
      3. Apply CharsNameOcrReplace rules
      4. Match against known operator names using find_oper() (profession, name)
         joint database lookup — exact MAA behavior

    KEY DIFFERENCE from previous version:
      - No more SequenceMatcher fuzzy match (0.45 cutoff was too low)
      - No more substring fallback heuristics
      - Uses MAA's find_oper(role, name) as the single authoritative matcher
      - If find_oper() returns None, the card is skipped (no guesswork)

    Returns (name, confidence_score).
    """
    from src.vision.ocr import ocr_engine

    scale = _scale_factor(card.screen_w)
    nm_x1, nm_y1, nm_x2, nm_y2 = _offset_rect(
        card.profession_icon_bbox, NAME_ROI_OFFSET, scale,
    )

    if nm_x2 <= nm_x1 or nm_y2 <= nm_y1:
        return "", 0.0

    name_crop = screenshot.crop((nm_x1, nm_y1, nm_x2, nm_y2))

    # Multi-pass OCR: try 3x upscale first, then 2x, then raw crop.
    # Different upscale factors help with different text sizes/backgrounds.
    # RapidOCR occasionally misses text at one scale but detects it at another.
    detections: list[dict] = []
    for upscale_factor in [3, 2]:
        prepped = _prep_name_for_ocr(name_crop, upscale=upscale_factor)
        if prepped is None:
            continue
        detections = ocr_engine.read_text(prepped)
        if detections:
            break

    # Last resort: raw crop (no upscale)
    if not detections:
        detections = ocr_engine.read_text(name_crop)

    if not detections:
        return "", 0.0

    # Combine all OCR text fragments — sort LEFT TO RIGHT (by bbox x position),
    # not by confidence. Chinese text reads left to right.
    detections_sorted = sorted(detections, key=lambda d: d["bbox"][0])
    all_raw = "".join(d["text"].strip() for d in detections_sorted)

    score = max(d.get("confidence", 0) for d in detections)

    if not all_raw:
        return "", 0.0

    # Apply CharsNameOcrReplace
    raw_cleaned = _apply_ocr_replace(all_raw, CHARS_NAME_OCR_REPLACE).strip()

    if not raw_cleaned:
        return "", 0.0

    # ── MAA-style find_oper(role, name) match — the primary matcher ──
    oper_data = find_oper(raw_cleaned, card.profession)
    if oper_data:
        return raw_cleaned, max(score, 0.9)

    # ── Vocabulary-constrained fuzzy match (equivalent to MAA's set_required) ──
    # OCR produces noisy text (especially for multi-char names on RapidOCR).
    # Constrain to known operator names using difflib — the same approach as
    # MAA's set_required() which restricts OCR output to a known character set.
    if all_oper_names:
        from difflib import get_close_matches
        candidates = all_oper_names

        # Profession-filtered first
        if card.profession:
            prof_enum = _PROFESSION_TO_ENUM.get(card.profession)
            if prof_enum:
                prof_names = {n for n, roles in _operator_db_by_role.items() if prof_enum in roles}
                if prof_names:
                    candidates = list(prof_names)

        matches = get_close_matches(raw_cleaned, candidates, n=3, cutoff=0.5)
        if matches:
            best = matches[0]
            # Require unambiguous: best match must be significantly better than runner-up
            if len(matches) == 1:
                return best, max(score, 0.7)
            # Check that best is clearly better
            sim1 = SequenceMatcher(None, raw_cleaned, best).ratio()
            sim2 = SequenceMatcher(None, raw_cleaned, matches[1]).ratio()
            if sim1 - sim2 > 0.1:
                return best, max(score, 0.7)

    # ── Exact match fallback against all names ──
    if all_oper_names:
        for name in all_oper_names:
            if raw_cleaned == name:
                return name, max(score, 0.9)

    # Substring match: OCR text contains known name (conservative)
    if all_oper_names:
        for name in all_oper_names:
            if len(name) >= 2 and name in raw_cleaned:
                return name, max(score, 0.8)

    # If we get here, the OCR result could not be matched to any known operator.
    # Return the raw text at low confidence — caller should decide whether to use it.
    return raw_cleaned, score * 0.3


def classify_potential_in_roi(
    screenshot: Image.Image,
    card: OperatorCard,
    threshold: float = POTENTIAL_TEMPLATE_THRESHOLD,
) -> int:
    """Template-match potential icon in precise ROI offset from flag rect.

    Uses MAA's BestMatcher approach: match against all potential templates (2-6),
    return the best match. Returns 1 (default) if no match above threshold.

    Returns 1-6 (default 1 = no extra potential icon detected).
    """
    from src.vision.template_match import template_matcher

    scale = _scale_factor(card.screen_w)
    pot_x1, pot_y1, pot_x2, pot_y2 = _offset_rect(
        card.profession_icon_bbox, POTENTIAL_ROI_OFFSET, scale,
    )

    if pot_x2 <= pot_x1 or pot_y2 <= pot_y1:
        return 1

    roi_img = screenshot.crop((pot_x1, pot_y1, pot_x2, pot_y2))

    # MAA BestMatcher: match all potential templates, keep best
    best_pot = 1
    best_score = 0.0
    for pot in range(2, 7):
        tname = f"potential_{pot}"
        if tname not in template_matcher.loaded_templates:
            continue
        result = template_matcher.match(roi_img, tname, threshold=threshold, grayscale=True)
        if result and result["score"] > best_score:
            best_score = result["score"]
            best_pot = pot

    return best_pot


# ═══════════════════════════════════════════════════════════════════════════
# One-shot full-card classification
# ═══════════════════════════════════════════════════════════════════════════

def classify_card(
    screenshot: Image.Image,
    card: OperatorCard,
    all_oper_names: list[str] | None = None,
) -> OperatorCard:
    """Fill all fields using precise ROI offsets from the profession flag rect.

    Each field has its own independent, narrowly-scoped detection:
      - Name: MAA OperNameAnalyzer boundary detection → OCR → CharsNameOcrReplace → find_oper()
      - Elite: BestMatcher E1/E2 template match (threshold: 0.9)
      - Level: OCR 1-2 digit number with NumberOcrReplace
      - Potential: BestMatcher potential 2-6 template match (threshold: 0.85)
    """
    # Name (most important — do first so we can short-circuit on failure)
    name, name_score = ocr_name_in_roi(screenshot, card, all_oper_names)
    card.name = name
    card.name_score = name_score

    # Populate rarity/id from database
    if name:
        oper_data = find_oper(name, card.profession)
        if oper_data:
            card.rarity = oper_data.get("rarity", 0)

    # Elite
    card.elite = classify_elite_in_roi(screenshot, card)

    # Level
    card.level = ocr_level_in_roi(screenshot, card)

    # Potential
    card.potential = classify_potential_in_roi(screenshot, card)

    if card.name:
        logger.debug(
            "  %s | %s | E%d Lv%d | 潜%d (flag=%.2f, name=%.2f)",
            card.name, card.profession, card.elite, card.level,
            card.potential, card.flag_score, card.name_score,
        )

    return card


def batch_classify_cards(
    screenshot: Image.Image,
    cards: list[OperatorCard],
    all_oper_names: list[str] | None = None,
) -> list[OperatorCard]:
    """Classify all cards on a screen. Returns cards with fields populated."""
    for card in cards:
        classify_card(screenshot, card, all_oper_names)
    return cards


# ═══════════════════════════════════════════════════════════════════════════
# Operator database — (profession, name) joint lookup (MAA find_oper)
# ═══════════════════════════════════════════════════════════════════════════

_operator_db: dict[str, dict] = {}             # name → first match {id, rarity, profession}
_operator_db_by_role: dict[str, dict[str, dict]] = {}  # name → {profession_enum → {id, rarity}}
_operator_db_loaded = False


def _load_operator_db() -> dict[str, dict]:
    """Load operator {id, rarity, profession_enum} from MAA battle_data.json.

    Builds two indices:
      - _operator_db: name → first match (backward compat)
      - _operator_db_by_role: name → {profession → {id, rarity}} (MAA find_oper)
    """
    global _operator_db, _operator_db_by_role, _operator_db_loaded
    if _operator_db_loaded:
        return _operator_db

    battle_data = Path("d:/vsworkspace/MaaAssistantArknights/resource/battle_data.json")
    if battle_data.exists():
        try:
            data = json.loads(battle_data.read_text(encoding="utf-8"))
            for char_id, char_info in data.get("chars", {}).items():
                if isinstance(char_info, dict) and "name" in char_info:
                    name = char_info["name"]
                    profession = char_info.get("profession", "")
                    rarity = char_info.get("rarity", 0)
                    entry = {"id": char_id, "rarity": rarity, "profession": profession}

                    # Primary index: name → first match (backward compat)
                    if name not in _operator_db:
                        _operator_db[name] = entry

                    # Secondary index: name → {profession → {id, rarity}} (MAA find_oper)
                    if name not in _operator_db_by_role:
                        _operator_db_by_role[name] = {}
                    if profession:
                        _operator_db_by_role[name][profession] = entry

            logger.info("Operator DB loaded: %d operators from battle_data.json", len(_operator_db))
        except Exception as e:
            logger.warning("Failed to load battle_data.json: %s", e)

    _operator_db_loaded = True
    return _operator_db


def find_oper(name: str, profession_cn: str = "") -> dict | None:
    """MAA BattleData::find_oper equivalent.

    Args:
        name: Operator name in Chinese.
        profession_cn: Profession in Chinese (e.g. "术师", "近卫").

    Returns:
        {id, rarity, profession} dict or None if not found.
        Prefers (profession, name) joint lookup; falls back to name-only.
    """
    _load_operator_db()

    if not name:
        return None

    # Joint (profession, name) lookup first (MAA primary path)
    if profession_cn and name in _operator_db_by_role:
        prof_enum = _PROFESSION_TO_ENUM.get(profession_cn, "")
        if prof_enum and prof_enum in _operator_db_by_role[name]:
            return _operator_db_by_role[name][prof_enum]
        # If profession not found by role, try all professions for this name
        if _operator_db_by_role[name]:
            return next(iter(_operator_db_by_role[name].values()))

    # Fallback: name-only lookup
    return _operator_db.get(name)


def load_operator_names() -> list[str]:
    """Load the full list of known operator names from multiple sources."""
    names: list[str] = []

    # Source 1: MAA battle_data.json
    _load_operator_db()
    names = list(_operator_db.keys())

    # Source 2: MAA recruitment.json (supplement)
    maa_resource = Path("d:/vsworkspace/MaaAssistantArknights/resource")
    recruit_json = maa_resource / "recruitment.json"
    if recruit_json.exists():
        try:
            data = json.loads(recruit_json.read_text(encoding="utf-8"))
            for op in data.get("operators", []):
                if op.get("name") and op["name"] not in names:
                    names.append(op["name"])
        except Exception:
            pass

    # Source 3: Hardcoded fallback for common operators
    if not names:
        names = [
            "阿米娅", "银灰", "能天使", "艾雅法拉", "安洁莉娜", "闪灵", "夜莺", "星熊",
            "塞雷娅", "推进之王", "伊芙利特", "黑", "赫拉格", "莫斯提马", "麦哲伦",
            "煌", "阿", "年", "W", "温蒂", "早露", "铃兰", "棘刺", "森蚺",
            "史尔特尔", "泥岩", "山", "空弦", "夕", "凯尔希", "傀影", "歌蕾蒂娅",
        ]

    return sorted(set(names))


# ═══════════════════════════════════════════════════════════════════════════
# Template loading
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_profession_templates() -> None:
    """Lazily load profession flag templates from data/templates/arknights/profession/.

    Templates are sourced from MAA resource/template/OperBox/OperBoxFlagRole*.png.
    """
    from src.vision.template_match import template_matcher

    if any(t in template_matcher.loaded_templates for t in PROFESSION_TEMPLATE_NAMES):
        return

    prof_dir = Path(config.DATA_DIR) / "templates" / "arknights" / "profession"
    if prof_dir.exists():
        for png in prof_dir.glob("*.png"):
            name = png.stem  # "role_1", "role_2", ...
            template_matcher.load_template(name, png)
        logger.info("Loaded %d profession flag templates", len(list(prof_dir.glob("*.png"))))
    else:
        logger.warning("Profession template directory not found: %s", prof_dir)


def _ensure_operbox_templates() -> None:
    """Lazily load all OperBox templates (elite, potential).

    Templates are sourced from MAA resource/template/OperBox/OperBoxFlagElite*.png
    and OperBoxPotential*.png.
    """
    from src.vision.template_match import template_matcher

    needed = (
        [f"operbox_elite_{i}" for i in (1, 2)]
        + [f"potential_{i}" for i in range(2, 7)]
    )
    if all(t in template_matcher.loaded_templates for t in needed):
        return

    tmpl_dir = Path(config.DATA_DIR) / "templates" / "arknights"
    if not tmpl_dir.exists():
        logger.warning("Template directory not found: %s", tmpl_dir)
        return

    loaded = 0
    for png in tmpl_dir.glob("*.png"):
        name = png.stem
        if name in needed and name not in template_matcher.loaded_templates:
            template_matcher.load_template(name, png)
            loaded += 1
    if loaded:
        logger.info("Loaded %d operbox templates", loaded)
