"""Arknights game adapter — game-specific screen understanding and VLM→OCR mapping."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# VLM outputs English button names, but OCR can't read Arknights art fonts.
# Map common VLM outputs to their Chinese equivalents for OCR search.
VLM_TO_CN: dict[str, str] = {
    "mission": "任务",
    "terminal": "终端",
    "base": "基建",
    "store": "采购中心",
    "operator": "干员",
    "squads": "编队",
    "friends": "好友",
    "archives": "档案",
    "recruitment": "公开招募",
    "combat": "作战",
    "battle": "作战",
    "supply": "物资筹备",
    "chip": "芯片搜索",
    "annihilation": "剿灭作战",
    "main theme": "主线",
    "main story": "主线",
    "event": "活动",
    "shop": "采购中心",
    "credit store": "信用商店",
    "auto deploy": "代理指挥",
    "start": "开始行动",
    "begin": "开始行动",
    "mission complete": "行动结束",
}

# Known Arknights screens and their key buttons
SCREENS = {
    "main": {
        "name": "主界面",
        "buttons": ["作战", "干员", "采购中心", "基建", "任务", "招募"],
    },
    "battle_select": {
        "name": "作战选择",
        "buttons": ["物资筹备", "芯片搜索", "主线", "活动", "剿灭作战"],
    },
    "base": {
        "name": "基建",
        "buttons": ["制造站", "贸易站", "发电站", "控制中枢", "宿舍"],
    },
}


def identify_screen(ocr_texts: list[str]) -> str | None:
    """Identify current Arknights screen from OCR text."""
    for screen_id, screen_info in SCREENS.items():
        matches = sum(1 for btn in screen_info["buttons"] if any(btn in t for t in ocr_texts))
        if matches >= 2:
            return screen_id
    return None


def get_vlm_to_cn() -> dict[str, str]:
    """Return the VLM English→Chinese mapping for this game."""
    return VLM_TO_CN
