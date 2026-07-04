"""Credit shop tool — deterministic credit store shopping for Arknights.

Replaces the LLM-driven credit-shop skill with a fast, reliable script that:
1. Navigates from main screen to 采购中心 → 信用交易所
2. Collects daily credits (收取信用)
3. OCR-scans the 5×2 grid for available items with prices
4. Builds a purchase plan (priority-sorted, budget-constrained)
5. Buys each item (tap → confirm → close popup)
6. Screenshots the final state + notifies user
7. Returns to main screen

Zero LLM involvement — the agent just calls credit_shop(), then subtask_done.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time

from src.device.adb import get_adb
from src.tools.registry import ToolOutput, registry
from src.utils.dhash import compute_dhash, hamming_distance
from src.vision.ocr import ocr_engine

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

# Purchase priority: key fragments of item names → priority (lower = buy first)
_PRIORITY_MAP: list[tuple[str, int]] = [
    ("招聘许可", 0),
    ("加急许可", 1),
    ("作战记录", 2),
    ("技巧概要", 3),
]

# Fixed positions (percentage of screen)
_MAIN_STORE_PCT = (0.85, 0.93)

# The credit shop is a 5-column × 2-row grid
_GRID_COLS = 5
_ITEM_GRID_Y_TOP = 0.28        # Top of item grid
_ITEM_GRID_Y_MID = 0.55         # Split between upper/lower rows
_ITEM_GRID_Y_BOT = 0.95         # Bottom of item grid

# Stock markers — items with these are unavailable
_OUT_OF_STOCK_MARKERS = ("OUTOFST", "OFSTOCK")

# Confirmation dialog keywords
_CONFIRM_KW = ("确认", "购买物品", "购买", "确定", "是", "花费", "CONFIRM", "BUY")
_POPUP_KW = ("获得物品", "获得", "GET", "ITEM")
# Popups that may appear on the credit shop screen (before the shop grid)
_SHOP_POPUP_KW = ("好友", "剿灭", "得分", "最高", "记录", "FRIEND", "ANNIHILATION")

# Timing
_CLICK_DELAY = 0.5
_POPUP_POLL_INTERVAL = 0.3
_DHASH_THRESHOLD = 5


# ── Helpers ─────────────────────────────────────────────────────────────

def _ok(data: dict) -> ToolOutput:
    data.setdefault("success", True)
    return ToolOutput(text=json.dumps(data, ensure_ascii=False))


def _error(msg: str) -> ToolOutput:
    return ToolOutput(text=json.dumps({"success": False, "error": msg}, ensure_ascii=False))


def _notify_screenshot(message: str) -> None:
    try:
        from src.agent.screen_injector import capture_screen_jpeg
        ctx = getattr(threading.current_thread(), '_terra_agent_ctx', None)
        if ctx is None:
            return
        image_b64 = capture_screen_jpeg()
        ctx._notify(message, notify_type="screenshot", image_b64=image_b64)
        logger.info("credit_shop: notify sent — %s", message[:120])
    except Exception:
        logger.warning("credit_shop: notify failed (non-critical)", exc_info=True)


def _wait_for_screen_change(adb, pre_dhash, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(_POPUP_POLL_INTERVAL)
        curr = adb.get_screenshot_image()
        dist = hamming_distance(pre_dhash, compute_dhash(curr))
        if dist >= _DHASH_THRESHOLD:
            return True
    return False


def _wait_for_text(adb, keywords: tuple[str, ...], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(_POPUP_POLL_INTERVAL)
        dets = ocr_engine.read_text(adb.get_screenshot_image())
        for d in dets:
            for kw in keywords:
                if kw in d["text"]:
                    return True
    return False


def _wait_for_text_gone(adb, keywords: tuple[str, ...], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(_POPUP_POLL_INTERVAL)
        dets = ocr_engine.read_text(adb.get_screenshot_image())
        if not any(any(kw in d["text"] for kw in keywords) for d in dets):
            return True
    return False


def _find_and_tap(adb, target: str, w: int, h: int,
                  y_min: int = 0, y_max: int = 99999) -> bool:
    """OCR-scan the screen for `target` text and tap its center.
    Returns True if found and tapped, False otherwise."""
    dets = ocr_engine.read_text(adb.get_screenshot_image())
    for d in dets:
        cy = d["center"][1]
        if cy < y_min or cy > y_max:
            continue
        if target in d["text"]:
            cx, cy = d["center"]
            logger.info("credit_shop: found '%s' at (%d,%d), tapping", target, cx, cy)
            adb.shell("input", "tap", str(cx), str(cy))
            return True
    # Diagnostic: log what was found in the target area
    nearby = [d["text"] for d in dets if y_min <= d["center"][1] <= y_max]
    logger.info("credit_shop: '%s' not found in y [%d,%d]. Nearby texts: %s",
               target, y_min, y_max, ", ".join(nearby[:15]))
    return False


def _item_priority(name: str) -> int:
    name_lower = name.lower()
    for kw, pri in _PRIORITY_MAP:
        if kw in name_lower or kw in name:
            return pri
    return 99


def _scan_items(adb, w: int, h: int) -> list[dict]:
    """Scan the credit shop grid for available items.

    The shop is a 5-column × 2-row grid.  We scan the full item area
    with generous bounds, collect ALL OCR detections for diagnostic
    logging, then partition into columns to extract name+price pairs.
    """
    col_w = w // _GRID_COLS
    y_top = int(h * 0.22)       # Start higher — header area
    y_mid = int(h * 0.58)       # Split between upper/lower rows
    y_bot = int(h * 0.88)       # End lower — below the grid

    dets = ocr_engine.read_text(adb.get_screenshot_image())

    # Filter to item area
    area_dets = [d for d in dets if y_top <= d["center"][1] <= y_bot]

    # ── Diagnostic: log ALL OCR detections in the item area ──
    logger.info("credit_shop: %d total OCR detections, %d in item area (y %d-%d)",
               len(dets), len(area_dets), y_top, y_bot)
    for d in sorted(area_dets, key=lambda d: (d["center"][1], d["center"][0])):
        logger.info("credit_shop:   OCR [%s] at (%d,%d) bbox=(%d,%d,%d,%d)",
                   d["text"], d["center"][0], d["center"][1],
                   d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3])

    # ── Check for "信用交易所" / "信用" header to verify we're on the right screen ──
    shop_header_found = any("信用" in d["text"] for d in dets
                           if d["center"][1] < int(h * 0.15))
    if not shop_header_found:
        logger.warning("credit_shop: credit shop header NOT found — may not be on shop screen")

    items: list[dict] = []

    for col in range(_GRID_COLS):
        x_min = col * col_w
        x_max = (col + 1) * col_w

        col_dets = [d for d in area_dets if x_min <= d["center"][0] <= x_max]

        # Split into upper (row 1) and lower (row 2)
        upper = [d for d in col_dets if d["center"][1] < y_mid]
        lower = [d for d in col_dets if d["center"][1] >= y_mid]

        for band_name, band_dets in [("upper", upper), ("lower", lower)]:
            if not band_dets:
                continue

            band_texts = " ".join(d["text"] for d in band_dets)

            # Check for out-of-stock markers
            if any(m in band_texts for m in _OUT_OF_STOCK_MARKERS):
                continue

            # Find price: look for numbers in plausible range (10-600).
            # OCR frequently glues the price to adjacent numbers:
            #   160 → "16040", 200 → "20050" or "200100"
            # Strategy: try clean numbers first, then extract leading digits.
            price = 0
            for d in band_dets:
                t = d["text"].strip()
                # 1) Clean number: "200", "160"
                m = re.match(r'^(\d{2,3})$', t)
                if m and 10 <= int(m.group(1)) <= 600:
                    price = int(m.group(1))
                    break
                # 2) Discount: "80/160"
                m = re.search(r'(\d{2,3})/(\d{2,3})', t)
                if m:
                    p = int(m.group(1))
                    if 10 <= p <= 600:
                        price = p
                        break
                # 3) OCR-glued: "16040"→160, "20050"→200, "200100"→200
                #    Take the first 3-digit chunk that's a plausible price
                m = re.match(r'^(\d{3})(?:\d{2,3})$', t)
                if m and 10 <= int(m.group(1)) <= 600:
                    price = int(m.group(1))
                    break

            if not price:
                # Log unrecognized texts for debugging
                logger.info("credit_shop: col %d %s — no price found, texts: %s",
                           col, band_name, band_texts)
                continue

            # Find name: prefer Chinese text. English garbage like
            # "SUORES"/"PreclicalArts" must NOT beat real Chinese names.
            name = ""
            name_chinese = ""
            for d in band_dets:
                t = d["text"].strip()
                if len(t) < 1:
                    continue
                # Skip discount badges and pure numbers
                if re.match(r'^[-−]?\d{1,3}%$', t):
                    continue
                if re.match(r'^\d{1,4}$', t) or re.match(r'^\d{2,3}/\d{2,3}$', t):
                    continue
                if re.match(r'^\d{3}\d{2,3}$', t):
                    continue  # OCR-glued price like "16040"
                # Skip out-of-stock markers
                if any(m in t for m in _OUT_OF_STOCK_MARKERS):
                    continue
                has_cjk = any('一' <= ch <= '鿿' for ch in t)
                if has_cjk:
                    if len(t) > len(name_chinese):
                        name_chinese = t
                else:
                    # English/other — only use if no Chinese found at all
                    if not name_chinese and len(t) > len(name) and len(t) >= 3:
                        name = t

            name = name_chinese or name

            if name and price:
                # Click target: center of column, average Y of the band
                band_ys = [d["center"][1] for d in band_dets]
                click_y = sum(band_ys) // len(band_ys)
                click_x = (x_min + x_max) // 2

                items.append({
                    "name": name,
                    "price": price,
                    "priority": _item_priority(name),
                    "col": col,
                    "click": (click_x, click_y),
                })

    # ── Diagnostic summary ──
    if not items:
        logger.warning("credit_shop: NO items extracted from %d area detections. "
                       "Full OCR text in area: %s",
                       len(area_dets),
                       ", ".join(f"'{d['text']}'" for d in sorted(
                           area_dets, key=lambda d: d["center"][1])))

    return items


# ── Main tool ───────────────────────────────────────────────────────────

def credit_shop() -> ToolOutput:
    adb = get_adb()
    w, h = adb.get_screen_size()

    # ── Step 1: Navigate to 采购中心 ─────────────────────────────────
    img = adb.get_screenshot_image()
    pre_dhash = compute_dhash(img)

    if not _find_and_tap(adb, "采购中心", w, h):
        sx, sy = int(w * _MAIN_STORE_PCT[0]), int(h * _MAIN_STORE_PCT[1])
        logger.info("credit_shop: OCR miss for 采购中心, using fixed (%d,%d)", sx, sy)
        adb.shell("input", "tap", str(sx), str(sy))

    if not _wait_for_screen_change(adb, pre_dhash, timeout=5.0):
        return _error("点击采购中心后画面无变化，可能已在采购中心或不在主界面")

    time.sleep(0.5)

    # ── Step 2: Navigate to 信用交易所 ───────────────────────────────
    # "信用交易所" may appear anywhere on the procurement screen — don't
    # restrict to top 25%.  Fall back to a second attempt with back+retry.
    if not _find_and_tap(adb, "信用交易所", w, h):
        # Maybe we're not on the procurement screen — try going back and
        # tapping 采购中心 again with fixed position
        logger.warning("credit_shop: 信用交易所 not found, retrying navigation")
        adb.press_back()
        time.sleep(0.6)
        sx, sy = int(w * _MAIN_STORE_PCT[0]), int(h * _MAIN_STORE_PCT[1])
        adb.shell("input", "tap", str(sx), str(sy))
        time.sleep(1.5)
        if not _find_and_tap(adb, "信用交易所", w, h):
            return _error("未找到信用交易所入口，请在采购中心界面手动操作后重试")

    time.sleep(0.8)

    # ── Step 3: Collect daily credits ────────────────────────────────
    _find_and_tap(adb, "收取信用", w, h, y_max=int(h * 0.15))
    time.sleep(1.0)

    # ── Step 3.5: Dismiss any overlay popups (friend annihilation score, etc.) ─
    # These popups appear on top of the shop grid and block item scanning.
    for _attempt in range(3):
        dets = ocr_engine.read_text(adb.get_screenshot_image())
        popup_hit = any(
            any(kw in d["text"] for kw in _SHOP_POPUP_KW)
            for d in dets if d["center"][1] > int(h * 0.25)
        )
        if not popup_hit:
            break
        logger.info("credit_shop: shop popup detected, dismissing (attempt %d)", _attempt + 1)
        # Try tapping center-right (common close button area) then back
        adb.shell("input", "tap", str(int(w * 0.85)), str(int(h * 0.15)))
        time.sleep(0.5)
        adb.press_back()
        time.sleep(0.5)
    else:
        logger.warning("credit_shop: shop popup may still be present after %d attempts", 3)

    # ── Step 4: Scan items ───────────────────────────────────────────
    items = _scan_items(adb, w, h)

    logger.info("credit_shop: scanned %d available items: %s",
               len(items),
               ", ".join(f"{i['name']}({i['price']})" for i in items))

    if not items:
        _notify_screenshot("信用商店：无可购买商品")
        adb.press_back()
        time.sleep(0.5)
        adb.press_back()
        time.sleep(0.3)
        return _ok({"bought": 0, "total": 0, "items": [], "message": "无可购买商品"})

    # ── Step 5: Build purchase plan ──────────────────────────────────
    items.sort(key=lambda i: (i["priority"], i["price"]))

    # Read budget from the credit counter at the top of the screen.
    # Format: "信用 123/300" or just a bare number like "200".
    budget = 300  # default
    shop_dets = ocr_engine.read_text(adb.get_screenshot_image())
    # Look for "信用" or "CREDIT" near the top of the screen
    for d in shop_dets:
        if d["center"][1] > int(h * 0.20):
            continue  # Budget is always top-left area
        t = d["text"].strip()
        # Pattern: "123/300" or "信用 123"
        m = re.search(r'(\d{2,4})\s*/\s*\d{2,4}', t)
        if m:
            budget = int(m.group(1))
            logger.info("credit_shop: budget=%d from text '%s'", budget, t)
            break
    if budget == 300:
        # Try bare number near the top that looks like a credit count
        for d in shop_dets:
            if d["center"][1] > int(h * 0.12):
                continue
            t = d["text"].strip()
            m = re.match(r'^(\d{2,4})$', t)
            if m and 0 < int(m.group(1)) <= 300:
                budget = int(m.group(1))
                logger.info("credit_shop: budget=%d from bare number '%s'", budget, t)
                break
    # Log all top-20% texts for budget debugging
    top_texts = [d["text"] for d in shop_dets if d["center"][1] < int(h * 0.20)]
    logger.info("credit_shop: top texts for budget: %s", ", ".join(top_texts))

    plan: list[dict] = []
    remaining = budget
    for item in items:
        if item["price"] <= remaining:
            plan.append(item)
            remaining -= item["price"]

    logger.info("credit_shop: budget=%d, plan=%d items: %s",
               budget, len(plan),
               ", ".join(f"{i['name']}({i['price']})" for i in plan))

    if not plan:
        item_list = ", ".join(f"{i['name']}({i['price']})" for i in items)
        _notify_screenshot(f"信用商店：余额不足（信用{budget}），有{len(items)}件商品：{item_list}")
        adb.press_back()
        time.sleep(0.5)
        adb.press_back()
        time.sleep(0.3)
        return _ok({
            "bought": 0, "total": len(items), "items": items,
            "message": f"信用不足，余额{budget}",
        })

    # ── Step 6: Execute purchases ────────────────────────────────────
    bought: list[dict] = []
    for item in plan:
        name = item["name"]
        price = item["price"]
        cx, cy = item["click"]
        logger.info("credit_shop: buying '%s' (%d credits) at (%d,%d)", name, price, cx, cy)

        adb.shell("input", "tap", str(cx), str(cy))

        # Wait for confirmation dialog
        if _wait_for_text(adb, _CONFIRM_KW, timeout=3.0):
            # Tap the confirm button — try OCR first, then fixed position
            tapped = False
            for kw in _CONFIRM_KW:
                if _find_and_tap(adb, kw, w, h):
                    tapped = True
                    break
            if not tapped:
                # Fallback: tap bottom-right of dialog (~common confirm button area)
                logger.warning("credit_shop: confirm button not found via OCR, using fixed position")
                adb.shell("input", "tap", str(int(w * 0.80)), str(int(h * 0.82)))
            time.sleep(0.3)

            # Wait for reward popup and dismiss
            _wait_for_text(adb, _POPUP_KW, timeout=3.0)
            time.sleep(0.3)
            adb.press_back()
            time.sleep(0.3)
        else:
            # No confirm dialog — maybe purchase failed, or dialog is different
            logger.warning("credit_shop: no confirm dialog for '%s', checking popup", name)
            if _wait_for_text(adb, _POPUP_KW, timeout=1.0):
                _wait_for_text_gone(adb, _POPUP_KW, timeout=3.0)
            adb.press_back()
            time.sleep(0.3)

        bought.append(item)
        logger.info("credit_shop: bought '%s' ✓", name)
        time.sleep(0.3)

    # ── Step 7: Screenshot + notify ──────────────────────────────────
    summary = ", ".join(f"{i['name']}({i['price']})" for i in bought)
    total_cost = sum(i["price"] for i in bought)
    _notify_screenshot(
        f"信用商店购买完成：{len(bought)}件（{summary}），花费{total_cost}，剩余{budget - total_cost}信用"
    )

    # ── Step 8: Return to main screen ────────────────────────────────
    adb.press_back()
    time.sleep(0.5)
    adb.press_back()
    time.sleep(0.3)

    logger.info("credit_shop: done — %d/%d items bought", len(bought), len(plan))

    return _ok({
        "bought": len(bought), "planned": len(plan),
        "total_items": len(items), "budget": budget,
        "spent": total_cost, "remaining": budget - total_cost,
        "items": bought,
    })


# ── Register ────────────────────────────────────────────────────────────

def _adb_check() -> bool:
    try:
        from src.device.emulator import emulator_manager
        return emulator_manager.first_online is not None
    except Exception:
        return False


registry.register(
    name="credit_shop",
    game="arknights",
    check_fn=_adb_check,
    description=(
        "【信用商店 — 确定性脚本】自动导航到信用交易所 → 收信用 → 扫描商品 → "
        "按优先级自动购买（招聘许可>加急许可>作战记录>技巧概要>材料） → 截图通知 → 返回主界面。\n"
        "前置条件：必须在明日方舟主界面。\n"
        "特点：零LLM参与，全自动处理选购决策。\n"
        "🔴 工具已内置截图通知！调用后直接 subtask_done。"
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
    handler=credit_shop,
)
