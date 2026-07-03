"""Base collect tool — deterministic notification bell reward collection for Arknights.

Replaces the LLM-driven base-collect skill with a fast, reliable script that:
1. Verifies we're on the base screen
2. Clicks the bell icon at the known fixed position
3. Waits for the notification panel to appear (dHash polling)
4. Re-OCR + click the first available tab, repeat until no tabs remain
   (tabs slide into first position after collection — same spot each time)
5. Screenshots the panel + notifies user (before closing panel)
6. Closes the panel and returns results

Zero LLM involvement — the agent just calls base_collect() and receives the outcome.
"""

from __future__ import annotations

import json
import logging
import threading
import time

from src.device.adb import get_adb
from src.tools.registry import ToolOutput, registry
from src.utils.dhash import compute_dhash, hamming_distance
from src.vision.ocr import ocr_engine

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

# Tabs to collect, in priority order (top-to-bottom as they appear in the panel)
_COLLECT_TABS: tuple[str, ...] = (
    "可收获",    # Manufacturing station products
    "订单交付",  # Trade station orders
    "干员信赖",  # Operator trust
    "队列轮换",  # Queue rotation (one-click!)
    "干员休整",  # Operators finished resting
)

# Tabs that exist in the panel but are NOT collectable
_SKIP_TABS: frozenset[str] = frozenset({"线索搜集", "待办事项"})

# Base screen verification: at least one of these must be present in OCR
_BASE_SCREEN_KW: tuple[str, ...] = ("控制中枢", "制造站", "贸易站")

# Bell icon position (verified across multiple devices / resolutions)
_BELL_X_PCT = 0.94
_BELL_Y_PCT = 0.17

# Timing
_PANEL_TIMEOUT = 3.0       # Max seconds to wait for panel after bell click
_PANEL_POLL_INTERVAL = 0.2  # Seconds between dHash checks
_TAB_CLICK_DELAY = 1.0      # Seconds to wait after clicking a tab (server anim)
_BACK_DELAY = 0.3           # Seconds to wait after pressing back

# dHash threshold for detecting meaningful screen changes
_DHASH_CHANGE_THRESHOLD = 5


# ── Helpers ─────────────────────────────────────────────────────────────

def _ok(data: dict) -> ToolOutput:
    data.setdefault("success", True)
    return ToolOutput(text=json.dumps(data, ensure_ascii=False))


def _error(msg: str) -> ToolOutput:
    return ToolOutput(text=json.dumps({"success": False, "error": msg}, ensure_ascii=False))


def _is_base_screen(ocr_texts: set[str]) -> bool:
    """Check whether the current screen is the base infrastructure view."""
    return any(
        any(kw in t for t in ocr_texts)
        for kw in _BASE_SCREEN_KW
    )


def _notify_panel_screenshot(message: str) -> None:
    """Send a WeChat notification with the current screen (panel open).

    Best-effort — failures are logged but never raised.  This runs
    BEFORE the panel is closed so the user sees the collection result.
    """
    try:
        from src.agent.screen_injector import capture_screen_jpeg
        ctx = getattr(threading.current_thread(), '_terra_agent_ctx', None)
        if ctx is None:
            logger.warning("base_collect: no agent context, skipping notify")
            return
        image_b64 = capture_screen_jpeg()
        ctx._notify(message, notify_type="screenshot", image_b64=image_b64)
        logger.info("base_collect: notify sent — %s", message[:120])
    except Exception:
        logger.warning("base_collect: notify failed (non-critical)", exc_info=True)


# ── Main tool ───────────────────────────────────────────────────────────

def base_collect() -> ToolOutput:
    """Collect all base rewards from the notification bell panel.

    Deterministic — no LLM involvement.  The function handles the entire
    flow internally and returns a structured result.

    Returns:
        ToolOutput with JSON containing:
          - collected: number of tabs that were collected
          - total_tabs: total tabs found
          - tabs: list of {tab, collected} per tab
    """
    adb = get_adb()
    w, h = adb.get_screen_size()

    # Panel covers roughly the bottom 45% of the screen.
    # We OCR from this Y coordinate downwards for both tab discovery
    # and content-change verification.
    panel_y_min = int(h * 0.55)

    # ── Step 1: Verify we are on the base screen ────────────────────
    img = adb.get_screenshot_image()
    detections = ocr_engine.read_text(img)
    all_texts = {d["text"] for d in detections}

    if not _is_base_screen(all_texts):
        sample = ", ".join(sorted(all_texts)[:12]) if all_texts else "(empty)"
        return _error(
            f"不在基建界面。当前画面文字：{sample}。"
            f"请先用 adb_tap('基建') 进入基建界面，再调用 base_collect()。"
        )

    logger.info("base_collect: confirmed on base screen")

    # ── Step 2: Click the bell icon (precise — no noise) ────────────
    # The bell icon is tiny and adjacent to the emergency warning icon.
    # adb.tap() adds ±5px random noise which can shift the click onto
    # the wrong icon.  Use a direct input tap with zero noise.
    bell_x = int(w * _BELL_X_PCT)
    bell_y = int(h * _BELL_Y_PCT)

    logger.info("base_collect: clicking bell at (%d, %d)", bell_x, bell_y)
    pre_dhash = compute_dhash(img)
    adb.shell("input", "tap", str(bell_x), str(bell_y))

    # ── Step 3: Wait for the notification panel to appear ───────────
    deadline = time.monotonic() + _PANEL_TIMEOUT
    panel_opened = False
    while time.monotonic() < deadline:
        time.sleep(_PANEL_POLL_INTERVAL)
        curr = adb.get_screenshot_image()
        curr_dhash = compute_dhash(curr)
        dist = hamming_distance(pre_dhash, curr_dhash)
        if dist >= _DHASH_CHANGE_THRESHOLD:
            logger.info("base_collect: panel opened (dHash dist=%d)", dist)
            panel_opened = True
            break

    if not panel_opened:
        return _error(
            f"铃铛点击后 {_PANEL_TIMEOUT:.0f}s 内面板未出现。"
            f"可能原因：坐标偏移 / 已在面板中 / 不在基建界面。"
        )

    # Let panel finish its slide-up animation before OCR
    time.sleep(0.3)

    # ── Step 4: Initial diagnostic — log all panel texts ────────────
    _initial_dets = ocr_engine.read_text(adb.get_screenshot_image())
    _panel_texts: list[str] = []
    for d in _initial_dets:
        cy = d["center"][1]
        if cy < panel_y_min:
            continue
        _panel_texts.append(f"'{d['text']}'(y={cy})")
    logger.info("base_collect: all panel texts (y>=%d, %d items): %s",
               panel_y_min, len(_panel_texts),
               ", ".join(_panel_texts) if _panel_texts else "(none)")

    # ── Step 5: Click each tab — no verification, just fire and move on
    # Re-OCR every iteration for fresh coordinates (tabs shift after
    # collection).  User verifies via the final screenshot.
    clicked: list[str] = []
    max_clicks = len(_COLLECT_TABS)

    for _ in range(max_clicks):
        dets = ocr_engine.read_text(adb.get_screenshot_image())

        first_tab: tuple[str, dict] | None = None
        for tab_name in _COLLECT_TABS:
            if tab_name in clicked:
                continue
            for d in dets:
                cy = d["center"][1]
                if cy < panel_y_min:
                    continue
                if len(d["text"]) > 20:
                    continue
                if tab_name in d["text"] or d["text"] in tab_name:
                    first_tab = (tab_name, d)
                    break
            if first_tab:
                break

        if first_tab is None:
            logger.info("base_collect: no more tabs — done")
            break

        tab_name, det = first_tab
        cx, cy = det["center"]
        logger.info("base_collect: clicking tab '%s' at (%d, %d)", tab_name, cx, cy)
        adb.tap(cx, cy)

        # Wait for the tab text to actually disappear before moving on.
        # Fixed sleep isn't reliable — tab animations vary in duration.
        # Poll OCR until this tab's text is gone (max 3s).
        wait_deadline = time.monotonic() + 3.0
        while time.monotonic() < wait_deadline:
            time.sleep(0.3)
            dets_after = ocr_engine.read_text(adb.get_screenshot_image())
            gone = not any(
                (tab_name in d["text"] or d["text"] in tab_name)
                for d in dets_after
                if d["center"][1] >= panel_y_min and len(d["text"]) <= 20
            )
            if gone:
                break

        clicked.append(tab_name)

    if not clicked:
        _notify_panel_screenshot("基建收菜：面板中无可收取项")
        adb.press_back()
        time.sleep(_BACK_DELAY)
        return _ok({
            "collected": 0,
            "total_tabs": 0,
            "tabs": [],
            "message": "面板中无可收取项（可能已全部收取完毕）",
        })

    # ── Step 6: Notify user BEFORE closing the panel ─────────────────
    tab_names_cn = ", ".join(clicked)
    _notify_panel_screenshot(f"基建收菜完成：收取了 {len(clicked)} 项（{tab_names_cn}）")

    # ── Step 7: Close the panel ─────────────────────────────────────
    adb.press_back()
    time.sleep(_BACK_DELAY)

    logger.info("base_collect: done — %d tabs: %s", len(clicked), tab_names_cn)

    return _ok({
        "collected": len(clicked),
        "total_tabs": len(clicked),
        "tabs": clicked,
    })


# ── Register ────────────────────────────────────────────────────────────

def _adb_check() -> bool:
    """Check that ADB is available before offering the tool to the LLM."""
    try:
        from src.device.emulator import emulator_manager
        return emulator_manager.first_online is not None
    except Exception:
        return False


registry.register(
    name="base_collect",
    game="arknights",
    check_fn=_adb_check,
    description=(
        "【基建收菜 — 确定性脚本】一键收取基建铃铛面板中的所有可收取项，包括队列轮换。\n"
        "前置条件：必须在基建界面（有 控制中枢/制造站/贸易站 等文字）。\n"
        "工具内部自动处理：点击铃铛 → 等待面板出现 → OCR识别可用tab → 逐个收取 → 截图通知 → 关闭面板。\n"
        "收取项：可收获（制造站）、订单交付（贸易站）、干员信赖、队列轮换、干员休整。\n"
        "自动跳过：线索搜集（非可收取项）、不存在的tab（无可收内容时不显示）。\n"
        "特点：零LLM参与，确定性执行，比手动 adb_tap 快 10 倍以上。\n"
        "🔴 工具已内置截图通知！调用后直接 subtask_done，不要重复调 notify_with_screen！"
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
    handler=base_collect,
)
