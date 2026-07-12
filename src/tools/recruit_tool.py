"""Recruit tool — deterministic public recruitment for Arknights.

Two-phase fully automated script based on proven LLM workflow from logs:

Phase 1 — Collect ready candidates:
  For each "聘用候选人" button (OCR-sorted by Y):
    tap → SKIP animation → dismiss result popup → wait for slot list

Phase 2 — Start new recruitments:
  For each "开始招募干员" button (OCR-sorted by Y):
    tap → OCR tags → call optimizer → decide (skip/refresh/run) →
    tap tags in grid → set time → ✓ confirm → handle post-confirm → wait for slot list

Zero LLM involvement — the agent just calls recruit(), then subtask_done.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

from src.device.adb import get_adb
from src.tools.registry import ToolOutput, registry
from src.utils.dhash import compute_dhash, hamming_distance
from src.vision.ocr import ocr_engine

logger = logging.getLogger(__name__)

# ── Known tag whitelist (loaded from recruit_tags.json) ────────────────
# Like MAA: only these strings are valid recruitment tags. Every OCR text
# in the tag zone is normalized then matched against this whitelist.

_KNOWN_TAGS: set[str] = set()
_KNOWN_TAGS_LOADED = False


def _load_known_tags() -> None:
    global _KNOWN_TAGS, _KNOWN_TAGS_LOADED
    if _KNOWN_TAGS_LOADED:
        return
    tag_path = Path(__file__).parent.parent / "knowledge" / "arknights" / "recruit_tags.json"
    try:
        raw = json.loads(tag_path.read_text("utf-8"))
        _KNOWN_TAGS = set(raw.get("tag_index", {}).keys())
        logger.info("recruit: loaded %d known tags from whitelist", len(_KNOWN_TAGS))
        _KNOWN_TAGS_LOADED = True
    except Exception:
        logger.warning("recruit: failed to load tag whitelist, fallback to empty", exc_info=True)
        _KNOWN_TAGS_LOADED = True


# ── Verified coordinates (1920×1080 landscape, as percentages) ─────────

# Hour ▼ — tap once to go from 01 → 09 (wraps around)
# Verified: (668, 445) on 1920×1080 from run_20260701_135026.log
_HOUR_DOWN = (0.35, 0.41)

# Confirm ✓ — verified: (1440, 860) on 1920×1080
_CONFIRM_PCT = (0.75, 0.80)

# Refresh confirmation dialog ✓
_REFRESH_CONFIRM = (0.66, 0.70)

# ── Recognizable keywords for screen state detection ────────────────────
_SLOT_LIST_KW = ("开始招募", "聘用候选人", "联络次数", "停止招募")
_TAG_SCREEN_KW = ("招募时限", "职业需求", "可获得的干员", "招募说明")
_CONFIRM_DIALOG_KW = ("加急许可", "立即完成")
_REFRESH_DIALOG_KW = ("消耗", "联络机会")
_RESULT_KW = ("获得", "资质凭证", "信物", "资质信物")
_SERVER_BUSY_KW = ("正在提交反馈至神经",)
_MAIN_SCREEN_KW = ("采购中心", "基建", "Terminal", "档案", "任务", "编队")
_EXIT_DIALOG_KW = ("退出游戏", "确认退出")

# Timing
_DHASH_THRESHOLD = 5
_POLL = 0.3


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
        ctx._notify(message, notify_type="screenshot", image_b64=capture_screen_jpeg())
        logger.info("recruit: notify — %s", message[:120])
    except Exception:
        logger.warning("recruit: notify failed", exc_info=True)


def _screen_changed(adb, pre_hash, timeout: float = 5.0) -> bool:
    """Wait for screen to change from pre_hash within timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(_POLL)
        if hamming_distance(pre_hash, compute_dhash(adb.get_screenshot_image())) >= _DHASH_THRESHOLD:
            return True
    return False


def _wait_for_keywords(adb, keywords: tuple[str, ...], timeout: float = 5.0,
                       require_all: bool = False) -> str:
    """Poll screen until keywords appear. Returns joined OCR text on match, '' on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(_POLL)
        texts = " ".join(d["text"] for d in ocr_engine.read_text(adb.get_screenshot_image()))
        if require_all:
            if all(kw in texts for kw in keywords):
                return texts
        else:
            if any(kw in texts for kw in keywords):
                return texts
    return ""


def _wait_for_keywords_gone(adb, keywords: tuple[str, ...], timeout: float = 8.0) -> bool:
    """Poll screen until keywords disappear."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(_POLL)
        texts = " ".join(d["text"] for d in ocr_engine.read_text(adb.get_screenshot_image()))
        if not any(kw in texts for kw in keywords):
            return True
    return False


def _find_tap(adb, target: str, w: int, h: int,
              y_min: int = 0, y_max: int = 99999) -> bool:
    """OCR-find target text and tap its center. Returns True if tapped."""
    for d in ocr_engine.read_text(adb.get_screenshot_image()):
        if y_min <= d["center"][1] <= y_max and target in d["text"]:
            adb.shell("input", "tap", str(d["center"][0]), str(d["center"][1]))
            logger.info("recruit: tap '%s' at (%d,%d)", target, d["center"][0], d["center"][1])
            return True
    return False


def _find_all_ocr(adb, target: str) -> list[dict]:
    """Return all OCR detections containing target text, sorted by Y position."""
    dets = ocr_engine.read_text(adb.get_screenshot_image())
    matches = [d for d in dets if target in d["text"]]
    matches.sort(key=lambda d: d["center"][1])
    return matches


# ── Contacts ────────────────────────────────────────────────────────────

def _read_contacts(adb, w: int, h: int) -> int:
    """Read 联络次数 remaining from the recruitment list screen."""
    dets = ocr_engine.read_text(adb.get_screenshot_image())
    top = sorted(
        [d for d in dets if d["center"][1] < int(h * 0.15)],
        key=lambda d: d["center"][0],
    )
    # Try single text with "联络次数X/Y"
    for d in top:
        m = re.search(r'联络次数(\d+)\s*/\s*(\d+)', d["text"])
        if m:
            return int(m.group(1))
    # Try joining adjacent texts (OCR split handling)
    joined = "".join(d["text"] for d in top)
    m = re.search(r'联络次数(\d+)\s*/\s*(\d+)', joined)
    if m:
        return int(m.group(1))
    # Fallback: number near "联络"
    for i, d in enumerate(top):
        if "联络" in d["text"] or "次数" in d["text"]:
            for j in range(i, min(i + 3, len(top))):
                n = re.search(r'(\d+)', top[j]["text"])
                if n:
                    val = int(n.group(1))
                    if val <= 3:
                        logger.info("recruit: contacts=%d (fallback from '%s')", val, top[j]["text"])
                        return val
    logger.info("recruit: contacts not readable, assuming 0")
    return 0


# ── Tag OCR ─────────────────────────────────────────────────────────────

def _ocr_real_tags(adb, w: int, h: int) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """OCR the tag grid area, return canonical tag names + their screen positions.

    Like MAA: each OCR-detected text in the tag zone is normalized,
    then fuzzy-matched against the known tag whitelist from recruit_tags.json.

    Returns:
        (tag_names, {tag_name: (center_x, center_y)})
    """
    _load_known_tags()

    tag_top = int(h * 0.20)
    tag_bot = int(h * 0.65)
    dets = ocr_engine.read_text(adb.get_screenshot_image())

    tag_names: list[str] = []
    tag_positions: dict[str, tuple[int, int]] = {}
    zone_raw: list[str] = []

    for d in dets:
        if d["center"][1] < tag_top or d["center"][1] > tag_bot:
            continue
        raw = d["text"].strip()
        zone_raw.append(raw)
        cx, cy = d["center"]

        # Normalize: strip leading bullet, trim whitespace
        t = re.sub(r'^[·••]\s*', '', raw)
        t = t.strip()
        if not t:
            continue

        matched = _match_known_tag(t)
        if matched and matched not in tag_positions:
            tag_names.append(matched)
            tag_positions[matched] = (cx, cy)
        elif not matched:
            logger.debug("recruit: dropped '%s' (no whitelist match)", t)

    logger.info("recruit: zone texts=%s", zone_raw)
    logger.info("recruit: real tags=%s", tag_names)
    return tag_names, tag_positions


def _match_known_tag(text: str) -> str | None:
    """Match a normalized OCR text against the known tag whitelist."""
    _load_known_tags()
    if not _KNOWN_TAGS:
        return text  # fallback: return as-is
    if text in _KNOWN_TAGS:
        return text
    best_score, best_tag = 0.0, None
    for known in _KNOWN_TAGS:
        score = SequenceMatcher(None, text, known).ratio()
        if score > best_score:
            best_score, best_tag = score, known
    if best_score >= 0.65 and best_tag is not None:
        logger.info("recruit: fuzzy match '%s' → '%s' (%.2f)", text, best_tag, best_score)
        return best_tag
    return None


# ── Tag tapping ─────────────────────────────────────────────────────────

def _tap_tag(adb, tag: str, cx: int, cy: int) -> None:
    """Tap a tag at its pre-captured OCR position."""
    logger.info("recruit: tap tag '%s' at (%d,%d)", tag, cx, cy)
    adb.shell("input", "tap", str(cx), str(cy))
    time.sleep(0.3)


# ── Time setting ────────────────────────────────────────────────────────

def _set_9h(adb, w: int, h: int):
    """Set recruitment time to 9:00 by tapping hour ▼ once (01 wraps to 09)."""
    hx, hy = int(w * _HOUR_DOWN[0]), int(h * _HOUR_DOWN[1])
    logger.info("recruit: set 9h — tap ▼ at (%d,%d)", hx, hy)
    adb.shell("input", "tap", str(hx), str(hy))
    time.sleep(0.3)
    # Verify: OCR should show '09 : 00' now, not '01 : 00'
    after = ocr_engine.read_text(adb.get_screenshot_image())
    for d in after:
        if '09' in d["text"] and '00' in d["text"]:
            logger.info("recruit: set 9h — verified 09:00 in OCR")
            return
    # Time didn't change — try again with slightly different Y
    logger.warning("recruit: set 9h — time not confirmed in OCR, retrying")
    adb.shell("input", "tap", str(hx), str(hy + int(h * 0.02)))
    time.sleep(0.3)


# ── Confirm ─────────────────────────────────────────────────────────────

def _tap_confirm(adb, w: int, h: int) -> bool:
    """Tap the recruitment ✓ confirm button. Returns True if screen changed."""
    cx, cy = int(w * _CONFIRM_PCT[0]), int(h * _CONFIRM_PCT[1])
    logger.info("recruit: confirm ✓ at (%d,%d)", cx, cy)
    before = compute_dhash(adb.get_screenshot_image())
    adb.shell("input", "tap", str(cx), str(cy))
    time.sleep(0.8)
    return _screen_changed(adb, before, timeout=5.0)


# ── Optimizer bridge ────────────────────────────────────────────────────

def _optimize(tags: list[str], strategy: str) -> dict:
    """Call the internal tag optimizer and return parsed result for one slot."""
    from src.tools.recruit_optimizer import optimize_recruit_tags as _opt
    raw = _opt(tags=[tags], strategy=strategy)
    data = json.loads(raw.text)
    if not data.get("slots"):
        return {"success": False}
    slot = data["slots"][0]
    best = slot.get("best_combo", {})
    return {
        "success": True,
        "tags": tags,
        "selected": best.get("selected_tags", []),
        "tier": best.get("guarantee_tier", 0),
        "time": best.get("recommended_time", "1:00"),
        "action": best.get("slot_action", "run"),
        "label": best.get("guarantee_label", ""),
        "ops": best.get("matched_operators", [])[:5],
        "has_top": best.get("has_top_senior", False),
        "has_senior": best.get("has_senior", False),
    }


# ═════════════════════════════════════════════════════════════════════════
# Phase 1: Collect ready candidates
# ═════════════════════════════════════════════════════════════════════════

def _poll_skip(adb, w: int, h: int, timeout: float = 6.0) -> bool:
    """Poll for SKIP button during recruitment animation, tap as soon as found.

    Also handles the case where the result screen appears instantly (fast device
    or no animation) — if result keywords are detected, returns True immediately.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        texts = " ".join(d["text"] for d in ocr_engine.read_text(adb.get_screenshot_image()))
        # Result screen already visible — no need to skip
        if any(kw in texts for kw in _RESULT_KW):
            return True
        # Look for SKIP and tap it
        if _find_tap(adb, "SKIP", w, h):
            logger.info("recruit: SKIP tapped")
            return True
        time.sleep(0.4)
    return False


def _collect_candidates(adb, w: int, h: int) -> int:
    """Collect all ready candidates (聘用候选人). Returns count collected."""
    collected = 0
    max_rounds = 4  # safety: at most 4 rounds (one per slot)

    for _round in range(max_rounds):
        candidates = _find_all_ocr(adb, "聘用候选人")
        if not candidates:
            break

        # Take the first (topmost) candidate
        d = candidates[0]
        cx, cy = d["center"]
        logger.info("recruit: phase1 — collect candidate at (%d,%d)", cx, cy)

        adb.shell("input", "tap", str(cx), str(cy))

        # ── Poll for SKIP during animation (up to 6s) ──
        _poll_skip(adb, w, h, timeout=6.0)

        # After SKIP, the operator reveal screen appears (character art +
        # operator name, "获得" text, 资质凭证 etc.).  Tap the screen
        # center to dismiss it — this is the game's expected interaction.
        # back() is unreliable here because it's not a popup, it's a
        # full-screen reveal.
        time.sleep(0.3)

        # ── Dismiss operator reveal screen ──
        logger.info("recruit: phase1 — tapping to dismiss operator reveal")
        for _tap in range(3):
            adb.shell("input", "tap", str(int(w * 0.50)), str(int(h * 0.70)))
            time.sleep(0.3)
            # Check if we're back on the slot list
            if _wait_for_keywords(adb, _SLOT_LIST_KW, timeout=2.0):
                break
            logger.info("recruit: phase1 — tap %d didn't return to slot list, retrying", _tap + 1)
        else:
            # Taps didn't work — try back as last resort
            logger.warning("recruit: phase1 — tap failed, trying back")
            adb.press_back()
            time.sleep(0.5)
            if not _wait_for_keywords(adb, _SLOT_LIST_KW, timeout=4.0):
                logger.warning("recruit: phase1 — back also failed, trying one more tap")
                adb.shell("input", "tap", str(int(w * 0.50)), str(int(h * 0.70)))
                time.sleep(0.3)
                _wait_for_keywords(adb, _SLOT_LIST_KW, timeout=3.0)

        # ── Only count if we confirmed return to slot list ──
        texts = " ".join(d["text"] for d in ocr_engine.read_text(adb.get_screenshot_image()))
        if any(kw in texts for kw in _SLOT_LIST_KW):
            collected += 1
            logger.info("recruit: phase1 — candidate %d collected and confirmed on slot list", collected)
        else:
            logger.warning("recruit: phase1 — candidate slot may not have been collected; not on slot list")
            # Don't increment collected — the dismiss failed

        time.sleep(0.2)

    logger.info("recruit: phase1 — collected %d candidates total", collected)
    return collected


# ═════════════════════════════════════════════════════════════════════════
# Phase 2: Start new recruitments
# ═════════════════════════════════════════════════════════════════════════

def _start_new_recruits(adb, w: int, h: int, contacts: int,
                        strategy: str) -> list[dict]:
    """Start new recruitments for all empty slots. Returns results list."""
    results: list[dict] = []
    remaining_contacts = contacts
    max_rounds = 4

    # ── Validate we're on the recruitment list ──
    texts = " ".join(d["text"] for d in ocr_engine.read_text(adb.get_screenshot_image()))
    if not any(kw in texts for kw in _SLOT_LIST_KW):
        logger.warning("recruit: phase2 — not on recruitment list at start, pressing back to recover")
        for _ in range(4):
            adb.press_back()
            time.sleep(0.5)
            texts = " ".join(d["text"] for d in ocr_engine.read_text(adb.get_screenshot_image()))
            if any(kw in texts for kw in _SLOT_LIST_KW):
                logger.info("recruit: phase2 — recovered to slot list")
                break
        else:
            logger.warning("recruit: phase2 — could not recover to slot list, attempting to proceed anyway")

    for _round in range(max_rounds):
        empty_slots = _find_all_ocr(adb, "开始招募干员")
        if not empty_slots:
            # Log what IS visible so we can debug the "missing 4th slot" issue
            all_texts = [d["text"] for d in ocr_engine.read_text(adb.get_screenshot_image())
                        if d["center"][1] > int(h * 0.50)]
            logger.info("recruit: phase2 — no more empty slots. Screen texts (bottom half): %s",
                       sorted(set(all_texts)))
            break

        # Take first (topmost) empty slot
        d = empty_slots[0]
        slot_idx = _round  # 0-indexed within this phase
        result = _process_one_slot(adb, w, h, d["center"], slot_idx,
                                   remaining_contacts, strategy)
        results.append(result)

        # Update remaining contacts
        refreshes = result.get("refreshes", 0)
        if refreshes > 0:
            remaining_contacts -= refreshes

    logger.info("recruit: phase2 — %d slots processed", len(results))
    return results


def _process_one_slot(adb, w: int, h: int, click_pos: tuple[int, int],
                      slot_idx: int, contacts: int,
                      strategy: str) -> dict:
    """Process a single empty recruitment slot. Returns result dict."""
    cx, cy = click_pos
    logger.info("recruit: phase2 — slot %d at (%d,%d)", slot_idx, cx, cy)

    # ── Open the slot ──
    before = compute_dhash(adb.get_screenshot_image())
    adb.shell("input", "tap", str(cx), str(cy))
    if not _screen_changed(adb, before, timeout=6.0):
        logger.warning("recruit: slot %d didn't open", slot_idx)
        return {"slot": slot_idx, "error": "栏位打不开"}

    time.sleep(0.3)

    # ── Wait for tag selection screen ──
    if not _wait_for_keywords(adb, _TAG_SCREEN_KW, timeout=3.0):
        # Might already be on tag screen — try OCR anyway
        logger.warning("recruit: slot %d — tag screen keywords not found, proceeding anyway", slot_idx)

    remaining_contacts = contacts
    refreshes_used = 0

    for attempt in range(4):  # max 4 attempts (original + 3 refreshes)
        # ── Wait for server communication to finish ──
        _wait_for_keywords_gone(adb, _SERVER_BUSY_KW, timeout=5.0)
        time.sleep(0.2)

        tags, tag_positions = _ocr_real_tags(adb, w, h)
        if not tags:
            logger.warning("recruit: slot %d — no tags found", slot_idx)
            adb.press_back()
            time.sleep(0.5)
            return {"slot": slot_idx, "error": "无标签"}

        opt = _optimize(tags, strategy)
        if not opt["success"]:
            adb.press_back()
            time.sleep(0.5)
            return {"slot": slot_idx, "error": "优化器失败"}

        action = opt["action"]
        selected = opt["selected"]
        tier = opt["tier"]
        time_req = opt["time"]

        # ── Top senior operator → skip ──
        if opt["has_top"]:
            logger.warning("recruit: slot %d — 高资! skipping", slot_idx)
            adb.press_back()
            time.sleep(0.5)
            return {"slot": slot_idx, "action": "skip",
                    "reason": "高级资深干员", "tags": tags}

        # ── Refresh if recommended and contacts available ──
        if action == "refresh" and remaining_contacts > 0 and refreshes_used < 3:
            logger.info("recruit: slot %d — refresh %d/%d (contacts left: %d)",
                       slot_idx, refreshes_used + 1, contacts, remaining_contacts)
            if _do_refresh(adb, w, h):
                refreshes_used += 1
                remaining_contacts -= 1
                time.sleep(0.5)
                continue
            else:
                logger.warning("recruit: slot %d — refresh failed, falling through to run", slot_idx)

        # ── Execute: tap tags as optimizer instructed ──
        logger.info("recruit: slot %d — run tags=%s time=%s tier=%d",
                   slot_idx, selected, time_req, tier)
        for tag in selected:
            pos = tag_positions.get(tag)
            if pos is None:
                # Fallback: re-OCR to find this tag (should be rare with whitelist)
                logger.warning("recruit: tag '%s' position not pre-captured, falling back to OCR", tag)
                _load_known_tags()
                dets = ocr_engine.read_text(adb.get_screenshot_image())
                tx = ty = None
                for d in dets:
                    t = re.sub(r'^[·••]\s*', '', d["text"].strip())
                    if t and _match_known_tag(t) == tag:
                        tx, ty = d["center"]
                        break
                if tx is None:
                    logger.warning("recruit: tag '%s' not found on screen", tag)
                    continue
                _tap_tag(adb, tag, tx, ty)
            else:
                _tap_tag(adb, tag, pos[0], pos[1])

        # ── Set time ──
        # Use the optimizer's recommendation directly.  It knows which
        # tags benefit from specific timers (e.g. 支援机械=1:00 for robots,
        # 9:00 for 4★ hard guarantee).
        actual_time = time_req
        if actual_time == "9:00":
            _set_9h(adb, w, h)
            time.sleep(0.3)

        # ── Confirm recruitment ──
        _tap_confirm(adb, w, h)
        time.sleep(0.5)

        # ── Handle post-confirm dialogs ──
        _handle_post_confirm(adb, w, h)

        # ── Wait for slot list screen ──
        _wait_for_keywords(adb, _SLOT_LIST_KW, timeout=6.0)
        # Handle lingering server communication
        _wait_for_keywords_gone(adb, _SERVER_BUSY_KW, timeout=3.0)
        time.sleep(0.3)

        return {"slot": slot_idx, "action": "run", "tags": tags,
                "selected": selected, "tier": tier, "time": actual_time,
                "label": opt["label"], "ops": opt["ops"],
                "refreshes": refreshes_used}

    # Ran out of attempts
    adb.press_back()
    time.sleep(0.5)
    return {"slot": slot_idx, "error": "重试耗尽"}


def _do_refresh(adb, w: int, h: int) -> bool:
    """Execute one tag refresh cycle. Returns True if refresh was completed."""
    # Tap "点击刷新标签" button — search the whole screen (no y_max restriction)
    found = (_find_tap(adb, "点击刷新标签", w, h)
             or _find_tap(adb, "刷新标签", w, h)
             or _find_tap(adb, "刷新", w, h))
    if not found:
        # Fallback: OCR scan for the button text position, then tap
        logger.warning("recruit: refresh button not found via OCR, using fixed position")
        adb.shell("input", "tap", str(int(w * 0.50)), str(int(h * 0.13)))
    time.sleep(0.5)

    # Handle "是否消耗1次联络机会？" confirmation dialog
    if _wait_for_keywords(adb, _REFRESH_DIALOG_KW, timeout=2.0):
        time.sleep(0.2)
        rx, ry = int(w * _REFRESH_CONFIRM[0]), int(h * _REFRESH_CONFIRM[1])
        logger.info("recruit: refresh confirm at (%.2f,%.2f) → (%d,%d)",
                   _REFRESH_CONFIRM[0], _REFRESH_CONFIRM[1], rx, ry)
        before = compute_dhash(adb.get_screenshot_image())
        adb.shell("input", "tap", str(rx), str(ry))
        time.sleep(0.3)
        # Wait for dialog to close and tags to refresh
        _screen_changed(adb, before, timeout=4.0)
        time.sleep(0.5)
        return True

    logger.warning("recruit: refresh — no confirmation dialog appeared")
    return False


def _handle_post_confirm(adb, w: int, h: int) -> None:
    """Handle dialogs that may appear after tapping ✓ confirm.

    Common scenarios:
    1. 加急许可 prompt: "是否消耗一张加急许可立即完成招募？"
    2. Server communication: "正在提交反馈至神经……"
    3. Direct result screen showing 获得/资质凭证/信物
    """
    time.sleep(0.5)

    for _ in range(3):  # at most 3 dialog layers
        texts = " ".join(d["text"] for d in ocr_engine.read_text(adb.get_screenshot_image()))

        # Check for 加急许可 dialog
        if any(kw in texts for kw in _CONFIRM_DIALOG_KW):
            logger.info("recruit: post-confirm — 加急许可 dialog, tapping ✓")
            # The ✓ button on this dialog is in roughly the same position
            adb.shell("input", "tap", str(int(w * 0.75)), str(int(h * 0.70)))
            time.sleep(0.5)
            continue

        # Check for server communication
        if any(kw in texts for kw in _SERVER_BUSY_KW):
            logger.info("recruit: post-confirm — waiting for server communication")
            _wait_for_keywords_gone(adb, _SERVER_BUSY_KW, timeout=5.0)
            time.sleep(0.3)
            continue

        # Check for result screen (获得/资质凭证/信物) — dismiss it
        if any(kw in texts for kw in _RESULT_KW):
            logger.info("recruit: post-confirm — dismissing result popup")
            adb.shell("input", "tap", str(int(w * 0.50)), str(int(h * 0.85)))
            time.sleep(0.4)
            adb.press_back()
            time.sleep(0.3)
            continue

        # Check if we're already back on the slot list screen
        if any(kw in texts for kw in _SLOT_LIST_KW):
            logger.info("recruit: post-confirm — already on slot list")
            break

        # Nothing recognized — try pressing back once
        adb.press_back()
        time.sleep(0.3)
        break

    # Final: ensure server communication is done
    _wait_for_keywords_gone(adb, _SERVER_BUSY_KW, timeout=3.0)


# ═════════════════════════════════════════════════════════════════════════
# Back to main screen
# ═════════════════════════════════════════════════════════════════════════

def _back_to_main(adb, w: int, h: int) -> None:
    """Navigate back to the Arknights main screen."""
    for _ in range(5):
        texts = " ".join(d["text"] for d in ocr_engine.read_text(adb.get_screenshot_image()))

        # Check exit dialog
        if any(kw in texts for kw in _EXIT_DIALOG_KW):
            logger.info("recruit: hit exit dialog, cancelling")
            _find_tap(adb, "取消", w, h)
            time.sleep(0.4)
            break

        # Check main screen
        if sum(1 for kw in _MAIN_SCREEN_KW if kw in texts) >= 2:
            logger.info("recruit: back to main screen confirmed")
            break

        adb.press_back()
        time.sleep(0.5)
    else:
        logger.warning("recruit: could not confirm return to main screen")


# ═════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════

def recruit(strategy: str = "collection") -> ToolOutput:
    """Fully automated public recruitment for Arknights.

    Two-phase execution:
      1. Collect all ready candidates (聘用候选人)
      2. Start new recruitments for empty slots (开始招募干员)

    Per-slot logic: OCR tags → optimizer → skip/refresh/run → tap tags →
    set time → ✓ confirm → handle post-confirm dialogs.

    Args:
        strategy: "collection" (图鉴优先) or "yellow_cert" (黄票优先)

    Returns:
        ToolOutput with summary of processed slots.
    """
    adb = get_adb()
    w, h = adb.get_screen_size()
    logger.info("recruit: start — strategy=%s, screen=%dx%d", strategy, w, h)
    _load_known_tags()  # ensure tag whitelist is loaded before OCR

    # ── Step 1: Navigate to recruitment ──
    if not _find_tap(adb, "公开招募", w, h):
        return _error("未找到公开招募按钮")
    time.sleep(1.0)

    # Wait for recruitment list screen
    if not _wait_for_keywords(adb, ("公开招募", "联络次数"), timeout=5.0):
        logger.warning("recruit: recruitment screen not confirmed, proceeding anyway")
    time.sleep(0.3)

    # ── Step 2: Phase 1 — Collect ready candidates ──
    logger.info("recruit: === Phase 1: Collect candidates ===")
    collected = _collect_candidates(adb, w, h)

    # ── Validate slot list before Phase 2 ──
    # Phase 1 should leave us on the slot list, but if something went wrong
    # (e.g. animation interrupted navigation), press back repeatedly to recover.
    texts = " ".join(d["text"] for d in ocr_engine.read_text(adb.get_screenshot_image()))
    if not any(kw in texts for kw in _SLOT_LIST_KW):
        logger.warning("recruit: not on slot list after Phase 1, recovering with repeated back")
        for _ in range(4):
            adb.press_back()
            time.sleep(0.5)
            texts = " ".join(d["text"] for d in ocr_engine.read_text(adb.get_screenshot_image()))
            if any(kw in texts for kw in _SLOT_LIST_KW):
                logger.info("recruit: recovered to slot list between phases")
                break

    # ── Step 3: Phase 2 — Start new recruitments ──
    logger.info("recruit: === Phase 2: Start new recruitments ===")
    contacts = _read_contacts(adb, w, h)
    logger.info("recruit: %d contacts remaining", contacts)

    results = _start_new_recruits(adb, w, h, contacts, strategy)

    # ── Step 4: Build summary ──
    started = sum(1 for r in results if r.get("action") == "run")
    skipped = sum(1 for r in results if r.get("action") == "skip")
    errs = sum(1 for r in results if "error" in r)

    parts = []
    if collected: parts.append(f"领取{collected}个已完成候选人")
    if started: parts.append(f"{started}栏位已招募")
    if skipped: parts.append(f"{skipped}跳过高资")
    if errs: parts.append(f"{errs}失败")

    summary = "，".join(parts) if parts else "无操作（所有栏位均在招募中）"
    _notify_screenshot(f"公招完成：{summary}")

    # ── Step 5: Back to main ──
    _back_to_main(adb, w, h)

    logger.info("recruit: done — collected=%d, started=%d, skipped=%d, errors=%d",
               collected, started, skipped, errs)
    return _ok({
        "collected": collected,
        "slots": len(results),
        "started": started,
        "skipped": skipped,
        "errors": errs,
        "details": results,
        "summary": summary,
    })


# ── Register ────────────────────────────────────────────────────────────

registry.register(
    name="recruit",
    game="arknights",
    description=(
        "【公开招募 — 全自动脚本】完成所有空闲栏位的标签优选与确认招募。\n"
        "两阶段处理：先领取已完成候选人→再为空白栏位开始新招募。\n"
        "流程：主界面→公开招募→逐栏OCR标签→内部优选器→选标签→调时间→✓确认。\n"
        "策略：collection(图鉴优先)/yellow_cert(黄票优先)。\n"
        "无保底标签自动刷新（消耗联络次数）。遇到高资自动跳过。\n"
        "前置条件：明日方舟主界面。\n"
        "🔴 内置截图通知，调用后直接 subtask_done。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "enum": ["collection", "yellow_cert"],
                "description": "collection=图鉴优先（默认）、yellow_cert=黄票优先",
            },
        },
    },
    handler=recruit,
)
