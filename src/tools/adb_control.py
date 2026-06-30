"""ADB control tools: tap (with OCR/template localization), swipe, back.

Each tool internally does: capture → locate → execute → verify → return result.
This encapsulates game UI complexity behind simple tool interfaces.

Self-registers with the tool registry at import time.
"""

from __future__ import annotations

import json
import logging
import random
import time
from difflib import SequenceMatcher

from PIL import Image

from config.settings import config as app_config
from src.device.adb import get_adb
from src.tools import screen_cache
from src.tools.registry import registry, ToolOutput
from src.utils.dhash import compute_dhash, dhash_to_hex, hamming_distance
from src.utils.hash import compute_image_hash
from src.vision.ocr import ocr_engine

logger = logging.getLogger(__name__)

# Icons with no readable text — skip OCR, tap via VLM coordinate directly
# Icons with no readable text — skip OCR, tap via VLM coordinate directly.
# NOTE: single-letter targets like "X"/"x" are EXCLUDED — they appear as
# readable text in game UI (e.g. "X10 quantity", "Episode X") and would
# cause false positives if they bypassed OCR verification.
_ICON_TARGETS = {"返回", "返回箭头", "关闭", "×", "主页", "◀", "▶", "＜", "＞",
                 "铃铛", "通知铃铛", "NOTIFICATION", "notification", "通知", "bell", "铃"}

# ── OCR detection cache (populated by screen_injector, consumed by adb_tap) ──
# screen_injector already runs OCR on every injected screenshot.  adb_tap
# can reuse those results instead of calling ocr_engine.read_text() again
# (~300ms saved per tap on a cache hit).  Keyed by screen dHash hex.
_ocr_cache: dict[str, list[dict]] = {}
_OCR_CACHE_MAX = 32


def cache_ocr_detections(screen_hash: str, detections: list[dict]) -> None:
    """Store full OCR detections for reuse by adb_tap. Called by screen_injector."""
    if not screen_hash or not detections:
        return
    _ocr_cache[screen_hash] = detections
    # LRU eviction: keep the most recent N entries
    while len(_ocr_cache) > _OCR_CACHE_MAX:
        oldest = next(iter(_ocr_cache))
        del _ocr_cache[oldest]

# Top-anchored targets — always at the top of scrollable lists. When the target
# is not found on screen, fail with scroll-back hint instead of fuzzy-matching a
# nearby button or tapping a stale cached coordinate.  This prevents the agent
# from clicking a bare "领取" button and thinking it clicked "全部领取".
_PINNED_TARGETS = {
    "全部领取",
    "一键领取",
    "一键派遣",
    "一键赠送",
    "全部收取",
    "信用交易所购物",
    "主线",
    "终端",
}

# Checkbox/state-toggle targets — tap then OCR-verify the region for X or digit
_CHECKBOX_TARGETS = {"代理指挥", "代理指挥^"}

# Fast-mode skill coordinates: populated by skill_run, consumed by adb_tap.
# Keyed by (device_serial, skill_name) to prevent cross-device contamination.
_skill_coords: dict[tuple[str, str], tuple[int, int]] = {}


def clear_skill_coords() -> None:
    """Clear fast-mode skill coordinate cache. Call in test setup/teardown."""
    _skill_coords.clear()


def _screen_hash(img: Image.Image) -> str:
    """Perceptual hash (dHash) for screen cache lookup.

    dHash tolerates minor pixel differences between ADB captures of the same
    screen, giving cache hits where MD5-based exact hashing would always miss.
    """
    from src.utils.dhash import compute_dhash, dhash_to_hex
    return dhash_to_hex(compute_dhash(img))


def _tap_with_screen_check(adb, x: int, y: int) -> dict[str, bool | int]:
    """Tap at pixel coordinates and report whether the screen actually changed.

    Captures dHash before and after the tap, compares with Hamming distance.
    Returns a dict with 'screen_changed' (bool) and 'dhash_distance' (int)
    suitable for merging into the tool output JSON.
    """
    try:
        pre_img = adb.get_screenshot_image()
        pre_dhash = compute_dhash(pre_img)
    except Exception:
        # If pre-tap screenshot fails, tap anyway and report unknown
        adb.tap(x, y)
        time.sleep(0.5)
        return {"screen_changed": None, "dhash_distance": -1}

    adb.tap(x, y)
    time.sleep(0.5)

    try:
        post_img = adb.get_screenshot_image()
        post_dhash = compute_dhash(post_img)
        dist = hamming_distance(pre_dhash, post_dhash)
        return {
            "screen_changed": dist >= 5,
            "dhash_distance": dist,
        }
    except Exception:
        return {"screen_changed": None, "dhash_distance": -1}


# ── Icon fallback: hardcoded coordinates for non-text targets ──
# Icons like the bell (铃铛) have no OCR-readable text. When OCR cache
# misses (which is expected), use verified coordinates instead of wasting
# LLM iterations guessing.
_ICON_FALLBACK_COORDS: dict[str, tuple[float, float]] = {
    "铃铛": (0.94, 0.13),
    "通知铃铛": (0.94, 0.13),
    "NOTIFICATION": (0.94, 0.13),
    "notification": (0.94, 0.13),
    "bell": (0.94, 0.13),
    "通知": (0.94, 0.13),
    "铃": (0.94, 0.13),
}


def _try_icon_fallback_tap(target: str, adb) -> ToolOutput | None:
    """Tap an icon target using a verified hardcoded coordinate.

    Called when OCR/cache can't find the target (expected for pixel icons).
    Returns a ToolOutput, or None if the target is not a supported icon.
    """
    fb = _ICON_FALLBACK_COORDS.get(target)
    if fb is None:
        return None

    w, h = adb.get_screen_size()
    x, y = int(w * fb[0]), int(h * fb[1])
    _random_delay()
    sc = _tap_with_screen_check(adb, x, y)
    out = {
        "success": True,
        "target": target,
        "method": "icon_fallback_coord",
        "position": [x, y],
        "pct": [fb[0], fb[1]],
        "screen_changed": sc["screen_changed"],
        "dhash_distance": sc["dhash_distance"],
    }
    if sc["screen_changed"] is False:
        out["_hint"] = (
            "硬编码坐标点击后画面未变化。尝试 magnify 放大右上角 → "
            "查看底部是否有'可收获'按钮 → tap_magnified 直接点击。"
        )
    return ToolOutput(text=json.dumps(out, ensure_ascii=False))


def _adb_available() -> bool:
    try:
        from src.device.emulator import emulator_manager
        return emulator_manager.first_online is not None
    except Exception:
        return False


def _random_delay() -> None:
    delay_ms = random.randint(app_config.adb.action_delay_min_ms, app_config.adb.action_delay_max_ms)
    time.sleep(delay_ms / 1000.0)


def _get_screen_ocr_texts() -> list[str]:
    """Get current screen OCR texts from the agent context, if available."""
    import threading
    ctx = getattr(threading.current_thread(), '_terra_agent_ctx', None)
    if ctx is None:
        return []
    return list(ctx.state.last_ocr_texts) if ctx.state.last_ocr_texts else []


def _is_annihilation_nav_screen(ocr_texts: list[str]) -> bool:
    """Return True if the screen is Terminal's annihilation entry area.

    The annihilation entrance on the Terminal TO-DO LIST is labeled "合成玉"
    (shows weekly reward progress, e.g. "合成玉 0/1800").  Tapping it is
    NAVIGATION into the annihilation map, NOT a currency purchase.

    Stable identifiers (no event/map names):
      - "合成玉" + "/1800" — the weekly cap is always exactly 1800.
        No other screen combines these two strings.
    """
    _joined = " ".join(ocr_texts)
    return "/1800" in _joined and "合成玉" in _joined


def _check_safety(target: str) -> dict | None:
    """Check if target triggers dangerous keyword. Returns error dict or None."""
    from src.tools.safety import check_dangerous, check_confirmation_required
    from src.tools.registry import get_current_game
    game = get_current_game()
    danger = check_dangerous(target, game=game)
    if danger:
        # ── Context override: Terminal Annihilation entry ──
        if danger["keyword"] == "合成玉":
            _screen_ocr = _get_screen_ocr_texts()
            if _screen_ocr and _is_annihilation_nav_screen(_screen_ocr):
                logger.info(
                    "Safety override: '%s' on Terminal annihilation entry "
                    "→ navigation, not purchase", target)
                return None
        logger.warning("SAFETY BLOCK: %s", danger["keyword"])
        return {"success": False, "blocked": True, **danger}
    if check_confirmation_required(target, game=game):
        logger.info("SAFETY CONFIRM: '%s' needs user confirmation", target)
        # Don't block, but flag for the LLM
    return None


def _check_daily() -> dict | None:
    """Check daily action limit (read-only, does NOT increment). Returns error dict or None."""
    from src.tools.safety import check_daily_limit
    limit = check_daily_limit()
    if limit:
        logger.warning("DAILY LIMIT REACHED: %d actions", limit["count"])
        return {"success": False, "blocked": True, **limit}
    return None


def _commit_daily() -> None:
    """Commit a successful daily action to the counter."""
    from src.tools.safety import commit_daily_action
    commit_daily_action()


def _check_screen_safety() -> dict | None:
    """Check if current screen OCR contains dangerous keywords.

    Used by adb_tap_position which has no text target but can hit
    any UI element including purchase/confirm buttons.
    """
    import threading
    ctx = getattr(threading.current_thread(), '_terra_agent_ctx', None)
    if ctx is None:
        return None
    ocr_texts = ctx.state.last_ocr_texts or []
    if not ocr_texts:
        return None
    # ── Context override: Terminal Annihilation entry ──
    if _is_annihilation_nav_screen(ocr_texts):
        return None  # Navigation, not a purchase
    from src.tools.safety import check_dangerous
    from src.tools.registry import get_current_game
    game = get_current_game()
    for text in ocr_texts:
        danger = check_dangerous(text, game=game)
        if danger:
            logger.warning("SAFETY BLOCK (screen): '%s' on screen", danger["keyword"])
            return {"success": False, "blocked": True, **danger}
    return None


# VLM outputs English button names, but OCR can't read Arknights art fonts.
# Map common VLM outputs to their Chinese equivalents for OCR search.
# Moved to src/games/{game}/adapter.py — this is a lazy load fallback.
# Keyed by game_id to prevent cross-game contamination in multi-agent setups.
_VLM_TO_CN_CACHE: dict[str, dict[str, str]] = {}


def _get_vlm_to_cn() -> dict[str, str]:
    """Get the VLM→CN mapping for the current game from its GamePlugin."""
    from src.tools.registry import get_current_game
    from src.games.registry import get_game_registry
    game = get_current_game()
    if game in _VLM_TO_CN_CACHE:
        return _VLM_TO_CN_CACHE[game]
    try:
        plugin = get_game_registry().get(game)
        _VLM_TO_CN_CACHE[game] = plugin.get_vlm_adapter() if plugin else {}
    except Exception:
        _VLM_TO_CN_CACHE[game] = {}
    return _VLM_TO_CN_CACHE[game]




# ── Visual decoration characters that OCR cannot (reliably) detect ──
# Game UI often uses >>, <<, →, ▼, ▲, ▶, ◀, ＞, ＜ as decorative suffixes
# or prefixes on buttons (e.g. "挑战>>", "▼更多", "◀返回").  OCR either
# drops these entirely or misreads them as noise.  Stripping them produces
# a clean search term that matches the readable text OCR actually returns.
_VISUAL_DECORATION_CHARS: list[str] = [
    ">>", "<<", ">>>", "<<<", "→", "←", "↑", "↓",
    "▼", "▲", "▶", "◀", "＞", "＜", "»", "«",
]

_VISUAL_DECORATION_SET: set[str] = set(_VISUAL_DECORATION_CHARS)


def _build_search_terms(target: str) -> list[str]:
    """Build OCR search terms including Chinese equivalents for English VLM output.

    If the target is an English button name VLM would use (e.g. 'Mission'),
    adds the Chinese equivalent (e.g. '任务') that OCR can actually read.
    """
    terms = [target]
    vlm_to_cn = _get_vlm_to_cn()

    # English VLM output → Chinese OCR target
    cn = vlm_to_cn.get(target.lower())
    if cn and cn not in terms:
        terms.append(cn)

    # Also try partial matches: "Mission任务" could be parsed
    for en, cn_val in vlm_to_cn.items():
        if en in target.lower() and cn_val not in terms:
            terms.append(cn_val)

    # Number selector variants: ×6 → [×6, x6, 6] so OCR can match the digit alone.
    # OCR often fails on the × symbol but reliably reads the number.
    num_digits = "".join(ch for ch in target if ch.isdigit())
    if num_digits and num_digits not in terms:
        terms.append(num_digits)

    # ── Visual decoration stripping ──────────────────────────────────
    # Game UI buttons like "挑战>>", "▼更多", "▶确认" render with decorative
    # glyphs that OCR typically misses.  Strip them to generate clean
    # search terms that match the readable portion OCR actually returns.
    stripped = target
    for deco in _VISUAL_DECORATION_CHARS:
        if stripped.endswith(deco):
            stripped = stripped[: -len(deco)]
            break
        elif stripped.startswith(deco):
            stripped = stripped[len(deco):]
            break
    if stripped and stripped != target and stripped not in terms:
        terms.append(stripped)

    # ── OCR-dropped leading "一" ────────────────────────────────────
    # PP-OCRv4 routinely misses the leading 一 in button labels like
    # 一键领取 / 一键派遣 / 一键赠送 because the character is too thin
    # in game fonts.  Add a variant without the leading 一 so substring
    # matching catches "键领取" etc. directly instead of relying on fuzzy.
    if target.startswith("一") and len(target) >= 3:
        stripped_yi = target[1:]
        if stripped_yi not in terms:
            terms.append(stripped_yi)

    return terms


def _fuzzy_match(target: str, ocr_texts: list[str], threshold: float = 0.6) -> str | None:
    """Find the closest OCR text using sequence similarity.

    Returns the matching OCR text, or None if no match above threshold.
    """
    target_lower = target.lower()
    best_score = 0.0
    best_text = None

    for text in ocr_texts:
        text_lower = text.lower()
        # Substring match: require BOTH strings to be >=3 chars AND have meaningful overlap.
        # Single-char fallback (e.g. "1" from "制造站1") must not match "14/5".
        min_len = min(len(text_lower), len(target_lower))
        if min_len >= 3 and (target_lower in text_lower or text_lower in target_lower):
            return text
        # Skip huge length mismatches (e.g. "in" vs "Terminal")
        if len(text_lower) < 2 or len(target_lower) < 2:
            continue
        len_ratio = min(len(text_lower), len(target_lower)) / max(len(text_lower), len(target_lower))
        if len_ratio < 0.4:
            continue
        # Reject when OCR text is a short fragment inside a longer target.
        # Example: "领取" (2 chars) inside "全部领取" (4 chars) — the OCR only
        # found a bare "领取" button, not the full "全部领取" anchor button.
        # Require the matched text to be at least 75% of the target length.
        if text_lower in target_lower and len(text_lower) < len(target_lower) * 0.75:
            logger.debug("Fuzzy: rejecting fragment '%s' ← target '%s' (len ratio %.2f)",
                         text, target, len(text_lower) / len(target_lower))
            continue
        score = SequenceMatcher(None, target_lower, text_lower).ratio()
        if score > best_score:
            best_score = score
            best_text = text

    # Short target strings need higher precision to avoid false matches.
    # e.g. "1-7" must not fuzzy-match to "1-1", and "1" must never match "14/5".
    if len(target) <= 3:
        _thresh = max(threshold, 0.85) if len(target) >= 2 else 0.95
    else:
        _thresh = threshold
    if best_score >= _thresh and best_text is not None:
        logger.debug("Fuzzy match: '%s' → '%s' (%.2f)", target, best_text, best_score)
        return best_text
    return None


def _match_score(detection: dict, search_term: str, match_method: str = "substring") -> float:
    """Score a detection candidate against the search term.  Returns 0–1, higher = better.

    Three dimensions:
      1. Text-match quality (0–0.65): ratio of term length to OCR text length.
         Exact length → 0.65; short term inside long text gets penalised heavily.
      2. Bbox compactness   (0–0.25): compact rectangles look like button labels;
         wide rectangles look like descriptions / compound menu items.
      3. OCR raw confidence (0–0.10): the engine's own confidence score.

    Parameters
    ----------
    detection : dict   – OCR detection with keys "text", "bbox", "confidence".
    search_term : str  – one of the terms returned by _build_search_terms().
    match_method : str – "substring" (Step 1) or "fuzzy" (Step 2 fallback).
    """
    dtext = detection["text"]
    bbox = detection["bbox"]          # (x1, y1, x2, y2)
    conf = float(detection.get("confidence", 0.5))

    # ── 1. Text-match quality (0 – 0.65) ─────────────────────────────
    if match_method == "substring":
        tlen = len(search_term)
        dlen = len(dtext)
        if tlen == dlen:
            text_score = 0.65                              # exact-length match
        elif tlen < dlen:
            ratio = tlen / dlen
            # Prefix match: search term is the start of the OCR text.
            # This happens when OCR picks up extra speech/dialogue text
            # after the target button label (e.g. "十分实用的理论" vs
            # "十分实用的理论！那我们为什么不试试看呢？").  Prefix is a
            # strong signal — score it like a 70%+ ratio match.
            is_prefix = dtext.startswith(search_term)
            if is_prefix and ratio >= 0.3:
                text_score = 0.55                          # prefix of longer text
            elif ratio >= 0.7:
                text_score = 0.55                          # most of the text (e.g. "行动" in "开始行动")
            elif ratio >= 0.5:
                text_score = 0.35                          # half  (e.g. "干员" in "干员寻访")
            else:
                text_score = 0.20                          # small fragment in long text
        else:  # tlen > dlen (OCR truncated)
            text_score = 0.45
    else:  # fuzzy — use SequenceMatcher directly
        from difflib import SequenceMatcher as _SM
        text_score = _SM(None, search_term.lower(), dtext.lower()).ratio() * 0.65

    # ── 2. Bbox compactness (0 – 0.25) ───────────────────────────────
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    if bbox_w <= 120 and bbox_h <= 60:
        bbox_score = 0.25              # compact → button label
    elif bbox_w <= 250 and bbox_h <= 100:
        bbox_score = 0.15              # medium → could be either
    elif bbox_w <= 400:
        bbox_score = 0.08              # wide → description or compound entry
    else:
        bbox_score = 0.02              # very wide → paragraph / notification text

    # ── 3. OCR confidence (0 – 0.10) ─────────────────────────────────
    conf_score = conf * 0.10

    return text_score + bbox_score + conf_score


# Minimum score to accept a single candidate as a click target.
# Below this threshold the match is too ambiguous to risk a tap.
_MIN_TAP_SCORE = 0.35


def _cache_key(target: str, nth: int = 0) -> str:
    """Build a screen_cache-compatible key that distinguishes nth matches.

    When a screen has multiple identically-named buttons (e.g. 6 copies of
    "前往" on a schedule page), caching a single coordinate under the bare
    target name causes nth=1 to return the coordinate for nth=5.  Including
    nth in the key fixes this.
    """
    if nth > 0:
        return f"{target}:nth={nth}"
    return target


def adb_tap(target: str = "", nth: int = 0,
            x_pct: float | None = None, y_pct: float | None = None) -> ToolOutput:
    """Tap a UI element by text. Tool internally handles OCR and coordinate lookup
    — just pass the button name.

    Args:
        target: Text label of the button (Chinese or English).
        nth: Which match to pick when target appears multiple times on screen.
             1 = first (topmost), 2 = second, etc. 0 = auto (best match, default).
             Use this for pages with multiple "前往" / "加速" / "挑战" buttons.
        x_pct: Optional — if provided, redirects to adb_tap_position.
               LLM sometimes confuses adb_tap with adb_tap_position and passes
               coordinate params; accept them and route correctly.
        y_pct: Optional — must be provided together with x_pct.
    """
    # ── Auto-redirect: LLM passed x_pct/y_pct (confused adb_tap with adb_tap_position) ──
    if x_pct is not None and y_pct is not None:
        logger.info("adb_tap: x_pct=%.3f y_pct=%.3f → redirecting to adb_tap_position", x_pct, y_pct)
        return adb_tap_position(x_pct=x_pct, y_pct=y_pct)
    if (x_pct is not None) != (y_pct is not None):
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": "x_pct 和 y_pct 必须同时提供，或都不提供。请只用 target 参数，或同时给 x_pct+y_pct。"
        }, ensure_ascii=False))

    adb = get_adb()
    img = adb.get_screenshot_image()
    shash = _screen_hash(img)

    # Check cache first — VLM already confirmed this button is here.
    # Try all variants (Chinese/English) since cache key may differ from LLM's target.
    # Use nth-aware cache key so "前往" nth=1 and nth=5 return different coordinates.
    #
    # P3: prepare OCR texts for composite cache key lookup.
    # Defined here (not inside nth==0) so VLM fallback at bottom can also use it.
    _ctx_ocr: list[str] | None = None
    try:
        import threading as _thr
        _actx = getattr(_thr.current_thread(), '_terra_agent_ctx', None)
        if _actx is not None:
            _ctx_ocr = list(_actx.state.last_ocr_texts) if _actx.state.last_ocr_texts else None
    except Exception:
        pass

    # IMPORTANT: skip cache when nth > 0.  nth ordering changes when tasks are
    # completed/removed from a list (e.g. sign-in disappears → 思绪漫步 becomes
    # the first "前往").  Crop OCR can verify the text exists but cannot tell
    # which Nth occurrence it is — so a stale cached coordinate for nth=1 will
    # pass verification but point to the wrong button.
    if nth == 0:

        cache_term = _cache_key(target, nth)
        cached = None
        for term in _build_search_terms(cache_term):
            cached = screen_cache.get(shash, term, device_serial=adb.serial,
                                      ocr_texts=_ctx_ocr)
            if cached is not None:
                break
        if cached is None:
            for term in _build_search_terms(target):
                cached = screen_cache.get(shash, term, device_serial=adb.serial,
                                          ocr_texts=_ctx_ocr)
                if cached is not None:
                    break
    else:
        cached = None
    if cached is not None:
        cx, cy, x1, y1, x2, y2 = cached
        is_icon = target in _ICON_TARGETS
        if is_icon:
            logger.debug("Cache hit (icon): '%s' tapped at %s", target, (cx, cy))
            _random_delay()
            sc = _tap_with_screen_check(adb, cx, cy)
            result = {
                "success": True,
                "target": target,
                "method": "cached_icon",
                "position": [cx, cy],
                "screen_changed": sc["screen_changed"],
                "dhash_distance": sc["dhash_distance"],
            }
            if sc["screen_changed"] is False:
                result["_hint"] = "画面没有变化。图标缓存坐标可能已过期。"
            return ToolOutput(text=json.dumps(result, ensure_ascii=False))
        # Text button: verify with crop OCR for precision
        region_text = ocr_engine.read_region(img, x1, y1, x2 - x1, y2 - y1)
        search_terms = _build_search_terms(target)
        for term in search_terms:
            if term.lower() in region_text.lower():
                _random_delay()
                sc = _tap_with_screen_check(adb, cx, cy)
                logger.debug("Cache hit: '%s' tapped at %s (screen_changed=%s)",
                           target, (cx, cy), sc["screen_changed"])
                result = {
                    "success": True,
                    "target": target,
                    "method": "cached",
                    "position": [cx, cy],
                    "screen_changed": sc["screen_changed"],
                    "dhash_distance": sc["dhash_distance"],
                }
                if sc["screen_changed"] is False:
                    result["_hint"] = "画面没有变化。缓存坐标可能已过期。"
                return ToolOutput(text=json.dumps(result, ensure_ascii=False))
        # Text not found in crop — for pinned targets that may have scrolled
        # off-screen, fail with a scroll-back hint instead of blind-tapping.
        is_pinned = target in _PINNED_TARGETS
        if is_pinned:
            return ToolOutput(text=json.dumps({
                "success": False,
                "target": target,
                "method": "cached_stale",
                "message": (
                    f"'{target}' 不在缓存的屏幕位置上了（可能已滚出视图）。"
                    f"'{target}' 通常在列表/面板顶部，请先向上滚动到顶端后再点击。"
                ),
            }, ensure_ascii=False))
        # Crop OCR failed — page state may have changed (e.g. "前往" became
        # "领取" after task completion).  Blind-tapping a stale coordinate
        # risks hitting a completely different button.
        # Fall through to full OCR scan to find the actual current position.
        logger.warning(
            "Stale cache: '%s' not at cached %s (screen hash %s), "
            "falling through to full OCR",
            target, (cx, cy), shash[:8],
        )

    # ── Icon fast-path: hardcoded coordinate for non-text targets ──
    # Icons like the bell (铃铛) have no OCR-readable text.  Cache misses
    # are expected.  Skip the full OCR scan and use a verified coordinate.
    if cached is None and target in _ICON_TARGETS:
        fb_result = _try_icon_fallback_tap(target, adb)
        if fb_result is not None:
            return fb_result

    # Full OCR scan — check cache first (populated by screen_injector,
    # which already OCR'd this screenshot on the last injection cycle).
    if shash and shash in _ocr_cache:
        all_detections = _ocr_cache[shash]
        logger.debug("adb_tap: OCR cache hit for '%s' (%d detections, hash=%s)",
                      target, len(all_detections), shash[:8])
    else:
        all_detections = ocr_engine.read_text(img)
    search_detections = all_detections

    if not all_detections:
        return ToolOutput(text=json.dumps({
            "success": False,
            "target": target,
            "message": (
                "OCR 未检测到任何文字，画面可能为空或加载中。"
                "💡 如果刚执行过导航操作 → 画面可能是过渡中的黑屏/加载页，等等系统自动注入新截图即可，不要重复点击。"
                "如果等了 10 秒以上仍无文字 → 调用 ask_user() 确认模拟器状态。"
            ),
        }, ensure_ascii=False))

    search_terms = _build_search_terms(target)
    logger.debug("adb_tap search terms for '%s': %s", target, search_terms)

    # Step 1: Substring match with multi-dimensional scoring.
    # Collect ALL candidates above threshold, not just the best one.
    # When nth is specified, the Nth match (sorted top-to-bottom) is used —
    # critical for pages with multiple "前往" / "加速" / "挑战" buttons.
    is_pinned = target in _PINNED_TARGETS
    candidates: list[tuple[dict, float, bool]] = []  # (detection, score, fuzzy)
    seen_positions: set[tuple[int, int]] = set()
    for d in search_detections:
        for term in search_terms:
            if term.lower() in d["text"].lower():
                if term.isdigit() and len(term) <= 2 and "-" in d["text"]:
                    continue
                if is_pinned and len(d["text"]) <= len(target) * 0.5:
                    continue
                score = _match_score(d, term, "substring")
                if score >= _MIN_TAP_SCORE:
                    pos = (d["center"][0], d["center"][1])
                    if pos not in seen_positions:
                        seen_positions.add(pos)
                        candidates.append((d, score, False))

    # Step 1.5: Neighbor merge — OCR sometimes splits a single UI label
    # into horizontally-adjacent fragments (e.g. "闪亮之旅" → "闪亮" + "之族"
    # because the font styling causes OCR to see two separate text blocks).
    # Merge adjacent detections and re-check substring match before falling
    # back to fuzzy matching.
    if not candidates:
        _H_MERGE_GAP = 48   # max horizontal gap in pixels between mergeable fragments
        _V_MERGE_TOL = 12   # max vertical drift (same line)
        for i, di in enumerate(search_detections):
            for j, dj in enumerate(search_detections):
                if i >= j:
                    continue
                # Must be on the same visual line
                yi, yj = di["center"][1], dj["center"][1]
                if abs(yi - yj) > _V_MERGE_TOL:
                    continue
                # j must be to the right of i, with a small gap (not overlapping)
                xi_right = di["bbox"][2]
                xj_left = dj["bbox"][0]
                gap = xj_left - xi_right
                if gap < 0 or gap > _H_MERGE_GAP:
                    continue
                merged = di["text"] + dj["text"]
                for term in search_terms:
                    if term.lower() in merged.lower():
                        # synthetic detection from merged fragments
                        syn = {
                            "text": merged,
                            "center": [
                                (di["center"][0] + dj["center"][0]) // 2,
                                (yi + yj) // 2,
                            ],
                            "bbox": [
                                di["bbox"][0], min(di["bbox"][1], dj["bbox"][1]),
                                dj["bbox"][2], max(di["bbox"][3], dj["bbox"][3]),
                            ],
                            "confidence": min(
                                float(di.get("confidence", 0.5)),
                                float(dj.get("confidence", 0.5)),
                            ),
                        }
                        score = _match_score(syn, term, "substring")
                        if score >= _MIN_TAP_SCORE:
                            pos = (syn["center"][0], syn["center"][1])
                            if pos not in seen_positions:
                                seen_positions.add(pos)
                                candidates.append((syn, score, False))
                                logger.info(
                                    "adb_tap: neighbor merge '%s'+'%s' → '%s' matched '%s' score=%.2f",
                                    di["text"], dj["text"], merged, term, score,
                                )
                        break  # one term match is enough for this pair
                if candidates:
                    break  # one good candidate is enough
            if candidates:
                break

    # Step 2: Fuzzy match fallback
    if not candidates:
        ocr_texts = [d["text"] for d in search_detections]
        for term in search_terms:
            matched_text = _fuzzy_match(term, ocr_texts)
            if matched_text:
                for d in search_detections:
                    if SequenceMatcher(None, d["text"].lower(), matched_text.lower()).ratio() >= 0.6:
                        score = _match_score(d, term, "fuzzy")
                        if score >= _MIN_TAP_SCORE:
                            pos = (d["center"][0], d["center"][1])
                            if pos not in seen_positions:
                                seen_positions.add(pos)
                                candidates.append((d, score, True))
                if candidates:
                    break

    # Sort candidates by Y position (top to bottom) for stable nth ordering
    candidates.sort(key=lambda x: (x[0]["center"][1], x[0]["center"][0]))

    if not candidates:
        available_texts = [d["text"] for d in all_detections[:20]]
        return ToolOutput(text=json.dumps({
            "success": False,
            "target": target,
            "available_texts": available_texts,
            "message": (
                f"'{target}' 未在画面中找到。"
                f"请确认目标是否真的在屏幕上，或用 magnify 查看。"
            ),
        }, ensure_ascii=False))

    # ── nth selection ──
    total = len(candidates)
    if nth > 0 and nth <= total:
        best, best_score, matched_via_fuzzy = candidates[nth - 1]
        logger.info("adb_tap: nth=%d/%d matched '%s' score=%.2f", nth, total, best.get("text", "?"), best_score)
    elif nth > total:
        texts = [c[0]["text"] for c in candidates]
        centers = [c[0]["center"] for c in candidates]
        return ToolOutput(text=json.dumps({
            "success": False,
            "target": target,
            "nth": nth,
            "total_matches": total,
            "match_texts": texts,
            "message": (
                f"'{target}' 在画面中出现了 {total} 次（{', '.join(texts)}），"
                f"但你指定了 nth={nth}（第 {nth} 个），超出范围。"
                f"匹配按从上到下排列：1={texts[0] if texts else '?'}，请选择有效的 nth 值。"
            ),
        }, ensure_ascii=False))
    else:
        # nth=0 or not specified: use best score (current behavior)
        best, best_score, matched_via_fuzzy = max(candidates, key=lambda x: x[1])
        if total > 1:
            logger.info("adb_tap: auto-picked from %d '%s' matches (use nth=N for Nth)", total, target)

    if best:
        safety_check = _check_safety(target)
        if safety_check:
            return ToolOutput(text=json.dumps(safety_check, ensure_ascii=False))

        daily_check = _check_daily()
        if daily_check:
            return ToolOutput(text=json.dumps(daily_check, ensure_ascii=False))

        center = best["center"]
        bbox = best["bbox"]
        conf = best["confidence"]
        method_parts = [f"ocr({conf:.2f})", f"score={best_score:.2f}"]
        if matched_via_fuzzy:
            method_parts.append(f"fuzzy:'{best['text']}'")
        used_method = " ".join(method_parts)

        logger.debug(
            "adb_tap: '%s' → '%s' score=%.2f (text+bbox+conf) method=%s",
            target, best["text"], best_score, used_method,
        )

        # Cache for next time — use nth-aware key so each nth gets its own slot
        screen_cache.set(shash, _cache_key(target, nth), center[0], center[1],
                         bbox[0], bbox[1], bbox[2], bbox[3],
                         device_serial=adb.serial)

        _random_delay()
        sc = _tap_with_screen_check(adb, center[0], center[1])
        _commit_daily()

        result = {
            "success": True,
            "target": target,
            "matched_text": best["text"],
            "method": used_method,
            "position": center,
            "screen_changed": sc["screen_changed"],
            "dhash_distance": sc["dhash_distance"],
        }
        if sc["screen_changed"] is False:
            result["_hint"] = "画面没有变化。点击可能未命中，尝试备用坐标或 magnify。"
            logger.warning(
                "adb_tap: '%s' → '%s' screen UNCHANGED (dhash dist=%d)",
                target, best["text"], sc["dhash_distance"],
            )

        # Checkbox verification: tap then OCR the region to confirm state change
        if target in _CHECKBOX_TARGETS:
            try:
                import re as _re
                img2 = adb.get_screenshot_image()
                region_text = ocr_engine.read_region(img2, bbox[0]-10, bbox[1]-10,
                                                     bbox[2]-bbox[0]+20, bbox[3]-bbox[1]+20)
                if _re.search(r"X\d+", region_text):
                    result["checkbox_state"] = f"已开启 ({region_text.strip()})"
                elif _re.search(r"\d+", region_text):
                    result["checkbox_state"] = f"已开启 (检测到数字: {region_text.strip()})"
                else:
                    result["checkbox_state"] = f"未检测到X/数字标志，文本: {region_text.strip()}"
            except Exception:
                result["checkbox_state"] = "验证失败"

        return ToolOutput(text=json.dumps(result, ensure_ascii=False))

    # OCR failed — try VLM-cached coordinate as fallback (for icon-only buttons)
    vlm_fallback = None
    for term in _build_search_terms(_cache_key(target, nth)):
        vlm_fallback = screen_cache.get(shash, term, device_serial=adb.serial,
                                          ocr_texts=_ctx_ocr)
        if vlm_fallback is not None:
            break
    if vlm_fallback is not None:
        cx, cy, *_ = vlm_fallback
        _random_delay()
        sc = _tap_with_screen_check(adb, cx, cy)
        _commit_daily()
        result = {
            "success": True,
            "target": target,
            "method": "vlm_cached",
            "position": [cx, cy],
            "screen_changed": sc["screen_changed"],
            "dhash_distance": sc["dhash_distance"],
        }
        if sc["screen_changed"] is False:
            result["_hint"] = "画面没有变化。VLM 缓存坐标可能已过期。"
        return ToolOutput(text=json.dumps(result, ensure_ascii=False))

    available_texts = [d["text"] for d in all_detections[:20]]
    # Build a hint: if the available texts look like target-page content
    # (e.g. "等级/稀有度/职业" → operator list), tell the LLM the previous
    # tap likely succeeded and it should work with the current screen.
    page_hint = ""
    _target_page_markers = {
        "干员": ["等级", "稀有度", "晋升", "职业"],
        "基建": ["制造站", "贸易站", "发电站", "控制中枢"],
        "作战": ["开始行动", "编队", "代理指挥"],
        "采购": ["时装", "礼包", "信用", "组合包"],
    }
    for marker_page, markers in _target_page_markers.items():
        if target in marker_page or any(m in str(available_texts) for m in markers):
            if target not in str(available_texts):
                page_hint = (
                    f"\n💡 当前画面文字中出现了 {marker_page}列表特征 "
                    f"({'/'.join(markers[:3])}) — 上一轮点击可能已成功！"
                    f"请基于当前画面继续操作，不要重复点击'{target}'。"
                )
            break

    # Pinned-target hint: if a top-anchored button isn't on screen,
    # it likely scrolled off. Tell the agent to scroll back up instead
    # of letting it guess coordinates or pick a different button.
    pinned_hint = ""
    if target in _PINNED_TARGETS:
        pinned_hint = (
            f"\n🔝 '{target}' 是列表顶部的固定按钮，如果当前不在画面上，"
            f"说明列表被滚动了。请用 adb_scroll direction=prev distance=full "
            f"滚回顶端后再点。不要用 adb_tap_position 猜测坐标！"
        )

    # Generic fallback action. Pinned targets override this with their own
    # scroll-back instruction, so don't suggest coordinate-guessing then.
    generic_next = ""
    if not pinned_hint:
        generic_next = (
            f"\n下一步：从 available_texts 选一个实际存在的文字重试 adb_tap，"
            f"或直接用 adb_tap_position 猜坐标。不要用 magnify——它比 adb_tap_position 慢 3 倍，"
            f"而且大部分情况下 adb_tap_position 一次就中。"
        )

    return ToolOutput(text=json.dumps({
        "success": False,
        "target": target,
        "available_texts": available_texts,
        "message": (
            f"找不到'{target}'。画面文字：{', '.join(available_texts[:15])}。"
            f"{pinned_hint}"
            f"{page_hint}"
            f"{generic_next}"
        ),
    }, ensure_ascii=False))


def adb_swipe(direction: str, distance: str = "half", area: str = "center") -> ToolOutput:
    """Swipe on screen.

    Args:
        direction: "up", "down", "left", "right"
        distance: "half" (half screen), "full", or "small"
        area: Where to start the swipe — "top", "center", "bottom", "left_edge", "right_edge",
              "top_bar", or "bottom_bar". Use "top_bar" for episode/chapter tab bars.
    """
    adb = get_adb()
    w, h = adb.get_screen_size()

    # Determine swipe start position based on area
    area_positions = {
        "top": (w // 2, h // 4),
        "center": (w // 2, h // 2),
        "bottom": (w // 2, 3 * h // 4),
        "top_bar": (w // 2, h // 8),
        "bottom_bar": (w // 2, 7 * h // 8),
        "left_edge": (w // 6, h // 2),
        "right_edge": (5 * w // 6, h // 2),
    }
    cx, cy = area_positions.get(area, (w // 2, h // 2))

    ratios = {"half": 0.5, "full": 0.8, "small": 0.25}
    ratio = ratios.get(distance, 0.5)

    offsets = {
        "up": (0, int(-h * 0.4)),
        "down": (0, int(h * 0.4)),
        "left": (int(-w * ratio), 0),
        "right": (int(w * ratio), 0),
    }
    dx, dy = offsets.get(direction, (0, 0))

    _random_delay()
    adb.swipe(cx, cy, cx + dx, cy + dy)
    return ToolOutput(text=json.dumps({
        "success": True,
        "direction": direction,
        "distance": distance,
        "area": area,
        "from": [cx, cy],
    }))


# ── Semantic → physical direction mapping ──────────────────────────
# LLM says "next page" (content intent); the table maps that to the
# correct ADB swipe direction.  This mapping is a physical fact —
# finger-left → content-left on every platform ADB touches.
#
#             │ horizontal       │ vertical
#  ───────────┼──────────────────┼──────────
#  next       │ left             │ up
#  prev       │ right            │ down
#  more       │ left  (continue) │ up  (continue)

_SEMANTIC_TO_PHYSICAL: dict[str, dict[str, str]] = {
    "horizontal": {"next": "left",  "prev": "right", "more": "left"},
    "vertical":   {"next": "up",    "prev": "down",  "more": "up"},
}


def adb_scroll(direction: str, axis: str = "horizontal",
               distance: str = "half", area: str = "center") -> ToolOutput:
    """Semantic scrolling — describe WHERE content should move, not HOW your finger moves.

    LLM never needs to reason about finger-direction vs content-direction.
    The mapping is a physical constant, guaranteed correct.

    Args:
        direction: "next" (see next page / more content),
                   "prev" (go back to previous page),
                   "more" (continue current direction — use during continuous scanning)
        axis: "horizontal" (left-right scroll, e.g. operator grid, chapter tabs)
              or "vertical" (up-down scroll, e.g. stage list, shop panel)
        distance: "half" (default), "full", or "small"
        area: Where to start the swipe — "center" (default), "top", "bottom",
              "left_edge", "right_edge", "top_bar", "bottom_bar"
    """
    axis_map = _SEMANTIC_TO_PHYSICAL.get(axis)
    if axis_map is None:
        return ToolOutput(text=json.dumps({
            "success": False,
            "error": f"Unknown axis '{axis}'. Must be 'horizontal' or 'vertical'.",
        }, ensure_ascii=False))

    phys_dir = axis_map.get(direction)
    if phys_dir is None:
        return ToolOutput(text=json.dumps({
            "success": False,
            "error": f"Unknown direction '{direction}'. Must be 'next', 'prev', or 'more'.",
        }, ensure_ascii=False))

    logger.debug("adb_scroll: %s/%s → physical swipe %s", direction, axis, phys_dir)
    result = adb_swipe(phys_dir, distance=distance, area=area)
    # Replace physical direction with semantic direction in the output text
    # so the LLM sees "方向: next" rather than "方向: left".
    try:
        data = json.loads(result.text)
        data["direction"] = direction          # "next" / "prev" / "more"
        data["physical"] = phys_dir            # "left" / "right" / "up" / "down"
        data["axis"] = axis
        result.text = json.dumps(data, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass
    return result


def adb_tap_position(x_pct: float, y_pct: float) -> ToolOutput:
    """Tap at a screen-relative position when the target has no text (e.g. arrow icons).

    Args:
        x_pct: X position as percentage of screen width (0.0 - 1.0). 0.5 = center.
        y_pct: Y position as percentage of screen height (0.0 - 1.0). 0.1 = near top.
    """
    adb = get_adb()
    w, h = adb.get_screen_size()
    x = int(w * x_pct)
    y = int(h * y_pct)

    # ── Safety: same checks as adb_tap ──
    # Check screen-level dangerous keywords (positional tap can hit any UI element)
    safety_check = _check_screen_safety()
    if safety_check:
        return ToolOutput(text=json.dumps(safety_check, ensure_ascii=False))
    daily_check = _check_daily()
    if daily_check:
        return ToolOutput(text=json.dumps(daily_check, ensure_ascii=False))

    _random_delay()
    sc = _tap_with_screen_check(adb, x, y)
    _commit_daily()

    result = {
        "success": True,
        "position": [x, y],
        "pct": [x_pct, y_pct],
        "screen_changed": sc["screen_changed"],
        "dhash_distance": sc["dhash_distance"],
    }
    if sc["screen_changed"] is False:
        result["message"] = (
            "画面没有变化。点击可能未命中目标，"
            "尝试备用坐标或 magnify 放大定位。"
        )
    return ToolOutput(text=json.dumps(result, ensure_ascii=False))


# ── Row-based smart targeting (P0 fix for same-name buttons like "前往") ──
# When a page has 5+ "前往" buttons (schedule pages, task lists), nth-based
# selection fails because the LLM doesn't know which Y position maps to which
# task.  adb_tap_smart lets the LLM specify BOTH the target button text AND
# the row's task-description text, matching by row-proximity instead of nth.

_ROW_Y_RANGE = 60  # max vertical distance for two OCR boxes to be "in the same row"


def _find_row_bbox(
    row_text: str,
    all_detections: list[dict],
    nth: int = 0,
) -> tuple[int, int] | None:
    """Find the Y range (y_min, y_max) of a row containing `row_text`.

    Returns None if the row text is not found on screen.
    When nth > 0, selects the nth matching row (top-to-bottom).
    """
    row_candidates: list[dict] = []
    for d in all_detections:
        if row_text.lower() in d["text"].lower():
            row_candidates.append(d)
    if not row_candidates:
        # Try partial match — long task descriptions may be split across OCR boxes
        row_chars: set[str] = set(row_text.replace(" ", ""))
        for d in all_detections:
            overlap = sum(1 for ch in row_chars if ch in d["text"])
            if overlap >= min(3, len(row_chars) * 0.5):
                row_candidates.append(d)
    if not row_candidates:
        return None
    row_candidates.sort(key=lambda d: d["center"][1])  # top to bottom
    if nth > 0 and nth <= len(row_candidates):
        best = row_candidates[nth - 1]
    else:
        # Pick the one with the longest text (most likely the full task description)
        best = max(row_candidates, key=lambda d: len(d["text"]))
    bbox = best["bbox"]
    return (bbox[1], bbox[3])


def adb_tap_smart(
    target: str,
    row_text: str = "",
    nth: int = 0,
) -> ToolOutput:
    """Tap a button by finding it in the same row as a task description.

    Use this when a page has many same-named buttons (e.g. 5 "前往" buttons
    on a schedule/task page) and nth-based adb_tap keeps hitting the wrong one.
    Specify both the button text AND the task description text in the same row.

    Args:
        target: The button label to tap (e.g. "前往", "领取", "挑战").
        row_text: Text in the same row that identifies which task/entry
                  the button belongs to (e.g. "在时尚对决中完成1次挑战").
        nth: When row_text appears multiple times, which row (1=first/top).
             Default 0 = auto (longest match).
    """
    adb = get_adb()
    img = adb.get_screenshot_image()
    shash = _screen_hash(img)

    # Use cached OCR if available
    if shash and shash in _ocr_cache:
        all_detections = _ocr_cache[shash]
        logger.debug("adb_tap_smart: OCR cache hit (%d detections)", len(all_detections))
    else:
        all_detections = ocr_engine.read_text(img)

    if not all_detections:
        return ToolOutput(text=json.dumps({
            "success": False,
            "target": target,
            "row_text": row_text,
            "message": "OCR 未检测到任何文字，画面可能为空或加载中。",
        }, ensure_ascii=False))

    # Step 1: Find the target row's Y range
    row_bbox = None
    row_center_y = None
    if row_text:
        row_bbox = _find_row_bbox(row_text, all_detections, nth=nth if nth > 0 else 0)
        if row_bbox is not None:
            row_center_y = (row_bbox[0] + row_bbox[1]) / 2

    # Step 2: Find all candidate buttons matching the target
    search_terms = _build_search_terms(target)
    candidates: list[tuple[dict, float]] = []
    seen_positions: set[tuple[int, int]] = set()
    for d in all_detections:
        for term in search_terms:
            if term.lower() in d["text"].lower():
                if term.isdigit() and len(term) <= 2 and "-" in d["text"]:
                    continue
                score = _match_score(d, term, "substring")
                if score >= _MIN_TAP_SCORE:
                    pos = (d["center"][0], d["center"][1])
                    if pos not in seen_positions:
                        seen_positions.add(pos)
                        candidates.append((d, score))

    if not candidates:
        available_texts = [d["text"] for d in all_detections[:20]]
        return ToolOutput(text=json.dumps({
            "success": False,
            "target": target,
            "row_text": row_text,
            "available_texts": available_texts,
            "message": f"'{target}' 未在画面中找到。",
        }, ensure_ascii=False))

    # Step 3: Filter candidates by row proximity
    candidates.sort(key=lambda x: x[0]["center"][1])  # top to bottom

    if row_center_y is not None:
        # Keep only candidates whose center Y is within _ROW_Y_RANGE of the row center
        row_candidates = [
            (d, s) for d, s in candidates
            if abs(d["center"][1] - row_center_y) <= _ROW_Y_RANGE
        ]
        if row_candidates:
            # Within the row, pick the rightmost candidate (buttons are usually on the right)
            row_candidates.sort(key=lambda x: -x[0]["center"][0])
            best, best_score = row_candidates[0]
            logger.info(
                "adb_tap_smart: row-matched '%s' via row_text='%s' (y=%.0f, %d candidates in row)",
                target, row_text[:40], row_center_y, len(row_candidates),
            )
        else:
            # No candidate in exact row — fall back to closest candidate by Y
            candidates.sort(key=lambda x: abs(x[0]["center"][1] - row_center_y))
            best, best_score = candidates[0]
            logger.info(
                "adb_tap_smart: no exact row match for '%s', using closest Y candidate '%s'",
                row_text[:40], best["text"],
            )
    else:
        # No row_text specified — fall back to standard best-match
        if nth > 0 and nth <= len(candidates):
            best, best_score = candidates[nth - 1]
        else:
            best, best_score = max(candidates, key=lambda x: x[1])
        if len(candidates) > 1:
            logger.info(
                "adb_tap_smart: auto-picked from %d '%s' matches (provide row_text for precision)",
                len(candidates), target,
            )

    # Step 4: Execute the tap
    safety_check = _check_safety(target)
    if safety_check:
        return ToolOutput(text=json.dumps(safety_check, ensure_ascii=False))
    daily_check = _check_daily()
    if daily_check:
        return ToolOutput(text=json.dumps(daily_check, ensure_ascii=False))

    center = best["center"]
    bbox = best["bbox"]
    conf = best["confidence"]
    method = f"smart(ocr={conf:.2f}, score={best_score:.2f})"
    if row_text:
        method += f", row='{row_text[:30]}'"

    # Cache for subsequent taps on the same screen
    screen_cache.set(shash, _cache_key(target, nth), center[0], center[1],
                     bbox[0], bbox[1], bbox[2], bbox[3],
                     device_serial=adb.serial)

    _random_delay()
    sc = _tap_with_screen_check(adb, center[0], center[1])
    _commit_daily()

    result = {
        "success": True,
        "target": target,
        "row_text": row_text,
        "matched_text": best["text"],
        "method": method,
        "position": center,
        "screen_changed": sc["screen_changed"],
        "dhash_distance": sc["dhash_distance"],
    }
    if sc["screen_changed"] is False:
        result["_hint"] = "画面没有变化。点击可能未命中目标行。"
    return ToolOutput(text=json.dumps(result, ensure_ascii=False))


def dismiss_all_popups(max_dismissals: int = 10) -> ToolOutput:
    """Dismiss all popups by repeatedly pressing back until the screen stabilizes.

    Uses dHash comparison to detect when each popup closes and the screen
    transitions to either the next popup or the underlying game UI.

    Args:
        max_dismissals: Maximum number of popups to dismiss (safety cap).
                        Default 10 is safe — games rarely show more than 10
                        stacked popups on login.

    Returns how many popups were dismissed and the final screen state.
    """
    from src.utils.dhash import compute_dhash, dhash_to_hex, hamming_distance

    adb = get_adb()
    dismissed = 0
    no_change_streak = 0

    for i in range(max_dismissals):
        # Capture current screen
        img = adb.get_screenshot_image()
        pre_dhash = dhash_to_hex(compute_dhash(img))

        # Press back
        _random_delay()
        adb.press_back()
        time.sleep(0.6)  # Wait for popup close animation

        # Capture new screen
        img2 = adb.get_screenshot_image()
        post_dhash = dhash_to_hex(compute_dhash(img2))

        try:
            dist = hamming_distance(
                int(pre_dhash, 16) if isinstance(pre_dhash, str) else pre_dhash,
                int(post_dhash, 16) if isinstance(post_dhash, str) else post_dhash,
            )
        except Exception:
            dist = 0

        if dist >= 13:
            # Screen changed significantly — popup was dismissed
            dismissed += 1
            no_change_streak = 0
            logger.info(
                "dismiss_all_popups: #%d dismissed (dHash dist=%d)",
                dismissed, dist,
            )
            continue
        elif dist <= 10:
            # Screen barely changed — popup didn't respond to back
            no_change_streak += 1
            if no_change_streak >= 2:
                # Try tapping center of screen (some popups need a tap, not back)
                if no_change_streak == 2:
                    w, h = adb.get_screen_size()
                    adb.tap(w // 2, h // 2)
                    time.sleep(0.6)
                elif no_change_streak >= 3:
                    logger.info(
                        "dismiss_all_popups: screen stable after %d dismissals, "
                        "streak=%d — stopping",
                        dismissed, no_change_streak,
                    )
                    break
        else:
            # Intermediate change (11-12) — popup partially dismissed
            dismissed += 1
            no_change_streak = 0

    return ToolOutput(text=json.dumps({
        "success": True,
        "dismissed": dismissed,
        "message": (
            f"关闭了 {dismissed} 个弹窗。"
            if dismissed > 0
            else "没有检测到弹窗，或弹窗不需要关闭。"
        ),
    }, ensure_ascii=False))


def adb_back() -> ToolOutput:
    """Press the Android back button. Includes screen-change detection."""
    adb = get_adb()
    try:
        pre_img = adb.get_screenshot_image()
        pre_dhash = compute_dhash(pre_img)
    except Exception:
        pre_dhash = None
    _random_delay()
    adb.press_back()
    time.sleep(0.15)
    result: dict = {"success": True}
    if pre_dhash is not None:
        try:
            post_img = adb.get_screenshot_image()
            post_dhash = compute_dhash(post_img)
            dist = hamming_distance(pre_dhash, post_dhash)
            result["screen_changed"] = dist >= 5
            result["dhash_distance"] = dist
        except Exception:
            result["screen_changed"] = None
    return ToolOutput(text=json.dumps(result))


# Register tools
registry.register(
    name="adb_tap",
    description=(
        "[首选] 点击屏幕上可见的文字按钮。只需传入按钮上的文字（中文或英文），工具内部自动处理 OCR 识别和坐标定位。\n"
        "使用时机：目标按钮有可读的文字标签（如 '开始行动'、'任务'、'Terminal'）。\n"
        "不要使用：目标是无文字的图标按钮（◀ ▶ ×）→ 用 adb_tap_position。\n"
        "失败处理：同一按钮连续失败 2 次 → 调用 magnify() 放大查看，不要反复点击。\n"
        "注意：用 OCR 实际输出的文字去 tap（如 Termlnal 对应 Terminal），不要纠正 OCR 结果。\n"
        "nth 参数：当页面有多个同名按钮（如5个'前往'）时，用 nth=1 选第一个（最上面）、nth=2 选第二个，以此类推。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "按钮上的精确文字，从当前截图中 OCR 实际输出的文字中选择，优先使用中文。"},
            "nth": {"type": "integer", "description": "可选。当 target 在页面上出现多次时，指定第几个（从上到下排序，1=第一个）。不填=自动选最匹配的。常用于多个'前往'/'加速'/'挑战'按钮的区分。"},
        },
        "required": ["target"],
    },
    handler=adb_tap,
    check_fn=_adb_available,
)

registry.register(
    name="adb_swipe",
    description=(
        "[备选] 直接用物理方向滑动屏幕。优先使用 adb_scroll 避免方向搞反。\n"
        "方向说明：up=手指上滑内容上移看下方；down=手指下滑内容下移看上方。\n"
        "left=手指左滑内容左移看右侧；right=手指右滑内容右移看左侧。\n"
        "使用时机：仅在完全确定需要特定物理方向时使用。不确定方向时用 adb_scroll。\n"
        "★ 滑动后检查系统自动注入的截图是否变化：hash 不变 = 滑动无效 = 可能已到列表边界。\n"
        "同一方向连续滑动 2 次屏幕 hash 不变 → 已到头 → 换方向。不要凭感觉提前换方向。\n"
        "同一方向连续滑动超过 3 次仍未找到目标 → 调用 ask_user()，不要继续试。\n"
        "area 参数：列表滚动用 'center'（默认），顶部标签栏切换用 'top_bar'。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
            "distance": {"type": "string", "enum": ["half", "full", "small"], "description": "Default: half"},
            "area": {"type": "string", "enum": ["top", "center", "bottom", "top_bar", "bottom_bar", "left_edge", "right_edge"], "description": "Where to start the swipe. 'top_bar' for chapter tabs, 'center' (default) for general scrolling."},
        },
        "required": ["direction"],
    },
    handler=adb_swipe,
    check_fn=_adb_available,
)

registry.register(
    name="adb_scroll",
    description=(
        "[首选] 语义滚动 — 描述内容的移动方向而非手指方向，避免方向搞反。\n"
        "direction: 'next'=看下一页内容, 'prev'=回看上一页, 'more'=继续当前方向（连续扫描用）。\n"
        "axis: 'horizontal'=水平滚动(干员列表/标签栏), 'vertical'=垂直滚动(关卡列表/商店)。\n"
        "使用时机：需要滚动列表查看当前屏幕外的内容时。\n"
        "不要使用：当目标按钮已知且在屏幕上可见时 → 直接用 adb_tap。\n"
        "★ 不确定 axis 时：看截图判断列表是横排还是竖排，横排用 horizontal，竖排用 vertical。\n"
        "如果猜错了一两次再换就好——这比猜手指方向简单得多。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["next", "prev", "more"],
                "description": "next=看下一页, prev=回看上一页, more=继续当前方向(连续扫描)"
            },
            "axis": {
                "type": "string",
                "enum": ["horizontal", "vertical"],
                "description": "horizontal=水平滚动(横排列表), vertical=垂直滚动(竖排列表)"
            },
            "distance": {"type": "string", "enum": ["half", "full", "small"], "description": "Default: half"},
            "area": {"type": "string", "enum": ["top", "center", "bottom", "top_bar", "bottom_bar", "left_edge", "right_edge"], "description": "Where to start the swipe. Default: center"},
        },
        "required": ["direction", "axis"],
    },
    handler=adb_scroll,
    check_fn=_adb_available,
)

registry.register(
    name="adb_tap_position",
    description=(
        "[备选] 点击无文字标签的屏幕位置，使用百分比坐标（0-1）。\n"
        "使用时机：目标是无文字的图标按钮（如 ◀ ▶ × 关闭 等纯图标元素），且你还没有调用过 magnify()。\n"
        "⚠️ 重要：如果你已经调用了 magnify() 并拿到了高清截图 → 绝对不要再用 adb_tap_position 盲猜百分比坐标！\n"
        "必须走 magnify() → 从坐标标尺读出像素坐标 → mark_position() 确认 → tap_magnified() 点击。\n"
        "adb_tap_position 的百分比坐标是盲猜——你猜 7 次也猜不中对勾按钮的位置。\n"
        "更优先的选项：如果弹窗关不掉，先用 adb_back() 试试——Android 返回键能关掉绝大多数弹窗。\n"
        "不要使用：目标有文字标签 → 用 adb_tap。微小目标（复选框、次数选择器）→ 用 magnify→tap_magnified 流程。\n"
        "坐标约束：x_pct 和 y_pct 必须在 0.05~0.95 之间，不要用 0.0 或 1.0（会被系统栏遮挡）。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "x_pct": {"type": "number", "description": "X position as fraction of screen width (0.05-0.95)"},
            "y_pct": {"type": "number", "description": "Y position as fraction of screen height (0.05-0.95)"},
        },
        "required": ["x_pct", "y_pct"],
    },
    handler=adb_tap_position,
    check_fn=_adb_available,
)

registry.register(
    name="adb_tap_smart",
    description=(
        "[推荐] 当页面有多个同名按钮时使用。通过指定目标按钮文字 + 同行任务描述文字来精确定位。\n"
        "解决 adb_tap 用 nth 参数在一个有5个'前往'按钮的页面上点错按钮的问题。\n"
        "示例：adb_tap_smart(target='前往', row_text='在时尚对决中完成1次挑战') — 在包含'在时尚对决中完成1次挑战'的行中找到'前往'按钮并点击。\n"
        "示例：adb_tap_smart(target='领取', row_text='完成每日签到') — 在包含'完成每日签到'的行中找到'领取'按钮。\n"
        "不要使用：只有一个同名按钮 → 直接用 adb_tap。目标按钮不在任何任务描述行上 → 用 adb_tap_position。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "要点击的按钮文字，如'前往'、'领取'、'挑战'。"},
            "row_text": {"type": "string", "description": "目标按钮所在行的任务描述文字，用于区分多个同名按钮。选该行中最独特的部分文字（如'在时尚对决中完成1次挑战'而非仅仅'挑战'）。"},
            "nth": {"type": "integer", "description": "可选。当 row_text 也出现多次时，指定第几个匹配行（从上到下）。默认=0自动选最长匹配。"},
        },
        "required": ["target"],
    },
    handler=adb_tap_smart,
    check_fn=_adb_available,
)

registry.register(
    name="dismiss_all_popups",
    description=(
        "[推荐用于弹窗风暴] 自动连续按返回键关闭所有弹窗，直到画面稳定。\n"
        "使用时机：进入游戏后面对多个叠加弹窗（活动公告、充值活动、礼包推广等），"
        "不需要每个弹窗单独调用 adb_back。一次调用处理所有弹窗。\n"
        "工作原理：反复按返回键，用 dHash 检测每个弹窗是否已关闭（画面变化 > 阈值 → 弹窗关闭 → 继续下一个）。\n"
        "连续3次画面无变化 → 自动停止（已到底层UI或所有弹窗已清除）。\n"
        "安全上限：最多关闭10个弹窗。\n"
        "返回：关闭了多少个弹窗。\n"
        "不要使用：只有一个已知弹窗 → 直接用 adb_back。弹窗需要特殊交互而非返回键关闭 → 用其他方法。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "max_dismissals": {"type": "integer", "description": "最多关闭几个弹窗（默认10，安全上限）。"},
        },
    },
    handler=dismiss_all_popups,
    check_fn=_adb_available,
)

registry.register(
    name="adb_back",
    description="Press Android back button to return to previous screen.",
    parameters={"type": "object", "properties": {}},
    handler=adb_back,
    check_fn=_adb_available,
)
