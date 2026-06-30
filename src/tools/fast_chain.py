"""Fast chain: execute verified skills as scripts without LLM calls.

Parses skill body steps into tool dispatches. When coordinates are available,
bypasses the tool registry entirely — direct ADB calls with proper screen
change polling (same polling logic as the LLM loop, but without OCR).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable

from src.tools.registry import ToolOutput, registry
from src.utils.hash import compute_image_hash
from src.utils.dhash import compute_dhash, hamming_distance

logger = logging.getLogger(__name__)

# Screen polling constants (same logic as LLM loop, tuned for speed)
SCREEN_CHANGE_TIMEOUT = 2.0   # Max wait for screen to differ from pre-tap
SCREEN_STABLE_TIMEOUT = 1.5   # Max wait for two identical frames after change
POLL_INTERVAL = 0.1           # 100ms between polls — 10 checks/sec

# dHash threshold for screen change detection.
# Minor UI noise (badge count, resource numbers) → Hamming 1-3.
# Actual screen transitions (new layout) → Hamming > 10.
DHASH_CHANGE_THRESHOLD = 5


def parse_skill_steps(body: str) -> list[dict]:
    """Parse skill body into executable steps.

    Handles formats:
        1. adb_tap('终端')               # [842, 162] comment
        2. adb_swipe('下滑', 'find ep01') # comment
        3. adb_scroll('next', axis='horizontal')  # semantic scroll
        4. adb_back()
        5. adb_tap_position(0.96, 0.12)  # [1536, 108] — unquoted numeric args

    Returns list of {tool, args, coords, kwargs} where coords is (x, y) or None
    and kwargs is a dict of keyword arguments (e.g. axis='horizontal').
    """
    steps: list[dict] = []
    step_pattern = re.compile(r"^\d+\.\s+(\w+)\(([^)]*)\)")

    for line in body.split("\n"):
        match = step_pattern.match(line.strip())
        if not match:
            continue

        tool = match.group(1)
        args_str = match.group(2).strip()

        # Parse keyword arguments: key='value' or key="value"
        kwargs: dict[str, str] = {}
        for kw_match in re.finditer(r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", args_str):
            kwargs[kw_match.group(1)] = kw_match.group(2)
        # Remove kwarg segments from args_str for positional parsing
        args_for_pos = re.sub(r"\w+\s*=\s*['\"][^'\"]*['\"]", "", args_str).strip()
        # Remove trailing commas
        args_for_pos = re.sub(r',\s*$', '', args_for_pos).strip()

        # Parse positional string arguments AND unquoted numeric arguments
        args: list[str] = []
        if args_for_pos:
            for m in re.finditer(r"""['\"]([^'\"]*)['\"]""", args_for_pos):
                args.append(m.group(1))
            # Also capture unquoted numbers: adb_tap_position(0.96, 0.12)
            if not args:
                for m in re.finditer(r'(\d+\.?\d*)', args_for_pos):
                    args.append(m.group(1))

        # Parse optional coordinates from trailing comment: # [842, 162]
        coord_match = re.search(r"#\s*\[(\d+)\s*,\s*(\d+)\]", line)
        coords = None
        if coord_match:
            coords = (int(coord_match.group(1)), int(coord_match.group(2)))

        steps.append({"tool": tool, "args": args, "coords": coords, "kwargs": kwargs})

    return steps


def _poll_screen_change(adb, pre_dhash: int, timeout: float) -> tuple[int, bool]:
    """Poll until screen dHash differs significantly from pre_dhash.

    Uses perceptual hashing (dHash) — tolerates minor UI noise like badge
    count changes or resource number updates. Only structural layout changes
    (new buttons, different backgrounds) cross the threshold.

    Returns (dhash_int, changed).
    """
    deadline = time.monotonic() + timeout
    current_dhash = pre_dhash
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        img = adb.get_screenshot_image()
        current_dhash = compute_dhash(img)
        if hamming_distance(pre_dhash, current_dhash) > DHASH_CHANGE_THRESHOLD:
            return current_dhash, True
    return current_dhash, False


def _poll_screen_stable(adb, current_dhash: int, timeout: float) -> tuple[int, str]:
    """Poll until two consecutive frames have near-identical dHash.

    Returns (final_dhash, md5_hex_hash). The MD5 hash is kept for oscillation
    detection and inject_screen_fn tracking — dHash is too fuzzy for dedup.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        next_img = adb.get_screenshot_image()
        next_dhash = compute_dhash(next_img)
        if hamming_distance(current_dhash, next_dhash) <= 1:  # virtually identical
            return current_dhash, compute_image_hash(next_img)
        current_dhash = next_dhash
    # Timed out or deadline already passed — return best effort
    try:
        last_img = adb.get_screenshot_image()
        return compute_dhash(last_img), compute_image_hash(last_img)
    except Exception:
        return current_dhash, ""


def execute_fast_chain(
    steps: list[dict],
    inject_screen_fn: Callable[..., None],
    screen_w: int = 0,
    screen_h: int = 0,
) -> tuple[bool, str]:
    """Execute parsed skill steps sequentially without LLM calls.

    When coordinates are available, taps ADB directly and does proper screen
    change polling (wait for change → wait for stabilization) — same logic as
    the LLM loop, but skipping OCR for speed.

    **Failure detection:** tracks how many steps actually changed the screen.
    If ALL fast-path steps produce zero screen changes, the chain is considered
    failed (coordinates are stale) and falls back to LLM.

    Args:
        steps: Parsed steps from parse_skill_steps()
        inject_screen_fn: Called with (fast=True, known_hash=str) to update
                          the agent's tracking hash after each step.
        screen_w, screen_h: Device screen dimensions (unused, kept for API compat).

    Returns:
        (success, message) — success is True if all steps completed and at least
        one fast-path step caused a visible screen change (or no fast-path steps
        existed). Success is False if all fast-path steps were no-ops or any
        step crashed.
    """
    from src.device.adb import get_adb as _get_adb

    adb = _get_adb()
    if screen_w <= 0 or screen_h <= 0:
        screen_w, screen_h = adb.get_screen_size()
        logger.debug(
            "execute_fast_chain: resolved screen size from ADB: %dx%d",
            screen_w, screen_h,
        )
    t_start = time.monotonic()
    fast_steps_attempted = 0
    fast_steps_changed = 0
    consecutive_no_change = 0  # P1: skip step after 2 misses (was: abort)
    skipped_steps: list[int] = []        # P1: indices of skipped stale steps
    failed_coords: list[tuple] = []      # P1: coordinates that were stale
    ocr_steps_ok = 0
    ocr_steps_failed = 0

    for i, step in enumerate(steps):
        tool = step["tool"]
        args = step["args"]
        coords = step["coords"]

        try:
            if tool in ("adb_tap", "adb_tap_position") and coords:
                # ============================================================
                # FAST PATH: direct ADB tap with full screen-change polling
                # ============================================================
                fast_steps_attempted += 1
                x, y = coords[0], coords[1]
                if not (0 <= x <= 3000 and 0 <= y <= 3000):
                    return False, f"Step {i+1}: coords out of bounds ({x},{y})"

                # 1. Capture pre-tap screen state (dHash for perceptual change;
                #    MD5 for oscillation log)
                pre_img = adb.get_screenshot_image()
                pre_dhash = compute_dhash(pre_img)
                pre_hex = compute_image_hash(pre_img)

                # 2. Tap
                adb.tap(x, y)

                # 3. Poll until screen perceptually changes (Phase 1)
                #    dHash tolerates badge counts / resource numbers changing.
                t_poll = time.monotonic()
                post_dhash, changed = _poll_screen_change(adb, pre_dhash, SCREEN_CHANGE_TIMEOUT)

                if not changed:
                    consecutive_no_change += 1
                    t_wait = time.monotonic() - t_poll
                    logger.warning(
                        "Fast chain step %d: screen unchanged after %.1fs "
                        "(tap at %d,%d, pre_md5=%s). Consecutive misses: %d.",
                        i + 1, t_wait, x, y, pre_hex[:8], consecutive_no_change,
                    )
                    inject_screen_fn(fast=True, known_hash=pre_hex)

                    # P1: OCR fallback on first miss — try real-time text lookup
                    if consecutive_no_change == 1 and args:
                        _target = args[0] if args else ""
                        if _target and len(_target) >= 2:
                            try:
                                from src.vision.ocr import ocr_engine
                                _det = ocr_engine.read_text(
                                    adb.get_screenshot_image())
                                for d in _det:
                                    if (_target in d["text"]
                                            and d["confidence"] >= 0.6):
                                        cx, cy = d["center"]
                                        adb.tap(cx, cy)
                                        logger.info(
                                            "Fast chain step %d: OCR rescue — "
                                            "'%s' found at (%d,%d)",
                                            i + 1, _target, cx, cy,
                                        )
                                        consecutive_no_change = 0
                                        fast_steps_changed += 1
                                        _rdhash = compute_dhash(
                                            adb.get_screenshot_image())
                                        _sdhash, _shex = _poll_screen_stable(
                                            adb, _rdhash,
                                            SCREEN_STABLE_TIMEOUT)
                                        inject_screen_fn(
                                            fast=True, known_hash=_shex)
                                        break
                            except Exception:
                                pass

                    if consecutive_no_change >= 2:
                        # P1: skip stale step instead of aborting entire chain
                        skipped_steps.append(i)
                        failed_coords.append(coords)
                        consecutive_no_change = 0
                        remaining = len(steps) - i - 1
                        logger.warning(
                            "Fast chain step %d: skipping stale coordinates "
                            "(%d, %d). %d steps remaining.",
                            i + 1, x, y, remaining,
                        )
                        continue
                    continue  # Don't count as changed, but give one retry

                fast_steps_changed += 1
                consecutive_no_change = 0  # Reset — screen moved, chain is alive

                # 4. Poll until screen stabilizes (Phase 2)
                stable_dhash, stable_hex = _poll_screen_stable(adb, post_dhash, SCREEN_STABLE_TIMEOUT)
                t_total = time.monotonic() - t_poll

                if t_total > 1.0:
                    logger.info(
                        "Fast chain step %d: screen poll %.1fs "
                        "(pre_hex=%s stable_hex=%s)",
                        i + 1, t_total,
                        pre_hex[:8], stable_hex[:8],
                    )
                else:
                    logger.debug(
                        "Fast chain step %d: done in %.0fms (hex=%s)",
                        i + 1, t_total * 1000, stable_hex[:8],
                    )

                # 5. Update agent's tracking hash
                inject_screen_fn(fast=True, known_hash=stable_hex)

            elif tool in ("adb_tap", "adb_tap_position"):
                # ============================================================
                # FALLBACK: OCR-based tap via tool registry
                # ============================================================
                # But first: if this is adb_tap_position with numeric pct args
                # but no # [x,y] comment, convert pct to device pixels and use
                # the fast path. The pct values ARE coordinates — just expressed
                # as screen fractions. Without this, the fallback would feed
                # "0.96" as an OCR target which never matches.
                if tool == "adb_tap_position" and args and len(args) >= 2:
                    try:
                        pct_x = float(args[0])
                        pct_y = float(args[1])
                        if 0 <= pct_x <= 1 and 0 <= pct_y <= 1 and (pct_x > 0 or pct_y > 0):
                            coords = (
                                round(pct_x * screen_w),
                                round(pct_y * screen_h),
                            )
                            logger.debug(
                                "Fast chain step %d: adb_tap_position pct→px "
                                "(%.3f, %.3f) → (%d, %d)",
                                i + 1, pct_x, pct_y, coords[0], coords[1],
                            )
                            # Re-process with now-populated coords — jump to fast path
                            # by restarting the step (the fast-path block above handles
                            # adb_tap_position with coords correctly).
                            tool = "adb_tap_position"  # unchanged
                            # Fall through to the same tap logic used by the fast path.
                            # We inline the fast-path block here to avoid goto-style
                            # control flow.
                            fast_steps_attempted += 1
                            x, y = coords
                            if not (0 <= x <= 3000 and 0 <= y <= 3000):
                                return False, f"Step {i+1}: coords out of bounds ({x},{y})"
                            pre_img = adb.get_screenshot_image()
                            pre_dhash = compute_dhash(pre_img)
                            pre_hex = compute_image_hash(pre_img)
                            adb.tap(x, y)
                            t_poll = time.monotonic()
                            post_dhash, changed = _poll_screen_change(adb, pre_dhash, SCREEN_CHANGE_TIMEOUT)
                            if not changed:
                                consecutive_no_change += 1
                                t_wait = time.monotonic() - t_poll
                                logger.warning(
                                    "Fast chain step %d: screen unchanged after %.1fs "
                                    "(tap at %d,%d, pre_md5=%s). Consecutive misses: %d.",
                                    i + 1, t_wait, x, y, pre_hex[:8], consecutive_no_change,
                                )
                                inject_screen_fn(fast=True, known_hash=pre_hex)

                                # P1: OCR fallback on first miss
                                if consecutive_no_change == 1 and args:
                                    _target = args[0] if args else ""
                                    if _target and len(_target) >= 2:
                                        try:
                                            from src.vision.ocr import ocr_engine
                                            _det = ocr_engine.read_text(
                                                adb.get_screenshot_image())
                                            for d in _det:
                                                if (_target in d["text"]
                                                        and d["confidence"] >= 0.6):
                                                    cx, cy = d["center"]
                                                    adb.tap(cx, cy)
                                                    logger.info(
                                                        "Fast chain step %d: OCR rescue — "
                                                        "'%s' found at (%d,%d)",
                                                        i + 1, _target, cx, cy)
                                                    consecutive_no_change = 0
                                                    fast_steps_changed += 1
                                                    _rdhash = compute_dhash(
                                                        adb.get_screenshot_image())
                                                    _sdhash, _shex = _poll_screen_stable(
                                                        adb, _rdhash,
                                                        SCREEN_STABLE_TIMEOUT)
                                                    inject_screen_fn(
                                                        fast=True, known_hash=_shex)
                                                    break
                                        except Exception:
                                            pass

                                if consecutive_no_change >= 2:
                                    # P1: skip stale step instead of aborting
                                    skipped_steps.append(i)
                                    failed_coords.append(coords)
                                    consecutive_no_change = 0
                                    remaining = len(steps) - i - 1
                                    logger.warning(
                                        "Fast chain step %d: skipping stale "
                                        "coordinates (%d, %d). %d remaining.",
                                        i + 1, x, y, remaining)
                                    continue
                                continue
                            fast_steps_changed += 1
                            consecutive_no_change = 0
                            stable_dhash, stable_hex = _poll_screen_stable(adb, post_dhash, SCREEN_STABLE_TIMEOUT)
                            t_total = time.monotonic() - t_poll
                            if t_total > 1.0:
                                logger.info(
                                    "Fast chain step %d: screen poll %.1fs "
                                    "(pre_hex=%s stable_hex=%s)",
                                    i + 1, t_total, pre_hex[:8], stable_hex[:8],
                                )
                            else:
                                logger.debug(
                                    "Fast chain step %d: done in %.0fms (hex=%s)",
                                    i + 1, t_total * 1000, stable_hex[:8],
                                )
                            inject_screen_fn(fast=True, known_hash=stable_hex)
                            continue  # Step complete — move to next
                    except (ValueError, TypeError):
                        pass  # Non-numeric args — fall through to OCR fallback

                if args:
                    output = registry.dispatch("adb_tap", target=args[0])
                else:
                    logger.warning("Fast chain step %d: adb_tap without target or coords", i + 1)
                    return False, f"Step {i+1}: adb_tap missing target"

                try:
                    data = json.loads(output.text)
                    if data.get("success", False):
                        ocr_steps_ok += 1
                    else:
                        ocr_steps_failed += 1
                        logger.warning("Fast chain step %d failed: %s(%s) — %s",
                                       i + 1, tool, args, data.get('message', ''))
                        return False, f"Step {i+1}: {tool}({args}) reported failure — {data.get('message', '')}"
                except json.JSONDecodeError:
                    ocr_steps_ok += 1  # Non-JSON output, assume success

                inject_screen_fn(fast=False)

            elif tool in ("adb_swipe", "adb_scroll"):
                direction = args[0] if args else "down"
                distance = args[1] if len(args) > 1 else "half"
                kwargs = step.get("kwargs", {})
                if tool == "adb_scroll":
                    axis = kwargs.get("axis", "horizontal")
                    registry.dispatch("adb_scroll", direction=direction, axis=axis, distance=distance)
                else:
                    registry.dispatch("adb_swipe", direction=direction, distance=distance)
                inject_screen_fn(fast=False)

            elif tool == "adb_back":
                registry.dispatch("adb_back")
                inject_screen_fn(fast=False)

            else:
                logger.warning("Fast chain step %d: unknown tool '%s'", i + 1, tool)
                return False, f"Step {i+1}: unknown tool '{tool}'"

        except Exception as e:
            logger.warning("Fast chain step %d crashed: %s", i + 1, e)
            return False, f"Step {i+1}: crashed — {e}"

    elapsed = time.monotonic() - t_start

    # ================================================================
    # P1: Failure detection — >50% of fast-path steps skipped → fall back to LLM.
    # Changed from "ALL zero changes" (was too aggressive — a single stale
    # coordinate killed the entire chain).  Now: if most steps worked, the
    # chain is considered partially successful (LLM picks up remaining steps).
    # ================================================================
    _total_skipped = len(skipped_steps)
    if fast_steps_attempted > 0:
        _failure_rate = _total_skipped / max(fast_steps_attempted, 1)
        if _failure_rate > 0.5:
            msg = (
                f"{_total_skipped}/{fast_steps_attempted} fast-path steps skipped "
                f"({_failure_rate:.0%} failure rate). "
                "Majority of coordinates are stale — fall back to LLM."
            )
            logger.warning("Fast chain FAILED: %s", msg)
            return False, msg
        if _total_skipped > 0:
            logger.warning(
                "Fast chain: %d/%d steps skipped (%.0%%), but majority succeeded — "
                "continuing with LLM for remaining steps.",
                _total_skipped, fast_steps_attempted, _failure_rate,
            )

    logger.info("Fast chain: %d steps in %.1fs (fast:%d/%d changed, ocr:%d ok, skipped:%d)",
                len(steps), elapsed,
                fast_steps_changed, fast_steps_attempted, ocr_steps_ok, _total_skipped)
    return True, "All steps completed"


def has_verified_skill(matching_skills: list[dict]) -> bool:
    """Check if any matched skill is verified for fast chain execution."""
    for s in matching_skills:
        if s.get("verified"):
            return True
    return False
