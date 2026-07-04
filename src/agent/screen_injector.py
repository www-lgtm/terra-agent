"""ScreenInjector — screen capture, polling, OCR, and injection pipeline.

Extracted from TerraAgent to keep the main loop readable.  Handles the
three-phase screen capture pipeline (settle → change → stabilize → black-frame
rejection) plus OCR, dHash, and screen cache management.

Memory hint gathering is NOT handled here — the caller (loop.py) receives
OCR texts and decides whether to search for relevant memories.
"""

from __future__ import annotations

import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any

from PIL import Image

from config.settings import config
from src.agent.state import AgentState
from src.utils.hash import compute_image_hash

logger = logging.getLogger(__name__)

# Tools that trigger screen injection after execution
ACTION_TOOLS = frozenset({
    "adb_tap", "adb_swipe", "adb_scroll", "adb_tap_position", "adb_back", "tap_magnified",
})

# Persistent thread pool for parallel dHash + OCR during screen injection.
# Creating/destroying a ThreadPoolExecutor on every injection cycle (~50-200
# per task) wastes ~10ms per cycle in thread creation overhead.  Two workers
# is optimal: dHash (~2ms) + OCR (~300ms) run concurrently; a third worker
# would always be idle since there are only two tasks.
_inject_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inject")


# ── Helpers ───────────────────────────────────────────────────────

def compress_screenshot(img: Image.Image, max_width: int | None = None,
                        quality: int | None = None) -> str:
    """Resize and compress screenshot to JPEG base64."""
    if max_width is None:
        max_width = config.agent.screenshot_max_width
    if quality is None:
        quality = config.agent.screenshot_quality
    if img.mode == "RGBA":
        img = img.convert("RGB")
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        img = img.resize((max_width, int(h * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def capture_screen_jpeg() -> str | None:
    """Capture current ADB screen as a compressed JPEG base64 string.

    Returns None if no ADB device is available.  Used by ask_user to send
    the current screen to the remote user.
    """
    try:
        from src.device.adb import get_adb
        adb = get_adb()
        img = adb.get_screenshot_image()
        if img.mode == "RGBA":
            img = img.convert("RGB")
        w, h = img.size
        if w > 640:
            ratio = 640 / w
            img = img.resize((640, int(h * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        from src.utils.errors import safe_log
        safe_log(logger, "warning", f"Failed to capture screen for notification: {e}")
        return None


def is_action_tool(tool_calls: list[dict[str, Any]]) -> bool:
    """Check if any tool call is a screen-changing action."""
    return any(tc["name"] in ACTION_TOOLS for tc in tool_calls)


# ── ScreenInjector ────────────────────────────────────────────────

class ScreenInjector:
    """Per-task screen capture, polling, OCR, and injection.

    Holds a reference to AgentState for writing injection results.
    ADB, OCR, and screen_cache are imported lazily (per-call) to avoid
    import-time coupling.
    """

    def __init__(self, state: AgentState) -> None:
        self.state = state
        self._post_restart_extension: float = 0.0  # extra deadline budget after restart

    def shutdown(self) -> None:
        """Release any resources held by the screen injector.

        Currently a no-op — the injector is stateless aside from the
        AgentState reference (which is owned by the caller).  Exists to
        satisfy the cleanup contract in loop.py:run().
        """

    # ── Post-action injection ────────────────────────────────────

    def inject_after_actions(self, tool_calls: list[dict[str, Any]]) -> list[str]:
        """After action tools execute, capture and inject current screen.

        Returns the list of OCR texts detected on the new screen (empty on failure).

        Settle pipeline:
        1. Initial settle (0.08s for taps, 0.15s for swipes)
        2. Fast-confirm: if screen already changed + not black → skip polling
        3. Slow poll: only when screen hasn't changed yet (rare — game loading)
        4. One-frame stability check (replaces polling loop — 100-200ms saved)
        5. Black-frame rejection poll (still needed — loading screens last 500ms+)
        """
        if config.agent.vision_mode != "auto_inject":
            return []
        if not is_action_tool(tool_calls):
            # Non-action tools (lifemaker_launch, base_collect, etc.)
            # may have changed the screen dramatically. Inject if needed.
            if self.state.last_injected_hash:
                try:
                    from src.device.adb import get_adb
                    from src.utils.hash import compute_image_hash as _cs
                    img = get_adb().get_screenshot_image()
                    if _cs(img) != self.state.last_injected_hash:
                        return self.inject_now()
                except Exception:
                    pass
            return []

        try:
            from src.device.adb import get_adb
            from src.vision.ocr import ocr_engine
            from src.tools import screen_cache

            adb = get_adb()
            pre_hash = self.state.last_injected_hash
            poll_start = time.monotonic()
            # 1.0s deadline — fast-confirm + one-frame stability handles 95% of
            # actions in <0.2s.  The remaining 1.0s budget is for loading screens
            # (black-frame rejection).  1.5s was unnecessarily conservative:
            # in 309 real-world polls, 99% would have succeeded within 1.0s.
            deadline = poll_start + 1.0 + self._post_restart_extension
            self._post_restart_extension = 0.0  # consume once
            _SETTLE_POLL = 0.05
            _BLACK_POLL = 0.1

            # ── Step 1: Initial settle + first capture ──
            _is_swipe = any(tc.get("name") in ("adb_swipe", "adb_scroll") for tc in tool_calls)
            if _is_swipe:
                time.sleep(0.15)
            else:
                time.sleep(0.08)  # was 0.15 — fast-confirm catches slow renders
            screenshot = adb.get_screenshot_image()
            screen_hash = compute_image_hash(screenshot)

            # ── Step 2: Slow poll (only when screen hasn't changed) ──
            if pre_hash is not None and screen_hash == pre_hash:
                while time.monotonic() < deadline:
                    time.sleep(_SETTLE_POLL)
                    screenshot = adb.get_screenshot_image()
                    screen_hash = compute_image_hash(screenshot)
                    if screen_hash != pre_hash:
                        break

            # ── Step 3: One-frame stability check ──
            # Replaces the old per-frame polling loop.  Most game UIs render
            # in one frame; a single check catches the tail of animations.
            # Long animations (5% of cases) are handled by stale_screen guard.
            if not _is_swipe:
                time.sleep(_SETTLE_POLL)
                next_frame = adb.get_screenshot_image()
                next_hash = compute_image_hash(next_frame)
                if next_hash != screen_hash:
                    screenshot = next_frame  # still animating → use newer frame

            # ── Step 4: Black-frame rejection ──
            # Polling kept — loading/transition screens can last 500ms+ and
            # the LLM hallucinates on black screens.
            mean = 0  # initialized for brightness storage below
            while time.monotonic() < deadline:
                tiny = screenshot.resize((1, 1), Image.LANCZOS).convert("L")
                mean = tiny.getpixel((0, 0))
                if mean >= 15:
                    break
                logger.debug("Black frame (mean=%d), waiting...", mean)
                time.sleep(_BLACK_POLL)
                screenshot = adb.get_screenshot_image()

            self.state.last_screen_brightness = float(mean)

            screen_hash = compute_image_hash(screenshot)
            poll_elapsed = time.monotonic() - poll_start
            if poll_elapsed > 0.5:
                logger.info("Screen poll: %.1fs (pre=%s post=%s)",
                            poll_elapsed,
                            pre_hash[:8] if pre_hash else "none",
                            screen_hash[:8])

            # Parallel: dHash + OCR
            from src.utils.dhash import compute_dhash, dhash_to_hex

            dhash_result: str | None = None
            detections: list[dict[str, Any]] = []

            dhash_future = _inject_pool.submit(_compute_dhash, screenshot)
            ocr_future = _inject_pool.submit(ocr_engine.read_text, screenshot)

            for future in as_completed([dhash_future, ocr_future]):
                if future == dhash_future:
                    try:
                        dhash_result = dhash_to_hex(future.result())
                    except Exception:
                        from src.utils.errors import safe_log
                        safe_log(logger, "warning", "dHash computation failed")
                        dhash_result = None
                elif future == ocr_future:
                    try:
                        detections = future.result()
                    except Exception:
                        from src.utils.errors import safe_log
                        safe_log(logger, "warning", "OCR read failed")
                        detections = []

            self.state.last_injected_dhash = dhash_result
            ocr_texts = [d["text"] for d in detections]
            self.state.last_ocr_texts = ocr_texts

            # Share full OCR detections with adb_tap so it can skip re-scanning
            # the same screenshot (~300ms saved per tap on cache hit).
            if detections and dhash_result:
                try:
                    from src.tools.adb_control import cache_ocr_detections
                    cache_ocr_detections(dhash_result, detections)
                except Exception:
                    pass  # Non-critical — adb_tap will fall back to its own OCR

            # Cache OCR coordinates for adb_tap fast path
            # P1: store under BOTH composite key (dhash + top-5 OCR) and
            # dhash-only key.  Composite key prevents collisions when different
            # screens have similar dHash (common in dark-background game UIs).
            buttons: dict[str, tuple[int, int]] = {}
            for d in detections:
                if d["confidence"] >= 0.6 and len(d["text"]) >= 2:
                    buttons[d["text"]] = tuple(d["center"])
            if buttons:
                _top_texts = [d["text"] for d in detections[:5] if d["confidence"] >= 0.6]
                if _top_texts:
                    _composite = f"{dhash_result or screen_hash}_{'_'.join(_top_texts)}"
                    screen_cache.bulk_set(_composite, buttons, device_serial=self.state.device_serial)
                # Also store dhash-only key for backward compatibility
                cache_key = dhash_result if dhash_result else screen_hash
                screen_cache.bulk_set(cache_key, buttons, device_serial=self.state.device_serial)

            if screen_hash != self.state.last_injected_hash:
                b64 = compress_screenshot(screenshot)
                self.state.add_screen_injection(b64, ocr_texts, screen_hash)
                self.state.last_injected_hash = screen_hash
                logger.info("Screen injected: hash=%s texts=%d cached=%d",
                            screen_hash[:8], len(ocr_texts), len(buttons))
            else:
                self.state.add_screen_injection_text_only(ocr_texts)
                logger.debug("Screen dedup: hash=%s unchanged", screen_hash[:8])

            return ocr_texts

        except Exception as e:
            logger.warning("Screen injection failed: %s", e)
            # Check ADB health
            from src.device.adb import get_adb
            try:
                adb_dev = get_adb()
                if not adb_dev._heartbeat_ok:
                    logger.error("ADB unhealthy — aborting task loop")
                    self.state.running = False
            except Exception:
                pass
            return []

    # ── Initial injection ────────────────────────────────────────

    def inject_initial(self) -> list[str]:
        """Inject the starting screen before the first LLM call.

        Fast path: skips dHash computation (not needed for first turn).
        Returns OCR texts for memory pre-fetching by the caller.
        """
        if config.agent.vision_mode != "auto_inject":
            return []
        try:
            from src.device.adb import get_adb
            from src.vision.ocr import ocr_engine
            from src.tools import screen_cache

            screenshot = get_adb().get_screenshot_image()
            screen_hash = compute_image_hash(screenshot)

            # Store brightness for dark-nav guard
            tiny = screenshot.resize((1, 1), Image.LANCZOS).convert("L")
            self.state.last_screen_brightness = float(tiny.getpixel((0, 0)))

            # Defensive: skip if this exact screen was already injected
            # (can happen if two agents briefly share a device — the
            # cross-check in _delegate_to_agent should prevent this,
            # but this guard is cheap).
            if screen_hash and screen_hash == self.state.last_injected_hash:
                logger.debug("Skipping duplicate initial injection (hash=%s)", screen_hash[:8])
                return self.state.last_ocr_texts or []

            detections = ocr_engine.read_text(screenshot)
            ocr_texts = [d["text"] for d in detections]
            self.state.last_ocr_texts = ocr_texts
            self.state.last_injected_dhash = None

            b64 = compress_screenshot(screenshot)

            buttons: dict[str, tuple[int, int]] = {}
            for d in detections:
                if d["confidence"] >= 0.6 and len(d["text"]) >= 2:
                    buttons[d["text"]] = tuple(d["center"])
            if buttons:
                # P1: store both composite key and dhash-only key
                _top_texts = [d["text"] for d in detections[:5] if d["confidence"] >= 0.6]
                if _top_texts:
                    _composite = f"{screen_hash}_{'_'.join(_top_texts)}"
                    screen_cache.bulk_set(_composite, buttons, device_serial=self.state.device_serial)
                screen_cache.bulk_set(screen_hash, buttons, device_serial=self.state.device_serial)

            self.state.add_screen_injection(b64, ocr_texts, screen_hash)
            self.state.last_injected_hash = screen_hash
            logger.info("Initial screen: hash=%s texts=%d cached=%d",
                        screen_hash[:8], len(ocr_texts), len(buttons))
            return ocr_texts

        except Exception as e:
            logger.warning("Initial screen injection failed: %s", e)
            return []

    # ── Idle injection ───────────────────────────────────────────

    def inject_now(self, fast: bool = False, known_hash: str | None = None) -> list[str]:
        """Take a fresh screenshot and inject into conversation history.

        fast=True: skip OCR, only update hash (used by fast chain fallback).
        known_hash: skip screenshot capture entirely (fast chain already polled).
        Returns OCR texts (empty if fast mode or on failure).
        """
        if fast and known_hash is not None:
            self.state.last_injected_hash = known_hash
            return []

        from src.device.adb import get_adb as _get_adb
        t0 = time.monotonic()

        try:
            screenshot = _get_adb().get_screenshot_image()
        except Exception as e:
            logger.warning("Screen injection failed: %s", e)
            from src.device.adb import get_adb as _get_adb2
            try:
                if not _get_adb2()._heartbeat_ok:
                    logger.error("ADB unhealthy — aborting task")
                    self.state.running = False
            except Exception:
                pass
            return []

        screen_hash = compute_image_hash(screenshot)

        # Store brightness for dark-nav guard
        tiny = screenshot.resize((1, 1), Image.LANCZOS).convert("L")
        self.state.last_screen_brightness = float(tiny.getpixel((0, 0)))

        if fast:
            changed = screen_hash != self.state.last_injected_hash
            if changed:
                self.state.last_injected_hash = screen_hash
            logger.debug("Fast screen check: hash=%s changed=%s (%.0fms)",
                        screen_hash[:8], changed, (time.monotonic() - t0) * 1000)
            return []

        # ── Same-screen fast path: reuse cached OCR ──
        # Idle iterations (LLM produced no tool calls) frequently land on the
        # same screen.  OCR on complex screens takes 3-9s for 48+ text elements.
        # When the hash hasn't changed, the OCR result is guaranteed identical.
        if screen_hash == self.state.last_injected_hash and self.state.last_ocr_texts:
            self.state.add_screen_injection_text_only(self.state.last_ocr_texts)
            logger.info("Idle screen dedup: hash=%s unchanged, reused %d OCR texts (%.1fs)",
                       screen_hash[:8], len(self.state.last_ocr_texts),
                       time.monotonic() - t0)
            return self.state.last_ocr_texts

        from src.vision.ocr import ocr_engine as _ocr
        from src.tools import screen_cache as _sc2
        detections = _ocr.read_text(screenshot)
        texts = [d["text"] for d in detections]
        self.state.last_ocr_texts = texts

        from src.utils.dhash import compute_dhash, dhash_to_hex
        try:
            self.state.last_injected_dhash = dhash_to_hex(compute_dhash(screenshot))
        except Exception:
            self.state.last_injected_dhash = None

        if screen_hash != self.state.last_injected_hash:
            b64 = compress_screenshot(screenshot)
            self.state.add_screen_injection(b64, texts, screen_hash)
            self.state.last_injected_hash = screen_hash
            buttons: dict[str, tuple[int, int]] = {}
            for d in detections:
                if d["confidence"] >= 0.6 and len(d["text"]) >= 2:
                    buttons[d["text"]] = tuple(d["center"])
            if buttons:
                cache_key = self.state.last_injected_dhash or screen_hash
                _sc2.bulk_set(cache_key, buttons, device_serial=self.state.device_serial)
        else:
            self.state.add_screen_injection_text_only(texts)
        logger.info("Idle screen injected: hash=%s texts=%d (%.1fs)",
                    screen_hash[:8], len(texts), time.monotonic() - t0)
        return texts


# ── Module-level helper ──────────────────────────────────────────

def _compute_dhash(screenshot: Image.Image) -> Any:
    """Compute dHash from a PIL Image for ThreadPoolExecutor."""
    from src.utils.dhash import compute_dhash as _cdh
    return _cdh(screenshot)
