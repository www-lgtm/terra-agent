"""IdleWatcher — dHash-based idle screen watcher.

Replaces wasteful LLM calls during wait periods (loading screens, auto-battle,
transitions).  Uses perceptual dHash + Hamming distance to detect genuine
screen transitions while ignoring animation noise from Live2D backgrounds,
character idle loops, and particle effects.

Why dHash instead of MD5:
  MD5 changes on every frame when a Live2D character breathes — one byte
  difference in the pixel array produces a completely different hash.
  dHash (9×8 → 64-bit) compares relative brightness of adjacent pixels at
  thumbnail scale.  A Live2D idle animation shifts pixels by a fraction
  of a thumbnail cell — invisible to dHash.  Hamming distance ≤ 10
  reliably means "same scene, just animation noise."

Design: zero keywords, zero OCR, zero LLM calls.  Pure dHash polling.

Scenario guide:
  Static screen (menu, settings)    → dHash stays ≤10 → gentle exit 15s
  Loading → game UI                 → dHash jumps >12 → stable? → exit ✓
  Live2D menu (animated background) → dHash stays ≤10 → watcher waits ✓
  Auto-battle                       → dHash stays ≤10 → waits for end screen
  Battle ends → settlement          → dHash jumps >12 → new stable → exit ✓
  User interrupt                    → instant exit

Safety nets:
  - Gentle exit: 3 polls (~3s) with distance ≤10 from start → LLM fallback
  - Max polls (10 ≈ 10s): LLM fallback for any scenario
  - Max total time (40s): absolute ceiling
  - User interrupt: Event.wait() wakes instantly
  - Cancellation: checks _pending_cancel each poll

Replaces the abandoned BattlePoller.  History:
  - BattlePoller: keyword scoring ("作战中" +3.0) → false positives on
    settings/overview screens containing those words → removed.
  - IdleWatcher v1: MD5 exact match → "一辈子醒不来" on dynamic scenes → removed.
  - IdleWatcher v2 (current): dHash + Hamming distance → handles animation noise.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.utils.dhash import compute_dhash, hamming_distance, dhash_to_hex

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────

POLL_INTERVAL = 0.8           # seconds between polls (was 1.0)
SAME_SCENE_MAX = 10           # Hamming distance ≤ this → same scene (animation ok)
TRANSITION_MIN = 13           # Hamming distance ≥ this → screen has moved on
STABILITY_POLLS = 3           # consecutive polls with pairwise distance ≤6 → stable
STABILITY_DISTANCE = 6        # max pairwise distance for consecutive frames to be "stable"
GENTLE_EXIT_POLLS = 3         # polls where distance stays ≤SAME_SCENE_MAX → fallback (~3s)
MAX_POLLS = 10                # safety: force LLM fallback after ~10s total (was 20)
MAX_TOTAL_TIME = 20.0         # absolute ceiling in seconds (was 40.0)
HEARTBEAT_INTERVAL = 8        # polls between heartbeat notifications


class IdleWatcher:
    """Per-task dHash-based watcher.  One instance per agent task.

    Usage (in loop.py, the no-tool-calls branch):

        pre_dhash = self.state.last_injected_dhash
        if self.loop_guard.idle_streak >= 2 and pre_dhash:
            changed, new_dhash = self.idle_watcher.watch(pre_dhash)
            if changed:
                self.screen_injector.inject_now()
                continue
        self.screen_injector.inject_now()
    """

    def __init__(self, state: Any, agent: Any = None) -> None:
        from src.agent.state import AgentState
        self.state: AgentState = state
        self._agent = agent  # optional TerraAgent ref for heartbeat

    # ── Public API ──────────────────────────────────────────────────

    def watch(self, pre_dhash_hex: str) -> tuple[bool, str | None]:
        """Watch screen until it perceptually changes and stabilizes.

        Args:
            pre_dhash_hex: dHash hex string (from state.last_injected_dhash)
                           of the screen when we entered idle watching.

        Returns:
            (progress_made, new_dhash_hex_or_none).
            - (True, new_dhash): Screen changed → settled → ready for LLM.
            - (False, None): Fallback (timeout / gentle exit / interrupt).
        """
        from src.utils.dhash import hex_to_dhash
        from src.device.adb import get_adb

        try:
            pre_dhash = hex_to_dhash(pre_dhash_hex)
        except (ValueError, TypeError):
            logger.warning("IdleWatcher: invalid pre_dhash_hex=%s", pre_dhash_hex)
            return False, None

        try:
            adb = get_adb()
        except Exception:
            logger.warning("IdleWatcher: ADB unavailable")
            return False, None

        t0 = time.monotonic()
        poll_count = 0
        stable_count = 0
        transition_seen = False          # dHash jumped > TRANSITION_MIN from start
        transition_dhash: int | None = None  # what hash we transitioned TO
        last_heartbeat = 0.0

        logger.info(
            "IdleWatcher: pre_dhash=%s, poll=%.1fs, same≤%d, transition≥%d, "
            "stable=%d×≤%d, gentle=%d, max=%d, ceiling=%.0fs",
            pre_dhash_hex[:8], POLL_INTERVAL, SAME_SCENE_MAX, TRANSITION_MIN,
            STABILITY_POLLS, STABILITY_DISTANCE, GENTLE_EXIT_POLLS,
            MAX_POLLS, MAX_TOTAL_TIME,
        )

        while self.state.running:
            # ── Safety ceilings ──
            elapsed = time.monotonic() - t0
            if poll_count >= MAX_POLLS:
                logger.info(
                    "IdleWatcher: max polls (%d) after %.0fs — LLM fallback",
                    MAX_POLLS, elapsed,
                )
                return False, None
            if elapsed >= MAX_TOTAL_TIME:
                logger.info("IdleWatcher: max time (%.0fs) reached", MAX_TOTAL_TIME)
                return False, None

            # ── Event-based sleep ──
            self.state._interrupt_event.wait(timeout=POLL_INTERVAL)
            self.state._interrupt_event.clear()
            poll_count += 1

            # ── User interrupt ──
            intr = self.state.pop_interrupt()
            if intr:
                logger.info("IdleWatcher: user interrupt at poll #%d — '%s'",
                           poll_count, intr[:80])
                self.state.add_message("user", f"[用户指令 — 必须执行] {intr}")
                return False, None

            # ── Cancellation ──
            if self.state._pending_cancel and self.state.interrupt_zone == "safe":
                logger.info("IdleWatcher: cancel at poll #%d", poll_count)
                return False, None

            # ── Capture + dHash ──
            try:
                screenshot = adb.get_screenshot_image()
                cur_dhash = compute_dhash(screenshot)
            except Exception:
                logger.warning("IdleWatcher: screenshot failed at poll #%d", poll_count)
                continue

            dist_from_start = hamming_distance(cur_dhash, pre_dhash)

            # ── Not transitioned yet: waiting for a real screen change ──
            if not transition_seen:
                if dist_from_start >= TRANSITION_MIN:
                    # Genuine transition detected
                    transition_seen = True
                    transition_dhash = cur_dhash
                    stable_count = 1
                    logger.info(
                        "IdleWatcher: transition at poll #%d / %.1fs "
                        "(pre=%s → cur=%s, dist=%d)",
                        poll_count, elapsed,
                        pre_dhash_hex[:8], dhash_to_hex(cur_dhash)[:8],
                        dist_from_start,
                    )
                    continue  # Next poll checks stability

                # Still "same scene" — keep waiting
                if poll_count >= GENTLE_EXIT_POLLS:
                    logger.info(
                        "IdleWatcher: gentle exit after %d polls / %.0fs "
                        "(scene unchanged, dist=%d ≤ %d)",
                        poll_count, elapsed, dist_from_start, SAME_SCENE_MAX,
                    )
                    return False, None

                # ── Dark-screen early-exit ──
                # If no transition has been seen and the screen is persistently
                # near-identical to the start (dist ≤ 3 for 3+ consecutive polls),
                # this is likely a static dark/loading screen. Exit early.
                if poll_count >= 3 and dist_from_start <= 3:
                    logger.info(
                        "IdleWatcher: dark-screen early exit after %d polls / %.0fs "
                        "(static screen, dist=%d ≤ 3)",
                        poll_count, elapsed, dist_from_start,
                    )
                    return False, None

                # Heartbeat
                if poll_count % HEARTBEAT_INTERVAL == 0 and self._agent is not None:
                    now = time.monotonic()
                    if now - last_heartbeat > 60.0:
                        self._agent._heartbeat(
                            f"等待画面变化中（已等 ~{int(elapsed)}s）"
                        )
                        last_heartbeat = now

                continue  # Still waiting for transition

            # ── Post-transition: waiting for stability ──
            pair_dist = hamming_distance(cur_dhash, transition_dhash)
            if pair_dist <= STABILITY_DISTANCE:
                stable_count += 1
            else:
                stable_count = 1
                transition_dhash = cur_dhash  # Reset anchor to latest frame

            if stable_count >= STABILITY_POLLS:
                new_hex = dhash_to_hex(transition_dhash)
                logger.info(
                    "IdleWatcher: stable new screen at poll #%d / %.1fs "
                    "(pre=%s → stable=%s, dist=%d, %d consecutive)",
                    poll_count, elapsed,
                    pre_dhash_hex[:8], new_hex[:8],
                    hamming_distance(transition_dhash, pre_dhash),
                    stable_count,
                )
                return True, new_hex

            # Periodic log
            if poll_count % 15 == 0:
                logger.debug(
                    "IdleWatcher: poll #%d / %.0fs, "
                    "transitioned d=%d, stable=%d/%d (pair_dist=%d)",
                    poll_count, elapsed,
                    hamming_distance(cur_dhash, pre_dhash),
                    stable_count, STABILITY_POLLS, pair_dist,
                )

        # state.running became False
        logger.info("IdleWatcher: agent stopped after %d polls", poll_count)
        return False, None
