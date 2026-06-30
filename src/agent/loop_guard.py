"""LoopGuard — per-task loop protection for TerraAgent.

Extracted from TerraAgent._run_loop() to keep the main loop readable and
to make each guard independently testable.  All guard state is isolated here;
the main loop only sees (triggered, warning_message) tuples.

Guards:
  - repeat: same tool + same input 3+ times → ask_user
  - burst: 6+ consecutive taps with repeats → ask_user
  - scroll: same direction 8+ swipes without a tap/back in between → ask_user
  - magnify: 3+ consecutive magnify calls → ask_user
  - page_revisit: same conceptual page (OCR Jaccard >0.7) visited 3+ times → warn/stop
  - stale_screen: 4+ actions with no visible screen change → ask_user
  - idle_cooldown: consecutive no-tool-call iterations → sleep
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class LoopGuard:
    """Per-task guard state.  One instance per task execution."""

    def __init__(self) -> None:
        # ── Repeat detection ──
        self.repeat_tracker: dict[str, int] = {}
        self.last_repeat_key: str = ""

        # ── Tap burst detection ──
        self.burst_tracker: dict[str, int] = {}
        self.last_op_type: str = ""
        self.consecutive_op_count: int = 0

        # ── Scroll same-direction detection ──
        self.scroll_dir_tracker: dict[str, int] = {}

        # ── Magnify streak ──
        self.magnify_streak: int = 0

        # ── Stale screen ──
        self.stale_screen_count: int = 0

        # ── Page fingerprint (conceptual loop) ──
        self.page_visits: list[tuple[frozenset[str], int, str]] = []
        self.loop_detection_count: int = 0

        # ── Idle cooldown ──
        self.idle_streak: int = 0

        # ── Back-button loop detection ──
        self.consecutive_back_count: int = 0

        # ── Popup-stuck detection ──
        # When OCR shows ≤3 texts with high overlap across iterations,
        # the agent is likely stuck on a popup/dialog. Screen hash changes
        # from failed taps are NOT real progress — do not reset guards.
        self.popup_stuck_streak: int = 0
        self._last_popup_texts: frozenset[str] | None = None

    # ── Reset ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all guard state for a new task."""
        self.repeat_tracker.clear()
        self.last_repeat_key = ""
        self.last_op_type = ""
        self.consecutive_op_count = 0
        self.burst_tracker.clear()
        self.magnify_streak = 0
        self.stale_screen_count = 0
        self.page_visits.clear()
        self.loop_detection_count = 0
        self.scroll_dir_tracker.clear()
        self.idle_streak = 0
        self.consecutive_back_count = 0
        self.popup_stuck_streak = 0
        self._last_popup_texts = None

    # ── Repeat key builder (static, shared with interrupt handler) ─

    @staticmethod
    def build_repeat_key(tool_name: str, tool_input: dict[str, Any]) -> str:
        """Build a dedup key for a tool call.

        adb_tap_position coordinates are bucketed to nearest 0.05 to group
        taps in the same screen area.
        """
        if tool_name == "adb_tap_position":
            x_b = round(tool_input.get("x_pct", 0) / 0.05) * 0.05
            y_b = round(tool_input.get("y_pct", 0) / 0.05) * 0.05
            return f"adb_tap_position:({x_b:.2f},{y_b:.2f})"
        if tool_name in ("adb_tap",):
            return f"{tool_name}:{tool_input}"
        if tool_name == "adb_swipe":
            return (
                f"adb_swipe:{tool_input.get('direction','')}:"
                f"{tool_input.get('distance','')}:{tool_input.get('area','')}"
            )
        if tool_name == "adb_scroll":
            return (
                f"adb_scroll:{tool_input.get('direction','')}:"
                f"{tool_input.get('axis','')}:"
                f"{tool_input.get('distance','')}:{tool_input.get('area','')}"
            )
        return f"{tool_name}:{sorted(tool_input.items())}"

    # ── Guards (each returns (triggered, warning_message)) ─────────

    def check_repeat(self, repeat_key: str, tool_name: str,
                     tool_input: dict[str, Any]) -> tuple[bool, str]:
        """Detect exact-repeat: same tool+input 3+ times.

        Returns (True, warning_msg) if the agent should stop and ask_user.
        """
        if repeat_key != self.last_repeat_key:
            self.repeat_tracker.clear()
            self.last_repeat_key = repeat_key

        self.repeat_tracker[repeat_key] = self.repeat_tracker.get(repeat_key, 0) + 1
        if self.repeat_tracker[repeat_key] >= 3:
            msg = (
                f"[系统提示] 你已连续执行 {tool_name}({tool_input}) "
                f"{self.repeat_tracker[repeat_key]} 次，画面没有变化。"
                "停止重试，立即调用 ask_user() 询问用户。"
            )
            return True, msg
        return False, ""

    def check_burst(self, tool_name: str, repeat_key: str) -> tuple[bool, str]:
        """Detect tap burst: 6+ consecutive taps with at least one repeated target."""
        op_type = "tap" if tool_name in ("adb_tap", "adb_tap_position") else tool_name

        if op_type != self.last_op_type:
            self.burst_tracker.clear()
            self.consecutive_op_count = 1
            self.last_op_type = op_type
        else:
            self.burst_tracker[repeat_key] = self.burst_tracker.get(repeat_key, 0) + 1
            self.consecutive_op_count += 1

        if (
            op_type == "tap"
            and self.consecutive_op_count >= 6
            and any(v >= 2 for v in self.burst_tracker.values())
        ):
            msg = (
                f"[系统提示] 你已经连续点击了 {self.consecutive_op_count} 次，"
                "且存在重复操作。说明你找不到正确的点击目标。"
                "停止点击，立即调用 ask_user() 询问用户。"
            )
            return True, msg
        return False, ""

    def check_scroll(self, tool_name: str,
                     tool_input: dict[str, Any]) -> tuple[bool, str]:
        """Detect same-direction scroll loop: 8+ swipes without a non-swipe break.

        Raised from 4→8 to avoid false-positives during list-scanning tasks
        (box-scan can legitimately need 10+ same-direction swipes).
        ScrollTracker handles precise boundary detection via OCR comparison;
        this guard is the last-resort safety net."""
        if tool_name in ("adb_swipe", "adb_scroll"):
            if tool_name == "adb_scroll":
                # Convert semantic direction → physical for readable messages
                _d = tool_input.get("direction", "more")
                _a = tool_input.get("axis", "horizontal")
                _MAP = {("next","horizontal"): "left", ("next","vertical"): "up",
                        ("prev","horizontal"): "right", ("prev","vertical"): "down",
                        ("more","horizontal"): "left", ("more","vertical"): "up"}
                swipe_dir = _MAP.get((_d, _a), _d)
            else:
                swipe_dir = tool_input.get("direction", "")
            if swipe_dir:
                self.scroll_dir_tracker.setdefault(swipe_dir, 0)
                self.scroll_dir_tracker[swipe_dir] += 1
                if self.scroll_dir_tracker[swipe_dir] >= 8:
                    msg = (
                        f"[系统提示] 你已经向 {swipe_dir} 方向连续滚动了 "
                        f"{self.scroll_dir_tracker[swipe_dir]} 次还没找到目标。立即停止滑动。"
                        "检查右上角是否有列表/网格切换按钮（通常两个小图标），切换到另一种视图；"
                        "或检查左上角是否有返回箭头，用 adb_back 回到上级重新导航。"
                        "如果两条都试了还是找不到 → 立即 ask_user()。"
                    )
                    self.scroll_dir_tracker.clear()
                    return True, msg
        else:
            self.scroll_dir_tracker.clear()
        return False, ""

    def check_magnify(self, tool_name: str) -> tuple[bool, str]:
        """Detect magnify streak: 2+ consecutive magnify calls."""
        if tool_name == "magnify":
            self.magnify_streak += 1
            if self.magnify_streak >= 2:
                msg = (
                    f"[系统提示] 你已经连续使用 magnify/mark/tap 瞄准同一区域 "
                    f"{self.magnify_streak} 次，仍未成功。说明你的坐标估计有问题。\n"
                    "请尝试：1) adb_tap(\"目标文字\") 用文字匹配点击；"
                    "2) ask_user() 询问用户具体位置。"
                    "不要继续 magnify 循环。"
                )
                self.magnify_streak = 0
                return True, msg
        return False, ""

    def check_page_revisit(
        self,
        ocr_texts: list[str],
        current_dhash: str | None,
        on_failure: Callable[..., None] | None = None,
    ) -> tuple[bool, str]:
        """Detect conceptual page loops via OCR Jaccard similarity.

        Uses top-20 OCR texts as a page fingerprint.  When the same page is
        visited 3+ times, it means the agent is stuck in a semantic loop.

        Returns:
            (True, msg) — 1st detection: warn; 2nd+: force-stop signal.
            The caller must handle force-stop by checking loop_detection_count >= 2.
        """
        if not ocr_texts:
            return False, ""

        page_texts = frozenset(ocr_texts[:20])
        if len(page_texts) < 3:
            return False, ""

        for i, (prev_texts, count, _dhash) in enumerate(self.page_visits):
            intersection = len(page_texts & prev_texts)
            union = len(page_texts | prev_texts)
            overlap = intersection / union if union > 0 else 0.0

            if overlap > 0.7:
                new_count = count + 1
                self.page_visits[i] = (prev_texts, new_count, current_dhash or "")
                if new_count >= 3:
                    self.loop_detection_count += 1
                    detail = (
                        f"Same page visited {new_count}× "
                        f"(OCR overlap {overlap:.2f}). "
                        f"Screen OCR: {', '.join(list(page_texts)[:15])}"
                    )
                    logger.warning(
                        "Screen loop #%d: page visited %d× (overlap=%.2f)",
                        self.loop_detection_count, new_count, overlap,
                    )
                    if on_failure:
                        on_failure("screen_loop", detail=detail)

                    if self.loop_detection_count == 1:
                        msg = (
                            f"[系统提示] 你已经重复访问同一页面 {new_count} 次 "
                            f"(OCR 相似度 {overlap:.0%})，说明操作无效或目标已达成。"
                            "停止当前操作，立即调用 ask_user() 询问用户。"
                            f"当前画面: {', '.join(list(page_texts)[:12])}"
                        )
                    else:
                        # 2nd+ detection — force stop
                        msg = (
                            "[系统通知] 检测到重复操作环路且未被纠正，任务已强制终止。"
                            "请告知用户当前进度并重新派发。"
                        )
                    self.page_visits.pop(i)
                    return True, msg
                return False, ""

        self.page_visits.append((page_texts, 1, current_dhash or ""))
        if len(self.page_visits) > 10:
            self.page_visits.pop(0)
        return False, ""

    @property
    def force_stop_requested(self) -> bool:
        """True when page_revisit has been triggered 2+ times."""
        return self.loop_detection_count >= 2

    def check_stale_screen(self, screen_changed: bool,
                           last_user_hint: str = "") -> tuple[bool, str]:
        """Detect when 4+ actions produce zero visible screen change."""
        if not screen_changed:
            self.stale_screen_count += 1
            if self.stale_screen_count >= 4:
                hint = ""
                if last_user_hint:
                    hint = (
                        f"用户刚才告诉你「{last_user_hint[:80]}」，"
                        "但你接下来的操作没有产生效果。请重新理解用户的指导。"
                    )
                msg = (
                    f"[系统提示] 你已经连续执行了 {self.stale_screen_count} 个操作"
                    "但画面完全没有变化。说明你的点击没有命中目标。"
                    "不要继续猜测坐标，立即调用 ask_user() 询问用户具体位置。"
                    + hint
                )
                self.stale_screen_count = 0
                return True, msg
        else:
            self.stale_screen_count = 0
        return False, ""

    def check_popup_stuck(self, ocr_texts: list[str]) -> tuple[bool, str]:
        """Detect popup/dialog stuck: very few texts, high overlap across iterations.

        When the agent is stuck on a popup (e.g. "获得物资" confirmation dialog),
        screen hash changes from failed taps are noise, not progress. This guard
        prevents record_screen_change from resetting the real guards (burst, repeat,
        magnify) that would otherwise catch the flailing.
        """
        if not ocr_texts or len(ocr_texts) > 3:
            self.popup_stuck_streak = 0
            self._last_popup_texts = None
            return False, ""

        current = frozenset(ocr_texts)
        if self._last_popup_texts is not None:
            intersection = len(current & self._last_popup_texts)
            union = len(current | self._last_popup_texts)
            overlap = intersection / union if union > 0 else 0.0
            if overlap > 0.5:
                self.popup_stuck_streak += 1
            else:
                self.popup_stuck_streak = max(0, self.popup_stuck_streak - 1)
        else:
            self.popup_stuck_streak = 1

        self._last_popup_texts = current

        if self.popup_stuck_streak >= 3:
            return True, ""
        return False, ""

    def record_screen_change(self, changed: bool, ocr_texts: list[str] | None = None) -> None:
        """Call after action tools: reset state when screen actually moved.

        When the screen changes, we know the agent made real progress, so:
        - exact-repeat tracking is reset (the next operation is a new context)
        - burst tracker resets (tap sequence ended)
        - magnify streak resets (targeting loop ended)

        EXCEPTION: when popup-stuck is detected (≤3 OCR texts with high overlap
        across 3+ iterations), hash changes are noise from failed taps on a
        persistent popup — do NOT reset guards.
        """
        # Check popup-stuck BEFORE deciding to reset guards
        if ocr_texts:
            is_popup_stuck, _ = self.check_popup_stuck(ocr_texts)
        else:
            is_popup_stuck = self.popup_stuck_streak >= 3

        if changed and not is_popup_stuck:
            self.repeat_tracker.clear()
            self.burst_tracker.clear()
            self.magnify_streak = 0
            self.stale_screen_count = 0

    def reset_action_counters(self) -> None:
        """Reset all per-action counters (called when idle streak ends)."""
        self.idle_streak = 0
        self.repeat_tracker.clear()
        self.burst_tracker.clear()
        self.scroll_dir_tracker.clear()
        self.magnify_streak = 0
        self.consecutive_back_count = 0

    # ── Idle cooldown ──────────────────────────────────────────────

    def tick_back(self, pre_hash: str | None = None,
                  post_hash: str | None = None) -> int:
        """Increment consecutive back count and return it.

        P3: now hash-aware — only increments when back didn't change the
        screen.  A back that successfully navigated to a new screen resets
        the counter because it was genuinely useful.
        """
        if pre_hash and post_hash and pre_hash != post_hash:
            # Back changed the screen — it worked, reset counter
            self.consecutive_back_count = 0
            return 0
        self.consecutive_back_count += 1
        return self.consecutive_back_count

    def tick_idle(self) -> int:
        """Increment idle streak and return the new count (no cooldown).

        The IdleWatcher (dHash-based) now runs first and gates whether we
        need to wait at all.  Cooldown is only applied AFTER the watcher
        confirms the screen is truly static — see get_idle_cooldown().
        """
        self.idle_streak += 1
        return self.idle_streak

    def get_idle_cooldown(self) -> float:
        """Return a SMALL cooldown for truly static screens (IdleWatcher gave up).

        IdleWatcher already spent ~3-5s polling the screen.  If it found no
        change, the screen is truly static — add a brief backoff before the
        expensive LLM safety-net call to avoid spamming the LLM on a frozen
        screen.  Much smaller than before because IdleWatcher is the real gate.
        """
        if self.idle_streak < 2:
            return 0.0
        # Gentle ramp: 2→0.8s, 3→1.1s, 4→1.4s, …, cap at 2s
        return min(0.5 + (self.idle_streak - 1) * 0.3, 2.0)
