"""Agent state management — Anthropic message format."""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.guard_context import GuardContext


@dataclass
class FailureSignal:
    """A failure event captured during task execution — the raw material for
    high-quality memory extraction.

    Stored on AgentState and passed to the background reviewer's memory
    extraction sub-agent so it can see exactly where the main agent got stuck,
    on which screen, and what strategy switch eventually resolved it.
    """

    timestamp: float = field(default_factory=time.time)
    iteration: int = 0
    signal_type: str = ""  # "repeat_stuck"|"tap_burst"|"scroll_loop"|"magnify_streak"|"stale_screen"|"tool_failure"
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    screen_hash: str | None = None   # MD5 hash from injection label
    screen_dhash: str | None = None  # dHash hex for visual matching
    ocr_texts: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class AgentNotification:
    """Structured notification from a game agent to the Concierge/gateway layer.

    Replaces the loose (msg, notify_type, image_b64, agent_id) kwarg tuple.
    Concierge and gateway code consume this to send WeChat messages, update
    slot state, and route follow-up user replies.
    """

    type: str = "progress"          # "progress" | "ask_user" | "complete" | "error" | "screenshot"
    agent_id: str = ""              # AgentHandle.agent_id for slot resolution
    message: str = ""               # Human-readable message text
    image_b64: str | None = None    # JPEG screenshot (ask_user only)
    metadata: dict[str, Any] = field(default_factory=dict)  # extensible payload

    def __post_init__(self) -> None:
        if self.type not in ("progress", "ask_user", "complete", "error", "screenshot"):
            raise ValueError(f"Invalid notification type: {self.type}")


@dataclass
class AgentState:
    """Mutable state for a single agent session."""

    game: str = "arknights"
    device_serial: str = ""
    running: bool = False
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    last_skill_name: str = ""
    task_description: str = ""
    task_type: str = ""
    matching_skills: list[dict[str, Any]] = field(default_factory=list)

    # Screen injection (Always-On Vision)
    last_injected_hash: str | None = None
    last_injected_dhash: str | None = None  # dHash hex for visual memory matching
    screen_w: int = 1600
    screen_h: int = 900

    # Failure-driven memory extraction (Phase 0)
    failure_signals: list[FailureSignal] = field(default_factory=list)

    # Memory injection tracking (Phase 1 — feedback loop)
    injected_memory_ids: list[int] = field(default_factory=list)
    last_ocr_texts: list[str] = field(default_factory=list)
    last_screen_brightness: float = 0.0  # mean pixel brightness 0-255 of last injected screen

    # Learning engine (Phase 1)
    task_execution_id: int | None = None
    injection_feedback_tracker: Any | None = None  # InjectionFeedbackTracker instance

    # Async interrupt support
    # on_notify(msg, notify_type="progress", image_b64=None, agent_id=None)
    # — Called by the agent to send progress / questions / screenshots
    # to external channels (WeChat).  agent_id identifies which agent
    # (and thus which GameSlot) the notification originates from.
    on_notify: Callable[..., None] | None = None
    agent_handle: Any | None = None  # AgentHandle ref, set by run_async for identity
    _waiting_for_user: bool = False
    _interrupt_queue: deque[str] = field(default_factory=deque)
    _interrupt_lock: threading.Lock = field(default_factory=threading.Lock)
    _interrupt_event: threading.Event = field(default_factory=threading.Event)

    # ── Concierge integration (Phase 1) ──
    # 游戏 agent 线程写入，concierge 管家线程只读
    # Python GIL 保护单字段赋值的原子性
    current_activity: str = ""              # 当前正在执行的操作描述
    last_progress_text: str = ""            # 最近一次 _progress() 的消息
    started_at: float = 0.0                 # 任务开始时间戳 (time.time())
    interrupt_zone: str = "safe"            # "safe" | "battle" | "critical"
    interrupt_zone_detail: str = ""         # 额外上下文描述
    _pending_cancel: bool = False           # 管家请求取消，等安全点执行
    summary_since_last_check: str = ""      # 累积进度摘要
    last_summary_iteration: int = 0         # 上次写摘要时的 iteration_count
    _ask_user_count: int = 0                # ask_user 调用次数（兜底触发复盘）
    _user_msg_count: int = 0                # 用户在任务期间发来消息次数
    _task_completed: bool = False           # task_complete() 已调用，阻止 re-entry
    skill_fast_chain_result: bool | None = None  # fast_chain 执行结果（None=未使用）
    completed_subtasks: set[str] = field(default_factory=set)  # 幂等去重
    skipped_subtasks: set[str] = field(default_factory=set)    # 用户明确跳过的不相关技能
    total_input_tokens: int = 0       # cumulative token tracking
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0  # cumulative cache read tokens (billed at ¥0.02/M)
    total_cache_create_tokens: int = 0  # cumulative cache create tokens (billed at ¥1/M)
    user_task: str = ""               # user task instruction, moved to system prompt for caching
    guard: Any = None  # GuardContext | None — runtime guard state, set by TerraAgent

    @property
    def token_cost_summary(self) -> str:
        """Human-readable token consumption summary for status/notifications."""
        inp = self.total_input_tokens
        out = self.total_output_tokens
        if inp == 0 and out == 0:
            return "尚未统计 token 消耗"
        parts: list[str] = []
        if inp >= 1_000_000:
            parts.append(f"输入 {inp / 1_000_000:.1f}M")
        elif inp >= 1_000:
            parts.append(f"输入 {inp / 1_000:.0f}k")
        else:
            parts.append(f"输入 {inp}")
        if out >= 1_000_000:
            parts.append(f"输出 {out / 1_000_000:.1f}M")
        elif out >= 1_000:
            parts.append(f"输出 {out / 1_000:.0f}k")
        else:
            parts.append(f"输出 {out}")
        return "，".join(parts)

    def inject_message(self, text: str) -> None:
        """Inject a user message into the queue (called from outside the agent thread)."""
        self._user_msg_count += 1
        with self._interrupt_lock:
            self._interrupt_queue.append(text)
        self._interrupt_event.set()

    def pop_interrupt(self) -> str | None:
        """Pop next interrupt message (called from inside the agent loop)."""
        with self._interrupt_lock:
            if self._interrupt_queue:
                return self._interrupt_queue.popleft()
            return None

    def has_pending_interrupt(self) -> bool:
        """Check if an interrupt is queued WITHOUT consuming it.

        Use this in non-primary code paths (idle cooldown, watchers) that
        want to wake early on user input but must not consume the interrupt —
        the main loop's pop_interrupt() at the top of each iteration is the
        canonical consumer.
        """
        with self._interrupt_lock:
            return len(self._interrupt_queue) > 0

    def add_message(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 400:
            from src.agent.compressor import compress_history
            self.conversation_history = compress_history(self.conversation_history)

    def add_tool_result(self, tool_call_id: str, name: str, result: str | list[dict[str, Any]]) -> None:
        self.conversation_history.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": result,
            }],
        })

    def add_screen_injection(self, img_b64: str, ocr_texts: list[str], screen_hash: str) -> None:
        """Inject screenshot + OCR as a user message (not tied to any tool_call).

        Keeps only the 3 most recent screen images in context.  Older
        screenshots are replaced with text-only markers (OCR text preserved).
        Each stale image costs ~2000 tokens per API call; after 200
        iterations, accumulated historical images dominate input cost.
        """
        if ocr_texts:
            label = f"[系统自动截图 — 当前屏幕 — HASH:{screen_hash}]\nOCR:{ocr_texts[:30]}"
        else:
            bright = self.last_screen_brightness >= 15
            if bright:
                label = (
                    f"[系统自动截图 — 当前屏幕 — HASH:{screen_hash}]\n"
                    f"OCR: 0 条文字 (画面亮度正常, mean={self.last_screen_brightness:.0f})。"
                    f"可能只是图形化文字（启动/登录画面按钮等）OCR 无法识别。"
                    f"如果通过 magnify 看到可操作按钮 → 使用 tap_magnified 点击。"
                    f"确认画面无操作可能且等待 3 轮以上无变化 → ask_user()。"
                )
            else:
                label = (
                    f"[系统自动截图 — 当前屏幕 — HASH:{screen_hash}]\n"
                    f"⚠️ 画面无任何文字且偏暗 (mean={self.last_screen_brightness:.0f}) — "
                    f"你在加载/黑屏/过渡动画中。"
                    f"不要导航！不要点击！不要滑动！等待画面变化。"
                    f"这不是桌面，不是主界面，不是任何可操作页面。"
                )

        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
            {"type": "text", "text": label},
        ]
        self.conversation_history.append({"role": "user", "content": content})

        # Strip old images AFTER adding the new one so it counts as one of
        # the keep_recent slots.  Only the 3 most recent screen images survive.
        self._strip_old_injection_images(keep_recent=1)

        # Debug: save screenshot to disk for visual inspection
        # try:
        #     import base64
        #     from pathlib import Path
        #     debug_dir = Path("data/debug_screenshots")
        #     debug_dir.mkdir(parents=True, exist_ok=True)
        #     fname = f"iter_{self.iteration_count:03d}_{screen_hash[:8]}_{len(ocr_texts)}txt.jpg"
        #     (debug_dir / fname).write_bytes(base64.b64decode(img_b64))
        # except Exception:
        #     logger.debug("Failed to save debug screenshot (non-critical)", exc_info=True)

    def _strip_previous_injection_image(self) -> None:
        """Remove the image block from the most recent screen injection message.

        Leaves the OCR text intact so the LLM retains context of what was on
        screen.  Replaces the content list with a text-only content list
        (NOT a bare string) — preserves `isinstance(content, list)` checks
        throughout the codebase and doesn't break API message immutability
        for retry scenarios.
        """
        for i in range(len(self.conversation_history) - 1, -1, -1):
            msg = self.conversation_history[i]
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            # Check if this is a screen injection: has image + text with OC
            has_image = False
            has_screen_label = False
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        has_image = True
                    elif block.get("type") == "text":
                        txt = str(block.get("text", ""))
                        if "HASH:" in txt:
                            has_screen_label = True
                            # Extract just the OCR portion for context
                            import re as _re
                            ocr_m = _re.search(r"OCR:([\w一-鿿, /-]+)", txt)
                            if ocr_m:
                                text_parts.append(f"OCR:{ocr_m.group(1)}")
            if has_image and has_screen_label:
                # Replace with text-only content list — preserves list type
                # so isinstance(content, list) checks don't break.
                ocr_context = ", ".join(text_parts) if text_parts else "已移除"
                msg["content"] = [
                    {"type": "text", "text": f"[上一屏截图已移除 — {ocr_context}]"}
                ]
                return

    def _strip_old_injection_images(self, keep_recent: int = 1) -> None:
        """Strip images from all old screen injections, keeping only the N most recent.

        Each screen injection image costs ~2000 tokens.  After 200 iterations
        without cleaning, accumulated historical images add 200K+ tokens to
        every late-iteration API call — 92% of cost comes from stale images
        the LLM no longer needs.

        Only counts injections that STILL HAVE an image block (already-
        stripped text-only injections are skipped).  This prevents double-
        stripping when _strip_previous_injection_image() runs first.
        """
        import re as _re
        # First pass: find indices of injections that still have images
        injection_indices: list[int] = []
        for i in range(len(self.conversation_history) - 1, -1, -1):
            msg = self.conversation_history[i]
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            has_image = False
            has_hash = False
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        has_image = True
                    elif block.get("type") == "text" and "HASH:" in str(block.get("text", "")):
                        has_hash = True
            if has_image and has_hash:
                injection_indices.append(i)

        if len(injection_indices) <= keep_recent:
            return  # Nothing to strip

        # Strip images from older injections (indices beyond keep_recent)
        for idx in injection_indices[keep_recent:]:
            msg = self.conversation_history[idx]
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new_content: list[dict] = []
            hash_val = ""
            for block in content:
                if not isinstance(block, dict):
                    new_content.append(block)
                    continue
                if block.get("type") == "image":
                    continue  # Drop the image
                if block.get("type") == "text":
                    txt = str(block.get("text", ""))
                    m = _re.search(r"HASH:([0-9a-f]+)", txt)
                    if m:
                        hash_val = m.group(1)
                new_content.append(block)
            if hash_val:
                new_content.append(
                    {"type": "text", "text": f"[截图已省略 — HASH:{hash_val}]"}
                )
            msg["content"] = new_content

    def add_screen_injection_text_only(self, ocr_texts: list[str]) -> None:
        """Same-screen dedup: only inject OCR text, skip the repeated image."""
        self.conversation_history.append({
            "role": "user",
            "content": f"[屏幕没变 — OCR与上轮相同]\nOCR:{ocr_texts[:30]}",
        })

    def add_assistant_with_tools(self, text: str, tool_calls: list[dict[str, Any]]) -> None:
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        for tc in tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": tc["input"],
            })
        self.conversation_history.append({"role": "assistant", "content": content})

    def clean_subtask_history(self, subtask_name: str, result: str) -> None:
        """Clean intermediate operations for a completed subtask from history.

        Scans backward to find where this subtask began (either a skill_run
        call or the periodic review referencing it), then drops all tap-by-tap
        detail between that point and the current screen.  Keeps only a
        summary record so the agent enters the next subtask with clean context.

        Preserves:
        - First 2 messages (task description + initial LLM response)
        - Messages before the subtask boundary (prior subtask completion records)
        - Last 3 messages (current screen injection + recent interaction)
        - Protected messages (user instructions, system hints, memory injections)
        """
        history = self.conversation_history
        if len(history) <= 10:
            return  # Not enough to clean

        # Find the subtask boundary — scan backward for the first message
        # that mentions the subtask name in a skill_run tool_use or a
        # periodic review checkpoint.
        boundary_idx: int | None = None
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Check for skill_run('subtask_name') in assistant tool_use
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        inp = block.get("input", {})
                        if (block.get("name") == "skill_run"
                                and isinstance(inp, dict)
                                and inp.get("name") == subtask_name):
                            boundary_idx = i
                            break
                if boundary_idx is not None:
                    break

            # Check for periodic review mentioning this subtask
            if isinstance(content, str) and role == "user":
                if ("[系统提示" in content and "回顾" in content
                        and subtask_name in content):
                    boundary_idx = i
                    break

        if boundary_idx is None:
            return  # Can't find boundary — don't clean

        # Identify protected messages (user instructions, system hints, etc.)
        PROTECTED_PREFIXES = (
            "[用户指令", "[用户回复]", "[Guidance]", "[系统提示",
            "[关联记忆", "[子任务完成]", "[已省略",
        )

        def _is_protected(msg: dict) -> bool:
            c = msg.get("content", "")
            if isinstance(c, str) and msg.get("role") == "user":
                stripped = c.strip()
                return any(stripped.startswith(p) for p in PROTECTED_PREFIXES)
            return False

        # Build cleaned history:
        #   [0:2] + pre-boundary non-detail + protected + marker + last 3
        keep_head = history[:2]  # Task description + initial response

        # Messages before the subtask boundary that are NOT detail messages
        pre_boundary: list[dict] = []
        for msg in history[2:boundary_idx]:
            if _is_protected(msg):
                pre_boundary.append(msg)
            # Drop tap-by-tap detail (assistant tool_use + user tool_result
            # pairs) from before this subtask boundary.

        # Protected messages in the subtask region we skip
        middle_protected: list[dict] = []
        for msg in history[boundary_idx:-3]:
            if _is_protected(msg):
                middle_protected.append(msg)

        keep_tail = history[-3:]

        marker = {
            "role": "user",
            "content": f"[子任务完成] {subtask_name} → {result}",
        }

        self.conversation_history = (
            keep_head
            + pre_boundary
            + middle_protected
            + [marker]
            + keep_tail
        )

    def reset(self) -> None:
        self.conversation_history.clear()
        self.iteration_count = 0
        self.last_skill_name = ""
        self.task_description = ""
        self.task_type = ""
        self.matching_skills.clear()
        self._waiting_for_user = False
        self.last_injected_hash = None
        self.last_injected_dhash = None
        self.failure_signals.clear()
        self.injected_memory_ids.clear()
        self.last_ocr_texts.clear()
        self.last_screen_brightness = 0.0
        self.current_activity = ""
        self.last_progress_text = ""
        self.started_at = 0.0
        self.interrupt_zone = "safe"
        self.interrupt_zone_detail = ""
        self._pending_cancel = False
        self.summary_since_last_check = ""
        self.last_summary_iteration = 0
        self._ask_user_count = 0
        self._user_msg_count = 0
        self._task_completed = False
        self.skill_fast_chain_result = None
        self.completed_subtasks.clear()
        self.skipped_subtasks.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_create_tokens = 0
        self.user_task = ""
        self.agent_handle = None
        with self._interrupt_lock:
            self._interrupt_queue.clear()
        self._interrupt_event.clear()
