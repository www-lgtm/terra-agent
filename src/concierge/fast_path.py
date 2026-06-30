"""快速通道 — 规则匹配，零 LLM 调用。

处理问候、状态查询、取消、追问确认等高频模式。
返回回复文本或 None（表示回退到直接委派）。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def try_fast_path(text: str, concierge: Any) -> tuple[str | None, bool]:
    """尝试快速路径处理用户消息。

    Returns:
        (reply_text, consumed_clarification)
        - reply_text: 回复文本或 None（回退到直接委派）
        - consumed_clarification: True 表示消费了追问确认
    """
    stripped = text.strip()

    # 1. 确认追问 — 用户回复了管家的上一条追问
    if concierge.session.pending_clarification is not None:
        _handle_clarification_response(stripped, concierge)
        # 追问已消费，返回 (None, True) 让慢速通道用注入的 refined 消息重新处理
        return None, True

    # 2. 问候
    if _is_greeting(stripped):
        return _handle_greeting(concierge), False

    # 3. 状态/进度查询
    if _is_status_query(stripped):
        return _handle_status(concierge), False

    # 4. 取消/停止
    if _is_cancel(stripped):
        return _handle_cancel(concierge), False

    return None, False


def _is_greeting(text: str) -> bool:
    small = text.lower().replace(" ", "")
    greetings = ["你好", "在吗", "在不在", "谢谢", "辛苦了", "早", "晚上好", "晚安", "嗨",
                 "hello", "hi", "hey"]
    if small in greetings:
        return True
    # 以问候词开头时，只有剩余部分是语气词/标点才算问候。
    # 防止 "你好帮我清体力" 被误判为 CHAT 而吞掉任务指令。
    for g in ["你好", "谢谢", "辛苦"]:
        if small.startswith(g):
            remainder = small[len(g):]
            if not remainder or all(
                c in "啊呀哦呢啦吧嘛噢哟哎诶哈呵嘻嘿哒嘞噜呐哇！!，,。.~～…、？? " for c in remainder
            ):
                return True
    return False


def _is_status_query(text: str) -> bool:
    kw = ["状态", "进度", "在干嘛", "怎么样了", "如何了", "在做什么", "情况", "运行"]
    return any(k in text for k in kw) and len(text) <= 20


def _is_cancel(text: str) -> bool:
    kw = ["停止", "取消", "算了", "别打了", "不要了"]
    return any(k in text for k in kw) and len(text) <= 15


# ── 处理函数 ──

def _handle_greeting(concierge: Any) -> str:
    if concierge._slots:
        return _handle_status(concierge)
    agent = concierge.session.current_agent
    if agent is not None and agent.state.running:
        state = agent.state
        activity = state.current_activity or state.task_description or "执行任务"
        return f"博士好！当前正在 {activity}（已执行 {state.iteration_count} 步）。有什么需要？"
    return "博士好！有什么我可以帮你的？"


def _handle_status(concierge: Any) -> str:
    # Multi-slot: use status panel
    if concierge._slots:
        from src.concierge.status_panel import format_status_panel
        return format_status_panel(concierge._slots)

    # Single agent: old format
    agent = concierge.session.current_agent
    if agent is None or not agent.state.running:
        return "目前没有正在执行的任务。博士想让我做什么？"

    state = agent.state
    task_label = state.task_description or "(未指定)"
    parts = [f"当前任务: {task_label}"]

    if state.current_activity:
        parts.append(f"正在: {state.current_activity}")

    parts.append(f"已执行 {state.iteration_count} 步")

    # Token consumption summary
    token_summary = state.token_cost_summary
    if token_summary and state.total_input_tokens > 0:
        parts.append(f"消耗 {token_summary}")

    import time
    if state.started_at > 0:
        elapsed = int((time.time() - state.started_at) // 60)
        if elapsed > 0:
            parts.append(f"已运行约 {elapsed} 分钟")
        else:
            parts.append("刚开始运行")

    return "，".join(parts) + "。"


def _handle_cancel(concierge: Any) -> str:
    """停止当前任务。委托给 tools.py _cancel_task 统一处理。"""
    from src.concierge.tools import _cancel_task
    return _cancel_task(concierge)


def _handle_clarification_response(text: str, concierge: Any) -> None:
    """用户回复了之前管家的追问（选 slot / 确认操作等）。"""
    pending_task = concierge.session.pending_task
    clarification = concierge.session.pending_clarification or ""
    # Save clarification type before clearing so router can re-route correctly
    concierge.session._last_clarification_type = clarification
    concierge.session.pending_clarification = None
    concierge.session.pending_task = None

    if not pending_task:
        return

    # Slot selection: user named a slot → resolve it and re-delegate
    if clarification.startswith("select_slot"):
        for s in concierge._slots:
            if s.match_label(text):
                concierge.session.agent_device_serial = s.device_serial
                refined = (
                    f"请执行任务: {pending_task}"
                    f"（用户选的是 {s.label}，设备 {s.device_serial}）"
                )
                concierge.session.conversation_history.append(
                    {"role": "user", "content": refined}
                )
                logger.info("Clarification resolved: slot=%s task=%s",
                           s.label, pending_task[:80])
                return
        # No slot matched — reinject the original task, ask LLM to re-prompt
        concierge.session.conversation_history.append(
            {"role": "user",
             "content": (
                 f"请执行任务: {pending_task}。"
                 f"用户回复「{text}」，但未匹配到任何账号。"
                 f"请重新询问用户要操作哪个号。"
             )}
        )
        return

    # Generic / concierge_confirm clarification: inject user's answer as context
    # and re-run through concierge LLM (if available) or direct delegate.
    refined = f"[用户补充] 管家之前问：「{clarification}」，用户回复「{text}」。请根据完整信息重新决定。"
    concierge.session.conversation_history.append({"role": "user", "content": refined})
    concierge.session.conversation_history.append(
        {"role": "assistant", "content": "明白，让我根据补充信息重新评估。"}
    )

    logger.info("Consumed pending clarification: %s", clarification[:80])
