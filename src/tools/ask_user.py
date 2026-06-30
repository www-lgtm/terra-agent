"""ask_user tool: LLM calls this when it doesn't know what to do next.

The handler is set by TerraAgent during initialization.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from src.tools.registry import registry, ToolOutput

logger = logging.getLogger(__name__)

_agent_ask_handler: Callable[[str], str] | None = None


def set_ask_handler(fn: Callable[[str], str] | None) -> None:
    """Set the callback for asking the user. Called by TerraAgent.__init__.

    DEPRECATED: Prefer setting ask_fn directly on the TerraAgent instance.
    This global setter remains for backward compatibility.
    """
    global _agent_ask_handler
    _agent_ask_handler = fn


def _get_ask_handler() -> Callable[[str], str] | None:
    """Get the current ask handler from thread-local agent context."""
    import threading
    ctx = getattr(threading.current_thread(), '_terra_agent_ctx', None)
    if ctx is not None and hasattr(ctx, 'ask_fn'):
        return ctx.ask_fn
    return _agent_ask_handler


def ask_user(question: str) -> ToolOutput:
    """Ask the user for help when you don't know what to do next.

    Use this when:
    - A tool failed and the available options don't match what you expected
    - The VLM description doesn't match the screen you thought you were on
    - You've tried the same action twice and it still fails
    - You need the user to confirm a dangerous operation

    Args:
        question: What you want to ask. Be specific — include what you expected,
                  what you see instead, and what options you're considering.
    """
    handler = _get_ask_handler()
    if handler is None:
        logger.info("ask_user called in async mode, pausing for user: %s", question[:200])
        return ToolOutput(
            text=json.dumps({"question": question, "message": question}, ensure_ascii=False),
            needs_user=True,
        )

    logger.info("Asking user: %s", question[:200])
    answer = handler(question)
    logger.info("User answered: %s", answer[:200])

    return ToolOutput(text=json.dumps({
        "success": True,
        "question": question,
        "answer": answer,
    }, ensure_ascii=False))


registry.register(
    name="ask_user",
    description=(
        "向用户求助 —— 这不是放弃，而是最高效的做法。用户看一眼画面就能告诉你的东西，"
        "你花 10 轮也猜不出来。以下情况必须立即调用 ask_user，不要尝试第 4 次：\n"
        "1) 同一操作（tap/swipe/magnify）换了 3 种不同方式都失败 → ask_user\n"
        "2) 不确定滑动方向（该左滑还是右滑）→ ask_user，不要猜\n"
        "3) 屏幕变了但你不确定发生了什么 → ask_user 让用户确认\n"
        "4) 需要用户做选择或确认消耗资源 → ask_user\n"
        "提问要具体：你期望看到什么、实际看到什么、你在考虑哪几个选项。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "要问用户的具体问题——你期望什么、实际看到什么、在考虑哪些选项"},
        },
        "required": ["question"],
    },
    handler=ask_user,
)
