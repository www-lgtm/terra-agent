"""notify_with_screen tool: send screenshot + text to WeChat without pausing the task.

Unlike ask_user (which pauses execution waiting for a reply), this tool
captures the screen, sends it to the user via the same WeChat notification
pipeline, and returns immediately — the task continues uninterrupted.

Use cases:
- "每日奖励已领完，截图确认"
- "高级资深干员出现，截图通知"
- 子技能失败时附带画面告知用户
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.tools.registry import registry, ToolOutput

logger = logging.getLogger(__name__)


def notify_with_screen(message: str) -> ToolOutput:
    """Capture the current screen and send it with a message to WeChat.

    Does NOT pause the task.  The notification is fire-and-forget —
    the task continues immediately.

    Args:
        message: What to tell the user.  Be specific: what's done,
                 what the screenshot shows, what to expect next.
    """
    # Get the agent from thread-local context
    import threading
    ctx = getattr(threading.current_thread(), '_terra_agent_ctx', None)
    if ctx is None:
        logger.warning("notify_with_screen: no agent context — skipping")
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": "无法获取当前agent上下文。此工具只能在 agent 线程内调用。",
        }, ensure_ascii=False))

    try:
        from src.agent.screen_injector import capture_screen_jpeg
        image_b64 = capture_screen_jpeg()
    except Exception as e:
        logger.warning("notify_with_screen: screen capture failed — %s", e)
        image_b64 = None

    # Use notify_type="screenshot" — triggers image delivery in concierge
    # router without adding to waiting_slots (unlike "ask_user").
    try:
        ctx._notify(message, notify_type="screenshot", image_b64=image_b64)
    except Exception as e:
        logger.warning("notify_with_screen: notify failed — %s", e)
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": f"通知发送失败: {e}",
        }, ensure_ascii=False))

    logger.info("notify_with_screen: sent — %s", message[:120])
    return ToolOutput(text=json.dumps({
        "success": True,
        "message": "已发送截图通知",
    }, ensure_ascii=False))


registry.register(
    name="notify_with_screen",
    description=(
        "截取**当前屏幕画面**发送到微信通知 —— 不暂停任务，发送后继续执行。\n"
        "🔴 截的是调用这一刻的屏幕！先确保画面正确再调用，不要在错误页面截图。\n"
        "🔴 基建收菜 → 必须在通知面板打开时截图，关闭面板后截的是基建俯视图，用户什么都看不到。\n"
        "🔴 任务奖励 → 必须在任务面板上截图，不要在回到主界面后再截。\n"
        "🔴 截图必须干净：不能有弹窗/动画遮挡。等动画结束、弹窗消失后再调用。\n"
        "正确顺序：notify_with_screen → subtask_done（不能反过来）。\n"
        "用于以下场景：\n"
        "- 子任务完成时通知用户成果（附上结果画面）\n"
        "- 发现重要画面截屏通知用户\n"
        "- 某步骤卡住时附带画面让用户了解当前状态\n"
        "注意：这不是 ask_user —— 用户不会回复，任务继续执行。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "发送给用户的消息文本——当前在做什么、截图显示的是什么",
            },
        },
        "required": ["message"],
    },
    handler=notify_with_screen,
)
