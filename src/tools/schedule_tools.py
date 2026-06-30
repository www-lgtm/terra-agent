"""Schedule management tools for the LLM agent.

Self-registers with the tool registry at import time. These are the LLM-callable
tools; the WeChat handler also uses schedule_db directly for fast keyword-based
management (list/delete/toggle/stop).
"""

from __future__ import annotations

import json
import logging
import time

from src.tools.registry import registry, ToolOutput

logger = logging.getLogger(__name__)


def schedule_create(
    name: str,
    schedule_type: str,
    schedule_value: str,
    task_description: str,
    one_shot: bool = False,
    game: str = "",
    task_type: str = "custom",
    slot_id: str = "",
) -> ToolOutput:
    """Create a scheduled task. Called by the LLM when the user wants to set up
    a timed / recurring task.

    Args:
        name: Short human-readable name (e.g. "早间清体力")
        schedule_type: "cron" for time-of-day patterns, "interval" for every-N
        schedule_value: Cron expression ("0 9 * * *") or interval ("30m", "2h", "1d")
        task_description: Natural language description of what to execute
        one_shot: If True, delete the task after its first execution
        game: Game name (default: "arknights")
        task_type: "skill", "custom", or "farming_plan"
        slot_id: (Phase 2) GameSlot id to bind this task to, e.g. "ark_main"
    """
    if schedule_type not in ("cron", "interval"):
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": f"Unknown schedule_type: {schedule_type}. Use 'cron' or 'interval'.",
        }, ensure_ascii=False))

    # Resolve game: explicit param > thread-local context > registry default
    if not game:
        from src.tools.registry import get_current_game
        game = get_current_game()

    from src.scheduler.schedule_db import schedule_db
    from src.scheduler.time_parser import calculate_next_run

    try:
        next_run_ts = calculate_next_run(schedule_type, schedule_value).timestamp()
    except ValueError as e:
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": f"Invalid schedule value: {e}",
        }, ensure_ascii=False))

    payload = {"custom_prompt": task_description}

    task_id = schedule_db.create(
        name=name,
        task_payload=payload,
        schedule_type=schedule_type,
        schedule_value=schedule_value,
        description=task_description,
        game=game,
        task_type=task_type,
        one_shot=one_shot,
        next_run=next_run_ts,
        slot_id=slot_id,
    )

    next_run_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(next_run_ts))
    logger.info("schedule_create tool: #%d %s (%s=%s) next=%s",
                 task_id, name, schedule_type, schedule_value, next_run_str)

    return ToolOutput(
        text=json.dumps({
            "success": True,
            "task_id": task_id,
            "name": name,
            "next_run": next_run_str,
            "message": (
                f"定时任务已创建: '{name}' (ID #{task_id}), 下次执行: {next_run_str}。"
                f"调度守护进程会在指定时间自动执行，你不需要等待。"
                f"你的任务已完成，请立即调用 task_complete() 结束本轮对话。"
            ),
        }, ensure_ascii=False),
        task_done=True,
    )


def schedule_list() -> ToolOutput:
    """List all scheduled tasks with their status."""
    from src.scheduler.schedule_db import schedule_db

    tasks = schedule_db.get_all()
    if not tasks:
        return ToolOutput(text=json.dumps({
            "success": True,
            "tasks": [],
            "count": 0,
            "message": "暂无定时任务。",
        }, ensure_ascii=False))

    result = []
    for t in tasks:
        next_run_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t["next_run"])) if t["next_run"] else "N/A"
        last_run_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t["last_run"])) if t["last_run"] else "从未"
        result.append({
            "id": t["id"],
            "name": t["name"],
            "description": t["description"],
            "schedule_type": t["schedule_type"],
            "schedule_value": t["schedule_value"],
            "enabled": bool(t["enabled"]),
            "one_shot": bool(t["one_shot"]),
            "next_run": next_run_str,
            "last_run": last_run_str,
            "run_count": t["run_count"],
        })

    return ToolOutput(text=json.dumps({
        "success": True,
        "tasks": result,
        "count": len(result),
    }, ensure_ascii=False))


def schedule_delete(task_id: int) -> ToolOutput:
    """Delete a scheduled task by its ID."""
    from src.scheduler.schedule_db import schedule_db

    task = schedule_db.get_by_id(task_id)
    if task is None:
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": f"未找到定时任务 #{task_id}",
        }, ensure_ascii=False))

    # Cancel any running instance before deleting
    try:
        from src.scheduler.cron_scheduler import get_engine
        get_engine().cancel_task(task_id)
    except Exception:
        pass

    schedule_db.delete(task_id)
    logger.info("schedule_delete tool: deleted #%d (%s)", task_id, task["name"])

    return ToolOutput(text=json.dumps({
        "success": True,
        "message": f"已删除定时任务 #{task_id} ({task['name']})",
    }, ensure_ascii=False))


def schedule_toggle(task_id: int, enabled: bool) -> ToolOutput:
    """Enable or disable (pause) a scheduled task by its ID.

    Args:
        task_id: The task's numeric ID
        enabled: True to enable, False to disable/pause
    """
    from src.scheduler.schedule_db import schedule_db

    task = schedule_db.get_by_id(task_id)
    if task is None:
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": f"未找到定时任务 #{task_id}",
        }, ensure_ascii=False))

    # Cancel running instance when disabling
    if not enabled:
        try:
            from src.scheduler.cron_scheduler import get_engine
            get_engine().cancel_task(task_id)
        except Exception:
            pass

    schedule_db.set_enabled(task_id, enabled)
    action = "启用" if enabled else "暂停"
    logger.info("schedule_toggle tool: %s #%d (%s)", action, task_id, task["name"])

    return ToolOutput(text=json.dumps({
        "success": True,
        "message": f"已{action}定时任务 #{task_id} ({task['name']})",
    }, ensure_ascii=False))


# ---- Tool Registration ----

registry.register(
    name="schedule_create",
    description="Create a timed / recurring task. Use when user says '定时', '每天', '每周', '每隔'. Supports cron expressions and interval strings.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Short name for the schedule (e.g. '早间清体力')"},
            "schedule_type": {"type": "string", "enum": ["cron", "interval"],
                              "description": "'cron' for time-of-day, 'interval' for every-N-minutes/hours"},
            "schedule_value": {"type": "string",
                               "description": "Cron: '0 9 * * *' (daily 9am, server local time). Interval: '30m', '2h', '1d'"},
            "task_description": {"type": "string",
                                 "description": "Natural language description of what the agent should execute"},
            "one_shot": {"type": "boolean",
                         "description": "True = run once then auto-delete. False = recurring"},
            "game": {"type": "string", "description": "Game name (default: arknights)"},
            "task_type": {"type": "string", "enum": ["skill", "custom", "farming_plan"],
                          "description": "Type of task payload"},
            "slot_id": {"type": "string",
                        "description": "(Phase 2) GameSlot ID to bind this task to (e.g. 'ark_main'). Leave empty for auto-dispatch."},
        },
        "required": ["name", "schedule_type", "schedule_value", "task_description"],
    },
    handler=schedule_create,
)

registry.register(
    name="schedule_list",
    description="List all scheduled (cron/timer) tasks with their status and next run times.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    handler=schedule_list,
)

registry.register(
    name="schedule_delete",
    description="Delete a scheduled task by its numeric ID.",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "The numeric ID of the task to delete"},
        },
        "required": ["task_id"],
    },
    handler=schedule_delete,
)

registry.register(
    name="schedule_toggle",
    description="Enable or disable (pause/resume) a scheduled task by its ID.",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "The numeric ID of the task"},
            "enabled": {"type": "boolean", "description": "True to enable, False to disable/pause"},
        },
        "required": ["task_id", "enabled"],
    },
    handler=schedule_toggle,
)
