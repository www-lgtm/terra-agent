"""Status panel formatter — per-slot, user-visible task display."""

from __future__ import annotations

from typing import Any


def format_status_panel(slots: list[Any]) -> str:
    """Format a status panel showing all GameSlots and their tasks.

    Example output:
        📊 Terra 状态

        📱 方舟主号 (5554)
          🟢 清体力-GT-6  战斗中·第4波  34步

        📱 方舟小号 (5556)
          ⚪ 空闲

        无正在运行的任务时显示简洁的空闲提示。
    """
    if not slots:
        return "没有配置游戏账号。发送「清体力」开始。"

    lines: list[str] = ["=== Terra 状态 ===", ""]

    running_count = 0
    queued_count = 0

    for s in slots:
        status_icon: str
        status_text: str
        handle = s.current_task  # AgentHandle | TerraAgent | None

        if handle is not None and (hasattr(handle, 'is_running') and handle.is_running
                                   or (not hasattr(handle, 'is_running') and handle.state.running)):
            running_count += 1
            agent = handle.agent if hasattr(handle, 'agent') else handle
            state = agent.state
            agent_id = f"#{handle.agent_id} " if hasattr(handle, 'agent_id') else ""
            activity = state.current_activity or state.task_description or "执行中"
            iter_str = f"  {state.iteration_count}步" if state.iteration_count else ""
            zone_tag = ""
            if state.interrupt_zone != "safe":
                zone_tag = f" [{state.interrupt_zone}]"
            status_icon = ">>"
            status_text = f"{agent_id}{activity}{zone_tag}{iter_str}"
        elif s.pending_tasks:
            status_icon = ".."
            next_task = s.pending_tasks[0]
            desc = getattr(next_task, 'task_description', str(next_task))[:30]
            status_text = f"排队: {desc}（共{len(s.pending_tasks)}个）"
            queued_count += 1
        else:
            status_icon = "--"
            status_text = "空闲"

        device_short = s.device_serial.replace("emulator-", "").replace("127.0.0.1:", "")
        lines.append(f"[{s.label}] ({device_short})")
        lines.append(f"  {status_icon} {status_text}")
        lines.append("")

    # Summary line
    parts: list[str] = []
    if running_count > 0:
        parts.append(f"{running_count} running")
    if queued_count > 0:
        parts.append(f"{queued_count} queued")
    if not parts:
        parts.append("all idle")
    lines.append("--- " + ", ".join(parts) + " ---")
    lines.append("")

    # ── Recently completed ──
    try:
        from src.concierge.agent_pool import get_completed_history
        history = get_completed_history(limit=3)
        if history:
            lines.append("最近完成:")
            for h in history:
                outcome_icon = "✓" if h.outcome == "success" else "✗"
                lines.append(f"  {outcome_icon} {h.label}")
    except Exception:
        pass

    lines.append("")
    lines.append("Hint: 'status' | 'slot-name task' | 'stop'")

    return "\n".join(lines)


def format_slot_question(candidates: list[Any]) -> str:
    """Format the 'which slot?' question when routing is ambiguous."""
    lines = ["你想操作哪个号？"]
    for s in candidates:
        status = "空闲" if s.is_free else s.status_text
        lines.append(f"  · {s.label}（{status}）")
    return "\n".join(lines)
