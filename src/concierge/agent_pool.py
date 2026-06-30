"""AgentHandle — lightweight wrapper around TerraAgent with lifecycle metadata.

Provides #N numbering, slot association, and clean cancel/status interfaces.
One AgentHandle per active task; completed handles are retained briefly for history.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentHandle:
    """Wrap a TerraAgent with an incrementing global agent_id and slot reference.

    The agent_id is a monotonically-increasing string ("1", "2", ...) that
    users can reference in commands like "停止#2" or "#1 进度".
    """

    agent_id: str                     # "1", "2", ... — global, never reused
    agent: Any                        # TerraAgent instance
    slot: Any | None = None           # GameSlot | None (None for Phase 1 no-slot mode)
    task_description: str = ""
    game: str = "arknights"
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    outcome: str = ""                 # "success" | "error" | "cancelled" | ""

    @property
    def is_running(self) -> bool:
        return self.agent is not None and getattr(self.agent.state, 'running', False)

    @property
    def status_text(self) -> str:
        """Human-readable one-line status for status panels."""
        state = self.agent.state if self.agent else None
        if state is None:
            return "已完成" if self.outcome else "启动中"
        if not state.running:
            return f"已完成 ({self.outcome})" if self.outcome else "已停止"
        activity = state.current_activity or self.task_description or "执行中"
        iters = f" {state.iteration_count}步" if state.iteration_count else ""
        return f"{activity}{iters}"

    @property
    def label(self) -> str:
        """Display label: '#N [slot_label] task'"""
        slot_part = f"[{self.slot.label}] " if self.slot else ""
        return f"#{self.agent_id} {slot_part}{self.task_description[:40]}"

    def cancel(self) -> None:
        """Gracefully cancel this agent's task."""
        if self.agent is not None:
            state = self.agent.state
            state._pending_cancel = True
            state.inject_message("用户要求停止当前任务。")
            state.running = False

    def mark_completed(self, outcome: str) -> None:
        """Record completion — keeps the handle for history display."""
        self.completed_at = time.time()
        self.outcome = outcome


# ── History ring buffer (keeps last N completed handles for status display) ──

_MAX_HISTORY = 20
_completed_handles: list[AgentHandle] = []
_completed_lock = threading.Lock()


def add_to_history(handle: AgentHandle) -> None:
    """Add a completed AgentHandle to the global history ring buffer."""
    with _completed_lock:
        _completed_handles.append(handle)
        if len(_completed_handles) > _MAX_HISTORY:
            _completed_handles.pop(0)


def get_completed_history(limit: int = 5) -> list[AgentHandle]:
    """Get most recently completed handles for status display."""
    with _completed_lock:
        return list(reversed(_completed_handles[-limit:]))
