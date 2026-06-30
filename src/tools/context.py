"""ToolContext — explicit per-task context for tool handlers.

Replaces the fragile thread-local pattern where tools read state from
threading.current_thread()._terra_agent_ctx and threading.local().

Every tool dispatch carries an optional ToolContext.  When present, tools
use it directly.  When absent (legacy or background calls), tools fall
back to the old thread-local mechanism for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolContext:
    """Per-task context passed explicitly to tool handlers via dispatch().

    All fields are optional — tools should handle None gracefully.
    """

    # ── Core identity ──
    game: str = "arknights"
    device_serial: str = ""

    # ── User interaction ──
    ask_fn: Callable[[str], str] | None = None   # CLI mode: synchronous ask
    notify_fn: Callable[..., None] | None = None  # WeChat mode: async notify

    # ── Screen state (snapshot at dispatch time) ──
    ocr_texts: list[str] = field(default_factory=list)
    screen_dhash: str | None = None
    screen_hash: str | None = None

    # ── AgentState reference (for injection tracking, etc.) ──
    agent_state: Any = None  # AgentState instance (avoid circular import)

    # ── Agent reference (for ask_user reply delivery) ──
    agent_ref: Any = None    # TerraAgent instance
