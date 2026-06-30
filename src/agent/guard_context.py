"""GuardContext — consolidated state for all runtime safety guards.

Extracted from TerraAgent instance variables scattered across _reset_trackers().
Concentrates 15+ variables into a single dataclass, reducing state sprawl and
making the guard logic testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GuardContext:
    """Runtime guard state for the agent main loop.

    All fields are reset at the start of each conversation turn (run/respond).
    """

    # ── Repeat guard ──
    last_tool_name: str = ""
    last_tool_input: str = ""
    repeat_count: int = 0

    # ── Burst guard ──
    burst_window: int = 0
    burst_start_iteration: int = 0

    # ── Scroll guard ──
    scroll_count: int = 0
    last_scroll_direction: str = ""

    # ── Resource consumption guard ──
    resource_warning_count: int = 0

    # ── Dark-screen navigation guard ──
    consecutive_nav_failures: int = 0
    last_nav_brightness: float = -1.0

    # ── Popup-stuck detection (Stage 1) ──
    stuck_target: str = ""
    stuck_screen_hash: str = ""
    stuck_count: int = 0

    # ── Stale-screen detection ──
    stale_screen_count: int = 0

    # ── Magnify guard ──
    magnify_streak: int = 0

    # ── Countdown skip ──
    cd_skip_until_iteration: int = 0

    # ── Wait-intent conflict ──
    wait_intent_conflict_streak: int = 0

    # ── Adb_back futility ──
    pending_back_pre_hash: str | None = None

    # ── Task completion guard ──
    task_complete_guard_count: int = 0

    # ── Idle tracking ──
    idle_streak: int = 0

    # ── Recent tool targets (for battle-vs-launch context) ──
    recent_tool_targets: list[str] = field(default_factory=list)

    def reset(self) -> None:
        """Reset all guard state for a new conversation turn."""
        self.last_tool_name = ""
        self.last_tool_input = ""
        self.repeat_count = 0
        self.burst_window = 0
        self.burst_start_iteration = 0
        self.scroll_count = 0
        self.last_scroll_direction = ""
        self.resource_warning_count = 0
        self.consecutive_nav_failures = 0
        self.last_nav_brightness = -1.0
        self.stuck_target = ""
        self.stuck_screen_hash = ""
        self.stuck_count = 0
        self.stale_screen_count = 0
        self.magnify_streak = 0
        self.cd_skip_until_iteration = 0
        self.wait_intent_conflict_streak = 0
        self.pending_back_pre_hash = None
        self.task_complete_guard_count = 0
        self.idle_streak = 0
        self.recent_tool_targets.clear()
