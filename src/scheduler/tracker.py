"""Progress tracker for multi-step material farming tasks.

Tracks progress through scheduled stages, accounting for daily sanity resets.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks progress of a material farming plan across multiple days."""

    def __init__(self) -> None:
        self._current_plan: dict[str, Any] = {}
        self._completed_runs: dict[str, int] = {}  # stage -> runs completed
        self._total_runs: dict[str, int] = {}       # stage -> total runs needed
        self._current_day: int = 0
        self._daily_sanity_used: int = 0

    def start_plan(self, plan: dict) -> None:
        """Load a scheduler plan for tracking."""
        self._current_plan = plan
        self._completed_runs.clear()
        self._total_runs.clear()
        self._current_day = 0
        self._daily_sanity_used = 0

        for stage_entry in plan.get("plan", []):
            stage = stage_entry["stage"]
            self._total_runs[stage] = stage_entry.get("runs", 0)

        operator = plan.get("operator", "?")
        elite = plan.get("elite_level", "?")
        logger.info("Tracking plan: %s E%s, %d stages, %d total sanity",
                     operator, elite, len(self._total_runs), plan.get("total_sanity", 0))

    def mark_runs(self, stage: str, runs: int) -> None:
        """Mark runs completed for a stage."""
        current = self._completed_runs.get(stage, 0)
        self._completed_runs[stage] = current + runs

    def mark_stage_complete(self, stage: str) -> None:
        """Mark all runs for a stage as complete."""
        self._completed_runs[stage] = self._total_runs.get(stage, 0)

    @property
    def progress(self) -> float:
        """Overall progress 0.0-1.0."""
        if not self._total_runs:
            return 0.0
        total = sum(self._total_runs.values())
        if total == 0:
            return 1.0
        completed = sum(min(self._completed_runs.get(s, 0), self._total_runs[s])
                        for s in self._total_runs)
        return completed / total

    @property
    def is_complete(self) -> bool:
        return self.progress >= 1.0

    def remaining(self) -> list[dict[str, Any]]:
        """Return stages with remaining runs."""
        result: list[dict[str, Any]] = []
        for stage_entry in self._current_plan.get("plan", []):
            stage = stage_entry["stage"]
            total = self._total_runs.get(stage, 0)
            completed = self._completed_runs.get(stage, 0)
            remaining = total - completed
            if remaining > 0:
                result.append({
                    "stage": stage,
                    "runs_remaining": remaining,
                    "runs_total": total,
                    "materials": stage_entry.get("materials", []),
                })
        return result

    def current_stage(self) -> dict[str, Any] | None:
        """Get the next stage that needs farming."""
        remaining = self.remaining()
        return remaining[0] if remaining else None

    def summary(self) -> str:
        """One-line progress summary."""
        pct = int(self.progress * 100)
        r = self.remaining()
        if not r:
            return "All stages complete."
        next_stage = r[0]
        return (
            f"Progress: {pct}% — "
            f"Next: {next_stage['stage']} "
            f"({next_stage['runs_remaining']}/{next_stage['runs_total']} runs remaining)"
        )
