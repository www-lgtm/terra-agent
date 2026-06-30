"""Arknights material DAG scheduler — operator promotion → farming plan.

Phase 2: full DAG with stage scheduling, sanity budget, and daily chip schedule.
Phase 1 was simple expected-runs estimation.

Moved from src/scheduler/engine.py (Phase B) — this module is Arknights-specific,
containing hardcoded stage drops, sanity costs, and chip schedules.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Sanity constants
SANITY_CAP = 135
SANITY_PER_HOUR = 6  # Natural recovery

# Stage data: {stage: {material: drops_per_run}}
# Phase 2: replace with Penguin Logistics API data
STAGE_DROPS: dict[str, dict[str, float]] = {
    "1-7": {"固源岩": 1.2},
    "S3-4": {"装置": 0.48},
    "S4-1": {"异铁组": 0.45},
    "S5-9": {"异铁": 0.52, "糖": 0.35},
    "GT-6": {"糖": 0.52},
    "CE-5": {"龙门币": 7500},
    "PR-C-2": {"近卫芯片组": 0.45, "特种芯片组": 0.45},
    "PR-A-2": {"狙击芯片组": 0.45},
}

STAGE_SANITY: dict[str, int] = {
    "1-7": 6,
    "S3-4": 18,
    "S4-1": 21,
    "S5-9": 18,
    "GT-6": 15,
    "CE-5": 30,
    "PR-C-2": 36,
    "PR-A-2": 36,
}

# Chip stage schedule: day_of_week (0=Mon) -> open stages
CHIP_SCHEDULE: dict[int, list[str]] = {
    0: [],
    1: ["PR-C-2"],   # Tue: Guard/Specialist
    2: ["PR-A-2"],   # Wed: Sniper/Caster
    3: ["PR-C-2"],   # Thu: Guard/Specialist
    4: ["PR-A-2"],   # Fri: Sniper/Caster
    5: ["PR-C-2", "PR-A-2"],  # Sat
    6: ["PR-C-2", "PR-A-2"],  # Sun: all open
}


class SchedulerEngine:
    """Schedules material farming based on promotion needs with DAG + sanity budget."""

    def plan(self, operator: str, elite_level: int = 2,
             current_sanity: int = 0) -> dict[str, Any]:
        """Create a farming plan for promoting an operator.

        Returns a plan with:
        - material_tree: full material dependency tree
        - stages: ordered list of stages to farm
        - total_sanity: total sanity needed
        - estimated_hours: time estimate including natural recovery
        - schedule: day-by-day schedule factoring chip availability
        """
        from src.games.arknights.materials import get_craft_requirements
        from src.games.arknights.operators import get_promotion_cost

        direct = get_promotion_cost(operator, elite_level)
        if not direct:
            return {
                "operator": operator,
                "elite_level": elite_level,
                "error": f"No promotion data for {operator} E{elite_level}",
                "plan": [],
            }

        # Expand full material tree
        material_tree: dict[str, int] = {}
        for material, count in direct.items():
            sub = get_craft_requirements(material, count)
            for mat, qty in sub.items():
                material_tree[mat] = material_tree.get(mat, 0) + qty

        # Plan stages for each material
        stage_plan = self._select_stages(material_tree)

        # Calculate sanity
        total_sanity = sum(s["sanity"] for s in stage_plan)
        total_runs = sum(s["runs"] for s in stage_plan)

        # Day-by-day schedule accounting for chip availability
        import copy
        schedule = self._build_schedule(copy.deepcopy(stage_plan), current_sanity)

        return {
            "operator": operator,
            "elite_level": elite_level,
            "material_tree": material_tree,
            "materials": direct,
            "total_sanity": total_sanity,
            "total_runs": total_runs,
            "estimated_days": schedule["total_days"],
            "estimated_hours": round(total_sanity / SANITY_PER_HOUR, 1),
            "plan": stage_plan,
            "schedule": schedule["days"],
        }

    @staticmethod
    def _select_stages(materials: dict[str, int]) -> list[dict[str, Any]]:
        """Select the best stage for each material.

        Returns list sorted: non-chip stages first, then chip stages.
        """
        stage_runs: dict[str, dict[str, Any]] = {}

        for material, quantity in materials.items():
            best_stage = None
            best_sanity = float("inf")

            for stage, drops in STAGE_DROPS.items():
                if material in drops:
                    rate = drops[material]
                    runs = max(1, int(quantity / rate) + (0 if quantity % rate == 0 else 1))
                    sanity = runs * STAGE_SANITY.get(stage, 18)
                    if sanity < best_sanity:
                        best_sanity = sanity
                        best_stage = {
                            "stage": stage,
                            "runs": runs,
                            "sanity": sanity,
                            "material": material,
                            "quantity": quantity,
                            "drop_rate": rate,
                        }

            if best_stage:
                key = best_stage["stage"]
                if key in stage_runs:
                    existing = stage_runs[key]
                    existing["runs"] += best_stage["runs"]
                    existing["sanity"] += best_stage["sanity"]
                    existing["materials"] = existing.get("materials", []) + [{
                        "name": material, "quantity": quantity,
                    }]
                else:
                    best_stage["materials"] = [{"name": material, "quantity": quantity}]
                    stage_runs[key] = best_stage

        result = sorted(stage_runs.values(), key=lambda s: (
            "PR-" in s["stage"],  # Chip stages last
            s["sanity"],           # Cheapest first within group
        ))
        return result

    def _build_schedule(self, stage_plan: list[dict[str, Any]],
                        current_sanity: int = 0) -> dict[str, Any]:
        """Build a day-by-day schedule accounting for sanity recovery and chip availability."""
        today = datetime.now()
        day_of_week = today.weekday()

        days: list[dict[str, Any]] = []
        remaining_sanity = max(0, SANITY_CAP - current_sanity)
        daily_budget = SANITY_CAP

        non_chip = [s for s in stage_plan if "PR-" not in s["stage"]]
        chip_stages = [s for s in stage_plan if "PR-" in s["stage"]]

        day_offset = 0

        for stage in non_chip:
            while stage["runs"] > 0:
                cost_per_run = STAGE_SANITY.get(stage["stage"], 18)
                affordable = remaining_sanity // cost_per_run
                runs_today = min(stage["runs"], affordable)

                if runs_today > 0:
                    days.append({
                        "day": day_offset,
                        "date": _fmt_date(today, day_offset),
                        "day_of_week": (day_of_week + day_offset) % 7,
                        "stage": stage["stage"],
                        "runs": runs_today,
                        "sanity_cost": runs_today * cost_per_run,
                        "type": "farm",
                    })
                    remaining_sanity -= runs_today * cost_per_run
                    stage["runs"] -= runs_today

                if stage["runs"] > 0:
                    day_offset += 1
                    remaining_sanity = daily_budget

        for stage in chip_stages:
            while stage["runs"] > 0:
                current_dow = (day_of_week + day_offset) % 7
                open_stages = CHIP_SCHEDULE.get(current_dow, [])

                if stage["stage"] in open_stages:
                    cost_per_run = STAGE_SANITY.get(stage["stage"], 36)
                    affordable = daily_budget // cost_per_run
                    runs_today = min(stage["runs"], affordable)

                    if runs_today > 0:
                        days.append({
                            "day": day_offset,
                            "date": _fmt_date(today, day_offset),
                            "day_of_week": current_dow,
                            "stage": stage["stage"],
                            "runs": runs_today,
                            "sanity_cost": runs_today * cost_per_run,
                            "type": "chip",
                        })
                        stage["runs"] -= runs_today

                if stage["runs"] > 0:
                    day_offset += 1

        return {
            "days": days,
            "total_days": day_offset + (1 if days else 0),
        }


def _fmt_date(base: datetime, offset_days: int) -> str:
    """Format a date with offset from base."""
    from datetime import timedelta
    d = base + timedelta(days=offset_days)
    return d.strftime("%m/%d")
