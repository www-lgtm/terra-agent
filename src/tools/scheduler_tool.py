"""Scheduler tool: generates farming plans for operator promotion.

Self-registers with the tool registry at import time.
"""

from __future__ import annotations

import json
import logging

from src.tools.registry import registry, ToolOutput

logger = logging.getLogger(__name__)


def plan_promotion(operator: str, elite_level: int = 2, current_sanity: int = 0) -> ToolOutput:
    """Generate a farming plan for promoting an operator.

    Args:
        operator: Operator name in Chinese (e.g. "银灰", "能天使").
        elite_level: Target elite level (1=E1, 2=E2).
        current_sanity: Current sanity available (0 = assume full).

    Returns the full material tree, stage plan, and estimated time.
    """
    from src.intelligence.arknights.material_scheduler import SchedulerEngine

    engine = SchedulerEngine()
    plan = engine.plan(operator, elite_level, current_sanity)

    if plan.get("error"):
        return ToolOutput(text=json.dumps({"success": False, "message": plan["error"]}, ensure_ascii=False))

    return ToolOutput(text=json.dumps({
        "success": True,
        "operator": operator,
        "elite_level": elite_level,
        "materials_needed": plan.get("material_tree", {}),
        "total_sanity": plan.get("total_sanity", 0),
        "total_runs": plan.get("total_runs", 0),
        "estimated_days": plan.get("estimated_days", 0),
        "estimated_hours": plan.get("estimated_hours", 0),
        "plan": plan.get("plan", []),
    }, ensure_ascii=False))


registry.register(
    name="plan_promotion",
    description="Generate a material farming plan for promoting an operator to target elite level. Returns material tree, stage plan with run counts, and sanity/time estimates.",
    parameters={
        "type": "object",
        "properties": {
            "operator": {"type": "string", "description": "Operator name in Chinese (e.g. '银灰', '能天使')"},
            "elite_level": {"type": "integer", "description": "Target elite level: 1 for E1, 2 for E2. Default: 2"},
            "current_sanity": {"type": "integer", "description": "Current sanity available. Default: 0 (assume full)"},
        },
        "required": ["operator"],
    },
    handler=plan_promotion,
)
