"""Intelligence module — game-specific smart tools.

These tools consume knowledge (static game data), skills (action procedures),
and memories (experience/pitfalls) to make informed recommendations to the
LLM agent.

Architecture:
    IntelligenceTool (ABC)
        ├── analyze(ctx, task) -> IntelligenceResult
        └── can_handle(task) -> bool

    IntelligenceRegistry
        ├── register(tool)
        └── query(ctx, task) -> list[IntelligenceResult]

Integration point:
    In TerraAgent.run(), after routing but before the first LLM call,
    intelligence_registry.query() is called and results are injected
    as [智能建议] user messages.
"""

from src.intelligence.base import (
    IntelligenceContext,
    IntelligenceResult,
    IntelligenceTool,
    IntelligenceRegistry,
)

__all__ = [
    "IntelligenceContext",
    "IntelligenceResult",
    "IntelligenceTool",
    "IntelligenceRegistry",
]


def register_default_tools(game: str = "arknights") -> IntelligenceRegistry:
    """Register all built-in intelligence tools for a game.

    Phase 4: delegates game-specific intelligence registration to GamePlugin.
    Cross-game tools (ChecklistGuard, SkillStalenessCheck) are registered here.

    Called once during TerraAgent initialization.  Idempotent.
    """
    from src.intelligence.base import get_intelligence_registry

    registry = get_intelligence_registry(game)

    # Guard: only register once per game per process
    if hasattr(registry, "_defaults_registered"):
        return registry
    registry._defaults_registered = True

    # Phase 2 tools — cross-game (registered for ALL games)
    from src.intelligence.checklist_guard import ChecklistGuard
    from src.intelligence.skill_staleness import SkillStalenessCheck

    registry.register(ChecklistGuard())
    registry.register(SkillStalenessCheck())

    # Phase 3+ tools — game-specific (delegated to GamePlugin)
    from src.games.registry import get_game_registry
    get_game_registry().activate_game_tools(game)

    return registry
