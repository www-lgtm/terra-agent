"""Abstract base classes for intelligence tools.

Intelligence tools consume knowledge (static game data), skills (how-to
procedures), and memories (experience/pitfalls) to make informed
recommendations to the LLM agent.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceContext:
    """Read-only context passed to intelligence tools on each query.

    Bundles all three data layers (knowledge, skills, memories) alongside
    the current screen state so tools can make context-aware recommendations.
    """

    game: str
    knowledge: Any = None    # KnowledgeBase instance
    skills: list[dict] = field(default_factory=list)    # Matching skills
    memories: list[dict] = field(default_factory=list)  # Relevant memories
    screen_dhash: str | None = None
    ocr_texts: list[str] = field(default_factory=list)


@dataclass
class IntelligenceResult:
    """Structured output from an intelligence tool.

    recommendation is a human-readable suggestion injected into the LLM's
    conversation context.  suggested_actions are optional structured action
    hints that the agent loop could use directly (future use).
    """

    recommendation: str
    confidence: float = 1.0       # 0.0 — 1.0
    source: str = ""              # "knowledge" | "skills" | "memories" | "hybrid"
    suggested_actions: list[dict] = field(default_factory=list)


class IntelligenceTool(ABC):
    """Base class for game-specific smart tools.

    Each tool is responsible for a specific domain (e.g. base scheduling,
    recruitment optimization, material planning) and declares whether it
    can handle a given user task via can_handle().
    """

    @abstractmethod
    def analyze(self, ctx: IntelligenceContext, task: str) -> IntelligenceResult | None:
        """Analyze the current context and task, returning a recommendation.

        Returns None if the tool has nothing useful to contribute.
        """
        ...

    @abstractmethod
    def can_handle(self, task: str) -> bool:
        """Check if this tool can handle the given task description."""
        ...


class IntelligenceRegistry:
    """Registry of IntelligenceTool instances, queried by the agent loop."""

    def __init__(self, game: str = "arknights") -> None:
        self.game = game
        self._tools: list[IntelligenceTool] = []

    def register(self, tool: IntelligenceTool) -> None:
        """Register a tool instance."""
        self._tools.append(tool)
        logger.info("Registered intelligence tool: %s", type(tool).__name__)

    def query(self, ctx: IntelligenceContext, task: str) -> list[IntelligenceResult]:
        """Query all tools that can handle the task.

        Returns a list of IntelligenceResult objects from tools that
        both can_handle() the task and produce a non-None analyze() result.
        """
        results: list[IntelligenceResult] = []
        for tool in self._tools:
            if not tool.can_handle(task):
                continue
            try:
                result = tool.analyze(ctx, task)
                if result is not None:
                    results.append(result)
            except Exception:
                logger.warning(
                    "Intelligence tool %s failed", type(tool).__name__, exc_info=True
                )
        return results


# Per-game singleton registries (lazily populated)
_registries: dict[str, IntelligenceRegistry] = {}


def get_intelligence_registry(game: str = "arknights") -> IntelligenceRegistry:
    """Get or create the IntelligenceRegistry for a game."""
    if game not in _registries:
        _registries[game] = IntelligenceRegistry(game)
    return _registries[game]
