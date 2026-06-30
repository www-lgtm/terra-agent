"""Game plugin registry — global singleton for multi-game support.

Replaces hardcoded keyword lists in router.py and config/prompts.py.
The agent loop queries this registry for game detection, task classification,
system prompt building, and tool registration.
"""

from __future__ import annotations

import logging
from typing import Any

from src.games.plugin import GamePlugin, GameManifest

logger = logging.getLogger(__name__)


class GameRegistry:
    """Thread-safe registry of GamePlugin instances.

    Plugins self-register at module import time.  The agent loop queries
    this registry at task start to detect the game and activate its tools.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, GamePlugin] = {}   # game_id → plugin
        self._default: str = "arknights"

    # ── Registration ──────────────────────────────────────────────

    def register(self, plugin: GamePlugin) -> None:
        """Register a game plugin.  Idempotent — re-registering updates."""
        game_id = plugin.manifest.id
        if game_id in self._plugins:
            logger.debug("Game plugin '%s' already registered — updating", game_id)
        self._plugins[game_id] = plugin
        logger.info("Registered game plugin: %s (%s)", plugin.manifest.name, game_id)

    # ── Lookup ────────────────────────────────────────────────────

    def get(self, game_id: str) -> GamePlugin | None:
        """Get a plugin by game ID."""
        return self._plugins.get(game_id)

    def get_default(self) -> GamePlugin | None:
        """Get the default game plugin. Falls back to first registered, then None."""
        plugin = self._plugins.get(self._default)
        if plugin is not None:
            return plugin
        values = list(self._plugins.values())
        return values[0] if values else None

    def list_all(self) -> list[GamePlugin]:
        """Return all registered plugins."""
        return list(self._plugins.values())

    def get_ids(self) -> list[str]:
        """Return all registered game IDs."""
        return list(self._plugins.keys())

    def get_game_name(self, game_id: str | None = None) -> str:
        """Get the display name for a game ID from its registered manifest.

        Returns the manifest.name if found, otherwise returns the game_id as-is.
        Falls back to the default game if game_id is None.
        """
        gid = game_id or self._default
        plugin = self._plugins.get(gid)
        if plugin is not None:
            return plugin.manifest.name
        return gid

    @property
    def default_game(self) -> str:
        return self._default

    # ── Game detection ────────────────────────────────────────────

    def detect_game(self, text: str, hint: str | None = None) -> str:
        """Detect which game a user message targets.

        Scores each registered game by keyword hit count, returns the
        game_id with the most matches.  Falls back to default.

        When `hint` is provided (e.g. from Concierge delegation) and no
        keywords match any game (all score 0), the hint is trusted.
        This prevents game-switching when the Concierge already determined
        the correct game but the task text has been normalized (e.g.
        "完成1999日常任务" → "完成日常任务" without the game keyword).
        """
        text_lower = text.lower()
        best_game = self._default
        best_score = 0

        for plugin in self._plugins.values():
            score = sum(1 for kw in plugin.manifest.keywords if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                best_game = plugin.manifest.id

        # When detection is ambiguous (no keywords matched) and the caller
        # already knows the game, trust the hint instead of falling back to
        # the default.  This is critical for Concierge→Agent game routing.
        if best_score == 0 and hint and hint != self._default:
            if hint in self._plugins:
                return hint

        return best_game

    # ── Delegated routing ─────────────────────────────────────────

    def classify_task(self, text: str, game_id: str | None = None) -> str:
        """Classify a user message into a task type."""
        plugin = self.get(game_id) if game_id else self.get_default()
        if plugin is None:
            return "unknown"
        return plugin.classify_task(text)

    def get_task_priority(self, text: str, game_id: str | None = None) -> int:
        """Get task priority from the game plugin."""
        plugin = self.get(game_id) if game_id else self.get_default()
        if plugin is None:
            return 5
        task_type = plugin.classify_task(text)
        return plugin.get_task_priority(task_type)

    def classify_schedule_intent(self, text: str, game_id: str | None = None) -> str:
        """Classify a message into a schedule management intent."""
        plugin = self.get(game_id) if game_id else self.get_default()
        if plugin is None:
            return ""
        return plugin.classify_schedule_intent(text)

    # ── System prompt ─────────────────────────────────────────────

    def get_system_prompt_append(self, game_id: str | None = None) -> str:
        """Get game-specific system prompt additions."""
        plugin = self.get(game_id) if game_id else self.get_default()
        if plugin is None:
            return ""
        return plugin.get_system_prompt_append()

    # ── Safety ────────────────────────────────────────────────────

    def get_dangerous_keywords(self, game_id: str | None = None) -> list[str]:
        plugin = self.get(game_id) if game_id else self.get_default()
        if plugin is None:
            return []
        return list(plugin.manifest.dangerous_keywords)

    def get_safe_compound_terms(self, game_id: str | None = None) -> list[str]:
        plugin = self.get(game_id) if game_id else self.get_default()
        if plugin is None:
            return []
        return list(plugin.manifest.safe_compound_terms)

    def get_confirmation_keywords(self, game_id: str | None = None) -> list[str]:
        plugin = self.get(game_id) if game_id else self.get_default()
        if plugin is None:
            return []
        return list(plugin.manifest.require_confirmation_keywords)

    # ── Knowledge tables ──────────────────────────────────────────

    def build_knowledge_tool_description(self) -> str:
        """Build the dynamic knowledge_query tool description with all games' tables."""
        lines: list[str] = []
        for plugin in self._plugins.values():
            hint = plugin.build_knowledge_tool_hint()
            if hint:
                lines.append(hint)
        if not lines:
            return ""
        return "可用数据表：\n" + "\n".join(lines)

    # ── Tool registration ─────────────────────────────────────────

    def activate_game_tools(self, game_id: str) -> None:
        """Register game-specific intelligence tools for the given game.

        Called when the agent loop detects the active game.
        Idempotent — subsequent calls for the same game are no-ops.
        """
        plugin = self.get(game_id)
        if plugin is None:
            logger.warning("Cannot activate tools for unknown game: %s", game_id)
            return

        try:
            plugin.register_intelligence_tools()
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", f"Intelligence tool registration failed for {game_id}")

        try:
            plugin.register_game_tools()
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", f"Game tool registration failed for {game_id}")


# ── Singleton ─────────────────────────────────────────────────────

_game_registry: GameRegistry | None = None


def get_game_registry() -> GameRegistry:
    """Get or lazily create the global game registry."""
    global _game_registry
    if _game_registry is None:
        _game_registry = GameRegistry()
        # Auto-register built-in games
        _register_builtin_games(_game_registry)
    return _game_registry


def _register_builtin_games(registry: GameRegistry) -> None:
    """Register all built-in game plugins."""
    try:
        from src.games.arknights.plugin import ArknightsGamePlugin
        registry.register(ArknightsGamePlugin())
    except ImportError as e:
        logger.warning("Failed to register Arknights plugin: %s", e)
    try:
        from src.games.reverse1999.plugin import Reverse1999GamePlugin
        registry.register(Reverse1999GamePlugin())
    except ImportError as e:
        logger.warning("Failed to register Reverse1999 plugin: %s", e)
    try:
        from src.games.lifemaker.plugin import LifemakerGamePlugin
        registry.register(LifemakerGamePlugin())
    except ImportError as e:
        logger.warning("Failed to register Lifemaker plugin: %s", e)
