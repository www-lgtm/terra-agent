"""UserGameContext — cross-message game context persistence.

Handles "switch to X" commands so users don't need to specify the game every time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _get_game_name(game_id: str) -> str:
    """Resolve game ID to display name via GameRegistry. Lazy import to avoid cycles."""
    from src.games.registry import get_game_registry
    return get_game_registry().get_game_name(game_id)


@dataclass
class UserGameContext:
    """Per-user context that persists across messages."""

    active_game: str = "arknights"
    active_slot_label: str = ""       # last slot the user interacted with
    last_mentioned_game: str = ""     # game mentioned in the last message

    def _game_keywords(self) -> dict[str, list[str]]:
        """Get game detection keywords from GameRegistry."""
        from src.games.registry import get_game_registry
        result: dict[str, list[str]] = {}
        for plugin in get_game_registry().list_all():
            result[plugin.manifest.id] = list(plugin.manifest.keywords)
        return result

    def handle_switch(self, text: str) -> str | None:
        """Detect game from message keywords. Updates active_game if a new game
        is mentioned. Returns switch confirmation if the game changed, else None.

        Does NOT consume the task — the message continues to task routing.
        """
        text_lower = text.strip().lower()

        detected = None
        best_score = 0
        for game_id, kws in self._game_keywords().items():
            score = sum(1 for kw in kws if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                detected = game_id

        if detected is None:
            return None

        # Update last_mentioned even if already active (for routing context)
        self.last_mentioned_game = detected

        if detected != self.active_game:
            old = self.active_game
            self.active_game = detected
            return f"好的，从{_get_game_name(old)}切换到{_get_game_name(detected)}。"
        return None

    def detect_game_in_text(self, text: str) -> str | None:
        """Detect game mentioned in text. Does NOT change active_game."""
        text_lower = text.lower()
        for game_id, kws in self._game_keywords().items():
            for kw in kws:
                if kw.lower() in text_lower:
                    self.last_mentioned_game = game_id
                    return game_id
        return None

    def effective_game(self, text: str) -> str:
        """Determine which game this message targets.

        Priority: explicit mention > active_game.
        """
        detected = self.detect_game_in_text(text)
        return detected if detected else self.active_game
