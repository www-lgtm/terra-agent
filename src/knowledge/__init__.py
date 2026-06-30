"""Game knowledge base — fast JSON lookups for LLM decision making.

Layer 1: Screen/UI knowledge (game primers, injected into prompts)
Layer 2: Game data (operators, stages, materials — queried on demand via tools)
Layer 3: Learned memory (existing memories/ + skills/ — dynamic injection)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KNOWLEDGE_DIR = Path(__file__).parent


class KnowledgeBase:
    """In-memory game knowledge index.

    Loads JSON files on demand per game. All lookups are O(1) dict access.
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    def query(
        self,
        game: str,
        table: str,
        *,
        filters: dict[str, Any] | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Query a game data table with mandatory filters or text query.

        Args:
            game: Game namespace (e.g. 'arknights')
            table: Table name (e.g. 'operators', 'stages', 'recruit_tags')
            filters: Dict of field→value to match (e.g. {'rarity': 6})
            query_str: Text search across name fields
            limit: Max results (default 20, enforced)

        Returns:
            List of matching records
        """
        if not filters and not query:
            raise ValueError(
                "knowledge_query requires 'filters' or 'query' parameter. "
                "Full table scan is not allowed — it would consume too many tokens."
            )

        data = self._load_table(game, table)
        limit = min(limit, 50)  # hard cap

        results: list[dict] = []
        for item in data:
            if filters and not self._match_filters(item, filters):
                continue
            if query and not self._match_query(item, query):
                continue
            results.append(item)
            if len(results) >= limit:
                break

        return results

    def get(self, game: str, table: str, key: str, key_field: str = "name") -> dict | None:
        """Get a single record by key field."""
        data = self._load_table(game, table)
        for item in data:
            if item.get(key_field) == key:
                return item
        return None

    def _load_table(self, game: str, table: str) -> list[dict]:
        """Load a game data table from disk, caching in memory."""
        cache_key = f"{game}/{table}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        file_path = _KNOWLEDGE_DIR / game / f"{table}.json"
        if not file_path.exists():
            logger.warning("Knowledge table not found: %s", file_path)
            self._cache[cache_key] = []
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.error("Failed to load knowledge table %s: %s", file_path, e)
            self._cache[cache_key] = []
            return []

        # Unwrap top-level container fields if present
        if isinstance(raw, dict):
            # Common patterns: {"operators": [...]}, {"materials": {...}}, {"stages": [...]}
            for key in ("operators", "skills", "stages", "main_story", "supply", "event"):
                if key in raw and isinstance(raw[key], list):
                    data = raw[key]
                    break
            else:
                # For materials: {"materials": {name: info, ...}}
                if "materials" in raw and isinstance(raw["materials"], dict):
                    data = [
                        {"name": k, **v}
                        for k, v in raw["materials"].items()
                    ]
                elif "tag_index" in raw:
                    # recruit_tags: return the operators directly
                    data = raw.get("operators", [])
                elif "skills_by_facility" in raw:
                    # base_skills: return flattened skills
                    data = raw.get("skills", [])
                elif "schedule" in raw:
                    data = [
                        {"day_name": k, **v}
                        for k, v in raw["schedule"].items()
                    ]
                else:
                    data = [raw]
        elif isinstance(raw, list):
            data = raw
        else:
            data = [raw]

        self._cache[cache_key] = data
        return data

    @staticmethod
    def _match_filters(item: dict, filters: dict) -> bool:
        for key, value in filters.items():
            item_val = item.get(key)
            if item_val is None:
                return False
            if isinstance(value, list):
                if not any(v in (item_val if isinstance(item_val, list) else [item_val]) for v in value):
                    return False
            elif item_val != value:
                return False
        return True

    @staticmethod
    def _match_query(item: dict, query: str) -> bool:
        q = query.lower()
        # Search in name, description, tags, day_name, chips fields
        for field in ("name", "description", "tags", "code", "desc", "day_name", "chips"):
            val = item.get(field)
            if val is None:
                continue
            if isinstance(val, list):
                if any(q in str(v).lower() for v in val):
                    return True
            elif q in str(val).lower():
                return True
        return False


# Singleton
knowledge_base = KnowledgeBase()


def get_knowledge_base() -> KnowledgeBase:
    return knowledge_base
