"""Knowledge query tool: LLM queries game data tables by filter or text search.

Game-aware — the `game` parameter defaults to the current thread's active game
(set by TerraAgent at task start).  The tool description is dynamically built
from all registered game manifests.

Self-registers with the tool registry at import time.
"""

from __future__ import annotations

import json
import logging

from src.knowledge import get_knowledge_base
from src.tools.registry import registry, ToolOutput, get_current_game

logger = logging.getLogger(__name__)


def _build_knowledge_description(game: str | None = None) -> str:
    """Build the knowledge_query tool description dynamically.

    Lists all registered games and their available tables.  Called lazily
    when tool definitions are built for the LLM.
    """
    from src.games.registry import get_game_registry

    table_hint = get_game_registry().build_knowledge_tool_description()

    base = (
        "查询游戏数据表。必须提供filters或query参数，不允许全表查询。\n"
        "game参数可指定游戏（默认当前活跃游戏）。\n"
    )

    if table_hint:
        base += table_hint + "\n"

    base += (
        "示例：knowledge_query(table='stages', query='GT-6') → 查关卡信息\n"
        "示例：knowledge_query(table='recruit_tags', filters={'rarity': 5}) → 五星公招干员\n"
        "示例：knowledge_query(game='reverse1999', table='characters', query='六星') → 跨游戏查询"
    )
    return base


def knowledge_query(
    table: str,
    game: str = "",
    filters: dict | None = None,
    query: str | None = None,
    limit: int = 20,
) -> ToolOutput:
    """Query game knowledge base for a specific game data table.

    MUST provide either 'filters' or 'query'. Full table scans are blocked.
    Default limit is 20, max 50.

    Args:
        table: Table name (see tool description for available tables per game)
        game: Game ID (default: current active game from agent context)
        filters: Key-value pairs to match, e.g. {"rarity": 5, "facility": "Trading"}
        query: Text search across name/description/tags fields
        limit: Max results (default 20)
    """
    kb = get_knowledge_base()

    # Resolve game: explicit param > thread-local context > default
    resolved_game = game if game else get_current_game()

    try:
        results = kb.query(resolved_game, table, filters=filters, query=query, limit=limit)
    except ValueError as e:
        return ToolOutput(text=json.dumps({
            "success": False,
            "error": str(e),
            "hint": "Provide 'filters' (e.g. {\"rarity\": 5}) or 'query' (e.g. \"贸易\") to narrow results.",
        }, ensure_ascii=False))
    except Exception as e:
        logger.error("knowledge_query failed: %s", e)
        return ToolOutput(text=json.dumps({"success": False, "error": str(e)}, ensure_ascii=False))

    summary = {
        "success": True,
        "game": resolved_game,
        "table": table,
        "count": len(results),
        "results": results,
    }

    if len(results) > 5 and results:
        summary["fields"] = list(results[0].keys())

    return ToolOutput(text=json.dumps(summary, ensure_ascii=False))


# Register with dynamic description (lazily resolved at tool listing time)
registry.register(
    name="knowledge_query",
    description="",  # Placeholder — description_fn provides the real one
    parameters={
        "type": "object",
        "properties": {
            "game": {
                "type": "string",
                "description": (
                    "Game ID to query. Default: current active game. "
                    "Set explicitly to query another game's data."
                ),
            },
            "table": {
                "type": "string",
                "description": "Table name to query (see tool description for available tables)",
            },
            "filters": {
                "type": "object",
                "description": "Key-value filters (e.g. {'rarity': 5, 'facility': 'Trading'})",
            },
            "query": {
                "type": "string",
                "description": "Text search across name/description/tags fields",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 20, max 50)",
            },
        },
        "required": ["table"],
    },
    handler=knowledge_query,
    description_fn=_build_knowledge_description,
)
