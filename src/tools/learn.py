"""Positive action-pattern memory tool — learn_action_pattern (Phase 1).

Unlike remember() which captures pitfalls from failure, learn_action_pattern()
captures positive, actionable mappings from screen state to action.  These
are stored as memories with source='action_pattern' and shown to the LLM as
[经验模式 — 正向参考] (distinct from [关联记忆 — 注意避坑]).

The LLM should call this when it discovers a reliable screen→action mapping
that it would want to reuse in future tasks.
"""

from __future__ import annotations

import json
import logging
import time as _time
from datetime import datetime, timezone
from pathlib import Path

from config.settings import config
from src.tools.registry import registry, ToolOutput

logger = logging.getLogger(__name__)

_MEMORIES_DIR = Path(config.DATA_DIR) / "memories"


def _get_current_game() -> str:
    """Read the current game from the active agent's context.

    When called from a background thread (no agent context), falls back
    to the GameRegistry's default game via the DI container.
    """
    import threading
    ctx = getattr(threading.current_thread(), "_terra_agent_ctx", None)
    if ctx is not None:
        from_ctx = getattr(ctx.state, "game", None)
        if from_ctx:
            return from_ctx
    try:
        from src.container import get_container
        return get_container().game_registry.default_game
    except Exception:
        return "arknights"  # ultimate fallback


def _get_current_screen_dhash() -> str | None:
    """Read the current screen's dHash from the active agent's context."""
    import threading
    ctx = getattr(threading.current_thread(), "_terra_agent_ctx", None)
    if ctx is None:
        return None
    return getattr(ctx.state, "last_injected_dhash", None)


def learn_action_pattern_tool(
    trigger_screen: str,
    action: str,
    expected_result: str = "",
    tags: str = "",
    game: str = "",
) -> ToolOutput:
    """Save a positive action pattern for future reference.

    Call this when you discover a reliable mapping from screen state to action.
    This is different from remember() — use remember() for pitfalls/warnings,
    use this for "when you see X, do Y" positive mappings.

    The pattern will be auto-injected when the agent encounters a similar screen.

    Args:
        trigger_screen: Describe the screen/UI context. Include visible OCR
                        texts, button names, or visual elements that identify
                        this screen state. E.g. "快捷编队界面，显示'开始'和'行动'
                        以及助战干员名称"
        action: What action to take. Be specific — include tool name, target,
                and any important parameters. E.g. "adb_tap_position(x=0.86, y=0.69)
                点击'开始行动'按钮（两字被OCR分割时用坐标代替文字匹配）"
        expected_result: What should happen after the action. E.g. "进入编队确认界面
                         或直接开始作战"
        tags: Comma-separated keywords (e.g. "快捷编队, 开始行动, 作战")
        game: Game ID. Auto-detected from agent context; pass explicitly when
              calling from background threads.
    """
    if not trigger_screen.strip():
        return ToolOutput(text=json.dumps(
            {"success": False, "error": "trigger_screen is required"},
            ensure_ascii=False,
        ))
    if not action.strip():
        return ToolOutput(text=json.dumps(
            {"success": False, "error": "action is required"},
            ensure_ascii=False,
        ))

    resolved_game = game if game else _get_current_game()
    screen_hash = _get_current_screen_dhash()

    # Build structured body
    body_parts = [
        f"【画面】{trigger_screen.strip()}",
        f"【操作】{action.strip()}",
    ]
    if expected_result.strip():
        body_parts.append(f"【结果】{expected_result.strip()}")
    body = "\n".join(body_parts)

    # Generate unique name
    name = f"ap{int(_time.time() * 1_000_000)}"

    # Build YAML frontmatter (compute before index so we can write file if new)
    tags_clean = tags.strip() if tags else ""
    now = datetime.now(tz=timezone.utc).isoformat()
    yaml_lines = [
        f"game: {resolved_game}",
        f"tags: [{', '.join(t.strip() for t in tags_clean.split(',') if t.strip())}]",
        "source: action_pattern",
        f"created: {now}",
    ]
    if screen_hash:
        yaml_lines.append(f"screen_hash: {screen_hash}")

    # Index first — _index_memory does dedup and may return an existing ID.
    # Only write the .md file if the memory is genuinely new.
    from src.tools.remember import _index_memory
    idx_id = _index_memory(name, resolved_game, tags_clean, body, screen_hash, source="action_pattern")
    if idx_id is None:
        return ToolOutput(text=json.dumps({"success": False, "error": "Failed to index action pattern"}, ensure_ascii=False))

    from src.memory.memory_db import memory_db as _mdb
    row = _mdb.conn.execute(
        "SELECT id, name FROM memories_data WHERE id = ?", (idx_id,)
    ).fetchone()
    is_new = row and row["name"] == name

    if is_new:
        mem_dir = _MEMORIES_DIR / resolved_game
        mem_dir.mkdir(parents=True, exist_ok=True)
        content = "---\n" + "\n".join(yaml_lines) + "\n---\n\n" + body
        file_path = mem_dir / f"{name}.md"
        file_path.write_text(content, encoding="utf-8")
        logger.info("Action pattern saved: %s/%s.md (id=%s)", resolved_game, name, idx_id)
    else:
        logger.info("Action pattern deduped to existing: id=%s", idx_id)

    return ToolOutput(text=json.dumps({
        "success": True,
        "name": name,
        "game": resolved_game,
        "type": "action_pattern",
        "message": "正向行动模式已保存，未来遇到相似画面时会自动推荐。",
    }, ensure_ascii=False))


# Register tool
registry.register(
    name="learn_action_pattern",
    description=(
        "学习一个正向行动模式：'当看到[画面特征]时，做[具体操作]'。\n"
        "与 remember() 不同：remember 记录的是坑点和教训（不要做什么），"
        "learn_action_pattern 记录的是正面的屏幕→操作映射（该做什么）。\n"
        "使用时机：发现了一个可靠的操作模式，希望在以后遇到相同画面时自动推荐。\n"
        "参数示例：trigger_screen='快捷编队界面，OCR显示开始/行动分开', "
        "action='adb_tap_position(x=0.86, y=0.69)直接点击坐标', "
        "expected_result='进入作战准备界面'"
    ),
    parameters={
        "type": "object",
        "properties": {
            "trigger_screen": {
                "type": "string",
                "description": "画面特征描述，包含可见的OCR文字、按钮、布局等识别信息",
            },
            "action": {
                "type": "string",
                "description": "要执行的具体操作，包括工具名称、目标、参数",
            },
            "expected_result": {
                "type": "string",
                "description": "操作后的预期结果（可选，帮助判断模式是否仍然有效）",
            },
            "tags": {
                "type": "string",
                "description": "逗号分隔的关键词（如 '快捷编队, 开始行动, 作战'）",
            },
            "game": {
                "type": "string",
                "description": "Game ID. Auto-detected; only pass when you're sure the auto-detection is wrong.",
            },
        },
        "required": ["trigger_screen", "action"],
    },
    handler=learn_action_pattern_tool,
)
