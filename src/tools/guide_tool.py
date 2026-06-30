"""Create guide tool — user describes a game operation, saves as type:guide skill.

The LLM parses the user's natural language description into structured steps,
then calls create_guide() to persist it as a properly formatted markdown file
in data/skills/<game>/, indexed in FTS5 for search.

Self-registers with the tool registry at import time.
"""

from __future__ import annotations

import json
import logging
import re

from src.tools.registry import registry, ToolOutput, get_current_game

logger = logging.getLogger(__name__)


def _sanitize_name(text: str) -> str:
    """Derive a safe skill name from natural language.

    Keeps Chinese characters, Latin letters, digits, hyphens.
    """
    safe = re.sub(r'[^\w一-鿿-]', '-', text.strip())
    safe = re.sub(r'-{2,}', '-', safe)
    safe = safe.strip('-')
    return safe[:40] if safe else "guide"


def create_guide(
    name: str,
    description: str,
    steps: str,
    game: str = "",
    pitfalls: str = "",
    tags: str = "",
    skill_type: str = "guide",
    verified: bool = False,
) -> ToolOutput:
    """Create or update a skill file from user's description.

    Args:
        name: Short name (e.g. "base-collect"). English/hyphens for filename.
        description: One-sentence summary of what this skill does.
        steps: Numbered steps, one per line. Natural language accepted.
        game: Game ID (default: current active game).
        pitfalls: Common mistakes to avoid, one per line.
        tags: Comma-separated keywords for search.
        skill_type: "guide" or "script". Guide = reference for LLM.
                    Script = executable with coordinates (verified=true).
        verified: Whether this skill has verified coordinates (script only).
    """
    resolved_game = game if game else get_current_game()

    if not name or not name.strip():
        return ToolOutput(text=json.dumps({
            "success": False, "error": "name is required"
        }, ensure_ascii=False))

    if not steps or not steps.strip():
        return ToolOutput(text=json.dumps({
            "success": False, "error": "steps is required"
        }, ensure_ascii=False))

    safe_name = _sanitize_name(name)

    # ── Parse steps ──
    step_lines: list[str] = []
    for line in steps.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading number: "1. xxx" or "1、xxx"
        line = re.sub(r'^\d+[.、．]\s*', '', line)
        if line:
            step_lines.append(line)

    # ── Natural language → tool format ──
    # Simple heuristics: if a step mentions a button name with "点击"/"点"/
    # "tap"/"进入" → convert to adb_tap('name'). Otherwise keep as-is.
    formatted_steps: list[str] = []
    for i, s in enumerate(step_lines):
        # Try to extract a button name from patterns like:
        #   "点击基建" → adb_tap('基建')
        #   "点右上角铃铛" → adb_tap_position(...) — too vague, keep as-is
        #   "返回主界面" → adb_back() + description
        tap_match = re.match(r'(?:点击|点|进入|打开)\s*[「『"【]?(.+?)[」』"】]?\s*$', s)
        if tap_match:
            target = tap_match.group(1).strip()
            formatted_steps.append(f"adb_tap('{target}')")
        elif "返回" in s and ("主界面" in s or "上一级" in s or "上级" in s):
            formatted_steps.append(f"adb_back()  # {s}")
        elif "等待" in s or "确认" in s:
            formatted_steps.append(f"# {s}")
        else:
            formatted_steps.append(s)

    # ── Parse pitfalls ──
    pitfall_lines: list[str] = []
    for line in pitfalls.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading "- " or "• " or number
        line = re.sub(r'^[-•]\s*', '', line)
        line = re.sub(r'^\d+[.、．]\s*', '', line)
        if line:
            pitfall_lines.append(line)

    # ── Parse tags ──
    tag_list: list[str] = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if skill_type not in tag_list:
        tag_list.append(skill_type)

    # ── Build markdown ──
    escaped_desc = description.replace('"', "'")[:120]
    verified_str = "true" if verified else "false"
    frontmatter = (
        "---\n"
        f"name: {safe_name}\n"
        f'description: "{escaped_desc}"\n'
        f"tags: [{', '.join(tag_list)}]\n"
        f"game: {resolved_game}\n"
        f"type: {skill_type}\n"
        f"verified: {verified_str}\n"
        "---"
    )

    body_lines = ["## Steps", ""]
    for i, s in enumerate(formatted_steps):
        body_lines.append(f"{i + 1}. {s}")
    body = "\n".join(body_lines)

    if pitfall_lines:
        body += "\n\n## Pitfalls\n"
        for p in pitfall_lines:
            body += f"- {p}\n"

    content = f"{frontmatter}\n\n{body}\n"

    # ── Save ──
    from src.skills.manager import get_skill_manager
    skill_mgr = get_skill_manager(resolved_game)
    path = skill_mgr.save(safe_name, content)

    logger.info("Guide created: %s (%d steps, %d pitfalls)", safe_name, len(formatted_steps), len(pitfall_lines))

    return ToolOutput(text=json.dumps({
        "success": True,
        "name": safe_name,
        "game": resolved_game,
        "steps": len(formatted_steps),
        "pitfalls": len(pitfall_lines),
        "path": str(path),
        "message": f"Guide '{safe_name}' 已保存。下次说 '{safe_name}' 或描述中的关键词即可匹配。",
    }, ensure_ascii=False))


# ── Registration ─────────────────────────────────────────────────────

registry.register(
    name="create_guide",
    description=(
        "创建或更新一个技能文件（skill）。接受自然语言描述，自动格式化。\n"
        "示例：create_guide(name='base-collect', description='基建收菜', game='arknights', "
        "steps='1. 点击基建\\n2. 点击通知铃铛\\n3. 点击可收获', "
        "pitfalls='- 加载画面需要等待', tags='基建,收菜,daily')"
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "技能名称（如 base-collect, farm-ce-5），英文/拼音，用于文件名",
            },
            "description": {
                "type": "string",
                "description": "一句话描述这个技能做什么",
            },
            "steps": {
                "type": "string",
                "description": (
                    "操作步骤，每行一步。自然语言即可。\n"
                    "'点击/进入/打开 XX' → adb_tap('XX')\n"
                    "'返回主界面' → adb_back()\n"
                    "纯描述步骤保留原样"
                ),
            },
            "game": {
                "type": "string",
                "description": "游戏 ID（默认当前活跃游戏）。如 arknights, reverse1999",
            },
            "pitfalls": {
                "type": "string",
                "description": "注意事项，每行一个（可选）",
            },
            "tags": {
                "type": "string",
                "description": "逗号分隔的关键词，用于搜索匹配（可选）",
            },
            "skill_type": {
                "type": "string",
                "description": "'guide'（指引，LLM 参考）或 'script'（脚本，含坐标可自动执行）。默认 guide",
            },
            "verified": {
                "type": "boolean",
                "description": "坐标是否已验证（script 专用）。默认 false",
            },
        },
        "required": ["name", "description", "steps", "game"],
    },
    handler=create_guide,
)
