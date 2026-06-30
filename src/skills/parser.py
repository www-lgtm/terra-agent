"""Parse skill Markdown files with YAML frontmatter."""

from __future__ import annotations

import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class SkillParser:
    """Parse SKILL.md files with YAML frontmatter + Markdown body."""

    @staticmethod
    def parse(content: str) -> dict[str, Any]:
        """Parse a skill markdown string into structured data.

        Returns dict with keys: name, description, tags, game, steps, pitfalls, raw.
        """
        frontmatter, body = SkillParser._split_frontmatter(content)

        meta: dict[str, Any] = {}
        if frontmatter:
            try:
                meta = yaml.safe_load(frontmatter) or {}
            except yaml.YAMLError as e:
                logger.warning("Invalid YAML frontmatter: %s", e)

        steps = SkillParser._extract_steps(body)
        pitfalls = SkillParser._extract_pitfalls(body)

        return {
            "name": meta.get("name", ""),
            "description": meta.get("description", ""),
            "tags": meta.get("tags", []),
            "game": meta.get("game", "arknights"),
            "verified": meta.get("verified", False),
            "type": meta.get("type", "script" if meta.get("verified") else "guide"),
            "version": meta.get("version", 0),
            "coords_verified_at": meta.get("coords_verified_at", ""),
            "subskills": meta.get("subskills", []),
            "steps": steps,
            "pitfalls": pitfalls,
            "body": body,
            "raw": content,
        }

    @staticmethod
    def _split_frontmatter(content: str) -> tuple[str, str]:
        """Split YAML frontmatter from markdown body."""
        content = content.strip()
        if not content.startswith("---"):
            return "", content

        parts = content.split("---", 2)
        if len(parts) < 3:
            return "", content
        return parts[1].strip(), parts[2].strip()

    @staticmethod
    def _extract_steps(body: str) -> list[dict]:
        """Extract numbered steps — delegates to the canonical parser in fast_chain.

        parse_skill_steps() is the single source of truth for step parsing.
        It handles coordinate extraction from # [x, y] comments and adb_swipe args,
        producing {tool, args, coords} dicts used by both display and execution.
        """
        # Lazy import to avoid circular dependency at module level
        from src.tools.fast_chain import parse_skill_steps
        return parse_skill_steps(body)

    @staticmethod
    def _extract_pitfalls(body: str) -> list[str]:
        """Extract pitfalls from the markdown body."""
        pitfalls: list[str] = []
        in_section = False
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("## Pitfalls") or line.startswith("## 注意事项"):
                in_section = True
                continue
            if in_section and line.startswith("##"):
                break
            if in_section and line.startswith("- "):
                pitfalls.append(line[2:])
        return pitfalls


def parse_skill_md(content: str) -> dict[str, Any]:
    return SkillParser.parse(content)
