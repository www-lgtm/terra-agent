"""Skill manager: load, save, list skills from data/skills/<game>/."""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

from config.settings import config
from src.skills.parser import parse_skill_md

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(config.DATA_DIR) / "skills"
_WRITE_LOCKS: dict[str, threading.Lock] = {}
_WRITE_LOCKS_LOCK = threading.Lock()


def _is_deprecated(raw_content: str, parsed_skill: dict[str, Any] | None = None) -> bool:
    """Check if a skill file has the deprecated marker in YAML frontmatter.

    Checks both the raw content (fast path) and the parsed frontmatter dict.
    This avoids false negatives from yaml.safe_load() dropping unknown fields.
    """
    # Fast path: scan raw content for the literal string
    if "deprecated:" in raw_content[:200]:
        try:
            # Check that it's "deprecated: true", not "deprecated: false"
            # by looking at the first YAML block
            lines = raw_content[:200].split("\n")
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("deprecated:"):
                    val = stripped.split(":", 1)[1].strip().lower()
                    return val in ("true", "1", "yes")
        except Exception:
            pass
    return False


class SkillManager:
    """Manages skill Markdown files on disk."""

    def __init__(self, game: str = "arknights", skill_dir: str = "") -> None:
        self.game = game
        # skill_dir is the on-disk directory name (e.g. "lifemaker" for game id "lifemakeover").
        # When omitted, fall back to using the game id directly (backward compat).
        self.base_dir = SKILLS_DIR / (skill_dir or game)

    def load(self, name: str) -> dict[str, Any] | None:
        """Load a skill by name. Searches subdirectories.

        Search order:
        1. Exact filename stem match (e.g. 'gt_6' for farm/gt_6.md)
           — skips deprecated files unless no alternative exists
        2. Parent directory match (e.g. 'farm' for farm/gt_6.md)
           — skips deprecated files unless no alternative exists
        3. Frontmatter 'name' field match (e.g. 'farm-gt-6' for farm/gt_6.md)
           — prefers non-deprecated over deprecated
        """
        # First pass: exact stem / directory match, skip deprecated
        for md_path in self.base_dir.rglob("*.md"):
            if md_path.stem == name or md_path.parent.name == name:
                content = md_path.read_text(encoding="utf-8")
                skill = parse_skill_md(content)
                # If deprecated, keep searching — a non-deprecated replacement
                # may exist with a different stem (e.g. credit-shop.md vs credit-shop-v2.md)
                if not _is_deprecated(content, skill):
                    return skill
                # Remember the first deprecated match as fallback
                if "_first_deprecated" not in self.__dict__:
                    pass  # We'll fall through to second pass

        # Second pass: parse all and check frontmatter name, prefer non-deprecated
        best_deprecated: dict[str, Any] | None = None
        for md_path in self.base_dir.rglob("*.md"):
            try:
                content = md_path.read_text(encoding="utf-8")
                skill = parse_skill_md(content)
                if skill.get("name") == name:
                    if _is_deprecated(content, skill):
                        if best_deprecated is None:
                            best_deprecated = skill
                    else:
                        return skill
            except Exception:
                continue

        # Fallback: return deprecated match if no non-deprecated found
        if best_deprecated is not None:
            logger.debug("Skill '%s' resolved to deprecated version (no replacement found)", name)
            return best_deprecated

        return None

    def save(self, name: str, content: str) -> Path:
        """Save a skill file. Creates parent directory if needed.
        Also indexes the skill in FTS5 for search.

        Thread-safe: uses per-skill Lock to prevent concurrent downgrade/upgrade races.
        """
        with _WRITE_LOCKS_LOCK:
            if name not in _WRITE_LOCKS:
                _WRITE_LOCKS[name] = threading.Lock()
        with _WRITE_LOCKS[name]:
            path = self._skill_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.info("Saved skill: %s", path)

        # Index in FTS5
        skill = parse_skill_md(content)
        if skill:
            from src.memory.skill_db import skill_db
            skill_db.index_skill(
                name=skill.get("name", name),
                description=skill.get("description", ""),
                tags=", ".join(skill.get("tags", [])),
                body=skill.get("body", ""),
                verified=skill.get("verified", False),
                skill_type=skill.get("type", "script" if skill.get("verified") else "guide"),
                game=self.game,
            )
            # cleanup_stale_skills() runs once per process in TerraAgent.__init__;
            # calling it on every save is O(n) file-system scan waste.

        return path

    def list_all(self) -> list[str]:
        """List all skill names for the current game."""
        if not self.base_dir.exists():
            return []
        names: list[str] = []
        for md_path in self.base_dir.rglob("*.md"):
            skill = parse_skill_md(md_path.read_text(encoding="utf-8"))
            name = skill.get("name") or md_path.stem
            names.append(name)
        return sorted(names)

    def search(self, query: str) -> list[dict[str, Any]]:
        """Keyword search returning parsed skill dicts (not just names).

        Splits query into CJK bigrams and alphanumeric tokens.  A skill must
        match at least ceil(term_count / 3) terms to be returned.

        Returns list of parsed skill dicts (same format as load()), avoiding
        the double-parse in callers that would otherwise do search() + load().
        """
        import re, math
        # Split into terms (CJK bigrams + alphabetic/word tokens ≥ 2 chars)
        terms: list[str] = []
        # Extract alphabetic/word tokens — filter single chars (1, 7, etc.)
        # because they match every skill's coordinate annotations
        word_tokens = re.findall(r'[a-zA-Z0-9]+', query)
        terms.extend(t.lower() for t in word_tokens if len(t) >= 2)
        # Extract CJK bigrams for better matching
        cjk = re.sub(r'[a-zA-Z0-9\s,，。.、：:；;！!？?()（）\[\]【】]', '', query)
        for i in range(len(cjk) - 1):
            bigram = cjk[i:i+2]
            if bigram not in terms:
                terms.append(bigram)
        if not terms:
            terms = [query.lower()]

        # Require proportional term hits to filter long-query noise.
        # Short queries (≤4 terms): need ≥1 hit (don't over-filter).
        # Medium queries (5-16 terms): need ≥2-4 hits.
        # Long verbose queries (30+ terms): need ≥8 hits → filters noise.
        if len(terms) <= 4:
            min_hits = 1
        else:
            min_hits = max(2, math.ceil(len(terms) / 4))

        if not self.base_dir.exists():
            return []

        results: list[dict[str, Any]] = []
        for md_path in self.base_dir.rglob("*.md"):
            try:
                skill = parse_skill_md(md_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            name = skill.get("name") or md_path.stem
            desc = skill.get("description", "")
            tags = " ".join(skill.get("tags", []))
            body = skill.get("body", "")
            searchable = f"{name} {desc} {tags} {body}".lower()
            hits = sum(1 for t in terms if t in searchable)
            if hits >= min_hits:
                results.append(skill)
        return results

    def delete(self, name: str) -> bool:
        path = self._skill_path(name)
        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        # Also remove from DB index
        try:
            from src.memory.skill_db import skill_db
            skill_db.remove_skill(name)
        except Exception:
            logger.debug("Failed to remove skill '%s' from DB index", name, exc_info=True)
        return deleted

    _SAFE_NAME_RE = re.compile(r'^[a-zA-Z0-9一-鿿_\-]+$')

    def _validate_name(self, name: str) -> None:
        """Reject names that contain path separators or unsafe characters.

        Prevents path traversal attacks where a malicious LLM output could
        write files outside the skills directory (e.g. name='../../malicious').
        """
        if not name:
            raise ValueError("Skill name must not be empty")
        if not self._SAFE_NAME_RE.match(name):
            raise ValueError(
                f"Skill name '{name}' contains unsafe characters. "
                f"Only letters, digits, Chinese characters, underscores, and hyphens are allowed."
            )
        # Double-check: resolved path must stay within base_dir
        resolved = (self.base_dir / f"{name}.md").resolve()
        if not str(resolved).startswith(str(self.base_dir.resolve())):
            raise ValueError(f"Skill name '{name}' escapes the skills directory.")

    def _skill_path(self, name: str) -> Path:
        self._validate_name(name)
        return self.base_dir / f"{name}.md"


# Default singleton (backward compat — defaults to arknights)
skill_manager = SkillManager()

# Game-aware cached factory — callers that know their game should use this
_skill_managers: dict[str, SkillManager] = {}


def get_skill_manager(game: str = "arknights") -> SkillManager:
    """Get or create a SkillManager for a specific game (cached).

    Resolves the on-disk skill directory from the GameRegistry.  This is
    critical when the game manifest's `id` differs from its `skill_dir`
    (e.g. game id "lifemakeover" maps to skill_dir "lifemaker").
    Without this resolution, SkillManager looks under data/skills/<game_id>/
    which may not exist, causing skill_list() to return empty.

    Prefer this over the module-level singleton when the game context is known.
    """
    if game not in _skill_managers:
        skill_dir = ""
        try:
            from src.games.registry import get_game_registry
            plugin = get_game_registry().get(game)
            if plugin is not None:
                skill_dir = plugin.manifest.skill_dir
        except Exception:
            pass
        _skill_managers[game] = SkillManager(game=game, skill_dir=skill_dir)
    return _skill_managers[game]
