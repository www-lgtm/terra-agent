"""Intent router: determines which game and what type of task.

Phase 3: delegates game-specific logic to GamePlugin registry.
Phase 2: FTS5 skill search integrated. Phase 1: keyword-based routing.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

from src.games.registry import get_game_registry
from src.memory.fts5_utils import build_search_terms, safe_fts5_term

logger = logging.getLogger(__name__)


# ── Game detection (delegates to GameRegistry) ────────────────────

def detect_game(text: str) -> str:
    """Detect which game a message targets via GameRegistry keyword scoring.

    Default: arknights.
    """
    return get_game_registry().detect_game(text)


# ── Task classification (delegates to game plugin) ───────────────

def classify_task(text: str, game: str = "arknights") -> str:
    """Classify a message into task type via the game plugin."""
    return get_game_registry().classify_task(text, game_id=game)


def get_priority(text: str, game: str = "arknights") -> int:
    """Get task priority (lower = higher priority) via the game plugin."""
    return get_game_registry().get_task_priority(text, game_id=game)


# ── Schedule intent classification ───────────────────────────────

def classify_schedule_intent(text: str, game: str = "arknights") -> str:
    """Classify a message into a schedule management intent.

    Returns:
        'create'  — user wants to create a new scheduled task
        'list'    — user wants to see all scheduled tasks
        'delete'  — user wants to remove a scheduled task
        'disable' — user wants to pause a task
        'enable'  — user wants to re-enable a paused task
        'stop'    — user wants to cancel a running task
        ''        — not a schedule-related message
    """
    return get_game_registry().classify_schedule_intent(text, game_id=game)


def extract_task_id(text: str) -> int | None:
    """Extract a task ID number from text like '取消定时任务#3' or '删除定时任务 3'.

    Returns the integer ID, or None if not found.
    """
    m = re.search(r'#?\s*(\d+)', text)
    if m:
        return int(m.group(1))
    return None


# ── Skill search (game-agnostic FTS5 + semantic rerank) ──────────

def search_skills(query: str, game: str = "arknights", limit: int = 5) -> list[dict[str, Any]]:
    """Search FTS5 skills index for matching skills.

    Pipeline:
    1. FTS5 full-text search (CJK bigrams + OR query)
    2. LLM semantic rerank when > limit candidates (Phase 3)
    3. Keyword fallback if FTS5 returns nothing
    """
    from src.memory.skill_db import skill_db

    results: list[dict[str, Any]] = []
    terms = build_search_terms(query)

    # Step 1: FTS5 search
    if terms:
        try:
            conn = skill_db.conn
            safe_terms = [safe_fts5_term(t) for t in terms]
            safe_terms = [t for t in safe_terms if t]
            if safe_terms:
                fts5_query = ' OR '.join(safe_terms)
                rows = conn.execute(
                    """SELECT d.name, d.description, d.tags, d.body, d.verified, d.type, f.rank
                       FROM skills_fts f
                       JOIN skills_data d ON f.rowid = d.id
                       WHERE skills_fts MATCH ? AND d.game = ?
                       ORDER BY f.rank
                       LIMIT ?""",
                    (fts5_query, game, limit * 2),  # Recall 2x for rerank pool
                ).fetchall()

                for r in rows:
                    results.append({
                        "name": r["name"],
                        "description": r["description"] or "",
                        "tags": r["tags"] or "",
                        "body": r["body"] or "",
                        "verified": bool(r["verified"]),
                        "type": r["type"] or "guide",
                    })
                logger.info("FTS5 skill search for '%s': %d results", query[:50], len(results))
        except Exception as e:
            logger.warning("FTS5 search failed, falling back to keyword search: %s", e)

    # Step 2: Rank without LLM — verified first, then FTS5 rank order.
    # Knowledge-type files are excluded from skill matching — they are
    # programmatic explore-engine output, not actionable for the LLM.
    results = [s for s in results if s.get("type") != "knowledge"]

    # ── Orchestrator dedup: when an orchestrator skill is matched,
    #     filter out its subskills from the results.  The orchestrator's
    #     Steps section already references them — showing them as
    #     separate "待完成子任务" confuses the LLM. ──
    orchestrators = [s for s in results if s.get("type") == "orchestrator"]
    if orchestrators:
        subskill_names: set[str] = set()
        for orch in orchestrators:
            for sub_name in orch.get("subskills", []):
                subskill_names.add(sub_name)
        if subskill_names:
            filtered = []
            for s in results:
                if s.get("name") in subskill_names and s.get("type") != "orchestrator":
                    logger.debug("Filtered subskill '%s' (covered by orchestrator)", s.get("name"))
                    continue
                filtered.append(s)
            results = filtered

    # ── Same-root-name dedup + priority ──
    # Version-bumped auto-generated skills (e.g. credit-shop-v2-3-v2,
    # base-collect-v2-2) should NOT crowd out the originals.  Per root name,
    # keep only the highest-priority skill: orchestrator > guide > script.
    # When same type, prefer the highest version number.
    # Also: never allow both an orchestrator AND its identically-named subskill
    # guide (e.g. daily orchestrator + daily guide) — orchestrator wins.
    if len(results) > 1:
        import re as _re
        _version_re = _re.compile(r'^(.*?)(?:-v\d+(?:-\d+)?)*$')
        def _root_name(name: str) -> str:
            m = _version_re.match(name)
            return m.group(1) if m else name
        _TYPE_PRIORITY = {"orchestrator": 3, "guide": 2, "script": 1}
        _by_root: dict[str, dict[str, Any]] = {}
        for s in results:
            root = _root_name(s.get("name", ""))
            s_type = s.get("type", "guide")
            s_priority = _TYPE_PRIORITY.get(s_type, 0)
            _name = s.get("name", "")
            # Extract version number for tiebreaking
            try:
                _parts = _name.split("-v")
                _s_ver = int(_parts[-1].split("-")[0]) if len(_parts) > 1 else 0
            except (ValueError, IndexError):
                _s_ver = 0
            if root not in _by_root:
                _by_root[root] = s
                _by_root[root]["_sort"] = (s_priority, _s_ver)
            else:
                _existing_pri, _existing_ver = _by_root[root].get("_sort", (0, 0))
                if s_priority > _existing_pri or (s_priority == _existing_pri and _s_ver > _existing_ver):
                    _by_root[root] = s
                    _by_root[root]["_sort"] = (s_priority, _s_ver)
        if len(_by_root) < len(results):
            logger.debug(
                "Root-name dedup: %d → %d skills (merged version variants)",
                len(results), len(_by_root),
            )
            results = list(_by_root.values())
        for s in results:
            s.pop("_sort", None)

    # ── Tag relevance filter + dominant match ──
    # Compute per-skill tag/name bigram hits, then:
    # 1. Drop zero-hit skills when others have hits
    # 2. When top skill's hits dominate (≥2x second place), keep only it.
    #    This prevents single-tag matches (e.g. base-collect matching via
    #    "基建" alone) from dragging in unrelated sub-skills. ──
    if len(results) > 1:
        from src.games.registry import get_game_registry
        game_name = get_game_registry().get_game_name(game)
        _clean_query = query.replace(game_name, "").strip()
        if not _clean_query:
            _clean_query = query
        query_bigrams = set(build_search_terms(_clean_query))
        for s in results:
            tag_text = s.get("tags", "")
            tags_lower = tag_text.lower()
            s["_tag_hits"] = sum(
                1 for bg in query_bigrams
                if bg in tags_lower or bg in s.get("name", "").lower()
            )
        max_hits = max(s["_tag_hits"] for s in results)
        if max_hits > 0:
            # Drop zero-hit skills
            if len(results) > 2:
                filtered_tag = [s for s in results if s["_tag_hits"] > 0]
                if filtered_tag:
                    logger.debug(
                        "Tag filter: %d → %d skills (dropped %d with no tag match)",
                        len(results), len(filtered_tag),
                        len(results) - len(filtered_tag),
                    )
                    results = filtered_tag
            # Recompute max after drop
            max_hits = max(s["_tag_hits"] for s in results)
            # Dominant match: top skill has ≥3x the tag hits of second place.
            # Raised from 2x because "日常+任务" in user's question about
            # annihilation was matching daily's tags and killing annihilation.
            if max_hits >= 3 and len(results) >= 2:
                sorted_hits = sorted(results, key=lambda s: -(s.get("_tag_hits", 0)))
                second_hits = sorted_hits[1].get("_tag_hits", 0)
                if max_hits >= second_hits * 3:
                    # Before dropping, protect skills whose name or tags appear
                    # verbatim in the query — user explicitly mentioned them.
                    keep = [sorted_hits[0]]
                    for s in sorted_hits[1:]:
                        skill_name = s.get("name", "")
                        tag_text = s.get("tags", "")
                        # Check if skill name or any tag word is a verbatim
                        # substring of the original (un-cleaned) query.
                        tag_words = [w.strip() for w in tag_text.replace(",", " ").split() if len(w.strip()) >= 2]
                        mentioned = (
                            skill_name in query
                            or any(tw in query for tw in tag_words)
                        )
                        if mentioned:
                            keep.append(s)
                    if len(keep) > 1:
                        logger.info(
                            "Dominant match softened: kept %d skills (query explicitly "
                            "mentions them) instead of dropping to 1 for query '%s'",
                            len(keep), query[:50],
                        )
                    else:
                        logger.info(
                            "Dominant skill match: '%s' (%d tag hits) >> '%s' (%d) — "
                            "dropping weaker match for query '%s'",
                            sorted_hits[0].get("name"), max_hits,
                            sorted_hits[1].get("name"), second_hits,
                            query[:50],
                        )
                    results = keep

    # ── Non-multi-task cap: specific queries get at most 2 skills.
    #     Sort by tag hits (desc) first so the most relevant survive. ──
    _multi_kw = ["日常", "周常", "全部", "所有", "每日", "每周"]
    if not any(kw in query for kw in _multi_kw) and len(results) > 2:
        results.sort(key=lambda s: -(s.get("_tag_hits", 0)))
        logger.debug("Non-multi-task cap: %d → 2 skills", len(results))
        results = results[:2]

    # Clean up temp key
    for s in results:
        s.pop("_tag_hits", None)

    results.sort(key=lambda s: (
        not s.get("verified"),
        -(s.get("type") == "orchestrator"),  # orchestrator before guide
        -(len(s.get("body", "")) > 20),
    ))
    if len(results) > limit:
        _schedule_async_rerank(query, results, top_n=limit)
        results = results[:limit]

    if results:
        return results

    # Step 3: Fallback keyword search via skill manager.
    from src.skills.manager import get_skill_manager
    skill_mgr = get_skill_manager(game)
    skills = skill_mgr.search(query)
    if not skills:
        return []

    for skill in skills[:limit]:
        results.append({
            "name": skill.get("name", ""),
            "description": skill.get("description", ""),
            "tags": ", ".join(skill.get("tags", [])),
            "body": skill.get("body", ""),
            "verified": skill.get("verified", False),
            "type": skill.get("type", "guide"),
        })
    return results


def _semantic_rerank_skills(query: str, candidates: list[dict[str, Any]],
                             top_n: int = 5) -> list[dict[str, Any]]:
    """LLM-based semantic relevance judge for skill search."""
    if not candidates or len(candidates) <= top_n:
        return candidates

    candidate_lines: list[str] = []
    for i, sk in enumerate(candidates):
        name = sk.get("name", "?")
        desc = sk.get("description", "")[:120]
        verified = "✓已验证" if sk.get("verified") else "✗未验证"
        candidate_lines.append(f"[{i}] {name} ({verified}) — {desc}")

    prompt = (
        "你是一个技能匹配判断器。给定用户的任务描述和一组候选技能，"
        "判断哪些技能最适合完成该任务。已验证的技能（✓）包含精确坐标，"
        "可以一键执行，应优先选择。\n\n"
        f"任务: {query[:300]}\n\n"
        "候选技能:\n" + "\n".join(candidate_lines) + "\n\n"
        "返回一个 JSON 数组，包含最相关的技能编号。只返回 JSON 数组，不要其他内容。"
        f"最多返回 {top_n} 个。示例: [0, 2]"
    )

    try:
        from src.llm.client import pooled_client, extract_text
        with pooled_client() as client:
            response = client.chat(
                system="你是技能匹配判断器。只输出 JSON 数组，不要任何解释或前言。",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
            )
            text = extract_text(response).strip()

            match = re.search(r"\[[\d,\s]*\]", text)
            if match:
                indices = json.loads(match.group())
                reranked = [candidates[i] for i in indices if 0 <= i < len(candidates)]
                if reranked:
                    logger.info("Semantic skill rerank: %d → %d relevant", len(candidates), len(reranked))
                    return reranked[:top_n]
    except Exception as e:
        logger.warning("Semantic skill rerank failed, falling back to FTS5 order: %s", e)

    return candidates[:top_n]


def _schedule_async_rerank(query: str, candidates: list[dict[str, Any]], top_n: int = 5) -> None:
    """Fire LLM semantic rerank in a background daemon thread."""
    t = threading.Thread(
        target=_semantic_rerank_skills,
        args=(query, candidates, top_n),
        daemon=True,
    )
    t.start()


# ── Full routing pipeline ─────────────────────────────────────────

def route_task(text: str, game: str = "arknights") -> dict[str, Any]:
    """Full routing pipeline: detect game, classify task, search skills.

    Returns a dict with game, task_type, priority, and matching skills.

    The `game` parameter carries the caller's game preference (e.g. from
    Concierge delegation).  When keyword detection is ambiguous (no keywords
    found), the hint is trusted so that Concierge-determined games are not
    silently overridden by the default.
    """
    registry = get_game_registry()
    # Pass game as hint so Concierge's game choice is respected when
    # the task text has been normalized (e.g. "1999" stripped by LLM)
    hint = game if game != registry.default_game else None
    detected_game = registry.detect_game(text, hint=hint)
    task_type = classify_task(text, game=detected_game)
    priority = get_priority(text, game=detected_game)

    skills = search_skills(text, game=detected_game)

    return {
        "game": detected_game,
        "task_type": task_type,
        "priority": priority,
        "matching_skills": skills,
    }


# ── Skill FTS5 index (re-exported from skill_db.py) ───────────────

from src.memory.skill_db import skill_db as _skill_db


def index_skill_fts(name: str, description: str, tags: str, body: str,
                    verified: bool = False) -> None:
    """Index or update a skill in the FTS5 table (delegates to skill_db)."""
    _skill_db.index_skill(name, description, tags, body, verified)


def _cleanup_stale_skills() -> None:
    """Delete skills_data rows whose .md files no longer exist (delegates to skill_db)."""
    _skill_db.cleanup_stale_skills()
