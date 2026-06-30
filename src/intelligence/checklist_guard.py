"""ChecklistGuard — pre-flight intelligence tool (Phase 2).

Before executing a task, scans relevant memories and produces a compact
"pre-flight checklist" of common pitfalls for this task type.  The checklist
is injected as [智能建议] into the conversation context.
"""

from __future__ import annotations

import logging

from src.intelligence.base import (
    IntelligenceContext,
    IntelligenceResult,
    IntelligenceTool,
)

logger = logging.getLogger(__name__)


class ChecklistGuard(IntelligenceTool):
    """Pre-flight pitfall checklist based on task type and memory history."""

    def can_handle(self, task: str) -> bool:
        """All tasks can benefit from a checklist."""
        return True

    def analyze(self, ctx: IntelligenceContext, task: str) -> IntelligenceResult | None:
        """Query memories for task-type-relevant pitfalls and format a checklist.

        Strategy:
        1. FTS5 search for memories matching the task description
        2. Filter to those with harm_count > 0 or source='action_pattern'
        3. Prioritize: high harm first (most important to avoid), then high help
        4. Format into a compact numbered list
        """
        if len(task) < 3:
            return None

        try:
            from src.tools.remember import _search_memory_internal
            from src.memory.memory_db import memory_db
        except ImportError:
            return None

        # Search for relevant memories — FTS5 (lexical) first
        candidates = _search_memory_internal(task, games=[ctx.game, "_shared"], limit=8)

        # ── Supplement: vector semantic search when FTS5 recall is low ──
        # FTS5 misses semantically-similar but lexically-different memories
        # (e.g. "刷1-7" vs "farm rocks").  Vector search fills this gap.
        if len(candidates) < 3:
            vec_candidates = _search_via_vector_for_checklist(task, ctx.game, limit=8)
            if vec_candidates:
                existing_ids = {m["id"] for m in candidates}
                for vc in vec_candidates:
                    if vc["id"] not in existing_ids:
                        vc["_source"] = "vector"
                        candidates.append(vc)
                        existing_ids.add(vc["id"])
                # Re-sort: FTS5 results first (lexical exact), then vector
                candidates.sort(key=lambda m: (0 if m.get("_source") != "vector" else 1, -(m.get("_score", 0))))

        if not candidates:
            return None

        # Enrich with help/harm stats
        enriched = []
        for m in candidates:
            row = memory_db.conn.execute(
                "SELECT help_count, harm_count FROM memories_data WHERE id=?",
                (m["id"],),
            ).fetchone()
            help_c = row["help_count"] if row else 0
            harm_c = row["harm_count"] if row else 0
            m["_help"] = help_c or 0
            m["_harm"] = harm_c or 0
            enriched.append(m)

        # Sort: high-harm first (critical pitfalls), then high-help (useful patterns)
        enriched.sort(key=lambda m: (-m["_harm"], -m["_help"]))

        # ── Dedup against active skill guides ──
        # When a skill guide already has detailed steps and pitfalls, injecting
        # overlapping memories only adds noise.  LLM gets confused when 4 sources
        # (skill guide + 3 memories) all talk about the same bell problem.
        skill_body = _get_skill_body(ctx)
        if skill_body:
            enriched = _filter_redundant_memories(enriched, skill_body)

        # Build checklist — at most 5 items
        items: list[str] = []
        for m in enriched[:5]:
            body_preview = (m.get("body", "") or "")[:120].replace("\n", " ")
            source = m.get("source", "")
            if source == "action_pattern":
                prefix = "✓ 正向模式"
            elif m["_harm"] > 0:
                prefix = f"⚠ 常见坑点（{m['_harm']}次无效）"
            elif m["_help"] > 0:
                prefix = f"💡 有效经验（{m['_help']}次帮助）"
            else:
                prefix = "📝 参考"
            items.append(f"{prefix}: {body_preview}")

        if not items:
            return None

        checklist = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(items))
        recommendation = (
            f"执行「{task[:40]}」前，参考以下历史经验：\n{checklist}"
        )

        return IntelligenceResult(
            recommendation=recommendation,
            confidence=0.7,
            source="memories",
        )


def _get_skill_body(ctx) -> str:
    """Extract skill guide body text from the intelligence context."""
    skill_parts: list[str] = []
    for s in (ctx.skills or []):
        if s.get("type") == "orchestrator":
            continue  # orchestrators just list sub-skills, no concrete steps
        body = (s.get("body", "") or "").strip()
        if body:
            skill_parts.append(body)
        pitfalls = s.get("pitfalls", "")
        if pitfalls:
            skill_parts.append(str(pitfalls).strip())
    return "\n".join(skill_parts)


def _filter_redundant_memories(memories: list[dict], skill_body: str) -> list[dict]:
    """Filter out memories that duplicate content already in the skill guide.

    When a skill guide already has verified coordinates or explicit pitfalls,
    injecting memories about the same problem just adds noise.
    """
    if not skill_body:
        return memories

    # Quick checks for skill guide content signals
    has_bell_coords = "adb_tap_position(0.94" in skill_body or \
                      "adb_tap_position(0.93" in skill_body or \
                      "adb_tap_position(0.91" in skill_body
    has_bell_pitfall = "铃铛" in skill_body and "很难点" in skill_body

    kept: list[dict] = []
    for m in memories:
        body = (m.get("body", "") or "")[:200]

        # Suppress bell memories when skill guide has verified coordinates
        if has_bell_coords and ("铃铛" in body or "bell" in body.lower() or
                                 "notification" in body.lower()):
            continue
        if has_bell_pitfall and ("铃铛" in body and "很难点" in body):
            continue

        # Check CJK bigram Jaccard overlap with skill body (>50% → suppress)
        if _cjk_overlap(body, skill_body) > 0.5:
            continue

        kept.append(m)

    return kept


def _cjk_overlap(a: str, b: str) -> float:
    """Compute CJK bigram Jaccard similarity between two short strings."""
    bigrams_a = {a[i:i+2] for i in range(len(a)-1)
                 if '一' <= a[i] <= '鿿' or '一' <= a[i+1] <= '鿿'}
    bigrams_b = {b[i:i+2] for i in range(len(b)-1)
                 if '一' <= b[i] <= '鿿' or '一' <= b[i+1] <= '鿿'}
    if not bigrams_a or not bigrams_b:
        return 0.0
    intersection = len(bigrams_a & bigrams_b)
    union = len(bigrams_a | bigrams_b)
    return intersection / union if union > 0 else 0.0


def _search_via_vector_for_checklist(query: str, game: str, limit: int = 8) -> list[dict]:
    """Vector semantic search for checklist-relevant memories.

    Used as a supplement to FTS5 when lexical search returns few results.
    Queries the vector_store for embeddings similar to the task description.
    """
    try:
        from src.memory.vector_store import get_vector_store
        vs = get_vector_store()
        if not vs.available:
            return []

        query_blob = vs.encode(query)
        if query_blob is None:
            return []

        from src.memory.memory_db import memory_db
        # Load all memories with embeddings for this game
        rows = memory_db.conn.execute(
            """SELECT id, name, game, tags, body, source, created, hits,
                      help_count, harm_count, injected_count, embedding
               FROM memories_data
               WHERE game IN (?, '_shared') AND embedding IS NOT NULL
                 AND deleted_at IS NULL""",
            (game,),
        ).fetchall()

        if not rows:
            return []

        # Compute similarity scores
        candidate_blobs = [(r["id"], r["embedding"]) for r in rows]
        scored = vs.similarity(query_blob, candidate_blobs)
        if not scored:
            return []

        # Build result dicts for top matches
        id_to_row = {r["id"]: r for r in rows}
        results: list[dict] = []
        for mid, sim in scored[:limit]:
            row = id_to_row.get(mid)
            if row is None or sim < 0.5:  # Minimum relevance threshold
                continue
            results.append({
                "id": row["id"],
                "name": row["name"],
                "game": row["game"],
                "tags": row["tags"] or "",
                "body": row["body"] or "",
                "source": row["source"] or "",
                "hits": row["hits"] or 0,
                "help_count": row["help_count"] or 0,
                "harm_count": row["harm_count"] or 0,
                "injected_count": row["injected_count"] or 0,
                "_score": sim,
                "_source": "vector",
            })
        logger.info("ChecklistGuard vector search: %d candidates (query: %.40s)", len(results), query)
        return results
    except Exception as e:
        from src.utils.errors import safe_log
        safe_log(logger, "warning", f"Vector checklist search failed: {e}")
        return []
