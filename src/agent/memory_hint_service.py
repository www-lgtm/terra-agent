"""MemoryHintService — multi-signal memory search pipeline.

Extracted from TerraAgent._gather_memory_hints() and _format_memory_hints().

Pipeline:
    1. dHash visual match — finds memories recorded on visually similar screens
    2. FTS5 text match — finds memories by OCR text + task description
    3. Semantic rerank via LLM to filter irrelevant candidates
    4. Format top-3 into a compact hint for the LLM agent

The service is **pure** — it does not mutate AgentState.  Injection tracking
(hits, injected_memory_ids, feedback_tracker) stays in TerraAgent.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)



class MemoryHintService:
    """Multi-signal memory search: dHash → FTS5 → Semantic Rerank → Format."""

    _MAX_SKIP_STREAK: int = 10  # was 5 — reduce forced LLM reranks

    def __init__(self, memory_db: Any, client_pool: Any = None,
                 config: Any = None) -> None:
        self._memory_db = memory_db
        self._client_pool = client_pool
        self._config = config
        # Internal state
        self._last_rerank_key: str = ""
        self._skip_streak: int = 0      # Consecutive rerank skips — force at 5
        self._last_rerank_time: float = 0.0
        self._rerank_min_interval: float = 15.0  # was 3.0s — LLM rerank costs ~500 tokens/call

    # ── Public API ──────────────────────────────────────────────────

    def gather(
        self,
        ocr_texts: list[str],
        dhash_hex: str | None,
        task_desc: str,
        skill_names: list[str],
        game: str,
    ) -> tuple[list[dict], str | None]:
        """Search for relevant memories. Returns (candidates, formatted_hint_or_none).

        Args:
            ocr_texts: Current screen OCR texts.
            dhash_hex: Perceptual hash of current screen (hex).
            task_desc: User's original task description.
            skill_names: Names of matching skills.
            game: Active game ID.

        Returns:
            (raw_candidates_list, formatted_hint_string) where formatted_hint_string
            is None when no relevant memories were found.  The candidates list has
            up to 3 entries after reranking.  The caller is responsible for
            incrementing hits and tracking injection IDs.
        """
        from src.tools.remember import (
            _search_memory_internal,
            _search_by_dhash,
            _semantic_rerank,
        )

        # Build a rich query from multiple text signals
        query_parts: list[str] = []
        if ocr_texts:
            query_parts.append(" ".join(ocr_texts[:15]))
        if task_desc:
            query_parts.append(task_desc)
        if skill_names:
            query_parts.append(" ".join(skill_names))
        query = " ".join(query_parts) if query_parts else ""
        if not query.strip() and not dhash_hex:
            return [], None

        candidates: list[dict[str, Any]] = []
        seen_ids: set[int] = set()

        # ── Signal 0: Skill-name direct match ──
        # When a skill like 'recruit' is active, pull all memories tagged
        # with that skill name.  This bypasses dHash visual matching which
        # often fails on near-identical UI (game clock, notifications, etc.)
        # even though the UI layout hasn't changed.  These get lower priority
        # than visual matches but are always included in the candidate pool.
        if skill_names:
            try:
                skill_query = " OR ".join(skill_names)
                skill_matches = _search_memory_internal(
                    skill_query, games=[game, "_shared"], limit=5,
                )
                for m in skill_matches:
                    if m["id"] not in seen_ids:
                        m["_source"] = "skill"
                        candidates.append(m)
                        seen_ids.add(m["id"])
            except Exception:
                from src.utils.errors import safe_log
                safe_log(logger, "warning", "Skill-tag memory search failed")

        # ── Signal 1: dHash visual matching ──
        if dhash_hex:
            try:
                visual = _search_by_dhash(dhash_hex, game, threshold=10, limit=5)
                for m in visual:
                    if m["id"] not in seen_ids:
                        m["_source"] = "visual"
                        candidates.append(m)
                        seen_ids.add(m["id"])
            except Exception:
                from src.utils.errors import safe_log
                safe_log(logger, "warning", "dHash memory search failed")

        # ── Signal 2: FTS5 text matching ──
        if query.strip():
            try:
                text_matches = _search_memory_internal(
                    query, games=[game, "_shared"], limit=5,
                )
                for m in text_matches:
                    if m["id"] not in seen_ids:
                        m["_source"] = "text"
                        candidates.append(m)
                        seen_ids.add(m["id"])
            except Exception:
                from src.utils.errors import safe_log
                safe_log(logger, "warning", "FTS5 memory search failed")

        if not candidates:
            return [], None

        # ── Merge: sort by dHash distance then by score ──
        candidates.sort(key=lambda m: (
            m.get("hamming_dist", 999),
            -m.get("_score", 0),
        ))
        candidates = candidates[:10]
        pre_rerank_pool = list(candidates)  # Snapshot for exploration swap

        # ── Signal 3a: Vector similarity re-rank (optional, no LLM cost) ──
        if query.strip():
            try:
                from src.memory.vector_store import get_vector_store
                vs = get_vector_store()
                if vs.available:
                    candidates = vs.search(query, candidates, top_n=10)
            except Exception:
                from src.utils.errors import safe_log
                safe_log(logger, "warning", "Vector search failed, falling back to FTS5 order")

        # ── Signal 3b: Semantic rerank (LLM-based, expensive) ──
        # Skip when OCR hasn't meaningfully changed since last rerank (>70% overlap).
        needs_rerank = any(
            m.get("_source") != "visual" for m in candidates
        ) and query.strip()

        # dHash early-exit: when any visual candidate has hamming_dist ≤3,
        # the screen is near-identical to a known memory's anchor.  Visual
        # matching already locked onto the right memories — LLM rerank is
        # unlikely to find anything better among FTS5 results mixed in.
        if needs_rerank and any(
            m.get("_source") == "visual" and m.get("hamming_dist", 99) <= 3
            for m in candidates
        ):
            logger.debug("Semantic rerank skipped: close visual match found "
                         "(dHash ≤3) for %d/%d candidates",
                         sum(1 for m in candidates if m.get("hamming_dist", 99) <= 3),
                         len(candidates))
            needs_rerank = False

        if needs_rerank:
            ocr_key = " ".join(sorted(ocr_texts[:15])) if ocr_texts else ""
            last_key = self._last_rerank_key
            _forced_rerank = False  # set when skip-streak exhaustion forces rerank
            # Jaccard similarity check
            if last_key and ocr_key:
                last_set = set(last_key.split())
                curr_set = set(ocr_key.split())
                union = len(last_set | curr_set)
                if union > 0:
                    overlap = len(last_set & curr_set) / union
                    if overlap > 0.85:  # was 0.7 — higher threshold = fewer reranks, saves ~500 tok/call
                        self._skip_streak += 1
                        if self._skip_streak >= self._MAX_SKIP_STREAK:
                            logger.info("Rerank forced after %d skips", self._skip_streak)
                            self._skip_streak = 0
                            _forced_rerank = True
                        else:
                            needs_rerank = False
            if needs_rerank:
                import time as _time
                now = _time.monotonic()
                if not _forced_rerank and now - self._last_rerank_time < self._rerank_min_interval:
                    candidates = candidates[:3]
                else:
                    self._skip_streak = 0  # reset on actual rerank
                    try:
                        candidates = _semantic_rerank(query, candidates, top_n=3)
                        self._last_rerank_key = ocr_key
                        self._last_rerank_time = now
                    except Exception:
                        from src.utils.errors import safe_log
                        safe_log(logger, "warning", "Semantic rerank failed")
                        candidates = candidates[:3]

        if not candidates:
            return [], None

        # ── Exploration swap: ~1/3 of the time, replace the lowest-ranked
        #     candidate with an unexplored one (injected_count == 0) from
        #     the pre-rerank pool.  This gives "ghost" memories a chance
        #     to prove themselves. ──
        candidates = self._exploration_swap(candidates, pre_rerank_pool)

        hint = self.format(candidates)
        return candidates, hint

    def reset_for_task(self) -> None:
        """Reset per-task state (called at the start of each new task)."""
        self._last_rerank_key = ""
        self._skip_streak = 0

    # ── Exploration / exploitation balance ──────────────────────────

    @staticmethod
    def _exploration_swap(
        selected: list[dict],
        pool: list[dict],
    ) -> list[dict]:
        """Occasionally swap a low-ranked selected candidate with an unexplored one.

        With ~1/3 probability, replaces the lowest-ranked selected candidate
        with one from the pool whose injected_count == 0 (never tried before).
        This gives "ghost" memories a chance to prove themselves and escape
        the never-injected → never-proven → never-injected cycle.

        Only swaps when pool has unexplored candidates that differ from
        all selected ones (avoids injecting a near-duplicate).
        """
        import random as _random

        # ~1/3 probability — enough to explore, not enough to degrade quality
        if _random.random() > 0.33:
            return selected

        if len(selected) < 2:
            return selected  # Need at least 2 to swap

        selected_ids = {m["id"] for m in selected}
        unexplored = [
            m for m in pool
            if m.get("injected_count", 0) == 0 and m["id"] not in selected_ids
        ]
        if not unexplored:
            return selected

        # Replace the lowest-ranked selected candidate (last in list)
        explorer = unexplored[0]  # First unexplored in pool order
        logger.info(
            "Exploration swap: injecting unproven memory id=%s '%s' "
            "instead of proven id=%s '%s'",
            explorer["id"], explorer.get("name", "?")[:20],
            selected[-1]["id"], selected[-1].get("name", "?")[:20],
        )
        return selected[:-1] + [explorer]

    # ── Formatting ──────────────────────────────────────────────────

    @staticmethod
    def format(memories: list[dict[str, Any]]) -> str:
        """Format memory dicts into a compact user message.

        Action patterns (source='action_pattern') are shown as [经验模式].
        Regular memories (pitfalls) are shown as [关联记忆].

        When two or more memories share a visual anchor (screen_hash), a
        conflict warning is appended so the LLM knows to reconcile them.
        """
        patterns = [m for m in memories if m.get("source") == "action_pattern"]
        pitfalls = [m for m in memories if m.get("source") != "action_pattern"]

        parts: list[str] = []

        if patterns:
            lines = ["[经验模式 — 正向参考：遇到相似画面时可以这样做]"]
            for i, m in enumerate(patterns):
                body_preview = m["body"][:180].replace("\n", " ")
                if len(m["body"]) > 180:
                    body_preview += "…"
                lines.append(f"✓ {i + 1}. {body_preview}")
            parts.append("\n".join(lines))

        if pitfalls:
            lines = ["[关联记忆 — 注意避坑：参考以下历史经验避免重复犯错]"]
            for i, m in enumerate(pitfalls):
                body_preview = m["body"][:180].replace("\n", " ")
                if len(m["body"]) > 180:
                    body_preview += "…"
                lines.append(f"⚠ {i + 1}. {body_preview}")
            parts.append("\n".join(lines))

        # ── Conflict detection: same-screen contradictory advice ──
        conflict_note = MemoryHintService._detect_conflicts(memories)
        if conflict_note:
            parts.append(conflict_note)

        return "\n\n".join(parts)

    @staticmethod
    def _detect_conflicts(memories: list[dict[str, Any]]) -> str | None:
        """Detect potentially conflicting memories injected for the same screen.

        Two memories conflict when they share a visual anchor (screen_hash)
        but appear to give contradictory advice — typically one tells the
        agent to do something and another warns against doing it.

        Strategy: group by overlapping screen_hashes.  For each group with
        2+ members, check if at least one is a pitfall (negative framing)
        and one is a pattern (positive framing).  Cross-screen memories
        (no screen_hash) are excluded — they reference different contexts.
        """
        # Collect memories with screen_hash (visual anchoring)
        anchored: list[dict] = []
        for m in memories:
            sh = m.get("screen_hash", "")
            if sh:
                anchored.append(m)

        if len(anchored) < 2:
            return None

        # Group by overlapping hashes
        groups: list[list[dict]] = []
        used: set[int] = set()
        for i, m1 in enumerate(anchored):
            if i in used:
                continue
            h1 = set(h.strip() for h in m1["screen_hash"].split(",") if h.strip())
            group = [m1]
            used.add(i)
            for j, m2 in enumerate(anchored):
                if j in used:
                    continue
                h2 = set(h.strip() for h in m2["screen_hash"].split(",") if h.strip())
                if h1 & h2:  # Share at least one hash
                    group.append(m2)
                    used.add(j)
            groups.append(group)

        # Check each group for mixed pattern/pitfall
        for group in groups:
            if len(group) < 2:
                continue
            has_pattern = any(m.get("source") == "action_pattern" for m in group)
            has_pitfall = any(
                m.get("source") != "action_pattern" for m in group
            )
            if has_pattern and has_pitfall:
                names = [m.get("name", "?")[:12] for m in group]
                # ── Confidence-weighted resolution ──
                # When one memory has much higher confidence, bias toward it.
                confs = [(m.get("confidence") or 0) for m in group]
                max_conf = max(confs) if confs else 0
                min_conf = min(confs) if confs else 0
                bias = ""
                if max_conf > 0 and max_conf - min_conf >= 0.3:
                    # Find the high-confidence memory
                    hi_mem = next((m for m in group if (m.get("confidence") or 0) == max_conf), None)
                    if hi_mem:
                        hi_name = hi_mem.get("name", "?")[:12]
                        hi_src = "正向模式" if hi_mem.get("source") == "action_pattern" else "坑点警告"
                        bias = (
                            f" 其中「{hi_name}」({hi_src})的可信度({max_conf:.0%})"
                            f"明显高于其他建议——建议优先采纳。"
                        )
                return (
                    "[⚠ 冲突提示] 以上记忆中有多条针对同一画面的建议——有的推荐做法，"
                    "有的警告坑点。请结合**当前实际画面**判断哪条更适用。"
                    f"{bias}"
                    f" (关联记忆: {', '.join(names)})"
                )

        return None
