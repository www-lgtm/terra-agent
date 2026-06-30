"""Memory tools: remember, search_memory, forget_memory.

Memories are stored as markdown files under data/memories/{game}/ and
indexed in SQLite FTS5. Separate from skills in both storage and table.

Internal functions (no ToolOutput wrapper) are exposed for programmatic use
by the agent loop, enabling auto-injection of relevant memories alongside
screenshot injections.
"""

from __future__ import annotations

import json
import logging
import time as _time
from datetime import datetime, timezone
from pathlib import Path

from config.settings import config
from src.memory.fts5_utils import build_search_terms, safe_fts5_term
from src.tools.registry import ToolOutput, registry

logger = logging.getLogger(__name__)

_MEMORIES_DIR = Path(config.DATA_DIR) / "memories"


def _memories_dir_for(game: str) -> Path:
    return _MEMORIES_DIR / game


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _search_fts5(query: str, games: list[str], limit: int = 10) -> list[dict]:
    """FTS5 search across specified games. Returns rows with bm25 rank."""
    from src.memory.memory_db import memory_db

    memory_db.rebuild_fts_if_dirty()  # Ensure index is up-to-date before search
    terms = build_search_terms(query)
    if not terms:
        return []

    safe_terms = [safe_fts5_term(t) for t in terms]
    safe_terms = [t for t in safe_terms if t]
    if not safe_terms:
        return []

    fts5_query = ' OR '.join(safe_terms)
    game_placeholders = ','.join(['?'] * len(games))

    try:
        conn = memory_db.conn
        rows = conn.execute(
            f"""SELECT m.id, m.name, m.game, m.tags, m.body, m.source, m.created, m.hits,
                      m.help_count, m.harm_count, m.last_helpful_at, m.injected_count,
                      m.confidence, f.rank
                FROM memories_fts f
                JOIN memories_data m ON f.rowid = m.id
                WHERE memories_fts MATCH ? AND m.game IN ({game_placeholders})
                  AND m.deleted_at IS NULL
                ORDER BY f.rank
                LIMIT ?""",
            (fts5_query, *games, limit),
        ).fetchall()

        results: list[dict] = []
        for r in rows:
            results.append({
                "id": r["id"],
                "name": r["name"],
                "game": r["game"],
                "tags": r["tags"] or "",
                "body": r["body"] or "",
                "source": r["source"] or "",
                "created": r["created"] or "",
                "hits": r["hits"] or 0,
                "help_count": r["help_count"] or 0,
                "harm_count": r["harm_count"] or 0,
                "last_helpful_at": r["last_helpful_at"],
                "injected_count": r["injected_count"] or 0,
                "confidence": r["confidence"],
                "bm25": r["rank"],
            })
        return results
    except Exception as e:
        logger.warning("FTS5 memory search failed: %s", e)
        return []


def _rank_memories(results: list[dict], top_n: int = 5) -> list[dict]:
    """Re-rank FTS5 results by bm25 * hits_weight * recency_weight * helpfulness_weight.

    Scoring dimensions:
      - bm25: lexical relevance from FTS5
      - hits: manual search popularity (each hit adds 15%)
      - recency: creation date decay (180-day half-life)
      - helpfulness: proven-helpful memories get a boost; the boost decays
        if the memory hasn't actually helped anyone recently (last_helpful_at).
        Harmful memories (harm > help) get a penalty.
    """
    now = datetime.now(tz=timezone.utc)

    for r in results:
        bm25 = r.get("bm25", 0)
        # Normalize bm25: FTS5 returns negative values, more negative = better
        bm25_score = max(0.1, abs(bm25) if bm25 < 0 else 0.1)

        hits = r.get("hits", 0)
        hits_w = 1.0 + hits * 0.15

        # Recency: memories older than 180 days get penalized
        created_str = r.get("created", "")
        days = 365
        if created_str:
            try:
                created = datetime.fromisoformat(created_str)
                days = (now - created).days
            except (ValueError, TypeError):
                pass
        recency_w = 1.0 / (1.0 + max(0, days) / 180.0)

        # ── Helpfulness weight (Phase 2) ──
        # Proven-helpful memories (help_count >= 3) get a boost tied to
        # how recently they actually helped.  The boost decays over 7 days
        # since the last helpful injection, so a memory that was useful 100
        # times last month doesn't outrank one that helped 5 times yesterday.
        help_c = r.get("help_count", 0) or 0
        harm_c = r.get("harm_count", 0) or 0
        helpfulness_w = 1.0
        if help_c >= 3:
            last_h = r.get("last_helpful_at")
            if last_h:
                try:
                    if isinstance(last_h, (int, float)):
                        days_since_help = (now - datetime.fromtimestamp(last_h, tz=timezone.utc)).days
                    else:
                        days_since_help = (now - datetime.fromisoformat(str(last_h))).days
                except (ValueError, TypeError, OSError):
                    days_since_help = 14  # Unknown recency → neutral
                # Boost: up to +80% for proven-helpful, decaying over 7 days
                helpfulness_w += 0.8 / (1.0 + max(0, days_since_help) / 7.0)
            else:
                # No last_helpful_at but high help count → data from before
                # the tracking existed.  Give a moderate boost.
                helpfulness_w += 0.3
        elif harm_c > help_c and help_c + harm_c >= 5:
            # Proven-harmful: penalty proportional to harm ratio
            total = help_c + harm_c
            harm_ratio = harm_c / total if total > 0 else 0
            helpfulness_w *= max(0.3, 1.0 - harm_ratio)

        r["_score"] = bm25_score * hits_w * recency_w * helpfulness_w

        # ── Cold-start quality boost ──
        # Unproven memories (few injections, no help/harm data) with high
        # intrinsic quality get a ranking boost.  This helps them escape the
        # never-injected → never-proven cycle.  Proven memories (>3 helps)
        # don't need this — their real-world track record speaks for itself.
        injected = r.get("injected_count", 0) or 0
        if help_c < 3 and harm_c == 0 and injected < 3:
            confidence = r.get("confidence")
            if confidence and isinstance(confidence, (int, float)) and confidence >= 0.7:
                r["_score"] *= 1.3  # 30% boost for high-quality unproven memories
                r["_cold_start_boost"] = True

    results.sort(key=lambda r: r["_score"], reverse=True)
    return results[:top_n]


def _increment_hits(memory_id: int) -> None:
    """Increment the hits counter for a memory."""
    from src.memory.memory_db import memory_db
    try:
        memory_db.conn.execute(
            "UPDATE memories_data SET hits = hits + 1 WHERE id = ?", (memory_id,)
        )
        memory_db.conn.commit()
    except Exception as e:
        logger.warning("Failed to increment hits for memory %d: %s", memory_id, e)


def _schedule_embedding_generation(memory_id: int, body: str) -> None:
    """Generate and store a vector embedding for a new/updated memory.

    Runs in a background thread so it never blocks the agent loop.
    Gracefully degrades if sentence-transformers is unavailable.
    """
    if not body or not body.strip():
        return

    def _run() -> None:
        try:
            from src.memory.vector_store import get_vector_store
            vs = get_vector_store()
            blob = vs.encode(body)
            if blob is not None:
                from src.memory.memory_db import memory_db
                memory_db.conn.execute(
                    "UPDATE memories_data SET embedding = ? WHERE id = ?",
                    (blob, memory_id),
                )
                memory_db.conn.commit()
                logger.debug("Embedding stored for memory #%d", memory_id)
        except Exception as e:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", f"Embedding generation for memory #{memory_id} failed: {e}")

    import threading
    t = threading.Thread(target=_run, daemon=True)
    t.start()


_QUALITY_SCORE_PROMPT = """对下面这条记忆做 4 维质量评分（1-5分），只输出 JSON：

评分维度：
1. screen_clarity: 画面特征是否具体可识别？（OCR关键词、UI描述）1=模糊 5=精确
2. error_specificity: 错误操作是否明确？1=含糊 5=一针见血
3. action_executability: 正确做法是否可直接执行？1=纯抽象建议 5=具体步骤
4. generalizability: 可否泛化到其他类似画面？1=一次性细节 5=可广泛复用

记忆：
{body}

输出: {{"screen_clarity": ?, "error_specificity": ?, "action_executability": ?, "generalizability": ?, "overall": ?.?}}"""


def _schedule_quality_scoring(memory_id: int, body: str, name: str, game: str) -> None:
    """Score a new memory's intrinsic quality via background LLM call.

    Stores the overall score in memories_data.confidence.  Unproven memories
    with high intrinsic quality get a ranking boost in _rank_memories(),
    helping them escape the never-injected → never-proven cold-start trap.

    Runs in a background thread; failures are silent (confidence stays NULL).
    """
    if not body or not body.strip():
        return

    def _run() -> None:
        try:
            from src.llm.client import pooled_client, extract_text
            from src.utils.llm_json import extract_json_block

            import json as _json

            MAX_RETRIES = 2
            messages: list[dict] = []
            text = ""

            for attempt in range(MAX_RETRIES):
                with pooled_client() as client:
                    if attempt == 0:
                        prompt = _QUALITY_SCORE_PROMPT.format(body=body[:1200])
                        messages = [{"role": "user", "content": prompt}]
                    else:
                        messages.append({"role": "assistant", "content": text})
                        messages.append({
                            "role": "user",
                            "content": f"上次输出无法解析为 JSON：{parse_err}。请只输出合法 JSON，从 {{ 开始。",
                        })
                    response = client.chat(
                        system="你是记忆质量评分器。只输出 JSON，不要任何解释或前言。",
                        messages=messages,
                        max_tokens=350,  # P2: was 180 — too low for CJK JSON (~5 fields × ~30 chars each = ~200 bytes in UTF-8, 500+ in CJK)
                    )
                text = extract_text(response).strip()
                block = extract_json_block(text)
                if block is None:
                    parse_err = "未找到 JSON 块"
                    logger.debug("Quality score attempt %d: no JSON block for '%s': %.80s",
                                attempt + 1, name, text)
                    continue
                try:
                    scores = _json.loads(block)
                except _json.JSONDecodeError as exc:
                    parse_err = str(exc)
                    logger.debug("Quality score attempt %d parse error for '%s': %s",
                                attempt + 1, name, exc)
                    continue

                overall = scores.get("overall", 0)
                if isinstance(overall, (int, float)) and 0 <= overall <= 5:
                    confidence = round(overall / 5.0, 2)  # Normalize to 0-1
                    from src.memory.memory_db import memory_db
                    memory_db.conn.execute(
                        "UPDATE memories_data SET confidence = ? WHERE id = ?",
                        (confidence, memory_id),
                    )
                    memory_db.conn.commit()
                    logger.info(
                        "Quality scored memory '%s' (id=%d): confidence=%.2f "
                        "(clarity=%d, specificity=%d, executability=%d, generalizability=%d)",
                        name, memory_id, confidence,
                        scores.get("screen_clarity", 0),
                        scores.get("error_specificity", 0),
                        scores.get("action_executability", 0),
                        scores.get("generalizability", 0),
                    )
                    return  # success — don't retry
                else:
                    parse_err = f"overall={overall} out of range"
                    logger.debug("Quality score out of range for '%s': %s", name, overall)

            # If we get here, all retries exhausted
            logger.debug("Quality score LLM failed after %d attempts for '%s': %.80s",
                        MAX_RETRIES, name, text)
        except Exception as e:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", f"Quality scoring for memory #{memory_id} failed: {e}")

    import threading
    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ── Game-specific keyword sets for cross-game memory promotion ──
_GAME_SPECIFIC_KW: dict[str, set[str]] = {
    "arknights": {
        "基建", "作战", "终端", "干员", "龙门币", "合成玉", "至纯源石",
        "寻访", "公开招募", "剿灭", "代理作战", "理智", "信用商店",
        "采购中心", "制造站", "贸易站", "发电站", "宿舍", "会客室",
        "控制中枢", "精英化", "专精", "模组", "技能概要", "作战记录",
        "1-7", "CE-5", "LS-5", "SK-5", "PR-", "GT-", "DM-", "RI-",
    },
    "reverse1999": {
        "乐章", "EP", "EP0", "网格", "洞悉", "荒原", "银月", "编队",
        "共鸣", "心相", "雨滴", "活动", "意志", "启示",
    },
}


def _is_game_agnostic(body: str, game: str) -> bool:
    kws = _GAME_SPECIFIC_KW.get(game, set())
    for g, gkws in _GAME_SPECIFIC_KW.items():
        if g != game:
            kws = kws | gkws
    body_lower = body.lower()
    for kw in kws:
        if kw.lower() in body_lower:
            return False
    return True


def _schedule_cross_game_promotion(memory_id: int, body: str, name: str, game: str) -> None:
    """Check if a memory is game-agnostic and promote to _shared."""
    if game == "_shared":
        return
    if not body or not body.strip():
        return

    def _run() -> None:
        try:
            if not _is_game_agnostic(body, game):
                return
            from src.memory.memory_db import memory_db
            existing = memory_db.conn.execute(
                """SELECT id FROM memories_data
                   WHERE game = '_shared' AND body LIKE ? LIMIT 1""",
                (f"%{body[:80]}%",),
            ).fetchone()
            if existing:
                return
            now = datetime.now(tz=timezone.utc).isoformat()
            row = memory_db.conn.execute(
                "SELECT tags FROM memories_data WHERE id = ?", (memory_id,)
            ).fetchone()
            tags = (row["tags"] or "") if row else ""
            shared_name = f"shared_{name}"
            memory_db.conn.execute(
                """INSERT INTO memories_data (name, game, tags, body, source, created, confidence)
                   VALUES (?, '_shared', ?, ?, 'cross_game', ?, ?)""",
                (shared_name, tags, body, now, None),
            )
            memory_db.mark_fts_dirty()
            memory_db.conn.commit()
            logger.info("Cross-game memory promoted to _shared: %s (from %s)", shared_name, game)
        except Exception as e:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", f"Cross-game promotion failed: {e}")

    import threading
    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ---- Programmatic (non-tool) search functions ----
# These are called by the agent loop for auto-injection, not by the LLM.


def _search_memory_internal(
    query: str, games: list[str] | None = None, limit: int = 10
) -> list[dict]:
    """Programmatic memory search — returns raw dicts, not ToolOutput.

    Called by the agent loop to auto-inject relevant memories alongside
    screenshots. Same FTS5 + ranking pipeline as search_memory_tool but
    skips the ToolOutput wrapper and hits increment (caller decides when
    to increment).
    """
    if not query.strip():
        return []

    games = games or ["arknights", "_shared"]
    raw = _search_fts5(query, games, limit=max(limit * 2, 15))
    return _rank_memories(raw, top_n=limit)


def _search_by_dhash(
    screen_hash_hex: str, game: str, threshold: int = 10, limit: int = 5
) -> list[dict]:
    """Find memories with a similar dHash (Hamming-distance visual matching).

    Only scans memories that have a non-NULL screen_hash. Hamming distance
    ≤ threshold means the screen "looks like" the one where the memory was
    originally recorded.

    Args:
        screen_hash_hex: 16-char hex dHash of the current screenshot.
        game: Game namespace (e.g. "arknights").
        threshold: Max Hamming distance (0–64). Default 10.
        limit: Max results to return.

    Returns:
        List of memory dicts with an added "hamming_dist" key, sorted by distance.
    """
    from src.memory.memory_db import memory_db
    from src.utils.dhash import hex_to_dhash, hamming_distance

    try:
        target_hash = hex_to_dhash(screen_hash_hex)
    except (ValueError, TypeError):
        from src.utils.errors import safe_log
        safe_log(logger, "warning", f"Invalid screen_hash_hex for dHash search: {screen_hash_hex}")
        return []

    conn = memory_db.conn
    rows = conn.execute(
        """SELECT id, name, game, tags, body, source, created, hits,
                  help_count, harm_count, last_helpful_at, injected_count, screen_hash
           FROM memories_data
           WHERE game = ? AND screen_hash IS NOT NULL AND deleted_at IS NULL""",
        (game,),
    ).fetchall()

    results: list[dict] = []
    for r in rows:
        if not r["screen_hash"]:
            continue
        # Unpack comma-separated hashes (merged memories anchor to multiple screens)
        best_dist = 65  # above max Hamming distance (64)
        for hash_str in r["screen_hash"].split(","):
            hash_str = hash_str.strip()
            if not hash_str:
                continue
            try:
                dist = hamming_distance(target_hash, hex_to_dhash(hash_str))
            except (ValueError, TypeError):
                continue
            if dist < best_dist:
                best_dist = dist
        if best_dist <= threshold:
            results.append({
                "id": r["id"],
                "name": r["name"],
                "game": r["game"],
                "tags": r["tags"] or "",
                "body": r["body"] or "",
                "source": r["source"] or "",
                "created": r["created"] or "",
                "hits": r["hits"] or 0,
                "help_count": r["help_count"] or 0,
                "harm_count": r["harm_count"] or 0,
                "last_helpful_at": r["last_helpful_at"],
                "injected_count": r["injected_count"] or 0,
                "screen_hash": r["screen_hash"],
                "hamming_dist": best_dist,
            })

    results.sort(key=lambda r: r["hamming_dist"])
    return results[:limit]


def _get_current_screen_hash() -> str | None:
    """Read the current screen's dHash from the active agent's context.

    Returns None when called from a background thread (e.g. memory extract
    sub-agent) where no ADB/screenshot context is available.
    """
    import threading
    ctx = getattr(threading.current_thread(), "_terra_agent_ctx", None)
    if ctx is None:
        return None
    return getattr(ctx.state, "last_injected_dhash", None)


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
        return "arknights"  # ultimate fallback — container not yet initialized


def _semantic_rerank(
    query: str, candidates: list[dict], top_n: int = 3
) -> list[dict]:
    """Re-rank candidate memories using LLM semantic relevance judgment.

    FTS5 and dHash provide lexical/visual recall but can miss semantic
    connections (e.g. "開始作戰" vs "开始行动").  This sends a short prompt
    to the LLM asking it to judge relevance of each candidate against the
    current scene description.

    Only called when there are candidates to re-rank — the call is skipped
    entirely when FTS5 and dHash both return nothing.

    Args:
        query: Current scene description (OCR texts + task context).
        candidates: Combined FTS5/dHash results (up to 10).
        top_n: Number of relevant memories to return.

    Returns:
        Candidates judged relevant, preserving original dicts.
    """
    if not candidates:
        return []

    # Build a compact prompt
    candidate_lines: list[str] = []
    for i, m in enumerate(candidates):
        body_preview = m["body"][:250].replace("\n", " ")
        source = m.get("_source", "text")
        d = m.get("hamming_dist")
        extra = f"[来源:{source}]" + (f" [视觉距离:{d}]" if d is not None else "")
        candidate_lines.append(f"[{i}] {body_preview} {extra}")

    prompt = f"""你是一个相关性判断器。给定当前场景描述和一组候选记忆，判断哪些记忆与**当前屏幕**真正相关。

关键原则：记忆可能在文字上与当前屏幕相似（共享关键词），但其描述的操作和上下文可能不适用于当前屏幕。
例如：一条关于"EP网格中滑动查找EP01"的记忆不应被匹配到"章节列表页"——虽然两者都包含"乐章收录"关键词。

当前场景: {query[:400]}

候选记忆:
{chr(10).join(candidate_lines)}

返回一个 JSON 数组，包含**与当前屏幕真正相关**的记忆编号。无关的或属于不同屏幕/界面的记忆不要包含。都不相关返回 []。
示例: [0, 2]
只返回 JSON 数组，不要其他内容。"""

    try:
        from src.llm.client import pooled_client, extract_text

        with pooled_client() as client:
            response = client.chat(
                system="你是相关性判断器。只输出 JSON 数组，不要任何解释或前言。",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            text = extract_text(response).strip()

        # Parse the JSON array
        import re
        match = re.search(r"\[[\d,\s]*\]", text)
        if match:
            indices = json.loads(match.group())
            relevant = [candidates[i] for i in indices if 0 <= i < len(candidates)]
            if relevant:
                logger.info(
                    "Semantic rerank: %d → %d relevant (query: %.50s)",
                    len(candidates), len(relevant), query,
                )
                return relevant[:top_n]
    except Exception as e:
        logger.warning("Semantic rerank failed, falling back to top-N: %s", e)

    # Fallback: return top candidates by existing score
    return candidates[:top_n]


# ---- Index helpers ----


def _index_memory(name: str, game: str, tags: str, body: str,
                  screen_hash: str | None = None,
                  source: str = "llm_discovery",
                  confidence: float | None = None) -> int | None:
    """Insert or update a memory in the FTS5 index. Returns the row id.

    Before creating, checks for near-duplicate memories via Jaccard similarity.
    Skips creation if similarity > 0.85 (true duplicate), merges if 0.6-0.85.

    Args:
        screen_hash: Optional dHash hex string from the current screenshot.
                     Set when the LLM calls remember() while looking at a screen.
        source: Memory source type.  'llm_discovery' (default) for remember(),
                'action_pattern' for learn_action_pattern(),
                'pattern_miner' for PatternMiner, 'manual' for user-created.
        confidence: Extraction LLM's self-reported confidence (0.0-1.0).
                    Stored for quality tracking and ranking.
    """
    from src.memory.memory_db import memory_db

    try:
        # Phase 1: Check for duplicates before creating
        dup = _check_duplicate_memory(game, body)
        if dup:
            sim = dup.get("_similarity", 0)
            if sim > 0.85:
                logger.info("Memory skipped (near-duplicate, similarity=%.2f): existing=%s",
                           sim, dup.get("name", "?"))
                # Increment hits on the existing memory as a lightweight feedback signal
                _increment_hits(dup["id"])
                return dup["id"]
            elif sim > 0.6:
                _merge_memory(dup["id"], body, tags, screen_hash)
                _increment_hits(dup["id"])
                return dup["id"]

        conn = memory_db.conn
        now_ts = _time.time()
        now = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()
        existing = conn.execute(
            "SELECT id, tags, body, screen_hash, source FROM memories_data WHERE name = ? AND game = ?", (name, game)
        ).fetchone()

        if existing:
            # Preserve existing source when updating (don't downgrade action_pattern → llm_discovery)
            existing_source = existing["source"] or source
            conn.execute(
                """UPDATE memories_data SET tags=?, body=?, screen_hash=?, source=?,
                   updated_at=?
                   WHERE name=? AND game=?""",
                (tags, body, screen_hash, existing_source, now_ts, name, game),
            )
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            conn.commit()
            # Audit log: update
            try:
                memory_db.log_audit(existing["id"], name, game, "update")
            except Exception:
                pass
            # Sync .md file — preserve existing source
            try:
                file_path = _memories_dir_for(game) / f"{name}.md"
                file_path.write_text(
                    _build_md_content(game, tags, body, screen_hash, now, source=existing_source),
                    encoding="utf-8",
                )
            except Exception:
                from src.utils.errors import safe_log
                safe_log(logger, "warning", "Failed to sync memory .md file")
            return existing["id"]
        else:
            cursor = conn.execute(
                """INSERT INTO memories_data (name, game, tags, body, source, created, screen_hash, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, game, tags, body, source, now, screen_hash, confidence),
            )
            conn.commit()
            memory_id = cursor.lastrowid
            # Audit log: create
            try:
                memory_db.log_audit(memory_id, name, game, "create")
            except Exception:
                pass
            # Async: generate embedding for the new memory (background thread)
            _schedule_embedding_generation(memory_id, body)
            # Async: score intrinsic quality for cold-start ranking (background thread)
            _schedule_quality_scoring(memory_id, body, name, game)
            # Async: promote game-agnostic memory to _shared (background thread)
            _schedule_cross_game_promotion(memory_id, body, name, game)
            return memory_id
    except Exception as e:
        logger.warning("Failed to index memory '%s': %s", name, e)
        return None


# ---- Memory lifecycle management (Phase 1) ----


def _check_duplicate_memory(game: str, body: str) -> dict | None:
    """Check if a memory with very similar content already exists.

    Uses FTS5 to find candidates, then computes Jaccard similarity on CJK
    bigrams for accurate duplication detection.

    Returns:
        Existing memory dict if similarity > 0.6, else None.
        Caller should skip creation if > 0.85, or merge if 0.6-0.85.
    """
    import re
    # Extract CJK bigrams for similarity comparison
    def _cjk_bigrams(text: str) -> set[str]:
        cjk = re.sub(r'[a-zA-Z0-9\s,，。.、：:；;！!？?()（）\[\]【】]', '', text)
        bigrams: set[str] = set()
        for i in range(len(cjk) - 1):
            bigrams.add(cjk[i:i+2])
        # Also add word tokens for non-CJK content
        for token in re.findall(r'[a-zA-Z0-9]{2,}', text.lower()):
            bigrams.add(token)
        return bigrams

    new_bigrams = _cjk_bigrams(body)
    if not new_bigrams:
        return None

    # FTS5 search for candidates (lexical overlap)
    candidates = _search_fts5(body, [game], limit=5)
    best_jaccard = 0.0
    best_match: dict | None = None

    for m in candidates:
        existing_bigrams = _cjk_bigrams(m.get("body", ""))
        if not existing_bigrams:
            continue
        intersection = len(new_bigrams & existing_bigrams)
        union = len(new_bigrams | existing_bigrams)
        if union == 0:
            continue
        similarity = intersection / union

        if similarity > best_jaccard:
            best_jaccard = similarity
            best_match = dict(m)
            best_match["_similarity"] = similarity

        if similarity > 0.6:
            logger.info("Duplicate memory found (Jaccard): %s (similarity=%.2f, existing_id=%d)",
                        m.get("name", "?"), similarity, m["id"])
            m["_similarity"] = similarity
            return m

    # ── Fallback: vector semantic similarity for lexically-different duplicates ──
    # Only triggered when FTS5+Jaccard found candidates but none crossed 0.6.
    # This catches semantically-identical memories expressed in different words.
    if best_match and 0.3 <= best_jaccard <= 0.6:
        vec_result = _check_duplicate_via_vector(body, best_match, game)
        if vec_result:
            return vec_result

    return None


def _check_duplicate_via_vector(body: str, best_found: dict, game: str) -> dict | None:
    """Secondary semantic duplicate check using vector embeddings.

    Called when Jaccard similarity is in the gray zone (0.3-0.6) — the text
    might be semantically identical despite different word choices.

    Returns the existing memory dict if vector cosine similarity > 0.85.
    """
    try:
        from src.memory.vector_store import get_vector_store
        vs = get_vector_store()
        if not vs.available:
            return None

        new_blob = vs.encode(body)
        if new_blob is None:
            return None

        from src.memory.memory_db import memory_db
        existing_id = best_found.get("id")
        if existing_id is None:
            return None

        row = memory_db.conn.execute(
            "SELECT embedding FROM memories_data WHERE id = ?", (existing_id,)
        ).fetchone()
        if not row or not row["embedding"]:
            return None

        scored = vs.similarity(new_blob, [(existing_id, row["embedding"])])
        if scored:
            sim = scored[0][1]  # cosine similarity (already normalized)
            if sim > 0.85:
                logger.info(
                    "Duplicate memory found (vector, cosine=%.3f, Jaccard=%.2f): %s (id=%d)",
                    sim, best_found.get("_similarity", 0),
                    best_found.get("name", "?"), existing_id,
                )
                best_found["_similarity"] = max(best_found.get("_similarity", 0), sim)
                best_found["_source"] = "vector_dedup"
                return best_found
    except Exception as e:
        from src.utils.errors import safe_log
        safe_log(logger, "warning", f"Vector duplicate check failed: {e}")

    return None


def _merge_memory(existing_id: int, new_body: str, new_tags: str,
                  new_screen_hash: str | None) -> bool:
    """Merge a new insight into an existing memory.

    Appends the new insight to the body with a timestamp divider.
    Updates tags to the union of old and new. Increments merge_count.
    Preserves the original source from the DB row.
    """
    from src.memory.memory_db import memory_db

    try:
        conn = memory_db.conn
        existing = conn.execute(
            "SELECT name, game, body, tags, screen_hash, source, merge_count FROM memories_data WHERE id = ?",
            (existing_id,),
        ).fetchone()
        if not existing:
            return False

        existing_source = existing["source"] or "llm_discovery"

        now = datetime.now(tz=timezone.utc).isoformat()
        merge_count = (existing["merge_count"] or 0) + 1
        # Cap merges at 3 to prevent unbounded body growth.
        # When the cap is reached, keep the latest version as the main body
        # and preserve the original seed version at the bottom for reference.
        if merge_count >= 3:
            original_body = existing["body"]
            # Extract the seed version: the body BEFORE the first "更新于" divider
            seed_divider = original_body.find("\n\n--- 更新于 ")
            if seed_divider > 0:
                seed_text = original_body[:seed_divider]
            else:
                seed_text = original_body
            # Truncate seed to reasonable size (first 300 chars, last merge retained)
            if len(seed_text) > 300:
                seed_text = seed_text[:300] + "…"
            merged_body = (
                f"[已合并 {merge_count} 次 — 以下为最新版本]\n\n{new_body}"
                f"\n\n--- 原始种子版本 ---\n{seed_text}"
            )
        else:
            merged_body = existing["body"] + f"\n\n--- 更新于 {now} ---\n\n{new_body}"

        # Union of tags
        old_tags = set(t.strip() for t in (existing["tags"] or "").split(",") if t.strip())
        new_tag_set = set(t.strip() for t in new_tags.split(",") if t.strip())
        merged_tags = ", ".join(sorted(old_tags | new_tag_set))

        # Keep existing screen_hash, add new one only if different.
        # Comma-separated for multi-screen visual matching (same trap on multiple screens).
        existing_hashes = set(
            h.strip() for h in (existing["screen_hash"] or "").split(",") if h.strip()
        )
        if new_screen_hash:
            existing_hashes.add(new_screen_hash)
        merged_hash = ", ".join(sorted(existing_hashes)) if existing_hashes else None

        conn.execute(
            """UPDATE memories_data SET body=?, tags=?, screen_hash=?, merge_count=?,
               updated_at=?
               WHERE id=?""",
            (merged_body, merged_tags, merged_hash, merge_count,
             _time.time(), existing_id),
        )
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        conn.commit()
        # Audit log: merge
        try:
            memory_db.log_audit(existing_id, existing["name"], existing["game"], "merge")
        except Exception:
            pass

        # Also update the .md file — preserve original source
        try:
            mem_game = existing["game"] or "arknights"
            mem_name = existing["name"]
            file_path = _memories_dir_for(mem_game) / f"{mem_name}.md"
            file_path.write_text(
                _build_md_content(mem_game, merged_tags, merged_body, merged_hash, now, source=existing_source),
                encoding="utf-8",
            )
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "Failed to sync merged memory .md file")

        logger.info("Memory merged: id=%d (merge_count=%d)", existing_id, merge_count)
        return True
    except Exception as e:
        logger.warning("Failed to merge memory %d: %s", existing_id, e)
        return False


def _build_md_content(game: str, tags: str, body: str, screen_hash: str | None, created: str = "",
                     source: str = "llm_discovery", extra_yaml: dict[str, str] | None = None) -> str:
    """Build the markdown content for a memory .md file."""
    yaml_lines = [
        f"game: {game}",
        f"tags: [{', '.join(t.strip() for t in tags.split(',') if t.strip())}]",
        f"source: {source}",
    ]
    if screen_hash:
        yaml_lines.append(f"screen_hash: {screen_hash}")
    if extra_yaml:
        for k, v in extra_yaml.items():
            yaml_lines.append(f"{k}: {v}")
    yaml_lines.append(f"created: {created or datetime.now(tz=timezone.utc).isoformat()}")
    return "---\n" + "\n".join(yaml_lines) + "\n---\n\n" + body


def cleanup_stale_memories(game: str = "arknights", stale_days: int = 7,
                           low_success_ratio: float = 0.2,
                           low_help_harm_ratio: float = 0.5) -> int:
    """Remove stale memories that are never used or have low success correlation.

    Game UIs change frequently, so the default stale_days is 7 (one week).
    Longer-lived memories should have the "永久" or "permanent" tag.

    Staleness criteria (either triggers deletion):
    1. Never injected, 0 hits, created > stale_days ago (ghost memory, 1 week)
    2. Injected >= 5 times with success ratio < low_success_ratio,
       AND created > stale_days*2 ago (proven-useless memory, 2 weeks)
    3. (Phase 1) Injected >= 8 times, has help/harm data, and
       help_count < harm_count * low_help_harm_ratio (proven-harmful memory)

    Memories with tag "永久" or "permanent" are NEVER deleted.
    The LLM can add these tags to memories about stable UI layouts.

    Returns count of deleted memories.
    """
    from src.memory.memory_db import memory_db

    try:
        conn = memory_db.conn
        now = datetime.now(tz=timezone.utc)
        cutoff_new = int((now.timestamp() - stale_days * 86400))
        cutoff_old = int((now.timestamp() - stale_days * 2 * 86400))

        # Find stale memories (include Phase 1 help/harm columns)
        rows = conn.execute(
            """SELECT id, name, game, tags, injected_count, injected_success_count,
               hits, created, help_count, harm_count FROM memories_data WHERE game = ?""",
            (game,),
        ).fetchall()

        stale_ids: list[int] = []
        for r in rows:
            # Tag-based protection: "永久" or "permanent" tags skip cleanup entirely
            tags = (r["tags"] or "").lower()
            if "永久" in tags or "permanent" in tags:
                continue

            created_ts = 0
            try:
                created_ts = datetime.fromisoformat(r["created"]).timestamp()
            except (ValueError, TypeError):
                pass

            injected = r["injected_count"] or 0
            injected_success = r["injected_success_count"] or 0
            hits = r["hits"] or 0
            help_count = r["help_count"] or 0
            harm_count = r["harm_count"] or 0

            # Criterion 1: never used (not injected AND never searched manually)
            if injected == 0 and hits == 0 and created_ts > 0 and created_ts < cutoff_new:
                stale_ids.append(r["id"])
                logger.debug("Stale memory (never used): %s/%s", r["game"], r["name"])
                continue

            # Criterion 2: low success rate after substantial injection sample
            if injected >= 5 and created_ts > 0 and created_ts < cutoff_old:
                if injected > 0 and (injected_success / injected) < low_success_ratio:
                    stale_ids.append(r["id"])
                    logger.debug("Stale memory (low success %.2f after %d injections): %s/%s",
                                injected_success / injected, injected, r["game"], r["name"])

            # Criterion 3 (Phase 1): proven-harmful after sufficient sample.
            # Require at least 6 injection attempts and harm_count >= 2 to
            # avoid deleting memories with a single bad rating (was 12/3).
            total_scored = help_count + harm_count
            if injected >= 6 and harm_count >= 2 and total_scored > 0:
                if (help_count / max(harm_count, 1)) < low_help_harm_ratio:
                    stale_ids.append(r["id"])
                    logger.debug("Stale memory (harmful: help=%d harm=%d after %d injections): %s/%s",
                                help_count, harm_count, injected, r["game"], r["name"])

        if not stale_ids:
            return 0

        # Delete DB rows — delete injection_log first, then audit_log, then memories_data
        # (injection_log has FK → memories_data, audit_log has FK → memories_data)
        if stale_ids:
            placeholders = ",".join("?" * len(stale_ids))
            conn.execute(f"DELETE FROM injection_log WHERE memory_id IN ({placeholders})", stale_ids)
            conn.execute(f"DELETE FROM memory_audit_log WHERE memory_id IN ({placeholders})", stale_ids)
            conn.execute(f"DELETE FROM memories_data WHERE id IN ({placeholders})", stale_ids)
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        conn.commit()

        # Delete .md files
        for r in rows:
            if r["id"] in stale_ids:
                try:
                    file_path = _memories_dir_for(r["game"]) / f"{r['name']}.md"
                    if file_path.exists():
                        file_path.unlink()
                except Exception:
                    pass

        # Purge soft-deleted memories older than 30 days (hard delete)
        try:
            purged = memory_db.purge_soft_deleted(older_than_days=30)
            if purged:
                logger.info("Purged %d soft-deleted memories (game=%s)", purged, game)
        except Exception as ex:
            logger.debug("Soft-delete purge skipped: %s", ex)

        logger.info("Cleaned up %d stale memories (game=%s)", len(stale_ids), game)
        return len(stale_ids)
    except Exception as e:
        logger.warning("Stale memory cleanup failed: %s", e)
        return 0


# ---- Tools ----

def remember_tool(insight: str, tags: str = "", screen_hash: str = "",
                 game: str = "") -> ToolOutput:
    """Save a discovery or insight as a memory for future reference.

    Call this when you discover a pattern, rule, or UI layout detail that
    would be useful in future tasks. The memory is saved per-game and
    searchable via search_memory.

    Args:
        insight: What you learned (natural language, any length)
        tags: Comma-separated keywords for categorization (e.g. "EP, 乐章, 网格")
        screen_hash: Optional dHash hex string. When called from the main loop,
                     auto-captured from the current screen. When called from the
                     background extraction sub-agent, passed explicitly from the
                     failure signal's dHash field.
        game: Game ID. Auto-detected from agent context; pass explicitly when
              calling from background threads (watcher, background review) where
              no agent context is available.
    """
    if not insight.strip():
        return ToolOutput(text=json.dumps({"success": False, "error": "insight is empty"}, ensure_ascii=False))

    resolved_game = game if game else _get_current_game()

    import time as _time
    name = f"m{int(_time.time() * 1_000_000)}"  # Microsecond timestamp avoids same-second collisions
    body = insight.strip()
    tags_clean = tags.strip() if tags else ""

    # Use explicit screen_hash if provided, otherwise auto-capture from current screen.
    # Auto-capture returns None when called from background threads (no ADB context).
    final_screen_hash = screen_hash if screen_hash else _get_current_screen_hash()

    # Index first — _index_memory does dedup and may return an existing ID.
    # Only write the .md file if the memory is genuinely new (not a duplicate).
    idx_id = _index_memory(name, resolved_game, tags_clean, body, final_screen_hash)
    if idx_id is None:
        return ToolOutput(text=json.dumps({"success": False, "error": "Failed to index memory"}, ensure_ascii=False))

    # Fetch the newly-inserted row to check if it was a dedup
    from src.memory.memory_db import memory_db as _mdb
    row = _mdb.conn.execute(
        "SELECT id, name FROM memories_data WHERE id = ?", (idx_id,)
    ).fetchone()
    is_new = row and row["name"] == name  # Dedup returns existing name ≠ our new name

    if is_new:
        _ensure_dir(_memories_dir_for(resolved_game))
        file_path = _memories_dir_for(resolved_game) / f"{name}.md"
        yaml_lines = [
            f"game: {resolved_game}",
            f"tags: [{', '.join(t.strip() for t in tags_clean.split(',') if t.strip())}]",
            "source: llm_discovery",
            f"created: {datetime.now(tz=timezone.utc).isoformat()}",
        ]
        if final_screen_hash:
            yaml_lines.insert(-1, f"screen_hash: {final_screen_hash}")
        content = "---\n" + "\n".join(yaml_lines) + "\n---\n\n" + body
        file_path.write_text(content, encoding="utf-8")
        logger.info("Memory saved: %s/%s.md (id=%s, hash=%s)", resolved_game, name, idx_id, final_screen_hash)
    else:
        logger.info("Memory deduped to existing: id=%s (body similarity match)", idx_id)

    return ToolOutput(text=json.dumps({
        "success": True,
        "name": name,
        "game": resolved_game,
        "message": "Memory saved and indexed for future search.",
    }, ensure_ascii=False))


def search_memory_tool(query: str) -> ToolOutput:
    """Search past memories for relevant experience.

    Call this when facing an unfamiliar page layout, encountering a UI
    element you haven't seen before, or when an action didn't work as
    expected. Returns top-5 memories ranked by relevance and usefulness.

    Args:
        query: What you're looking for (natural language, e.g. "EP 章节 导航")
    """
    if not query.strip():
        return ToolOutput(text=json.dumps({"success": False, "error": "query is empty"}, ensure_ascii=False))

    game = _get_current_game()
    games = [game, "_shared"]
    raw = _search_fts5(query, games, limit=10)
    ranked = _rank_memories(raw, top_n=5)

    items = []
    for r in ranked:
        # Increment hits for returned memories
        _increment_hits(r["id"])
        items.append({
            "name": r["name"],
            "game": r["game"],
            "tags": r["tags"],
            "body": r["body"][:300],  # Preview: first 300 chars
            "source": r["source"],
        })

    logger.info("search_memory: '%s' → %d results (searched %s)", query[:50], len(items), games)

    return ToolOutput(text=json.dumps({
        "success": True,
        "query": query,
        "count": len(items),
        "results": items,
    }, ensure_ascii=False))


def forget_memory_tool(query: str) -> ToolOutput:
    """Delete a memory matching the query. Use when a memory is wrong or outdated.

    Args:
        query: Keywords to match against memory name/tags/body
    """
    if not query.strip():
        return ToolOutput(text=json.dumps({"success": False, "error": "query is empty"}, ensure_ascii=False))

    from src.memory.memory_db import memory_db

    game = _get_current_game()
    games = [game, "_shared"]
    raw = _search_fts5(query, games, limit=5)
    if not raw:
        return ToolOutput(text=json.dumps({"success": False, "error": "No matching memories found"}, ensure_ascii=False))

    deleted = []
    for r in raw[:3]:  # Max 3 at a time
        name = r["name"]
        mem_game = r["game"]
        mid = r["id"]
        try:
            # Soft delete: mark as deleted, preserves data for audit/recovery
            ok = memory_db.soft_delete_memory(mid)
            if ok:
                deleted.append(f"{mem_game}/{name}")
        except Exception as e:
            logger.warning("Failed to delete memory %s: %s", name, e)

    memory_db.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
    memory_db.conn.commit()

    logger.info("forget_memory: deleted %s", deleted)
    return ToolOutput(text=json.dumps({
        "success": True,
        "deleted": deleted,
    }, ensure_ascii=False))


registry.register(
    name="remember",
    description=(
        "Save coordinates, button positions, or UI layout details for future tasks.\n"
        "**硬触发 — 以下情况必须立即调 remember()**：\n"
        "1. 同一个按钮连续失败 2+ 次后终于成功 → 记住这次的定位方式（nth/坐标/百分比）和按钮名\n"
        "2. 用 magnify → tap_magnified 精确定位了按钮 → 记住放大图坐标和屏幕实际坐标\n"
        "3. adb_tap_position 用百分比首次成功点到之前失败过的目标 → 记住百分比\n"
        "4. OCR 总是读不到某个按钮文字（如含符号的「挑战>>」「每日宣传」）→ 记住它的屏幕位置"
    ),
    parameters={
        "type": "object",
        "properties": {
            "insight": {"type": "string", "description": "What you learned. Include exact coordinates, nth values, percentages, and the button name."},
            "tags": {"type": "string", "description": "Comma-separated keywords (e.g. '挑战按钮, 时尚对决, 坐标')"},
            "screen_hash": {"type": "string", "description": "Optional dHash of the current screen. Auto-captured; only pass if the system doesn't fill it."},
            "game": {"type": "string", "description": "Game ID. Auto-detected; only pass when you're sure the auto-detection is wrong."},
        },
        "required": ["insight"],
    },
    handler=remember_tool,
)

registry.register(
    name="search_memory",
    description="Manually search past memories by keyword. Normally the system auto-injects relevant memories alongside screenshots — use this only when you need to find something specific not already shown.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What you're looking for (e.g. 'EP chapter navigation', 'checkbox toggle')"},
        },
        "required": ["query"],
    },
    handler=search_memory_tool,
)

registry.register(
    name="forget_memory",
    description="Delete a wrong or outdated memory.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords to match the memory to delete"},
        },
        "required": ["query"],
    },
    handler=forget_memory_tool,
)
