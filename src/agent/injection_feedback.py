"""Injection feedback tracker — per-injection outcome tracking (Phase 1).

Mirrors ConfidenceManager's adaptive-learning pattern.  Tracks each memory
injection during task execution, then scores them when the task finishes
based on whether the agent recovered after seeing the memory.

Scoring rules:
  - was_helpful=1: agent had a successful action within 3 turns after injection
  - was_helpful=-1: agent hit a failure signal within 3 turns after injection
  - was_helpful=0: no clear evidence either way (NULL in DB)

After scoring, updates memories_data.help_count / harm_count for lifecycle mgmt.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Window for attribution: how many messages after injection do we look at
# to determine if the memory helped?  Increased from 3 to 6 because agent
# recovery after a user correction typically takes 5-8 turns.
ATTRIBUTION_WINDOW = 2

# ── OCR overlap helpers for attribution weighting ──

def _ocr_bigrams(text: str) -> set[str]:
    """Extract CJK bigrams and alpha tokens from text for OCR overlap."""
    import re as _re
    cjk = _re.sub(r'[a-zA-Z0-9\s,，。.、：:；;！!？?()（）\[\]【】]', '', text)
    bigrams: set[str] = set()
    for i in range(len(cjk) - 1):
        bigrams.add(cjk[i:i+2])
    for token in _re.findall(r'[a-zA-Z0-9]{2,}', text.lower()):
        bigrams.add(token)
    return bigrams


def _extract_ocr_from_message(msg: dict) -> set[str]:
    """Extract OCR keywords from a screen-injection message label.

    Screen injections have labels like:
      "[系统自动截图 — 当前屏幕 — HASH:abc123]\\nOCR:基建, 制造站, 贸易站"
    """
    import re as _re
    content = msg.get("content", "")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", ""))
                break
    m = _re.search(r"OCR:([\w一-鿿, /-]+)", text)
    if m:
        words = {w.strip() for w in m.group(1).split(",") if len(w.strip()) >= 2}
        return words
    return set()


class InjectionFeedbackTracker:
    """Per-task tracker that records and scores memory injections.

    Attached to AgentState during a single task execution.  Call
    record_injection() when a memory is injected, and score_injections()
    when the task completes.
    """

    def __init__(self, task_execution_id: int) -> None:
        self._task_id = task_execution_id
        # Map: injection_log_id → (memory_id, injected_at, assistant_msg_index)
        self._pending: dict[int, tuple[int, float, int]] = {}
        # Track the conversation_history index at time of each injection
        self._injection_indices: dict[int, int] = {}  # injection_log_id → msg_index
        # Track the list of all injection log IDs for bulk scoring
        self._all_ids: list[int] = []

    def record_injection(self, memory_id: int, screen_dhash: str | None = None,
                         conversation_len: int = 0) -> int | None:
        """Log an injection and track its position in conversation history.

        Also increments memories_data.injected_count at injection time
        (moved from _log_execution's blanket batch — this is per-injection,
        not per-task).

        Args:
            memory_id: The memories_data.id that was injected.
            screen_dhash: dHash of the screen when injected (for context).
            conversation_len: Current length of conversation_history (used
                              to determine how many messages later to look).

        Returns:
            The injection_log row id, or None on failure.
        """
        from src.memory.memory_db import memory_db

        try:
            log_id = memory_db.log_injection(memory_id, self._task_id, screen_dhash)
        except Exception as e:
            logger.warning("Failed to log injection for memory %d: %s", memory_id, e)
            return None

        # Increment injected_count at injection time (atomic, per-memory)
        try:
            memory_db.conn.execute(
                "UPDATE memories_data SET injected_count = injected_count + 1 WHERE id = ?",
                (memory_id,),
            )
            memory_db.conn.commit()
        except Exception as e:
            logger.debug("Failed to increment injected_count for memory %d: %s", memory_id, e)

        self._pending[log_id] = (memory_id, 0.0, conversation_len)
        self._injection_indices[log_id] = conversation_len
        self._all_ids.append(log_id)
        return log_id

    def score_injections(self, conversation_history: list[dict[str, Any]],
                         failure_signals: list[dict[str, Any]],
                         task_success: bool) -> None:
        """Post-task: analyze conversation to determine which injections helped.

        For each recorded injection:
        - Look at the next ATTRIBUTION_WINDOW assistant messages.
        - If any contain a successful tool call → was_helpful=1
        - If a failure_signal was recorded within that window → was_helpful=-1
        - Otherwise → leave as NULL (no evidence)

        Then update memory help/harm stats in the DB.
        """
        from src.memory.memory_db import memory_db

        # Build a set of message indices where failure signals occurred
        failure_msg_indices: set[int] = set()
        for sig in failure_signals:
            # failure_signals use 'iteration' which corresponds roughly to
            # message index / 2 (each iteration = assistant + user/tool_result)
            # We use iteration * 2 as a conservative upper bound.
            iter_idx = sig.get("iteration", 0)
            failure_msg_indices.add(iter_idx * 2)

        # Collect per-memory outcomes: which were helpful, which were harmful
        helpful_memory_ids: set[int] = set()
        harmful_memory_ids: set[int] = set()
        weak_helpful_count = 0

        for log_id, (memory_id, _, conv_idx) in self._pending.items():
            was_helpful: int | None = None  # NULL = no evidence
            is_weak: bool = False

            # ── Load memory body for OCR overlap check ──
            mem_keywords: set[str] = set()
            try:
                row = memory_db.conn.execute(
                    "SELECT body FROM memories_data WHERE id = ?", (memory_id,)
                ).fetchone()
                if row and row["body"]:
                    mem_keywords = _ocr_bigrams(row["body"])
            except Exception:
                pass

            # ── Collect window OCR texts ──
            window_ocr_words: set[str] = set()
            for offset in range(ATTRIBUTION_WINDOW):
                check_idx = conv_idx + offset + 1
                if check_idx >= len(conversation_history):
                    break
                window_ocr_words |= _extract_ocr_from_message(conversation_history[check_idx])

            # Check the next ATTRIBUTION_WINDOW messages for success/failure evidence.
            for offset in range(ATTRIBUTION_WINDOW):
                check_idx = conv_idx + offset + 1
                if check_idx >= len(conversation_history):
                    break

                msg = conversation_history[check_idx]
                content = msg.get("content")

                # Check for tool results that indicate success
                # Use robust JSON parsing instead of fragile substring match
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            result_text = str(block.get("content", ""))
                            try:
                                import json as _json
                                data = _json.loads(result_text)
                                if isinstance(data, dict) and data.get("success") is True:
                                    was_helpful = 1
                                    break
                            except (_json.JSONDecodeError, TypeError):
                                # Fallback to substring for non-JSON results
                                if '"success": true' in result_text.lower() or '"success":true' in result_text.lower():
                                    was_helpful = 1
                                    break
                    if was_helpful:
                        break

                # Check for failure signal in this window
                if check_idx in failure_msg_indices:
                    was_helpful = -1
                    break

            # ── OCR-weighted confidence check ──
            if was_helpful is not None and mem_keywords and window_ocr_words:
                overlap = len(mem_keywords & window_ocr_words)
                total = len(mem_keywords)
                if total > 0:
                    overlap_ratio = overlap / total
                    if overlap_ratio < 0.3:
                        is_weak = True
                        logger.debug(
                            "Injection #%d (mem=%d): weak attribution (overlap=%.2f, was_helpful=%d)",
                            log_id, memory_id, overlap_ratio, was_helpful,
                        )

            # Commit score to DB
            if was_helpful is not None:
                try:
                    memory_db.score_injection(log_id, was_helpful == 1)
                except Exception as e:
                    logger.debug("Failed to score injection %d: %s", log_id, e)
                if was_helpful == 1:
                    if is_weak:
                        weak_helpful_count += 1
                    else:
                        helpful_memory_ids.add(memory_id)
                else:
                    harmful_memory_ids.add(memory_id)

        # Update aggregate stats per memory (help_count, harm_count, injected_success_count)
        scored_memory_ids = helpful_memory_ids | harmful_memory_ids
        for mid in scored_memory_ids:
            try:
                memory_db.update_memory_help_stats(mid)
            except Exception as e:
                logger.debug("Failed to update help stats for memory %d: %s", mid, e)

        # Increment injected_success_count only for proven-helpful memories
        if helpful_memory_ids:
            try:
                ids = list(helpful_memory_ids)
                placeholders = ",".join("?" * len(ids))
                memory_db.conn.execute(
                    f"UPDATE memories_data SET injected_success_count = injected_success_count + 1 WHERE id IN ({placeholders})",
                    ids,
                )
                memory_db.conn.commit()
            except Exception as e:
                logger.debug("Failed to update injected_success_count: %s", e)

        logger.info("Injection feedback: scored %d/%d injections (helpful=%d harmful=%d weak=%d) for task #%d (success=%s)",
                    len(scored_memory_ids), len(self._all_ids),
                    len(helpful_memory_ids), len(harmful_memory_ids),
                    weak_helpful_count,
                    self._task_id, task_success)
