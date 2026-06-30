"""ExecutionLogger — task execution recording and learning metrics.

Extracted from TerraAgent._log_execution().

Handles:
    1. Writing to history_db.log_execution()
    2. Completing task_executions record (memory_db.complete_task_execution())
    3. Scoring injection helpfulness (InjectionFeedbackTracker.score_injections())
    4. Saving conversation JSON snapshot to data/logs/
    5. Scheduling the periodic PatternMiner
"""

from __future__ import annotations

import json
import logging
import time as _time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionLogger:
    """Unified execution log: history + task_execution + injection scoring +
    JSON archive + PatternMiner scheduling + user corrections index."""

    # Path for lightweight user corrections index (read by PatternMiner)
    _CORRECTIONS_INDEX = "user_corrections.jsonl"

    def __init__(self, history_db: Any, memory_db: Any, config: Any) -> None:
        self._history_db = history_db
        self._memory_db = memory_db
        self._config = config

    def _append_user_corrections(self, state: Any, task_description: str) -> None:
        """Extract user correction messages and append to a lightweight JSONL index.

        PatternMiner's find_unlearned_guidance reads this file instead of
        scanning full conversation JSON logs (each multi-MB).  One line per
        correction — orders of magnitude faster.
        """
        import re as _re
        corrections: list[dict] = []
        seen: set[str] = set()
        for msg in state.conversation_history:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            # Extract user corrections from [用户指令] / [用户回复] markers
            for marker in ("[用户指令 — 必须执行] ", "[用户指令] ", "[用户回复] "):
                if marker in content:
                    text = content.split(marker, 1)[-1].strip()
                    if text and len(text) > 5 and text not in seen:
                        seen.add(text)
                        corrections.append({
                            "ts": _time.strftime("%Y%m%d_%H%M%S"),
                            "game": state.game,
                            "task": task_description[:200],
                            "correction": text[:300],
                        })
                    break

        if not corrections:
            return

        log_dir = Path(self._config.DATA_DIR) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        index_path = log_dir / self._CORRECTIONS_INDEX
        with open(index_path, "a", encoding="utf-8") as f:
            for c in corrections:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    def log(
        self,
        user_message: str,
        result: dict[str, Any],
        duration: float,
        state: Any,         # AgentState
        matching_skills: list[dict[str, Any]],
        task_type: str = "unknown",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Unified execution log entry — replaces _log_execution()."""

        # ── 1. History DB ──
        try:
            self._history_db.log_execution(
                game=state.game,
                task_type=result.get("task_type", task_type),
                task_description=user_message[:200],
                success=result.get("success", False),
                iterations=state.iteration_count,
                duration_seconds=duration,
                details={
                    "result": result,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )
        except Exception as e:
            logger.warning("Failed to log execution: %s", e)

        # ── 2. Task execution record (Learning Engine Phase 1) ──
        if state.task_execution_id:
            try:
                failure_types = ",".join(
                    sorted(set(s.signal_type for s in state.failure_signals))
                ) if state.failure_signals else ""

                skill_name = ""
                if matching_skills:
                    skill_name = ",".join(s.get("name", "") for s in matching_skills if s.get("name"))

                self._memory_db.complete_task_execution(
                    task_id=state.task_execution_id,
                    success=result.get("success", False),
                    iterations=state.iteration_count,
                    duration=duration,
                    failure_signal_types=failure_types,
                    skill_name=skill_name,
                    skill_fast_chain_success=state.skill_fast_chain_result,
                    user_interrupted=state._user_msg_count > 0 or state._ask_user_count > 0,
                )
                logger.debug("Task execution #%d completed", state.task_execution_id)
            except Exception as e:
                from src.utils.errors import safe_log
                safe_log(logger, "warning", f"Failed to complete task execution record: {e}")

        # ── 3. Injection scoring ──
        if state.injection_feedback_tracker:
            try:
                from dataclasses import asdict
                state.injection_feedback_tracker.score_injections(
                    conversation_history=state.conversation_history,
                    failure_signals=[asdict(s) for s in state.failure_signals],
                    task_success=result.get("success", False),
                )
            except Exception as e:
                from src.utils.errors import safe_log
                safe_log(logger, "warning", f"Failed to score injections: {e}")

        # ── 4. Save conversation JSON ──
        try:
            log_dir = Path(self._config.DATA_DIR) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = _time.strftime("%Y%m%d_%H%M%S")
            game_short = (state.game or "unknown")[:8]
            device_short = (state.device_serial or "unknown").replace("emulator-", "").replace("127.0.0.1:", "")[:10]
            outcome = "OK" if result.get("success") else "FAIL"
            fname = f"conv_{game_short}_{device_short}_{ts}_{outcome}.json"
            dump = {
                "timestamp": ts,
                "game": state.game,
                "device_serial": state.device_serial,
                "task": user_message,
                "success": result.get("success", False),
                "iterations": state.iteration_count,
                "duration_seconds": round(duration, 1),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "messages": state.conversation_history,
            }
            (log_dir / fname).write_text(
                json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed to save conversation: %s", e)

        # ── 5. Lightweight user corrections index (for PatternMiner) ──
        # Appends user corrections to a minimal JSONL file so PatternMiner's
        # find_unlearned_guidance can read this instead of scanning full
        # conversation logs (which may be multi-MB each).
        if state._user_msg_count > 0:
            try:
                self._append_user_corrections(state, user_message)
            except Exception as e:
                logger.debug("Failed to save user corrections index: %s", e)
