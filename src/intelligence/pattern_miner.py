"""Pattern Miner — cross-task statistical analysis engine (Phase 2).

Runs periodically (every N tasks or daily) to discover recurring patterns
across multiple task executions.  Each miner produces findings that can
be saved as memories (source='pattern_miner') for future injection.

Analyzers:
  - find_recurring_failures(): Cluster failure signals to find systemic issues
  - find_stale_skills(): Detect skills whose fast_chain success rate is dropping
  - find_high_value_memories(): Identify memories with high help/harm ratio
  - find_unlearned_guidance(): Find ask_user interactions not captured as memories
"""

from __future__ import annotations

import json
import logging
import re
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import config

logger = logging.getLogger(__name__)


class PatternMiner:
    """Analyzes execution history and injection logs to discover patterns."""

    def __init__(self, game: str = "arknights") -> None:
        self.game = game

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[dict]:
        """Run all miners. Returns list of finding dicts."""
        findings: list[dict] = []
        for method in [
            self.find_recurring_failures,
            self.find_stale_skills,
            self.find_deprecated_skills,
            self.find_high_value_memories,
            self.find_unlearned_guidance,
        ]:
            try:
                result = method()
                if result:
                    findings.extend(result)
            except Exception as e:
                logger.warning("Pattern miner %s failed: %s", method.__name__, e)
        return findings

    def run_and_save(self) -> int:
        """Run all miners and save findings as memories. Returns count saved."""
        findings = self.run()
        if not findings:
            logger.info("Pattern miner: no findings for game=%s", self.game)
            return 0

        count = 0
        for f in findings:
            try:
                self._save_finding(f)
                count += 1
            except Exception as e:
                logger.debug("Failed to save pattern finding: %s", e)

        logger.info("Pattern miner: saved %d findings for game=%s", count, self.game)
        return count

    # ------------------------------------------------------------------
    # Analyzer 1: Recurring failure patterns
    # ------------------------------------------------------------------

    def find_recurring_failures(self, lookback_tasks: int = 50,
                                min_occurrences: int = 3) -> list[dict]:
        """Find failure signals that keep happening across different tasks.

        Groups failure signals by (signal_type, top OCR texts cluster).
        """
        from src.memory.memory_db import memory_db

        rows = memory_db.conn.execute(
            """SELECT failure_signal_types, task_type, task_description
               FROM task_executions
               WHERE game=? AND failure_signal_types != ''
               ORDER BY finished_at DESC LIMIT ?""",
            (self.game, lookback_tasks),
        ).fetchall()

        if not rows:
            return []

        # Count signal types across tasks
        signal_counts: dict[str, int] = {}
        for r in rows:
            if not r["failure_signal_types"]:
                continue
            for sig in r["failure_signal_types"].split(","):
                sig = sig.strip()
                if sig:
                    signal_counts[sig] = signal_counts.get(sig, 0) + 1

        findings: list[dict] = []
        for sig_type, count in signal_counts.items():
            if count >= min_occurrences:
                # Calculate what % of failure-tasks have this signal
                total_failure_tasks = len(rows)
                pct = round(count / max(total_failure_tasks, 1) * 100)
                findings.append({
                    "type": "recurring_failure",
                    "signal_type": sig_type,
                    "occurrences": count,
                    "total_failure_tasks": total_failure_tasks,
                    "percentage": pct,
                    "body": (
                        f"【系统发现】在最近 {total_failure_tasks} 个有失败信号的任务中，"
                        f"'{sig_type}' 出现了 {count} 次（{pct}%）。\n"
                        f"这是目前最常见的卡住类型。如果遇到此类卡住，优先检查是否有相关的"
                        f"[经验模式] 或 [关联记忆] 可以参考。\n"
                        f"建议：下一次此信号触发时，agent 应立即切换策略而非继续重试。"
                    ),
                    "tags": f"pattern_miner, recurring, {sig_type}, 永久",
                })

        return findings

    # ------------------------------------------------------------------
    # Analyzer 2: Stale skill detection
    # ------------------------------------------------------------------

    def find_stale_skills(self, lookback_tasks: int = 50,
                          min_executions: int = 3,
                          stale_threshold: float = 0.5) -> list[dict]:
        """Find skills whose fast_chain success rate has dropped."""
        from src.memory.memory_db import memory_db

        rows = memory_db.conn.execute(
            """SELECT skill_name, skill_fast_chain_success, finished_at
               FROM task_executions
               WHERE game=? AND skill_name != ''
               ORDER BY finished_at DESC LIMIT ?""",
            (self.game, lookback_tasks),
        ).fetchall()

        if not rows:
            return []

        # Group by skill name (multi-skill tasks store comma-separated names)
        skill_stats: dict[str, dict] = {}
        for r in rows:
            raw_names = (r["skill_name"] or "").split(",")
            for name in (n.strip() for n in raw_names if n.strip()):
                if name not in skill_stats:
                    skill_stats[name] = {"total": 0, "fast_chain_ok": 0}
                skill_stats[name]["total"] += 1
                if r["skill_fast_chain_success"] == 1:
                    skill_stats[name]["fast_chain_ok"] += 1

        findings: list[dict] = []
        for name, stats in skill_stats.items():
            if stats["total"] >= min_executions:
                success_rate = stats["fast_chain_ok"] / stats["total"]
                if success_rate < stale_threshold:
                    findings.append({
                        "type": "stale_skill",
                        "skill_name": name,
                        "executions": stats["total"],
                        "fast_chain_success_rate": round(success_rate, 2),
                        "body": (
                            f"【系统发现】技能 '{name}' 最近 {stats['total']} 次执行中，"
                            f"fast_chain 成功率仅 {round(success_rate * 100)}%。"
                            f"坐标可能已过期（UI 布局变更或分辨率不匹配）。"
                            f"下次匹配到此技能时，建议使用手动执行而非 skill_run。"
                        ),
                        "tags": f"pattern_miner, stale_skill, {name}, 永久",
                    })

        return findings

    # ------------------------------------------------------------------
    # Analyzer 2b: Deprecated skill cleanup
    # ------------------------------------------------------------------

    def find_deprecated_skills(self, retention_days: int = 30) -> list[dict]:
        """Find deprecated skills older than retention_days and suggest cleanup.

        When a skill file is marked ``deprecated: true`` in frontmatter and
        hasn't been used or updated for 30+ days, it's safe to archive.
        """
        from pathlib import Path
        from config.settings import config as _cfg
        from src.skills.parser import SkillParser

        skills_dir = Path(_cfg.DATA_DIR) / "skills" / self.game
        if not skills_dir.exists():
            return []

        now = _time.time()
        cutoff = now - (retention_days * 86400)
        findings: list[dict] = []

        for md_path in skills_dir.rglob("*.md"):
            try:
                content = md_path.read_text(encoding="utf-8")
                skill = SkillParser.parse(content)
            except Exception:
                continue

            # Check frontmatter for deprecated marker
            if not content.startswith("---"):
                continue
            frontmatter_end = content.find("---", 3)
            if frontmatter_end < 0:
                continue
            fm = content[3:frontmatter_end].strip()
            if "deprecated: true" not in fm:
                continue

            # Check file modification time
            try:
                mtime = md_path.stat().st_mtime
            except OSError:
                mtime = now

            if mtime > cutoff:
                continue  # Still recent — don't suggest cleanup yet

            days_old = int((now - mtime) / 86400)
            name = skill.get("name", md_path.stem)
            replaces = ""
            for line in fm.split("\n"):
                if line.startswith("replaces:"):
                    replaces = line.split(":", 1)[1].strip()

            findings.append({
                "type": "deprecated_skill_cleanup",
                "skill_name": name,
                "skill_path": str(md_path),
                "days_since_deprecated": days_old,
                "replaces": replaces,
                "body": (
                    f"【清理建议】技能 '{name}' 已废弃 {days_old} 天"
                    + (f"（被 '{replaces}' 替代）" if replaces else "")
                    + "，建议归档或删除以保持技能目录整洁。"
                ),
                "tags": f"pattern_miner, deprecated, cleanup, {name}",
            })

        return findings

    # ------------------------------------------------------------------
    # Analyzer 3: High-value memory identification
    # ------------------------------------------------------------------

    def find_high_value_memories(self, min_helps: int = 3,
                                 max_harm_ratio: float = 0.2) -> list[dict]:
        """Identify memories that consistently help the agent recover.

        These memories could be promoted (shown first, given higher priority
        in semantic rerank, or pinned to system prompt for certain task types).
        """
        from src.memory.memory_db import memory_db

        rows = memory_db.conn.execute(
            """SELECT id, name, tags, body, help_count, harm_count, hits
               FROM memories_data
               WHERE game=? AND help_count >= ?
               ORDER BY help_count DESC LIMIT 20""",
            (self.game, min_helps),
        ).fetchall()

        findings: list[dict] = []
        for r in rows:
            help_c = r["help_count"] or 0
            harm_c = r["harm_count"] or 0
            total = help_c + harm_c
            if total == 0:
                continue
            harm_ratio = harm_c / total if total > 0 else 0
            if harm_ratio <= max_harm_ratio:
                # This is a high-value memory
                body_preview = (r["body"] or "")[:150]
                findings.append({
                    "type": "high_value_memory",
                    "memory_id": r["id"],
                    "memory_name": r["name"],
                    "help_count": help_c,
                    "harm_count": harm_c,
                    "body": (
                        f"【系统发现】记忆 '{r['name']}' 是一条高价值记忆："
                        f"帮助 agent 恢复了 {help_c} 次，仅 {harm_c} 次未生效。"
                        f"内容摘要：{body_preview}..."
                    ),
                    "tags": f"pattern_miner, high_value, 永久",
                })

        return findings

    # ------------------------------------------------------------------
    # Analyzer 4: Unlearned user guidance
    # ------------------------------------------------------------------

    def find_unlearned_guidance(self, lookback_logs: int = 200) -> list[dict]:
        """Scan recent user corrections for guidance not yet captured as memories.

        Reads the lightweight user_corrections.jsonl index (written by
        ExecutionLogger) instead of scanning full multi-MB JSON conversation
        logs.  Falls back to the old JSON-scanning approach if the index
        doesn't exist yet.
        """
        from src.memory.memory_db import memory_db

        log_dir = Path(config.DATA_DIR) / "logs"
        if not log_dir.exists():
            return []

        guidance_pairs: list[dict] = []

        # -- Fast path: read lightweight user corrections index --
        index_path = log_dir / "user_corrections.jsonl"
        if index_path.exists():
            try:
                lines = index_path.read_text(encoding="utf-8").strip().split("\n")
                for line in reversed(lines[-lookback_logs:]):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("game") == self.game:
                            guidance_pairs.append({
                                "task": entry.get("task", ""),
                                "reply": entry.get("correction", "")[:200],
                            })
                    except json.JSONDecodeError:
                        continue
            except Exception:
                pass

        # -- Fallback: scan JSON logs (backward compat) --
        if not guidance_pairs:
            log_files = sorted(log_dir.glob("conv_*.json"), reverse=True)[:20]
            if not log_files:
                return []
            for lf in log_files:
                try:
                    data = json.loads(lf.read_text(encoding="utf-8"))
                    messages = data.get("messages", [])
                except (json.JSONDecodeError, OSError):
                    continue
                for i, msg in enumerate(messages):
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                if block.get("name") == "ask_user":
                                    for j in range(i + 1, min(i + 15, len(messages))):
                                        next_msg = messages[j]
                                        next_content = next_msg.get("content", "")
                                        if isinstance(next_content, list):
                                            if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in next_content):
                                                continue
                                        if next_msg.get("role") == "user" and isinstance(next_content, str):
                                            reply = next_content.strip()
                                            if reply and len(reply) > 5:
                                                clean = reply
                                                for marker in ["[用户回复] ", "[用户指令 — 必须执行] ", "[用户指令] "]:
                                                    if marker in clean:
                                                        clean = clean.split(marker, 1)[-1].strip()
                                                        break
                                                if clean:
                                                    guidance_pairs.append({
                                                        "task": data.get("task", ""),
                                                        "reply": clean[:200],
                                                    })
                                                    break
                                    break

        if not guidance_pairs:
            return []

        # Check which guidance already has a corresponding memory
        unlearned: list[dict] = []
        for gp in guidance_pairs[:5]:
            reply = gp["reply"]
            existing = memory_db.conn.execute(
                """SELECT id FROM memories_data
                   WHERE game=? AND body LIKE ? LIMIT 1""",
                (self.game, f"%{reply[:30]}%"),
            ).fetchone()
            if not existing:
                unlearned.append({
                    "type": "unlearned_guidance",
                    "task": gp["task"],
                    "body": (
                        f"【待学习】用户曾在任务「{gp['task']}」中给出以下指导，"
                        f"但尚未被记录为记忆：\n\"{reply}\"\n"
                        f"下次执行类似任务时，agent 应记住此指导。"
                    ),
                    "tags": "pattern_miner, unlearned_guidance",
                })

        return unlearned

    def _save_finding(self, finding: dict) -> None:
        """Save a finding as a memory file and index it."""
        from src.memory.memory_db import memory_db
        from src.tools.remember import _index_memory

        name = f"pm{int(_time.time() * 1_000_000)}"
        body = finding.get("body", "")
        tags = finding.get("tags", "")
        finding_type = finding.get("type", "unknown")

        # Ensure directory
        mem_dir = Path(config.DATA_DIR) / "memories" / self.game
        mem_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(tz=timezone.utc).isoformat()
        yaml_lines = [
            f"game: {self.game}",
            f"tags: [{', '.join(t.strip() for t in tags.split(',') if t.strip())}]",
            f"source: pattern_miner",
            f"created: {now}",
            f"finding_type: {finding_type}",
        ]
        content = "---\n" + "\n".join(yaml_lines) + "\n---\n\n" + body

        file_path = mem_dir / f"{name}.md"
        file_path.write_text(content, encoding="utf-8")

        _index_memory(name, self.game, tags, body, screen_hash=None, source="pattern_miner")
        logger.info("Pattern finding saved: %s (%s)", name, finding_type)


# ------------------------------------------------------------------
# Debounced invocation
# ------------------------------------------------------------------

def should_run_pattern_miner(game: str = "arknights") -> bool:
    """Check if enough tasks have accumulated since the last miner run."""
    from src.memory.memory_db import memory_db

    interval = config.agent.learning_pattern_miner_interval
    key = f"pattern_miner_tasks_since_{game}"
    count_str = memory_db.get_learning_state(key)
    count = int(count_str) if count_str else 0
    return count >= interval


def reset_pattern_miner_counter(game: str = "arknights") -> None:
    """Reset the counter after a miner run."""
    from src.memory.memory_db import memory_db
    memory_db.set_learning_state(f"pattern_miner_tasks_since_{game}", "0")


def increment_pattern_miner_counter(game: str = "arknights") -> None:
    """Increment the task counter for pattern miner scheduling."""
    from src.memory.memory_db import memory_db
    key = f"pattern_miner_tasks_since_{game}"
    count_str = memory_db.get_learning_state(key)
    count = (int(count_str) if count_str else 0) + 1
    memory_db.set_learning_state(key, str(count))


def run_pattern_miner_if_due(game: str = "arknights") -> int:
    """Run the pattern miner if enough tasks have accumulated. Returns count of findings."""
    if not should_run_pattern_miner(game):
        return 0

    logger.info("Pattern miner triggered for game=%s", game)
    miner = PatternMiner(game)
    count = miner.run_and_save()
    reset_pattern_miner_counter(game)
    return count
