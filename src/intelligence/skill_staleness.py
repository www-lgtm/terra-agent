"""SkillStalenessCheck — pre-task skill freshness warning (Phase 2).

When a task matches a verified skill, checks recent task_executions to see
if that skill's fast_chain has been failing.  If so, downgrades the
recommendation to manual mode and warns the LLM.
"""

from __future__ import annotations

import logging

from src.intelligence.base import (
    IntelligenceContext,
    IntelligenceResult,
    IntelligenceTool,
)

logger = logging.getLogger(__name__)

# ── Auto-downgrade: number of consecutive fast_chain failures to trigger ──
AUTO_DOWNGRADE_FAILURE_COUNT = 2


def _downgrade_skill_to_guide(skill_name: str, game: str) -> bool:
    """Downgrade a script skill to guide type after repeated fast_chain failures.

    Only modifies frontmatter — type and verified status.  The body with
    coordinate annotations is preserved as-is so the LLM can reference them
    during manual execution.
    """
    from src.skills.manager import get_skill_manager

    skill_mgr = get_skill_manager(game)
    skill = skill_mgr.load(skill_name)
    if not skill:
        return False

    # Only downgrade if currently verified (has fast-chain level)
    if not skill.get("verified"):
        return False

    raw = skill.get("raw", "")
    if not raw:
        return False

    new_raw = raw
    new_raw = new_raw.replace("type: script", "type: guide")
    new_raw = new_raw.replace("verified: true", "verified: false")

    if new_raw == raw:
        # Nothing changed — may already be downgraded or using different format
        return False

    skill_mgr.save(skill_name, new_raw)
    logger.warning(
        "Skill '%s' auto-downgraded: script → guide (stale coordinates). "
        "Will use manual execution next time.",
        skill_name,
    )
    return True


class SkillStalenessCheck(IntelligenceTool):
    """Warns when a matched skill's fast-chain coordinates may be stale."""

    # How many recent executions to look at
    LOOKBACK = 10
    # Minimum executions before we can judge staleness (lowered from 3 → 2
    # so auto-downgrade triggers faster on repeated skill failures)
    MIN_EXECUTIONS = 2
    # Fast-chain success rate below which we warn
    STALE_THRESHOLD = 0.5
    # Time-based staleness: warn after N days without coordinate verification
    # (lowered from 14 → 10 days — game updates happen frequently)
    STALE_DAYS_WARN = 10
    STALE_DAYS_SOFT = 7

    def can_handle(self, task: str) -> bool:
        """Only relevant when skills are matched."""
        return True  # We check ctx.skills inside analyze()

    def analyze(self, ctx: IntelligenceContext, task: str) -> IntelligenceResult | None:
        """Check each matched skill's recent fast_chain success rate."""
        if not ctx.skills:
            return None

        verified_skills = [s for s in ctx.skills if s.get("verified")]
        if not verified_skills:
            return None

        from src.memory.memory_db import memory_db

        warnings: list[str] = []
        for skill in verified_skills:
            name = skill.get("name", "")
            if not name:
                continue

            # Query recent executions of this skill.
            # skill_name may be comma-separated (multi-skill tasks).
            rows = memory_db.conn.execute(
                """SELECT skill_fast_chain_success
                   FROM task_executions
                   WHERE game=? AND (skill_name=? OR skill_name LIKE ? OR skill_name LIKE ? OR skill_name LIKE ?)
                   ORDER BY finished_at DESC LIMIT ?""",
                (ctx.game, name, f"{name},%", f"%,{name},%", f"%,{name}", self.LOOKBACK),
            ).fetchall()

            if len(rows) < self.MIN_EXECUTIONS:
                continue  # Not enough data yet

            # Calculate success rate among non-NULL entries
            fc_results = [
                r["skill_fast_chain_success"] for r in rows
                if r["skill_fast_chain_success"] is not None
            ]
            if len(fc_results) < self.MIN_EXECUTIONS:
                continue

            success_rate = sum(1 for r in fc_results if r == 1) / len(fc_results)

            if success_rate < self.STALE_THRESHOLD:
                # ── Auto-downgrade: 2+ consecutive failures → guide ──
                if success_rate == 0.0 and len(fc_results) >= AUTO_DOWNGRADE_FAILURE_COUNT:
                    downgraded = _downgrade_skill_to_guide(name, ctx.game)
                    if downgraded:
                        warnings.append(
                            f"技能 '{name}' 已自动降级为 guide："
                            f"最近 {len(fc_results)} 次快速链全部失败。"
                            f"下次将用手动执行，成功后自动更新坐标。"
                        )
                    else:
                        warnings.append(
                            f"技能 '{name}' 最近 {len(fc_results)} 次 fast_chain 全部失败，"
                            f"坐标可能已过期。先调用 skill_run('{name}') 尝试，"
                            f"失败后再手动执行（手动成功后坐标会自动更新）。"
                        )
                else:
                    warnings.append(
                        f"技能 '{name}' 最近 {len(fc_results)} 次 fast_chain 执行成功率仅 "
                        f"{round(success_rate * 100)}%，坐标可能已过期。"
                        f"先调用 skill_run('{name}') 尝试——如果成功就赚了，"
                        f"失败了再手动执行。"
                    )
            elif success_rate < 0.8:
                warnings.append(
                    f"技能 '{name}' fast_chain 成功率 {round(success_rate * 100)}%，"
                    f"仍有风险。先试 skill_run('{name}')，失败后立即切换到手动。"
                )

            # ── Time-based staleness check ──
            # Even if fast-chain hasn't been used recently, coordinates that
            # haven't been verified in 10+ days are likely stale (game updates,
            # resolution changes, etc.).
            coords_verified_raw = skill.get("coords_verified_at", "")
            if coords_verified_raw:
                try:
                    from datetime import datetime, timezone, timedelta
                    if isinstance(coords_verified_raw, str):
                        verified_dt = datetime.fromisoformat(coords_verified_raw)
                    else:
                        verified_dt = datetime.fromtimestamp(float(coords_verified_raw), tz=timezone.utc)
                    days_since = (datetime.now(tz=timezone.utc) - verified_dt).days
                    if days_since > self.STALE_DAYS_WARN:
                        warnings.append(
                            f"技能 '{name}' 坐标已 {days_since} 天未验证，"
                            f"可能因游戏更新而偏移。建议闲置时验证。"
                        )
                    elif days_since > self.STALE_DAYS_SOFT:
                        warnings.append(
                            f"技能 '{name}' 坐标已 {days_since} 天未验证（接近过期）。"
                        )
                except (ValueError, TypeError, OSError):
                    pass

        if not warnings:
            return None

        recommendation = "⚠️ 技能坐标新鲜度警告：\n" + "\n".join(
            f"  • {w}" for w in warnings
        )

        return IntelligenceResult(
            recommendation=recommendation,
            confidence=0.85,
            source="skills+history",
        )
