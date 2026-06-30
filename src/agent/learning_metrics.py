"""Learning Metrics — quantify agent improvement over time (Phase 3).

Computes metrics across time windows (7d, 30d, all-time) by querying
task_executions and memories_data.  Outputs a human-readable dashboard.

Usage:
    python -m src.agent.learning_metrics [--game arknights] [--days 30]
"""

from __future__ import annotations

import json
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_metrics(game: str = "arknights", window_days: int = 30) -> dict[str, Any]:
    """Compute all learning metrics for the given game and time window.

    Returns a dict with keys: task_count, success_rate, avg_iterations,
    fast_chain_usage, memory_help_harm_ratio, ask_user_freq, first_try_rate,
    total_memories, total_skills, total_action_patterns.
    """
    from src.memory.memory_db import memory_db

    now = _time.time()
    cutoff = now - window_days * 86400 if window_days > 0 else 0

    conn = memory_db.conn

    # ---- Task-level metrics ----
    if cutoff > 0:
        tasks = conn.execute(
            """SELECT success, iterations, duration_seconds, failure_signal_types,
               skill_fast_chain_success, user_interrupted
               FROM task_executions
               WHERE game=? AND finished_at >= ?""",
            (game, cutoff),
        ).fetchall()
    else:
        tasks = conn.execute(
            """SELECT success, iterations, duration_seconds, failure_signal_types,
               skill_fast_chain_success, user_interrupted
               FROM task_executions WHERE game=?""",
            (game,),
        ).fetchall()

    task_count = len(tasks)
    if task_count == 0:
        return {"task_count": 0, "message": "No task data yet. Run some tasks first!"}

    # Success rate
    success_count = sum(1 for t in tasks if t["success"])
    success_rate = round(success_count / task_count, 3)

    # Average iterations
    avg_iterations = round(
        sum(t["iterations"] or 0 for t in tasks) / task_count, 1
    )

    # Fast chain usage rate (% of tasks that used fast_chain)
    fc_tasks = [t for t in tasks if t["skill_fast_chain_success"] is not None]
    fc_usage_rate = round(len(fc_tasks) / task_count, 3) if task_count > 0 else 0
    fc_success_rate = (
        round(sum(1 for t in fc_tasks if t["skill_fast_chain_success"] == 1) / len(fc_tasks), 3)
        if fc_tasks else 0
    )

    # ask_user frequency (fraction of tasks where user was interrupted)
    ask_count = sum(1 for t in tasks if t["user_interrupted"])
    ask_freq = round(ask_count / task_count, 3)

    # First-try rate: tasks with no failure signals
    first_try_count = sum(
        1 for t in tasks
        if not (t["failure_signal_types"] or "").strip()
    )
    first_try_rate = round(first_try_count / task_count, 3)

    # ---- Memory-level metrics (P2: consolidated from 3 queries to 1 GROUP BY) ----
    mem_stats = conn.execute(
        """SELECT source, COUNT(*) as cnt FROM memories_data WHERE game=? GROUP BY source""",
        (game,),
    ).fetchall()
    mem_count = sum(r["cnt"] for r in mem_stats)
    ap_count = next((r["cnt"] for r in mem_stats if r["source"] == "action_pattern"), 0)
    pm_count = next((r["cnt"] for r in mem_stats if r["source"] == "pattern_miner"), 0)

    # Help/harm ratio
    help_harm_row = conn.execute(
        """SELECT SUM(help_count) as total_help, SUM(harm_count) as total_harm
           FROM memories_data WHERE game=?""",
        (game,),
    ).fetchone()
    total_help = help_harm_row["total_help"] or 0
    total_harm = help_harm_row["total_harm"] or 0
    help_harm_ratio = round(total_help / max(total_harm, 1), 2)

    # ---- Skill-level metrics (per-game) ----
    skill_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM skills_data WHERE game=?",
        (game,),
    ).fetchone()["cnt"]

    verified_skill_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM skills_data WHERE game=? AND verified=1",
        (game,),
    ).fetchone()["cnt"]

    # ---- Trend data (per-day for the window) ----
    # P2: consolidated from window_days individual queries to a single SELECT
    # with Python-side aggregation.  90-day window used to do 90 round-trips.
    trend: list[dict] = []
    if window_days > 0 and window_days <= 90:
        all_tasks = conn.execute(
            """SELECT success, iterations, finished_at FROM task_executions
               WHERE game=? AND finished_at >= ?
               ORDER BY finished_at""",
            (game, cutoff),
        ).fetchall()
        # Group by day in Python (single pass)
        from collections import defaultdict
        day_buckets: dict[str, list] = defaultdict(list)
        for t in all_tasks:
            day_key = datetime.fromtimestamp(t["finished_at"], tz=timezone.utc).strftime("%m-%d")
            day_buckets[day_key].append(t)
        for day_key in sorted(day_buckets.keys()):
            day_tasks = day_buckets[day_key]
            d_success = sum(1 for t in day_tasks if t["success"])
            d_avg_iter = sum(t["iterations"] or 0 for t in day_tasks) / len(day_tasks)
            trend.append({
                "date": day_key,
                "tasks": len(day_tasks),
                "success_rate": round(d_success / len(day_tasks), 2),
                "avg_iterations": round(d_avg_iter, 1),
            })

    return {
        "game": game,
        "window_days": window_days,
        "task_count": task_count,
        "success_rate": success_rate,
        "avg_iterations": avg_iterations,
        "fast_chain_usage_rate": fc_usage_rate,
        "fast_chain_success_rate": fc_success_rate,
        "ask_user_frequency": ask_freq,
        "first_try_rate": first_try_rate,
        "help_harm_ratio": help_harm_ratio,
        "total_memories": mem_count,
        "total_action_patterns": ap_count,
        "total_pattern_findings": pm_count,
        "total_skills": skill_count,
        "total_verified_skills": verified_skill_count,
        "trend": trend,
    }


def print_dashboard(metrics: dict[str, Any]) -> None:
    """Print a human-readable learning dashboard to stdout."""
    if metrics.get("task_count", 0) == 0:
        print("📊 Terra Learning Dashboard")
        print("   No task data yet. Run some tasks first!")
        return

    game = metrics.get("game", "?")
    window = metrics.get("window_days", 30)
    window_label = f"近{window}天" if window > 0 else "全部历史"

    print(f"📊 Terra 学习仪表盘 — {game} ({window_label})")
    print(f"   总任务数: {metrics['task_count']}")
    print(f"   任务成功率: {metrics['success_rate']:.1%}")
    print(f"   平均迭代数: {metrics['avg_iterations']:.1f} 轮/任务")
    print(f"   快速链使用率: {metrics['fast_chain_usage_rate']:.1%}")
    print(f"   快速链成功率: {metrics['fast_chain_success_rate']:.1%}")
    print(f"   首次成功率: {metrics['first_try_rate']:.1%}")
    print(f"   用户干预频率: {metrics['ask_user_frequency']:.1%}")
    print(f"   记忆帮助/有害比: {metrics['help_harm_ratio']:.1f}")
    print(f"   总记忆数: {metrics['total_memories']} "
          f"(正向模式: {metrics['total_action_patterns']}, "
          f"模式发现: {metrics['total_pattern_findings']})")
    print(f"   总技能数: {metrics['total_skills']} "
          f"(已验证: {metrics['total_verified_skills']})")

    trend = metrics.get("trend", [])
    if trend:
        print(f"\n   每日趋势 ({len(trend)} 天):")
        print(f"   {'日期':<8} {'任务':<6} {'成功率':<10} {'平均迭代':<10}")
        for t in trend:
            print(f"   {t['date']:<8} {t['tasks']:<6} {t['success_rate']:<10.0%} {t['avg_iterations']:<10.1f}")

    # Learning health indicators
    print(f"\n   📈 学习健康度:")
    health_checks: list[tuple[str, bool]] = []

    # 1. Success rate should be trending up (last 7 days vs overall)
    recent_trend = trend[-7:] if len(trend) >= 7 else trend
    if len(recent_trend) >= 3:
        first_week_rate = sum(t["success_rate"] for t in recent_trend[:3]) / 3
        last_week_rate = sum(t["success_rate"] for t in recent_trend[-3:]) / 3
        improving = last_week_rate >= first_week_rate * 0.95  # Allow slight noise
        health_checks.append(("成功率趋势上升", improving))

    # 2. Fast chain usage > 10%
    health_checks.append(("快速链使用率 > 10%", metrics["fast_chain_usage_rate"] > 0.1))

    # 3. Help/harm ratio > 1.0
    health_checks.append(("记忆帮助比 > 1.0", metrics["help_harm_ratio"] > 1.0))

    # 4. At least some action patterns
    health_checks.append(("有正向行动模式记忆", metrics["total_action_patterns"] > 0))

    # 5. First-try rate > 30% (agent is learning)
    health_checks.append(("首次成功率 > 30%", metrics["first_try_rate"] > 0.3))

    for label, ok in health_checks:
        icon = "✅" if ok else "❌"
        print(f"     {icon} {label}")


def main() -> None:
    """CLI entry point: python -m src.agent.learning_metrics"""
    args = sys.argv[1:]

    game = "arknights"
    days = 30

    i = 0
    while i < len(args):
        if args[i] == "--game" and i + 1 < len(args):
            game = args[i + 1]
            i += 2
        elif args[i] == "--days" and i + 1 < len(args):
            try:
                days = int(args[i + 1])
            except ValueError:
                print(f"Invalid days value: {args[i + 1]}")
                return
            i += 2
        elif args[i] == "--all":
            days = 0
            i += 1
        else:
            i += 1

    metrics = compute_metrics(game=game, window_days=days)
    print_dashboard(metrics)


if __name__ == "__main__":
    main()
