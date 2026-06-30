"""Runtime context injector — injects day-of-week, game state, etc.

When an orchestrator skill (like daily) is matched, the LLM needs to know the
current runtime state to make conditional decisions (e.g. "is it Saturday? →
should I run annihilation?").  This module gathers that context and formats it
as a [系统上下文] message injected into the conversation.

Game-specific data (sanity, materials, etc.) is gathered via plugin hooks so
the core agent loop stays game-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RuntimeContext:
    """Current runtime state gathered at the start of a task."""

    day_of_week: int = 0          # 0=Monday, 6=Sunday
    day_name_cn: str = ""         # "周一"..."周日"
    time_of_day: str = ""         # "凌晨"/"早上"/"上午"/"中午"/"下午"/"晚上"
    hour: int = 0
    game: str = "arknights"
    device_serial: str = ""

    # Game-specific extra data from plugin hooks
    extra: dict[str, Any] = field(default_factory=dict)


_DAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _time_of_day(hour: int) -> str:
    if hour < 6:
        return "凌晨"
    if hour < 9:
        return "早上"
    if hour < 12:
        return "上午"
    if hour < 14:
        return "中午"
    if hour < 18:
        return "下午"
    return "晚上"


def gather_runtime_context(
    game: str = "arknights",
    device_serial: str = "",
) -> RuntimeContext:
    """Gather runtime context: time, day, and game-specific state.

    Game-specific data is gathered via plugin hooks so the core stays
    game-agnostic.  Currently Arknights provides sanity/annihilation state;
    Reverse1999 provides material stock hints.
    """
    now = datetime.now()
    dow = now.weekday()  # 0=Monday
    ctx = RuntimeContext(
        day_of_week=dow,
        day_name_cn=_DAY_NAMES[dow],
        time_of_day=_time_of_day(now.hour),
        hour=now.hour,
        game=game,
        device_serial=device_serial,
    )

    # ── Game-specific context via plugin hooks ──
    try:
        from src.games.registry import get_game_registry
        plugin = get_game_registry().get(game)
        if plugin and hasattr(plugin, "gather_runtime_context"):
            extra = plugin.gather_runtime_context(device_serial)
            if extra:
                ctx.extra = extra
                logger.debug("Runtime context extra: %s", ctx.extra)
    except Exception as e:
        from src.utils.errors import safe_log
        safe_log(
            logger, "warning",
            f"Game-specific context gathering failed for {game}: {e}",
        )

    return ctx


def inject_context_message(ctx: RuntimeContext) -> str:
    """Format a RuntimeContext as a [系统上下文] message for conversation injection.

    The message is concise enough to not dominate the prompt, but specific
    enough for the LLM to make conditional decisions (Saturday→annihilation,
    low materials→farm specific stage, etc.).
    """
    lines = [
        "[系统上下文 — 任务开始时的状态]",
        f"今天是{ctx.day_name_cn}，{ctx.time_of_day}。",
    ]

    # ── Arknights-specific context ──
    extra = ctx.extra
    if ctx.game == "arknights":
        sanity = extra.get("sanity_current")
        sanity_max = extra.get("sanity_max")
        if sanity is not None:
            lines.append(f"理智: {sanity}/{sanity_max if sanity_max else '?'}")
        annihilation_done = extra.get("annihilation_done")
        if annihilation_done is not None:
            if annihilation_done:
                lines.append("剿灭: 本周已完成")
            else:
                lines.append("剿灭: 本周尚未完成")

        # Recommendations
        suggestions: list[str] = []
        is_saturday = ctx.day_of_week == 5  # Saturday (Friday in 0-based, 5=Saturday)
        if is_saturday and annihilation_done is False:
            suggestions.append("今天是周六，建议先打剿灭再刷关，避免理智溢出。")
        if sanity is not None and sanity > 30:
            suggestions.append(f"理智充足 ({sanity})，剿灭完后可以刷材料关。")
        elif sanity is not None and sanity < 10:
            suggestions.append("理智不足，跳过刷关步骤。")
        if suggestions:
            lines.append("建议: " + " ".join(suggestions))

    # ── Reverse1999-specific context ──
    elif ctx.game == "reverse1999":
        material_hints = extra.get("material_hints", [])
        if material_hints:
            lines.append("仓库材料库存（低库存优先刷）:")
            for hint in material_hints[:5]:
                lines.append(f"  - {hint}")
            lines.append("建议: 优先刷库存最少的材料关。")
        else:
            lines.append("材料库存: 调用 scan_depot() 一键扫描")

    # ── Fallback (unknown game) ──
    else:
        if extra:
            items = [f"{k}: {v}" for k, v in extra.items()]
            lines.append("游戏状态: " + "; ".join(items))

    return "\n".join(lines)
