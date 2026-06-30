"""管家 LLM 系统提示词构建器。

单轮 LLM 调用：理解用户意图、分配到正确的游戏和设备、必要时确认。
不是 agent loop —— 最多一次 LLM 调用，无 screen 注入。
"""

from __future__ import annotations

from typing import Any


def build_concierge_system_prompt(
    games_info: str = "",
    slots_info: str = "",
    context: str = "",
) -> str:
    """Build the concierge LLM system prompt with game/device/rules context."""
    parts = [_PERSONA, _RULES]
    if games_info:
        parts.append("## 可用游戏\n\n" + games_info)
    if slots_info:
        parts.append("## 可用设备\n\n" + slots_info)
    if context:
        parts.append("## 当前上下文\n\n" + context)
    return "\n\n".join(parts)


# ── Persona ─────────────────────────────────────────────────────────

_PERSONA = """你是「博士的管家」，负责理解用户的游戏任务指令，将任务正确分配给对应的游戏和模拟器。

你不是游戏操作 Agent——你不知道游戏界面长什么样、不知道技能步骤。你只做三件事：
1. 理解用户想做什么（哪个游戏、什么任务）
2. 分配任务到正确的设备
3. 必要时向用户确认"""

# ── Rules ────────────────────────────────────────────────────────────

_RULES = """## 决策规则

### 单游戏单任务
用户指令明确指向一个游戏且任务清晰 → 直接调用 delegate_to_game()，分配后告知用户。

### 多游戏（重要 — 必须全部派发）
用户指令涉及多个游戏 → **必须为每个游戏分别调用一次 delegate_to_game()**。这是硬性要求，不能遗漏任何一个游戏。

- 有多少个游戏就调用多少次 delegate_to_game。不能合并、不能跳过、不能只派发一部分。
- 设备够用 → 直接分配。**不要先输出大段文字解释再调工具——工具调用本身就是回答。** 第一个工具调用之前最多用一句话总结。
- 设备不够 → **直接调用 launch_emulator() 启动新模拟器，不要问用户。** 多游戏请求意味着用户需要多开——这不需要确认。只有启动失败时才 ask_user() 告知。
- 上下文已经给出了明确的游戏拆分 → **不要质疑、不要修正、不要重新分析**，直接按拆分结果每个游戏各 delegate 一次。

### 模糊意图
用户指令含糊（"帮我搞一下"、"做日常"但未指定游戏）→ ask_user() 澄清。
不确定游戏是什么 → ask_user()。
不确定任务是什么 → ask_user()。

### 纯聊天
问候、感谢、无任务意图 → chat_with_user() 直接回复。

### 设备分配
- 设备上显示的「游戏」只是上一次委托记录，不是实时检测。Agent 无法自行关闭/切换应用。
- 优先使用游戏匹配的设备；如果设备是「未委派」→ 可以委托任意游戏
- 设备游戏不匹配或不明确 → ask_user() 确认
- 所有设备都忙 → 告知用户等待，或 ask_user()
- **多游戏请求：设备不够直接 launch_emulator()，不需要确认。单游戏请求：启动新模拟器前 ask_user() 确认。**

### 禁止行为
- 不要编造游戏名称——只使用「可用游戏」列表中的 game_id
- 不要把不同游戏的任务合并成一个 delegate 调用
- **不要遗漏多游戏请求中的任何一个游戏**
- **不要在工具调用前输出大段文字——直接调工具。工具调用结果就是你的回答。**"""


# ── Helper: build dynamic context from router state ──────────────────

def build_games_info(games: list[str] | None = None) -> str:
    """Build a human-readable list of available games and their keywords."""
    from src.games.registry import get_game_registry
    gr = get_game_registry()

    lines: list[str] = []
    target = set(games) if games else {p.manifest.id for p in gr.list_all()}

    for gid in target:
        plugin = gr.get(gid)
        if plugin is None:
            continue
        m = plugin.manifest
        kw_list = ", ".join(m.keywords[:4])
        lines.append("- **" + m.name + "** (game_id=`" + gid + "`): " + kw_list)

    return "\n".join(lines) if lines else "（无注册游戏）"


def build_slots_info(slots: list[Any]) -> str:
    """Build a human-readable list of available device slots."""
    if not slots:
        return "（无可用设备）"

    lines: list[str] = []
    for s in slots:
        label = getattr(s, "label", s.slot_id if hasattr(s, "slot_id") else "?")
        game = getattr(s, "game", "") or ""
        serial = getattr(s, "device_serial", "") or "?"
        has_task = getattr(s, "current_task", None) is not None
        if has_task:
            desc = ""
            if s.current_task:
                desc = getattr(s.current_task, "task_description", "")[:30]
            status = "忙碌中: " + desc
        else:
            status = "空闲"
        game_display = game if game else "未委派"
        lines.append("- " + label + " (`" + serial + "`): " + game_display + " (" + status + ")")

    free = [s for s in slots if s.is_free]
    free_count = str(len(free))
    total_count = str(len(slots))

    if free:
        free_names = [getattr(s, "label", "?") for s in free]
        summary = "共 " + total_count + " 个设备，" + free_count + " 个空闲 (" + ", ".join(free_names) + ")。"
    else:
        summary = "共 " + total_count + " 个设备，全部繁忙。"

    note = (
        "\n\n"
        "关于游戏显示: 「未委派」= 还没有任务被分配到此设备；"
        "其他游戏名 = 上次委派到此设备的游戏。"
        "这不是实时检测——它不知道屏幕当前实际显示什么。"
        "Agent 无法自行关闭/切换应用。不确定时 ask_user() 确认。"
    )

    return summary + note + "\n\n" + "\n".join(lines)


def build_multi_game_context(
    text: str,
    games: list[str],
    sub_tasks: dict[str, str],
    slots: list[Any] | None = None,
    emu_available: bool = False,
) -> str:
    """Build context block for multi-game detection results.

    Includes device availability so the LLM can decide whether to
    auto-dispatch or ask for confirmation (e.g. when devices are
    insufficient).
    """
    from src.games.registry import get_game_registry
    gr = get_game_registry()

    parts = ["用户原话：「" + text + "」", "", "**系统已检测到多游戏请求，拆分如下：**"]
    for g in games:
        plugin = gr.get(g)
        name = plugin.manifest.name if plugin else g
        task = sub_tasks.get(g, "日常任务")
        parts.append("  - " + name + " (`" + g + "`): " + task)

    # Device availability
    if slots:
        free = [s for s in slots if s.is_free]
        busy = [s for s in slots if not s.is_free]
        parts.append("")
        parts.append("**当前设备状态：**")
        parts.append(f"  - 可用设备: {len(free)} 个" + (
            f" ({', '.join(getattr(s, 'label', '?') for s in free)})" if free else ""
        ))
        if busy:
            busy_info = ", ".join(
                f"{getattr(s, 'label', '?')}({getattr(s, 'game', '')})" for s in busy
            )
            parts.append(f"  - 忙碌设备: {len(busy)} 个 ({busy_info})")
        if emu_available:
            parts.append("  - 可以启动新模拟器实例（launch_emulator），**直接调用，不需要问用户**")

    parts.append("")
    parts.append(
        "**你必须为以上 " + str(len(games)) + " 个游戏分别调用 delegate_to_game()。**"
        " 每个游戏调用一次，一共 " + str(len(games)) + " 次。"
        " 不要质疑拆分结果，直接按此执行。工具调用之前最多说一句话总结。"
    )
    if slots and len([s for s in slots if s.is_free]) < len(games):
        parts.append(
            "**空闲设备不足**（需要 " + str(len(games)) + " 个但只有 "
            + str(len([s for s in slots if s.is_free])) + " 个空闲）。"
            "**先调用 launch_emulator()，等模拟器启动成功后，再为每个游戏 delegate。**"
            "注意：必须先多开后派发——如果先派发任务，模拟器会在运行中被中断，不能启动。"
        )
    return "\n".join(parts)
