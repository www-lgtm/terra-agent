"""管家管理工具 — 委托、查询、取消、对话。

这些工具不注册到全局 ToolRegistry，管家有自己的 dispatch 字典。
工具 schema 根据可用游戏槽位动态生成，多游戏时 game 参数可选（有默认值）。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Tool schema builder (dynamic, depends on available slots) ──

def build_concierge_tool_schemas(slots: list[Any]) -> list[dict[str, Any]]:
    """根据可用设备槽构建工具 schema。

    设备池模式下 game 是运行时状态，所有已注册游戏都作为选项列出。
    """
    from src.games.registry import get_game_registry
    gr = get_game_registry()
    # List ALL registered games (not just what's currently on devices)
    all_games = gr.get_ids()
    game_names: dict[str, str] = {}
    for g in all_games:
        plugin = gr.get(g)
        game_names[g] = plugin.manifest.name if plugin else g

    delegate_props: dict[str, Any] = {
        "task": {
            "type": "string",
            "description": "要执行的任务描述，用中文，尽量具体（如「清体力 GT-6」「基建收菜」）",
        },
    }
    delegate_required = ["task"]

    if len(all_games) > 1:
        # Accept both game IDs AND Chinese display names in the enum.
        # LLMs frequently output the Chinese name (e.g. "明日方舟") instead
        # of the raw ID ("arknights"), causing schema validation to reject
        # the tool call.  Listing both avoids this silent failure.
        enum_values = list(all_games) + [game_names.get(g, g) for g in all_games]
        delegate_props["game"] = {
            "type": "string",
            "enum": enum_values,
            "description": "目标游戏: " + ", ".join(
                f"{g}={game_names.get(g, g)}" for g in all_games
            ),
        }
        delegate_required.insert(0, "game")

    # Build schemas list
    schemas = [
        {
            "name": "delegate_to_agent",
            "description": (
                "将游戏任务派发给智能体执行。智能体会自动操作手机完成指定任务。"
                "如果已有任务在运行，任务会排队等候。"
            ),
            "parameters": {
                "type": "object",
                "properties": delegate_props,
                "required": delegate_required,
            },
        },
        # Phase 3: delegate_batch — only when multiple slots of the same game exist
    ]

    # Check if batch routing is applicable (2+ slots of the same game)
    game_counts: dict[str, int] = {}
    for s in slots:
        game_counts[s.game] = game_counts.get(s.game, 0) + 1
    if any(c >= 2 for c in game_counts.values()):
        schemas.append({
            "name": "delegate_batch",
            "description": (
                "向多个账号批量派发同一任务。用于用户说「两个号都清体力」"
                "「全部刷GT-6」「方舟都基建收菜」等场景。"
                "空闲账号立即启动，忙碌账号自动排队。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "要执行的任务描述，去掉批量关键词后的纯任务。例如用户说「两个号都清体力」，task='清体力'",
                    },
                },
                "required": ["task"],
            },
        })

    schemas.extend([
        {
            "name": "check_agent_status",
            "description": "查看当前正在运行的游戏智能体状态快照（正在做什么、已执行多少步等）。",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "cancel_task",
            "description": "停止当前正在运行的游戏智能体任务。",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "start_emulator",
            "description": (
                "启动模拟器或已克隆的多开实例。如果主实例不在线，启动主实例。"
                "如果主实例在线，启动已克隆的多开实例（需要用户已用过一次多开器克隆）。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "restart_emulator",
            "description": (
                "重启指定的模拟器实例。target 可用模拟器名称、别名或 ADB 串口。"
                "如「重启MuMu」「重启雷电」「重启127.0.0.1:16384」。"
                "只在设备空闲（无任务运行）时允许重启。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "模拟器名称、别名或ADB串口",
                    },
                },
                "required": ["target"],
            },
        },
        {
            "name": "list_emulators",
            "description": (
                "查看当前所有已注册的模拟器清单——哪些在线、哪些离线、每个装了什么游戏。"
                "当用户问「哪些模拟器上有1999」「MuMu在线没」「有几个模拟器」时调用。"
                "调用前会自动扫描所有在线设备，所以每次调用返回的都是最新状态。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "scan_emulators",
            "description": (
                "全面扫描所有在线的模拟器设备——通过 adb shell pm list packages 自动检测"
                "每个设备上安装了哪些游戏，并更新模拟器清单。当用户说「扫描模拟器」「检查模拟器」"
                "「看看设备上有什么游戏」时调用。也适用于新设备首次接入后。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "chat_with_user",
            "description": (
                "向用户发送纯文本回复。用于闲聊、简单回应等不需要游戏操作的场景。"
                "调用此工具后本轮结束，回复内容会发送给用户。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "回复内容，用中文，1-3 句话为宜",
                    },
                },
                "required": ["message"],
            },
        },
        {
            "name": "create_guide",
            "description": (
                "保存一个游戏操作指引（guide）。用户说 /save 或 '存操作 基建收菜流程...' 时调用。"
                "steps 接受自然语言描述（如 '1. 点击基建\\n2. 点击铃铛'），会自动格式化。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "指引名称（英文/拼音，如 base-collect, farm-ce-5）",
                    },
                    "description": {
                        "type": "string",
                        "description": "一句话描述这个指引做什么",
                    },
                    "steps": {
                        "type": "string",
                        "description": "操作步骤，每行一步，自然语言即可",
                    },
                    "pitfalls": {
                        "type": "string",
                        "description": "注意事项，每行一个（可选）",
                    },
                    "tags": {
                        "type": "string",
                        "description": "逗号分隔的关键词（可选，如 基建,收菜,daily）",
                    },
                },
                "required": ["name", "description", "steps"],
            },
        },
    ])
    return schemas


# ── Tool handlers ──

def _delegate_to_agent(concierge: Any, task: str, game: str = "",
                       task_type: str = "custom",
                       _slot: Any = None,
                       preload_hint: str = "") -> str:
    """创建并启动 TerraAgent。slot 忙时自动排队。

    设备信号量生命周期：由 _on_agent_notify 在任务完成/出错时释放。
    game 参数可选——只有 1 个游戏时自动填充。
    _slot: Phase 3 — 强制指定目标 slot，用于批量派发。
    """
    import time as _time
    from src.agent.loop import TerraAgent
    from src.concierge.task_queue import enqueue_task, queue_size

    session = concierge.session
    slots: list[Any] = getattr(concierge, '_slots', [])

    # Normalize game: map Chinese names / keywords back to game IDs.
    # The LLM may pass the display name (e.g. "明日方舟") instead of the ID
    # ("arknights"), especially since the enum now lists both.
    if game:
        from src.games.registry import get_game_registry
        gr = get_game_registry()
        _game_name_map: dict[str, str] = {}
        for gid in gr.get_ids():
            plugin = gr.get(gid)
            if plugin:
                _game_name_map[plugin.manifest.name] = gid
                for kw in plugin.manifest.keywords:
                    _game_name_map[kw] = gid
        normalized = _game_name_map.get(game)
        if normalized:
            game = normalized

    # Auto-fill game: task context → active_game → registered games
    if not game:
        ctx_active = getattr(concierge.game_ctx, 'active_game', '') if hasattr(concierge, 'game_ctx') else ''
        if ctx_active:
            game = ctx_active
        else:
            from src.games.registry import get_game_registry
            all_games = get_game_registry().get_ids()
            if len(all_games) == 1:
                game = all_games[0]
            else:
                return "错误：未指定游戏。当前有多个游戏可用，请指定 game 参数。"

    # Find matching slot — _slot override for batch, else prefer idle device
    if _slot is not None:
        matching_slot = _slot
    elif slots:
        matching_slot = None
        # Prefer: device already on this game + idle
        on_game = [s for s in slots if s.game == game and s.is_free]
        if on_game:
            matching_slot = on_game[0]
        else:
            # Accept: any idle device (agent will switch games)
            idle = [s for s in slots if s.is_free]
            matching_slot = idle[0] if idle else None
            if matching_slot is None:
                # All busy — fall back to the session's device
                for s in slots:
                    if s.device_serial == session.agent_device_serial:
                        matching_slot = s
                        break
    else:
        matching_slot = None

    # ── Slot busy → queue ──
    if matching_slot and not matching_slot.is_free:
        enqueue_task(matching_slot, task, game=game, task_type=task_type)
        pos = queue_size(matching_slot)
        return (
            f"任务已加入排队（{matching_slot.label}，第 {pos} 位）: {task[:100]}。"
            f"当前任务完成后自动执行。"
        )

    device = matching_slot.device_serial if matching_slot else session.agent_device_serial
    session.agent_device_serial = device

    # Update slot.game to reflect what's about to run on this device
    if matching_slot and game:
        matching_slot.game = game

    # ── Slot free → execute now ──
    # Only stop existing task if running on the SAME device (single-slot backward compat)
    if not matching_slot:
        # No GameSlot config — use session-level agent tracking (Phase 1 mode)
        had_agent = session.current_agent is not None and session.current_agent.state.running
        if had_agent:
            old_agent = session.current_agent
            old_agent.state._pending_cancel = True
            old_agent.state.inject_message("管家派发了新任务，当前任务已取消。")
            old_agent.state.running = False
            session._device_owned = False
            logger.info("Concierge stopped existing agent on %s", device)

    # ADB health check — if offline, try rescanning before resorting to restart.
    # The emulator may have been started after bot init, so the device pool
    # might not have the right serial yet.
    from src.device.adb import get_adb as _get_adb
    import src.device.adb as _adb_mod
    adb_ok = False
    adb_dev = None
    try:
        adb_dev = _get_adb(serial=device)
        # Do a fresh heartbeat instead of trusting cached _heartbeat_ok.
        # The cached value may be stale from a transient failure.
        adb_ok = adb_dev.heartbeat()
    except RuntimeError:
        # Device not in pool — scan for online devices and try to register them
        from src.device.emulator import emulator_manager as _emu_mgr
        try:
            online_now = _emu_mgr.list_online
            for s in online_now:
                if s not in _adb_mod._adb_devices:
                    try:
                        from src.device.adb import init_adb as _init_adb
                        _init_adb(s)
                        logger.info("Discovered new device %s during delegation", s)
                        # Register with schedule engine so it can be reserved
                        if concierge.sched_engine:
                            concierge.sched_engine.add_device(s)
                    except Exception:
                        pass
        except Exception:
            pass
        # Retry with potentially newly discovered devices
        try:
            adb_dev = _get_adb(serial=device)
            adb_ok = adb_dev.heartbeat()
        except RuntimeError:
            # Still not found — try any device in the pool as last resort
            try:
                from src.device.adb import get_any_adb
                adb_dev = get_any_adb()
                adb_ok = adb_dev.heartbeat()
                if adb_ok:
                    device = adb_dev.serial  # remap to actual working device
                    session.agent_device_serial = device
                    logger.info("Remapped device %s → %s", "emulator-5554", device)
            except RuntimeError:
                adb_ok = False

    if not adb_ok:
        emu = getattr(concierge, 'emu_manager', None)
        if emu is not None:
            logger.info("Device %s offline — starting background restart", device)
            import threading

            def _restart_and_notify() -> None:
                """Restart emulator in background, then send WeChat notification."""
                result = emu.restart_emulator(device)
                logger.info("Restart result for %s: %s", device, result)

                # Send WeChat notification after restart completes
                bot = getattr(concierge, 'bot', None)
                loop = getattr(concierge, '_loop', None)
                user_id = getattr(concierge, 'user_id', None)
                if bot and loop and user_id:
                    import asyncio as _asyncio
                    if result == "ok":
                        # Find the new serial after restart (MuMu changes serial)
                        new_serial = device
                        try:
                            online = emu.list_online
                            if online and online[0] != device:
                                new_serial = online[0]
                        except Exception:
                            pass
                        msg = (
                            f"✅ 模拟器重启完成（{new_serial}）。"
                            "请重新发送指令继续任务。"
                        )
                    else:
                        msg = (
                            f"❌ 模拟器 {device} 重启失败（{result}）。"
                            "请检查电脑上的模拟器状态。"
                        )
                    try:
                        _asyncio.run_coroutine_threadsafe(
                            bot.send_message(user_id, msg), loop)
                    except Exception:
                        logger.exception("Failed to send restart notification")

            threading.Thread(
                target=_restart_and_notify, daemon=True,
            ).start()
            return (
                f"设备 {device} 离线，正在后台重启模拟器（约需 1 分钟）。"
                "重启完成后会通知你，届时重新发送指令即可。"
            )
        return f"设备 {device} 未连接。请确认模拟器是否在运行。发送「重启模拟器」来尝试恢复。"

    if concierge.sched_engine is not None:
        if matching_slot:
            # Phase 2: slot-level reservation
            if not concierge.sched_engine.reserve_slot(matching_slot.slot_id):
                return f"{matching_slot.label} 正忙，无法派发任务。"
        elif not session._device_owned:
            # Phase 1 fallback: device-level reservation
            if not concierge.sched_engine.reserve_device(device):
                return f"设备 {device} 正忙，无法派发任务。"
            session._device_owned = True

    agent = TerraAgent(device_serial=device, game=game)

    agent_id = str(session._agent_id_counter)
    session._agent_id_counter += 1

    from src.concierge.agent_pool import AgentHandle
    handle = AgentHandle(
        agent_id=agent_id, agent=agent, slot=matching_slot,
        task_description=task, game=game,
    )

    if matching_slot:
        matching_slot.current_task = handle
    else:
        # Phase 1: set session.current_agent only when no slots configured
        session.current_agent = agent

    def _notify(notif: Any) -> None:
        concierge._on_agent_notify(notif)

    try:
        agent.run_async(task, _notify, handle=handle)
    except Exception:
        # Clean up reservations if run_async fails.
        # Without this, the slot would be permanently deadlocked.
        logger.error("run_async failed — releasing slot/device reservations")
        if matching_slot and concierge.sched_engine:
            concierge.sched_engine.release_slot(matching_slot.slot_id)
        elif session._device_owned and concierge.sched_engine:
            concierge.sched_engine.release_device(device)
            session._device_owned = False
        if matching_slot:
            matching_slot.current_task = None
        raise

    # Inject game context hint as FIRST message (before the task).
    # Use inject_message() which is thread-safe (uses _interrupt_queue),
    # avoiding direct mutation of conversation_history from concierge thread.
    if preload_hint:
        agent.state.inject_message(preload_hint)

    logger.info("Concierge delegated task to agent on %s (game=%s): %s",
                 device, game, task[:100])
    from src.games.registry import get_game_registry
    game_label = get_game_registry().get_game_name(game)
    return f"任务已派发（{game_label}）: {task[:100]}（设备 {device}）"


def _check_agent_status(concierge: Any) -> str:
    """读取当前智能体状态快照。多 slot 时使用完整状态面板。"""
    # Multi-slot: use full status panel
    if concierge._slots and len(concierge._slots) > 1:
        from src.concierge.status_panel import format_status_panel
        return format_status_panel(concierge._slots)

    # Single agent (Phase 1)
    import time as _time

    session = concierge.session
    agent = session.current_agent

    if agent is None or not agent.state.running:
        return "当前没有正在运行的任务。"

    state = agent.state
    lines = [
        f"任务: {state.task_description or '(未指定)'}",
        f"游戏: {state.game}",
        f"设备: {state.device_serial or session.agent_device_serial}",
        f"已执行步数: {state.iteration_count}",
    ]

    if state.current_activity:
        lines.append(f"当前操作: {state.current_activity}")
    if state.last_progress_text and state.last_progress_text != state.current_activity:
        lines.append(f"最近进度: {state.last_progress_text[:200]}")
    if state.interrupt_zone != "safe":
        lines.append(f"所在区域: {state.interrupt_zone} ({state.interrupt_zone_detail})")

    if state.started_at > 0:
        elapsed = _time.time() - state.started_at
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        lines.append(f"已运行: {mins}分{secs}秒")

    return "\n".join(lines)


def _cancel_task(concierge: Any) -> str:
    """停止当前游戏智能体。

    多 slot 时仅停止 active_game 的 agent（一个游戏一个），
    不跨游戏停止。用户可用 #N stop 精确停止指定 agent。
    """
    # Multi-slot mode: stop only active game agents
    if concierge._slots and len(concierge._slots) > 1:
        active_game = concierge.game_ctx.active_game if concierge.game_ctx else "arknights"
        stopped = 0
        other_running = 0
        from src.concierge.router import _agent_running, _agent_from
        for s in concierge._slots:
            if not _agent_running(s.current_task):
                continue
            if s.game != active_game:
                other_running += 1
                continue
            handle = s.current_task
            if hasattr(handle, 'cancel'):
                handle.cancel()
            else:
                agent = _agent_from(handle)
                agent.state._pending_cancel = True
                agent.state.inject_message("用户通过管家要求停止当前任务。")
                agent.state.running = False
            # Phase 2: release slot semaphore so future tasks can use this slot
            if concierge.sched_engine:
                concierge.sched_engine.release_slot(s.slot_id)
            stopped += 1
        if stopped == 0:
            if other_running > 0:
                other_games = list(dict.fromkeys(
                    s.game for s in concierge._slots
                    if _agent_running(s.current_task)
                ))
                from src.games.registry import get_game_registry
                gr = get_game_registry()
                names = ", ".join(gr.get_game_name(g) for g in other_games)
                return (
                    f"当前 {gr.get_game_name(active_game)} 没有运行中的任务。"
                    f"正在运行的有 {names}。"
                    f"使用 #N stop 停止指定任务，或先说「切换到 {gr.get_game_name(other_games[0])}」再停止。"
                )
            return "当前没有正在运行的任务需要停止。"
        return f"已发出停止指令（{stopped} 个 {active_game} 任务）。智能体正在安全退出。"

    # Single agent (Phase 1 mode)
    session = concierge.session
    agent = session.current_agent

    if agent is None or not agent.state.running:
        return "当前没有正在运行的任务需要停止。"

    agent.state._pending_cancel = True
    agent.state.inject_message("用户通过管家要求停止当前任务。")
    agent.state.running = False
    logger.info("Concierge cancelled task on %s", session.agent_device_serial)

    return "已发出停止指令。智能体正在安全退出。"


# ── #N 引用路由 (Phase 2) ──
# TODO: When user says "#2 stop" or "#1 progress", find the agent by its
# global agent_id (from AgentHandle).  Currently stubbed — Phase 1 has only
# one agent so #N routing is unnecessary.

def _lookup_by_agent_num(concierge: Any, agent_num: str) -> Any | None:
    """Find AgentHandle by its #N number across all slots + session.

    Returns AgentHandle | None.
    Phase 1: always returns None (single agent, no #N routing needed).
    Phase 2: iterate slots + session.current_agent, match handle.agent_id.
    """
    if concierge is None:
        return None
    num = agent_num.lstrip("#")
    # Check slots
    for s in concierge._slots:
        handle = s.current_task
        if handle is not None and hasattr(handle, 'agent_id') and handle.agent_id == num:
            return handle
    # Check session-level agent (Phase 1 backward compat)
    agent = concierge.session.current_agent
    if agent is not None:
        # Phase 1: no AgentHandle, so just return the raw agent
        return agent
    return None


def _route_agent_command(concierge: Any, text: str) -> str | None:
    """Handle #N-prefixed management commands like '#2 stop', '#1 progress'.

    Returns a reply string if the command was handled, or None to fall through.
    Phase 1: always returns None.
    """
    import re
    m = re.match(r'#(\d+)\s*(.*)', text)
    if not m:
        return None
    num, cmd = m.group(1), m.group(2).strip()
    target = _lookup_by_agent_num(concierge, num)
    if target is None:
        return f"未找到 #{num}。可能已完成或编号错误。"

    # ── Phase 2: dispatch commands to target agent ──
    cmd_lower = cmd.lower().strip()
    agent = _agent_from(target)

    # stop / 停止
    if cmd_lower in ("stop", "停止", "取消"):
        from src.concierge.router import _agent_running
        if agent is not None and agent.state.running:
            agent.state._pending_cancel = True
            agent.state.inject_message("用户要求停止当前任务。")
            agent.state.running = False
            # Phase 2: release slot semaphore so the slot can be reused
            if hasattr(target, 'slot') and target.slot is not None and concierge.sched_engine:
                concierge.sched_engine.release_slot(target.slot.slot_id)
            logger.info("#%s stop: agent cancelled", num)
            return f"已向 #{num} 发出停止指令。"
        elif hasattr(target, 'completed_at') and target.completed_at:
            return f"#{num} 已完成（{getattr(target, 'outcome', 'unknown')}），无需停止。"
        return f"#{num} 当前空闲，无需停止。"

    # progress / status / 进度 / 状态
    if cmd_lower in ("progress", "status", "进度", "状态", ""):
        if agent is not None and agent.state.running:
            st = agent.state
            lines = [
                f"#{num} ({getattr(target, 'task_description', '?')[:60]}):",
                f"  当前: {st.current_activity or '启动中'}",
                f"  步数: {st.iteration_count}",
                f"  游戏: {st.game}",
            ]
            return "\n".join(lines)
        elif hasattr(target, 'outcome') and target.outcome:
            return f"#{num} 已完成（{target.outcome}）: {getattr(target, 'task_description', '')[:60]}"
        return f"#{num} 当前空闲。"

    # screenshot / 截图
    if cmd_lower in ("screenshot", "截图"):
        if agent is not None and agent.state.running:
            try:
                from src.agent.screen_injector import capture_screen_jpeg
                img_b64 = capture_screen_jpeg()
                if concierge.bot and concierge._loop:
                    import asyncio as _asyncio
                    _asyncio.run_coroutine_threadsafe(
                        concierge.bot.send_image(concierge.user_id, img_b64,
                                                 f"#{num} 当前屏幕"),
                        concierge._loop,
                    )
                return f"#{num} 截图已发送。"
            except Exception as exc:
                logger.error("#%s screenshot failed: %s", num, exc)
                return f"#{num} 截图失败: {exc}"
        return f"#{num} 未在运行，无法截图。"

    if cmd_lower:
        return f"#{num} 不支持的命令: {cmd}。可用: stop, progress, screenshot"
    return f"#{num} — 可用命令: stop, progress, screenshot"


# ── 批量路由 (Phase 3) ──

def _detect_batch(concierge: Any, text: str) -> tuple[list[Any], str] | None:
    """Detect batch-intent and return (slots, task) or None.

    Delegates to SlotRouter.detect_batch() for keyword matching and
    slot resolution. Returns the list of GameSlot objects + clean task.
    """
    slots: list[Any] = getattr(concierge, '_slots', [])
    if not slots or len(slots) <= 1:
        return None
    from src.concierge.slot_router import SlotRouter
    router = SlotRouter(slots, active_game=concierge.game_ctx.active_game, validate=False)
    result = router.detect_batch(text)
    if result is not None:
        return result.slots, result.task
    return None


def _delegate_batch(concierge: Any, task: str, game: str = "",
                    task_type: str = "custom") -> str:
    """向多个 GameSlot 批量派发同一任务。每个 slot 独立调度。

    空闲 slot 立即启动，忙碌 slot 自动排队。
    """
    import time as _time

    slots: list[Any] = getattr(concierge, '_slots', [])
    if not slots or len(slots) <= 1:
        # No batch slots — fall back to single delegate
        return _delegate_to_agent(concierge, task, game=game, task_type=task_type)

    # Filter to active game unless game specified.
    # In device-pool mode, unbound slots (game="") are also eligible.
    active_game = game or (concierge.game_ctx.active_game if concierge.game_ctx else "arknights")
    target_slots = [s for s in slots if s.game == active_game or not s.game]

    if len(target_slots) <= 1:
        if len(target_slots) == 1:
            return _delegate_to_agent(concierge, task, game=active_game, task_type=task_type)
        from src.games.registry import get_game_registry
        game_label = get_game_registry().get_game_name(active_game)
        return f"当前没有可用的设备来做 {game_label} 任务。"

    # Save + restore session device so batch loop doesn't corrupt state
    saved_device = concierge.session.agent_device_serial

    started = 0
    queued = 0
    from src.concierge.task_queue import enqueue_task

    for s in target_slots:
        if s.is_free:
            _delegate_to_agent(
                concierge, task, game=s.game, task_type=task_type, _slot=s,
            )
            started += 1
            logger.info("Batch: started task on slot %s (%s)", s.slot_id, s.label)
        else:
            enqueue_task(s, task, s.game, task_type)
            queued += 1
            logger.info("Batch: queued task on slot %s (%s)", s.slot_id, s.label)

    # Restore session device to pre-batch state
    concierge.session.agent_device_serial = saved_device

    from src.games.registry import get_game_registry
    gr = get_game_registry()
    game_label = gr.get_game_name(active_game)

    parts: list[str] = [f"已向 {len(target_slots)} 个 {game_label} 账号批量派发: {task[:80]}"]
    if started:
        parts.append(f"{started} 个已启动")
    if queued:
        parts.append(f"{queued} 个已排队")
    return "，".join(parts) + "。"


def _chat_with_user(concierge: Any, message: str) -> str:
    """存储要发送给用户的回复文本。"""
    concierge._pending_reply = message
    return f"回复已准备: {message[:80]}..."


# ── Dispatch table (handler names fixed) ──

def _start_emulator(concierge: Any) -> str:
    """启动模拟器。离线则启动主实例，在线则启动已克隆的多开实例。"""
    emu = getattr(concierge, 'emu_manager', None)
    if emu is None:
        return "模拟器管理器未初始化。"

    online_before = list(emu.list_online) if emu.list_online else []
    force = len(online_before) > 0  # 在线 → 启动多开；离线 → 启动主实例
    logger.info("Starting emulator — %d online, force_new=%s", len(online_before), force)

    new_serial = emu.launch_emulator(force_new=force)
    if new_serial is None:
        if force:
            return "启动多开实例失败。请确认已用多开器克隆过实例。"
        return "启动模拟器失败。请检查电脑上模拟器是否正常。"

    try:
        from src.device.adb import init_adb
        init_adb(new_serial)
    except Exception as e:
        return f"模拟器已启动但 ADB 连接失败 ({new_serial}): {e}"

    try:
        emu.start_health_monitor(new_serial)
    except Exception:
        pass

    # Auto-launch Arknights on the clone — it's a clone of an Arknights
    # emulator, so Arknights is guaranteed to be installed and logged in.
    # Don't make the agent fumble through the Android desktop.
    from src.tools.emulator_tools import _start_game_via_adb
    _start_game_via_adb(new_serial)

    # Create a device slot (game="" = unbound, will be set when task delegates)
    from src.concierge.slot_router import GameSlot
    port = new_serial.split(":")[-1] if ":" in new_serial else new_serial
    short = new_serial.replace("127.0.0.1:", "")
    aliases = list(dict.fromkeys([new_serial, short, port]))
    new_slot = GameSlot(
        slot_id=f"dev_{short.replace('.', '_')}",
        label=f"设备 {short}",
        aliases=aliases,
        game="",  # 不绑定游戏，运行时动态分配
        device_serial=new_serial,
    )
    concierge._slots.append(new_slot)
    # Rebuild tool schemas with the new slot
    concierge._tool_schemas = build_concierge_tool_schemas(concierge._slots)

    # Register with schedule engine
    if concierge.sched_engine:
        concierge.sched_engine.add_device(new_serial)
        concierge.sched_engine._slot_to_device[new_slot.slot_id] = new_serial

    return (
        f"✅ 新模拟器实例已启动（{new_serial}）。\n"
        f"设备已注册，现在可以派发任务了。"
    )


def _list_emulators(concierge: Any) -> str:
    """列出所有已注册的模拟器、在线状态、安装的游戏。会自动扫描在线设备。"""
    from src.device.emulator_inventory import get_emulator_inventory, format_inventory_for_agent
    from src.device.emulator import emulator_manager

    inventory = get_emulator_inventory()
    online_map = inventory.all_online_serials(emulator_manager)
    # Auto-discover games on all online devices
    online_serials = list(online_map.values())
    if online_serials:
        from config.settings import config as _cfg
        inventory.auto_discover_all(online_serials, adb_path=_cfg.adb.path)
    return format_inventory_for_agent(inventory, online_map)


def _scan_emulators(concierge: Any) -> str:
    """扫描所有在线模拟器，自动发现安装的游戏并更新清单。"""
    from src.device.emulator_inventory import get_emulator_inventory, format_inventory_for_agent
    from src.device.emulator import emulator_manager
    from config.settings import config as _cfg

    inventory = get_emulator_inventory()
    emulator_manager._probe_emu_ports()
    online_serials = emulator_manager.list_online
    if not online_serials:
        return "当前没有在线模拟器。"

    discovered = inventory.auto_discover_all(online_serials, adb_path=_cfg.adb.path)
    if not discovered:
        return "扫描完成，未发现新的游戏安装。"

    online_map = inventory.all_online_serials(emulator_manager)
    scan_lines = [f"扫描了 {len(discovered)} 台设备，发现游戏："]
    for serial, games in discovered.items():
        from src.games.registry import get_game_registry
        gr = get_game_registry()
        game_names = ", ".join(gr.get_game_name(g) for g in games)
        scan_lines.append(f"  {serial}: {game_names}")
    scan_lines.append("")
    scan_lines.append(format_inventory_for_agent(inventory, online_map))
    return "\n".join(scan_lines)


def _restart_emulator(concierge: Any, target: str) -> str:
    """重启指定模拟器。target 用名称、别名或 ADB 串口匹配。"""
    from src.device.emulator_inventory import get_emulator_inventory

    emu = getattr(concierge, 'emu_manager', None)
    if emu is None:
        return "模拟器管理器未初始化。"

    inventory = get_emulator_inventory()
    entry = inventory.find_by_alias(target)
    if entry is None:
        # Try direct serial match
        entry = inventory.find_by_serial(target)
    if entry is None:
        # Try port substring
        for e in inventory.list_all():
            if target in e.adb_ports or target in (e.current_serial or ""):
                entry = e
                break
    if entry is None:
        return f"未找到匹配的模拟器: {target}。发送「查看模拟器」查看可用模拟器。"

    serial = entry.current_serial
    if not serial:
        # Try to find online device on any of this entry's ports
        emu.discover()
        for port in entry.adb_ports:
            for dev_serial, state in emu._devices.items():
                if f":{port}" in dev_serial and state == "device":
                    serial = dev_serial
                    break
            if serial:
                break
        if not serial:
            return f"{entry.name} 当前不在线，无法重启。请先用 start_emulator 启动它。"

    # Check no active task on this device
    for s in concierge._slots:
        if s.device_serial == serial and not s.is_free:
            task_desc = getattr(s.current_task, 'task_description', '') or '执行中'
            return f"{entry.name} 正在执行任务（{task_desc[:60]}），无法重启。先停止任务再试。"

    result = emu.restart_emulator(serial)
    if result == "ok":
        entry.current_serial = ""  # Will be re-discovered by list_online
        return f"✅ {entry.name} 重启完成。"
    elif result == "already_running":
        return f"{entry.name} 正在重启中，请稍候。"
    return f"❌ {entry.name} 重启失败（{result}）。请检查电脑上的模拟器状态。"


# ── create_guide handler ─────────────────────────────────────────────

def _create_guide_handler(
    concierge: Any,
    name: str,
    description: str,
    steps: str,
    pitfalls: str = "",
    tags: str = "",
) -> str:
    """Concierge wrapper — fills in game from active context, calls guide_tool."""
    from src.tools.guide_tool import create_guide

    game = getattr(concierge.game_ctx, "active_game", "arknights") if hasattr(concierge, "game_ctx") else "arknights"
    result = create_guide(
        name=name, description=description, steps=steps,
        game=game, pitfalls=pitfalls, tags=tags,
    )
    import json
    data = json.loads(result.text)
    if data.get("success"):
        return data.get("message", f"已保存指引 {name}")
    return f"保存失败: {data.get('error', '未知错误')}"


# ── Concierge LLM tools (for the restored LLM-based concierge) ──


def build_concierge_llm_tools(slots: list[Any]) -> list[dict[str, Any]]:
    """Build tool schemas for the concierge LLM (single-turn, non-agent-loop).

    The concierge LLM has 3 tools:
      - delegate_to_game: delegate a task to a game agent on a device
      - ask_user: ask the user for clarification
      - chat_with_user: reply with text only (no action)
    """
    from src.games.registry import get_game_registry
    gr = get_game_registry()
    all_games = gr.get_ids()
    game_names: dict[str, str] = {}
    for g in all_games:
        plugin = gr.get(g)
        game_names[g] = plugin.manifest.name if plugin else g

    # delegate_to_game
    delegate_props: dict[str, Any] = {
        "game": {
            "type": "string",
            "enum": list(all_games) + [game_names.get(g, g) for g in all_games],
            "description": "目标游戏: " + ", ".join(
                f"{g}={game_names.get(g, g)}" for g in all_games
            ),
        },
        "task": {
            "type": "string",
            "description": (
                "任务描述，用中文。如「基建换班」「清体力」「完成日常任务」。"
                "**重要**：保留用户原文中的游戏标识词（如 1999、方舟、nikki），"
                "不要把「完成1999日常任务」简化为「完成日常任务」——Agent 依赖这些关键词识别目标游戏。"
            ),
        },
    }
    # If multiple slots, optionally specify which one
    if len(slots) > 1:
        slot_labels = []
        for s in slots:
            label = getattr(s, "label", s.slot_id if hasattr(s, "slot_id") else "?")
            slot_labels.append(label)
        delegate_props["slot"] = {
            "type": "string",
            "description": "目标设备（可选）。可用: " + ", ".join(slot_labels),
        }

    return [
        {
            "name": "delegate_to_game",
            "description": (
                "将任务委派给指定游戏的 Agent 执行。Agent 会自动操作手机。"
                "**每个游戏只能调用一次，不同游戏不能合并。**"
                "**多游戏请求必须为每一个游戏分别调用此工具，不能遗漏。**"
                "委派后任务立即开始执行，你会收到确认消息。"
            ),
            "parameters": {
                "type": "object",
                "properties": delegate_props,
                "required": ["game", "task"],
            },
        },
        {
            "name": "check_devices",
            "description": (
                "检查当前可用设备状态：有多少设备、每个设备上运行的是哪个游戏、"
                "是否有空闲设备。在委派前调用此工具确认设备够用。"
                "如果设备不够，建议用户启动新模拟器或选择使用哪些设备。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "launch_emulator",
            "description": (
                "启动一个新的模拟器实例。\n"
                "- 多游戏请求设备不够时：**直接调用，不需要确认。**\n"
                "- 单游戏请求：先 ask_user() 确认再调用。\n"
                "启动成功后新设备会自动注册，可以立即 delegate 任务过去。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "ask_user",
            "description": (
                "向用户提问，等待回复。用于以下场景：\n"
                "- 用户意图模糊，无法确定游戏或任务\n"
                "- 需要确认是否启动新模拟器\n"
                "- 设备分配有歧义\n"
                "调用后本轮结束，用户回复后你会被再次调用以继续处理。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "向用户提出的问题，用中文，简洁清晰",
                    },
                },
                "required": ["question"],
            },
        },
        {
            "name": "chat_with_user",
            "description": (
                "向用户发送纯文本回复。用于问候、感谢、闲聊等不需要游戏操作的场景。"
                "调用此工具后本轮结束。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "回复内容，用中文，1-3 句话",
                    },
                },
                "required": ["message"],
            },
        },
    ]


def _llm_delegate_to_game(concierge: Any, game: str, task: str,
                           slot: str = "") -> str:
    """Concierge LLM tool: delegate a task to a game agent."""
    # Resolve slot label to actual slot object if provided
    _slot_obj = None
    free_slots = [s for s in getattr(concierge, '_slots', []) if s.is_free]
    if slot and hasattr(concierge, '_slots'):
        for s in concierge._slots:
            if getattr(s, "label", "") == slot or \
               getattr(s, "slot_id", "") == slot:
                _slot_obj = s
                break

    if not _slot_obj and not free_slots and getattr(concierge, '_slots', []):
        # No free slots — task will be queued by _delegate_to_agent.
        # Include a warning so the LLM knows to tell the user.
        logger.warning("Concierge LLM: no free slots for %s — task will queue", game)

    result = _delegate_to_agent(concierge, task=task, game=game, _slot=_slot_obj)

    # Record delegated task for status tracking
    from src.games.registry import get_game_registry
    gr = get_game_registry()
    plugin = gr.get(game)
    game_name = plugin.manifest.name if plugin else game

    if _slot_obj:
        slot_info = f" ({getattr(_slot_obj, 'label', '?')})"
    else:
        slot_info = ""

    logger.info("Concierge LLM delegated: %s → %s%s: %s",
                task[:40], game_name, slot_info, result[:80])
    return f"[{game_name}]{slot_info} {task} — {result}"


def _llm_ask_user(concierge: Any, question: str) -> str:
    """Concierge LLM tool: ask the user a question, pause until reply.

    Sets pending_clarification to "concierge_confirm|..." so the user's
    reply comes back through process_message for the concierge LLM to
    continue processing.
    """
    concierge.session.pending_clarification = "concierge_confirm|" + question
    concierge.session.pending_task = question
    logger.info("Concierge LLM ask_user: %s", question[:100])
    return "[PENDING_CLARIFICATION] " + question


def _llm_chat_with_user(concierge: Any, message: str) -> str:
    """Concierge LLM tool: reply with text only."""
    concierge._pending_reply = message
    logger.info("Concierge LLM chat: %s", message[:100])
    return "[CHAT] " + message


def _llm_check_devices(concierge: Any) -> str:
    """Concierge LLM tool: return device availability status."""
    slots = getattr(concierge, '_slots', [])
    if not slots:
        return "当前没有可用的设备。请让用户先启动模拟器。"

    lines = []
    for s in slots:
        label = getattr(s, "label", s.slot_id if hasattr(s, "slot_id") else "?")
        game = getattr(s, "game", "") or "空闲"
        busy = "忙碌" if (getattr(s, "current_task", None) is not None) else "空闲"
        serial = getattr(s, "device_serial", "") or "?"
        lines.append(f"  {label} ({serial}): {game} / {busy}")

    free = [s for s in slots if s.is_free]
    summary = f"{len(slots)} 个设备，{len(free)} 个空闲。"
    return summary + "\n" + "\n".join(lines)


def _llm_launch_emulator(concierge: Any) -> str:
    """Concierge LLM tool: launch a new emulator instance."""
    emu = getattr(concierge, 'emu_manager', None)
    if emu is None:
        return "错误：没有可用的模拟器管理器。请告知用户无法启动新模拟器。"

    # Safety: don't launch if agents are running
    running = [s for s in getattr(concierge, '_slots', [])
               if getattr(s, "current_task", None) is not None]
    if running:
        return "错误：有任务正在运行，启动模拟器会断开 ADB 连接。告知用户等任务完成后再试。"

    try:
        ns = emu.launch_emulator(force_new=True)
        if ns:
            emu.start_health_monitor(ns)
            # Register new device with scheduler
            try:
                from src.scheduler.cron_scheduler import get_engine as _get_sched
                _sched = _get_sched()
                if ns not in _sched.device_serials:
                    _sched.add_device(ns)
            except Exception:
                pass

            # Create new slot
            from src.concierge.slot_router import GameSlot
            short = ns.replace("127.0.0.1:", "")
            new_slot = GameSlot(
                slot_id=f"dev_{short.replace('.', '_')[:6]}",
                label=f"模拟器 {short}",
                aliases=[ns, short],
                game="", device_serial=ns,
            )
            concierge._slots.append(new_slot)
            logger.info("Concierge LLM: launched new emulator %s → slot %s", ns, new_slot.slot_id)
            return f"✅ 新模拟器已启动：{ns}。现在可以委派任务了。"
        else:
            return "启动失败：launch_emulator 返回空。可能已经达到最大实例数。"
    except Exception as e:
        logger.error("Concierge LLM launch_emulator failed: %s", e)
        return f"启动失败: {e}"


# ── Concierge LLM tool dispatch (for LLM tool_calls → handler mapping) ──

CONCIERGE_LLM_TOOL_DISPATCH: dict[str, Callable[..., str]] = {
    "delegate_to_game": _llm_delegate_to_game,
    "check_devices": _llm_check_devices,
    "launch_emulator": _llm_launch_emulator,
    "ask_user": _llm_ask_user,
    "chat_with_user": _llm_chat_with_user,
}


CONCIERGE_TOOL_DISPATCH: dict[str, Callable[..., str]] = {
    "delegate_to_agent": _delegate_to_agent,
    "delegate_batch": _delegate_batch,
    "create_guide": _create_guide_handler,
    "check_agent_status": _check_agent_status,
    "cancel_task": _cancel_task,
    "start_emulator": _start_emulator,
    "list_emulators": _list_emulators,
    "scan_emulators": _scan_emulators,
    "restart_emulator": _restart_emulator,
    "chat_with_user": _chat_with_user,
}

# Legacy static schemas (fallback) — use build_concierge_tool_schemas() instead
CONCIERGE_TOOL_SCHEMAS = build_concierge_tool_schemas([])
