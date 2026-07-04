"""MessageRouter — 用户会话管理 + 消息路由 + 多游戏任务分发。

全部使用确定性逻辑（无 LLM）。每个 WeChat 用户对应一个实例。
"""

from __future__ import annotations

import atexit
import logging
import re
import threading
from typing import Any

from src.concierge.session import ConciergeSession
from src.concierge.tools import CONCIERGE_TOOL_DISPATCH
from src.concierge.fast_path import try_fast_path
from src.concierge.notification_buffer import NotificationBuffer

# ── Create-guide prefix patterns ────────────────────────────────────
# Shared with agent/loop.py:_match_create_guide_prefix.
# Detected here so /save bypasses the concierge LLM entirely.

_CREATE_GUIDE_PREFIXES = [
    "/save ", "/s ", "/save\n", "/s\n", "/save", "/s",
    "存操作 ", "存操作\n", "存操作",
    "加操作 ", "加操作\n", "加操作",
    "保存指引 ", "保存指引\n", "保存指引",
]


def _match_create_guide_prefix(message: str) -> str | None:
    """If message starts with a create-guide prefix, return the body after it."""
    if not message:
        return None
    msg = message.strip()
    for prefix in _CREATE_GUIDE_PREFIXES:
        if msg.startswith(prefix):
            body = msg[len(prefix):].strip()
            return body if body else msg
    return None

logger = logging.getLogger(__name__)


# ── Module-level helpers ──────────────────────────────────────────


def _extract_game_and_task(body: str) -> tuple[str, str]:
    """Extract game name and task from observation command body.

    When no registered game keywords match, the first word is the game
    name and the rest is the task description.
    Returns (game_name, task) — task is empty if body is a single word.
    """
    body = body.strip()
    if not body:
        return "", ""

    parts = body.split(None, 1)
    game = parts[0].strip().lower()
    task = parts[1].strip() if len(parts) > 1 else ""
    return game, task


def _agent_from(obj: Any) -> Any | None:
    """Extract TerraAgent from AgentHandle, TerraAgent, or None."""
    if obj is None:
        return None
    if hasattr(obj, 'agent'):
        return obj.agent
    return obj


def _agent_running(obj: Any) -> bool:
    """Check if the underlying TerraAgent is running."""
    agent = _agent_from(obj)
    return agent is not None and agent.state.running


class MessageRouter:
    """一个 WeChat 用户对应一个 MessageRouter 实例。

    线程模型：
    - process_message() 在 ThreadPoolExecutor 线程中串行执行（_process_lock）
    - _on_agent_notify() 在 TerraAgent 的 daemon 线程中被回调
    """

    def __init__(
        self,
        user_id: str,
        device_serial: str,
        bot: Any,
        sched_engine: Any,
        slots: list[Any] | None = None,
        emu_manager: Any = None,
    ) -> None:
        self.user_id = user_id
        self.session = ConciergeSession(
            user_id=user_id,
            agent_device_serial=device_serial,
        )
        self.bot = bot
        self.sched_engine = sched_engine
        self.emu_manager = emu_manager  # For auto-restarting offline devices
        self._process_lock = threading.Lock()
        self._notif_buf = NotificationBuffer()  # progress dedup + throttle

        # GameSlots — for multi-game/multi-account routing
        self._slots: list[Any] = slots or []

        # Game context — cross-message active_game persistence
        from src.concierge.game_context import UserGameContext
        if self.session.game_ctx is None:
            from config.settings import config
            self.session.game_ctx = UserGameContext(active_game=config.state.game)
        self.game_ctx = self.session.game_ctx

        import asyncio as _asyncio
        try:
            self._loop = _asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
            logger.warning("MessageRouter created outside asyncio — "
                           "notifications disabled")

        # Phase 3: recover persisted task queues from DB
        if self._slots:
            try:
                from src.concierge.task_queue import load_all_queues
                loaded = load_all_queues(self._slots)
                if loaded:
                    logger.info("MessageRouter %s: recovered %d queued tasks from DB",
                              user_id, loaded)
            except Exception:
                logger.debug("Queue recovery skipped (DB not available)")

        # Slot activity callback — called when an agent sends a notification
        # so the gateway can update last_active_slot / waiting_for_reply tracking.
        self._slot_activity_callback: Any = None

        # Observation learning — active recorder (None when not recording)
        self._active_recorder: Any = None

        # Session persistence — load snapshot on first message, save on every message
        self._snapshot_loaded = False

        # Register atexit hook for final save
        atexit.register(self._on_exit_save)

    def set_slot_activity_callback(self, cb: Any) -> None:
        """Register a callback invoked when an agent sends a notification.

        Called with (slot_id: str, event_type: str) where event_type is one of:
        "ask_user", "complete", "error", "progress".
        """
        self._slot_activity_callback = cb

    def _resolve_slot_id(self, agent_id: str | None) -> str | None:
        """Resolve agent_id to slot_id. Returns None if not found."""
        if not agent_id or not self._slots:
            if agent_id and not self._slots:
                logger.debug("_resolve_slot_id: agent_id=%s but no slots configured", agent_id)
            return None
        for s in self._slots:
            handle = s.current_task
            if handle is not None and hasattr(handle, 'agent_id') and handle.agent_id == agent_id:
                return s.slot_id
        logger.debug("_resolve_slot_id: agent_id=%s not found in %d slots (available: %s)",
                   agent_id, len(self._slots),
                   [(s.slot_id, getattr(s.current_task, 'agent_id', None) if s.current_task else None)
                    for s in self._slots])
        return None

    # ── Slot lookup helpers (Phase 2) ──────────────────────────────

    def _get_slot_for_device(self, device_serial: str) -> Any | None:
        """Find the GameSlot matching a device serial."""
        for s in self._slots:
            if s.device_serial == device_serial:
                return s
        return None

    def _get_slot_by_label(self, label: str) -> Any | None:
        """Find a GameSlot by label or alias (case-insensitive match)."""
        for s in self._slots:
            if s.match_label(label):
                return s
        return None

    def _detect_multi_game(self, text: str) -> list[str]:
        """Return list of game IDs mentioned in text, in order of first occurrence.
        Handles "方舟和1999", "明日方舟 重返未来", "arknights + reverse1999", etc.
        Only returns games that are actually registered (via GameRegistry keywords).
        """
        from src.games.registry import get_game_registry
        gr = get_game_registry()
        text_lower = text.lower()

        # Find the first keyword match position for each registered game
        game_positions: list[tuple[str, int]] = []
        for plugin in gr.list_all():
            earliest = None
            for kw in plugin.manifest.keywords:
                pos = text_lower.find(kw.lower())
                if pos >= 0 and (earliest is None or pos < earliest):
                    earliest = pos
            if earliest is not None:
                game_positions.append((plugin.manifest.id, earliest))

        # Sort by position → preserves the user's intended order
        game_positions.sort(key=lambda x: x[1])
        games = [g for g, _ in game_positions]

        # Deduplicate (same game mentioned multiple times)
        seen: set[str] = set()
        result: list[str] = []
        for g in games:
            if g not in seen:
                seen.add(g)
                result.append(g)
        return result

    def _split_multi_game_task(self, text: str) -> str | None:
        """Detect multi-game request and hand off to Concierge LLM.

        Detection + text splitting is deterministic (no LLM).  The LLM
        decides how to handle device constraints — it can confirm with the
        user, auto-launch emulators, or queue tasks.

        Skips messages that look like questions/complaints/corrections.
        """
        # ── Guard: skip questions, complaints, and corrections ──
        _non_task_markers = [
            "为什么", "怎么", "干嘛", "干啥", "搞错", "弄错", "不对",
            "不是", "错了", "怎么会", "咋回事", "什么意思",
        ]
        _has_question = "?" in text or "？" in text
        _has_complaint = any(m in text for m in _non_task_markers)
        if _has_question or _has_complaint:
            return None

        games = self._detect_multi_game(text)
        if len(games) < 2:
            return None

        sub_tasks = self._extract_sub_tasks(text, games)
        logger.info(
            "Multi-game text split: '%s' → %s",
            text[:80],
            {g: sub_tasks.get(g, "") for g in games},
        )

        from src.concierge.prompts import build_multi_game_context
        context = build_multi_game_context(
            text, games, sub_tasks,
            slots=self._slots,
            emu_available=self.emu_manager is not None,
        )
        return self._run_concierge_llm(
            text, context=context,
            expected_games=games, expected_tasks=sub_tasks,
        )

    # ── Concierge LLM (multi-turn agent loop) ──

    _CONCIERGE_MAX_TURNS = 5

    def _run_concierge_llm(self, text: str, context: str = "",
                            expected_games: list[str] | None = None,
                            expected_tasks: dict[str, str] | None = None,
                            ) -> str | None:
        """Multi-turn concierge agent loop.

        The Concierge is NOT single-turn.  It can call tools, see results,
        and continue — just like the main TerraAgent.  This is critical for
        multi-step workflows like "launch emulator → see new device →
        delegate tasks to all devices".

        max_turns=5 prevents runaway loops.  The loop stops when:
        - LLM calls ask_user → return the question to user
        - LLM calls chat_with_user (no tools) → return the text
        - No tool calls at all (plain text) → return text or fallback
        - All expected games are dispatched → combine results, return summary
        """
        from src.concierge.prompts import (
            build_concierge_system_prompt,
            build_games_info,
            build_slots_info,
        )
        from src.concierge.tools import (
            build_concierge_llm_tools,
            CONCIERGE_LLM_TOOL_DISPATCH,
        )

        games_info = build_games_info()
        system = build_concierge_system_prompt(
            games_info=games_info,
            slots_info=build_slots_info(self._slots),
            context=context if context else f"用户消息：「{text}」",
        )
        tools = build_concierge_llm_tools(self._slots)

        messages: list[dict] = [{"role": "user", "content": text}]
        all_results: list[str] = []
        delegated: set[str] = set()  # game IDs already delegated

        from src.llm.client import MiMoClient

        for turn in range(self._CONCIERGE_MAX_TURNS):
            try:
                client = MiMoClient()
                response = client.chat(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=1024,
                    temperature=0.3,
                )
            except Exception as e:
                logger.warning("Concierge LLM failed (turn %d): %s", turn, e)
                if turn == 0:
                    return self._fallback_direct_delegate(text)
                break

            # Extract text and tool calls
            tool_calls: list[dict] = []
            turn_text = ""
            for block in getattr(response, "content", []) or []:
                if getattr(block, "type", "") == "tool_use":
                    tool_calls.append({
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input": dict(getattr(block, "input", {}) or {}),
                    })
                elif getattr(block, "type", "") == "text":
                    turn_text += getattr(block, "text", "")

            # ── Stop condition: LLM wants to talk to user ──
            if not tool_calls:
                if turn_text.strip():
                    logger.info("Concierge LLM text reply (turn %d): %s", turn, turn_text[:100])
                    if all_results:
                        return turn_text.strip() + "\n" + "\n".join(all_results)
                    return turn_text.strip()
                logger.warning("Concierge LLM no tools and no text (turn %d) — fallback", turn)
                if all_results:
                    return "\n".join(all_results)
                return self._fallback_direct_delegate(text)

            # ── Execute tools; inject results; track state ──
            tool_results: list[dict] = []
            stop_loop = False

            for tc in tool_calls:
                name = tc["name"]
                handler = CONCIERGE_LLM_TOOL_DISPATCH.get(name)
                if handler is None:
                    continue

                try:
                    raw = handler(self, **tc["input"])
                except Exception as e:
                    logger.error("Concierge tool '%s' failed: %s", name, e)
                    raw = f"[{name}失败: {e}]"

                logger.info("Concierge LLM tool (turn %d): %s → %s", turn, name, str(raw)[:120])

                # Track delegated games
                if name == "delegate_to_game":
                    g = tc["input"].get("game", "")
                    if g:
                        delegated.add(g)
                    all_results.append(raw)
                elif name == "ask_user":
                    stop_loop = True
                    all_results.append(raw)
                elif name == "chat_with_user":
                    stop_loop = True
                    all_results.append(raw)
                else:
                    all_results.append(raw)

                # Build tool_result block for LLM to see
                tr_block: dict = {
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                }
                if isinstance(raw, str) and len(raw) > 4000:
                    tr_block["content"] = raw[:4000] + "…(已截断)"
                else:
                    tr_block["content"] = raw
                tool_results.append(tr_block)

            # Inject assistant + tool_results into conversation
            assistant_block: dict = {"role": "assistant", "content": []}
            if turn_text.strip():
                assistant_block["content"].append({"type": "text", "text": turn_text.strip()})
            for tc in tool_calls:
                assistant_block["content"].append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })
            messages.append(assistant_block)
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # ── Stop conditions ──
            if stop_loop:
                break

            # All expected games dispatched?
            if expected_games and set(expected_games) <= delegated:
                logger.info("Concierge: all %d games dispatched in %d turns",
                           len(expected_games), turn + 1)
                break

            # Refresh slots_info for next turn (launch_emulator may have added one)
            system = build_concierge_system_prompt(
                games_info=games_info,
                slots_info=build_slots_info(self._slots),
                context=context if context else "",
            )

        if all_results:
            return "\n".join(all_results)
        return None

    def _fallback_direct_delegate(self, text: str) -> str | None:
        """Fallback: directly delegate a single-game task without LLM."""
        detected = self._detect_multi_game(text)
        from config.settings import config as _cfg
        game = detected[0] if detected else _cfg.state.game

        from src.games.registry import get_game_registry
        gr = get_game_registry()
        game_name = gr.get_game_name(game)

        task = f"({game_name}) {text}"
        from src.concierge.tools import _delegate_to_agent
        result = _delegate_to_agent(self, task=task, game=game)
        logger.info("Fallback direct delegate: %s → %s", game_name, result[:100])
        return f"已为你启动 {game_name}: {text}"

    # ── 多游戏任务拆分（纯文本，不用 LLM） ──
    # MiMo 等模型反复拒绝输出纯 JSON，总是先写推理文本再输出 JSON，
    # max_tokens 不管设多高都会被推理耗尽。这是 LLM 训练偏好问题，
    # 不是 prompt 能修的。对于"完成方舟和1999日常"这类固定模式，
    # 纯文本处理完全够用——检测游戏名 → 去掉所有游戏关键词 → 得通用任务。

    # Deprecated: kept for external references / tests.
    _MULTI_GAME_SPLIT_SYSTEM = "（纯文本拆分，不再使用 LLM）"

    def _extract_sub_tasks(self, text: str, games: list[str]) -> dict[str, str]:
        """Split a multi-game message into per-game task descriptions.

        Deterministic — no LLM.  Uses game keywords as boundary markers to
        partition the text: each game gets the text segment between its
        keyword and the next game's keyword.  Falls back to shared-task
        mode when the partitions yield empty segments (e.g. "完成方舟和1999日常").
        """
        from src.games.registry import get_game_registry
        gr = get_game_registry()

        # Collect game keywords and find their positions
        game_kw: dict[str, set[str]] = {}
        for g in games:
            plugin = gr.get(g)
            game_kw[g] = set(plugin.manifest.keywords) if plugin else set()

        # Find the boundary where each game's keywords appear in the text.
        # For each game, record the earliest keyword position and the keyword
        # that matched — these are the partition boundaries.
        text_lower = text.lower()
        boundaries: list[tuple[str, int, int]] = []  # (game, start_pos, end_pos)
        for g in games:
            earliest_pos = len(text)
            earliest_len = 0
            for kw in game_kw[g]:
                pos = text_lower.find(kw.lower())
                if pos >= 0 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_len = len(kw)
            boundaries.append((g, earliest_pos, earliest_pos + earliest_len))

        # Sort by position → partition text between game boundaries
        boundaries.sort(key=lambda x: x[1])

        # Partition: each game gets text from its keyword end to the next
        # game's keyword start, or end of text for the last one.
        result: dict[str, str] = {}
        for i, (g, _start, kw_end) in enumerate(boundaries):
            if i + 1 < len(boundaries):
                segment_end = boundaries[i + 1][1]
            else:
                segment_end = len(text)
            segment = text[kw_end:segment_end]

            # Clean the segment: remove inter-game connectives and leading/
            # trailing punctuation.  Do NOT strip additional game keywords
            # from the segment — keywords that appear inside a task segment
            # are task-relevant words (e.g. "基建" in "基建换班"), not game
            # identifiers.  The matched game keyword is already outside the
            # segment (segment starts at kw_end).
            cleaned = segment
            # Strip leading/trailing punctuation & whitespace
            cleaned = re.sub(r'^[\s,，。.、]+|[\s,，。.、]+$', '', cleaned)
            # Remove inter-game connectives (whole words)
            cleaned = re.sub(r'\s*(和|与|及|然后|同时)\s*', ' ', cleaned)
            cleaned = cleaned.strip()

            # If partition yielded an empty segment, fall back to shared mode.
            # This handles "完成方舟和1999日常" where both games share one task.
            if not cleaned or len(cleaned) < 2:
                result[g] = ""  # flag for shared fallback
            else:
                result[g] = cleaned

        # Shared fallback: if any game got an empty segment, rebuild from
        # the full text by stripping ALL game keywords (original behavior).
        if any(not v for v in result.values()):
            all_kw: set[str] = set()
            for kws in game_kw.values():
                all_kw.update(kws)
            shared = text
            for kw in sorted(all_kw, key=len, reverse=True):
                shared = shared.replace(kw, "")
            shared = re.sub(r'\s*(和|与|及|、|然后|同时|完成)\s*', ' ', shared)
            shared = re.sub(r'\s+', ' ', shared).strip()
            if not shared or len(shared) < 2:
                shared = "日常任务"
            for g in games:
                if not result.get(g):
                    result[g] = shared

        logger.info("Multi-game text split: '%s' → %s", text[:60], result)
        return result

    # ── 管家级系统命令处理 ───────────────────────────────────────

    def _is_emulator_restart_command(self, text: str) -> bool:
        """检测模拟器重启/设备管理类命令。"""
        import re
        return bool(re.search(
            r'(?:重开|重启|打开|启动|开启|再开|新开|另开|多开)'
            r'(?:一下|一个|个)?(?:模拟器|设备|手机|这个)?',
            text,
        )) and not any(kw in text for kw in ['取消', '停止', '暂停'])

    def _is_direct_to_concierge(self, text: str) -> bool:
        """用户明确对管家说话，而非回应 agent 的追问。"""
        return any(kw in text for kw in ['管家', '你让管家', '让管家', '帮我开', '帮我重开'])

    def _handle_emulator_restart(self, text: str) -> str | None:
        """处理模拟器重启命令。

        场景：游戏 agent 报告模拟器连接断开 → 用户说"重开"
        → 管家接管，停止残废 agent，重启模拟器。
        """
        # ── 确定要重启的目标设备 ──
        target_serial: str | None = None

        # 1. 从消息中提取可能的设备标识
        import re
        # Check for explicit serial/port in message
        port_match = re.search(r'(?:127\.0\.0\.1:)?(\d{4,6})', text)
        if port_match:
            target_serial = f"127.0.0.1:{port_match.group(1)}"

        # 2. 从运行中 agent 的 slot 获取设备串口（优先离线/出问题的）
        if not target_serial:
            for s in self._slots:
                handle = s.current_task
                if handle and _agent_running(handle):
                    target_serial = s.device_serial
                    break

        # 3. Phase 1 fallback: session 级设备
        if not target_serial:
            target_serial = self.session.agent_device_serial

        # 4. 尝试 slot label 匹配
        if not target_serial:
            for s in self._slots:
                if s.match_label(text):
                    target_serial = s.device_serial
                    break

        # ── 停止所有运行中的 agent ──
        stopped = 0
        for s in self._slots:
            handle = s.current_task
            if handle and _agent_running(handle):
                agent = _agent_from(handle)
                if agent:
                    agent.state._pending_cancel = True
                    agent.state.inject_message(
                        "管家接管：正在重启模拟器。当前任务已暂停。")
                    agent.state.running = False
                    stopped += 1

        # Phase 1 fallback: session-level agent
        if stopped == 0:
            agent = self.session.current_agent
            if agent is not None and agent.state.running:
                agent.state._pending_cancel = True
                agent.state.inject_message(
                    "管家接管：正在重启模拟器。当前任务已暂停。")
                agent.state.running = False
                stopped += 1

        logger.info("Concierge handling emulator restart: stopped %d agents, target=%s",
                   stopped, target_serial)

        # ── 执行重启 ──
        if target_serial and target_serial != "emulator-5554":
            from src.concierge.tools import _restart_emulator
            result = _restart_emulator(self, target_serial)
        else:
            # 无明确目标设备 → 尝试启动模拟器（离线恢复场景）
            from src.concierge.tools import _start_emulator
            result = _start_emulator(self)

        # ── 通知 gateway 清除 waiting_for_reply 跟踪 ──
        if self._slot_activity_callback:
            for s in self._slots:
                if s.current_task and not _agent_running(s.current_task):
                    try:
                        self._slot_activity_callback(s.slot_id, "complete")
                    except Exception:
                        pass

        return result

    # ── 公共入口 ──

    def process_message(self, text: str) -> str | None:
        """处理用户消息（同步方法，同一用户串行化）。

        全部确定性逻辑，零 LLM 调用。
        """
        # Lazy-load snapshot on first message (after game_ctx is set up)
        if not self._snapshot_loaded:
            self._snapshot_loaded = True
            self.session.load_snapshot()

        result: str | None = None
        with self._process_lock:
            # 0. 管家级系统命令（优先于一切，无论是否有 agent 在运行）
            #    这些命令需要管家的工具能力，不能委派给游戏 agent。

            # 0a. 模拟器重启/设备管理
            if self._is_emulator_restart_command(text):
                return self._handle_emulator_restart(text)

            # 0b. 用户明确对管家说话 —— 清理掉 pending_clarification 和
            #     waiting_for_user 状态，防止后续消息继续注入给旧 agent
            if self._is_direct_to_concierge(text):
                self.session.pending_clarification = None
                self.session.pending_task = None
                logger.info("Direct-to-concierge message cleared pending state: %s", text[:80])

            # 1. #N 引用命令
            from src.concierge.tools import _route_agent_command
            agent_reply = _route_agent_command(self, text)
            if agent_reply is not None:
                return agent_reply

            # 2. Batch: "两个号都清体力"
            from src.concierge.tools import _detect_batch, _delegate_batch
            batch = _detect_batch(self, text)
            if batch is not None:
                batch_slots, batch_task = batch
                logger.info("Batch detected: %d slots for '%s'", len(batch_slots), batch_task[:60])
                return _delegate_batch(self, batch_task)

            # 3. Multi-game: "完成方舟和1999日常任务"
            multi_game = self._split_multi_game_task(text)
            if multi_game is not None:
                return multi_game

            # 4. Fast path: CHAT / STATUS / MANAGE / REPLY
            pending_before = self.session.pending_task
            fast_reply, consumed = try_fast_path(text, self)

            # Clarification consumed (e.g. user selected a slot or replied to
            # concierge LLM's ask_user).  _handle_clarification_response cleared
            # pending_task, so use the copy we saved.
            if consumed:
                task = pending_before or text
                # Check if this was a concierge_confirm — if so, re-run
                # the concierge LLM with the user's reply
                clarification = getattr(self.session, '_last_clarification_type', "") or ""
                if clarification.startswith("concierge_confirm"):
                    from src.concierge.prompts import build_games_info, build_slots_info
                    context = (
                        f"管家之前向用户提问：「{clarification.replace('concierge_confirm|', '')}」\n"
                        f"用户回复：「{text}」\n"
                        f"用户确认后重新执行原始意图。"
                    )
                    return self._run_concierge_llm(text, context=context)
                # Otherwise: standard slot-selection or generic — direct delegate
                game = self.game_ctx.active_game if self.game_ctx else ""
                return CONCIERGE_TOOL_DISPATCH["delegate_to_agent"](
                    self, task=task, game=game, task_type="custom",
                )

            if fast_reply is not None:
                return fast_reply

            # 4.5. Save/record guide: "/save 基建收菜..." / "存操作 ..."
            # These are create_guide tasks — delegate directly, no LLM needed.
            # The game agent detects the prefix in _setup_task_context and
            # enters create_guide mode (text-only, no screen interaction).
            create_guide_body = _match_create_guide_prefix(text)
            if create_guide_body is not None:
                from config.settings import config as _cfg2
                game = self.game_ctx.active_game if self.game_ctx else _cfg2.state.game
                # Detect game from body if it specifies one
                from src.agent.router import detect_game
                detected = detect_game(create_guide_body)
                if detected != game and detected != _cfg2.state.game:
                    game = detected
                return CONCIERGE_TOOL_DISPATCH["delegate_to_agent"](
                    self, task=text, game=game, task_type="create_guide",
                )

            # 4.6. Start new emulator for task: "再开一个模拟器完成1999日常"
            _emu_task = re.search(
                r'(?:再开|另开|新开|重开|多开|启动新)\S{0,3}(?:模拟器|设备|实例|个)\s*(.+)',
                text,
            )
            if _emu_task:
                task = _emu_task.group(1).strip()
                if task and self.emu_manager is not None:
                    from src.concierge.tools import _start_emulator as _emu_start_fn
                    from src.concierge.tools import _delegate_to_agent
                    start_reply = _emu_start_fn(self)
                    if "✅" in start_reply:
                        # Route task to the NEWLY created device
                        new_slot = self._slots[-1] if self._slots else None
                        game = self.game_ctx.active_game if self.game_ctx else ""
                        return start_reply + "\n" + _delegate_to_agent(
                            self, task=task, game=game, task_type="custom", _slot=new_slot,
                        )
                    return start_reply

            # 5. Everything else → concierge LLM
            # Check for all_busy first (all slots occupied with queued tasks)
            if self._slots and len(self._slots) > 1:
                free = [s for s in self._slots if s.is_free]
                if not free and self._slots:
                    from src.concierge.status_panel import format_slot_question
                    self.session.pending_clarification = "select_slot|all_busy"
                    self.session.pending_task = text
                    return format_slot_question(self._slots)

            # P1 fix: skip concierge LLM for simple single-game tasks.
            # "清体力", "刷GT-6", "完成日常" all go directly to agent.
            if self._is_simple_single_task(text):
                from config.settings import config
                from src.agent.router import detect_game
                game = detect_game(text)
                if not game or game == config.state.game:
                    game = self.game_ctx.active_game if self.game_ctx else config.state.game
                logger.info("Simple single-task detected — skipping concierge LLM: %s (game=%s)", text[:80], game)
                return CONCIERGE_TOOL_DISPATCH["delegate_to_agent"](
                    self, task=text, game=game, task_type="custom",
                )

            # Single-turn LLM decides: delegate, ask_user, or chat.
            # On LLM failure, falls back to direct delegate.
            result = self._run_concierge_llm(text)
            self.session.save_snapshot()
            return result

    def _on_exit_save(self) -> None:
        """atexit handler — saves session snapshot on process exit."""
        try:
            self.session.save_snapshot()
        except Exception as e:
            logger.warning("Failed to save session on exit: %s", e)

    def _is_simple_single_task(self, text: str) -> bool:
        """Deterministic check: is this a simple single-game task?

        When True, we can skip the concierge LLM entirely and delegate directly.
        This eliminates 100% of concierge LLM calls for simple task messages.
        """
        # Exclude: questions / complaints / multi-game / schedule / fuzzy
        _question_markers = ["?", "？", "为什么", "怎么", "干嘛", "干啥", "什么意思",
                             "好不好", "行不行", "能吗", "可以吗"]
        if any(kw in text for kw in _question_markers):
            return False
        # Exclude: complaints / feedback ("搞错了", "不对", etc.)
        _complaint_markers = ["搞错", "弄错", "不对", "不是", "怎么会", "错了", "咋回事"]
        if any(kw in text for kw in _complaint_markers):
            return False
        # Exclude: emulator management
        if self._is_emulator_restart_command(text):
            return False
        # Exclude: multi-game (already handled in step 3)
        if len(self._detect_multi_game(text)) > 1:
            return False
        # Exclude: schedule management (handled by gateway)
        _sched_kw = ["定时", "每天", "每日", "每小时", "每周", "取消定时", "暂停定时"]
        if any(kw in text for kw in _sched_kw):
            return False
        # Include: contains actionable task verbs (from all registered games)
        try:
            from src.games.registry import get_game_registry
            _all_verbs: list[str] = []
            for plugin in get_game_registry()._plugins.values():
                _all_verbs.extend(plugin.get_task_verbs())
            # Also include common cross-game verbs
            _all_verbs.extend(["完成", "启动", "截图", "升级", "日常", "进度", "farm"])
        except Exception:
            # Fallback if registry not available
            _all_verbs = ["刷", "打", "基建", "收菜", "换班", "清体力"]
        if any(kw in text for kw in _all_verbs):
            return True
        return False

    # ── Auto-dequeue ──

    def _try_auto_dequeue(self) -> None:
        """检查所有 slot 是否有排队任务需要启动。线程安全。"""
        if not self._slots:
            return
        from src.concierge.task_queue import dequeue_next
        import asyncio as _asyncio

        with self._process_lock:
            for s in self._slots:
                handle = s.current_task
                if handle is not None and not _agent_running(handle):
                    # Mark completed in history before clearing slot
                    if hasattr(handle, 'mark_completed'):
                        if not handle.completed_at:
                            handle.mark_completed(handle.outcome or "cancelled")
                        from src.concierge.agent_pool import add_to_history
                        add_to_history(handle)
                    s.current_task = None

                    # Phase 2: check slot availability before dequeueing
                    if (self.sched_engine is not None
                            and self.sched_engine.is_slot_busy(s.slot_id)):
                        continue  # Another task already acquired this slot

                    next_task = dequeue_next(s)
                    if next_task:
                        logger.info("Auto-dequeuing next task for slot %s: %s",
                                   s.label, next_task.task_description[:80])
                        self.session.agent_device_serial = s.device_serial
                        result = CONCIERGE_TOOL_DISPATCH["delegate_to_agent"](
                            self,
                            task=next_task.task_description,
                            game=next_task.game,
                            task_type=next_task.task_type,
                        )
                        self.session.conversation_history.append(
                            {"role": "user",
                             "content": f"[系统通知 — 自动启动排队任务] {result[:200]}"}
                        )
                        if self.bot and self._loop:
                            game_name = s.game_label
                            _asyncio.run_coroutine_threadsafe(
                                self.bot.send_message(
                                    self.user_id,
                                    f"[{game_name}] 自动启动排队任务: {next_task.task_description[:60]}",
                                ),
                                self._loop,
                            )
                # Continue to next slot — don't break (Phase 2: independent slots)

    # ── Observation learning commands ──────────────────────────────

    def process_observation_command(self, command: str, body: str) -> str | None:
        """Handle /record, /done, /stop commands for observation learning.

        Uses the same device reservation pattern as delegate_to_agent:
        reserve_device() on start, release_device() on stop/cancel/error.
        Sets session._device_owned so _on_agent_notify knows the lock is ours.
        """
        if command == "start":
            # ── Start observation ──
            if self._active_recorder is not None:
                return "已在观察中，请先 /done 完成或 /stop 取消当前观察。"

            # Detect game from message body (e.g. "/record 1999日常" → reverse1999),
            # then active_game from prior game switch.
            detected = self.game_ctx.detect_game_in_text(body)
            game = detected or self.game_ctx.active_game

            from config.settings import config as _cfg3
            # If nothing matched and the active_game is still the configured
            # default game, the user is likely recording a NEW game.
            # Take the first word as game name, rest as task.
            if not detected and game == _cfg3.state.game:
                guessed, rest = _extract_game_and_task(body)
                if guessed:
                    logger.info("Auto-detected new game from body: '%s' -> game=%s task=%s",
                               body, guessed, rest)
                    game = guessed
                    task_name = rest if rest else guessed
                    self.game_ctx.active_game = guessed
                else:
                    task_name = body.strip() or "未命名任务"
            else:
                task_name = body.strip() or "未命名任务"

            device = self.session.agent_device_serial

            # Check and reserve device (same pattern as delegate_to_agent)
            if self.sched_engine is not None:
                if self.sched_engine.is_device_busy(device):
                    return (
                        f"设备 {device} 正忙，无法开始观察。\n"
                        "请等当前任务完成后再试。"
                    )
                if not self.sched_engine.reserve_device(device):
                    return f"无法锁定设备 {device}，请稍后再试。"
                self.session._device_owned = True

            # Build a notify callback that sends messages to the user
            # during recording (timeout warnings, interrupted, etc.).
            import asyncio as _asyncio
            bot = self.bot
            user_id = self.user_id
            loop = self._loop

            def _notify_fn(msg: str) -> None:
                if bot and loop:
                    try:
                        _asyncio.run_coroutine_threadsafe(
                            bot.send_message(user_id, msg), loop,
                        )
                    except Exception:
                        pass

            try:
                from src.agent.observation_recorder import ObservationRecorder
                recorder = ObservationRecorder(
                    device_serial=device,
                    game=game,
                    task_name=task_name,
                    notify_fn=_notify_fn,
                )
                recorder.start()
                self._active_recorder = recorder

                from src.games.registry import get_game_registry
                game_name = get_game_registry().get_game_name(game) or game

                logger.info("Observation started: user=%s game=%s task=%s dev=%s",
                           self.user_id[:20], game, task_name, device)
                return (
                    f"👀 开始观察 **{game_name}**，请操作游戏。\n"
                    f"鼠标点击位置会被记录。\n"
                    f"完成后发送 /done {task_name} 生成指引。"
                )
            except Exception as e:
                logger.error("Failed to start observation: %s", e)
                self._release_observation_lock()
                return f"开始观察失败: {e}"

        elif command == "stop":
            # ── Stop and extract ──
            recorder = self._active_recorder
            if recorder is None:
                return "当前没有正在进行的观察。发送 /record 任务名 开始观察。"

            try:
                manifest_path = recorder.stop()
            except Exception as e:
                logger.error("Failed to stop observation: %s", e)
                manifest_path = ""
            finally:
                self._active_recorder = None
                self._release_observation_lock()

            if not manifest_path:
                return "观察记录保存失败，请重试。"

            game = recorder.game
            task_name = recorder.task_name

            self._spawn_extraction(
                manifest_path=manifest_path,
                game=game,
                task_name=task_name,
            )

            m = recorder.manifest
            return (
                f"📝 正在分析观察记录…\n"
                f"（{m.frame_count if m else 0} 帧，"
                f"{m.significant_count if m else 0} 个关键变化）\n"
                f"完成后会将生成的指引发送给你。"
            )

        elif command == "cancel":
            # ── Cancel observation ──
            recorder = self._active_recorder
            if recorder is None:
                return "当前没有正在进行的观察。"

            try:
                recorder.cancel()
            except Exception as e:
                logger.error("Failed to cancel observation: %s", e)
            finally:
                self._active_recorder = None
                self._release_observation_lock()

            logger.info("Observation cancelled: user=%s", self.user_id[:20])
            return "观察已取消。记录数据已删除。"

        return None

    def _release_observation_lock(self) -> None:
        """Release device lock held by an observation session."""
        if self.sched_engine is not None and self.session._device_owned:
            try:
                self.sched_engine.release_device(self.session.agent_device_serial)
            except Exception:
                pass
            self.session._device_owned = False

    def _spawn_extraction(
        self, manifest_path: str, game: str, task_name: str,
    ) -> None:
        """Spawn a background thread for guide extraction from observation.

        When extraction completes, sends the result to the user via WeChat.
        """
        import threading
        bot = self.bot
        user_id = self.user_id
        loop = self._loop

        def _run() -> None:
            import asyncio as _asyncio
            result_msg: str = ""

            def _on_done(msg: str) -> None:
                nonlocal result_msg
                result_msg = msg

            try:
                from src.agent.observation_extractor import ObservationExtractor
                extractor = ObservationExtractor()
                skill_name, skill_path = extractor.extract(
                    manifest_path=manifest_path,
                    game=game,
                    task_name=task_name,
                    on_done=_on_done,
                )

                if not skill_name and not result_msg:
                    result_msg = "抱歉，分析失败。观察记录已保留，可以稍后重试或手动查看。"

            except Exception as e:
                logger.error("Extraction thread crashed: %s", e)
                result_msg = f"分析出错: {e}"

            # Send result to user
            if result_msg and bot and loop:
                try:
                    _asyncio.run_coroutine_threadsafe(
                        bot.send_message(user_id, result_msg),
                        loop,
                    )
                except Exception as e:
                    logger.error("Failed to send extraction result: %s", e)

        t = threading.Thread(target=_run, daemon=True, name="obs-extract")
        t.start()

    # ── AgentHandle lookup (Phase 2) ────────────────────────────────

    def _lookup_handle_by_agent_id(self, agent_id: str) -> Any | None:
        """Find an AgentHandle by its #N agent_id across all slots.

        Searches active slots first, then completed history.
        Returns None if no match.
        """
        if not agent_id or not self._slots:
            return None
        for s in self._slots:
            handle = s.current_task
            if handle is not None and hasattr(handle, 'agent_id') and handle.agent_id == agent_id:
                return handle
        # Search completed history
        from src.concierge.agent_pool import get_completed_history
        for h in get_completed_history(limit=20):
            if getattr(h, 'agent_id', None) == agent_id:
                return h
        return None

    def _resolve_game_display(self, handle: Any) -> str:
        """Resolve a human-readable game name for notification labels.

        Priority: handle.game → handle.slot.game → agent.state.game → active_game.
        Returns Chinese name via GameRegistry, or the raw game ID as fallback.
        """
        game_id = ""
        if handle is not None:
            game_id = getattr(handle, 'game', '') or ''
            if not game_id and handle.slot:
                game_id = handle.slot.game or ''
            if not game_id:
                agent = _agent_from(handle)
                if agent is not None:
                    game_id = agent.state.game or ''
        if not game_id and self.game_ctx:
            game_id = self.game_ctx.active_game or ''
        if not game_id:
            return "agent"
        try:
            from src.games.registry import get_game_registry
            return get_game_registry().get_game_name(game_id) or game_id
        except Exception:
            return game_id

    # ── 来自游戏 agent 的回调（daemon 线程） ──

    def _on_agent_notify(self, notif: Any) -> None:
        """接收来自 TerraAgent 的结构化回调通知（daemon 线程调用）。

        接收一个 AgentNotification 对象（或向后兼容的旧式 tuple）。
        Progress is throttled via NotificationBuffer to prevent WeChat spam.
        Critical notifications (complete/error/ask_user) pass through immediately.
        """
        import asyncio

        # ── Backward compat: accept old-style (msg, notify_type, image_b64, agent_id) ──
        from src.agent.state import AgentNotification
        if isinstance(notif, AgentNotification):
            msg = notif.message
            notify_type = notif.type
            image_b64 = notif.image_b64
            agent_id = notif.agent_id or None
        else:
            # Old-style tuple/kwargs — unwrap gracefully
            logger.warning("_on_agent_notify received legacy notification format — upgrade callers")
            msg = notif[0] if isinstance(notif, (tuple, list)) else notif
            notify_type = notif[1] if isinstance(notif, (tuple, list)) and len(notif) > 1 else "progress"
            image_b64 = notif[2] if isinstance(notif, (tuple, list)) and len(notif) > 2 else None
            agent_id = notif[3] if isinstance(notif, (tuple, list)) and len(notif) > 3 else None

        if self.bot is None or self._loop is None:
            return

        # Resolve handle from agent_id for precise slot targeting
        handle: Any = None
        if agent_id:
            handle = self._lookup_handle_by_agent_id(agent_id)

        # Resolve game display name for notification labels
        game_display = self._resolve_game_display(handle)

        # Progress: throttle via buffer, label with game name so user knows
        # which game the progress is about in multi-agent scenarios.
        if notify_type == "progress":
            agent_label = game_display or (handle.slot.label if handle and handle.slot else (
                self.game_ctx.active_game if self.game_ctx else "agent"))
            if not self._notif_buf.should_push(agent_label, notify_type, msg):
                return  # Hold, don't push yet
            # Flush any buffered progress before sending fresh one
            for _label, _msg in self._notif_buf.flush():
                asyncio.run_coroutine_threadsafe(
                    self.bot.send_message(self.user_id, f"[{_label}] {_msg}"),
                    self._loop,
                )
            label_prefix = agent_label if agent_label else self.session.agent_device_serial
            asyncio.run_coroutine_threadsafe(
                self.bot.send_message(
                    self.user_id, f"[{label_prefix}] {msg}",
                ),
                self._loop,
            )
            return

        # ── Drain buffered progress BEFORE critical notifications ──
        # Without this, buffered progress is silently discarded when a
        # critical notification (complete/error/ask_user) arrives.
        for _label, _msg in self._notif_buf.drain_all():
            asyncio.run_coroutine_threadsafe(
                self.bot.send_message(self.user_id, f"[{_label}] {_msg}"),
                self._loop,
            )

        # ── Critical notifications: push immediately ──

        if notify_type == "complete":
            from config.settings import config as _cfg4
            completed_iters = 0
            resolved_game = _cfg4.state.game
            if handle is not None and hasattr(handle, 'mark_completed') and not handle.completed_at:
                agent = _agent_from(handle)
                if agent is not None and not agent.state.running:
                    handle.mark_completed("success")
                    completed_iters = agent.state.iteration_count
                    resolved_game = agent.state.game or resolved_game
                # Update slot.game so the device pool knows what game is active
                if handle.slot and resolved_game:
                    handle.slot.game = resolved_game
                # Release slot semaphore
                if handle.slot and self.sched_engine:
                    self.sched_engine.release_slot(handle.slot.slot_id)
            else:
                # Phase 1 fallback: iterate slots
                if self._slots:
                    for s in self._slots:
                        h = s.current_task
                        if h is not None and hasattr(h, 'mark_completed'):
                            a = _agent_from(h)
                            if a is not None and not a.state.running and not h.completed_at:
                                h.mark_completed("success")
                                completed_iters = a.state.iteration_count
                                resolved_game = a.state.game or resolved_game
                                # Phase 2: release slot semaphore in fallback
                                if h.slot and self.sched_engine:
                                    self.sched_engine.release_slot(h.slot.slot_id)
                if self.sched_engine is not None and self.session._device_owned:
                    self.sched_engine.release_device(self.session.agent_device_serial)
                    self.session._device_owned = False
            asyncio.run_coroutine_threadsafe(
                self.bot.send_message(self.user_id, f"[{game_display} ✓] {msg}"),
                self._loop,
            )
            self._try_auto_dequeue()
        elif notify_type == "error":
            from config.settings import config as _cfg5
            error_iters = 0
            resolved_game = _cfg5.state.game
            if handle is not None and hasattr(handle, 'mark_completed') and not handle.completed_at:
                agent = _agent_from(handle)
                if agent is not None and not agent.state.running:
                    handle.mark_completed("error")
                    error_iters = agent.state.iteration_count
                    resolved_game = agent.state.game or resolved_game
                if handle.slot and resolved_game:
                    handle.slot.game = resolved_game
                if handle.slot and self.sched_engine:
                    self.sched_engine.release_slot(handle.slot.slot_id)
            else:
                # Phase 1 fallback
                if self._slots:
                    for s in self._slots:
                        h = s.current_task
                        if h is not None and hasattr(h, 'mark_completed'):
                            a = _agent_from(h)
                            if a is not None and not a.state.running and not h.completed_at:
                                h.mark_completed("error")
                                error_iters = a.state.iteration_count
                                resolved_game = a.state.game or resolved_game
                                if h.slot and self.sched_engine:
                                    self.sched_engine.release_slot(h.slot.slot_id)
                if self.sched_engine is not None and self.session._device_owned:
                    self.sched_engine.release_device(self.session.agent_device_serial)
                    self.session._device_owned = False
            asyncio.run_coroutine_threadsafe(
                self.bot.send_message(self.user_id, f"[{game_display} ✗] {msg}"),
                self._loop,
            )
            self._try_auto_dequeue()
        elif notify_type == "ask_user":
            ask_label = f"[{game_display}] " if game_display else ""
            # Send full question as text first (no artificial truncation),
            # then send screenshot as image with short label
            asyncio.run_coroutine_threadsafe(
                self.bot.send_message(self.user_id, f"{ask_label}🤔 {msg}"), self._loop
            )
            if image_b64:
                asyncio.run_coroutine_threadsafe(
                    self.bot.send_image(self.user_id, image_b64, msg[:500]),
                    self._loop,
                )

        elif notify_type == "screenshot":
            # Fire-and-forget screenshot: same image path as ask_user,
            # but with a notification label (no 🤔) and WITHOUT adding
            # the slot to waiting_slots — the agent is NOT waiting for reply.
            label = f"[{game_display}] " if game_display else ""
            asyncio.run_coroutine_threadsafe(
                self.bot.send_message(self.user_id, f"{label}📸 {msg}"), self._loop
            )
            if image_b64:
                asyncio.run_coroutine_threadsafe(
                    self.bot.send_image(self.user_id, image_b64, msg[:500]),
                    self._loop,
                )

        # ── Slot activity callback ──
        # Notify the gateway so it can update last_active_slot / waiting_slot
        # tracking for intelligent multi-agent message routing.
        # "screenshot" type is EXCLUDED — the agent is not waiting for a reply.
        if self._slot_activity_callback and notify_type in ("ask_user", "complete", "error"):
            slot_id = self._resolve_slot_id(agent_id)
            if slot_id:
                try:
                    self._slot_activity_callback(slot_id, notify_type)
                except Exception:
                    pass  # Non-critical — don't break notification path

    # ── 设备离线事件处理 ──

    def on_device_offline(self, serial: str) -> None:
        """设备离线：尝试自动重启模拟器，同时取消任务并通知用户。"""
        import asyncio as _asyncio

        affected_slots = [s for s in self._slots if s.device_serial == serial]
        if not affected_slots:
            return

        # ── Auto-restart in background thread (never block health monitor) ──
        if self.emu_manager is not None:
            logger.info("Device %s offline — auto-restarting emulator", serial)
            import threading
            threading.Thread(
                target=self.emu_manager.restart_emulator,
                args=(serial,), daemon=True,
            ).start()

        for s in affected_slots:
            logger.warning("Device %s offline — slot %s marked unavailable",
                         serial, s.label)
            agent = _agent_from(s.current_task)
            if agent is not None and agent.state.running:
                agent.state._pending_cancel = True
                agent.state.inject_message("设备离线，任务已暂停。")
                agent.state.running = False

            if self.bot and self._loop:
                _asyncio.run_coroutine_threadsafe(
                    self.bot.send_message(
                        self.user_id,
                        f"⚠️ {s.label}（{serial}）设备离线，正在自动重启…",
                    ),
                    self._loop,
                )

    def on_device_restarted(self, new_serial: str) -> None:
        """设备重启后串口变更（MuMu 12 网络 ADB）：更新 slot 绑定。"""
        for s in self._slots:
            if s.device_serial != new_serial:
                # If this slot's old serial is no longer online but there's
                # a newly restarted device, remap the slot
                try:
                    from src.device.emulator import emulator_manager
                    old_serial = s.device_serial  # save before mutation
                    old_online = emulator_manager.is_online(old_serial)
                    if not old_online:
                        logger.info("Restart: slot %s remapped %s → %s",
                                   s.label, old_serial, new_serial)
                        s.device_serial = new_serial
                        # Update schedule engine's slot→device mapping
                        if self.sched_engine:
                            self.sched_engine._slot_to_device[s.slot_id] = new_serial
                        if self.session.agent_device_serial == old_serial:
                            self.session.agent_device_serial = new_serial
                    else:
                        logger.warning(
                            "Restart: slot %s old serial %s still online — "
                            "skipping remap to %s (may be a different device)",
                            s.label, old_serial, new_serial)
                except Exception:
                    logger.exception(
                        "Restart: failed to remap slot %s from %s to %s",
                        s.label, old_serial, new_serial)
        self.on_device_online(new_serial)

    def on_device_online(self, serial: str) -> None:
        """设备恢复：标记对应 slot 可用，通知用户。"""
        import asyncio as _asyncio

        affected_slots = [s for s in self._slots if s.device_serial == serial]
        if not affected_slots:
            return

        for s in affected_slots:
            logger.info("Device %s online — slot %s available", serial, s.label)

        if self.bot and self._loop:
            names = ", ".join(s.label for s in affected_slots)
            _asyncio.run_coroutine_threadsafe(
                self.bot.send_message(self.user_id, f"✅ {names} 设备已恢复。"),
                self._loop,
            )
