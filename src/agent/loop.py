"""Core conversation loop — the heart of Terra Agent.

Pattern:
1. Build three-layer system prompt
2. LLM inference → tool calls
3. Execute tools sequentially (same ADB device)
4. Return results to LLM
5. Check iteration budget
6. Repeat until task complete or budget exhausted
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from config.prompts import build_system_prompt, build_system_prompt_cached
from config.settings import config
from src.agent.compression_service import CompressionService
from src.agent.loop_guard import LoopGuard
from src.agent.memory_hint_service import MemoryHintService
from src.agent.router import route_task
from src.agent.screen_injector import (
    ScreenInjector, ACTION_TOOLS, compress_screenshot, capture_screen_jpeg, is_action_tool,
)
from src.agent.scroll_tracker import ScrollTracker
from src.agent.idle_watcher import IdleWatcher
from src.agent.state import AgentState
from src.llm.client import acquire_client, release_client, extract_text, extract_thinking, extract_tool_calls
from src.memory.history import history_db
from src.tools.registry import registry, ToolOutput, set_current_game
from src.utils.log_tags import set_agent_tag, clear_agent_tag

logger = logging.getLogger(__name__)


# ── Low-text loading-screen detection keywords (module level) ──────
_LOADING_SCREEN_KW: frozenset[str] = frozenset({
    "加载", "loading", "请稍候", "wait", "please wait",
    "启动", "starting", "启动中", "资源", "resource",
    "splash", "更新", "update", "初始化", "initializing",
    "连接", "connecting",
    # Arknights-specific loading screens
    "infrastructure",  # RHODES ISLAND INFRASTRUCTURE COMPLEX
    "进驻",  # 设施进驻说明, loading tip text
})


# ── Dark-screen navigation tools (module level) ──────────────────────
_DARK_NAV_TOOLS: frozenset[str] = frozenset({"adb_tap", "adb_scroll", "adb_swipe", "adb_tap_position", "adb_back"})

# ── Wait-intent keywords: LLM thinking says "wait" but tool calls say "act" ──
# These are INTENT PHRASES (not single words) to avoid matching meta-discussion
# where the LLM analyzes WHY it was blocked (e.g. "the word 'wait' triggers it").
#
# DESIGN NOTE: English negation patterns ("should not tap", "do not click", etc.)
# are deliberately EXCLUDED.  The LLM frequently uses these when reasoning about
# what NOT to do ("I should not tap 下载指引, I should tap 暂时不用 instead")
# — catching them would discard the CORRECT action alongside the wrong one.
# The Chinese patterns are retained because their negation forms ("不要操作")
# almost always express genuine idle/wait intent rather than button discrimination.
_WAIT_INTENT_KW: frozenset[str] = frozenset({
    # Chinese: clear wait/no-action intent
    "应该等待", "需要等待", "先等一下", "等加载",
    "不要操作", "不要导航", "不要滑动",
    "不能操作", "不进行任何操作", "不去点",
    "等待加载", "等待进入", "等待画面", "等待过渡",
    "先不操作", "先别动", "暂时不",
    "等待自动", "让它自动", "别点", "不要点",
    # English: positive wait intent only (NOT negations — see DESIGN NOTE above)
    "should wait", "need to wait", "going to wait", "i will wait",
    "i'll wait", "i am going to wait", "i'm going to wait",
    "do nothing", "let it auto",
})

# ── Resource consumption guard constants (module level) ──────────────

# Tier 1 — Strong: explicit consumption-confirmation keywords.
_CONSUME_CONFIRM_KW: tuple[str, ...] = (
    "当前技能等级", "是否升级", "是否确认",
    "精英化", "专精", "模组",
)
# Tier 2 — Weak: resource names (appear in consumption AND display contexts).
_CONSUME_RESOURCE_KW: tuple[str, ...] = (
    "龙门币", "作战记录", "技能概要",
    "双芯片", "聚合剂", "至纯源石", "合成玉", "寻访凭证",
)
# Tier 3 — Safe screens that should NEVER trigger the guard.
# P1 fix: extended with main-screen / farm-navigation keywords.
# Logs showed "SAFETY BLOCK (screen): '合成玉'" triggered on the main
# screen — 合成玉 was displayed as a resource count, not a purchase target.
_CONSUME_SAFE_SCREEN_KW: tuple[str, ...] = (
    "控制中枢", "进驻总览", "会客室", "制造站", "贸易站",
    "发电站", "宿舍", "基建", "设施列表", "进驻信息",
    "信用交易所", "可露希尔推荐",
    # Main screen navigation — resource display, not consumption
    "终端", "干员", "档案", "首页", "编队", "公开招募",
    "采购中心", "日常任务", "周常任务", "主线任务",
    "特勤任务", "任务列表", "情报", "剿灭作战",
    # Arknights resource display zones
    "理智", "合成玉", "至纯源石", "龙门币", "寻访凭证",
)
# Exit / log-out dialogs — clicking here does NOT consume in-game resources.
_CONSUME_EXIT_KW: tuple[str, ...] = (
    "是否确认退出游戏", "退出游戏", "确认退出",
    "确认登出", "登出账号", "退出登录",
    "返回登录", "切换账号登录",
)
_CONSUME_TAP_TARGETS: tuple[str, ...] = (
    "升级", "确认", "精英化", "专精", "购买", "兑换", "寻访", "招募",
)

# ── Implicit ask_user detection keywords (module level) ───────────────
# Matched against LLM text output when it forgets to call the ask_user tool.
_IMPLICIT_ASK_KW: tuple[str, ...] = (
    "模拟器卡", "无响应", "等你回复", "检查一下模拟器",
    "模拟器冻结", "emulator", "frozen", "please check",
    "请登录", "重新登录", "登录已过期", "登录失败",
    "验证码", "密码", "手机号", "账号",
    "需要你手动", "我没法帮你", "没法帮你操作",
    "需要你自己", "请你自己", "手动输入",
    "手动登录", "手动操作", "需要人工",
    "无法自动", "我无法", "需要你帮忙",
    "告诉我", "回复我", "完成后告诉我",
)

# ── Chat message detection keywords (module level) ────────────────────
_TASK_SIGNALS: tuple[str, ...] = (
    "刷", "清", "打", "基建", "收菜", "制造", "贸易",
    "招募", "升级", "精二", "精英", "材料", "规划",
    "定时", "每天", "截图", "重启", "启动", "连接",
    "任务", "日常", "周常", "剿灭", "活动", "关卡",
    # P1 fix: resource queries are task-related, not chat.
    # "还剩多少理智" / "还有多少合成玉" — these need game state access.
    "理智", "合成玉", "龙门币", "源石", "寻访",
    "体力", "剩余", "代币", "商店",
)
_CHAT_PATTERNS: tuple[str, ...] = (
    "在吗", "在不在", "你好", "谢谢", "辛苦了",
    "怎么样", "如何", "怎么用", "你能", "你会",
    "帮我看", "看看", "还剩", "还有多少", "多少",
)


def _try_parse_output(text: str) -> dict[str, Any] | None:
    """Parse tool output JSON once, return None on failure.

    Multiple post-processing checks (skill_run guard, box scan trigger,
    search_memory trigger, any_action_succeeded, stuck-target) each
    independently parse the same output.text.  This helper lets callers
    parse once at the top of the post-processing block.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_box_signal(results: list[dict[str, Any]]) -> int:
    """Count operator markers (E0/E1/E2) in search_memory results.

    Returns the count of operator-level markers found — >= 3 triggers an
    intelligence re-run (BaseScheduler needs fresh box data to build a schedule).
    """
    count = 0
    for r in results:
        body = r.get("body", "")
        elite_markers = len(re.findall(r'[Ee][012]\s*(?:LV\d+|级)?\s*[:：]', body))
        paren_markers = len(re.findall(r'[（(]\s*(?:精英)?\s*[Ee]?\d\s*[）)]', body))
        count += max(elite_markers, paren_markers)
    return count


def _consume_screen_ocr_scan(ocr_texts: list[str] | None) -> bool:
    """Pre-compute whether the current screen is a resource-consumption screen.

    Called once per iteration before the tool loop so the expensive OCR text
    scanning (6+ any() calls over keyword lists) runs once instead of per-tool.
    """
    if not ocr_texts:
        return False
    concat = " ".join(ocr_texts)
    has_confirm = any(kw in concat for kw in _CONSUME_CONFIRM_KW)
    if not has_confirm:
        return False
    is_safe = any(kw in concat for kw in _CONSUME_SAFE_SCREEN_KW)
    is_exit = any(kw in concat for kw in _CONSUME_EXIT_KW)
    return not is_safe and not is_exit


class IterationBudget:
    """Thread-safe iteration counter to prevent infinite loops.

    Budget is dynamic — it adjusts based on the number of matched skills
    (more skills / sub-tasks → more iterations needed).
    """

    def __init__(self, max_iterations: int = 20) -> None:
        self.max_total = max_iterations
        self._used = 0
        # Per-skill budget: estimated average iterations per sub-skill
        self.PER_SKILL = 50
        self.MIN_TOTAL = 200
        self.MAX_TOTAL = 800

    def adjust_for_skills(self, skill_count: int) -> None:
        """Adjust budget based on number of matched skills/sub-tasks.

        Each skill averages ~50 iterations (navigate + execute + verify).
        Floor is 200, ceiling is 800.  Farming tasks (1-7, etc.) match
        few skills but need many iterations — they hit the minimum.
        """
        if skill_count > 0:
            dynamic = max(self.MIN_TOTAL, skill_count * self.PER_SKILL)
            dynamic = min(dynamic, self.MAX_TOTAL)
            if dynamic > self.max_total:
                logger.info("Budget adjusted: %d → %d (skills=%d)",
                           self.max_total, dynamic, skill_count)
                self.max_total = dynamic

    def consume(self) -> bool:
        if self._used >= self.max_total:
            return False
        self._used += 1
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.max_total - self._used)

    @property
    def used(self) -> int:
        return self._used


class TerraAgent:
    """Main agent that runs the conversation loop for game automation."""

    def __init__(self, device_serial: str, game: str = "arknights",
                 ask_fn: Callable[[str], str] | None = None,
                 ocr_engine: Any | None = None,
                 container: Any = None) -> None:
        from src.container import get_container
        self.container = container if container is not None else get_container()
        self.state = AgentState(game=game, device_serial=device_serial)
        self.client = acquire_client()
        self.budget = IterationBudget(config.agent.max_iterations)
        self.ask_fn = ask_fn
        # _terra_agent_ctx is bound in _async_run() on the daemon thread,
        # NOT here — __init__ runs on the Concierge thread and binding
        # here would point to the wrong thread when two agents run concurrently.
        # OCR: use injected engine or default module singleton
        if ocr_engine is not None:
            ocr_engine.preload()
        else:
            from src.vision.ocr import ocr_engine as _default_ocr
            _default_ocr.preload()

        # ── Phase 2: injected services ──
        # Each agent gets its OWN MemoryHintService — the shared singleton's
        # _last_rerank_key field causes cross-agent rerank skips: when agent A
        # stores its OCR key, agent B's next gather() may see >70% overlap and
        # skip LLM reranking, delivering lower-quality memories.
        self.hint_service = MemoryHintService(
            self.container.memory_db,
            self.container.client_pool,
            self.container.config,
        )
        # Each agent gets its OWN CompressionService — sharing a singleton
        # causes cross-contamination: agent B can receive agent A's compressed
        # conversation history when _pending is consumed by the wrong thread.
        self.compressor = CompressionService(self.container.client_pool)
        self.execution_logger = self.container.execution_logger
        self.reviewer = self.container.review_trigger

        # DB startup maintenance now runs automatically in each DB module's __init__
        # (memory_db.py, skill_db.py — each with its own _startup_done guard).

        # ── Heartbeat tracking (multi-agent responsiveness) ──
        self._last_heartbeat_sent: float = 0.0  # monotonic timestamp of last heartbeat
        self._last_activity: float = 0.0        # monotonic timestamp of last meaningful action
        self._phase_enter: float = 0.0          # when current phase started (for concurrency diag)

    # ── Tracker reset (shared by run / respond / run_async) ──────────

    def _reset_trackers(self) -> None:
        """Reset all per-task state."""
        from src.agent.guard_context import GuardContext
        self.guard = GuardContext()
        self.state.guard = self.guard
        self.loop_guard = LoopGuard()
        self.screen_injector = ScreenInjector(state=self.state)
        self.scroll_tracker = ScrollTracker()
        self.idle_watcher = IdleWatcher(state=self.state, agent=self)
        self._wait_cycles = 0
        self._pending_back_pre_hash = None
        self._stuck_screen_hash = ""
        self._stuck_target = ""
        self._stuck_count = 0
        self._last_budget_warning_iter = 0  # Dedup subtask budget warnings
        self._last_output_tokens: int = 0    # Per-call output tokens (for guard)
        self._max_tokens_no_tool_streak: int = 0  # Consecutive max_tokens exhaustion
        self._no_tool_static_streak: int = 0  # Consecutive no-tool + static screen
        self._wait_intent_conflict_streak = 0  # Consecutive wait-intent conflicts
        self._dark_screen_since: float = 0.0  # Monotonic timestamp when dark-screen blocking began (0=none)
        self._dark_screen_hash: str = ""      # Screen hash when dark-screen blocking began
        # ── Repeated-thinking detection ──
        self._last_think_bigrams: set[str] = set()   # Chinese bigrams from previous LLM text
        self._repeat_think_streak: int = 0            # Consecutive near-identical thinking rounds
        self._subtask_iter: dict[str, int] = {}       # Per-subtask iteration counter
        self._subtask_iter_warned: set[str] = set()   # Subtasks that already got the 40-iter warning
        self._last_memory_hint = ""
        self._last_user_hint = ""
        self._last_review_skill_hash: str | None = None  # Dedup review injections
        self._cached_state_summary = ""  # Per-task cache — DB query, stable within task
        self.hint_service.reset_for_task()
        self.compressor.reset_for_task()
        # Injected memory dedup (fresh set per task)
        self._injected_ids_this_task: set[int] = set()
        # Build per-task ToolContext for explicit context passing (Phase 3)
        from src.tools.context import ToolContext
        self.tool_ctx = ToolContext(
            game=self.state.game,
            device_serial=self.state.device_serial,
            ask_fn=self.ask_fn,
            notify_fn=self.state.on_notify,
            agent_state=self.state,
            agent_ref=self,
        )
        from src.tools.adb_control import clear_skill_coords
        clear_skill_coords()

    def _inject_memory_hints(self) -> None:
        """Gather and inject relevant memories based on current screen OCR.

        Called after screen_injector operations.  Deduplicates via
        self._last_memory_hint to avoid repeating the same hint.

        Frequency-limited: after the first 5 iterations, subsequent calls
        only run every 3 iterations to reduce DB query + LLM rerank cost.
        Screen context rarely changes enough in consecutive turns to justify
        a full dHash→FTS5→rerank pipeline on every iteration.
        Reduced to every 5 iterations after warmup (was every 3) to save
        DB query + LLM reranking cost (~500 tokens per call).
        """
        if self.state.iteration_count > 5 and self.state.iteration_count % 15 != 0:
            return
        try:
            ocr_texts = list(self.state.last_ocr_texts) if self.state.last_ocr_texts else []
            memory_hints = self._gather_memory_hints(ocr_texts)
            if memory_hints and memory_hints != self._last_memory_hint:
                self.state.add_message("user", memory_hints)
                self._last_memory_hint = memory_hints
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "Memory hint gathering failed")

    def _inject_create_guide_instruction(self, body: str) -> None:
        """Inject instructions for create-guide mode — no game interaction.

        Still runs game detection so the user can switch games with /save.
        E.g. "/save reverse1999 抽卡流程" → detects reverse1999, switches.
        """
        # ── Detect game from body (same logic as normal commands) ──
        from src.agent.router import detect_game
        detected = detect_game(body)
        if detected != self.state.game:
            from src.tools.registry import set_current_game
            logger.info("Game switch via /save: %s → %s", self.state.game, detected)
            self.state.game = detected
            set_current_game(detected)
            self.tool_ctx.game = detected

        game_hint = self.state.game
        guide_msg = (
            "[系统指令 — 保存操作指引]\n"
            f"用户刚才用 /save 发起了一条指引保存请求。当前游戏: {game_hint}\n\n"
            "用户描述了一个游戏操作流程，你需要把它保存为指引。\n\n"
            "提取规则（只提取用户明确说到的，不能编造）：\n"
            "- name: 从描述中总结，英文/拼音，如 base-collect、farm-ce-5\n"
            "- description: 一句话描述这个指引做什么\n"
            "- steps: 用户描述的每个操作步骤，一行一步。用户说什么你写什么，\n"
            "  不要加用户没提到的步骤，也不要删用户提到的步骤。\n"
            "  自然语言即可（如 '点击基建'），create_guide 会自动格式化。\n"
            "- pitfalls: 用户明确说的注意事项。用户没提到就不填。\n"
            "- tags: 从步骤中提取关键词（如 基建、收菜、作战）。\n\n"
            f"执行：create_guide(name, description, steps, game='{game_hint}', pitfalls, tags)\n"
            "然后 task_complete()。回复「已保存指引 NAME」。\n\n"
            "⚠️ 不要截屏、不要 adb_tap、不要做任何游戏操作。"
        )
        self.state.add_message("user", guide_msg)
        self.state.task_type = "create_guide"
        self.state.task_description = body
        logger.info("Create guide flow activated (game=%s): %s", game_hint, body[:60])

    def _setup_task_context(self, user_message: str,
                            skip_intelligence: bool = False,
                            task_continuation: bool = False) -> dict[str, Any]:
        """Shared task bootstrap: route, create execution record, inject intelligence.

        Called by both run() and run_async() — eliminates ~30 duplicated lines.
        When task_continuation=True (used by respond() for mid-task user replies),
        routing, DB record creation, and continuation context are all skipped —
        the task's matching_skills and budget are already set from the original
        run() invocation.
        Returns the routing dict.
        """
        # ── Task continuation: reuse existing matching_skills + budget ──
        # respond() calls this to refresh skill text, but routing, DB record
        # creation, continuation context, and intelligence are already set from
        # the original run() invocation.  Skipping saves a DB write + FTS5 query
        # + intelligence LLM call.
        if task_continuation:
            return {
                "game": self.state.game,
                "task_type": self.state.task_type or "unknown",
                "matching_skills": self.state.matching_skills,
            }

        # ── Create guide: direct prefix match, before routing ──
        # Triggers: /save 基建收菜... | 存操作 基建收菜... | /s 基建收菜...
        create_guide_body = _match_create_guide_prefix(user_message)
        if create_guide_body is not None:
            self._inject_create_guide_instruction(create_guide_body)
            return {"game": self.state.game, "task_type": "create_guide", "matching_skills": []}

        # Route task to find matching skills
        routing = route_task(user_message, game=self.state.game)
        # Update game if detection produced a different result
        detected = routing.get("game", self.state.game)
        if detected != self.state.game:
            logger.info("Game switched: %s → %s", self.state.game, detected)
            self.state.game = detected
            set_current_game(detected)
            self.tool_ctx.game = detected  # Keep ToolContext in sync after game switch
        self.state.matching_skills = routing["matching_skills"]
        if routing["matching_skills"]:
            skill_names = [s["name"] for s in routing["matching_skills"]]
            logger.info("Matched skills: %s", skill_names)

        # Adjust iteration budget based on task complexity.
        # Simple per-skill budget — daily tasks use matched skill count.
        # 50 iter/skill is ample (6/25 run: 5 skills completed in 173 iters).
        self.budget.adjust_for_skills(max(len(routing["matching_skills"]), 1))

        # ── Runtime context injection ──
        # When an orchestrator skill is matched, inject day-of-week, game state,
        # and other runtime context so the LLM can make conditional decisions
        # (e.g. Saturday→annihilation, low sanity→skip farming).
        _has_orch = any(s.get("type") == "orchestrator" for s in self.state.matching_skills)
        if _has_orch:
            try:
                from src.agent.context_injector import gather_runtime_context, inject_context_message
                ctx = gather_runtime_context(
                    game=self.state.game,
                    device_serial=self.state.device_serial,
                )
                msg = inject_context_message(ctx)
                # Inject BEFORE the user message so it's background context
                self.state.add_message("user", msg)
                logger.info("Runtime context injected: day=%s, game=%s",
                           ctx.day_name_cn, ctx.game)
            except Exception:
                from src.utils.errors import safe_log
                safe_log(logger, "warning", "Runtime context injection failed (non-critical)")

        # ---- Learning Engine Phase 1: create task execution record ----
        self.state.task_execution_id = None
        self.state.injection_feedback_tracker = None
        try:
            from src.memory.memory_db import memory_db as _mdb
            task_id = _mdb.create_task_execution(
                game=self.state.game,
                task_type=routing["task_type"],
                task_description=user_message[:200],
            )
            self.state.task_execution_id = task_id
            from src.agent.injection_feedback import InjectionFeedbackTracker
            self.state.injection_feedback_tracker = InjectionFeedbackTracker(task_id)
            logger.debug("Task execution #%d created", task_id)
        except Exception as e:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", f"Failed to create task execution record: {e}")

        # ── Continuation context: "继续" / "接着" with no task description ──
        # When the user says "继续" after a task_complete, inject the previous
        # task's goal so the LLM doesn't start fresh on a random screen.
        _continuation = re.match(
            r'^(继续|接着做|接着|继续做|继续刚才|继续任务|接着任务)\s*$', user_message.strip())
        if _continuation:
            try:
                from src.memory.memory_db import memory_db
                recent = memory_db.get_recent_task_executions(
                    game=self.state.game, limit=3)
                for prev in recent:
                    desc = (prev.get("task_description") or "").strip()
                    if not desc or desc in ("(未指定)", "继续", "接着做", "继续任务"):
                        continue
                    if prev.get("success"):
                        self.state.add_message("user",
                            f"[系统提示 — 上一个已完成的任务] {desc}\n"
                            "请检查当前屏幕，判断是否需要继续刚才的任务。"
                            "如果上一个任务的目标尚未完全达成（如刷图只刷了一部分），"
                            "请继续执行而非开始其他任务。")
                    else:
                        self.state.add_message("user",
                            f"[系统提示 — 上一次未完成的任务] {desc}\n"
                            "该任务上次未能完成。如果当前屏幕允许，可以重试。")
                    break
            except Exception:
                from src.utils.errors import handle_error, DegradedError
                handle_error(logger, DegradedError("Failed to fetch recent task executions"),
                           "continuation_context")

        # ---- Learning Engine Phase 2: Intelligence tool recommendations ----
        if not skip_intelligence:
            try:
                from src.intelligence import register_default_tools
                from src.intelligence.base import IntelligenceContext, get_intelligence_registry
                register_default_tools(self.state.game)
                from src.knowledge import get_knowledge_base
                intel_ctx = IntelligenceContext(
                    game=self.state.game,
                    knowledge=get_knowledge_base(),
                    skills=self.state.matching_skills,
                    screen_dhash=self.state.last_injected_dhash,
                    ocr_texts=self.state.last_ocr_texts,
                )
                intel_results = get_intelligence_registry(self.state.game).query(intel_ctx, user_message)
                for r in intel_results:
                    self.state.add_message("user", f"[智能建议] {r.recommendation}")
                    logger.info("Intelligence injected: %s (confidence=%.2f)", r.source, r.confidence)
            except Exception as e:
                from src.utils.errors import safe_log
                safe_log(logger, "warning", f"Intelligence query skipped: {e}")

        return routing

    def _reinject_intelligence(self) -> None:
        """Re-run intelligence tools with fresh data (e.g. after box scan).

        Uses the cached task description to re-query intelligence tools
        (BaseScheduler, etc.) so the LLM gets updated results without
        requiring a full task reset.
        """
        try:
            from src.intelligence import register_default_tools
            from src.intelligence.base import IntelligenceContext, get_intelligence_registry
            from src.knowledge import get_knowledge_base
            register_default_tools(self.state.game)
            intel_ctx = IntelligenceContext(
                game=self.state.game,
                knowledge=get_knowledge_base(),
                skills=self.state.matching_skills,
                screen_dhash=self.state.last_injected_dhash,
                ocr_texts=self.state.last_ocr_texts,
            )
            task_desc = self.state.task_description or ""
            intel_results = get_intelligence_registry(self.state.game).query(intel_ctx, task_desc)
            if intel_results:
                self.state.add_message("user",
                    "[系统提示 — 数据已更新] 智能优化器已根据最新Box数据重新计算方案：")
                for r in intel_results:
                    self.state.add_message("user", f"[智能建议] {r.recommendation}")
                    logger.info("Intelligence re-injected: %s (confidence=%.2f)", r.source, r.confidence)
                # The [智能建议] above now contains the full room-by-room schedule.
                # Guide the LLM to use base_shift_maa for execution — never
                # attempt manual adb_tap room-by-room.
                self.state.add_message("user",
                    "[系统 — 排班执行指引]"
                    "执行基建换班时，使用 base_shift_maa 工具，不要手动 adb_tap。\n"
                    "1. 日常换班 → base_shift_maa(mode='default')\n"
                    "2. 执行优化器方案 → base_shift_maa(mode='custom', plan_index=N)\n"
                    "3. 队列轮换 → base_shift_maa(mode='rotation')\n"
                    "手动逐个房间换人需要 100+ 步操作，不可能完成。\n"
                    "base_shift_maa 由 MAA 引擎驱动，1-2 分钟完成全部换班。")
            else:
                logger.debug("Re-intelligence returned no results")
        except Exception as e:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", f"Intelligence re-run skipped: {e}")

    def _set_log_tag(self) -> None:
        """Set a per-agent log tag so multi-agent logs can be disambiguated."""
        from src.games.registry import get_game_registry
        game_name = get_game_registry().get_game_name(self.state.game)
        port = self.state.device_serial.split(":")[-1] if ":" in self.state.device_serial else self.state.device_serial
        set_agent_tag(f"{game_name}-{port}")

    # ── Public entry points ──────────────────────────────────────────

    def run(self, user_message: str) -> dict[str, Any]:
        """Execute one conversation turn from user message to completion."""
        from src.device.adb import bind_device_to_thread
        bind_device_to_thread(self.state.device_serial)
        self._set_log_tag()

        t0 = time.monotonic()
        self.state.reset()
        self.state.started_at = time.monotonic()  # wall-clock timeout
        self._reset_trackers()
        # Task instruction moved to system prompt (cached layer) instead of
        # being a conversation message (uncached, billed at full input rate).
        # The first user message will be the screen injection from inject_initial().
        self.state.user_task = user_message
        self.state.task_description = user_message
        self.state.conversation_history = []
        self.state.running = True

        set_current_game(self.state.game)  # Thread-local for tool handlers

        routing = self._setup_task_context(user_message)

        logger.info("Starting task: %s", user_message)

        # Capture screen dimensions for background review
        try:
            from src.device.adb import get_adb
            w, h = get_adb().get_screen_size()
            self.state.screen_w, self.state.screen_h = w, h
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "Failed to get screen size — using defaults")

        try:
            result = self._run_loop()
        finally:
            self.state.running = False
            self.screen_injector.shutdown()
            release_client(self.client)
            clear_agent_tag()
            duration = time.monotonic() - t0

        if not result.get("needs_input"):
            # ── Cost tracking (MiMo-V2.5 pricing: input ¥1/M, cache_read ¥0.02/M, output ¥2/M) ──
            # MiMo's usage.input_tokens excludes cache_read (unlike Anthropic which includes it).
            # Compute cost treating input_tokens as cache-miss, cache_read separately at 1/50 rate.
            _inp = self.state.total_input_tokens
            _out = self.state.total_output_tokens
            _cr = self.state.total_cache_read_tokens
            _cc = self.state.total_cache_create_tokens
            _cost_input = _inp * 1.0 / 1_000_000
            _cost_cache_read = _cr * 0.02 / 1_000_000
            _cost_cache_create = _cc * 1.0 / 1_000_000
            _cost_output = _out * 2.0 / 1_000_000
            _total_cost = _cost_input + _cost_cache_read + _cost_cache_create + _cost_output
            _total_tokens = _inp + _cr  # MiMo: input + cache_read = total tokens processed
            _cache_pct = (_cr / _total_tokens * 100) if _total_tokens > 0 else 0
            logger.info(
                "Cost: ¥%.2f (input=%.2f cache_read=%.2f cache_create=%.2f output=%.2f, "
                "%d iter, %.1fM total, cache_rate=%.0f%%)",
                _total_cost, _cost_input, _cost_cache_read, _cost_cache_create, _cost_output,
                self.state.iteration_count, _total_tokens / 1_000_000, _cache_pct,
            )
            self.execution_logger.log(
                user_message=user_message,
                result=result,
                duration=duration,
                state=self.state,
                matching_skills=self.state.matching_skills,
                task_type=routing.get("task_type", "unknown"),
                input_tokens=self.state.total_input_tokens,
                output_tokens=self.state.total_output_tokens,
            )
            self.reviewer.maybe_trigger(
                state=self.state,
                loop_guard=self.loop_guard,
                review_msgs=getattr(self, "_review_history", None),
                matching_skills=self.state.matching_skills,
                task_description=user_message,
            )

        return result

    def respond(self, answer: str) -> dict[str, Any]:
        """Continue a conversation after user guidance. Does NOT reset state."""
        if self.state._task_completed:
            logger.info("respond() blocked: task already completed")
            return {
                "success": True,
                "final_response": "任务已完成，无需继续。",
                "iterations": self.state.iteration_count,
                "task_completed": True,
            }
        self.state.add_message("user", f"[用户回复] {answer}")
        self.state.last_injected_hash = None
        self._reset_trackers()
        self._last_notify_time = 0.0  # Cooldown throttle for WeChat progress pushes
        self._recent_tool_targets: list[str] = []  # Battle-vs-launch context
        self.state.running = True
        set_current_game(self.state.game)  # Thread-local for tool handlers

        # Re-run task context setup to refresh skill_text (matching_skills
        # may have changed via user command).  skip_intelligence=True (results
        # rarely change mid-task) and task_continuation=True (routing, DB record,
        # budget are already set from the original run() invocation).
        routing = self._setup_task_context(self.state.task_description,
                                           skip_intelligence=True,
                                           task_continuation=True)

        try:
            result = self._run_loop()
        finally:
            self.state.running = False
            self.screen_injector.shutdown()
            release_client(self.client)

        # Log completion if task is actually done (not waiting for more input)
        if not result.get("needs_input"):
            # ── Cost tracking (MiMo-V2.5 pricing) ──
            _inp = self.state.total_input_tokens
            _out = self.state.total_output_tokens
            _cr = self.state.total_cache_read_tokens
            _cc = self.state.total_cache_create_tokens
            _cost_input = _inp * 1.0 / 1_000_000
            _cost_cache_read = _cr * 0.02 / 1_000_000
            _cost_cache_create = _cc * 1.0 / 1_000_000
            _cost_output = _out * 2.0 / 1_000_000
            _total_cost = _cost_input + _cost_cache_read + _cost_cache_create + _cost_output
            _total_tokens = _inp + _cr
            _cache_pct = (_cr / _total_tokens * 100) if _total_tokens > 0 else 0
            logger.info(
                "Cost: ¥%.2f (input=%.2f cache_read=%.2f cache_create=%.2f output=%.2f, "
                "%d iter, %.1fM total, cache_rate=%.0f%%)",
                _total_cost, _cost_input, _cost_cache_read, _cost_cache_create, _cost_output,
                self.state.iteration_count, _total_tokens / 1_000_000, _cache_pct,
            )
            # Duration is approximate since respond() doesn't track start time
            self.execution_logger.log(
                user_message=self.state.task_description,
                result=result,
                duration=0.0,
                state=self.state,
                matching_skills=self.state.matching_skills,
                task_type=routing.get("task_type", "unknown"),
            )
            self.reviewer.maybe_trigger(
                state=self.state,
                loop_guard=self.loop_guard,
                review_msgs=getattr(self, "_review_history", None),
                matching_skills=self.state.matching_skills,
                task_description=self.state.task_description,
            )

        return result

    def run_async(self, user_message: str, on_notify: Callable[..., None],
                  handle: Any = None) -> None:
        """Start task in background thread. Returns immediately.

        on_notify(text, image_b64=None) is called for: ask_user questions,
        errors, completion. When image_b64 is provided, the receiver should
        attempt to send the image alongside the text.

        handle is the AgentHandle for this task — stored on AgentState so
        _notify() can pass agent_id through the callback chain.
        """
        self.state.on_notify = on_notify
        self.state.reset()
        self.state.agent_handle = handle  # AFTER reset() — reset clears it
        # Task instruction moved to system prompt (cached layer)
        self.state.user_task = user_message
        self.state.task_description = user_message
        self.state.conversation_history = []
        self.state.running = True

        self._reset_trackers()

        self._setup_task_context(user_message)

        # Capture trace_id from calling thread for propagation to daemon thread
        from src.utils.trace import get_trace_id
        _captured_trace_id = get_trace_id()

        logger.info("Starting task (async): %s", user_message)

        t = threading.Thread(target=self._async_run, args=(_captured_trace_id,), daemon=True)
        t.start()

    def _async_run(self, trace_id: str = "unknown") -> None:
        """Background thread entry: run loop, notify on complete/error."""
        from src.device.adb import bind_device_to_thread
        from src.utils.trace import set_trace_id
        set_trace_id(trace_id)
        bind_device_to_thread(self.state.device_serial)

        # Bind agent to THIS daemon thread so ask_user tool finds the right context.
        # Must be here, not in __init__, because __init__ runs on the Concierge thread.
        import threading
        threading.current_thread()._terra_agent_ctx = self

        self._set_log_tag()
        set_current_game(self.state.game)  # Thread-local for tool handlers

        self.state.started_at = time.time()

        # Capture screen dimensions for background review (mirrors run())
        try:
            from src.device.adb import get_adb
            w, h = get_adb().get_screen_size()
            self.state.screen_w, self.state.screen_h = w, h
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "Failed to get screen size — using defaults")

        # Start real-time lesson watcher — runs concurrently, extracts
        # lessons from user corrections as they happen during the task.
        from src.agent.background_review import spawn_lesson_watcher
        spawn_lesson_watcher(self.state, game=self.state.game, loop_guard=self.loop_guard)

        t0 = time.monotonic()
        result: dict[str, Any] = {}
        try:
            try:
                result = self._run_loop()
                crash_notified = False
            except Exception as e:
                logger.error("Agent loop crashed: %s", e, exc_info=True)
                result = {"success": False, "error": str(e), "task_completed": False}
                crash_notified = True
                self._notify(f"出错了…{e}，要不再试试？", notify_type="error")

            duration = time.monotonic() - t0
            # ── Cost tracking (MiMo-V2.5 pricing) ──
            _inp = self.state.total_input_tokens
            _out = self.state.total_output_tokens
            _cr = self.state.total_cache_read_tokens
            _cc = self.state.total_cache_create_tokens
            _cost_input = _inp * 1.0 / 1_000_000
            _cost_cache_read = _cr * 0.02 / 1_000_000
            _cost_cache_create = _cc * 1.0 / 1_000_000
            _cost_output = _out * 2.0 / 1_000_000
            _total_cost = _cost_input + _cost_cache_read + _cost_cache_create + _cost_output
            _total_tokens = _inp + _cr
            _cache_pct = (_cr / _total_tokens * 100) if _total_tokens > 0 else 0
            logger.info(
                "Cost: ¥%.2f (input=%.2f cache_read=%.2f cache_create=%.2f output=%.2f, "
                "%d iter, %.1fM total, cache_rate=%.0f%%)",
                _total_cost, _cost_input, _cost_cache_read, _cost_cache_create, _cost_output,
                self.state.iteration_count, _total_tokens / 1_000_000, _cache_pct,
            )
            self.execution_logger.log(
                user_message=self.state.task_description,
                result=result,
                duration=duration,
                state=self.state,
                matching_skills=self.state.matching_skills,
                task_type="unknown",
            )

            # Mark running=False BEFORE background review and notify.
            # Otherwise user messages that arrive during review trigger
            # are injected to this agent instead of routing through Concierge.
            self.state.running = False

            # Trigger background review for any non-trivial task (≥10 iters),
            # not just completed ones. Failed tasks have valuable failure signals.
            self.reviewer.maybe_trigger(
                state=self.state,
                loop_guard=self.loop_guard,
                review_msgs=getattr(self, "_review_history", None),
                matching_skills=self.state.matching_skills,
                task_description=self.state.task_description,
            )

            if not crash_notified:
                if result.get("success"):
                    final_text = result.get("final_response", "")
                    if final_text and final_text != "搞定啦！":
                        summary = final_text
                    else:
                        task_desc = self.state.task_description or "任务"
                        iters = self.state.iteration_count
                        token_info = ""
                        if self.state.total_input_tokens > 0:
                            inp_k = self.state.total_input_tokens // 1000
                            token_info = f"，消耗 {inp_k}k tokens"
                        summary = f"{task_desc[:50]} 已完成（{iters} 步{token_info}）。博士还需要我做什么吗？"
                    self._notify(summary, notify_type="complete")
                elif self.state._pending_cancel:
                    # Clean cancel — concierge already sent the stop confirmation
                    # via fast_path → _cancel_task reply. Don't duplicate.
                    pass
                elif not result.get("needs_input"):
                    err = result.get("error", "出错了")
                    task_desc = self.state.task_description or "任务"
                    summary = f"{task_desc[:50]} 没能完成——{err}\n博士要我换个方式试试吗？"
                    self._notify(summary, notify_type="error")
        finally:
            # Safety: ensure running is False even if the block above throws
            self.state.running = False
            self.screen_injector.shutdown()
            release_client(self.client)
            clear_agent_tag()

    @staticmethod
    def _needs_confirmation(output: ToolOutput) -> str | None:
        """Check if a tool result requires user intervention.

        Returns a prompt string to show the user, or None if no intervention needed.
        """
        if not output.needs_user:
            return None
        try:
            data = json.loads(output.text)
        except json.JSONDecodeError:
            return None

        msg = data.get("message", "Agent needs your input.")
        texts = output.screen_texts
        hint = f"\n\nScreen text: {', '.join(texts[:12])}" if texts else ""
        return f"{msg}{hint}\n\nWhat should I do? (type a command or button name)"

    @staticmethod
    def _log_tool_result(tool_name: str, output: ToolOutput) -> None:
        """Log key tool outputs for debugging OCR quality."""
        if tool_name not in ("screenshot", "ocr_read", "adb_tap"):
            return
        try:
            data = json.loads(output.text)
        except json.JSONDecodeError:
            return

        if tool_name in ("screenshot", "ocr_read"):
            texts = data.get("texts", data.get("available_texts", output.screen_texts))
            if texts:
                logger.info("  -> OCR texts: %s", ", ".join(texts[:20]))
        elif tool_name == "adb_tap":
            success = data.get("success", False)
            matched = data.get("matched_text", "")
            method = data.get("method", "")
            if success:
                logger.info("  -> TAP OK: '%s' via %s", matched, method)
            else:
                texts = data.get("available_texts", [])
                logger.info("  -> TAP FAIL: target='%s', on screen: %s",
                            data.get("target"), ", ".join(texts[:15]))

    @staticmethod
    def _build_content(output: ToolOutput, tool_name: str = "") -> str | list[dict[str, Any]]:
        """Build Anthropic tool_result content from ToolOutput.

        For action tools (adb_*): trim the JSON to just {success, target, method}
        to reduce token bloat — the LLM already sees the screen via injection.
        For other tools: keep the full output.

        Returns a plain string if no images, or a list of content blocks.
        """
        text = output.text

        # Trim verbose fields from action tool results (LLM already sees screen).
        # screenshot/ocr_read are INFO-GATHERING tools, NOT action tools — their
        # full output is valuable context for the LLM and must not be trimmed.
        if output.images or tool_name in ("adb_tap", "adb_tap_position", "adb_swipe", "adb_scroll", "adb_back"):
            try:
                data = json.loads(text)
                # Keep only what the LLM actually needs to know
                slim: dict[str, Any] = {"success": data.get("success", False)}
                for k in ("target", "matched_text", "method", "direction", "distance",
                         "area", "message", "error", "blocked", "task_done",
                         "position", "screen_coords", "pct"):
                    if k in data:
                        slim[k] = data[k]
                text = json.dumps(slim, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass  # Non-JSON output, keep as-is

        if not output.images:
            return text
        content: list[dict[str, Any]] = []
        for img in output.images:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": img.media_type, "data": img.data},
            })
        content.append({"type": "text", "text": text})
        return content

    def _notify(self, msg: str, notify_type: str = "progress",
                image_b64: str | None = None) -> None:
        """Send a structured AgentNotification through the on_notify callback.

        Extracts agent_id from self.state.agent_handle so the concierge can
        route the notification to the correct GameSlot.  No-op when no
        on_notify callback is set.
        """
        if not self.state.on_notify:
            logger.debug("_notify skipped: no on_notify callback (type=%s)", notify_type)
            return
        agent_id: str = ""
        if self.state.agent_handle is not None:
            agent_id = getattr(self.state.agent_handle, 'agent_id', '') or ""
        from src.agent.state import AgentNotification
        notif = AgentNotification(
            type=notify_type,
            agent_id=agent_id,
            message=msg,
            image_b64=image_b64,
        )
        logger.info("_notify: type=%s agent=%s msg=%.100s", notify_type, agent_id[:8], msg)
        try:
            self.state.on_notify(notif)
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "on_notify failed (non-critical)")

    def _progress(self, msg: str, *, notify: bool = True) -> None:
        """Log and optionally send progress to WeChat.

        Set notify=False to log only without pushing to WeChat.
        Cooldown: at most one notification per 60 seconds (except
        ask_user which goes through _notify_with_screen directly).
        """
        logger.info("Progress: %s", msg)
        # Write to state for Concierge status queries
        self.state.last_progress_text = msg
        self.state.current_activity = msg
        if notify:
            now = time.monotonic()
            if now - getattr(self, "_last_notify_time", 0) < 60:
                return
            self._last_notify_time = now
            self._notify(msg, notify_type="progress")

    def _notify_with_screen(self, msg: str) -> None:
        """Like _progress, but also captures + sends the current screen as JPEG.

        Used for ask_user() so the remote user can see what the agent is
        looking at when it needs help.

        Captures screen first, then sends text + image in one notification
        so the user sees the question text followed by the actual screenshot
        (no "[截图]" placeholder text).
        """
        logger.info("Notify with screen: %s", msg[:80])
        # Capture image first — send text + image together
        image_b64 = None
        try:
            image_b64 = capture_screen_jpeg()
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "screen capture failed, will send text-only")
        self._notify(msg, notify_type="ask_user", image_b64=image_b64)

    def _persist_checkpoint(self) -> None:
        """Persist lightweight checkpoint (completed_subtasks) to task_executions."""
        if not self.state.task_execution_id:
            return
        try:
            import json as _json
            from src.memory.memory_db import memory_db
            checkpoint = _json.dumps({
                "completed_subtasks": list(self.state.completed_subtasks),
                "iteration": self.state.iteration_count,
                "checkpoint_at": time.time(),
            })
            memory_db.conn.execute(
                "UPDATE task_executions SET failure_signal_types=? WHERE id=?",
                (checkpoint, self.state.task_execution_id),
            )
            memory_db.conn.commit()
        except Exception as e:
            from src.utils.errors import handle_error, DegradedError
            handle_error(logger, DegradedError(f"Checkpoint persistence failed: {e}"),
                       "checkpoint_save")

    # ── Shared tool execution helper ──────────────────────────────────
    # Used by both the main loop and interrupt handler to eliminate the
    # ~100 lines of duplicated guard+dispatch+ask_user logic.

    def _execute_tool_safe(self, tc: dict, registry: Any) -> str:
        """Execute one tool call with common guards. Returns "ok" | "break" | "wait_user".

        - "ok" → continue to next tool call
        - "break" → stop tool execution for this iteration
        - "wait_user" → stop and wait for user reply (ask_user triggered)
        """
        tool_name = tc["name"]
        tool_input = tc.get("input", {})

        # Update Concierge-visible activity
        self.state.current_activity = f"{tool_name}({str(tool_input)[:80]})"

        # ── Repeat guard ──
        repeat_key = LoopGuard.build_repeat_key(tool_name, tool_input)
        triggered, msg = self.loop_guard.check_repeat(repeat_key, tool_name, tool_input)
        if triggered:
            logger.warning("Repeat: %s x%d", repeat_key, self.loop_guard.repeat_tracker[repeat_key])
            self.state.add_message("user", msg)
            return "break"

        # ── Burst guard ──
        triggered, msg = self.loop_guard.check_burst(tool_name, repeat_key)
        if triggered:
            logger.warning("Tap burst: %d taps", self.loop_guard.consecutive_op_count)
            self.state.add_message("user", msg)
            return "break"

        # ── Execute ──
        t_tool = time.monotonic()
        logger.info("Executing tool: %s(%s)", tool_name, tool_input)
        try:
            output = registry.dispatch(tool_name, ctx=self.tool_ctx, **tool_input)
        except Exception as e:
            from src.tools.registry import ToolOutput
            output = ToolOutput(text=json.dumps({"error": str(e)}))

        dt = time.monotonic() - t_tool
        if dt > 3.0:
            logger.debug("Slow tool: %s(%.3fs)", tool_name, dt)

        self.state.add_tool_result(tc["id"], tc["name"],
                                   self._build_content(output, tool_name=tc["name"]))

        # ── ask_user handling ──
        confirmation = self._needs_confirmation(output)
        if confirmation:
            self.state._ask_user_count += 1
            self._notify_with_screen(f"🤔 {confirmation}")
            self._wait_cycles = 0
            self.state._waiting_for_user = True
            reply = self.state.pop_interrupt()
            if reply:
                self.state.add_message("user", f"[用户回复] {reply}")
                self.state._waiting_for_user = False
                logger.info("Immediate interrupt reply caught")
            return "wait_user"

        return "ok"

    def _heartbeat(self, reason: str) -> None:
        """Send heartbeat notification during long blocking operations.

        Throttled to at most once per 120 seconds so the user knows the
        agent is alive without being spammed.  Safe to call from the
        owning daemon thread only (no lock needed).
        """
        elapsed = time.monotonic() - self._last_heartbeat_sent
        if elapsed < 300.0:
            return
        self._last_heartbeat_sent = time.monotonic()
        logger.info("Heartbeat: %s", reason)
        if self.state.on_notify:
            self._notify(reason, notify_type="progress")

    def _capture_failure(self, signal_type: str, tool_name: str = "",
                         tool_input: dict | None = None, detail: str = "") -> None:
        """Record a failure signal for later memory extraction.

        Captures the current screen context (hash, dHash, OCR texts) at the
        moment of failure, so the background reviewer can see exactly where
        the agent got stuck.
        """
        from src.agent.state import FailureSignal

        signal = FailureSignal(
            timestamp=time.time(),
            iteration=self.state.iteration_count,
            signal_type=signal_type,
            tool_name=tool_name,
            tool_input=tool_input or {},
            screen_hash=self.state.last_injected_hash,
            screen_dhash=self.state.last_injected_dhash,
            ocr_texts=list(self.state.last_ocr_texts) if self.state.last_ocr_texts else [],
            detail=detail,
        )
        self.state.failure_signals.append(signal)
        logger.debug("Failure signal captured: %s (iter=%d, hash=%s)",
                     signal_type, self.state.iteration_count,
                     (self.state.last_injected_hash or "none")[:8])

    # _inject_screen_after_actions → ScreenInjector.inject_after_actions
    # _inject_initial_screenshot  → ScreenInjector.inject_initial
    # _inject_screen_now          → ScreenInjector.inject_now
    # (full implementations in src/agent/screen_injector.py)

    def _run_loop(self) -> dict[str, Any]:
        tools = registry.get_definitions(
            game=self.state.game,
            skill_names=[s["name"] for s in (self.state.matching_skills or [])],
        )
        final_text = ""

        while self.state.running:
            # ── Task-completed guard: prevent re-entry after task_complete() ──
            if self.state._task_completed:
                logger.info("Task already completed — breaking loop (iter=%d)",
                           self.state.iteration_count)
                break

            # ── Wall-clock timeout guard ──
            if self.state.started_at > 0:
                elapsed_task = time.monotonic() - self.state.started_at
                timeout = config.agent.task_timeout_seconds
                if elapsed_task > timeout:
                    logger.warning(
                        "Task timeout: %.0fs elapsed > %.0fs limit. Stopping.",
                        elapsed_task, timeout,
                    )
                    self.state.add_message("user",
                        f"[系统提示] 任务已运行 {elapsed_task:.0f} 秒，超过 {timeout:.0f} 秒限制，自动终止。")
                    self.state.running = False
                    break

            # ── Waiting for user reply: skip LLM, poll interrupt ──
            # Budget is NOT consumed while waiting — the user is the bottleneck.
            if self.state._waiting_for_user:
                reply = self.state.pop_interrupt()
                if reply:
                    self.state.add_message("user", f"[用户回复] {reply}")
                    self.state._waiting_for_user = False
                    self._wait_cycles = 0
                    logger.info("User reply received, continuing loop")
                    continue
                self._wait_cycles += 1
                wait_cycles = self._wait_cycles
                if wait_cycles % 30 == 0:
                    logger.info("Still waiting for user reply (%ds)", wait_cycles)
                # Heartbeat: every 180s (3 min), tell user the agent is still
                # alive and waiting.  Throttled by _heartbeat's 300s guard
                # so the user gets ~2 reminders before the 10 min timeout.
                if wait_cycles % 180 == 0:
                    wait_mins = wait_cycles // 60
                    self._heartbeat(
                        f"还在等你的回复（已等 {wait_mins} 分钟）…有答案了告诉我。"
                    )
                # 10 min hard timeout — exit gracefully.
                if wait_cycles >= 600:
                    logger.warning("User unresponsive for 10 min — exiting task")
                    self.state.running = False
                    self.state._waiting_for_user = False
                    self._notify("等你 10 分钟没回，我先睡了…有需要再叫我。", notify_type="progress")
                    break
                # Event-based wait: interrupt injection sets the event,
                # waking us immediately — no 1s poll lag for user replies.
                self.state._interrupt_event.wait(timeout=1.0)
                self.state._interrupt_event.clear()
                continue

            # ── Periodic activity heartbeat ──
            # If the agent has been doing something (not waiting for user) but
            # hasn't produced visible output in 45s, send a heartbeat so the
            # user knows it's alive (multi-agent responsiveness).
            if (not self.state._waiting_for_user
                    and self._last_activity > 0
                    and time.monotonic() - self._last_activity > 45):
                self._heartbeat(
                    f"工作中（{self.state.current_activity or '思考中'}）"
                )

            self.state.iteration_count += 1

            # ── Subtask iteration timeout ──
            # Individual subtasks should complete within 30-40 iterations.
            # If one subtask hogs > 40 iterations, inject a warning; at > 60,
            # auto-force completion to give the remaining subtasks a chance.
            # Orchestrator-type skills are NEVER tracked — they are the overall
            # task plan, not an individual subtask that can be "skipped".
            _completed = self.state.completed_subtasks or set()
            _skills = self.state.matching_skills or []
            _current_skill = ""
            for _s in (_skills or []):
                if _s["name"] not in _completed and _s.get("type") != "orchestrator":
                    _current_skill = _s["name"]
                    break
            if _current_skill:
                _sk_iter = self._subtask_iter.get(_current_skill, 0) + 1
                self._subtask_iter[_current_skill] = _sk_iter
                _MAX_SUBTASK_ITER = 60
                _WARN_SUBTASK_ITER = 40
                if _sk_iter >= _MAX_SUBTASK_ITER:
                    logger.warning(
                        "Subtask '%s' exceeded max iterations (%d ≥ %d) — auto-force skipping",
                        _current_skill, _sk_iter, _MAX_SUBTASK_ITER,
                    )
                    self.state.add_message("user",
                        f"[系统 — 子任务超时] 子任务 '{_current_skill}' 已消耗 {_sk_iter} 轮迭代"
                        f"（上限 {_MAX_SUBTASK_ITER}），自动跳过。"
                        f"请立即调用 subtask_done('{_current_skill}', 'auto-timeout') "
                        f"并继续下一个子任务。")
                    del self._subtask_iter[_current_skill]
                    self._subtask_iter_warned.discard(_current_skill)
                elif _sk_iter >= _WARN_SUBTASK_ITER and _current_skill not in self._subtask_iter_warned:
                    logger.warning(
                        "Subtask '%s' approaching iteration limit (%d ≥ %d) — injecting warning",
                        _current_skill, _sk_iter, _WARN_SUBTASK_ITER,
                    )
                    _remaining_skills = [_s["name"] for _s in _skills
                                         if _s["name"] not in _completed and _s["name"] != _current_skill]
                    _msg = (f"[系统提示] 子任务 '{_current_skill}' 已消耗 {_sk_iter} 轮迭代，"
                            f"超过预期（≤{_WARN_SUBTASK_ITER}轮）。")
                    if _remaining_skills:
                        _msg += (f" 还有 {len(_remaining_skills)} 个子任务在排队："
                                f"{', '.join(_remaining_skills[:4])}。")
                    _msg += (f" 如果卡住了：1) 检查这是否是已知的困难操作（如图形按钮'挑战>>'）；"
                            f"2) 考虑调用 subtask_done('{_current_skill}', 'auto-skip') 跳过；"
                            f"3) 或换一种方式（adb_tap_position 猜坐标 / 返回键重来）。")
                    self.state.add_message("user", _msg)
                    self._subtask_iter_warned.add(_current_skill)

            # Check for Concierge-requested cancellation (safe zone only)
            if self.state._pending_cancel and self.state.interrupt_zone == "safe":
                logger.info("Concierge cancellation accepted in safe zone")
                self.state.running = False
                self.state.add_message("user",
                    "[系统提示] 管家要求停止当前任务。任务已安全退出。")
                break

            # Check for user interrupt (status query, command change, ask_user reply)
            interrupt = self.state.pop_interrupt()
            if interrupt:
                self._last_user_hint = interrupt
                self.state.add_message("user", f"[用户指令 — 必须执行] {interrupt}")
                try:
                    from src.games.registry import get_game_registry
                    game_append = get_game_registry().get_system_prompt_append(game_id=self.state.game)
                    response = self.client.chat(
                        system=build_system_prompt_cached(
                            game_append=game_append,
                            skill_text=self._build_skill_context(),
                            state_summary="",
                            screen_w=self.state.screen_w,
                            screen_h=self.state.screen_h,
                            user_task=self.state.user_task,
                        ),
                        messages=self.state.conversation_history,
                        tools=tools,
                    )
                except Exception as e:
                    logger.error("Interrupt LLM call failed: %s", e)
                    self._notify(f"出了点问题: {e}", notify_type="error")
                    self.state.conversation_history.pop()
                    continue

                i_text = extract_text(response)
                i_tools = extract_tool_calls(response)
                i_thinking = extract_thinking(response)
                if i_thinking or i_text:
                    parts: list[str] = []
                    if i_thinking:
                        parts.append(f"[thinking]\n{i_thinking}")
                    if i_text:
                        parts.append(f"[text]\n{i_text}")
                    logger.info("LLM:\n%s", "\n".join(parts))

                if i_tools:
                    # User gave a command → execute, reply via on_notify
                    self.state.add_assistant_with_tools(i_text, i_tools)
                    _interrupt_tool_names: set[str] = set()
                    for tc in i_tools:
                        _interrupt_tool_names.add(tc["name"])
                        result = self._execute_tool_safe(tc, registry)
                        if result == "break":
                            break
                        if result == "wait_user":
                            break  # Stop tool execution, wait for user at top of loop
                    # Only notify + inject if not waiting for user (ask_user path exits early)
                    if not self.state._waiting_for_user:
                        self._notify(i_text or "好的，已处理。", notify_type="progress")
                        self.screen_injector.inject_after_actions(i_tools)
                        self._inject_memory_hints()
                else:
                    # Status query → reply via on_notify, remove from history
                    self._notify(i_text or "还在进行中。", notify_type="progress")
                    self.state.conversation_history.pop()
                continue

            # Inject initial screenshot if none has been injected yet.
            # Skip in create_guide mode — no game interaction needed.
            if self.state.last_injected_hash is None and self.state.task_type != "create_guide":
                self.screen_injector.inject_initial()
                self._inject_memory_hints()

            # Build system prompt for this iteration (cached: stable+context reuse)
            # Cache state summary — game-level DB query, stable within task
            if not self._cached_state_summary:
                self._cached_state_summary = history_db.get_state_summary(self.state.game)
            skill_text = self._build_skill_context()
            from src.games.registry import get_game_registry
            game_append = get_game_registry().get_system_prompt_append(game_id=self.state.game)
            system = build_system_prompt_cached(
                game_append=game_append,
                skill_text=skill_text,
                state_summary=self._cached_state_summary,
                screen_w=self.state.screen_w,
                screen_h=self.state.screen_h,
                user_task=self.state.user_task,
            )

            # Save snapshot for background review before compression strips tool data.
            # Only deep-copy when history is close to compression threshold —
            # avoids copying 30+ MB of base64 images every single iteration.
            if len(self.state.conversation_history) > CompressionService.MESSAGE_THRESHOLD - 10:
                self._review_history = list(self.state.conversation_history)

            # Async compression via CompressionService
            if len(self.state.conversation_history) > CompressionService.MESSAGE_THRESHOLD:
                replacement = self.compressor.check_and_swap(
                    self.state.conversation_history
                )
                if replacement is not None:
                    self.state.conversation_history = replacement

            # Periodic review: every 12 iterations, remind LLM to check its plan.
            # Include game identity so the agent never forgets which game it's
            # on — critical in multi-agent setups where compression may drop
            # the initial identity hint.
            # First review at iter 12 (gives the agent enough time to make
            # meaningful progress), then every 12 thereafter.
            REVIEW_INTERVAL = 12
            if (
                self.state.iteration_count > REVIEW_INTERVAL
                and self.state.iteration_count % REVIEW_INTERVAL == 0
            ):
                from src.games.registry import get_game_registry
                game_name = get_game_registry().get_game_name(self.state.game)
                # Build detailed skill review — survives compression because
                # it's re-injected fresh every 12 iterations from AgentState,
                # not from compressed conversation history.
                skill_blocks: list[str] = []
                for s in self.state.matching_skills:
                    skill_blocks.append(_format_skill_for_review(s))
                skill_detail = "\n\n---\n\n".join(skill_blocks) if skill_blocks else "（无匹配技能）"
                # P1: Check for stale verified skills and warn during review
                stale_warning = ""
                try:
                    stale_names: list[str] = []
                    for s in self.state.matching_skills:
                        if not s.get("verified") or s.get("type") != "script":
                            continue
                        coords_at = s.get("coords_verified_at", "")
                        if coords_at:
                            from datetime import datetime, timezone
                            try:
                                if isinstance(coords_at, str):
                                    vdt = datetime.fromisoformat(coords_at)
                                else:
                                    vdt = datetime.fromtimestamp(float(coords_at), tz=timezone.utc)
                                days = (datetime.now(tz=timezone.utc) - vdt).days
                                if days > 10:
                                    stale_names.append(f"{s['name']}（{days}天）")
                            except Exception as e:
                                logger.debug("Failed to parse coords_verified_at for skill '%s': %s",
                                           s.get('name', '?'), e)
                    if stale_names:
                        stale_warning = (
                            f"\n\n⚠️ 以下技能坐标可能已过期: {', '.join(stale_names)}。"
                            f"如果 skill_run 失败 → 立即切换到手动执行，不要反复重试。"
                            f"任务完成后用 /save 保存新坐标。"
                        )
                except Exception as e:
                    from src.utils.errors import handle_error, DegradedError
                    handle_error(logger, DegradedError(f"Stale warning builder failed: {e}"),
                               "stale_warning")

                review_text = (
                    f"[系统提示 — 第 {self.state.iteration_count} 轮回顾]"
                    f" 你负责 **{game_name}**，设备 {self.state.device_serial}。"
                    f"如果不是这个游戏界面 → 先退回到桌面再启动 {game_name}。\n"
                    f"当前时间: {_fmt_time_now()}。\n"
                    f"## 你必须完成的全部子任务及执行细节:\n\n{skill_detail}"
                    f"{stale_warning}\n\n"
                    "逐一回答:\n"
                    "1. 上面哪些子任务已完成？（标记 ✓）\n"
                    "2. 哪些还没做？\n"
                    "3. 当前在做什么？是否卡住了？\n"
                    "4. 下一个最快能完成的是哪个？\n"
                    "全部子任务确实做完了才调 task_complete。没做完就继续。")

                # Dedup: skip re-injecting identical skill detail.
                # Compression strips the initial system prompt after ~40 messages,
                # but the review itself is a protected message.  Re-injecting the
                # same text every 12 iterations wastes ~500-3000 tokens each time.
                # Only inject if skill_detail (the heavy part) has changed since
                # the last review.
                import hashlib as _hl
                _detail_hash = _hl.md5(skill_detail.encode()).hexdigest()
                if getattr(self, '_last_review_skill_hash', None) != _detail_hash:
                    self._last_review_skill_hash = _detail_hash
                    self.state.add_message("user", review_text)
                else:
                    # Hash unchanged — skip injection entirely.
                    # The original review message (now protected from compression)
                    # is still in conversation history.  Re-injecting even a
                    # lightweight reminder every 12 iterations wastes tokens.
                    pass

            # ── Checkpoint hint (Tier 4): encourage saving intermediate results ──
            # Long data-collection tasks (box-scan, etc.) benefit from frequent saves.
            # The "perfect scan" impulse destroys data when direction gets confused.
            # Stagger vs REVIEW_INTERVAL (12): at 11, the first collision is iter 132
            # (LCM of 12 and 11), reducing stacked system prompts.
            CHECKPOINT_INTERVAL = 11
            if (
                self.state.iteration_count > 20
                and self.state.iteration_count % CHECKPOINT_INTERVAL == 0
            ):
                self.state.add_message("user",
                    "[系统 — 检查点] 任务已运行较长时间。如果已收集到有效数据，"
                    "建议现在就保存一次中间结果（task_complete 或写入文件）——"
                    "不完整的数据比全丢了好。不要在方向错乱中失去已扫描的内容。")

            # Call LLM — unified non-streaming path.
            # Previously WeChat mode used streaming (chat_stream_collect) even
            # though on_text_delta was never passed — MiMo's streaming reports
            # inflated input tokens via the _est_input fallback (~15K/call vs
            # ~1.3K actual from usage.input_tokens), and takes 2x longer (avg
            # 3s vs 1.5s).  Non-streaming is faster, cheaper, and equally
            # reliable for tool-call extraction.
            iter_t0 = time.monotonic()
            self._phase_enter = iter_t0
            if self.state.iteration_count % 5 == 0:
                logger.info("[phase] enter_llm iter=%d game=%s t=%.3f",
                          self.state.iteration_count, self.state.game, iter_t0)
            # Mark a stable message in conversation history as a prompt-cache
            # breakpoint so that Anthropic caches everything before it (system
            # prompt + older messages) and only transmits recent messages fresh.
            self._mark_conversation_cache_point()
            try:
                response = self.client.chat(
                    system=system,
                    messages=self.state.conversation_history,
                    tools=tools,
                )
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                return {"success": False, "error": str(e), "iterations": self.state.iteration_count}

            text = extract_text(response)
            tool_calls = extract_tool_calls(response)
            thinking = extract_thinking(response)
            usage = getattr(response, 'usage', None)
            if usage:
                self.state.total_input_tokens += getattr(usage, 'input_tokens', 0) or 0
                self.state.total_output_tokens += getattr(usage, 'output_tokens', 0) or 0
                self.state.total_cache_read_tokens += getattr(usage, 'cache_read_input_tokens', None) or 0
                self.state.total_cache_create_tokens += getattr(usage, 'cache_creation_input_tokens', None) or 0
            self._last_output_tokens = getattr(usage, 'output_tokens', 0) if usage else 0

            # ── Repeated-thinking detection ──
            # LLM sometimes falls into a death-spiral: same reasoning repeated
            # across multiple rounds with no tool calls.  Detect this via
            # Jaccard similarity of Chinese character bigrams and inject a
            # forceful break-out message BEFORE it burns through 3 rounds of
            # max_tokens exhaustion (which costs ¥).  This is complementary to
            # the max_tokens_no_tool_streak guard below — it catches identical
            # thinking even when output fits within the token budget.
            _all_text = (thinking or "") + "\n" + (text or "")
            _cur_bigrams = _extract_chinese_bigrams(_all_text)
            if self._last_think_bigrams and _cur_bigrams:
                _intersection = len(_cur_bigrams & self._last_think_bigrams)
                _union = len(_cur_bigrams | self._last_think_bigrams)
                _jaccard = _intersection / _union if _union > 0 else 0.0
                if _jaccard > 0.80 and not tool_calls:
                    self._repeat_think_streak += 1
                    logger.warning(
                        "Repeat-thinking detected: Jaccard=%.2f (streak=%d, iter=%d)",
                        _jaccard, self._repeat_think_streak, self.state.iteration_count,
                    )
                    if self._repeat_think_streak >= 2:
                        _completed = self.state.completed_subtasks or set()
                        _skills = self.state.matching_skills or []
                        _current_skill = ""
                        for _s in (_skills or []):
                            if _s["name"] not in _completed:
                                _current_skill = _s["name"]
                                break
                        _force_name = _current_skill or "unknown"
                        self.state.add_message("user",
                            f"[系统 — 重复推理检测] 连续 {self._repeat_think_streak} 轮推理内容高度重复"
                            f"（Jaccard={_jaccard:.1%}）且无操作。立即停止重复推理！\n"
                            f"如果当前操作确实无法执行（按钮找不到/页面卡住），"
                            f"调用 subtask_done('{_force_name}', 'auto-stuck') 跳过。\n"
                            f"如果能操作：直接执行一个工具调用——不要多想，直接操作。")
                        self._repeat_think_streak = 0  # Reset to avoid double-injection
                        self._max_tokens_no_tool_streak = 0
                        self.loop_guard.idle_streak = 0
                else:
                    self._repeat_think_streak = max(0, self._repeat_think_streak - 1)
            elif _cur_bigrams:
                self._repeat_think_streak = max(0, self._repeat_think_streak - 1)
            self._last_think_bigrams = _cur_bigrams

            if thinking or text:
                parts: list[str] = []
                if thinking:
                    parts.append(f"[thinking]\n{thinking}")
                if text:
                    final_text = text
                    parts.append(f"[text]\n{text}")
                logger.info("LLM:\n%s", "\n".join(parts))
            if not thinking and not text:
                block_types = [b.type for b in response.content]
                logger.info("LLM response has no thinking or text. Types: %s", block_types)

            # Mark activity — the agent just completed an LLM call
            self._last_activity = time.monotonic()

            # ── Output guard: detect max_tokens exhaustion with no tool calls ──
            # When the model burns its entire output budget on repetitive text
            # without emitting any tool call, this is pathological — not normal
            # "waiting for loading" behavior (which uses 50-150 tokens).
            # Inject a forceful system message to break the repetition loop.
            _max_tok = 900  # Output guard fires when output ≥ 765 tokens with no tool calls
            if (
                not tool_calls
                and self._last_output_tokens >= int(_max_tok * 0.85)
            ):
                self._max_tokens_no_tool_streak += 1
                logger.warning(
                    "Output guard: max_tokens exhaustion detected "
                    "(output=%d, streak=%d, iter=%d) — injecting recovery prompt",
                    self._last_output_tokens, self._max_tokens_no_tool_streak,
                    self.state.iteration_count,
                )
                # Build context-aware recovery message
                _completed = self.state.completed_subtasks or set()
                _skills = self.state.matching_skills or []
                _current_skill = ""
                for _s in _skills:
                    if _s["name"] not in _completed:
                        _current_skill = _s["name"]
                        break

                # Tiered escalation: light nudge at streak 1-2, force-skip at >= 3.
                # 52 false-triggers occurred when threshold was 1 — most were loading
                # screens where the LLM correctly had no tool calls but simply needed
                # more time.  Only force-skip after 3 CONSECUTIVE exhaustions.
                if self._max_tokens_no_tool_streak >= 3:
                    _force_name = _current_skill or "unknown"
                    self.state.add_message("user",
                        f"[系统 — 自动恢复] LLM 输出 {self._last_output_tokens} tokens "
                        f"耗尽 max_tokens 且无工具调用（疑似重复输出）。当前子任务 '{_force_name}' "
                        f"自动跳过。请立即调用 subtask_done('{_force_name}', 'auto-forced') "
                        f"并继续下一个子任务。不要回头重试该子任务。")
                    logger.warning(
                        "Output guard: auto-forcing subtask completion for '%s' (streak=%d)",
                        _force_name, self._max_tokens_no_tool_streak)
                    # Reset idle streak so we don't double-penalize
                    self.loop_guard.idle_streak = 0
                elif self._max_tokens_no_tool_streak >= 1:
                    # Light nudge — warn but don't force-skip yet.
                    # Strengthened: explicit anti-repetition instruction to break
                    # the "repeat→truncate→re-derive→repeat" death spiral.
                    self.state.add_message("user",
                        f"[系统提醒] 上轮输出 {self._last_output_tokens} tokens 触及上限，"
                        "疑似中英双语重复或反复复读同一段内容。\n"
                        "🔴 禁止中英双语重复。thinking 只写中文 2-3 句。text 只写中文一句话。\n"
                        "🔴 禁止复读前几轮的推理——前面推导过的不要再写一遍。\n"
                        "如果画面正在加载/转场，继续等待即可。如果卡住了，换一种方式操作。")
                # Add the LLM text to history so context is preserved
                if text:
                    self.state.add_message("assistant", text)
                self.screen_injector.inject_now()
                self._inject_memory_hints()
                continue

            # No tool calls — LLM may be waiting or just chatting. Inject fresh
            # screenshot so it can see progress, then continue. Budget is the
            # only hard stop (besides ask_user pending reply).
            if not tool_calls:
                # First-iteration-no-tools guard: if the user's message was pure
                # chat / test / greeting and the LLM already replied text-only,
                # end the conversation — no need to idle-loop for screenshots.
                if self.state.iteration_count == 1 and text:
                    if self._is_chat_message(self.state.task_description or ""):
                        logger.info("Chat message detected on first turn — ending conversation.")
                        if text:
                            self.state.add_message("assistant", text)
                        return {
                            "success": True,
                            "final_response": text,
                            "iterations": self.state.iteration_count,
                            "task_completed": True,
                        }

                logger.info("No tool calls — injecting fresh screenshot and continuing.")
                if text:
                    self.state.add_message("assistant", text)

                # Check if the text looks like an implicit ask_user
                # (LLM forgot to call ask_user tool and output text instead).
                # This is a safety net — the system prompt already tells the LLM
                # to call ask_user(), but when it doesn't, we catch it here.
                if text and any(kw in text for kw in _IMPLICIT_ASK_KW):
                    logger.info("Detected implicit ask_user in text — notifying user and waiting")
                    self._notify(text, notify_type="ask_user")
                    # Don't inject screenshot — ADB may be dead. Wait for user.
                    self.state._waiting_for_user = True
                    continue

                # Guard against screenshot injection when ADB is unhealthy
                from src.device.adb import get_adb as _adb_health
                try:
                    adb_ok = _adb_health()._heartbeat_ok
                except Exception:
                    adb_ok = False
                if not adb_ok:
                    logger.warning("ADB unhealthy — skipping screenshot, asking user")
                    self._notify(
                        "模拟器连接断开了，请检查模拟器是否还在运行。",
                        notify_type="error",
                    )
                    self.state.running = False
                    continue

                # ── Idle watcher FIRST: skip LLM while waiting for screen change ──
                # Uses perceptual dHash + Hamming distance — tolerates
                # Live2D animation noise while detecting real transitions.
                # Zero keywords, zero OCR, zero LLM calls.

                # Runs BEFORE any cooldown — if the screen has already changed
                # (common during loading sequences), act immediately.
                self.loop_guard.tick_idle()  # increment streak
                _pre_idle_dhash = self.state.last_injected_dhash
                if self.loop_guard.idle_streak >= 2 and _pre_idle_dhash:
                    changed, _new_dhash = self.idle_watcher.watch(_pre_idle_dhash)
                    if changed:
                        # Screen is changing — reset static streak, normal waiting
                        self._no_tool_static_streak = 0
                        self.screen_injector.inject_now()
                        self._inject_memory_hints()
                        iter_total = time.monotonic() - iter_t0
                        if iter_total > 3.0:
                            logger.debug("[timing] iter=%d idle+watch total=%.1fs game=%s",
                                       self.state.iteration_count, iter_total, self.state.game)
                        continue
                    # Watcher gave up — screen is truly static.
                    # Track consecutive static-screen no-tool iterations for
                    # escalation (separate from idle_streak which resets at 5).
                    self._no_tool_static_streak += 1
                    # Apply a small cooldown before the LLM safety-net call
                    # to avoid spamming the LLM on a frozen screen.
                    cooldown = self.loop_guard.get_idle_cooldown()
                    if cooldown > 0:
                        logger.info("Idle cooldown: %.1fs (streak=%d, static=%d, watcher exhausted)",
                                   cooldown, self.loop_guard.idle_streak,
                                   self._no_tool_static_streak)
                        # ── Escalation ladder for static screen + no tool calls ──
                        # Build context for subtask-aware hints
                        _completed = self.state.completed_subtasks or set()
                        _skills = self.state.matching_skills or []
                        _current_skill = ""
                        for _s in (_skills or []):
                            if _s["name"] not in _completed:
                                _current_skill = _s["name"]
                                break
                        _skill_hint = ""
                        if _current_skill:
                            _skill_hint = (f"如果子任务 '{_current_skill}' 已完成，"
                                           f"调用 subtask_done('{_current_skill}') 然后继续。")
                        # Streak 3: gentle nudge with subtask awareness
                        if self.loop_guard.idle_streak == 3:
                            self.state.add_message("user",
                                f"[系统提示] 已连续 {self.loop_guard.idle_streak} 轮无操作且画面未变。"
                                f"{_skill_hint or '如果当前步骤已完成，不要继续等待——执行下一步。'}")
                        # Streak 5: stronger push
                        elif self.loop_guard.idle_streak >= 5:
                            self.state.add_message("user",
                                f"[系统提示] 已连续 {self.loop_guard.idle_streak} 轮无操作且画面未变。"
                                "页面可能卡住了——点击 tab / 返回重进 / 滑动刷新。"
                                f"{_skill_hint}")
                            self.loop_guard.idle_streak = 0  # reset to avoid spam
                        # Static streak 8+: auto-force subtask completion
                        if self._no_tool_static_streak >= 8:
                            _force_name = _current_skill or "unknown"
                            logger.warning(
                                "Escalation: auto-forcing subtask '%s' after %d static no-tool rounds",
                                _force_name, self._no_tool_static_streak)
                            self.state.add_message("user",
                                f"[系统 — 自动恢复] 画面已连续 {self._no_tool_static_streak} 轮完全静止"
                                f"且无任何操作。当前子任务 '{_force_name}' 自动标记为完成。"
                                f"请立即调用 subtask_done('{_force_name}', 'auto-escalated') "
                                f"然后继续下一个子任务。")
                            self._no_tool_static_streak = 0
                            self.loop_guard.idle_streak = 0
                        elapsed_cooldown = 0.0
                        while elapsed_cooldown < cooldown and self.state.running:
                            chunk = min(1.0, cooldown - elapsed_cooldown)
                            self.state._interrupt_event.wait(timeout=chunk)
                            self.state._interrupt_event.clear()
                            elapsed_cooldown += chunk
                            if self.state.has_pending_interrupt():
                                self.loop_guard.reset_action_counters()
                                break
                            if self.state._waiting_for_user:
                                break
                            if self.state._pending_cancel and self.state.interrupt_zone == "safe":
                                self.state.running = False
                                self.state.add_message("user",
                                    "[系统提示] 管家要求停止当前任务。任务已安全退出。")
                                break
                elif self.loop_guard.idle_streak == 1:
                    # First idle: no watcher yet, immediate fallback to LLM
                    pass

                self.screen_injector.inject_now()
                self._inject_memory_hints()
                iter_total = time.monotonic() - iter_t0
                if iter_total > 3.0:
                    logger.debug("[timing] iter=%d idle total=%.1fs game=%s",
                               self.state.iteration_count, iter_total, self.state.game)
                continue

            # ── Wait-intent guard (MUST run BEFORE add_assistant_with_tools) ──
            # If the LLM's own thinking says "wait / loading / do not act"
            # but it simultaneously issued navigation tool calls, discard the
            # entire response — including the contradictory assistant message.
            # Inject a correction and let the next round re-observe.
            #
            # SAFETY VALVE: if ANY tool call targets text visible in the current
            # screen OCR, the LLM is interacting with real UI — NOT hallucinating
            # on a black/loading screen.  Skip the guard in that case.
            # This prevents false-positives when the LLM's thinking contains
            # hedging language ("先等一下...关闭弹窗") while issuing a valid
            # interaction with a visible popup button.
            #
            # P0 FIX — consecutive conflict degradation (logs showed 40+ conflicts
            # across 2 days, each wasting ~15s).  After 3 consecutive conflicts:
            # stop discarding and execute anyway.  Also: adb_back is a safe
            # retreat — never block it even on the first conflict.
            if thinking:
                _thinking_lower = thinking.lower()
                _wait_hit = any(kw in _thinking_lower for kw in _WAIT_INTENT_KW)
                if _wait_hit:
                    _conflict_tools = [
                        tc["name"] for tc in tool_calls
                        if tc["name"] in _DARK_NAV_TOOLS
                    ]
                    if _conflict_tools:
                        # Safety valve: if any tool call targets OCR-visible text,
                        # the LLM is interacting with real UI — allow it.
                        _ocr_now = [t.lower() for t in (self.state.last_ocr_texts or [])]
                        _targets_on_screen = False
                        for tc in tool_calls:
                            target = str(tc.get("input", {}).get("target", "")).lower()
                            if target and len(target) >= 2 and any(target in ot for ot in _ocr_now):
                                _targets_on_screen = True
                                break
                        # P0: adb_back is always a safe retreat — never block.
                        _only_back = all(tc == "adb_back" for tc in _conflict_tools)
                        if _only_back and not _targets_on_screen:
                            self._wait_intent_conflict_streak = 0
                            logger.debug(
                                "Wait-intent bypass: adb_back retreat allowed (iter %d)",
                                self.state.iteration_count,
                            )
                            # Fall through — don't discard
                        elif not _targets_on_screen:
                            self._wait_intent_conflict_streak += 1
                            if self._wait_intent_conflict_streak >= 3:
                                # Before executing anyway, check if the screen is
                                # genuinely still loading.  If OCR contains loading
                                # keywords (RHODES ISLAND, INFRASTRUCTURE, etc.),
                                # forcing a tap through just wastes the attempt on
                                # a screen with no interactive elements.
                                _ocr_now_lower = " ".join(
                                    (self.state.last_ocr_texts or [])
                                ).lower()
                                _still_loading = any(
                                    kw in _ocr_now_lower
                                    for kw in _LOADING_SCREEN_KW
                                ) and len(self.state.last_ocr_texts or []) < 10
                                if _still_loading:
                                    logger.warning(
                                        "Wait-intent streak=%d but screen still loading "
                                        "(OCR=%d) — injecting screenshot, NOT executing",
                                        self._wait_intent_conflict_streak,
                                        len(self.state.last_ocr_texts or []),
                                    )
                                    self._wait_intent_conflict_streak = 0
                                    self.screen_injector.inject_now(fast=True)
                                    continue
                                # 3 consecutive conflicts + screen is loaded → execute.
                                logger.warning(
                                    "Wait-intent conflict streak=%d — executing anyway (iter %d)",
                                    self._wait_intent_conflict_streak,
                                    self.state.iteration_count,
                                )
                                self._wait_intent_conflict_streak = 0
                                self.state.add_message("user",
                                    "[系统提醒] 当前画面可能正在加载中，"
                                    "但你已经等待了3轮。先执行操作看看效果吧。")
                                # Fall through — execute tools
                            else:
                                logger.warning(
                                    "Wait-intent conflict: thinking says wait but "
                                    "tool calls are %s (iter %d, streak=%d/3). Discarding response.",
                                    _conflict_tools, self.state.iteration_count,
                                    self._wait_intent_conflict_streak,
                                )
                                self.state.add_message("user",
                                    "[系统 — 检测到矛盾] 你的思考说等待/不操作，但你发出了操作指令。"
                                    "如果确实需要操作（关闭弹窗、跳过对话等）→ 直接做，不要犹豫。"
                                    "如果确实应该等待加载/过渡完成 → 移除操作指令。"
                                    f"连续 {self._wait_intent_conflict_streak}/3 次矛盾后系统自动放行。")
                                # Inject fresh screenshot so the next LLM call sees
                                # the current screen — prevents looping on the same
                                # stale image + correction message.
                                self.screen_injector.inject_now(fast=True)
                                continue

            # Record assistant message with tool calls
            self.loop_guard.reset_action_counters()
            self._no_tool_static_streak = 0
            self._max_tokens_no_tool_streak = 0
            self._repeat_think_streak = 0
            self.state.add_assistant_with_tools(text, tool_calls)

            # ── Dark-screen navigation guard ──
            # When OCR returns 0 texts (loading/black/splash screen with no UI text),
            # the LLM sometimes hallucinates that it's on a launcher/desktop and
            # starts navigating. Intercept: inject a correction and skip the action.
            #
            # Brightness-aware: game splash/login screens (e.g. 开始唤醒 button,
            # 鹰角网络 logo) render text as bitmapped art — OCR returns 0 but the
            # screen is bright and actionable.  True black/loading frames have
            # mean pixel brightness < 15.
            #
            # Also guards low-text screens (1-3 OCR texts) where calling adb_back
            # or navigation tools during game loading can accidentally exit the app.
            _nav_tools = [tc for tc in tool_calls if tc["name"] in _DARK_NAV_TOOLS]
            _ocr_count = len(self.state.last_ocr_texts) if self.state.last_ocr_texts else 0
            if _nav_tools and self.state.iteration_count > 1:
                _nav_names = ", ".join(tc["name"] for tc in _nav_tools)
                _brightness = self.state.last_screen_brightness
                _is_dark = _brightness < 15.0

                if _ocr_count == 0:
                    if _is_dark:
                        # ── Dark-screen escape valve ──
                        # After 120s of the same dark screen, the device may be
                        # frozen in a loading loop, biubiu ad-abyss, or a crash
                        # state.  Stop blocking and let adb_back escape.
                        _cur_hash = self.state.last_injected_hash or ""
                        _now_mono = time.monotonic()
                        if self._dark_screen_since == 0.0 or self._dark_screen_hash != _cur_hash:
                            self._dark_screen_since = _now_mono
                            self._dark_screen_hash = _cur_hash
                        _dark_elapsed = _now_mono - self._dark_screen_since
                        _DARK_ESCAPE_TIMEOUT = 120.0
                        if _dark_elapsed > _DARK_ESCAPE_TIMEOUT:
                            _back_calls = [tc for tc in _nav_tools if tc["name"] == "adb_back"]
                            if _back_calls:
                                logger.warning(
                                    "Dark-screen escape: allowing adb_back after %.0fs of darkness (hash=%s)",
                                    _dark_elapsed, _cur_hash[:8],
                                )
                                self._dark_screen_since = 0.0
                                self._dark_screen_hash = ""
                                self.state.add_message("user",
                                    f"[系统 — 黑屏超时放行] 画面已全黑 {_dark_elapsed:.0f} 秒，"
                                    "允许 adb_back 尝试跳出。")
                                # Fall through — allow adb_back
                            else:
                                # Not adb_back — still block, but tell LLM to try back
                                logger.warning(
                                    "Dark-screen escape: %s blocked after %.0fs — suggesting adb_back",
                                    _nav_names, _dark_elapsed,
                                )
                                self.state.add_message("user",
                                    f"[系统提示] 画面已全黑 {_dark_elapsed:.0f} 秒。"
                                    "尝试用 adb_back() 退出当前状态，不要用点击/滑动。")
                                continue
                        else:
                            # Still within timeout — block navigation
                            logger.warning(
                                "Dark-screen navigation blocked: %s on screen with 0 OCR texts (iter %d, hash=%s, brightness=%.0f, dark_for=%.0fs)",
                                _nav_names, self.state.iteration_count,
                                (_cur_hash or "none")[:8],
                                _brightness, _dark_elapsed,
                            )
                            self._capture_failure(
                                "dark_nav_blocked", _nav_tools[0]["name"], _nav_tools[0].get("input", {}),
                                f"Tried {_nav_names} on 0-OCR screen (hash={_cur_hash}, brightness={_brightness:.0f})",
                            )
                            self.state.add_message("user",
                                "[系统阻止] 当前画面没有任何文字且画面偏暗——你在加载/过渡/黑屏中。"
                                "这不是桌面，不是主界面，不是任何可操作页面。"
                                "不要导航！不要点击！不要滑动！等待画面变化。"
                                "如果已经等了 3 轮以上画面仍然全黑 → ask_user()。")
                            continue  # Skip tool execution, re-inject screen next iteration
                    else:
                        # Bright but 0 OCR — warn, do NOT block
                        logger.warning(
                            "Bright-screen 0-OCR: %s allowed (iter %d, hash=%s, brightness=%.0f)",
                            _nav_names, self.state.iteration_count,
                            (self.state.last_injected_hash or "none")[:8],
                            _brightness,
                        )
                        self._capture_failure(
                            "bright_zero_ocr_warned", _nav_tools[0]["name"], _nav_tools[0].get("input", {}),
                            f"Tried {_nav_names} on bright 0-OCR screen (hash={self.state.last_injected_hash}, brightness={_brightness:.0f})",
                        )
                        self.state.add_message("user",
                            "[系统提示] OCR 检测到 0 条文字，但画面亮度正常——"
                            "可能有图形化按钮/文字 OCR 无法识别。"
                            "导航已放行，但建议先用 magnify() 确认目标可见再操作。"
                            "如果 tap 无反应 → 使用 tap_magnified 精确点击。")
                        # Fall through — allow tool execution

                elif _ocr_count <= 3 and not _is_dark:
                    # Low-text screen (1-3 OCR texts, not dark).
                    # Only block adb_back when OCR texts contain loading/
                    # splash indicators — legitimate popups and menus
                    # should NOT be blocked.
                    _back_calls = [tc for tc in _nav_tools if tc["name"] == "adb_back"]
                    if _back_calls and self.state.last_ocr_texts:
                        _texts_lower = " ".join(self.state.last_ocr_texts).lower()
                        _is_loading = any(kw in _texts_lower for kw in _LOADING_SCREEN_KW)
                        if _is_loading:
                            logger.warning(
                                "Loading-screen adb_back blocked: %d OCR texts (iter %d, hash=%s, brightness=%.0f)",
                                _ocr_count, self.state.iteration_count,
                                (self.state.last_injected_hash or "none")[:8],
                                _brightness,
                            )
                            self._capture_failure(
                                "loading_back_blocked", "adb_back", _back_calls[0].get("input", {}),
                                f"adb_back called on loading screen (ocr_count={_ocr_count}, hash={self.state.last_injected_hash})",
                            )
                            self.state.add_message("user",
                                "[系统阻止] 当前画面文字包含加载/启动信号——"
                                "你很可能在游戏加载画面中，按返回键可能退出应用。"
                                "这不是可操作页面！不要按返回！等待画面变化。")
                            continue  # Skip tool execution, re-inject screen next iteration

            # Phase log: LLM call finished, entering tool execution
            if tool_calls and self.state.iteration_count % 5 == 0:
                _llm_elapsed = time.monotonic() - self._phase_enter
                logger.info("[phase] enter_tools iter=%d game=%s llm=%.1fs t=%.3f",
                          self.state.iteration_count, self.state.game, _llm_elapsed, time.monotonic())

            # Execute tools sequentially
            any_action_succeeded = False

            # ── Pre-compute OCR-derived state (once per iteration) ──
            # These depend on the current screen OCR, which is the same for
            # all tool calls in this iteration (screen hasn't changed yet).
            _is_consume_screen = _consume_screen_ocr_scan(
                self.state.last_ocr_texts)
            _ocr_count = len(self.state.last_ocr_texts) if self.state.last_ocr_texts else 0

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_input = tc.get("input", {})

                # Update Concierge-visible activity
                self.state.current_activity = f"{tool_name}({str(tool_input)[:80]})"

                # Build repeat key via LoopGuard
                repeat_key = LoopGuard.build_repeat_key(tool_name, tool_input)

                # ── Repeat guard ──
                triggered, msg = self.loop_guard.check_repeat(repeat_key, tool_name, tool_input)
                if triggered:
                    logger.warning("Repeat: %s x%d", repeat_key, self.loop_guard.repeat_tracker[repeat_key])
                    self._capture_failure("repeat_stuck", tool_name, tool_input,
                        f"Repeat {tool_name} {self.loop_guard.repeat_tracker[repeat_key]}×")
                    self.state.add_message("user", msg)
                    break

                # ── Burst guard ──
                triggered, msg = self.loop_guard.check_burst(tool_name, repeat_key)
                if triggered:
                    logger.warning("Tap burst: %d taps", self.loop_guard.consecutive_op_count)
                    self._capture_failure("tap_burst", tool_name, tool_input,
                        f"{self.loop_guard.consecutive_op_count} taps, keys: {dict(self.loop_guard.burst_tracker)}")
                    self.state.add_message("user", msg)
                    break

                # ── Scroll guard ──
                triggered, msg = self.loop_guard.check_scroll(tool_name, tool_input)
                if triggered:
                    logger.warning("Scroll loop: %s", tool_input.get("direction", ""))
                    self._capture_failure("scroll_loop", tool_name, tool_input,
                        f"Scroll {tool_input.get('direction','')} loop")
                    self.state.add_message("user", msg)
                    break

                # ── Resource consumption guard (P1: graduated response) ──
                # OCR scanning is pre-computed once per iteration (see
                # _is_consume_screen above the tool loop).  Only tool-specific
                # conditions are checked inside the loop.
                _is_consume_tap = (
                    tool_name in ("adb_tap", "tap_magnified")
                    and any(kw in str(tool_input.get("target", "")) for kw in _CONSUME_TAP_TARGETS)
                )
                _is_consume_pos = (
                    tool_name == "adb_tap_position"
                    and _is_consume_screen
                )
                if _is_consume_screen and (_is_consume_tap or _is_consume_pos):
                    # P1: check farm/daily task context — override block when
                    # the task is explicitly about farming or daily chores.
                    _is_farm_task = any(kw in (self.state.task_description or "")
                                       for kw in ["刷", "farm", "体力", "理智", "日常", "1-7"])
                    if _is_farm_task and self.state._ask_user_count == 0:
                        logger.info(
                            "Resource guard: allowing %s in farm context (desc=%s)",
                            tool_name, self.state.task_description[:60],
                        )
                        # Allow — farm tasks are expected to interact with
                        # resource-adjacent UI during normal gameplay.
                    else:
                        logger.warning(
                            "Resource consumption blocked: %s on upgrade/consumption screen "
                            "(ocr_kw matched, ask_user_count=%d, farm=%s)",
                            tool_name, self.state._ask_user_count, _is_farm_task,
                        )
                        self._capture_failure(
                            "resource_consume_blocked", tool_name, tool_input,
                            f"Tried {tool_name} on consumption screen, "
                            f"ask_user_count={self.state._ask_user_count}",
                        )
                        self.state.add_message("user",
                            "[系统阻止 — 资源消耗红线] 当前画面显示有资源消耗操作（升级/精英化/购买）。"
                            "**任何消耗材料/货币/资源的操作必须先 ask_user() 获得用户同意！**"
                            "如果你认为这个操作安全或用户已同意，调用 ask_user() 让用户确认。"
                            "如果你已经消耗了资源，立即停止，承认错误并 ask_user()。")
                        break  # Stop tool execution, re-inject screen next iteration

                # P1: soft warning for resource keywords without confirm —
                # resource display on main screens should not block navigation.
                if not _is_consume_screen and (_is_consume_tap or _is_consume_pos):
                    _concat = " ".join(self.state.last_ocr_texts or [])
                    _has_resource = any(kw in _concat for kw in _CONSUME_RESOURCE_KW)
                    if _has_resource:
                        logger.info(
                            "Resource guard: soft warning — resource on screen "
                            "but no confirm UI (tool=%s, target=%s)",
                            tool_name, str(tool_input.get("target", ""))[:40],
                        )
                        # Don't block — just log. If this was a real consumption
                        # screen, _is_consume_screen would have caught it.

                t_tool = time.monotonic()
                logger.info("Executing tool: %s(%s)", tool_name, tool_input)
                # Refresh tool context with current screen state (Phase 3)
                self.tool_ctx.ocr_texts = list(self.state.last_ocr_texts) if self.state.last_ocr_texts else []
                self.tool_ctx.screen_dhash = self.state.last_injected_dhash
                self.tool_ctx.screen_hash = self.state.last_injected_hash

                # ── Loading-screen ask_user guard ──
                # Block ask_user when the screen looks like a loading/startup
                # screen (near-empty OCR).  P3: uses screen-state and a higher
                # iteration ceiling instead of the old hard cap of 5.
                #   - Guard active: iter ≤ 30 AND OCR ≤ 2 texts AND question
                #     contains loading/startup keywords AND not a real problem.
                #   - After iter 30 or if screen has >2 OCR texts: guard off.
                _LOADING_GUARD_MAX = 30
                if (tool_name == "ask_user"
                        and self.state.iteration_count <= _LOADING_GUARD_MAX):
                    _ocr_now = self.state.last_ocr_texts or []
                    _q = str(tool_input.get("question", ""))
                    _is_loading_q = any(kw in _q for kw in [
                        "加载", "启动", "loading", "黑屏",
                    ])
                    _is_real_problem = any(kw in _q for kw in [
                        "登录", "过期", "密码", "验证码", "账号",
                        "重新登录", "连接失败", "网络",
                    ])
                    if _is_loading_q and not _is_real_problem and len(_ocr_now) <= 2:
                        logger.info(
                            "Blocking early ask_user on loading screen (iter=%d, ocr=%d)",
                            self.state.iteration_count, len(_ocr_now),
                        )
                        self.state.add_message("user",
                            "[系统阻止] 当前是游戏启动/加载画面，这是正常现象，不是卡死。"
                            "自己等待画面变化即可，不要问用户。用户不在屏幕前，帮不了你。"
                            "等待加载完成后继续执行任务。")
                        output = ToolOutput(
                            text=json.dumps({
                                "blocked": True,
                                "message": "加载画面是正常现象，自行等待即可，不要问用户。",
                            }),
                        )
                    else:
                        output = registry.dispatch(tool_name, ctx=self.tool_ctx, **tool_input)
                elif tool_name == "task_complete":
                    # ── Premature task_complete guard ──
                    # Multi-skill threshold: 20 iterations (first review fires
                    # at iter 16, needed context is loaded).
                    # Single-skill threshold: 8 iterations — even solo tasks
                    # need time to navigate + execute + verify.
                    # P2: added single-skill guard (was unprotected — a solo
                    # skill could call task_complete at iter 2 with no block).
                    skills = self.state.matching_skills
                    _completed = self.state.completed_subtasks or set()
                    _min_iters = 20 if len(skills) > 1 else 8
                    if self.state.iteration_count < _min_iters:
                        _all_done = _completed and len(skills) > 1 and all(
                            s["name"] in _completed for s in skills
                            if s.get("type") != "orchestrator"
                        )
                        if not _all_done:
                            if len(skills) > 1:
                                skill_names = [s["name"] for s in skills]
                                logger.warning(
                                    "task_complete BLOCKED: %d skills at iter=%d: %s",
                                    len(skills), self.state.iteration_count, skill_names,
                                )
                                self.state.add_message("user",
                                    f"[系统提示] 匹配了 {len(skills)} 个子任务: {', '.join(skill_names)}。"
                                    f"才跑了 {self.state.iteration_count} 轮，不太可能全做完。"
                                    f"继续执行剩余任务。全做完后再调 task_complete。")
                            else:
                                logger.warning(
                                    "task_complete BLOCKED: single skill at iter=%d",
                                    self.state.iteration_count,
                                )
                                self.state.add_message("user",
                                    f"[系统提示] 才跑了 {self.state.iteration_count} 轮，"
                                    "任务不太可能已经完成。需要继续执行。"
                                    "确认任务目标达成后再调 task_complete。")
                            output = ToolOutput(
                                text=json.dumps({
                                    "blocked": True,
                                    "message": "请继续执行任务，确认完成后再调 task_complete。",
                                }),
                            )
                        else:
                            output = registry.dispatch(tool_name, ctx=self.tool_ctx, **tool_input)
                    else:
                        output = registry.dispatch(tool_name, ctx=self.tool_ctx, **tool_input)
                else:
                    output = registry.dispatch(tool_name, ctx=self.tool_ctx, **tool_input)
                elapsed = time.monotonic() - t_tool
                logger.info("Tool %s done (%.1fs)", tool_name, elapsed)
                # ── Diagnostic: log task_done for task_complete / subtask_done ──
                if tool_name in ("task_complete", "subtask_done"):
                    logger.info("  -> %s: task_done=%s subtask_done=%s",
                               tool_name, output.task_done, output.subtask_done)
                self._log_tool_result(tool_name, output)
                self.state.add_tool_result(tc["id"], tool_name, self._build_content(output, tool_name=tc["name"]))

                # ── Check task/subtask completion BEFORE all post-processing ──
                # Post-processing code below (stuck-target, magnify guard, failure
                # capture, ask_user) may add messages or alter state — but the
                # completion flag must be respected FIRST.  The old location (after
                # all post-processing) was never reached for unclear reasons; this
                # position was verified to work in the 6/25 run pattern.
                if output.task_done:
                    self.state.running = False
                    self.state._task_completed = True
                    try:
                        data = json.loads(output.text)
                        summary = data.get("summary", final_text)
                    except (json.JSONDecodeError, TypeError):
                        summary = final_text
                    logger.info("Task completed: returning from _run_loop (iter=%d)",
                               self.state.iteration_count)
                    return {
                        "success": True,
                        "final_response": summary,
                        "iterations": self.state.iteration_count,
                        "task_completed": True,
                    }

                # ── Parse output once for all post-processing checks ──
                # Multiple guards below each parse output.text independently.
                # Parse once here so all guards share the same parsed dict.
                _parsed = _try_parse_output(output.text)
                _tool_success = _parsed.get("success") if _parsed else False

                # Prevent re-running a stale skill: if skill_run failed with
                # coordinate/stale errors, tell the LLM not to call it again.
                if tool_name == "skill_run" and _parsed:
                    if not _parsed.get("success") and _parsed.get("message", "").find("执行失败") != -1:
                        skill_name_failed = _parsed.get("skill_name", "")
                        self.state.add_message("user",
                            f"[系统提示] 技能 '{skill_name_failed}' 执行失败，坐标已过期。"
                            "不要再次调用 skill_run，请观察当前画面手动完成剩余步骤。")

                # After a successful box scan, re-run intelligence (BaseScheduler/optimizer)
                # so the LLM gets fresh results based on the new operator data.
                if tool_name == "scan_operator_box" and _parsed:
                    total = _parsed.get("total", 0)
                    if _parsed.get("success") and total >= 3:
                        logger.info("Box scan found %d operators — re-running intelligence", total)
                        self._reinject_intelligence()
                    else:
                        logger.debug("Box scan returned %d operators (success=%s) — skipping re-inject",
                                   total, _parsed.get("success"))

                # After search_memory finds box data (operator names + elite levels),
                # re-run BaseScheduler so the LLM gets an optimized schedule directly
                # instead of manually querying each operator's skills one-by-one.
                if tool_name == "search_memory" and _parsed:
                    results = _parsed.get("results", [])
                    box_signal_count = _parse_box_signal(results)
                    if box_signal_count >= 3:
                        logger.info(
                            "search_memory found box data (%d operator markers) — re-running intelligence",
                            box_signal_count,
                        )
                        self._reinject_intelligence()

                # After adb_back, escalate the warning with back count so the
                # agent stops sooner when stuck in an app that eats back keys
                # (e.g. 明日方舟 login screen, splash screen, another game).
                # P3: hash-aware — deferred check after inject_after_actions
                # so we reuse the screenshot it already captures (no extra ~150ms).
                if tool_name == "adb_back":
                    _pre_back_hash = self.state.last_injected_hash
                    # Defer: tick_back will run after inject_after_actions
                    # updates last_injected_hash with the actual post-back screen.
                    self._pending_back_pre_hash = _pre_back_hash

                # Reset back count when non-back action tools succeed
                if tool_name in ACTION_TOOLS and tool_name != "adb_back" and _tool_success:
                    self.loop_guard.consecutive_back_count = 0

                if tc["name"] in ACTION_TOOLS:
                    if _tool_success:
                        any_action_succeeded = True
                        self._wait_intent_conflict_streak = 0  # P0: reset conflict streak
                    elif _parsed:
                        self._capture_failure(
                            "tool_failure", tool_name, tool_input,
                            f"{tool_name} 失败: {_parsed.get('error', _parsed.get('message', 'unknown'))}"
                        )
                    elif not _tool_success:
                        # Unparsed failure (non-JSON output) — still worth capturing
                        self._capture_failure(
                            "tool_failure_low", tool_name, tool_input,
                            f"{tool_name} 失败 (unparsed): {str(output.text)[:200]}"
                        )
                # ── Low-priority failure capture for non-action tools ──
                # Guards (repeat/burst/scroll/stale/dark_nav) already capture
                # high-priority signals.  This captures the rest: OCR failures,
                # ADB glitches, vision tool errors, etc.  These are marked
                # "tool_failure_low" so background_review can weight them less
                # but pattern_miner can aggregate them for statistical power.
                elif not _tool_success and tool_name not in ("screenshot", "ocr_read"):
                    # screenshot/ocr_read failures are infrastructure, not agent behavior
                    self._capture_failure(
                        "tool_failure_low", tool_name, tool_input,
                        f"{tool_name} 失败: {_parsed.get('error', _parsed.get('message', 'unknown'))}" if _parsed else f"{tool_name} returned success=false (unparsed)"
                    )
                # ── Stuck-target backoff (for adb_tap failures inside ACTION_TOOLS) ──
            # P1: two-stage escalation instead of one-shot.
            #   Stage 1 (count >= 3): hint to use magnify/mark/tap_magnified
            #   Stage 2 (count >= 6): escalate — go back to MAIN SCREEN,
            #                          re-enter from scratch.  Then reset
            #                          backoff so further escalation can
            #                          reach ask_user if needed.
            # ── Stuck-target backoff (ALL tap tools — P2 fix expanded from adb_tap-only) ──
            # Previously only tracked adb_tap failures.  adb_tap_position and
            # tap_magnified always return success=True (the ADB tap command itself
            # never fails) even when the tap misses the intended icon.  This caused
            # 30-round death-spirals on small icons like the base bell (铃铛).
            # Now we extract a normalized target from ALL tap tools and count
            # "sticky-failures" — repeated taps in the same area on the same screen.
            _tap_tools = {"adb_tap", "adb_tap_position", "tap_magnified"}
            if tc["name"] in ACTION_TOOLS and tool_name in _tap_tools:
                # Extract normalized target — maps 铃铛/NOTIFICATION/通知铃铛 to same key
                if tool_name == "adb_tap":
                    _raw = tool_input.get("target", "")
                elif tool_name == "tap_magnified":
                    _raw = tool_input.get("target", "")
                    if not _raw:
                        _raw = f"mag({tool_input.get('x',0)},{tool_input.get('y',0)})"
                else:  # adb_tap_position
                    # Bucket coordinates to nearest 0.05 so nearby taps aggregate
                    # into the same stuck target (0.94,0.13 and 0.93,0.10 → same bucket)
                    _x_b = round(tool_input.get("x_pct", 0) / 0.05) * 0.05
                    _y_b = round(tool_input.get("y_pct", 0) / 0.05) * 0.05
                    _raw = f"pct({_x_b:.2f},{_y_b:.2f})"
                target = _normalize_stuck_target(_raw)

                # Positional tools auto-count (always "succeed" at ADB level).
                # Track by normalized target ONLY — NOT by screen hash.
                # Base UI animations cause minor dHash changes (3c985b9d→61985b51)
                # even when the tap had no effect. Hash-independent tracking
                # ensures the counter accumulates across animation noise.
                _is_positional = tool_name in ("adb_tap_position", "tap_magnified")
                if not _tool_success or _is_positional:
                    if target == self._stuck_target:
                        self._stuck_count += 1
                    else:
                        self._stuck_target = target
                        self._stuck_count = 1

                _STAGE1 = 2
                _STAGE2 = 5

                # ── Bell-icon fast escalation: tiny icons (铃铛) fail more often
                # and require fewer retries before force-skipping.
                _is_bell = "铃铛" in str(tool_input) or "铃" in str(tool_input) or \
                           "notification" in str(tool_input).lower()
                if _is_bell:
                    _STAGE1 = 1
                    _STAGE2 = 3

                if self._stuck_count == _STAGE1:
                    logger.warning(
                        "Stuck-target stage 1: '%s' failed %d× (tool=%s iter=%d)",
                        self._stuck_target, self._stuck_count, tool_name,
                        self.state.iteration_count,
                    )
                    self._capture_failure(
                        "stuck_target", tool_name, tool_input,
                        f"'{self._stuck_target}' failed {self._stuck_count}×",
                    )
                    # ── Include verified fix in the stage 1 message ──
                    _fix_hint = ""
                    if _is_bell:
                        _fix_hint = (
                            "\n**铃铛点击的正确做法（不要猜测坐标）**：\n"
                            "1. adb_tap_position(0.94, 0.13) — 经验证的最稳定坐标\n"
                            "2. 如果仍失败：magnify() 放大右上角 → "
                            "查看底部是否有'可收获'/'点击全部收获' → tap_magnified 直接点\n"
                            "3. 不要用 adb_tap('铃铛') 或 adb_tap('NOTIFICATION') — OCR 无法识别该图标"
                        )
                    self.state.add_message("user",
                        f"[系统 — 目标卡住] 在同一画面点击目标区域"
                        f"失败 {self._stuck_count} 次。{_fix_hint}\n"
                        "**立即切换策略（按优先级）**：\n"
                        "0. adb_back() 关掉可能遮挡目标的弹窗\n"
                        "1. 如果已经试了 magnify + tap_magnified 仍失败 → 不要继续试\n"
                        "2. adb_back() 返回上一级重进，或直接 subtask_done 跳过该子任务\n"
                        "3. 多次尝试仍失败 → ask_user() 求助")

                elif self._stuck_count >= _STAGE2:
                    logger.warning(
                        "Stuck-target stage 2: '%s' failed %d×, force-skipping",
                        self._stuck_target, self._stuck_count,
                    )
                    self._capture_failure(
                        "stuck_target_stage2", tool_name, tool_input,
                        f"'{self._stuck_target}' failed {self._stuck_count}× — escalated",
                    )
                    _completed = self.state.completed_subtasks or set()
                    _skills = self.state.matching_skills or []
                    _current_skill = ""
                    for _s in (_skills or []):
                        if _s["name"] not in _completed:
                            _current_skill = _s["name"]
                            break
                    _force_name = _current_skill or "unknown"
                    self.state.add_message("user",
                        f"[系统 — 强制跳过] 同一目标已连续失败 {self._stuck_count} 次，"
                        f"判定为不可完成。立即调用 subtask_done('{_force_name}', 'auto-stuck') "
                        f"跳过当前子任务，继续下一个。不要继续尝试该目标！")
                    self._stuck_count = 0
                    self._stuck_target = ""

            elif tc["name"] in ACTION_TOOLS and tool_name not in _tap_tools:
                # Non-tap action (adb_back, adb_scroll, adb_swipe, etc.) —
                # the agent is trying a different strategy.  Reset the stuck
                # counter so the next attempt at the same target gets a fresh
                # count — BUT only if the action actually changed the screen.
                # adb_back on an already-open screen is a common counter-reset
                # exploit: the LLM presses back (no effect) then resumes tapping.
                _sc = _parsed.get("screen_changed") if _parsed else None
                if tool_name == "adb_back" and _sc is False:
                    # Back had no effect — don't reset, keep the stuck count
                    pass
                else:
                    self._stuck_count = 0
                    self._stuck_target = ""

                # ── Magnify guard ──
                triggered, msg = self.loop_guard.check_magnify(tc["name"])
                if triggered:
                    logger.warning("Magnify streak: %d consecutive calls", self.loop_guard.magnify_streak)
                    self._capture_failure("magnify_streak", "magnify", tool_input,
                        f"Magnify {self.loop_guard.magnify_streak}× consecutive")
                    self.state.add_message("user", msg)

                # ── Post-restart injection smoothing ──
                # After restart_emulator, the game needs 2-5s to traverse
                # black → splash → login screens.  Extend the next inject
                # deadline by 1.5s so inject_after_actions captures a
                # settled frame instead of a mid-transition one.
                if tool_name == "restart_emulator" and _tool_success:
                    self.screen_injector._post_restart_extension = 1.5

                # Check for user intervention
                confirmation = self._needs_confirmation(output)
                if confirmation:
                    self.state._ask_user_count += 1
                    if self.ask_fn:
                        answer = self.ask_fn(confirmation)
                        self.state.add_message("user", f"[Guidance] {answer}")
                    elif self.state.on_notify:
                        # Async mode — notify user, then wait at top of loop (no LLM burn)
                        self._notify_with_screen(f"🤔 {confirmation}")
                        self._wait_cycles = 0
                        self.state._waiting_for_user = True
                        # Quick poll: if user replied super fast, grab it now
                        immediate = self.state.pop_interrupt()
                        if immediate:
                            self.state.add_message("user", f"[用户回复] {immediate}")
                            self.state._waiting_for_user = False
                            logger.info("Immediate user reply caught")
                    else:
                        return {
                            "success": False,
                            "needs_input": True,
                            "question": confirmation,
                            "iterations": self.state.iteration_count,
                        }
                    break  # No more tool execution while waiting for user

                # ── Subtask boundary cleanup ──
                # When the agent calls subtask_done(), clean intermediate
                # operations for that subtask from conversation history.
                # This keeps context size manageable without relying on
                # lossy LLM summarization.  The agent continues from the
                # clean context into the next subtask.
                if output.subtask_done:
                    name = output.subtask_name
                    result = output.subtask_result
                    logger.info(
                        "Subtask complete: '%s' → '%s'. Cleaning conversation history.",
                        name, result,
                    )
                    self.state.clean_subtask_history(name, result)
                    self.state.completed_subtasks.add(name)
                    self._subtask_iter.pop(name, None)  # reset iteration counter
                    self._subtask_iter_warned.discard(name)
                    self._persist_checkpoint()

                    # ── P0: All subtasks done → early exit ──
                    # Avoid burning remaining budget on post-completion wandering.
                    _ms = self.state.matching_skills or []
                    _cs = self.state.completed_subtasks or set()
                    if _ms and _cs:
                        _all_done = all(
                            s["name"] in _cs for s in _ms
                            if s.get("type") != "orchestrator"
                        )
                        if _all_done:
                            logger.info(
                                "All subtasks completed (%d/%d), early exit at iter %d",
                                len(_cs), len([s for s in _ms if s.get("type") != "orchestrator"]),
                                self.state.iteration_count,
                            )
                            # Let the loop exit normally — the next guard will
                            # break when budget.consume() fails or LLM calls
                            # task_complete.  We just log for visibility.
                    # Inject the current screen after cleaning so the agent
                    # sees a fresh image immediately (otherwise the screen
                    # injection at line 1792 only fires when
                    # any_action_succeeded and next iteration starts).
                    # Force-screen-inject will happen naturally on the next
                    # iteration — the tail of clean_subtask_history already
                    # contains the latest screen.

            # ── Post-tool-loop termination: if _task_completed was set by
            # the per-tool check above, exit immediately.  Belt-and-suspenders
            # — catches edge cases where the per-tool return is swallowed.
            if self.state._task_completed:
                logger.info("Post-loop termination: _task_completed set (iter=%d), exiting",
                           self.state.iteration_count)
                self.state.running = False
                return {
                    "success": True,
                    "final_response": final_text,
                    "iterations": self.state.iteration_count,
                    "task_completed": True,
                }

            # After action tools, inject current screen for next LLM turn
            if any_action_succeeded:
                pre_hash = self.state.last_injected_hash
                pre_ocr = list(self.state.last_ocr_texts) if self.state.last_ocr_texts else []

                # ── Record pre-swipe state for scroll boundary detection (Tier 2) ──
                _SCROLL_TOOLS = {"adb_swipe", "adb_scroll"}
                swipe_calls = [tc for tc in tool_calls if tc["name"] in _SCROLL_TOOLS]
                for sw in swipe_calls:
                    if sw["name"] == "adb_scroll":
                        # Convert semantic direction → physical for scroll_tracker
                        _sc = sw.get("input", {})
                        _axis = _sc.get("axis", "horizontal")
                        _sdir = _sc.get("direction", "more")
                        _PHYS_MAP = {
                            ("next", "horizontal"): "left",  ("next", "vertical"): "up",
                            ("prev", "horizontal"): "right", ("prev", "vertical"): "down",
                            ("more", "horizontal"): "left",  ("more", "vertical"): "up",
                        }
                        direction = _PHYS_MAP.get((_sdir, _axis), "left")
                    else:
                        direction = sw.get("input", {}).get("direction", "")
                    self.scroll_tracker.record_pre_swipe(direction, pre_ocr, pre_hash)

                self.screen_injector.inject_after_actions(tool_calls)
                self._inject_memory_hints()

                # P3: deferred adb_back futility check — uses the screenshot
                # already captured by inject_after_actions instead of a separate
                # ~150ms screencap call per back.
                if self._pending_back_pre_hash is not None:
                    _post = self.state.last_injected_hash
                    _back_count = self.loop_guard.tick_back(
                        pre_hash=self._pending_back_pre_hash, post_hash=_post)
                    if _back_count >= 3:
                        logger.warning(
                            "adb_back futility: %d consecutive backs "
                            "(pre=%s post=%s)",
                            _back_count, self._pending_back_pre_hash[:8],
                            (_post or "none")[:8],
                        )
                        self.state.add_message("user",
                            f"[系统提示] adb_back 已连续 {_back_count} 次无效。"
                            "返回键对你当前所在的界面完全不起作用——你可能卡在了"
                            "启动画面、登录界面、或其他应用的首页。\n"
                            "**立即切换策略，禁止再调 adb_back()：**\n"
                            "1. adb_launch_app() 用包名直接启动目标游戏\n"
                            "2. 按 Home 键回到桌面\n"
                            "3. 两种都无效 → ask_user()")
                    elif _back_count >= 2:
                        logger.warning(
                            "adb_back no-op: %d consecutive backs "
                            "(pre=%s post=%s)",
                            _back_count, self._pending_back_pre_hash[:8],
                            (_post or "none")[:8],
                        )
                        self.state.add_message("user",
                            f"[系统提示] adb_back 连续 {_back_count} 次，"
                            "你可能卡在了无法返回的界面。不要继续 adb_back()——"
                            "点击画面左上角的 ← / 首页 图标来退出，"
                            "或点击画面空白区域。")
                    self._pending_back_pre_hash = None

                # Mark activity — the agent just completed a tool-execution cycle
                self._last_activity = time.monotonic()

                # ── Iteration timing (debug, multi-agent visibility) ──
                iter_total = self._last_activity - iter_t0
                if iter_total > 3.0 or self.state.iteration_count % 20 == 0:
                    logger.info("[timing] iter=%d total=%.1fs game=%s",
                              self.state.iteration_count, iter_total, self.state.game)

                # ── Scroll boundary analysis (Tier 2+3) ──
                if swipe_calls:
                    post_ocr = list(self.state.last_ocr_texts) if self.state.last_ocr_texts else []
                    hint = self.scroll_tracker.analyze_post_swipe(
                        self.state.last_injected_hash, post_ocr
                    )
                    if hint:
                        self.state.add_message("user", hint)
                    progress = self.scroll_tracker.build_progress_hint()
                    if progress:
                        self.state.add_message("user", progress)

                # Periodic progress logging (every 5 iterations, log only).
                # WeChat is too noisy for iteration-level updates — user gets
                # a task-completion summary at the end which is sufficient.
                if self.state.iteration_count % 5 == 0:
                    if final_text and len(final_text) > 5:
                        summary = final_text[:120] + "…" if len(final_text) > 120 else final_text
                        self._progress(summary, notify=False)
                    else:
                        tool_names = [tc["name"] for tc in tool_calls]
                        self._progress(f"第 {self.state.iteration_count} 步: {', '.join(tool_names)}", notify=False)
                # Screen changed → real progress. Reset guard state.
                screen_changed = self.state.last_injected_hash != pre_hash
                post_ocr = list(self.state.last_ocr_texts) if self.state.last_ocr_texts else []
                self.loop_guard.record_screen_change(screen_changed, ocr_texts=post_ocr)

                # Popup-stuck detection: agent is flailing on a persistent popup/dialog
                # with very few OCR texts (e.g. "获得物资" confirmation). Inject hint.
                if self.loop_guard.popup_stuck_streak >= 3:
                    popup_texts = ", ".join(post_ocr[:5]) if post_ocr else "无文字"
                    logger.warning(
                        "Popup stuck: streak=%d, texts=[%s] — injecting adb_back hint",
                        self.loop_guard.popup_stuck_streak, popup_texts,
                    )
                    self.state.add_message("user", (
                        f"[系统提示] 你已经连续 {self.loop_guard.popup_stuck_streak} 轮停在同一个弹窗上"
                        f"（OCR: {popup_texts}），画面hash在变但弹窗没消失。"
                        "这说明你一直在做无效点击。立即停止盲猜坐标！"
                        "第一步：调用 adb_back() — Android返回键通常能直接关闭这个弹窗。"
                        "第二步：如果 adb_back() 无效，用 magnify() → 从坐标标尺读数 → tap_magnified()。"
                        "第三步：仍然关不掉 → ask_user()。"
                    ))

                if not screen_changed:
                    triggered, msg = self.loop_guard.check_stale_screen(
                        False, self._last_user_hint)
                    if triggered:
                        logger.warning("Screen staleness: %d actions with no screen change",
                                      self.loop_guard.stale_screen_count)
                        self._capture_failure("stale_screen", "", {},
                            f"{self.loop_guard.stale_screen_count}× no screen change")
                        hint = ""
                        if self._last_user_hint:
                            hint = (f"User said «{self._last_user_hint[:80]}» but actions had no effect.")
                            self._last_user_hint = ""
                        self.state.add_message("user", msg)


        # Loop exited — determine why
        if not self.state.running:
            logger.info("Task cancelled (state.running=False) after %d iterations",
                       self.state.iteration_count)
            return {
                "success": False,
                "error": "任务被取消",
                "final_response": final_text,
                "iterations": self.state.iteration_count,
                "task_completed": False,
            }

        # Loop exited (not via cancellation) — should not normally reach here
        # since task_complete() sets state.running=False.
        logger.warning("Loop exited unexpectedly after %d iterations", self.state.iteration_count)
        return {
            "success": False,
            "error": "loop_exited_unexpectedly",
            "final_response": final_text,
            "iterations": self.state.iteration_count,
        }

    def _build_skill_context(self) -> str:
        """Build skill text for system prompt injection.

        Three skill types:
        - orchestrator: parent task that dispatches sub-skills via skill_run()
        - script: verified, has coordinates → hide body, push skill_run()
        - guide:  not verified or explicitly marked guide → show body for LLM reference

        Knowledge-type files (explore engine output) are NOT injected —
        their content (screen descriptions) overlaps with guide steps and OCR.
        The explore graph JSON is available for programmatic consumers.
        """
        skills = self.state.matching_skills
        if not skills:
            return ""

        # Filter out knowledge — screen descriptions are redundant with
        # OCR injection + guide steps + memory
        actionable = [s for s in skills if s.get("type") != "knowledge"]
        if not actionable:
            return ""

        # ── Orchestrator mode: when an orchestrator is matched, it IS the
        #     main task.  Show it as the sole plan with its subskills as
        #     available skill_run() targets.  Do NOT list other matched
        #     skills as parallel sub-tasks — the orchestrator's Steps
        #     already define the execution order. ──
        orchestrators = [s for s in actionable if s.get("type") == "orchestrator"]
        if orchestrators:
            orch = orchestrators[0]
            return _expand_orchestrator_body(orch)

        # Prefer script skills, then guide with most body content.
        # Script/verified skills stay compact (skill_run handles them; Fix 1
        # in skill_run.py provides body on failure).  Guide skills need their
        # full body injected so the agent knows what to do without having to
        # call skill_run just to discover the steps.
        def _rank(s):
            score = 0
            is_script = s.get("type") == "script" or s.get("verified")
            if is_script:
                score += 10
            if s.get("body") and "[" in s.get("body", ""):
                score += 5
            return -score

        best = sorted(actionable, key=_rank)[0]

        # Build output — list all skills for multi-skill tasks so the LLM
        # knows the full scope from iteration 1, not just at periodic review
        # (iteration 8).  Without this, the agent often misses sub-tasks and
        # calls task_complete prematurely, wasting iteration budget.
        parts: list[str] = []
        if len(actionable) > 1:
            names = [s.get("name", "?") for s in actionable]
            parts.append(
                f"## 待完成子任务 ({len(names)} 个)\n"
                + ", ".join(names)
                + "\n> 以上全部子任务必须完成才能调用 task_complete。"
            )

        # Expand ALL guide-type skills so the agent has full instructions for
        # every sub-task from iteration 1.  Script/verified skills stay compact
        # — skill_run() handles execution, and on failure the body is injected
        # inline by skill_run.py (Fix 1).  Knowledge/orchestrator types are skipped.
        #
        # CRITICAL: when a verified script or script-type skill exists, do NOT
        # inject ANY guide body — not even a redirect.  Any mention of the guide
        # (name, description, redirect message) gives the agent an alternative
        # path that competes with skill_run().  The agent will consistently choose
        # the manual path over skill_run(), wasting 50+ iterations.
        # Fallback: if skill_run fails, Fix 1 injects the full guide body.
        has_verified = any(
            s.get("verified") or s.get("type") == "script"
            for s in actionable
        )
        guides_shown = False
        if not has_verified:
            for s in sorted(actionable, key=_rank):
                is_guide = (
                    s.get("type") != "script"
                    and not s.get("verified")
                    and s.get("type") != "orchestrator"
                )
                if is_guide:
                    parts.append(_expand_skill_body(s))
                    guides_shown = True

        # Fallback: if no guide skills were expanded (all scripts), expand the
        # best-ranked one — same behavior as before Fix 2.
        if not guides_shown:
            parts.append(_expand_skill_body(best))
        return "\n\n".join(parts)


    @staticmethod
    def _is_chat_message(text: str) -> bool:
        """Check if a message is pure chat/test (no actionable game task).

        Returns True for greetings, small talk, questions without a task,
        and any message that doesn't name a specific game operation.
        """
        if not text:
            return True
        stripped = text.strip()

        # Explicit task signals — if present, this IS a task
        for kw in _TASK_SIGNALS:
            if kw in stripped:
                return False

        # Very short messages without task signals → chat
        if len(stripped) <= 10:
            return True

        # Messages about status/questions without task verbs
        for pat in _CHAT_PATTERNS:
            if pat in stripped:
                return True

        return False

    # ---- Memory auto-injection ----

    def _mark_conversation_cache_point(self) -> None:
        """Mark a stable user message in conversation history as a cache breakpoint.

        Anthropic prompt caching allows up to 4 ``cache_control`` breakpoints
        per request.  The system prompt already uses 3 (stable + task + context).
        This method uses the 4th slot to cache the **conversation prefix** so
        that only the most recent 3-4 screenshot cycles are transmitted fresh.

        **Stability strategy**: find a screenshot-injection message whose image
        has *already* been stripped by ``_strip_old_injection_images`` (i.e.,
        the 4th+ most recent injection).  These messages are text-only and their
        content never changes across iterations — the cache prefix is identical
        call-to-call, guaranteeing a hit.

        In the first 3-4 iterations no messages have been stripped yet; we
        fall back to the oldest screenshot that still has an image.  The cold-
        start history is short enough that a miss is negligible.
        """
        # 1. Remove all existing cache_control from conversation messages so
        #    we never exceed the 4-breakpoint limit.
        for msg in self.state.conversation_history:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        del block["cache_control"]

        # 2. Walk backwards to find the first screenshot message whose image
        #    has already been stripped.  Its text content is permanent.
        for i in range(len(self.state.conversation_history) - 1, -1, -1):
            msg = self.state.conversation_history[i]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            # A stripped injection has its original OCR text block plus a
            # "[截图已省略 — HASH:...]" marker (added by _strip_old_injection_images).
            stripped = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    txt = str(block.get("text", ""))
                    if "截图已省略" in txt or "上一屏截图已移除" in txt:
                        stripped = True
                        break
            if not stripped:
                continue

            # 3. Place cache_control on the *last* text block of this message.
            #    Everything before it (system prompt + older messages + earlier
            #    blocks in this message) enters the cached prefix.
            for block in reversed(content):
                if isinstance(block, dict) and block.get("type") == "text":
                    block["cache_control"] = {"type": "ephemeral"}
                    return

        # 4. Cold-start fallback: no stripped messages yet (iteration ≤ 3).
        #    Mark the earliest screenshot that still has its image.
        for i in range(len(self.state.conversation_history)):
            msg = self.state.conversation_history[i]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    if "HASH:" in str(block.get("text", "")):
                        block["cache_control"] = {"type": "ephemeral"}
                        return

    def _gather_memory_hints(self, ocr_texts: list[str]) -> str | None:
        """Search for relevant memories via MemoryHintService and return formatted hints.

        The service handles dHash → FTS5 → semantic rerank pipeline.
        Injection tracking (hits, dedup, feedback) stays here since it
        touches AgentState.
        """
        from src.tools.remember import _increment_hits

        skill_names = [s.get("name", "") for s in (self.state.matching_skills or [])]
        candidates, hint = self.hint_service.gather(
            ocr_texts=ocr_texts,
            dhash_hex=self.state.last_injected_dhash,
            task_desc=self.state.task_description or "",
            skill_names=skill_names,
            game=self.state.game,
        )

        if not candidates:
            return None

        # ── Fix 10: suppress old anti-skill_run memories ──
        # Memories stored before Fix 2A contain "不要用 skill_run" / "建议使用手动执行而非 skill_run".
        # When at least one matched skill IS verified (has coordinates), suppress these
        # memories to avoid poisoning the LLM against skill_run.
        _has_any_verified = any(
            s.get("verified") or s.get("type") == "script"
            for s in (self.state.matching_skills or [])
        )
        if _has_any_verified and hint:
            import re as _re
            # Strip lines that explicitly discourage skill_run usage
            hint = _re.sub(
                r'[^\n]*fast_chain 成功率仅 0%[^\n]*建议使用手动执行而非 skill_run[^\n]*\n?',
                '', hint,
            )
            hint = _re.sub(
                r'[^\n]*fast_chain 成功率仅 0%[^\n]*坐标可能已过期[^\n]*\n?',
                '', hint,
            )
            # Also strip standalone "不要用 skill_run" lines
            hint = _re.sub(
                r'[^\n]*不要[再调]*用 skill_run[^\n]*\n?',
                '', hint,
            )
            if not hint.strip():
                return None

        # Increment hits and track injected IDs for feedback loop (Phase 1)
        for m in candidates:
            mid = m["id"]
            try:
                _increment_hits(mid)
                if mid not in self.state.injected_memory_ids:
                    self.state.injected_memory_ids.append(mid)
                if mid in self._injected_ids_this_task:
                    continue
                self._injected_ids_this_task.add(mid)
                if self.state.injection_feedback_tracker:
                    self.state.injection_feedback_tracker.record_injection(
                        memory_id=mid,
                        screen_dhash=self.state.last_injected_dhash,
                        conversation_len=len(self.state.conversation_history),
                    )
            except Exception as e:
                logger.debug("Failed to process memory hint candidate %s: %s", mid, e)

        return hint


# ── Create guide prefix detection ───────────────────────────────────

_CREATE_GUIDE_PREFIXES = [
    "/save ", "/s ", "/save\n", "/s\n", "/save", "/s",
    "存操作 ", "存操作\n", "存操作",
    "加操作 ", "加操作\n", "加操作",
]


def _expand_orchestrator_body(skill: dict[str, Any]) -> str:
    """Format an orchestrator skill for system prompt injection.

    Orchestrator skills coordinate multiple sub-skills.  The body contains
    the execution plan; subskills are listed as available skill_run() targets.
    """
    from src.skills.manager import get_skill_manager

    name = skill.get("name", "")
    desc = skill.get("description", "")
    body = skill.get("body", "")
    subskills = skill.get("subskills", [])

    lines = [
        f"## 主任务: {name}",
        f"> {desc}",
        "",
    ]

    # ── Available sub-skills ──
    if subskills:
        game = skill.get("game", "arknights")
        skill_mgr = get_skill_manager(game)
        # Split verified (skill_run) vs unverified (manual only)
        verified_subs: list[str] = []
        unverified_subs: list[str] = []
        missing_subs: list[str] = []
        for sub_name in subskills:
            sub = skill_mgr.load(sub_name)
            if sub and sub.get("verified"):
                verified_subs.append(sub_name)
            elif sub:
                unverified_subs.append(sub_name)
            else:
                missing_subs.append(sub_name)

        if verified_subs:
            lines.append("### 已验证子技能（调用 skill_run 一键执行，极快且免费）")
            for sub_name in verified_subs:
                lines.append(f"- ✅ **{sub_name}** — skill_run('{sub_name}')")
            lines.append("> 🔴 每个子技能完成后必须 notify_with_screen 截图，然后 subtask_done。详见执行计划。")
            lines.append("")
        if unverified_subs:
            lines.append("### 未验证子技能（无快速链坐标，必须手动逐步执行）")
            lines.append("> 🔴 每个子技能完成后必须在**结果界面** notify_with_screen 截图，不能退了界面再截。然后 subtask_done。")
            for sub_name in unverified_subs:
                sub = skill_mgr.load(sub_name)
                sub_desc = sub.get("description", "") if sub else ""
                lines.append(f"- 📋 **{sub_name}** — {sub_desc}")
                # Expand only first 3 steps as a launch hint; full body is
                # injected on the first periodic review (iter 12).  This
                # prevents the initial system prompt from ballooning when
                # many unverified sub-skills are matched.
                if sub and sub.get("body"):
                    sub_body = sub.get("body", "")
                    # Skip blank lines so we pack more meaningful content
                    steps = [line for line in sub_body.strip().split("\n") if line.strip()]
                    if len(steps) <= 20:
                        # Short body — show complete
                        preview = "\n".join(steps)
                    else:
                        # Long body — show first 12 content lines
                        preview = "\n".join(steps[:12])
                        preview += f"\n> （完整 {len(steps)} 行有意义内容将在首次回顾时注入，包含截图/验证指令）"
                    lines.append(f"\n  **{sub_name} 操作步骤：**\n")
                    lines.append(preview)
            lines.append("")
        if missing_subs:
            for sub_name in missing_subs:
                lines.append(f"- ⚠️ **{sub_name}** — 技能文件未找到，需手动执行")
            lines.append("")

    # ── Execution plan ──
    lines.append("### 执行计划")
    lines.append(body)
    lines.append("")
    lines.append("> 已验证用 skill_run()，未验证按步骤手动执行。全部完成后 task_complete()。")

    return "\n".join(lines)


def _build_phase_checklist(body: str) -> str:
    """Extract stage/phase headings from guide body and build a progress checklist.

    Guide skills use patterns like '## 阶段一：启动游戏' or '## 阶段二：...'
    to mark major execution stages.  This function extracts them and builds a
    markdown checklist so the LLM has a concrete progress tracker — it must
    complete phase N before moving to phase N+1.
    """
    import re
    phases: list[str] = []
    for line in body.split("\n"):
        line = line.strip()
        # Match "## 阶段X：名称" or "## 阶段X: 名称" or numbered patterns
        m = re.match(r'^#{1,3}\s*(?:阶段|Phase|Stage)\s*([\d一二三四五六七八九十]+)[：:]\s*(.+)', line)
        if m:
            num = m.group(1)
            title = m.group(2).strip()
            phases.append(f"阶段{num}：{title}")
    if not phases:
        # Fallback: look for step-numbered headings like "1. ## 阶段一：..."
        for line in body.split("\n"):
            line = line.strip()
            m = re.match(r'^\d+\.\s*#{1,3}\s*(?:阶段|Phase|Stage)\s*([\d一二三四五六七八九十]+)[：:]\s*(.+)', line)
            if m:
                num = m.group(1)
                title = m.group(2).strip()
                phases.append(f"阶段{num}：{title}")
    if not phases:
        return ""

    lines: list[str] = [
        "## 执行进度追踪",
        "",
    ]
    for i, p in enumerate(phases):
        if i == 0:
            lines.append(f"- [ ] {p}  ← **当前阶段 — 立即开始**")
        else:
            lines.append(f"- [ ] {p}")
    lines.append("")
    lines.append(
        "> 按阶段顺序执行。卡住超过5步 → ask_user()。"
    )
    return "\n".join(lines)


def _expand_skill_body(skill: dict[str, Any]) -> str:
    """Format a single skill for system prompt injection."""
    name = skill.get("name", "")
    desc = skill.get("description", "")
    body = skill.get("body", "")
    skill_type = skill.get("type", "script" if skill.get("verified") else "guide")
    is_script = skill_type == "script"

    if is_script and skill.get("verified"):
        return (
            f"### {name}\n{desc}\n\n"
            f"> **⚠️ 该技能已验证，包含精确坐标步骤。你的唯一任务："
            f"回到主界面后，立即调用 skill_run('{name}') 一键执行全部步骤。"
            f"不要手动执行技能中的任何步骤，不要用 adb_tap/position 重复技能里的坐标。**\n"
        )

    if is_script:
        # Script but not yet verified — still prefer skill_run, warn about coordinates
        return (
            f"### {name}\n{desc}\n\n"
            f"> **该技能标记为脚本类型，但坐标尚未验证。**"
            f"尝试 skill_run('{name}')——如果快速链失败（坐标过期），"
            f"改为手动执行以下步骤：\n\n{body}"
        )

    # Guide skill — steps are a contract, not a suggestion.
    checklist = _build_phase_checklist(body)
    return (
        f"### {name}\n{desc}\n\n"
        f"> 按以下步骤顺序执行。画面不匹配 → ask_user()。\n\n"
        f"{checklist}\n\n"
        f"{body}"
    )


def _format_skill_for_review(skill: dict[str, Any]) -> str:
    """Format one skill's Steps + Pitfalls for periodic review injection.

    Unlike _expand_skill_body (one-shot system prompt injection), this is
    re-injected every 12 iterations into conversation history as a [系统提示]
    message — immune to compression.  Re-showing full guide content prevents
    the LLM from "forgetting" what to do after many tool-call rounds.

    Guide skills get their full body; script skills get a compact instruction
    to call skill_run() plus pitfalls.
    """
    name = skill.get("name", "")
    desc = skill.get("description", "")
    body = skill.get("body", "")
    pitfalls = skill.get("pitfalls", [])
    skill_type = skill.get("type", "")
    is_script = skill_type == "script" or skill.get("verified")
    is_orchestrator = skill_type == "orchestrator"
    verified = skill.get("verified", False)
    subskills = skill.get("subskills", [])

    lines = [f"### {name}"]
    if desc:
        lines.append(f"> {desc}")

    if is_orchestrator:
        lines.append(f"\n🎯 编排任务 — 你负责按顺序调度以下子技能：")
        if subskills:
            from src.skills.manager import get_skill_manager
            game = skill.get("game", "arknights")
            _rv_skm = get_skill_manager(game)
            verified_count = 0
            for sub_name in subskills:
                sub = _rv_skm.load(sub_name)
                if sub and sub.get("verified"):
                    lines.append(f"  - [ ] ✅ skill_run('{sub_name}') 一键执行")
                    lines.append(f"        🔴 完成后立即 notify_with_screen 截图 → subtask_done('{sub_name}')")
                    verified_count += 1
                elif sub:
                    # Unverified guide — expand body inline, never suggest skill_run
                    sub_body = sub.get("body", "")
                    lines.append(f"  - [ ] 📋 **{sub_name}** — 手动执行（无快速链坐标）")
                    lines.append(f"        🔴 完成后必须在结果界面 notify_with_screen 截图 → subtask_done('{sub_name}')")
                    if sub_body:
                        lines.append(f"    {sub_body.strip()[:500]}")
                else:
                    lines.append(f"  - [ ] ⚠️ **{sub_name}** — 技能文件未找到，需手动执行")
            if verified_count > 0:
                lines.append(f"\n> ✅ {verified_count} 个子技能可用 skill_run 一键执行（极快且免费）。"
                            f" 每个子技能完成后必须先 notify_with_screen 截图再 subtask_done。")
        lines.append(f"\n执行计划：")
        if body:
            lines.append(f"\n{body}")
    elif is_script and verified:
        lines.append(
            f"\n✅ 已验证脚本 — 调用 skill_run('{name}') 一键执行，不要手动操作。"
        )
    elif is_script:
        lines.append(
            f"\n⚠️ 脚本（坐标未验证）— 先尝试 skill_run('{name}')，"
            f"失败则手动执行以下步骤："
        )
        if body:
            lines.append(f"\n{body}")
    else:
        # Guide skill — full body re-injected for review
        lines.append(f"\n📋 按以下步骤执行，画面不匹配时 ask_user():")
        # Phase checklist — survive compression via periodic re-injection
        checklist = _build_phase_checklist(body)
        if checklist:
            lines.append(f"\n{checklist}")
        # Re-inject pitfalls prominently — this is the agent's second chance to read them
        if pitfalls:
            lines.append(f"\n🔴 注意事项:")
            for p in pitfalls:
                lines.append(f"  - {p}")
        if body:
            lines.append(f"\n{body}")

    return "\n".join(lines)


def _fmt_time_now() -> str:
    """Return current time as a compact Chinese string, e.g. '周六 14:30（北京时间）'."""
    from datetime import datetime
    now = datetime.now()
    days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return f"{days[now.weekday()]} {now.hour:02d}:{now.minute:02d}（北京时间）"


# ── Stuck-target normalization ──────────────────────────────────────────
# Maps related target names to canonical forms so the stuck-target counter
# aggregates failures across different tool types (adb_tap('铃铛'),
# tap_magnified(target='通知铃铛'), adb_tap('NOTIFICATION'), etc.)

_STUCK_TARGET_ALIASES: dict[str, str] = {
    "铃铛": "铃铛", "铃": "铃铛", "通知铃铛": "铃铛",
    "通知铃铛蓝色图标": "铃铛", "notification": "铃铛",
}


def _normalize_stuck_target(raw: str) -> str:
    """Normalize a tap target name to a canonical form for stuck detection.

    Maps aliases (铃铛/通知铃铛/NOTIFICATION) to the same key so repeated
    failed taps on the same UI element get counted together, even when the
    agent switches tools (adb_tap → adb_tap_position → tap_magnified).
    """
    if not raw:
        return raw
    lower = raw.lower().strip()
    # Direct alias lookup (longest-match-first)
    for alias, canonical in sorted(_STUCK_TARGET_ALIASES.items(),
                                   key=lambda x: -len(x[0])):
        if alias in lower:
            return canonical
    # Positional targets — normalize coordinate precision to 2 decimals
    if lower.startswith("pct(") or lower.startswith("mag("):
        return lower
    return lower


def _extract_chinese_bigrams(text: str) -> set[str]:
    """Extract Chinese character bigrams from text for Jaccard similarity comparison.

    Used by the repeated-thinking detector to detect when the LLM is rehashing
    the same reasoning across consecutive rounds.  Only Chinese characters are
    considered (ignores English, numbers, punctuation) since the agent's
    operational thinking is primarily in Chinese.
    """
    import re
    chinese = re.findall(r'[一-鿿]+', text)
    all_chars = "".join(chinese)
    if len(all_chars) < 4:
        return set()
    # Character bigrams — captures phrase-level similarity, not just word overlap
    return {all_chars[i:i + 2] for i in range(len(all_chars) - 1)}


def _match_create_guide_prefix(message: str) -> str | None:
    """If message starts with a create-guide prefix, return the body after it.

    Prefixes are short and voice-friendly:
      /save 基建收菜...   (English, unambiguous)
      /s 基建收菜...      (even shorter)
      存操作 基建收菜...   (Chinese, natural)
      加操作 基建收菜...   (Chinese, natural)

    Returns None if no prefix matches.
    """
    if not message:
        return None
    msg = message.strip()
    for prefix in _CREATE_GUIDE_PREFIXES:
        if msg.startswith(prefix):
            body = msg[len(prefix):].strip()
            return body if body else msg  # If nothing after prefix, use full msg
    return None


def run_conversation(agent: TerraAgent, user_message: str) -> dict[str, Any]:
    """Thin wrapper for consistency with Hermes interface."""
    return agent.run(user_message)
