"""Background reviewer — programmatic skill extraction + LLM memory extraction.

Phase 1: Walks conversation messages to extract the successful action chain
         (steps + verified coordinates in one pass). No LLM involved — the
         conversation structure itself encodes the causal chain. adb_back
         marks dead-end boundaries; actions before the last adb_back are
         discarded.

Phase 2: Multi-mode memory extraction sub-agent that adapts to the task
         outcome.  Smooth successes → extract positive action patterns via
         learn_action_pattern().  Stuck-but-recovered → extract both pitfalls
         (remember) and patterns.  Failed → extract root-cause analysis.

Phase 3 (v2): Two-round extraction with self-verification, conversation-
         context-aware prompts, and a quality feedback loop that shows bad
         examples from prior extractions.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from config.settings import config

logger = logging.getLogger(__name__)

# ---- Productive action tools that can be part of a skill ----

_PRODUCTIVE_TOOLS = frozenset({
    "adb_tap", "adb_tap_position", "tap_magnified", "adb_swipe", "adb_scroll",
    "adb_back",
})

# ---- Memory extraction (Phase 2, LLM-based) ----
# Three mode-specific system prompts.  The mode is chosen by
# _classify_extraction_mode() based on failure signals and task outcome.

_EXTRACT_SYS_PATTERN = """你是正向经验提取器。任务执行**完全流畅**—无卡住、无重复操作、无失败信号。

## 提取标准

你的目标是找出这次任务中**值得复用的高效操作**。满足以下全部条件才提取：
1. 涉及了**特定的 UI 判断**（如"OCR 把按钮分成了两个词需要用坐标"）
2. 或涉及了**非标准导航路径**（如"不是点 A 进 B，而是要先到 C 才有入口"）
3. 这个知识在类似任务中**可能再次需要**

## 禁止提取

- 标准操作流程（Phase 1 会自动提取为 skill）
- 常识（"点击开始行动进入战斗"）
- 如果操作成功时使用了精确坐标（adb_tap_position / tap_magnified），请将坐标保存到记忆中。
  同设备同游戏的 UI 坐标是稳定的。格式：坐标: pct(0.85, 0.55) 或 坐标: [1467, 890]
- 如果坐标不可用（纯 OCR tap），用方向描述代替（"右下角蓝色按钮"）
- 对特定关卡/活动的布局描述 → 尽可能泛化到同类型界面

{payload}

## 指令

对每个值得保存的正向模式调用 learn_action_pattern(trigger_screen, action, expected_result, tags)。
trigger_screen 必须包含画面的 OCR 文字。如果没有任何有价值的经验，回复"无"。\n"
"""

_EXTRACT_SYS_HYBRID = """你是经验提取器。任务**有卡住但最终成功**。

## 核心任务：找出最浪费时间的操作错误

仔细阅读 **"用户纠正/中断"** 部分（如果有的话）。用户的每条纠正都直接指出了 agent 的错误操作。对于每条用户纠正，提取：

**类型 A — 用户明确纠正的坑点（最高优先级）** → 调用 remember()
用户说"你没做 X"、"你为什么做 Y"、"应该做 Z"——这就是最精准的操作经验。
例子：
- 用户说"你没点代理指挥"→ 经验：关卡详情页必须先勾选代理指挥复选框再点开始行动，否则战斗无法自动进行
- 用户说"我不是让你刷 td-8 吗"→ 经验：OCR 模糊匹配可能点到错误关卡，关键目标要用坐标确认

**类型 B — 系统检测到卡住的坑点** → 调用 remember()
系统信号（repeat_stuck, stale_screen 等）也指向操作问题，但用户纠正的优先级更高。

**类型 C — 最终成功的正向模式** → 调用 learn_action_pattern()

## 提取规则
- 置信度 >= 0.5 即可保存
- insight 必须包含：卡住时的画面 OCR + 错误操作 + 正确做法
- 禁止：纯常识、一次性细节
- 如果操作成功使用了坐标，请在记忆中保存。格式：坐标: pct(x, y) 或 坐标: [x, y]

{payload}

## 指令

优先处理"用户纠正"部分的反馈。每条用户纠正至少考虑一条经验。
类型 A/B → remember(insight, tags, screen_hash)，screen_hash 从失败信号获取。
类型 C → learn_action_pattern(trigger_screen, action, expected_result, tags)。
如无有价值经验，回复"无"。\n"
"""

_EXTRACT_SYS_PITFALL = """你是根因分析器。任务**失败了**。

## 核心任务：找出导致任务失败的根本原因

不要只报告表面现象（"这个 tap 失败了"），要回答：**为什么整体任务没有完成？**

## ⚠️ 最高优先级：逐条分析用户纠正

如果 payload 中有"用户纠正/中断"部分，**必须逐条分析每一条**。用户说的每一句话都指向一个具体操作错误：
1. 搞清楚用户发现了什么错误
2. 提取为 remember() 经验
3. 经验必须包含：画面 OCR 文字 + 错误操作 + 正确做法

示例（对应常见用户纠正）：
- "你什么时候完成战斗了，那是加载小贴士" → 加载画面的提示文字与游戏进度无关，禁止据此声称任务完成
- "没完成啊，你要点开任务面板看" → 必须打开游戏内任务面板核实子任务实际状态再 call task_complete
- "不是说了很多回不要擅自消耗物资吗" → 任何购买/消耗操作前必须先 ask_user 确认

## 分析框架
1. **用户纠正了什么？** ← 最高优先级，每条必须给出一条 remember()
2. **时间都花在哪了？** 哪些操作在反复重试？每次重试耗时多少秒？
3. **如果重来一次，第一步应该做什么不同？**

## 提取标准
- 每个 insight 结构：画面 OCR + 错误操作 + 根因 + 正确做法
- 禁止：常识（"等待加载完成"）、模糊建议
- 如果纠正涉及具体 UI 位置，且该位置在任务中曾以坐标方式成功操作，请将坐标保存到记忆中
- **不要提取泛泛的经验** — 要说清楚具体画面和具体错误

{payload}

## 指令

逐条分析用户纠正 → remember()。再检查卡住事件中反复重试的操作 → remember()。
如无有价值的教训，回复"无"。\n"
"""

# Mode → (system_prompt, tool_whitelist)
# tool_whitelist: which tools the extractor may call in Round 2.
_EXTRACTION_MODE_CONFIG: dict[str, tuple[str, list[str]]] = {
    "pattern": (_EXTRACT_SYS_PATTERN, ["learn_action_pattern"]),
    "hybrid":  (_EXTRACT_SYS_HYBRID,  ["remember", "learn_action_pattern"]),
    "pitfall": (_EXTRACT_SYS_PITFALL, ["remember"]),
}


def _classify_extraction_mode(
    failure_signals: list[dict[str, Any]],
    task_success: bool,
    chain_len: int,
    iterations: int = 0,
    has_user_feedback: bool = False,
    ask_user_count: int = 0,
) -> str:
    """Determine which extraction mode to use based on task outcome.

    Returns: 'pattern' | 'hybrid' | 'pitfall' | 'skip'

    User feedback is the strongest signal — always use hybrid mode when the
    user corrected or guided the agent, because user corrections are precise
    and must be extracted.  Without user feedback, the mode is chosen by
    failure signals and iteration count.
    """
    has_failures = bool(failure_signals)
    agent_stuck = ask_user_count > 0  # Agent called ask_user — needed help
    high_iter = iterations >= 12  # Something was eating time

    # ── User feedback is always hybrid ──
    # User corrections are the highest-value signal.  Hybrid mode has the
    # "逐条处理用户纠正" rule that pitfall lacks.
    if has_user_feedback:
        return "hybrid"

    if task_success:
        if high_iter or agent_stuck:
            # "Success" but took forever or needed help — something was broken
            return "hybrid"
        if not has_failures and chain_len >= 3:
            return "pattern"
        if has_failures:
            return "hybrid"
        if chain_len < 3 and not agent_stuck:
            return "skip"
    else:
        if has_failures or agent_stuck or high_iter:
            return "pitfall"
        if iterations >= 5:
            # Failed with no explicit signals — still worth a look
            return "hybrid"
        return "skip"

    return "skip"


def _get_bad_examples(game: str, limit: int = 3) -> list[str]:
    """Query previously-extracted memories proven harmful (harm > help).

    These are injected into the extraction prompt as negative examples so
    the extractor LLM learns to avoid producing similar low-quality output.
    """
    from src.memory.memory_db import memory_db

    try:
        rows = memory_db.conn.execute(
            """SELECT body FROM memories_data
               WHERE game=? AND harm_count > help_count
               AND harm_count >= 2 AND source = 'llm_discovery'
               ORDER BY harm_count DESC LIMIT ?""",
            (game, limit),
        ).fetchall()
        return [r["body"][:200] for r in rows]
    except Exception:
        return []


def _build_extraction_payload(
    failure_signals: list[dict[str, Any]],
    success_chain: list[dict[str, Any]],
    task_description: str,
    task_success: bool,
    iterations: int,
    conversation_text: str = "",
    bad_examples: list[str] | None = None,
    user_feedback: list[str] | None = None,
) -> str:
    """Build structured extraction payload with user feedback prioritized (v3).

    User corrections from WeChat interrupts are the HIGHEST-SIGNAL data
    for memory extraction — they tell us exactly what the LLM got wrong.
    These are placed at the top of the payload so the extraction LLM sees
    them first, before any other context.
    """
    parts: list[str] = []

    parts.append(f"## 任务描述\n{task_description}\n")
    parts.append(f"## 任务结果\n{'成功' if task_success else '失败'}（{iterations} 轮）\n")

    # ── User feedback: MOST IMPORTANT section (v3) ─────────────────
    if user_feedback:
        parts.append("## ⚠️ 用户纠正/中断（最重要 — 从这里提取经验！）\n")
        parts.append("下面是用户在执行过程中发来的纠正和反馈，每条都指向了一个导致卡住或失败的操作问题。请逐一分析每条用户反馈，搞清楚：用户发现的操作错误是什么？正确的做法应该是什么？\n")
        for i, fb in enumerate(user_feedback):
            parts.append(f"### 用户纠正 {i+1}\n{fb}\n")
        parts.append("")

    # ── Failure signals ────────────────────────────────────────────
    if failure_signals:
        parts.append("## 卡住事件\n")
        for i, sig in enumerate(failure_signals):
            parts.append(f"### 事件 {i+1}: {sig.get('signal_type', '?')}")
            parts.append(f"- 回合: {sig.get('iteration', '?')}")
            parts.append(f"- 工具: {sig.get('tool_name', '?')}")
            parts.append(f"- 参数: {sig.get('tool_input', {})}")
            parts.append(f"- 详情: {sig.get('detail', '')}")
            ocr = sig.get('ocr_texts', [])
            if ocr:
                parts.append(f"- 当前画面 OCR: {', '.join(ocr[:20])}")
            dhash = sig.get('screen_dhash', '')
            if dhash:
                parts.append(f"- 画面 dHash: {dhash}")
            parts.append("")
    else:
        parts.append("## 卡住事件\n（无 — 任务执行流畅，无卡住信号）\n")

    # ── Success chain ───────────────────────────────────────────────
    if success_chain:
        parts.append("## 成功的操作链\n")
        for j, step in enumerate(success_chain):
            tool = step.get("tool", "?")
            target = step.get("target", "")
            coords = step.get("coords")
            coord_str = f" [{coords[0]}, {coords[1]}]" if coords else ""
            parts.append(f"{j+1}. {tool}('{target}'){coord_str}")
        parts.append("")

    # ── Conversation transcript (supporting context) ────────────────
    if conversation_text:
        parts.append("## 对话记录（辅助参考）\n")
        parts.append(conversation_text[:6000])
        parts.append("")

    # ── Bad examples (quality feedback) ─────────────────────────────
    if bad_examples:
        parts.append("## ⚠️ 历史教训 — 不要生成类似这些的低质量记忆\n")
        parts.append("以下记忆是之前提取的，但后来被证实无效（harm > help）：\n")
        for k, ex in enumerate(bad_examples):
            parts.append(f"{k+1}. {ex[:200]}")
        parts.append("\n请避免生成与以上类似的质量低下的记忆。\n")

    return "\n".join(parts)


# ====================================================================
# Phase 1: Programmatic skill extraction (no LLM)
# ====================================================================


# Regex to extract screen hash from injection labels like
# "[系统自动截图 — 当前屏幕 — HASH:7eddcf76fec1b54e]"
_HASH_RE = re.compile(r"HASH:([0-9a-fA-F]{16})")

# Regex to extract OCR texts from injection labels like "OCR:基建, 制造站, 贸易站"
_OCR_TEXTS_RE = re.compile(r"OCR:([\w一-鿿, /-]+)")

# Main screen markers — a screen is "main" if >=2 of these appear in OCR.
# Both Chinese and English UI variants are supported (Arknights supports both).
# Per-game markers are defined separately because each game has different UI.
_MAIN_MARKERS_BY_GAME: dict[str, list[str]] = {
    "arknights": [
        "基建", "作战", "终端", "Base", "Terminal", "Squad", "干员", "Operator",
        "采购", "Store", "公开招募", "Recruit", "干员寻访", "Headhunt",
    ],
    "reverse1999": [
        "作战", "荒原", "洞悉", "角色", "任务", "银月", "编队",
        "Battle", "Wilderness", "Insight", "Character", "Mission",
        "Squad", "主界面", "首页",
        "不休荒", "仓库", "邮件", "NEW",
    ],
    "lifemakeover": [
        "日程", "协会", "闪亮之旅", "时尚对决", "心意之期",
        "商城", "点击开始", "开始游戏", "Life Makeover",
        "代言女王", "事件簿", "灵感碰撞", "每日签到",
    ],
}
# Fallback: when no game-specific markers are defined, use arknights.
_MAIN_MARKERS_DEFAULT = _MAIN_MARKERS_BY_GAME["arknights"]


def _extract_hash_from_message(content: str | list[dict[str, Any]]) -> str | None:
    """Extract screen hash from a screen-injection message, or None.

    Handles both fresh injections (HASH:xxxxxxxx) and stripped ones
    ([上一屏截图已移除 — OCR:...]) — the stripped format still carries
    OCR text we can use for main screen detection even without a hash.
    """
    if isinstance(content, str):
        m = _HASH_RE.search(content)
        return m.group(1) if m else None
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                m = _HASH_RE.search(str(block.get("text", "")))
                if m:
                    return m.group(1)
    return None


def _is_any_screen_injection(content: str | list[dict[str, Any]]) -> bool:
    """Check if content looks like a screen injection message (fresh or stripped).

    Returns True for:
    - [系统自动截图 — 当前屏幕 — HASH:xxxxxxxx]
    - [上一屏截图已移除 — OCR:...]
    """
    test_text = ""
    if isinstance(content, str):
        test_text = content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                test_text = str(block.get("text", ""))
                break
    return bool(_HASH_RE.search(test_text) or "上一屏截图已移除" in test_text)


def _extract_ocr_texts_from_message(content: str | list[dict[str, Any]]) -> list[str]:
    """Extract OCR texts from a screen-injection label like 'OCR:基建, 制造站'."""
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", ""))
                break
    m = _OCR_TEXTS_RE.search(text)
    if m:
        return [t.strip() for t in m.group(1).split(",") if t.strip()]
    return []


def _extract_success_chain(
    messages: list[dict[str, Any]],
    screen_w: int = 1080,
    screen_h: int = 1920,
    game: str = "arknights",
    matched_skill_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract SUB-CHAINS from conversation messages, split at main screens.

    When an orchestrator runs multiple sub-skills (base-collect → credit-shop
    → recruit), each returns to the main screen between skills.  We use the
    main screen as a natural splitter to produce one sub-chain per skill.

    Each sub-chain is a dict: {skill_name, steps=[...]}.
    skill_name is derived from the nearest skill_run('X') call in the segment.

    START:  first main-screen sighting (≥2 markers in OCR).
    SPLIT:  subsequent main-screen sightings → finalize current sub-chain,
            start a new one.
    END:    task_complete() tool_use — finalize the last sub-chain.

    Returns:
        List of {skill_name, steps} dicts.  Sub-chains without an attributable
        skill_name are included with skill_name="" (caller decides whether to
        use matched_skill_names as fallback).
    """
    # Step 1: Find task_complete as the endpoint.
    task_complete_msg_idx: int | None = None
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        role = msg.get("role", "")
        if role != "assistant" or not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if block.get("name") == "task_complete":
                    task_complete_msg_idx = i
                    break
        if task_complete_msg_idx is not None:
            break

    if task_complete_msg_idx is not None:
        logger.debug("task_complete at message %d/%d", task_complete_msg_idx, len(messages))

    # Step 2: Walk messages, splitting at main screens.
    markers = _MAIN_MARKERS_BY_GAME.get(game, _MAIN_MARKERS_DEFAULT)
    sub_chains: list[dict[str, Any]] = []
    current_steps: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    started = False
    main_screen_count = 0

    # Track the most recent skill_run target seen since last main screen.
    _current_skill_name = ""
    # Fix 9: track consecutive adb_back calls for chain split detection.
    # When the agent backs out to main screen between sub-tasks, 2+ consecutive
    # successful adb_back calls signal a sub-task boundary — flush the chain.
    _consecutive_backs = 0

    def _flush_chain() -> None:
        """Finalize the current sub-chain and start a fresh one."""
        nonlocal current_steps, _current_skill_name, _consecutive_backs
        if current_steps:
            sub_chains.append({
                "skill_name": _current_skill_name,
                "steps": current_steps,
            })
            logger.debug("Sub-chain flushed: %d steps, skill='%s'",
                        len(current_steps), _current_skill_name)
        current_steps = []
        _current_skill_name = ""
        _consecutive_backs = 0

    def _process_pending(data: dict[str, Any]) -> None:
        """Extract a skill step from the pending tool_use + tool_result pair."""
        nonlocal pending, started, _consecutive_backs
        if pending is None:
            return
        if not data.get("success"):
            pending = None
            return
        step = _build_skill_step(pending, data, screen_w, screen_h)
        if step:
            if not started:
                # Fallback: first successful productive step starts the chain
                started = True
                current_steps.clear()
                logger.info("Chain start: fallback (no main screen) — %s", step.get("tool", "?"))
            current_steps.append(step)
            # ── Fix 9: adb_back-triggered chain split ──
            # When 2+ consecutive successful adb_back steps occur, flush the
            # current chain — the agent is backing out between sub-tasks.
            if step.get("tool") == "adb_back":
                _consecutive_backs += 1
                if _consecutive_backs >= 2:
                    logger.info(
                        "Chain split: %d consecutive adb_back after %d steps",
                        _consecutive_backs, len(current_steps))
                    _flush_chain()
                    # Re-start immediately after flush
                    started = True
            else:
                _consecutive_backs = 0
        pending = None

    for msg_idx, msg in enumerate(messages):
        if task_complete_msg_idx is not None and msg_idx >= task_complete_msg_idx:
            break

        content = msg.get("content", "")
        role = msg.get("role", "")

        # ── Main screen detection: split point ──
        # Use _is_any_screen_injection (not just _extract_hash_from_message)
        # because _strip_previous_injection_image removes HASH: from historical
        # injections — only the last one retains its hash.  Stripped messages
        # still carry OCR texts in "[上一屏截图已移除 — OCR:...]" format.
        if _is_any_screen_injection(content):
            ocr = _extract_ocr_texts_from_message(content)
            hits = sum(1 for m in markers if any(m in t for t in ocr))
            if hits >= 2:
                main_screen_count += 1
                _screen_hash = _extract_hash_from_message(content) or "(stripped)"
                if main_screen_count == 1:
                    # First main screen: discard fallback steps (pre-game actions),
                    # start fresh.  Do NOT _flush_chain() — that would save the
                    # fallback garbage as a sub-chain with no skill attribution.
                    started = True
                    current_steps.clear()
                    _current_skill_name = ""
                    logger.info("Chain start: main screen #1 hash=%s (%d markers)",
                               _screen_hash[:8], hits)
                elif main_screen_count >= 2:
                    _flush_chain()
                    started = True
                    logger.info("Chain split: main screen #%d hash=%s (%d markers)",
                               main_screen_count, _screen_hash[:8], hits)
            continue

        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            continue

        # ── Assistant blocks ──
        if role == "assistant":
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    name = block.get("name", "")
                    # Track skill_run calls for sub-chain attribution
                    if name == "skill_run":
                        inp = block.get("input", {})
                        skill_target = inp.get("name", "")
                        if skill_target:
                            _current_skill_name = skill_target
                    # ── Fix 9: subtask_done as explicit chain boundary ──
                    # When the LLM declares a sub-task complete, flush the
                    # current chain so the next sub-task starts a fresh one.
                    if name == "subtask_done":
                        pre_flush_steps = len(current_steps)
                        _flush_chain()
                        started = True
                        _consecutive_backs = 0
                        inp = block.get("input", {})
                        sub_name = inp.get("name", "")
                        if sub_name:
                            _current_skill_name = sub_name
                        logger.info(
                            "Chain split: subtask_done('%s') after %d steps",
                            sub_name, pre_flush_steps,
                        )
                    if name in _PRODUCTIVE_TOOLS:
                        pending = {
                            "tool_use_id": block.get("id", ""),
                            "tool": name,
                            "input": block.get("input", {}),
                        }

        # ── User blocks: tool_result ──
        elif role == "user":
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue
                if block.get("tool_use_id", "") != pending["tool_use_id"] if pending else "":
                    continue
                r = str(block.get("content", ""))
                try:
                    data = json.loads(r)
                except json.JSONDecodeError:
                    pending = None
                    continue
                _process_pending(data)

    # Flush the final sub-chain
    _flush_chain()

    # ── Fix 3: OCR-target based attribution for chains without skill_run ──
    # When the LLM manually executed steps (no skill_run call was made),
    # _current_skill_name stays empty.  This fallback matches each un-attributed
    # sub-chain against matched guide skills by comparing OCR tap targets in the
    # chain with keyword/target mentions in the skill body.
    if matched_skill_names:
        from src.skills.manager import get_skill_manager as _attr_skm
        _attr_mgr = _attr_skm(game)
        for c in sub_chains:
            if c["skill_name"]:
                continue  # Already attributed
            steps = c["steps"]
            if not steps:
                continue
            # Collect OCR targets from chain steps (adb_tap target text)
            chain_targets: set[str] = set()
            for s in steps:
                if s.get("tool") == "adb_tap" and s.get("target"):
                    chain_targets.add(s["target"])
                if s.get("tool") == "adb_tap_position" and s.get("args"):
                    for a in s["args"]:
                        if isinstance(a, str) and len(a) >= 2:
                            chain_targets.add(a)
            if not chain_targets:
                continue
            # Try to match against guide/script (non-orchestrator, non-verified) skills
            best_score = 0
            best_name = ""
            for mn in matched_skill_names:
                ms = _attr_mgr.load(mn)
                if not ms or ms.get("type") == "orchestrator" or ms.get("verified"):
                    continue
                body = ms.get("body", "")
                name_lower = mn.lower()
                # Count how many chain targets appear in the skill body or name
                hits = sum(1 for t in chain_targets
                          if t in body or t.lower() in name_lower)
                if hits > best_score:
                    best_score = hits
                    best_name = mn
            # Require at least 40% of chain targets to match (or 2+ absolute hits)
            overlap = best_score / max(len(chain_targets), 1)
            if best_name and (overlap >= 0.25 or best_score >= 2):
                c["skill_name"] = best_name
                logger.info(
                    "Chain attributed to '%s' via OCR-target matching "
                    "(%d/%d targets, %.0f%% overlap, steps=%d)",
                    best_name, best_score, len(chain_targets),
                    overlap * 100, len(steps),
                )

    if sub_chains:
        logger.info("Extracted %d sub-chains (main screens=%d): %s",
                   len(sub_chains), main_screen_count,
                   [(c["skill_name"], len(c["steps"])) for c in sub_chains])
    else:
        logger.debug("No sub-chains extracted (main screens=%d)", main_screen_count)

    return sub_chains


def _build_skill_step(
    action: dict[str, Any],
    result_data: dict[str, Any],
    screen_w: int,
    screen_h: int,
) -> dict[str, Any] | None:
    """Convert a single action + its successful tool_result into a skill step.

    Each step has: {tool, target?, args?, coords?} matching what
    parse_skill_steps() in fast_chain.py expects.
    """
    tool = action["tool"]
    inp = action["input"]

    if tool == "adb_tap":
        target = inp.get("target", "")
        pos = result_data.get("position")
        coords = (int(pos[0]), int(pos[1])) if (pos and len(pos) == 2) else None
        return {"tool": "adb_tap", "target": target, "coords": coords}

    elif tool == "adb_tap_position":
        x_pct = inp.get("x_pct", 0)
        y_pct = inp.get("y_pct", 0)
        # Prefer verified absolute position from tool_result, fall back to pct-based
        pos = result_data.get("position")
        if pos and len(pos) == 2:
            coords = (int(pos[0]), int(pos[1]))
        else:
            coords = (int(x_pct * screen_w), int(y_pct * screen_h))
        return {"tool": "adb_tap_position", "args": [x_pct, y_pct], "coords": coords}

    elif tool == "tap_magnified":
        # tap_magnified returns screen_coords — verified absolute device coords
        pos = result_data.get("screen_coords") or result_data.get("position")
        if pos and len(pos) == 2:
            x, y = int(pos[0]), int(pos[1])
            x_pct = round(x / screen_w, 3)
            y_pct = round(y / screen_h, 3)
            return {
                "tool": "adb_tap_position",
                "args": [x_pct, y_pct],
                "coords": (x, y),
            }
        return None

    elif tool in ("adb_swipe", "adb_scroll"):
        direction = inp.get("direction", "down")
        distance = inp.get("distance", "half")
        axis = inp.get("axis", "")
        if tool == "adb_scroll":
            return {"tool": "adb_scroll", "args": [direction, distance],
                    "kwargs": {"axis": axis, "direction": direction, "distance": distance},
                    "coords": None}
        return {"tool": "adb_swipe", "args": [direction, distance], "coords": None}

    elif tool == "adb_back":
        return {"tool": "adb_back", "args": [], "coords": None}

    return None


# ---- Skill file generation ----


def _extract_guide_keywords(body: str) -> set[str]:
    """Extract target keywords from a guide skill body for chain attribution.

    Parses the guide's Steps section for button labels and other text targets
    that the LLM would use in adb_tap/scroll/navigate operations.  These
    keywords are used by the orchestrator fallback splitter to assign each
    action-chain step to the sub-skill whose guide body mentions similar targets.

    Extracts:
      - adb_tap('X') / adb_tap("X") / adb_tap(target='X') targets
      - adb_tap_smart(target='X', ...) targets
      - Chinese-quoted text: 「X」, "X", 'X'
      - Bold markers: **X**
      - adb_scroll / adb_swipe direction keywords
    """
    import re
    keywords: set[str] = set()

    # adb_tap('X'), adb_tap("X"), adb_tap(target='X')
    for m in re.finditer(
        r"adb_tap\s*\(\s*(?:target\s*=\s*)?['\"]([^'\"]{1,20})['\"]",
        body,
    ):
        t = m.group(1).strip()
        if t and len(t) >= 1:
            keywords.add(t.lower())
    # adb_tap({'target': 'X'}) — dict/JSON format (used in older guides)
    for m in re.finditer(
        r"adb_tap\s*\(\s*\{[^}]*['\"]target['\"]\s*:\s*['\"]([^'\"]{1,20})['\"]",
        body,
    ):
        t = m.group(1).strip()
        if t:
            keywords.add(t.lower())

    # adb_tap_smart(target='X', row_text='Y')
    for m in re.finditer(
        r"adb_tap_smart\s*\(.*?target\s*=\s*['\"]([^'\"]+)['\"]",
        body,
    ):
        t = m.group(1).strip()
        if t:
            keywords.add(t.lower())

    # 「X」, "X", 'X' — Chinese/quoted button references in guide prose
    for m in re.finditer(r"[「「]([^」」]{1,15})[」」]", body):
        keywords.add(m.group(1).strip().lower())
    for m in re.finditer(r"\*\*([^*]{1,15})\*\*", body):
        keywords.add(m.group(1).strip().lower())

    # Prominent Chinese nouns that are likely button labels (3-8 chars)
    # in guide prose descriptions like "点击 **日程** 进入日程页面"
    for m in re.finditer(r"点击\s*['\"「]?([一-鿿]{2,8})['\"」]?", body):
        keywords.add(m.group(1).strip().lower())

    # Score keywords: longer = more specific (keep the raw set, scoring
    # happens in the matcher via substring containment)
    return {k for k in keywords if len(k) >= 1}


def _generate_skill_from_chain(
    chain: list[dict[str, Any]],
    task_description: str,
    task_type: str,
    game: str,
    preferred_name: str = "",
    force_unverified: bool = False,
) -> tuple[str | None, bool]:
    """Write a skill .md file from an extracted action chain.

    Auto-verifies the skill if ≥80% of action steps have verified coordinates.
    Before creating a new file, checks if a skill with the same derived name
    already exists and merges when the step chains are substantially similar
    (≥70% overlap), avoiding duplicate skills from repeated tasks.

    Args:
        preferred_name: If non-empty and an existing skill with this name exists,
            uses it as the target (version-bumping if needed) instead of deriving
            a new name from task_description + chain targets.

    Returns (skill_name, verified).

    Only generates type=script files — guides are hand-written, not auto-generated.
    """
    if not chain:
        return None, False

    from src.skills.manager import get_skill_manager
    skill_mgr = get_skill_manager(game)

    # ── Fix 4: Use preferred_name (from sub-chain attribution) when available ──
    if preferred_name:
        existing = skill_mgr.load(preferred_name)
        if existing:
            _existing_type = existing.get("type", "")
            # Guides are hand-authored — NEVER overwrite them.
            # Generate a version-bumped script variant instead.
            if _existing_type == "guide":
                name = _bump_skill_version(preferred_name, 1, skill_mgr)
                logger.info(
                    "Skill '%s' is a guide — creating script variant '%s'",
                    preferred_name, name,
                )
            elif _existing_type == "script":
                # Always version-bump auto-generated scripts — never overwrite
                # in place.  Even if overlap % is high, a new run's chain may
                # have subtle differences.  Keep old versions for comparison.
                old_ver = existing.get("version", 0)
                name = _bump_skill_version(preferred_name, old_ver + 1, skill_mgr)
                logger.info(
                    "Skill '%s' version-bumped to '%s' (preferred_name, was v%d)",
                    preferred_name, name, old_ver,
                )
            else:
                name = _bump_skill_version(preferred_name, 1, skill_mgr)
        else:
            name = preferred_name

        # ── Attribution validation ──
        # Verify the chain's targets actually match the named skill.
        # Runs BEFORE the version-bump decision — even if the skill file exists,
        # reject the attribution if the chain content is for a different skill.
        # Without this, a base-collect chain gets named "credit-shop" because
        # OCR attribution guessed wrong in _extract_success_chain.
        _attr_skill = skill_mgr.load(preferred_name)
        if _attr_skill and _attr_skill.get("body"):
            _attr_kw = _extract_guide_keywords(_attr_skill["body"])
            if _attr_kw:
                _chain_targets = set()
                for _s in chain:
                    _t = _s.get("target", "")
                    if _t:
                        _chain_targets.add(_t.lower())
                _match_count = sum(1 for _k in _attr_kw if any(_k in _ct or _ct in _k for _ct in _chain_targets))
                _min_matches = max(3, len(_attr_kw) * 0.30)
                if _match_count < _min_matches:
                    logger.warning(
                        "Attribution rejected: '%s' has only %d/%d keyword matches "
                        "with chain (need ≥%d). Skipping write entirely — "
                        "chain content doesn't match attributed skill.",
                        preferred_name, _match_count, len(_attr_kw), int(_min_matches),
                    )
                    # Chain doesn't match the attributed skill → don't write anything.
                    # Previously we fell through to _derive_skill_name, which created
                    # mislabeled garbage (e.g. "base-collect" with credit shop steps).
                    return None, False
    else:
        name = _derive_skill_name(task_type, chain, task_description)

    tags = _derive_tags(task_type, chain, task_description, preferred_name=preferred_name)
    description = task_description[:80].replace('"', "'")

    # ── Dedup: check existing skill with same name ──
    # (skill_mgr already imported at top of function)
    existing = skill_mgr.load(name)
    if existing and existing.get("body"):
        # NEVER overwrite hand-authored guides — bump to a script variant
        if existing.get("type") == "guide":
            old_name = name
            name = _bump_skill_version(name, 1, skill_mgr)
            logger.info(
                "Skill '%s' is a guide — creating script variant '%s'",
                old_name, name,
            )
            existing = None  # force fresh generation, not dedup-merge
        # Compare step chains: reuse the normalized signature logic from
        # skill_refiner to be OCR-tolerant.
        from src.tools.fast_chain import parse_skill_steps as _parse
        old_steps = _parse(existing["body"])
        _CHAIN_DEDUP_THRESHOLD = 0.7

        def _chain_sig(steps: list[dict]) -> set[str]:
            """Normalized step signature set for dedup comparison."""
            sigs: set[str] = set()
            for s in steps:
                tool = s.get("tool", "")
                args = s.get("args", [])
                normalized = [re.sub(r'\s+', '', str(a)) for a in args]
                sigs.add(f"{tool}({','.join(normalized)})")
            return sigs

        old_sigs = _chain_sig(old_steps)
        new_sigs = _chain_sig(chain)
        if old_sigs and new_sigs:
            union = len(old_sigs | new_sigs)
            intersection = len(old_sigs & new_sigs)
            overlap = intersection / max(union, 1)
            if overlap >= _CHAIN_DEDUP_THRESHOLD:
                logger.info(
                    "Skill '%s' dedup: %.0f%% overlap with existing — updating",
                    name, overlap * 100,
                )
                # Update the existing skill with the new chain (better coords,
                # fresher timestamps). Preserve any manual pitfalls section.
                pitfalls = _extract_pitfalls_section_from_body(existing["body"])
                verified_line = "verified: true\n" if _compute_verified(chain, skill_name=name) else ""
                stats_line = _compute_stats_line(chain)
                new_body = _chain_to_skill_body(chain)
                if pitfalls:
                    new_body += "\n\n" + pitfalls
                frontmatter = (
                    "---\n"
                    f"name: {name}\n"
                    f'description: "{description}"\n'
                    f"tags: [{tags}]\n"
                    f"game: {game}\n"
                    f"type: script\n"
                    f"{verified_line}"
                    f"{stats_line}"
                    "---"
                )
                content = f"{frontmatter}\n\n{new_body}\n"
                skill_mgr.save(name, content)
                logger.info("Skill '%s' updated (dedup merge)", name)
                return name, False if force_unverified else _compute_verified(chain, skill_name=name)
            else:
                # Different chain → version-bump, don't overwrite
                old_name = name
                old_ver = existing.get("version", 0)
                name = _bump_skill_version(old_name, old_ver + 1, skill_mgr)
                logger.info(
                    "Skill '%s' differs (%.0f%% overlap < %d%%) — "
                    "version-bumped to '%s'",
                    old_name, overlap * 100, int(_CHAIN_DEDUP_THRESHOLD * 100), name,
                )
        else:
            # Existing script has no parsable steps — version-bump to be safe
            old_name = name
            name = _bump_skill_version(old_name, 1, skill_mgr)
            logger.info(
                "Skill '%s' has unparsable steps — version-bumped to '%s'",
                old_name, name,
            )

    # Determine verification BEFORE building content (avoid double-save).
    # Verified only when ≥80% of action steps have coordinates and the chain has
    # ≥2 steps for context (see _compute_verified).
    # Fix 8: force_unverified → user corrected during task, skip auto-verification.
    is_verified = False if force_unverified else _compute_verified(chain, skill_name=name)

    # Build skill body
    body = _chain_to_skill_body(chain)

    # Include verified + stats in frontmatter on first (and only) write
    verified_line = "verified: true\n" if is_verified else ""
    stats_line = _compute_stats_line(chain)
    frontmatter = (
        "---\n"
        f"name: {name}\n"
        f'description: "{description}"\n'
        f"tags: [{tags}]\n"
        f"game: {game}\n"
        f"type: script\n"
        f"{verified_line}"
        f"{stats_line}"
        "---"
    )
    content = f"{frontmatter}\n\n{body}\n"

    skill_mgr.save(name, content)

    if is_verified:
        logger.info(
            "Skill '%s' auto-verified (%d/%d steps with coordinates, %.0f%%)",
            name,
            sum(1 for s in chain if s["tool"] in ("adb_tap", "adb_tap_position") and s.get("coords")),
            len([s for s in chain if s["tool"] in ("adb_tap", "adb_tap_position")]),
            _coord_ratio(chain) * 100,
        )
    elif [s for s in chain if s["tool"] in ("adb_tap", "adb_tap_position")]:
        logger.warning(
            "Skill '%s': only %d/%d steps have coordinates (%.0f%%) — NOT auto-verifying",
            name,
            sum(1 for s in chain if s["tool"] in ("adb_tap", "adb_tap_position") and s.get("coords")),
            len([s for s in chain if s["tool"] in ("adb_tap", "adb_tap_position")]),
            _coord_ratio(chain) * 100,
        )

    return name, is_verified


# ── Skill chain helpers (extracted from _generate_skill_from_chain) ──

# Skills that should NEVER be auto-verified because they involve dynamic
# content requiring LLM judgment (variable inventory, random tags, etc.).
# These are always generated as type=script, verified=false — the LLM sees
# the guide body and makes decisions, not blindly follows coordinates.
_NEVER_AUTO_VERIFY: frozenset[str] = frozenset({
    "credit-shop",  # Dynamic inventory + variable discounts → magnify needed
})


def _compute_verified(chain: list[dict[str, Any]], threshold: float = 0.8,
                      skill_name: str = "") -> bool:
    """Check if >=threshold of action steps have coordinates and the chain is reliable.

    Fix 5: Removed the hard requirement for at least one adb_tap_position / tap_magnified
    step.  OCR-based adb_tap() calls also return valid pixel coordinates (the center
    of the detected text box) in tool_result.position — these are equally reliable.
    Lowered threshold from 1.0 to 0.8 to tolerate occasional OCR failures where
    position wasn't captured (e.g. adb_tap succeeded but the tool_result didn't
    include coordinates).
    """
    if skill_name and skill_name in _NEVER_AUTO_VERIFY:
        return False
    action_steps = [s for s in chain if s["tool"] in ("adb_tap", "adb_tap_position")]
    if not action_steps or len(action_steps) < 2:
        return False  # need at least 2 steps for context
    return (
        sum(1 for s in action_steps if s.get("coords")) / len(action_steps)
        >= threshold
    )


def _coord_ratio(chain: list[dict[str, Any]]) -> float:
    """Fraction of action steps with verified coordinates."""
    action_steps = [s for s in chain if s["tool"] in ("adb_tap", "adb_tap_position")]
    if not action_steps:
        return 1.0
    return sum(1 for s in action_steps if s.get("coords")) / len(action_steps)


def _compute_stats_line(chain: list[dict[str, Any]]) -> str:
    """Build step_count/coord_count frontmatter line."""
    action_steps = [s for s in chain if s["tool"] in ("adb_tap", "adb_tap_position")]
    step_count = len(action_steps)
    coord_count = sum(1 for s in action_steps if s.get("coords"))
    return f"step_count: {step_count}\ncoord_count: {coord_count}\n"


def _chain_to_skill_body(chain: list[dict[str, Any]]) -> str:
    """Format a chain into Markdown body."""
    lines = ["## Steps"]
    for i, step in enumerate(chain):
        line = _format_step_line(i + 1, step)
        lines.append(line)
    return "\n".join(lines)


def _extract_pitfalls_section_from_body(body: str) -> str:
    """Extract ## Pitfalls section from existing skill body."""
    m = re.search(r'(##\s*(?:Pitfalls|注意事项).*)', body, re.DOTALL)
    return m.group(1).strip() if m else ""



def _format_step_line(num: int, step: dict[str, Any]) -> str:
    """Format a single step for the skill .md file."""
    tool = step["tool"]

    if tool == "adb_tap":
        target = step["target"]
        if step.get("coords"):
            x, y = step["coords"]
            return f"{num}. adb_tap('{target}')  # [{x}, {y}]"
        return f"{num}. adb_tap('{target}')"

    if tool == "adb_tap_position":
        args = step.get("args", [0, 0])
        if step.get("coords"):
            x, y = step["coords"]
            return f"{num}. adb_tap_position({args[0]}, {args[1]})  # [{x}, {y}]"
        return f"{num}. adb_tap_position({args[0]}, {args[1]})"

    if tool in ("adb_swipe", "adb_scroll"):
        args = step.get("args", ["down", "half"])
        kwargs = step.get("kwargs", {})
        if tool == "adb_scroll":
            axis = kwargs.get("axis", "horizontal")
            return f"{num}. adb_scroll('{args[0]}', axis='{axis}', distance='{args[1]}')"
        return f"{num}. adb_swipe('{args[0]}', '{args[1]}')"

    return f"{num}. {tool}()"


def _bump_skill_version(base_name: str, new_version: int,
                        skill_mgr) -> str:
    """Bump a skill name to the next version (e.g. base-collect → base-collect-v2).

    Checks if the target name already exists and increments until a free slot
    is found.  Used when _generate_skill_from_chain is called with preferred_name
    and the existing skill is already a script (not a guide being promoted).

    Capped at _MAX_SKILL_VERSIONS: once v2 exists, returns v2 (will overwrite
    in place) instead of creating v3, v4, ... endlessly.
    """
    _MAX_SKILL_VERSIONS = 2
    # Cap version nesting: strip existing `-vN` or `-vN-M` suffix
    stripped = re.sub(r'-v\d+(?:-\d+)?$', '', base_name)
    if stripped != base_name:
        logger.info(
            "Version cap: stripped '%s' -> '%s' to prevent nesting inflation",
            base_name, stripped,
        )
        base_name = stripped
    candidate = f"{base_name}-v{new_version}"
    while skill_mgr.load(candidate):
        if new_version >= _MAX_SKILL_VERSIONS:
            logger.info(
                "Version cap reached: '%s' exists (v%d ≥ max v%d) — "
                "will overwrite in place",
                candidate, new_version, _MAX_SKILL_VERSIONS,
            )
            return candidate  # Return existing name, caller will overwrite
        new_version += 1
        candidate = f"{base_name}-v{new_version}"
    return candidate


def _derive_skill_name(task_type: str, chain: list[dict[str, Any]],
                       task_description: str = "") -> str:
    """Derive a searchable skill name from task description (primary) and chain.

    Extracts game identifiers, alphanumeric codes, and meaningful CJK segments
    from task_description.  Chain targets are only used as fallback.
    """
    keywords: list[str] = []
    if task_description:
        # 1. Game identifiers (longer match first)
        game_kw = re.findall(
            r'(?:明日方舟|重返未来1999|以闪亮之名|'
            r'reverse1999|arknights|lifemakeover|'
            r'1999)',
            task_description, re.IGNORECASE,
        )
        keywords.extend(game_kw)

        # 2. Alpha-numeric identifiers
        alnum = re.findall(
            r'(?<!\d)\d{1,2}-\d{1,2}(?!\d)|'          # 1-7, 3-4
            r'[A-Z]{2,3}[-\s]?\d{1,2}|'               # CE-5, GT-6, PR-C-2
            r'ep\d{1,2}|'                               # ep01
            r'(?<![a-zA-Z])\d{3,}(?![a-zA-Z])',        # standalone numbers >=3 digits
            task_description, re.IGNORECASE,
        )
        # Normalize stage codes
        keywords.extend(m.strip().replace(' ', '-').upper() for m in alnum)

        keywords = list(dict.fromkeys(keywords))  # dedup

    base = task_type if (task_type and task_type != "unknown") else "task"

    if keywords:
        return f"{base}-{'-'.join(keywords[:3])}"

    # Fallback: chain targets
    chain_targets: list[str] = []
    for step in chain:
        if step["tool"] == "adb_tap" and step.get("target"):
            t = step["target"]
            t = re.sub(r"[^\w一-鿿-]", "", t)
            if t:
                chain_targets.append(t)
    if chain_targets:
        return f"{base}-{'-'.join(chain_targets[:3])}"
    return f"{base}-auto"


def _derive_tags(task_type: str, chain: list[dict[str, Any]],
                 task_description: str = "", preferred_name: str = "") -> str:
    """Derive skill tags from task description, task type, sub-skill tags, and chain content.

    Priority: sub-skill tags (from preferred_name) > game name > task_type > chain targets.
    Chain adb_tap targets are only used as supplement (last resort) — they're
    step-level text, not meaningful categorization tags.
    """
    tags: list[str] = []

    # ── Priority 1: Sub-skill's own tags (when attributed to a known skill) ──
    if preferred_name:
        from src.skills.manager import get_skill_manager as _dt_skm
        import threading as _dt_thr
        _game = getattr(_dt_thr.current_thread(), '_terra_game', None) or "arknights"
        try:
            _dt_mgr = _dt_skm(_game)
            _sub_skill = _dt_mgr.load(preferred_name)
            if _sub_skill and _sub_skill.get("tags"):
                _sub_tags = [t.strip() for t in _sub_skill["tags"].split(",") if t.strip()]
                for t in _sub_tags:
                    if t not in tags:
                        tags.append(t)
        except Exception:
            pass

    if task_description:
        # Game name (only if not already in tags from sub-skill)
        if re.search(r'明日方舟|arknights', task_description, re.IGNORECASE):
            _add_if_new(tags, "明日方舟")
        if re.search(r'重返.*1999|reverse.*1999|1999', task_description, re.IGNORECASE):
            _add_if_new(tags, "重返未来1999")
        if re.search(r'以闪|lifemakeover', task_description, re.IGNORECASE):
            _add_if_new(tags, "以闪")
        # Stage codes
        alnum = re.findall(
            r'(?<!\d)\d{1,2}-\d{1,2}(?!\d)|'
            r'[A-Z]{2,3}[-\s]?\d{1,2}|'
            r'ep\d{1,2}|'
            r'(?<![a-zA-Z])\d{3,}(?![a-zA-Z])',
            task_description, re.IGNORECASE,
        )
        for a in alnum:
            a = a.strip().replace(' ', '-').upper()
            _add_if_new(tags, a)

    if task_type and task_type != "unknown":
        _add_if_new(tags, task_type)

    # Chain targets (supplement only — lowest priority, fill up to 8 max)
    for step in chain:
        if step["tool"] == "adb_tap" and step.get("target"):
            t = step["target"]
            _add_if_new(tags, t)
            if len(tags) >= 8:
                break

    return ", ".join(tags[:8])


def _add_if_new(tags: list[str], item: str) -> None:
    """Append item to tags list if not already present."""
    if item and item not in tags:
        tags.append(item)


def _build_conversation_text(messages: list[dict[str, Any]]) -> str:
    """Build a text summary of the conversation for memory extraction prompt."""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            sub_parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        sub_parts.append(str(block.get("text", "")))
                    elif block.get("type") == "tool_use":
                        sub_parts.append(f"[工具调用: {block.get('name', '?')}]")
                    elif block.get("type") == "tool_result":
                        r = str(block.get("content", ""))[:300]
                        sub_parts.append(f"[结果: {r}]")
            content = " ".join(sub_parts)
        parts.append(f"{role}: {content}")
    return "\n".join(parts)[:12000]


def _extract_user_feedback(conversation_text: str) -> list[str]:
    """Extract user interrupt/correction messages from conversation text.

    User corrections (from WeChat interrupts) are the single most valuable
    signal for memory extraction — they tell us exactly what the LLM did
    wrong and what it should have done.  The extraction LLM must see these
    prominently, not buried in a 12000-char conversation dump.
    """
    import re as _re
    feedback: list[str] = []

    # Patterns for user interrupt/feedback messages
    patterns = [
        r"\[用户指令\s*[-—]*\s*必须执行\]\s*(.+?)(?=\n|$)",
        r"\[用户指令\]\s*(.+?)(?=\n|$)",
        r"\[用户回复\]\s*(.+?)(?=\n|$)",
    ]

    for pattern in patterns:
        for m in _re.finditer(pattern, conversation_text):
            text = m.group(1).strip()
            if text and len(text) > 2:
                # Skip system-originated messages (not actual user feedback)
                if text.startswith("[系统") or text.startswith("上述操作"):
                    continue
                if text not in feedback:
                    feedback.append(text)

    return feedback


def _extract_learning_signals_from_summary(conversation_text: str) -> list[str]:
    """Parse [USER_CORRECTION]/[FAILURE]/[RESOLUTION] tags from compression summaries.

    When the compressor runs, its enhanced prompt asks the LLM to tag:
      [USER_CORRECTION] — user corrected the agent
      [FAILURE] — agent was stuck in a loop
      [RESOLUTION] — the strategy that finally worked

    These tags survive compression and can be parsed here to feed structured
    learning signals into the extraction payload — ensuring that valuable
    lessons aren't lost even when the full conversation is too long to review.
    """
    import re as _re
    signals: list[str] = []

    # Match compressed summary blocks: [Conversation summary: ...]
    # Use greedy .+ so nested [TAG] brackets inside the summary don't
    # prematurely terminate the match (non-greedy .+? would stop at first ]).
    summary_match = _re.findall(
        r'\[Conversation summary:\s*(.+)\]',
        conversation_text,
        _re.DOTALL,
    )
    if not summary_match:
        return signals

    for summary in summary_match:
        # Extract tagged sections
        for tag in ("USER_CORRECTION", "FAILURE", "RESOLUTION"):
            pattern = rf'\[{tag}\]\s*(.+?)(?=\[|$)'
            for m in _re.findall(pattern, summary, _re.DOTALL):
                text = m.strip()
                if text and len(text) > 5:
                    signals.append(f"[压缩摘要-{tag}] {text}")

    if signals:
        logger.info(
            "Extracted %d learning signals from compression summaries",
            len(signals),
        )

    return signals


# ====================================================================
# Main entry point
# ====================================================================


def spawn_background_review(
    messages: list[dict[str, Any]],
    game: str = "arknights",
    task_description: str = "",
    task_type: str = "unknown",
    screen_w: int = 1080,
    screen_h: int = 1920,
    failure_signals: list[dict[str, Any]] | None = None,
    matched_skill_names: list[str] | None = None,
    iterations: int = 0,
    ask_user_count: int = 0,
    user_msg_count: int = 0,
) -> None:
    """Spawn a daemon thread for background skill + memory extraction.

    Phase 1 (programmatic): Extract the success chain from conversation
    messages and generate a skill file — no LLM involved. Coordinates
    are verified from tool_results.

    Phase 2 (LLM): Multi-mode memory extraction.  The extraction mode is
    chosen dynamically based on task outcome (smooth success / stuck-but-
    recovered / failed).  Two-round extraction with self-verification.

    Phase 3 (skill refinement): If a previously-matched skill's fast_chain
    failed and the LLM manually succeeded, compare old vs new coordinates
    and auto-update the skill file.
    """

    def _run() -> None:
        try:
            logger.info(
                "Background review: %d messages, game=%s, screen=%dx%d, failures=%d",
                len(messages), game, screen_w, screen_h,
                len(failure_signals or []),
            )

            # --- Phase 1: Programmatic skill extraction ---
            sub_chains = _extract_success_chain(
                messages, screen_w, screen_h, game=game,
                matched_skill_names=matched_skill_names,
            )
            task_success = any(c["steps"] for c in sub_chains)
            generated_skill_names: list[str] = []

            # ── Orchestrator guard ──
            _has_orchestrator = False
            _orchestrator_name = ""
            if matched_skill_names:
                from src.skills.manager import get_skill_manager as _bg_skm
                _bg_mgr = _bg_skm(game)
                for _mn in matched_skill_names:
                    _ms = _bg_mgr.load(_mn)
                    if _ms and _ms.get("type") == "orchestrator":
                        _has_orchestrator = True
                        _orchestrator_name = _mn
                        logger.info("Orchestrator '%s' matched — blocking global generation", _mn)
                        break

            _had_correction = user_msg_count > 0

            # ── Orchestrator subskills fallback chain split ──
            # When an orchestrator is matched but _extract_success_chain produced
            # a single giant unattributed chain (steps > 20), the normal split
            # triggers (main screen / subtask_done / adb_back) all failed —
            # typically because:
            #   - the game has no main-screen markers in _MAIN_MARKERS_BY_GAME
            #   - the agent didn't call subtask_done between sub-tasks
            #   - there were no 2+ consecutive adb_back between tasks
            #
            # Fallback: use the orchestrator's subskills list + keyword matching
            # on each step's target text to partition the giant chain into segments,
            # one per sub-skill.  This is the last-resort split — if it succeeds,
            # each sub-skill guide gets coordinates and can be auto-verified.
            if (_has_orchestrator and _orchestrator_name
                    and len(sub_chains) == 1):
                _giant = sub_chains[0]
                _giant_steps = _giant.get("steps", [])
                _attributed_name = _giant.get("skill_name", "")
                _orch = _bg_mgr.load(_orchestrator_name)
                _subskill_names = _orch.get("subskills", []) if _orch else []
                # Trigger when the single chain is too large for one skill.
                # Previously required skill_name="" which skipped the fallback
                # when OCR attribution assigned the giant chain to one sub-skill
                # (e.g. "credit-shop") even though it contained the ENTIRE task run.
                _needs_split = (not _attributed_name
                                or len(_giant_steps) > 20
                                or _attributed_name in _subskill_names)
                if _needs_split and len(_giant_steps) > 20 and len(_subskill_names) >= 2:
                    # Build keyword sets for each sub-skill from its guide body
                    _sub_keywords: dict[str, set[str]] = {}
                    for _sn in _subskill_names:
                        _sk = _bg_mgr.load(_sn)
                        _body = _sk.get("body", "") if _sk else ""
                        _kw = _extract_guide_keywords(_body)
                        if _kw:
                            _sub_keywords[_sn] = _kw
                    if len(_sub_keywords) >= 2:
                        # Assign each step to the best-matching sub-skill
                        _assignments: list[tuple[int, str]] = []  # (step_idx, skill_name)
                        for _si, _step in enumerate(_giant_steps):
                            _best_name = ""
                            _best_score = 0
                            _target = str(_step.get("target", "")).lower()
                            _tool = str(_step.get("tool", ""))
                            # Combine target + tool for better matching
                            _search = f"{_target} {_tool}"
                            for _sn, _kw_set in _sub_keywords.items():
                                _score = sum(1 for _k in _kw_set if _k in _search)
                                if _score > _best_score:
                                    _best_score = _score
                                    _best_name = _sn
                            if _best_name:
                                _assignments.append((_si, _best_name))
                        # Partition: find transition points where assignment changes
                        if _assignments:
                            _segments: list[list[dict]] = []
                            _seg_names: list[str] = []
                            _cur_seg: list[dict] = []
                            _cur_name = _assignments[0][1]
                            for _si, _sn in _assignments:
                                if _sn != _cur_name:
                                    if _cur_seg:
                                        _segments.append(_cur_seg)
                                        _seg_names.append(_cur_name)
                                    _cur_seg = [_giant_steps[_si]]
                                    _cur_name = _sn
                                else:
                                    _cur_seg.append(_giant_steps[_si])
                            if _cur_seg:
                                _segments.append(_cur_seg)
                                _seg_names.append(_cur_name)
                            # Replace the giant chain with split segments
                            if len(_segments) >= 2:
                                logger.info(
                                    "Orchestrator fallback split: %d steps → %d segments (%s)",
                                    len(_giant_steps), len(_segments),
                                    ", ".join(f"{n}({len(s)})" for n, s in zip(_seg_names, _segments)),
                                )
                                sub_chains = [
                                    {"skill_name": name, "steps": steps}
                                    for name, steps in zip(_seg_names, _segments)
                                ]

            # ── Global kill switch: auto-generation produces garbage ──
            # with bad coordinates (loading screens, misclicks) and garbage
            # names (base-Operator-仓库.md etc.). Hand-author skills instead.
            if not config.agent.enable_skill_generation:
                logger.debug("Skill generation disabled (enable_skill_generation=False)")

            for sub in sub_chains if config.agent.enable_skill_generation else []:
                steps = sub.get("steps", [])
                skill_name = sub.get("skill_name", "")

                if not steps:
                    logger.warning(
                        "Sub-chain: empty steps list — chain extraction produced "
                        "no actionable steps (skill_name=%s). This means the "
                        "background_review Phase 1 could not extract a tool chain "
                        "from the conversation.",
                        skill_name or "(none)",
                    )
                    continue

                # ── Attribution fallback: if no skill_run was detected but
                #     exactly one non-orchestrator guide skill is matched,
                #     attribute the chain to that skill. ──
                # Fix 6: orchestrator no longer blocks this fallback.  When
                # daily orchestrator is matched alongside base-collect/credit-shop
                # guides, and the LLM manually executed those guides without
                # calling skill_run(), we still want attribution to work.
                if not skill_name and matched_skill_names:
                    guide_matches = []
                    for mn in matched_skill_names:
                        ms = _bg_mgr.load(mn)
                        if ms and ms.get("type") in ("guide", "script") and not ms.get("verified"):
                            guide_matches.append(mn)
                    if len(guide_matches) == 1:
                        skill_name = guide_matches[0]
                        logger.info("Sub-chain attributed to '%s' (sole matched guide)", skill_name)

                # ── Minimum step gate: at least 3 productive steps to generate ──
                if len(steps) < 3:
                    logger.warning(
                        "Sub-chain too short (%d steps < 3) — skipping", len(steps))
                    continue

                if not skill_name:
                    logger.info(
                        "Sub-chain unattributed (%d steps) — generating new skill via "
                        "_derive_skill_name. Matched skills: %s",
                        len(steps), matched_skill_names,
                    )
                    # Fall through – generate without preferred_name (Fix 7)

                # ── Skip orchestrator-type skills ──
                _attr_skill = _bg_mgr.load(skill_name) if skill_name else None
                if _attr_skill and _attr_skill.get("type") == "orchestrator":
                    continue

                tt = task_type
                if tt == "unknown" and task_description:
                    from src.games.registry import get_game_registry
                    tt = get_game_registry().classify_task(task_description, game_id=game)

                # ── Fix 9: NEVER auto-generate script variants from hand-authored ──
                # guides.  Background review from real runs always captures stale/
                # wrong coordinates (loading screens, misclicks, magnify skew).
                # These auto-generated scripts pollute the skill index, override
                # hand-authored guides via version-bump, and cause the agent to
                # blindly follow broken coordinates instead of reading the guide.
                # Guides are hand-maintained — refinement (Phase 3) handles
                # coordinate updates for existing script files only.
                _attr_skill_type = _attr_skill.get("type", "") if _attr_skill else ""
                if _attr_skill_type == "guide":
                    logger.info(
                        "Skipping script generation for guide '%s' — hand-authored "
                        "guides are never auto-converted to scripts. Refinement only.",
                        skill_name,
                    )
                    continue

                # ── Fix 8: user corrections → still generate, but don't auto-verify ──
                # The chain may still contain correct steps; just be conservative
                # about auto-marking it as verified.
                force_unverified = _had_correction
                if force_unverified:
                    logger.info(
                        "Generating skill for '%s' with verified=false (user corrected "
                        "during task — conservative auto-verification)", skill_name or "(derived)",
                    )

                generated_name, verified = _generate_skill_from_chain(
                    steps, task_description, tt, game,
                    preferred_name=skill_name or "",
                    force_unverified=force_unverified,
                )
                if generated_name:
                    generated_skill_names.append(generated_name)
                logger.info(
                    "Sub-skill '%s' → script '%s' (verified=%s, %d steps)",
                    skill_name, generated_name, verified, len(steps),
                )

            # --- Phase 3: Skill refinement check (per sub-chain) ---
            # For each sub-chain attributed to a known skill, check if the old
            # skill file needs coordinate/structure updates.
            # Guard: disabled by default — produces garbage v2 variants.
            if not config.agent.enable_skill_refinement:
                logger.debug("Skill refinement disabled (enable_skill_refinement=False)")
            elif matched_skill_names and sub_chains and task_success:
                from src.skills.manager import get_skill_manager as _rf_skm
                _rf_mgr = _rf_skm(game)
                from src.tools.skill_refiner import (
                    check_and_refine_skill,
                    auto_refine_from_observation,
                )
                for sub in sub_chains:
                    steps = sub.get("steps", [])
                    s_name = sub.get("skill_name", "")
                    if not steps or not s_name:
                        continue
                    if s_name in generated_skill_names:
                        continue  # Already generated this turn
                    # Only refine existing skill files (not orchestrators).
                    # Verified gate REMOVED: unverified guide skills with
                    # observation data can now be auto-filled via observation.
                    _existing = _rf_mgr.load(s_name)
                    if not _existing or _existing.get("type") == "orchestrator":
                        continue
                    try:
                        result = check_and_refine_skill(
                            skill_name=s_name,
                            new_chain=steps,
                            game=game,
                            screen_w=screen_w,
                            screen_h=screen_h,
                        )
                        if result:
                            logger.info("Skill refined: %s v%d (%s)",
                                       result["skill_name"], result["version"],
                                       result.get("action", "unknown"))
                            if result.get("action") in ("coords_updated", "body_replaced"):
                                try:
                                    obs_result = auto_refine_from_observation(
                                        s_name, game,
                                        screen_w=screen_w, screen_h=screen_h,
                                    )
                                    if obs_result:
                                        logger.info(
                                            "Observation coords patched: %s (%d steps)",
                                            obs_result["skill_name"],
                                            obs_result["patched_steps"],
                                        )
                                except Exception:
                                    pass  # Non-critical
                    except Exception as e:
                        from src.utils.errors import safe_log
                        safe_log(logger, "warning", f"Skill refinement check skipped for '{s_name}': {e}")

            # --- Phase 2: Memory extraction (v3: user-feedback-aware) ---
            # Cost optimization: skip LLM extraction for clean successful tasks.
            # Successful runs produce minimal reusable memory — Phase 1 already
            # captured the tool chain, and the LLM extraction almost always
            # returns empty (parse_ok=False), wasting ~5-7 LLM rounds (~28s).
            _has_failures = bool(failure_signals)
            if task_success and not _has_failures:
                logger.info(
                    "Memory extraction skipped (task succeeded with no failures "
                    "— Phase 1 chain extraction is sufficient)")
                return
            _total_steps = sum(len(c["steps"]) for c in sub_chains)
            mode = _classify_extraction_mode(
                failure_signals=list(failure_signals or []),
                task_success=task_success,
                chain_len=_total_steps,
                iterations=iterations,
                has_user_feedback=user_msg_count > 0,
                ask_user_count=ask_user_count,
            )

            # Pattern mode + skill generated: skip LLM extraction entirely.
            # Smooth tasks produce few reusable memory patterns beyond what
            # Phase 1 already captured as a skill file.  LLM extraction in
            # pattern mode almost always returns empty, wasting ~3 LLM rounds.
            if mode == "pattern" and generated_skill_names:
                logger.info(
                    "Memory extraction skipped (pattern mode, %d skill(s) "
                    "already generated by Phase 1 — LLM has nothing to add)",
                    len(generated_skill_names),
                )
                return

            if mode == "skip":
                logger.debug("Memory extraction skipped (mode=%s, success=%s, failures=%d, steps=%d)",
                            mode, task_success, len(failure_signals or []), _total_steps)
                return

            msg_count = len(messages)
            conv_text = _build_conversation_text(messages)
            bad_examples = _get_bad_examples(game)
            user_feedback = _extract_user_feedback(conv_text)
            # ── Extract learning signals from compressed summaries ──
            # When compression ran during the task, the compressed summary may
            # contain tagged [USER_CORRECTION]/[FAILURE]/[RESOLUTION] markers.
            # Parse these out and feed them as additional user feedback so the
            # extraction LLM can learn from conversations too long to review in full.
            summary_signals = _extract_learning_signals_from_summary(conv_text)
            if summary_signals:
                user_feedback = (user_feedback or []) + summary_signals

            _flat_chain = [s for c in sub_chains for s in c.get("steps", [])]
            payload = _build_extraction_payload(
                failure_signals=list(failure_signals or []),
                success_chain=_flat_chain,
                task_description=task_description,
                task_success=task_success,
                iterations=msg_count,
                conversation_text=conv_text,
                bad_examples=bad_examples,
                user_feedback=user_feedback,
            )
            _run_memory_extract_subagent(
                mode=mode,
                payload=payload,
                failure_signals=list(failure_signals or []),
                game=game,
            )

        except Exception as e:
            logger.warning("Background review failed: %s", e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


# ====================================================================
# Real-time lesson watcher (runs concurrently with agent loop)
# ====================================================================

_WATCH_PROMPT = """你是即时经验提取器。你在后台观察一个游戏自动化 agent 的执行过程。

刚才发生了一件值得记住的事（用户纠正 / agent 卡住循环）。你的任务：**提取一条具体可操作的经验，让 agent 下次遇到同样情况时不再浪费时间。**

## 提取规则

1. 经验必须包含：**画面特征（OCR 文字）** + **错误操作** + **正确做法**
2. 一条经验只绑定一个具体场景
3. 禁止：常识、模糊建议
4. 如果相关操作使用了坐标（adb_tap_position / tap_magnified），请将坐标写入 insight，格式：坐标: pct(x, y) 或 坐标: [x, y]
5. insight 格式必须为：**"当 OCR 包含 [具体OCR文字] 时，不要 [具体错误操作]，应该 [具体正确做法]"**
6. 如果这件事不值得记住（纯一次性故障），回复"无"

{payload}

## 指令

调用 remember(insight, tags) 保存。参数说明：
- insight: 结构化经验，格式严格按照上面规则 4
- tags: 逗号分隔的关键词

如果无价值，回复"无"。不要调用任何工具。"""


def spawn_lesson_watcher(
    state: Any,
    game: str = "arknights",
    loop_guard: Any = None,
) -> None:
    """Spawn a daemon thread that watches the agent loop in real time.

    When the watcher detects a user correction or agent loop, it immediately
    spawns a focused mini-review to extract the lesson — no need to wait
    until the task completes.

    This runs CONCURRENTLY with the agent loop so lessons are captured
    while the context is fresh.
    """
    import time as _time

    last_user_count = state._user_msg_count
    last_loop_count = loop_guard.loop_detection_count if loop_guard else 0
    last_stale_count = loop_guard.stale_screen_count if loop_guard else 0

    def _watch() -> None:
        nonlocal last_user_count, last_loop_count, last_stale_count

        while state.running:
            _time.sleep(8)  # Poll interval — frequent enough, not CPU-heavy

            if not state.running:
                break

            # ── User correction detected ──
            current_user_count = state._user_msg_count
            if current_user_count > last_user_count:
                last_user_count = current_user_count
                # Extract the last few messages for context
                hist = list(state.conversation_history)
                # Find the most recent user-injected message
                recent = _extract_recent_context(hist, tail=8)
                if recent:
                    logger.info("Watcher: user correction detected — extracting lesson")
                    _run_watcher_extraction(
                        trigger="用户纠正",
                        context=recent,
                        game=game,
                    )

            # ── Screen loop detected ──
            current_loop_count = loop_guard.loop_detection_count if loop_guard else 0
            if current_loop_count > last_loop_count:
                last_loop_count = current_loop_count
                hist = list(state.conversation_history)
                recent = _extract_recent_context(hist, tail=10)
                if recent:
                    logger.info("Watcher: loop detection #%d — extracting lesson", current_loop_count)
                    _run_watcher_extraction(
                        trigger="agent 卡住循环",
                        context=recent,
                        game=game,
                    )

            # ── Stale screen detection (settings trap, exit-dialog loop, etc.) ──
            current_stale_count = loop_guard.stale_screen_count if loop_guard else 0
            if current_stale_count > last_stale_count and current_stale_count >= 4:
                last_stale_count = current_stale_count
                hist = list(state.conversation_history)
                recent = _extract_recent_context(hist, tail=12)
                if recent:
                    logger.info("Watcher: stale screen streak=%d — extracting lesson", current_stale_count)
                    _run_watcher_extraction(
                        trigger=f"agent 连续 {current_stale_count} 次操作画面没变（可能陷在弹窗/设置/退出确认循环中）",
                        context=recent,
                        game=game,
                    )

    thread = threading.Thread(target=_watch, daemon=True)
    thread.start()
    logger.debug("Lesson watcher started for game=%s", game)


def _extract_recent_context(
    history: list[dict[str, Any]],
    tail: int = 8,
) -> str:
    """Extract the last N messages as a compact text summary for the watcher."""
    recent = history[-tail:] if len(history) > tail else history
    parts: list[str] = []
    for m in recent:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            sub: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        sub.append(str(block.get("text", ""))[:200])
                    elif block.get("type") == "tool_use":
                        sub.append(f"[调用 {block.get('name', '?')}]")
                    elif block.get("type") == "tool_result":
                        r = str(block.get("content", ""))[:150]
                        sub.append(f"[结果: {r}]")
            content = " ".join(sub)
        elif isinstance(content, str):
            content = content[:300]
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _run_watcher_extraction(
    trigger: str,
    context: str,
    game: str,
) -> None:
    """Run a focused single-round LLM extraction for a specific event.

    Much lighter than the full two-round background review — no candidate
    identification, no self-verification. Just one LLM call → remember().
    """
    from src.llm.client import MiMoClient, extract_text, extract_tool_calls

    payload = f"## 触发事件\n{trigger}\n\n## 最近的对话记录\n{context}\n"
    system = _WATCH_PROMPT.format(payload=payload)

    # Build a minimal registry with just remember()
    from src.tools.registry import ToolRegistry
    registry = ToolRegistry()

    def _remember(insight: str, tags: str = "", screen_hash: str = "") -> Any:
        from src.tools.remember import remember_tool
        result = remember_tool(insight=insight, tags=tags, screen_hash=screen_hash, game=game)
        try:
            data = json.loads(result.text)
            name = data.get("name", "")
            if data.get("success") and name:
                logger.info("Watcher extracted memory: %s — %s", name, insight[:100])
        except json.JSONDecodeError:
            pass
        return result

    registry.register(
        name="remember",
        description="Save a concrete lesson: what screen, what went wrong, correct action.",
        parameters={
            "type": "object",
            "properties": {
                "insight": {
                    "type": "string",
                    "description": "Structured lesson: screen OCR + wrong action + correct action",
                },
                "tags": {
                    "type": "string",
                    "description": "Comma-separated keywords",
                },
                "screen_hash": {
                    "type": "string",
                    "description": "Screen dHash hex (from failure signal or OCR context)",
                },
            },
            "required": ["insight"],
        },
        handler=_remember,
    )

    from src.llm.client import pooled_client
    try:
        with pooled_client() as client:
            response = client.chat(
                system=system,
                messages=[{"role": "user", "content": context}],
                tools=registry.get_definitions(),
                max_tokens=512,
            )
            text = extract_text(response)
            tool_calls = extract_tool_calls(response)

            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "")
                    if name == "remember":
                        try:
                            out = registry.dispatch(name, **tc.get("input", {}))
                            logger.info("Watcher lesson saved: %s", out.text[:150])
                        except Exception as e:
                            logger.warning("Watcher remember failed: %s", e)
            elif text:
                logger.debug("Watcher: no lesson extracted — %s", text[:100])
    except Exception as e:
        logger.warning("Watcher extraction LLM failed: %s", e)


def _run_failure_signal_fallback_extraction(
    client: Any,
    failure_signals: list[dict[str, Any]],
    tool_whitelist: list[str],
    base_system: str,
    newly_created: list[str],
    game: str = "arknights",
) -> None:
    """Simplified extraction from failure signals alone — no conversation dump.

    Used as a fallback when Round 1 returns empty (huge conversation overwhelms
    the LLM).  Failure signals carry screen OCR + dHash + tool context which
    is often sufficient to identify trap patterns like:
    - adb_back → exit dialog → X → settings → back → exit dialog → ...
    - stale screen: 4+ actions with no visual change
    - repeat stuck: same action 3+ times

    Single LLM call with tools, no multi-round complexity.
    """
    from src.llm.client import extract_text as _et, extract_tool_calls as _etc
    from src.tools.registry import ToolRegistry

    # Build compact failure signal summary
    lines = ["## 任务执行中检测到的卡住事件\n"]
    # Deduplicate: group similar signals
    seen_types: dict[str, int] = {}
    for sig in failure_signals:
        st = sig.get("signal_type", "unknown")
        seen_types[st] = seen_types.get(st, 0) + 1

    lines.append("信号类型汇总: " + ", ".join(
        f"{t}×{c}" for t, c in seen_types.items()
    ))
    lines.append("")

    for i, sig in enumerate(failure_signals[:12]):  # Max 12 signals
        lines.append(f"### 事件 {i+1}: {sig.get('signal_type', '?')}")
        lines.append(f"- 回合: {sig.get('iteration', '?')}")
        lines.append(f"- 工具: {sig.get('tool_name', '?')} {sig.get('tool_input', {})}")
        lines.append(f"- 详情: {sig.get('detail', '')}")
        ocr = sig.get('ocr_texts', [])
        if ocr:
            lines.append(f"- 画面 OCR: {', '.join(ocr[:15])}")
        dhash = sig.get('screen_dhash', '')
        if dhash:
            lines.append(f"- 画面 dHash: {dhash}")
        lines.append("")

    fallback_payload = "\n".join(lines)
    fallback_sys = (
        base_system
        + "\n\n【简化模式】以上是系统检测到的卡住事件，不包含完整对话记录。"
        "从这些信号中找出反复出现的陷阱模式（如设置→返回→退出确认→返回→设置的循环），"
        "为每种陷阱调用 remember() 保存一条可操作的经验。"
        "格式：当 OCR 包含 [X] 时，不要 [Y]，应该 [Z]。"
        f"\n\n{fallback_payload}"
    )

    # Build minimal tool registry
    restricted = ToolRegistry()
    if "remember" in tool_whitelist:
        from src.tools.remember import remember_tool
        # Wrap to inject the correct game — fallback runs on background thread
        # where _terra_agent_ctx is NOT set.
        _game = game
        def _remember_fallback(insight: str, tags: str = "", screen_hash: str = "") -> Any:
            return remember_tool(insight=insight, tags=tags, screen_hash=screen_hash, game=_game)
        restricted.register(
            name="remember",
            description="Save a pitfall lesson from failure signals.",
            parameters={
                "type": "object",
                "properties": {
                    "insight": {
                        "type": "string",
                        "description": "Structured lesson: screen OCR + wrong action + correct action",
                    },
                    "tags": {"type": "string", "description": "Comma-separated keywords"},
                    "screen_hash": {"type": "string", "description": "dHash hex from failure signal"},
                },
                "required": ["insight"],
            },
            handler=_remember_fallback,
        )

    try:
        response = client.chat(
            system=fallback_sys,
            messages=[{"role": "user", "content": fallback_payload}],
            tools=restricted.get_definitions(),
            max_tokens=1024,
        )
    except Exception as e:
        logger.warning("Failure-signal fallback LLM call failed: %s", e)
        return

    text = _et(response)
    tool_calls = _etc(response)

    if tool_calls:
        for tc in tool_calls:
            name = tc.get("name", "")
            if name in tool_whitelist:
                try:
                    out = restricted.dispatch(name, **tc.get("input", {}))
                    logger.info("Fallback memory extracted via %s: %s", name, out.text[:200])
                    try:
                        data = json.loads(out.text)
                        if data.get("success") and data.get("name"):
                            newly_created.append(data["name"])
                    except json.JSONDecodeError:
                        pass
                except Exception as e:
                    logger.warning("Fallback tool '%s' failed: %s", name, e)
    elif text:
        logger.info("Fallback extraction: LLM returned text but no tool calls: %s", text[:200])
    else:
        logger.info("Fallback extraction: no memories extracted from failure signals")


# ====================================================================
# Phase 2: Memory extraction sub-agent (LLM-based, v2)
# ====================================================================


def _run_memory_extract_subagent(
    mode: str,
    payload: str,
    failure_signals: list[dict[str, Any]],
    game: str = "arknights",
) -> None:
    """Multi-mode, two-round memory extraction sub-agent (v3).

    Round 1 (no tools): LLM analyzes the full conversation context and
    outputs structured JSON candidates.  Robust parsing with fallbacks.

    Round 2 (with tools): LLM performs a relaxed self-verification check
    (2/3 YES passes, not 3/3), then calls remember() / learn_action_pattern().

    Fallback (Round 3): If Round 2 produces zero tool calls despite having
    candidates, retry once with a simplified single-round prompt.

    The mode controls:
    - Which system prompt is used (pattern / hybrid / pitfall)
    - Which tools are available in Round 2

    After extraction, auto-fills screen_hash on any memory where the LLM
    forgot to pass it explicitly.
    """
    from src.llm.client import MiMoClient, extract_text, extract_tool_calls
    from src.tools.registry import ToolRegistry

    # Look up mode config
    sys_prompt, tool_whitelist = _EXTRACTION_MODE_CONFIG.get(
        mode, (_EXTRACT_SYS_HYBRID, ["remember", "learn_action_pattern"])
    )
    system = sys_prompt.format(payload=payload)

    logger.info("Memory extraction: mode=%s, tools=%s, failures=%d",
               mode, tool_whitelist, len(failure_signals))

    newly_created: list[str] = []

    # ---- Tool wrappers ----

    def _remember_from_extraction(insight: str, tags: str = "", screen_hash: str = "") -> Any:
        from src.tools.remember import remember_tool
        # Drop obviously bad hashes before they hit the DB
        clean_hash = screen_hash if _is_valid_screen_hash(screen_hash) else ""
        result = remember_tool(insight=insight, tags=tags, screen_hash=clean_hash, game=game)
        try:
            data = json.loads(result.text)
            if data.get("success") and data.get("name"):
                newly_created.append(data["name"])
        except json.JSONDecodeError:
            pass
        return result

    def _learn_ap_from_extraction(trigger_screen: str, action: str,
                                   expected_result: str = "", tags: str = "") -> Any:
        from src.tools.learn import learn_action_pattern_tool
        result = learn_action_pattern_tool(
            trigger_screen=trigger_screen,
            action=action,
            expected_result=expected_result,
            tags=tags,
            game=game,
        )
        try:
            data = json.loads(result.text)
            if data.get("success") and data.get("name"):
                newly_created.append(data["name"])
        except json.JSONDecodeError:
            pass
        return result

    # ---- Build restricted registry ----
    restricted = ToolRegistry()
    if "remember" in tool_whitelist:
        restricted.register(
            name="remember",
            description="Save a pitfall lesson: what screen, what went wrong, what to do instead.",
            parameters={
                "type": "object",
                "properties": {
                    "insight": {
                        "type": "string",
                        "description": "MUST include: (1) screen OCR texts visible, (2) what action failed, (3) what worked instead or what to try next time",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated keywords (e.g. 'EP, 乐章, 网格')",
                    },
                    "screen_hash": {
                        "type": "string",
                        "description": "dHash hex from failure signal's '画面 dHash' field. Required.",
                    },
                },
                "required": ["insight"],
            },
            handler=_remember_from_extraction,
        )
    if "learn_action_pattern" in tool_whitelist:
        restricted.register(
            name="learn_action_pattern",
            description="Save a positive pattern: 'when screen shows [X], do [Y]'.",
            parameters={
                "type": "object",
                "properties": {
                    "trigger_screen": {
                        "type": "string",
                        "description": "Screen description with visible OCR texts — when you see this, act",
                    },
                    "action": {
                        "type": "string",
                        "description": "Specific action: tool name + target + key params",
                    },
                    "expected_result": {
                        "type": "string",
                        "description": "What should happen after the action (helps detect staleness)",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated keywords",
                    },
                },
                "required": ["trigger_screen", "action"],
            },
            handler=_learn_ap_from_extraction,
        )

    tools = restricted.get_definitions()
    # Phase 4 note: this function has multiple sequential LLM calls sharing
    # one client; pooled_client() context manager not used here to avoid
    # re-indenting 190 lines. Client is safely closed in all exit paths.
    client = MiMoClient()
    messages: list[dict[str, Any]] = [{"role": "user", "content": payload}]

    # ================================================================
    # Round 1: Candidate identification (no tools)
    # ================================================================
    round1_sys = (
        system
        + "\n\n【第一轮】先仔细分析对话记录和失败信号，不要调用任何工具。"
        "找出所有值得保存的经验。输出一个 JSON 数组，每个元素为：\n"
        '{{"type": "pitfall"|"pattern", "trigger_screen": "...", '
        '"wrong_action": "...", "correct_action": "...", '
        '"screen_ocr": "...", "confidence": 0.X, "reasoning": "..."}}\n'
        "如果没有任何有价值的经验，输出空数组 []。只输出 JSON，不要前言或解释。"
    )

    try:
        response_1 = client.chat(
            system=round1_sys,
            messages=messages,
            tools=[],  # No tools in Round 1
            max_tokens=1024,
        )
    except Exception as e:
        logger.warning("Memory extract Round 1 LLM call failed: %s", e)
        client.close()
        return

    text_1 = extract_text(response_1)
    # Fallback: if extract_text returned empty but the LLM DID produce output
    # (e.g. all content was in thinking blocks), try to parse all text-like
    # blocks from the raw response.
    if not text_1.strip():
        for block in getattr(response_1, 'content', []) or []:
            if hasattr(block, 'text') and block.text:
                text_1 += block.text
        # Also try thinking blocks
        if not text_1.strip():
            for block in getattr(response_1, 'content', []) or []:
                if hasattr(block, 'thinking') and block.thinking:
                    text_1 += block.thinking
    logger.debug("Memory extract Round 1: %s", text_1[:500] if text_1 else "(empty)")

    # ── Robust JSON parsing (v3) ──────────────────────────────────────
    candidates: list[dict] = []
    parse_ok = False

    # Strategy 1: Greedy match the outermost JSON array
    match = re.search(r"\[[\s\S]*\]", text_1)
    if match:
        try:
            candidates = json.loads(match.group())
            parse_ok = True
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 2: Extract individual JSON objects {…} if array parse failed
    if not parse_ok:
        obj_matches = re.findall(r"\{[^{}]*\}", text_1)
        for om in obj_matches:
            try:
                obj = json.loads(om)
                if isinstance(obj, dict) and "type" in obj:
                    candidates.append(obj)
                    parse_ok = True
            except (json.JSONDecodeError, TypeError):
                pass

    # Strategy 3: If still nothing, attempt to fix trailing comma / unclosed
    if not parse_ok and "[" in text_1:
        # Try to find the last complete object before any truncation
        try:
            fixed = text_1[text_1.index("["):]
            fixed = re.sub(r",\s*\]", "]", fixed)  # Remove trailing comma
            # Find last complete }, then close the array
            last_obj_end = fixed.rfind("}")
            if last_obj_end > 0:
                fixed = fixed[:last_obj_end + 1] + "]"
                candidates = json.loads(fixed)
                parse_ok = True
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if not candidates:
        logger.info(
            "Memory extraction: no candidates found in Round 1 (parse_ok=%s, text_len=%d)",
            parse_ok, len(text_1),
        )

        # ── Fallback: if we have failure signals with OCR data, feed them
        #     directly to a simplified single-round extraction.  Round 1 can
        #     fail when the conversation is huge and the LLM gets overwhelmed.
        #     The failure signals alone often contain enough OCR + dHash data
        #     to produce useful memories.
        if failure_signals and len(failure_signals) >= 2:
            logger.info(
                "Memory extraction: Round 1 empty, falling back to failure-signal-"
                "only extraction (%d signals)", len(failure_signals),
            )
            _run_failure_signal_fallback_extraction(
                client, failure_signals, tool_whitelist, system, newly_created, game,
            )

        client.close()
        return

    logger.info("Memory extraction: %d candidates identified, starting Round 2", len(candidates))

    # ================================================================
    # Round 2: Relaxed self-verification + tool calls (v3)
    # ================================================================
    # Relaxed from "all 3 must be YES" to "at least 2 of 3 YES passes".
    # This prevents the LLM from being overly conservative and producing
    # zero memories when valid patterns exist.
    round2_sys = (
        system
        + "\n\n【第二轮】对每条候选做 3 问自检（通过 ≥2/3 即可写入，不需要全YES）：\n"
        "1. 记忆描述的画面特征和 OCR 文字是否与对话记录一致？\n"
        "2. 如果把这个经验注入到原对话的失败位置，agent 是否真的能避免错误？\n"
        "3. 这个经验是否可以泛化到其他类似画面（而非一次性细节）？\n\n"
        "通过 ≥2 条自检的候选，调用工具写入。同画面+同操作的候选合并为一条。\n"
        "⚠️ 重要：不要过度保守。即使对泛化性有疑虑，只要前2问是YES就应该写入。"
    )

    messages.append({"role": "assistant", "content": text_1})
    messages.append({
        "role": "user",
        "content": (
            f"已识别 {len(candidates)} 个候选。现在对每条做自检（≥2/3 即可写入），"
            f"通过的调用 {'remember' if 'remember' in tool_whitelist else ''}"
            f"{' 或 ' if len(tool_whitelist) > 1 else ''}"
            f"{'learn_action_pattern' if 'learn_action_pattern' in tool_whitelist else ''}"
            f" 写入。不通过的跳过。"
        ),
    })

    def _run_extraction_round(sys_msg: str, msgs: list[dict], tools_list: list[dict],
                              max_tok: int) -> tuple[str, list[dict]]:
        """Run one extraction LLM call. Returns (text, tool_calls)."""
        try:
            resp = client.chat(
                system=sys_msg,
                messages=msgs,
                tools=tools_list,
                max_tokens=max_tok,
            )
            return extract_text(resp), extract_tool_calls(resp)
        except Exception as e:
            logger.warning("Memory extract LLM call failed: %s", e)
            return "", []

    text_2, tool_calls = _run_extraction_round(round2_sys, messages, tools, 2048)

    # ── Fallback (Round 3): If Round 2 produced 0 tool calls despite having
    #     candidates, retry ONCE with a simplified single-round prompt that
    #     skips the self-verification step and directly asks the LLM to write.
    # ──
    if not tool_calls and candidates:
        logger.info(
            "Memory extraction Round 2 produced 0 tool calls — retrying with "
            "simplified fallback prompt (%d candidates)",
            len(candidates),
        )
        fallback_sys = (
            system
            + "\n\n【回退模式】第二轮过于保守未写入任何经验。现在直接写入："
            "将所有值得保存的候选（画面特征清晰、可操作、能帮到未来执行）"
            "直接调用工具写入。只要不是纯常识或一次性细节就写入。"
            f"\n候选列表：{json.dumps(candidates, ensure_ascii=False)[:2000]}"
        )
        text_2, tool_calls = _run_extraction_round(
            fallback_sys, messages, tools, 2048,
        )

    if not tool_calls:
        logger.info("Memory extraction complete (no tools called): %s", text_2[:200])
    else:
        # Dispatch tool calls
        for tc in tool_calls:
            name = tc.get("name", "")
            if name in tool_whitelist:
                try:
                    out = restricted.dispatch(name, **tc.get("input", {}))
                    logger.info("Memory extracted via %s: %s", name, out.text[:200])
                except Exception as e:
                    logger.warning("Memory extract tool '%s' failed: %s", name, e)

    # ---- Post-process: auto-fill missing screen_hash ----
    if newly_created:
        if failure_signals:
            _auto_fill_screen_hashes(newly_created, failure_signals)
        else:
            # Pattern mode: try to backfill screen_hash from the success chain
            # steps embedded in the payload. Each step description includes OCR
            # keywords that we can match against the memory body.
            _auto_fill_screen_hashes_from_payload(newly_created, payload)

    # ---- Post-process: backfill confidence from Round 1 candidates ----
    if newly_created and candidates:
        _backfill_confidence(newly_created, candidates)

    # ---- Record extraction stats ----
    try:
        from src.memory.memory_db import memory_db
        memory_db.set_learning_state(
            "last_extraction_stats",
            json.dumps({
                "mode": mode,
                "candidates": len(candidates),
                "created": len(newly_created),
            }),
        )
    except Exception:
        from src.utils.errors import safe_log
        safe_log(logger, "warning", "Failed to save extraction stats")

    logger.info("Memory extraction done: mode=%s, candidates=%d, created=%d",
               mode, len(candidates), len(newly_created))
    client.close()


def _backfill_confidence(
    memory_names: list[str],
    candidates: list[dict],
) -> None:
    """Backfill confidence scores from Round 1 candidates to newly created memories.

    The extraction LLM self-reports confidence (0.0-1.0) in Round 1 candidate
    JSON.  We match by CJK bigram + word token overlap—same approach as
    _check_duplicate_memory—because Chinese text lacks whitespace word boundaries.
    """
    from src.memory.memory_db import memory_db as _mdb

    def _tokenize(text: str) -> set[str]:
        """Extract CJK bigrams and alphanumeric tokens for fuzzy matching."""
        tokens: set[str] = set()
        # Alphanumeric tokens >= 2 chars
        for token in re.findall(r'[a-zA-Z0-9]{2,}', text.lower()):
            tokens.add(token)
        # CJK bigrams
        cjk = re.sub(r'[a-zA-Z0-9\s,，。.、：:；;！!？?()（）\[\]【】]', '', text)
        for i in range(len(cjk) - 1):
            tokens.add(cjk[i:i+2])
        return tokens

    try:
        conn = _mdb.conn
        for mem_name in memory_names:
            row = conn.execute(
                "SELECT id, body, confidence FROM memories_data WHERE name = ?",
                (mem_name,),
            ).fetchone()
            if not row:
                continue
            if row["confidence"] is not None:
                continue  # Already has confidence (set by tool caller)

            mem_body = row["body"] or ""
            if not mem_body:
                continue
            mem_tokens = _tokenize(mem_body)
            if not mem_tokens:
                continue

            # Find the best-matching candidate by token overlap
            best_conf = None
            best_hits = 0
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                c_body = str(c.get("correct_action", "")) + " " + str(c.get("wrong_action", ""))
                c_trigger = c.get("trigger_screen", c.get("screen_ocr", ""))
                c_body += " " + str(c_trigger)
                c_tokens = _tokenize(c_body)
                if not c_tokens:
                    continue
                hits = len(mem_tokens & c_tokens)
                if hits > best_hits:
                    best_hits = hits
                    conf = c.get("confidence")
                    if isinstance(conf, (int, float)) and 0 <= conf <= 1:
                        best_conf = conf

            if best_conf is not None and best_hits >= 3:
                conn.execute(
                    "UPDATE memories_data SET confidence = ? WHERE id = ?",
                    (best_conf, row["id"]),
                )
                conn.commit()
                logger.debug(
                    "Backfilled confidence=%.2f for memory '%s' (%d token hits)",
                    best_conf, mem_name, best_hits,
                )
    except Exception as e:
        from src.utils.errors import safe_log
        safe_log(logger, "warning", f"Confidence backfill failed: {e}")


def _is_valid_screen_hash(hash_str: str) -> bool:
    """Reject obviously bad dHash values that will break visual matching.

    A valid dHash is a 16-char hex string with reasonable entropy; bad hashes
    include all-same-nibble patterns (0000..., ffff...), repeating byte pairs
    (6969..., aaaa...), or recognizable ASCII text masquerading as hex
    (e.g. 696961696961614d decodes to 'iiaiaiaM').

    Real dHashes can legitimately have bytes in the printable ASCII range
    (e.g. f655226c5a6051f0 has 'Ul\"lZ`Q'), but they always contain at least
    some non-printable bytes (> 0x7E or < 0x20).  A dHash composed entirely
    of standard ASCII letters/numbers is text corruption, not a real hash.
    """
    if not hash_str or len(hash_str) != 16:
        return False
    # Must be valid hex
    try:
        int(hash_str, 16)
    except (ValueError, TypeError):
        return False
    # Reject all-same-nibble (no entropy — all 0s, all Fs, all As, etc.)
    if len(set(hash_str)) <= 2:
        return False
    # Reject 2-byte repeating patterns: 6969..., aaaa..., etc.
    byte_pairs = [hash_str[i:i+2] for i in range(0, 16, 2)]
    if len(set(byte_pairs)) <= 2:
        return False
    # Reject hashes that look like ASCII text: ALL 8 bytes fall in the
    # printable ASCII range (0x20-0x7E).  Real dHashes always have some
    # non-printable bytes; 8/8 printable means it's text corruption.
    printable_count = sum(
        1 for i in range(0, 16, 2)
        if 0x20 <= int(hash_str[i:i+2], 16) <= 0x7E
    )
    if printable_count >= 8:
        return False
    return True


def _auto_fill_screen_hashes(
    memory_names: list[str],
    failure_signals: list[dict[str, Any]],
) -> None:
    """For each new memory without a screen_hash, find the best-matching failure
    signal via OCR-word substring matching and update the memory.

    This guarantees visual anchoring even when the LLM ignores the
    screen_hash parameter in the remember tool call.
    """
    from src.memory.memory_db import memory_db as _mdb

    try:
        conn = _mdb.conn
        for mem_name in memory_names:
            row = conn.execute(
                "SELECT id, body, screen_hash FROM memories_data WHERE name = ?",
                (mem_name,),
            ).fetchone()
            if not row:
                continue
            if row["screen_hash"]:
                continue

            mem_body = row["body"] or ""
            if not mem_body:
                continue

            best_signal = None
            best_hits = 0
            for sig in failure_signals:
                sig_dhash = sig.get("screen_dhash", "")
                if not sig_dhash:
                    continue
                ocr_texts = sig.get("ocr_texts", [])
                if not ocr_texts:
                    continue
                hits = sum(1 for word in ocr_texts if len(word) >= 2 and word in mem_body)
                if hits > best_hits:
                    best_hits = hits
                    best_signal = sig

            if best_signal and best_hits >= 2:
                dhash = best_signal["screen_dhash"]
                if not _is_valid_screen_hash(dhash):
                    logger.debug(
                        "Skipping bad screen_hash '%s' for memory '%s'",
                        dhash, mem_name,
                    )
                    continue
                conn.execute(
                    "UPDATE memories_data SET screen_hash = ? WHERE id = ?",
                    (dhash, row["id"]),
                )
                conn.commit()
                logger.info(
                    "Auto-filled screen_hash for memory '%s' from failure signal "
                    "(%d OCR word hits, dhash=%s)",
                    mem_name, best_hits, dhash,
                )
    except Exception as e:
        logger.warning("Auto-fill screen_hash failed: %s", e)


def _auto_fill_screen_hashes_from_payload(
    memory_names: list[str],
    payload: str,
) -> None:
    """Pattern-mode variant: extract screen hashes from conversation transcript
    embedded in the payload and match against new memories via OCR word overlap.

    Pattern-mode tasks have no failure_signals, so the standard auto-fill
    can't run.  Instead we parse HASH: markers and OCR: labels from the
    conversation text inside the payload, then match against memory bodies
    the same way.
    """
    from src.memory.memory_db import memory_db as _mdb

    # Extract (hash, ocr_words) pairs from payload conversation text
    screen_entries: list[tuple[str, set[str]]] = []
    # Match each HASH:… OCR:… block
    hash_positions = [m.start() for m in re.finditer(r"HASH:([0-9a-fA-F]{16})", payload)]
    for hp in hash_positions:
        hash_match = re.match(r"HASH:([0-9a-fA-F]{16})", payload[hp:])
        if not hash_match:
            continue
        dhash = hash_match.group(1)
        # Look for OCR: text within the next 300 chars after HASH:
        tail = payload[hp + hash_match.end():hp + hash_match.end() + 300]
        ocr_match = re.search(r"OCR:([\w一-鿿, /-]+)", tail)
        if ocr_match:
            words = {w.strip() for w in ocr_match.group(1).split(",") if len(w.strip()) >= 2}
        else:
            words = set()
        screen_entries.append((dhash, words))

    if not screen_entries:
        logger.debug("No HASH:OCR: pairs found in payload — cannot auto-fill screen_hash")
        return

    try:
        conn = _mdb.conn
        for mem_name in memory_names:
            row = conn.execute(
                "SELECT id, body, screen_hash FROM memories_data WHERE name = ?",
                (mem_name,),
            ).fetchone()
            if not row:
                continue
            if row["screen_hash"]:
                continue

            mem_body = row["body"] or ""
            if not mem_body:
                continue

            best_hash = None
            best_hits = 0
            for dhash, words in screen_entries:
                if not words:
                    continue
                hits = sum(1 for w in words if w in mem_body)
                if hits > best_hits:
                    best_hits = hits
                    best_hash = dhash

            if best_hash and best_hits >= 2:
                if not _is_valid_screen_hash(best_hash):
                    logger.debug(
                        "Skipping bad screen_hash '%s' for pattern-mode memory '%s'",
                        best_hash, mem_name,
                    )
                    continue
                conn.execute(
                    "UPDATE memories_data SET screen_hash = ? WHERE id = ?",
                    (best_hash, row["id"]),
                )
                conn.commit()
                logger.info(
                    "Auto-filled screen_hash for pattern-mode memory '%s' from payload "
                    "(%d OCR word hits, dhash=%s)",
                    mem_name, best_hits, best_hash,
                )
    except Exception as e:
        logger.warning("Auto-fill screen_hash from payload failed: %s", e)
