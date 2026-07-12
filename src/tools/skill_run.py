"""Skill tools: skill_run, skill_list.

Self-registers with the tool registry at import time.
"""

from __future__ import annotations

import json
import logging

from src.tools.registry import registry, ToolOutput

logger = logging.getLogger(__name__)

def skill_run(name: str) -> ToolOutput:
    """Execute a verified skill with fast chain (direct ADB + dHash polling).

    Runs all steps without LLM calls. Calls skill_run('base-collect') only
    AFTER navigating to the main screen. The skill's first step assumes you
    start from the main screen.

    Returns success/failure; on failure, LLM should re-enter the loop.
    """
    from src.skills.manager import get_skill_manager
    from src.tools.registry import get_current_game
    from src.tools.fast_chain import parse_skill_steps, execute_fast_chain
    from src.device.adb import get_adb

    game = get_current_game()
    skill = get_skill_manager(game).load(name)
    if skill is None:
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": f"技能'{name}'不存在。可用技能见skill_list。",
        }, ensure_ascii=False))

    body = skill.get("body", "")
    steps = parse_skill_steps(body)
    if not steps:
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": f"技能'{name}'没有可执行步骤。",
        }, ensure_ascii=False))

    has_coords = any(s.get("coords") for s in steps)
    if not has_coords:
        # ── Auto-redirect to dedicated tool if one exists ──────────────
        # Many sub-skills now have dedicated deterministic tools
        # (e.g. base-collect → base_collect).  When skill_run() is
        # called for such a skill, redirect to the tool automatically
        # instead of returning the skill text for manual execution.
        # This is a safety net — the LLM should call the tool directly,
        # but if it calls skill_run instead, it still works.
        tool_name = name.replace("-", "_")
        tool_entry = registry.get(tool_name)
        if tool_entry is not None:
            logger.info("skill_run: '%s' → auto-redirecting to %s()", name, tool_name)
            try:
                result = tool_entry.handler()
                if isinstance(result, ToolOutput):
                    return result
                return ToolOutput(text=result)
            except Exception as e:
                logger.warning("skill_run: '%s' → %s() redirect failed: %s", name, tool_name, e)
                # Fall through to manual execution below

        # Re-inject full Steps + Pitfalls so the LLM has the instructions
        # right in front of it instead of relying on stale system prompt memory.
        # Also inject as a protected conversation message so the instructions
        # survive clean_subtask_history — otherwise the LLM forgets everything
        # as soon as it calls subtask_done after the first step.
        pitfalls = skill.get("pitfalls", [])
        pitfalls_text = "\n".join(f"  - {p}" for p in pitfalls) if pitfalls else "（无）"
        import threading, re
        ctx = getattr(threading.current_thread(), '_terra_agent_ctx', None)
        if ctx is not None:
            # Compact 1-line hint so the LLM doesn't skim past it.
            # Long skill bodies get ignored; a single actionable line sticks.
            _coord = re.search(r'adb_tap_position\(([\d.]+),\s*([\d.]+)\)', body)
            if name == "annihilation":
                _hint = ("[系统提示 技能：annihilation] adb_tap_position(0.88,0.23)进终端→页面固定模块里找[合成玉]"
                         "（没有=已打完subtask_done skip），别滑动轮播找剿灭字！")
            elif name == "farm-1-7":
                _hint = ("[系统提示 技能：farm-1-7] 第一步：adb_tap_position(0.88, 0.23)进终端"
                         " → 右下角「上次作战」→ 不是1-7才去曲谱找 → 代理×6开打")
            elif _coord:
                _hint = (f"[系统提示 技能：{name}] 入口坐标=adb_tap_position({_coord.group(1)}, {_coord.group(2)})"
                         f" — 不要找其他入口，直接用这个坐标！")
            elif "终端" in body and "右侧面板顶部" in body:
                _hint = f"[系统提示 技能：{name}] 终端在右侧面板顶部(0.88,0.23)，别找底部导航"
            else:
                _first = body.strip().split('\n')[0].strip()
                _hint = f"[系统提示 技能：{name}] {_first}"
            ctx.state.add_message("user", _hint)
        return ToolOutput(text=json.dumps({
            "success": "manual",
            "message": (
                f"技能'{name}'没有坐标（verified=false），快速链无法自动执行。"
                f"⚠️ 请按照以下步骤手动操作，不可跳过！\n"
                f"\n--- {name} 操作步骤 ---\n"
                f"{body.strip()}\n"
                f"--- 避坑警告 ---\n"
                f"{pitfalls_text}"
            ),
        }, ensure_ascii=False))

    logger.info("skill_run: executing '%s' (%d steps)", name, len(steps))

    # Capture the last known screen hash from fast_chain so we can sync it
    # back to the agent's state when execution completes.
    _last_known_hash: list[str | None] = [None]

    def _inject(fast=False, known_hash=""):
        if known_hash:
            _last_known_hash[0] = known_hash

    try:
        adb = get_adb()
        w, h = adb.get_screen_size()
        ok, msg = execute_fast_chain(steps, _inject, w, h)
    except Exception as e:
        logger.warning("skill_run crashed: %s", e)
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": f"技能'{name}'执行异常: {e}",
        }, ensure_ascii=False))

    # Sync the final screen hash back to the agent's state so the loop
    # knows the screen changed (avoids stale-screen false positive).
    # Also record fast_chain success/failure for execution log feedback.
    if _last_known_hash[0]:
        import threading
        ctx = getattr(threading.current_thread(), '_terra_agent_ctx', None)
        if ctx is not None:
            ctx.state.last_injected_hash = _last_known_hash[0]

    # Record fast_chain result in AgentState so execution_logger can write
    # skill_fast_chain_success to task_executions. This unblocks
    # SkillStalenessCheck and PatternMiner stale skill detection.
    # Fix 1: Only record when the skill actually HAD coordinates.  Without
    # this guard, skill_run() on an unverified guide skill (no coordinates)
    # leaks False into the per-skill fast_chain success metric, polluting
    # SkillStalenessCheck for every matched skill in the comma-separated
    # skill_name list of the task_executions record.
    import threading as _t
    _agent_ctx = getattr(_t.current_thread(), '_terra_agent_ctx', None)
    if _agent_ctx is not None and has_coords:
        _agent_ctx.state.skill_fast_chain_result = ok

    if ok:
        return ToolOutput(
            text=json.dumps({
                "success": True,
                "skill_name": name,
                "message": f"技能'{name}'执行完成。{msg}",
            }, ensure_ascii=False),
            task_done=False,
        )
    else:
        # ---- Phase 3: Mark skill as potentially stale ----
        # Guard: disabled by default — auto-stale marking feeds the broken
        # skill_refiner pipeline and produces garbage v2 variants.
        from config.settings import config as _cfg
        if _cfg.agent.enable_skill_refinement and ("coordinates" in msg.lower() or "stale" in msg.lower() or "screen change" in msg.lower()):
            try:
                from src.tools.skill_refiner import mark_skill_potentially_stale
                mark_skill_potentially_stale(name, game=game)
                logger.info("Skill '%s' marked for refinement (fast_chain failure)", name)
            except Exception as e:
                logger.debug("Failed to mark skill as stale: %s", e)

        # Re-inject full Steps + Pitfalls so the LLM can fall back to
        # manual execution with complete instructions — critical for
        # verified script skills whose body is hidden from the system prompt.
        pitfalls = skill.get("pitfalls", [])
        pitfalls_text = "\n".join(f"  - {p}" for p in pitfalls) if pitfalls else "（无）"
        return ToolOutput(text=json.dumps({
            "success": False,
            "skill_name": name,
            "message": (
                f"技能'{name}'快速链执行失败: {msg}。请手动完成剩余步骤。\n"
                f"\n--- {name} 操作步骤 ---\n"
                f"{body.strip()}\n"
                f"--- 避坑警告 ---\n"
                f"{pitfalls_text}"
            ),
        }, ensure_ascii=False))


def skill_list() -> ToolOutput:
    """List all available skills."""
    from src.skills.manager import get_skill_manager
    from src.tools.registry import get_current_game

    game = get_current_game()
    mgr = get_skill_manager(game)
    names = mgr.list_all()
    if not names:
        return ToolOutput(text=json.dumps({"success": True, "skills": [], "message": "暂无可用技能。"}, ensure_ascii=False))

    return ToolOutput(text=json.dumps({
        "success": True,
        "skills": [{"name": n, "description": mgr.load(n).get("description", "") if mgr.load(n) else ""} for n in names],
    }, ensure_ascii=False))


def task_complete(summary: str = "") -> ToolOutput:
    """Mark the current task as complete with an optional summary."""
    from src.tools.adb_control import clear_skill_coords
    clear_skill_coords()
    return ToolOutput(
        text=json.dumps({"success": True, "summary": summary}, ensure_ascii=False),
        task_done=True,
    )


def subtask_done(name: str, result: str = "") -> ToolOutput:
    """Mark a subtask as complete. Cleans intermediate operations from history
    AND pushes a notification to WeChat.

    Call this after finishing each subtask (e.g. recruit, base-collect,
    credit-shop) so the conversation context stays clean for the next subtask
    and the user gets a progress update.

    Args:
        name: Subtask name matching the skill name (e.g. 'recruit', 'base-collect')
        result: Short outcome summary (e.g. '四星干员，狙击+新手tag组合')
    """
    # ── Idempotency guard: skip if already completed ──
    import threading
    ctx = getattr(threading.current_thread(), '_terra_agent_ctx', None)
    if ctx is not None and name in ctx.state.completed_subtasks:
        logger.info(
            "subtask_done: '%s' already completed — idempotent no-op", name,
        )
        return ToolOutput(
            text=json.dumps({
                "success": True,
                "subtask_name": name,
                "message": f"子任务'{name}'已完成（重复调用已忽略）。",
            }, ensure_ascii=False),
            subtask_done=False,
        )

    # ── Note: subtask_done does NOT send its own WeChat notification. ──
    # The LLM must call notify_with_screen FIRST (while still on the result
    # screen) to capture the actual screenshot, THEN call subtask_done to
    # clean context.  This avoids screenshots of an empty main screen taken
    # after navigation.

    return ToolOutput(
        text=json.dumps({
            "success": True,
            "subtask_name": name,
            "result": result,
            "message": f"子任务'{name}'已标记完成。上下文将被清理。",
        }, ensure_ascii=False),
        subtask_done=True,
        subtask_name=name,
        subtask_result=result,
    )


def _normalize_skill_content(raw: str) -> str:
    """Ensure skill content has proper markdown body sections.

    If the frontmatter contains steps:/pitfalls: lists but the body has no
    ## Steps / ## Pitfalls sections, convert them so the parser can read them.
    """
    from src.skills.parser import SkillParser

    # If body already has ## Steps or ## Pitfalls, it's fine
    _, body = SkillParser._split_frontmatter(raw)
    has_body_steps = "## Steps" in body or "## 步骤" in body
    has_body_pitfalls = "## Pitfalls" in body or "## 注意事项" in body

    if has_body_steps and has_body_pitfalls:
        return raw  # Already well-formed

    # Try to parse frontmatter YAML to extract steps/pitfalls
    fm, _ = SkillParser._split_frontmatter(raw)
    if not fm:
        return raw

    try:
        import yaml
        meta = yaml.safe_load(fm) or {}
    except Exception:
        return raw

    fm_steps = meta.get("steps")
    fm_pitfalls = meta.get("pitfalls")

    # No frontmatter steps/pitfalls to promote — nothing to do
    if not fm_steps and not fm_pitfalls:
        return raw

    # Build body sections from frontmatter data
    body_parts: list[str] = [body.strip()] if body.strip() else []

    if fm_steps and not has_body_steps:
        body_parts.append("## Steps")
        if isinstance(fm_steps, list):
            for i, s in enumerate(fm_steps):
                s = str(s).strip()
                body_parts.append(f"{i + 1}. {s}")
        else:
            body_parts.append(str(fm_steps))

    if fm_pitfalls and not has_body_pitfalls:
        body_parts.append("## Pitfalls")
        if isinstance(fm_pitfalls, list):
            for p in fm_pitfalls:
                body_parts.append(f"- {str(p).strip()}")
        else:
            body_parts.append(str(fm_pitfalls))

    new_body = "\n\n".join(body_parts)
    return f"---\n{fm}\n---\n\n{new_body}\n"


def skill_manage(action: str, name: str = "", content: str = "") -> ToolOutput:
    """Create, update, or delete a skill file. Used by the background reviewer sub-agent.

    Args:
        action: "write_file" to create/update, "delete" to remove.
        name: Skill name (used as filename stem).
        content: Full markdown content for the skill (YAML frontmatter + body).
    """
    from src.skills.manager import get_skill_manager
    from src.tools.registry import get_current_game

    game = get_current_game()
    mgr = get_skill_manager(game)

    if action == "delete":
        if not name:
            return ToolOutput(text=json.dumps({"success": False, "message": "name is required for delete"}))
        ok = mgr.delete(name)
        return ToolOutput(text=json.dumps({
            "success": ok,
            "message": f"Skill '{name}' deleted." if ok else f"Skill '{name}' not found.",
        }, ensure_ascii=False))

    if action == "write_file":
        if not name or not content:
            return ToolOutput(text=json.dumps({"success": False, "message": "name and content are required for write_file"}))
        try:
            # Normalize: convert YAML-frontmatter steps/pitfalls lists into
            # proper ## Steps / ## Pitfalls markdown body sections.
            normalized = _normalize_skill_content(content)
            path = mgr.save(name, normalized)
            return ToolOutput(text=json.dumps({
                "success": True,
                "skill_name": name,
                "path": str(path),
                "message": f"Skill '{name}' saved successfully.",
            }, ensure_ascii=False))
        except Exception as e:
            return ToolOutput(text=json.dumps({"success": False, "message": str(e)}))

    return ToolOutput(text=json.dumps({"success": False, "message": f"Unknown action: {action}. Use 'write_file' or 'delete'."}))


registry.register(
    name="skill_run",
    description=(
        "一键执行已验证的技能。技能中的所有步骤会自动用精确坐标完成（无需 LLM 逐步判断），"
        "比手动执行快 10 倍以上。\n"
        "使用时机：回到主界面后，如果 Active Skill 中有一个已验证技能，立即调用此工具。\n"
        "不要手动重复技能中的坐标步骤 —— 此工具直接读取并执行全部步骤。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name (e.g. 'farm-gt-6')"},
        },
        "required": ["name"],
    },
    handler=skill_run,
)

registry.register(
    name="skill_list",
    description="List all available skills.",
    parameters={"type": "object", "properties": {}},
    handler=skill_list,
)

registry.register(
    name="task_complete",
    description=(
        "Mark the current task as complete. "
        "🔴 调用前必须确认所有子任务都已完成且已截图通知用户。\n"
        "自检清单：\n"
        "1. 所有子任务都调了 notify_with_screen + subtask_done？\n"
        "2. 任务面板奖励已全部领完（只看顶部，不要滚动）？\n"
        "3. notify_with_screen 截图已发送？\n"
        "任何一个是'否' → 先完成它再调 task_complete。\n"
        "Write a warm, informative summary telling the user what you accomplished, "
        "key results (e.g. sanity spent, items obtained), and how many steps it took. "
        "Use natural Chinese — as if reporting to your commanding officer."
    ),
    parameters={
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "A warm summary of what was accomplished. Include: what task was done, "
                    "key results (sanity/items), how many steps. In Chinese, natural tone."
                ),
            },
        },
    },
    handler=task_complete,
)

registry.register(
    name="subtask_done",
    description=(
        "清理子任务中间消息，释放 LLM 上下文。\n"
        "🔴 调用前必须先调 notify_with_screen——趁着还在结果画面上截图发给用户。\n"
        "🔴 绝对不要先关闭面板/返回再调 notify_with_screen——那样截到的是空白界面。\n"
        "正确顺序：notify_with_screen(\"基建产物已全部收取\") → subtask_done('base-collect', '收了4个制造站+2个贸易站订单')\n"
        "每个子任务调一次。IMPORTANT: 确认子任务真正完成了再调，不要提前。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Subtask name matching the skill name (e.g. 'recruit', 'base-collect', 'credit-shop')",
            },
            "result": {
                "type": "string",
                "description": "Short outcome summary in Chinese (e.g. '四星干员，狙击+新手tag组合', '信用商店清除并购买了3件物品')",
            },
        },
        "required": ["name"],
    },
    handler=subtask_done,
)

registry.register(
    name="skill_manage",
    description="Create, update, or delete a skill file. Use write_file to create/update, delete to remove.",
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["write_file", "delete"],
                       "description": "write_file to create/update, delete to remove"},
            "name": {"type": "string", "description": "Skill name"},
            "content": {"type": "string", "description": "Full markdown content (YAML frontmatter + body). Required for write_file."},
        },
        "required": ["action"],
    },
    handler=skill_manage,
)
