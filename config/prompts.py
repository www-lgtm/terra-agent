"""Layer-based system prompt builder with cache-breakpoint support.

Layer 1 (stable):   persona + common rules + game-specific append (cached)
Layer 2 (context):  matched skill content (changes with skills)
Layer 3 (volatile): state snapshot + current time (per-turn, never cached)

Game-specific append text is injected by callers via the game_append parameter.
This module has ZERO imports from src/ — it is a pure configuration layer.
"""

from datetime import datetime, timezone


# ============================================================
# Persona
# ============================================================

_PERSONA = "你是 Terra，游戏自动化助手。接口是微信，用自然中文回复。不确定时问用户，不要猜。"


def get_persona(game: str = "") -> str:
    """Return the unified persona block."""
    return _PERSONA


# ============================================================
# Stable layer — common operational rules (all games)
# ============================================================

_COMMON_STABLE = """## 标记说明

🔴 = 违反会导致任务失败。其他规则按常识执行即可。

## 工作原理

每次操作后系统自动注入截图 + OCR 文字，这是你唯一的信息源。
等待加载/转场时只输出文字不调工具，系统会自动注入新截图。
自己做等待，不要 ask_user() 问用户"加载好了告诉我"。

## 输出格式

🔴 thinking 最多 3 句中文，格式：当前画面 → 要做什么 → 用什么工具。禁止列出完整计划、重复历史步骤。
🔴 text 一句话中文，15-30 字。用户看不到 thinking，关键信息必须写在 text。
🔴 禁止中英双语重复。每条信息只写一种语言。
🔴 禁止复读。不要把上一轮的 thinking/text 重新输出一遍。画面没变就一句话概括状态。

## 核心操作规则

- 🔴 **Skill > OCR**：Skill 指令说点 A → 只能点 A。OCR 只是帮你定位 A，不是让你改目标。
- **触屏方向**：上滑=看下方，下滑=看上方。收到"滑动边界"→ 立即换方向。
- **弹窗**：关不掉 → adb_back() → dismiss_all_popups() → magnify() → tap_magnified() → ask_user()。不要猜百分比坐标。
- **失败即问**：同一操作 3 种方法都失败 → ask_user()。不确定 → ask_user()。调用后等回复。
- **资源消耗**：任何消耗游戏内资源的操作必须先 ask_user() 确认。不确定是否消耗 → 问。
- **多子任务**：每个子任务完成 → subtask_done(name, result)。全部完成 → task_complete()。
- **通信**：需要用户做任何事 → 必须调 ask_user()。纯文本输出用户看不见。
- **自动战斗**：严禁 notify_with_screen。等操作结束、回到可操作界面后再通知。
- **记住经验**：同一按钮失败 2+ 次后终于点成功了 → remember() 保存定位方式。
- **[用户指令] 优先一切**。同名按钮用 adb_tap_smart(target, row_text) 精确定位。"""


# ── Internal builders ────────────────────────────────────────────

def _build_stable(game_append: str = "") -> str:
    """Concatenate: persona + common rules + game-specific append.

    game_append is injected by the caller (resolved from GameRegistry).
    This module has zero imports from src/.
    """
    result = _PERSONA + "\n\n" + _COMMON_STABLE
    if game_append:
        result += "\n\n" + game_append
    return result


def _build_task(user_task: str) -> str:
    """User task instruction — cached per task, changes only on new task."""
    if not user_task:
        return ""
    return f"## 当前任务\n\n{user_task}"


def _build_context(skill_text: str) -> str:
    if not skill_text:
        return (
            "## 可用技能\n\n"
            "未匹配到已知技能。请观察当前画面，从可见的按钮和文字出发逐步探索。"
            "优先考虑：返回主界面 -> 从主界面导航到目标功能。"
        )
    return f"## 当前技能\n\n{skill_text}"


def _build_volatile(state_summary: str, now: str, screen_w: int = 0, screen_h: int = 0) -> str:
    screen_line = ""
    if screen_w and screen_h:
        screen_line = f"\n屏幕分辨率: {screen_w}x{screen_h}"
    summary_line = state_summary if state_summary else ""
    parts = ["## 当前状态"]
    if summary_line:
        parts.append(summary_line)
    if screen_line.strip():
        parts.append(screen_line.strip())
    parts.append(f"\n当前时间: {now}")
    return "\n".join(parts)


# ── Public API ───────────────────────────────────────────────────

def build_system_prompt_parts(
    game_append: str = "",
    skill_text: str = "",
    state_summary: str = "",
    screen_w: int = 0,
    screen_h: int = 0,
    user_task: str = "",
) -> dict[str, str]:
    """Build the three prompt layers as separate strings."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return {
        "stable": _build_stable(game_append),
        "task": _build_task(user_task),
        "context": _build_context(skill_text),
        "volatile": _build_volatile(state_summary, now, screen_w, screen_h),
    }


def build_system_prompt(
    game_append: str = "",
    skill_text: str = "",
    state_summary: str = "",
    screen_w: int = 0,
    screen_h: int = 0,
    user_task: str = "",
) -> str:
    """Build the full system prompt as a single string."""
    parts = build_system_prompt_parts(game_append, skill_text, state_summary, screen_w, screen_h, user_task=user_task)
    if parts["task"]:
        return f"{parts['stable']}\n\n{parts['task']}\n\n{parts['context']}\n\n{parts['volatile']}"
    return f"{parts['stable']}\n\n{parts['context']}\n\n{parts['volatile']}"


def build_system_prompt_cached(
    game_append: str = "",
    skill_text: str = "",
    state_summary: str = "",
    screen_w: int = 0,
    screen_h: int = 0,
    user_task: str = "",
) -> list[dict]:
    """Build system prompt with THREE independent cache breakpoints.

    Layer 1 (stable): persona + rules + game_append — almost never changes.
    Layer 1.5 (task): user task instruction — cached per task (new).
    Layer 2 (context): skill text — changes only on skill switch.
    Layer 3 (volatile): state + time — changes every iteration (never cached).

    Anthropic's prompt caching uses PREFIX matching:
    - Layer 1 is almost always a cache HIT (same persona+rules across tasks)
    - Layer 1.5 is a cache HIT within the same task
    - Layer 2 is a cache HIT when the same skill continues
    - Even on task switch, Layer 1 still HITs

    Previously the user task instruction lived in the first conversation
    message (uncached, billed at full input rate every iteration).  Moving
    it into the system prompt as a cached layer saves ~200-500 tokens per
    iteration at cache-read pricing (¥0.02/M vs ¥1/M for cache-miss input).
    """
    parts = build_system_prompt_parts(game_append, skill_text, state_summary, screen_w, screen_h, user_task=user_task)
    blocks: list[dict] = [
        {"type": "text", "text": parts["stable"],
         "cache_control": {"type": "ephemeral"}},
    ]
    # Only emit task layer when non-empty (keeps cache clean on empty)
    if parts["task"]:
        blocks.append(
            {"type": "text", "text": parts["task"],
             "cache_control": {"type": "ephemeral"}},
        )
    blocks.append(
        {"type": "text", "text": parts["context"],
         "cache_control": {"type": "ephemeral"}},
    )
    blocks.append(
        {"type": "text", "text": parts["volatile"]},
    )
    return blocks
