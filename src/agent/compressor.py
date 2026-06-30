"""Context compressor: truncation-based context window management.

Phase 3: Pure truncation — no LLM summarization.
Phase 2 was LLM summarization (removed — summaries described stale state and
confused the agent when they contradicted the live screenshot).
Phase 1 was simple truncation.

Key design decisions:
- No LLM-generated semantic summaries.  A stale summary describing the screen
  from 15 seconds ago contradicts the current injected screenshot, causing the
  agent to oscillate between the summary and the real image.
- Protected messages (user instructions, system hints, memory injections,
  periodic reviews) pass through verbatim.
- Successful action exemplars (productive taps with confirmed success) are
  preserved so the agent doesn't forget which coordinates/targets worked.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

TRUNCATION_TRIGGER = 40  # Start compressing when we exceed this many messages


def compress_history(
    messages: list[dict[str, Any]],
    keep_first: int = 2,
    keep_last: int = 25,
) -> list[dict[str, Any]]:
    """Compress conversation history by truncating the middle segment.

    Strategy:
    - Keep first `keep_first` messages (user task + initial response)
    - Keep protected messages verbatim (user instructions, system hints, etc.)
    - Keep successful action exemplars (productive taps with confirmed results)
    - Keep last `keep_last` messages (recent context including current screen)
    - Drop the rest with a neutral counter marker (no semantic summary)

    The periodic REVIEW_INTERVAL injection in loop.py re-injects full skill
    Steps+Pitfalls every 12 iterations, so dropped middle detail is not lost.
    """
    if len(messages) <= TRUNCATION_TRIGGER:
        return messages

    first = messages[:keep_first]
    last = messages[-keep_last:] if keep_last > 0 else []
    middle = messages[keep_first:-keep_last] if keep_last > 0 else messages[keep_first:]

    if not middle:
        return messages

    # Protect user instruction/guidance/reply messages from compression — keep them verbatim.
    # Also protect system-injected hints (loop warnings, memory hints).
    # These are string-typed user messages that start with a bracket tag like
    # [用户指令], [用户回复], [Guidance], [系统提示], [关联记忆].
    protected: list[dict[str, Any]] = []
    compressible: list[dict[str, Any]] = []
    for m in middle:
        content = m.get("content", "")
        role = m.get("role", "")
        if isinstance(content, str) and role == "user":
            stripped = content.strip()
            if stripped.startswith("[用户指令") or stripped.startswith("[用户回复]") \
                    or stripped.startswith("[Guidance]") or stripped.startswith("[系统提示") \
                    or stripped.startswith("[关联记忆") \
                    or stripped.startswith("[子任务完成]") \
                    or stripped.startswith("[已省略"):
                protected.append(m)
                continue
        compressible.append(m)

    # ── Preserve successful action exemplars ──
    # Repeated operations (e.g. 4 recruitment slots all set to 9h) are valuable
    # reference material — if we compress them away, the LLM forgets what
    # coordinates/targets worked and starts guessing from scratch.
    exemplars = _pick_success_exemplars(compressible)
    if exemplars:
        # Remove exemplars from compressible so they aren't summarized
        exemplar_ids = {id(m): True for m in exemplars}
        compressible = [m for m in compressible if id(m) not in exemplar_ids]
        protected.extend(exemplars)
        logger.debug("Preserved %d success exemplar(s) through compression", len(exemplars))

    # Count standalone (non-tool-result) messages for the "worth it" check
    standalone = [m for m in compressible if not _is_tool_result(m)]
    if len(standalone) <= 4 and not exemplars:
        return messages  # Not worth compressing

    dropped_count = len(compressible)
    action_summary = _extract_action_summary(compressible)
    summary_msg = {
        "role": "user",
        "content": (
            f"[已省略 {dropped_count} 条中间消息，"
            f"保留首部 {keep_first} 条 + 受保护 {len(protected)} 条 + 尾部 {keep_last} 条。"
            f"省略期间操作: {action_summary}]"
        ),
    }

    logger.info(
        "Truncated %d messages → %d + %d protected + marker + %d (dropped %d middle)",
        len(messages), len(first), len(protected), len(last), dropped_count,
    )
    result = first + protected + [summary_msg] + last
    # ── Hard cap: force truncate if still too large ──
    MAX_MESSAGES = 400
    if len(result) > MAX_MESSAGES:
        # P2: use MAX_MESSAGES as the actual cap target, not keep_first+100.
        # The old code produced 102 messages while logging "hard cap=200".
        result = result[:MAX_MESSAGES]
        logger.warning(
            "Forced truncation: %d → %d messages (hard cap=%d)",
            len(first + protected + [summary_msg] + last), len(result), MAX_MESSAGES,
        )
    return result


def _tool_key(msg: dict[str, Any]) -> tuple[str, str]:
    """Build a dedup key from an assistant message's productive tool call.

    Groups: same tool + same target (or rounded position for position-based taps).
    """
    content = msg.get("content", [])
    if not isinstance(content, list):
        return ("", "")
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name", "")
            inp = block.get("input", {})
            target = inp.get("target", "")
            if not target:
                # Position-based tools: use rounded percentage
                x = inp.get("x_pct", inp.get("x", ""))
                y = inp.get("y_pct", inp.get("y", ""))
                if x and y:
                    target = f"pos({round(float(x), 3)},{round(float(y), 3)})"
                else:
                    target = str(inp)[:80]
            return (name, target[:60])
    return ("", "")


def _extract_action_summary(messages: list[dict[str, Any]]) -> str:
    """Build a one-line summary of actions taken in the dropped messages.

    Scans tool_use blocks for action names + targets, deduplicates,
    and returns a compact Chinese summary like "点击 任务/基建/终端,
    skill_run credit-shop, 滑动×3".  This gives the LLM a rough sense of
    what happened in the truncated segment without any LLM summarization
    call (zero added cost).
    """
    tap_targets: list[str] = []
    skill_runs: list[str] = []
    other_tools: dict[str, int] = {}
    scroll_count = 0

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            inp = block.get("input", {})
            if name in ("adb_tap", "adb_tap_position", "tap_magnified"):
                target = inp.get("target", "")
                if target and target not in tap_targets:
                    tap_targets.append(target)
            elif name == "skill_run":
                sn = inp.get("name", "")
                if sn and sn not in skill_runs:
                    skill_runs.append(sn)
            elif name in ("adb_scroll", "adb_swipe"):
                scroll_count += 1
            else:
                other_tools[name] = other_tools.get(name, 0) + 1

    parts: list[str] = []
    if tap_targets:
        parts.append(f"点击 {', '.join(tap_targets[:6])}" + (f"等{len(tap_targets)}个" if len(tap_targets) > 6 else ""))
    if skill_runs:
        parts.append(f"skill_run {', '.join(skill_runs[:4])}" + (f"等{len(skill_runs)}个" if len(skill_runs) > 4 else ""))
    if scroll_count:
        parts.append(f"滑动×{scroll_count}")
    for tname, count in sorted(other_tools.items(), key=lambda x: -x[1])[:3]:
        parts.append(f"{tname}×{count}")

    return "; ".join(parts) if parts else "无关键操作"


# Productive tools whose success patterns we want to preserve through compression.
_PRODUCTIVE_TOOLS = frozenset({"adb_tap", "adb_tap_position", "tap_magnified"})


def _pick_success_exemplars(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract successful action-result pairs from compressible messages.

    Returns the kept assistant+user message pairs so they survive compression.
    Deduplicated by (tool_name, target) — one exemplar per distinct operation.
    """
    # tool_use_id → assistant message that issued it
    pending: dict[str, dict[str, Any]] = {}
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    kept_ids: set[str] = set()  # Avoid duplicate message IDs in output

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue

        if role == "assistant":
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") in _PRODUCTIVE_TOOLS:
                    tid = block.get("id", "")
                    if tid:
                        pending[tid] = msg

        elif role == "user":
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue
                tid = block.get("tool_use_id", "")
                if tid not in pending:
                    continue
                # Check success in the result payload
                result = block.get("content", "")
                if isinstance(result, list):
                    texts = [str(b.get("text", "")) for b in result
                             if isinstance(b, dict) and b.get("type") == "text"]
                    result = " ".join(texts)
                result_str = str(result)
                if '"success": true' in result_str.lower() or '"success":True' in result_str:
                    asst_msg = pending[tid]
                    key = _tool_key(asst_msg)
                    if key not in seen:
                        seen.add(key)
                        # Keep the assistant message with tool_use
                        asst_id = id(asst_msg)
                        if asst_id not in kept_ids:
                            kept.append(asst_msg)
                            kept_ids.add(asst_id)
                        # Keep the user message with tool_result
                        msg_id = id(msg)
                        if msg_id not in kept_ids:
                            kept.append(msg)
                            kept_ids.add(msg_id)
                del pending[tid]

    return kept


def _is_tool_result(msg: dict[str, Any]) -> bool:
    """Check if a message is a tool_result."""
    content = msg.get("content", "")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return True
    return False
