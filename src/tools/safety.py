"""Safety tool: intercept dangerous operations before execution.

Self-registers with the tool registry at import time.
"""

from __future__ import annotations

import json
import logging
import threading

from config.settings import config

logger = logging.getLogger(__name__)

# Track daily action count (thread-safe)
_daily_count: int = 0
_daily_lock = threading.Lock()


def check_dangerous(target_text: str, game: str) -> dict | None:
    """Check if a target text contains dangerous keywords.

    Returns None if safe, or a warning dict if dangerous.
    Safe compound terms (e.g. "源石订单") override dangerous keyword matches.

    Uses per-game keyword lists from GameRegistry, falling back to config.
    """
    safe_terms = config.safety.effective_safe_compound_terms(game)
    dangerous_kw = config.safety.effective_dangerous_keywords(game)

    # Check safe compound terms first — they override dangerous keywords
    for safe_term in safe_terms:
        if safe_term in target_text:
            logger.debug("Safety: safe compound '%s' found in '%s', allowing", safe_term, target_text)
            return None

    for keyword in dangerous_kw:
        if keyword in target_text:
            return {
                "dangerous": True,
                "keyword": keyword,
                "message": f"检测到危险关键词'{keyword}'！该操作可能消耗付费货币，已拦截。如确需执行请明确说明。",
            }
    return None


def check_confirmation_required(target_text: str, game: str) -> bool:
    """Check if a target text requires user confirmation."""
    for keyword in config.safety.effective_confirmation_keywords(game):
        if keyword in target_text:
            return True
    return False


def check_daily_limit() -> dict | None:
    """Check if daily action limit is exceeded (read-only, does NOT increment).

    Call this BEFORE executing an action to see if the limit has been reached.
    Call commit_daily_action() AFTER a successful action to increment the counter.
    """
    global _daily_count
    with _daily_lock:
        current = _daily_count
    if current >= config.safety.max_daily_actions:
        return {
            "limit_exceeded": True,
            "count": current,
            "max": config.safety.max_daily_actions,
            "message": f"今日操作次数已达上限({config.safety.max_daily_actions})，已暂停。明天重置或手动清除。",
        }
    return None


def commit_daily_action() -> None:
    """Increment the daily action counter. Call AFTER a successful action."""
    global _daily_count
    with _daily_lock:
        _daily_count += 1


def reset_daily_count() -> None:
    """Reset the daily action counter. Call in test setup/teardown."""
    global _daily_count
    with _daily_lock:
        _daily_count = 0


def get_daily_count() -> int:
    """Get current daily action count for debugging."""
    with _daily_lock:
        return _daily_count
