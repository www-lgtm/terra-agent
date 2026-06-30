"""Structured error handling for Terra Agent.

Defines a clear taxonomy: Recoverable (retry), Degraded (skip/fallback),
Fatal (abort).  Plus a unified `handle_error()` dispatcher so callers
don't need to know the hierarchy.
"""

from __future__ import annotations

import enum
import logging
from typing import Literal


# ── Exception hierarchy ──────────────────────────────────────────────

class TerraError(Exception):
    """Base for all Terra Agent errors."""
    pass


class RecoverableError(TerraError):
    """Transient — retry with backoff.

    Use for: temporary ADB disconnects, OCR timeouts, stale skill coordinates,
    transient network issues.
    """
    pass


class DegradedError(TerraError):
    """Non-fatal but not worth retrying — skip the operation, continue.

    Use for: DB write failures (data survives in memory), cache misses,
    optional feature failures, non-critical subsystem errors.
    """
    pass


class FatalError(TerraError):
    """Unrecoverable — abort the task immediately.

    Use for: invalid LLM API credentials, database corruption, unrecoverable
    filesystem errors.
    """
    pass


# ── Error action ────────────────────────────────────────────────────

class ErrorAction(enum.Enum):
    RETRY = "retry"
    SKIP = "skip"
    DEGRADE = "degrade"
    ABORT = "abort"


def classify_error(exc: Exception) -> ErrorAction:
    """Map an exception to the recommended handling action."""
    if isinstance(exc, FatalError):
        return ErrorAction.ABORT
    if isinstance(exc, DegradedError):
        return ErrorAction.DEGRADE
    if isinstance(exc, RecoverableError):
        return ErrorAction.RETRY
    # Unknown exceptions: be safe — degrade
    return ErrorAction.DEGRADE


def handle_error(
    logger: logging.Logger,
    exc: Exception,
    context: str = "",
    *,
    exc_info: bool = False,
) -> ErrorAction:
    """Log + classify.  One-line error handling for catch blocks.

    Usage:
        except Exception as e:
            action = handle_error(logger, e, "memory hint injection")
            if action == ErrorAction.ABORT:
                raise
            # Otherwise continue degraded
    """
    action = classify_error(exc)
    msg = f"[{action.value}] {context}: {exc}" if context else f"[{action.value}] {exc}"

    if action == ErrorAction.ABORT:
        logger.error(msg, exc_info=exc_info)
    elif action == ErrorAction.RETRY:
        logger.warning(msg, exc_info=exc_info)
    else:
        logger.warning(msg, exc_info=exc_info)

    return action


def safe_log(
    logger: logging.Logger,
    level: Literal["warning", "error"],
    msg: str,
    exc_info: bool = False,
) -> None:
    """Unified logging exit point. Prefer handle_error() for new code.

    Args:
        logger: The logger instance from the calling module.
        level: "warning" for recoverable, "error" for fatal.
        msg: Human-readable message.
        exc_info: When True, include the full traceback.
    """
    if level == "warning":
        logger.warning(msg, exc_info=exc_info)
    elif level == "error":
        logger.error(msg, exc_info=exc_info)
