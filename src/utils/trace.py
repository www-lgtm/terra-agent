"""Thread-safe trace ID for request correlation.

A single trace_id follows a user message through the entire processing chain:
WeixinBot → MessageRouter → TerraAgent → LLM calls → tool execution.

Usage:
    from src.utils.trace import (
        generate_trace_id, set_trace_id, get_trace_id, trace_context,
    )
    set_trace_id(generate_trace_id())
    print(get_trace_id())   # "a1b2c3d4e5f6"
    with trace_context("custom_id"):
        ...  # all nested calls see "custom_id"
    # trace_id reverts when exiting the context manager
"""

from __future__ import annotations

import contextlib
import threading
import uuid

_log_trace_id: threading.local = threading.local()


def generate_trace_id() -> str:
    """Generate a short unique trace ID (12 hex chars)."""
    return uuid.uuid4().hex[:12]


def set_trace_id(tid: str) -> None:
    """Set the trace_id for the current thread."""
    _log_trace_id.value = tid


def get_trace_id() -> str:
    """Get the current thread's trace_id, or 'unknown' if not set."""
    return getattr(_log_trace_id, "value", "unknown")


@contextlib.contextmanager
def trace_context(tid: str | None = None):
    """Set a trace_id for the duration of a context block. Thread-safe.

    If tid is None, generates a new one.

    Usage:
        with trace_context():
            do_work()  # all nested calls see the same trace_id
    """
    prev = getattr(_log_trace_id, "value", None)
    set_trace_id(tid or generate_trace_id())
    try:
        yield
    finally:
        if prev is not None:
            set_trace_id(prev)
        else:
            try:
                delattr(_log_trace_id, "value")
            except AttributeError:
                pass
