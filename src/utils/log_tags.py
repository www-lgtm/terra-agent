"""Per-agent log tag — injects game+device identifier into every log line.

In multi-agent scenarios, all log output is interleaved in a single file.
Without tags, it's impossible to tell which agent produced which log line.

Usage:
    from src.utils.log_tags import set_agent_tag, clear_agent_tag, AgentTagFilter

    # In agent thread startup:
    set_agent_tag("明日方舟-16384")

    # In agent thread cleanup:
    clear_agent_tag()

Tags are injected via a logging.Filter that adds `agent_tag` to every LogRecord.
The root logger's format string should include `%(agent_tag)s`.
"""

from __future__ import annotations

import logging
import threading

_agent_tag: threading.local = threading.local()


def set_agent_tag(tag: str) -> None:
    """Set the agent tag for the current thread."""
    _agent_tag.value = tag


def clear_agent_tag() -> None:
    """Clear the agent tag for the current thread."""
    _agent_tag.value = ""


def get_agent_tag() -> str:
    """Get the agent tag for the current thread. Returns '' if not set."""
    return getattr(_agent_tag, "value", "") or ""


class AgentTagFilter(logging.Filter):
    """Inject agent_tag and trace_id into every LogRecord.

    Add to root logger handlers: root_logger.handlers[0].addFilter(AgentTagFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        tag = get_agent_tag()
        record.agent_tag = f"[{tag}] " if tag else ""
        from src.utils.trace import get_trace_id
        record.trace_id = f"[trace={get_trace_id()}] " if get_trace_id() != "unknown" else ""
        return True
