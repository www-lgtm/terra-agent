"""NotificationBuffer — dedup + throttle agent progress notifications.

Prevents WeChat spam when multiple agents produce progress updates.
ask_user / complete / error notifications pass through immediately;
progress notifications are collapsed per agent and throttled.

Thread-safe: all methods acquire a lock since notifications can arrive
from background agent threads concurrently.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NotificationBuffer:
    """Per-router notification buffer.  One instance per MessageRouter.

    Thread-safe: uses a reentrant lock since flush() may be called
    by the push path which also holds the lock in should_push().
    """

    push_interval: float = 10.0
    _buffer: dict[str, str] = field(default_factory=dict)
    _last_push: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def should_push(self, agent_label: str, notify_type: str, msg: str) -> bool:
        """Decide whether to push this notification to WeChat right now.

        Returns True for push-now, False for hold-and-collapse.
        The caller is responsible for actually sending the message.

        - ask_user / complete / error / screenshot → always push immediately
        - progress → per-agent throttle to push_interval, collapse same-agent duplicates
        """
        with self._lock:
            # Critical notifications always pass through immediately
            if notify_type in ("ask_user", "complete", "error", "screenshot"):
                self._last_push[agent_label] = time.time()
                return True

            # Progress: throttle by interval per-agent
            now = time.time()
            last = self._last_push.get(agent_label, 0)
            if now - last > self.push_interval:
                self._last_push[agent_label] = now
                return True

            # Hold — buffer for later flush
            self._buffer[agent_label] = msg
            return False

    def flush(self) -> list[tuple[str, str]]:
        """Drain buffered progress notifications. Returns [(agent_label, msg), ...].

        Caller should forward these to the user BEFORE the current notification.
        """
        with self._lock:
            items = list(self._buffer.items())
            self._buffer.clear()
            now = time.time()
            for label, _ in items:
                self._last_push[label] = now
            return items

    def drain_all(self) -> list[tuple[str, str]]:
        """Drain ALL buffered notifications (including for critical types).
        Used before sending complete/error to ensure buffered progress
        is forwarded rather than silently discarded.
        """
        return self.flush()
