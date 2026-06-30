"""Circuit breaker for cascading failure protection.

When a downstream dependency (LLM API, ADB, OCR) starts failing repeatedly,
the circuit breaker "opens" and fast-fails subsequent calls without hitting
the failing dependency.  After a recovery timeout, it transitions to "half-open"
and allows one trial call.  If that succeeds, the circuit "closes" (back to
normal).  If it fails, the circuit re-opens.

Usage:
    from src.utils.circuit_breaker import CircuitBreaker, CircuitOpenError

    llm_breaker = CircuitBreaker("llm", failure_threshold=5, recovery_timeout=60)

    try:
        result = llm_breaker.call(lambda: some_llm_call())
    except CircuitOpenError:
        notify_user("LLM 服务暂时不可用，请稍后重试")
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    """Raised when call() is invoked on an open circuit breaker."""


class CircuitBreaker:
    """Thread-safe circuit breaker for a single dependency."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._state: str = "closed"  # closed | open | half_open
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            self._transition()
            return self._state

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    def _transition(self) -> None:
        """Internal: transition open→half_open if recovery timeout elapsed."""
        if self._state == "open" and time.monotonic() - self._last_failure_time >= self.recovery_timeout:
            self._state = "half_open"
            logger.info("Circuit '%s' → half_open (recovery timeout elapsed)", self.name)

    def record_success(self) -> None:
        with self._lock:
            if self._state == "half_open":
                logger.info("Circuit '%s' → closed (trial succeeded)", self.name)
            self._failure_count = 0
            self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == "half_open" or self._failure_count >= self.failure_threshold:
                if self._state != "open":
                    logger.warning(
                        "Circuit '%s' → open (%d failures, threshold=%d)",
                        self.name, self._failure_count, self.failure_threshold,
                    )
                self._state = "open"

    def call(self, fn, *args, **kwargs):
        """Call fn(*args, **kwargs) with circuit breaker protection.

        Returns fn's result on success.
        Raises CircuitOpenError if the circuit is open.
        Re-raises fn's exception on failure (after recording it).
        """
        with self._lock:
            self._transition()
            if self._state == "open":
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is open "
                    f"({self._failure_count} failures, "
                    f"retry in ~{self.recovery_timeout - (time.monotonic() - self._last_failure_time):.0f}s)"
                )

        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()
            return result


# ── Global circuit breaker instances ──

# LLM API: 10 consecutive failures within 120s → open for 60s
# Threshold raised from 5 to match increased retry budget (up to 7 conn retries
# + 3 backup-key retries per call).  With 5, a single bad call chain would open
# the circuit before the backup key had a chance.
llm_breaker = CircuitBreaker("llm", failure_threshold=10, recovery_timeout=60.0)

# ADB: 3 consecutive failures → open for 30s (ADB recovers faster)
adb_breaker = CircuitBreaker("adb", failure_threshold=3, recovery_timeout=30.0)

# OCR: 5 consecutive failures → open for 60s
ocr_breaker = CircuitBreaker("ocr", failure_threshold=5, recovery_timeout=60.0)
