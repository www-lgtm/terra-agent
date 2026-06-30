"""MiMo-V2.5 client via Anthropic Messages API.

Endpoint: https://api.xiaomimimo.com/anthropic
Auth: api-key header (MiMo uses api-key, not the standard x-api-key)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from anthropic import Anthropic, APIStatusError, APIConnectionError, APIResponseValidationError
from anthropic.types import Message, MessageParam, ToolParam

from config.settings import config

logger = logging.getLogger(__name__)

# ── Retry helpers ────────────────────────────────────────────────────

_RETRY_MAX_ATTEMPTS = 5          # total retries per key for status/rate-limit errors
_RETRY_BACKOFF_BASE = 1.5        # seconds, multiplied exponentially for status errors
_RETRY_BACKOFF_CONN_BASE = 2.0   # seconds for connection errors (TCP drops recover within seconds)
_RETRY_MAX_ATTEMPTS_CONN = 5     # connection errors: 2+4+8+16+32 = 62s window, sufficient for transient drops
_RETRY_BACKOFF_CAP = 30.0        # cap any single retry delay at 30s


def _is_connection_error(exc: Exception) -> bool:
    """Return True if the error is a network-level (TCP/DNS) failure.

    These errors are typically transient infrastructure issues, not rate limits
    or API-level rejections.  They benefit from longer backoff and more retries.
    """
    if isinstance(exc, (httpx.NetworkError, httpx.TimeoutException,
                         httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, APIConnectionError):
        return True
    # Low-level SDK parse errors during stream processing often result from
    # truncated responses due to connection drops — treat as connection errors.
    msg = str(exc).lower()
    if any(phrase in msg for phrase in (
        "unexpected event", "message_delta before",
        "list index", "index out of range",
    )):
        return True
    return False


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is likely transient (network/HTTP 5xx)."""
    if _is_connection_error(exc):
        return True
    if isinstance(exc, APIResponseValidationError):
        return False
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429 or (500 <= exc.status_code < 600):
            return True
        # Retry 400 when image data is corrupted (transient encoding issue)
        if exc.status_code == 400 and "corrupted" in str(exc).lower():
            return True
        return False
    # Catch low-level SDK errors (IndexError, KeyError, AttributeError) during
    # response parsing.  Almost always transient malformed-SSE issues.
    if isinstance(exc, (IndexError, KeyError, AttributeError)):
        return True
    return False


def _retry_call(fn, *args, max_attempts: int | None = None, **kwargs):
    """Call fn(*args, **kwargs) with circuit breaker + exponential backoff.

    Only retries on transient errors (network, 5xx). Re-raises immediately
    on 4xx errors or after exhausting retries.  Connection errors get longer
    backoff and more attempts than status/rate-limit errors.
    """
    from src.utils.circuit_breaker import llm_breaker, CircuitOpenError

    # Fast-fail if circuit is open
    try:
        llm_breaker.call(lambda: None)  # Probe-only call
    except CircuitOpenError as e:
        logger.warning("LLM circuit breaker open — fast-failing: %s", e)
        raise

    last_exc: Exception | None = None
    first_exc_is_conn: bool | None = None  # lock error class on first failure
    for attempt in range(max_attempts if max_attempts is not None else _RETRY_MAX_ATTEMPTS):
        try:
            result = fn(*args, **kwargs)
            llm_breaker.record_success()
            return result
        except Exception as exc:
            last_exc = exc
            llm_breaker.record_failure()
            if not _is_retryable(exc):
                raise
            # Lock error class on first retryable failure — prevents a connection
            # error from consuming status-error retry budget and vice versa
            if first_exc_is_conn is None:
                first_exc_is_conn = _is_connection_error(exc)
            # Determine attempt cap and backoff based on error class
            if first_exc_is_conn:
                cap = _RETRY_MAX_ATTEMPTS_CONN if max_attempts is None else max_attempts
                base = _RETRY_BACKOFF_CONN_BASE
            else:
                cap = _RETRY_MAX_ATTEMPTS if max_attempts is None else max_attempts
                base = _RETRY_BACKOFF_BASE
            if attempt >= cap - 1:
                raise
            delay = min(base ** (attempt + 1), _RETRY_BACKOFF_CAP)
            logger.warning(
                "LLM call failed (attempt %d/%d, retry in %.1fs)%s: %s",
                attempt + 1, cap, delay,
                " [conn]" if first_exc_is_conn else "",
                exc,
            )
            time.sleep(delay)
    # Should be unreachable, but re-raise for type safety
    if last_exc is not None:
        raise last_exc


class MiMoClient:
    """Anthropic-compatible client for MiMo-V2.5.

    MiMo uses `api-key` header for auth, not the standard `x-api-key`.

    api_key_index selects which API key from config to use (round-robin
    across comma-separated values in MIMO_API_KEY).  When multiple keys
    are configured, different agents use different keys to bypass per-key
    concurrency limits on the API server.

    Timeout tiering: two httpx clients with different timeouts.
    - Fast client (18s): for small requests (estimated input < 5K tokens).
      These should complete quickly; a short timeout lets them fail fast
      and retry on another key instead of queuing.
    - Slow client (60s): for large requests (estimated input ≥ 5K tokens).
      These need more time for inference + potential queuing.
    """

    _FAST_TIMEOUT = 35.0   # Small requests: enough for vision inference (was 18)
    _SLOW_TIMEOUT = 120.0  # Large/image-heavy requests: vision models need time
    _TOKEN_ESTIMATE_THRESHOLD = 5000  # Tokens: switch point

    def __init__(self, api_key_index: int = 0) -> None:
        self._api_key_index = api_key_index
        self._active_key = config.llm.get_api_key(api_key_index)
        self._httpx_client = httpx.Client(timeout=self._FAST_TIMEOUT)
        self._httpx_client_slow = httpx.Client(timeout=self._SLOW_TIMEOUT)
        self._client = Anthropic(
            base_url=config.llm.base_url,
            api_key=self._active_key,
            max_retries=0,  # disable SDK retry — we handle retries in _retry_call
            default_headers={
                "api-key": self._active_key,
                "anthropic-beta": "prompt-caching-2024-07-31",
            },
            http_client=self._httpx_client,
        )
        self._client_slow = Anthropic(
            base_url=config.llm.base_url,
            api_key=self._active_key,
            max_retries=0,  # disable SDK retry — we handle retries in _retry_call
            default_headers={
                "api-key": self._active_key,
                "anthropic-beta": "prompt-caching-2024-07-31",
            },
            http_client=self._httpx_client_slow,
        )
        self.model = config.llm.model
        self.max_tokens = config.llm.max_tokens
        self._closed = False

    def close(self) -> None:
        """Release underlying HTTP resources (connection pool, sockets).

        Safe to call multiple times. After closing, chat() calls raise.
        """
        if not self._closed:
            self._httpx_client.close()
            self._httpx_client_slow.close()
            self._closed = True

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, Any]],
                         system: str | list[dict[str, Any]] | None = None) -> int:
        """Rough token estimate for client timeout selection and budget tracking.

        CJK chars: ~1.5 chars/token.  ASCII: ~4 chars/token.
        Uses weighted counting so Chinese-heavy system prompts aren't
        underestimated by 2-3x (as len//4 would do).

        Images: ~1500-2500 tokens each for a typical 480-800px screenshot
        in a vision model.  Conservative estimate ensures image-heavy
        requests route to the slow client (60s timeout) instead of the
        fast client (18s), which would kill them mid-inference.

        When `system` is provided, also estimates system prompt tokens.
        Used as fallback when the streaming API doesn't report input_tokens.

        Returns an upper-bound estimate suitable for timeout selection.
        """
        def _text_tokens(text: str) -> int:
            if not text:
                return 0
            cjk = sum(1 for ch in text if '一' <= ch <= '鿿'
                      or '㐀' <= ch <= '䶿'
                      or '豈' <= ch <= '﫿'
                      or '　' <= ch <= '〿')  # CJK punctuation
            return cjk * 2 // 3 + (len(text) - cjk) // 4

        total = 0
        # System prompt estimation
        if system is not None:
            if isinstance(system, str):
                total += _text_tokens(system)
            elif isinstance(system, list):
                for block in system:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += _text_tokens(str(block.get("text", "")))
        # Messages estimation
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += _text_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        total += _text_tokens(str(block.get("text", "")))
                    elif block.get("type") == "image":
                        total += 2000  # Per-image vision token estimate
        return total

    def _pick_client(self, messages: list[dict[str, Any]]) -> Anthropic:
        """Select fast or slow Anthropic client based on estimated input size."""
        est = self._estimate_tokens(messages)
        if est >= self._TOKEN_ESTIMATE_THRESHOLD:
            return self._client_slow
        return self._client

    def chat(
        self,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking: dict[str, Any] | None = None,
    ) -> Message:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else config.llm.temperature,
            "system": system,
            "messages": self._convert_messages(messages),
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
        if thinking is not None:
            kwargs["thinking"] = thinking

        t0 = time.monotonic()
        picked = self._pick_client(messages)

        # ── Primary key ──
        try:
            response = _retry_call(picked.messages.create, **kwargs)
        except Exception as e:
            # Connection errors are usually infrastructure-level — try a
            # different API key before giving up.  Status/rate-limit errors
            # are key-wide, so key rotation won't help (retries handled above).
            keys = config.llm.api_keys
            if not _is_connection_error(e) or len(keys) <= 1:
                raise

            backup_index = (self._api_key_index + 1) % len(keys)
            # Don't rotate back to ourselves
            if backup_index == self._api_key_index:
                raise
            backup_key = config.llm.get_api_key(backup_index)
            logger.warning(
                "Primary key (#%d) exhausted — trying backup key (#%d)",
                self._api_key_index, backup_index,
            )

            backup_client = Anthropic(
                base_url=config.llm.base_url,
                api_key=backup_key,
                max_retries=0,
                default_headers={
                    "api-key": backup_key,
                    "anthropic-beta": "prompt-caching-2024-07-31",
                },
                http_client=httpx.Client(timeout=self._SLOW_TIMEOUT),
            )
            try:
                response = _retry_call(
                    backup_client.messages.create,
                    **kwargs,
                    max_attempts=3,  # shorter leash on backup
                )
                logger.info("Backup key (#%d) succeeded", backup_index)
            except Exception as e2:
                logger.error("Backup key (#%d) also failed: %s", backup_index, e2)
                raise e2
            finally:
                backup_client._client.close()  # type: ignore[attr-defined]

        elapsed = time.monotonic() - t0

        usage = response.usage
        _cache_info = ""
        if usage:
            _cr = getattr(usage, 'cache_read_input_tokens', None)
            _cc = getattr(usage, 'cache_creation_input_tokens', None)
            if _cr:
                _cache_info = f" cache_read={_cr}"
            if _cc:
                _cache_info += f" cache_create={_cc}"
        logger.info(
            "API: %.1fs input=%s output=%s%s",
            elapsed,
            usage.input_tokens if usage else "?",
            usage.output_tokens if usage else "?",
            _cache_info,
        )
        return response

    def chat_stream(
        self,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": config.llm.temperature,
            "system": system,
            "messages": self._convert_messages(messages),
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
        picked = self._pick_client(messages)
        return picked.messages.stream(**kwargs)

    def chat_stream_collect(
        self,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_text_delta: Any = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, list[dict[str, Any]], str, int, int, int, int]:
        """Stream LLM response, collecting text, tool_calls, thinking, and token counts.

        Calls on_text_delta(text_chunk) for each text delta (real-time WeChat).
        Returns (full_text, tool_calls, thinking, input_tokens, output_tokens,
        cache_read_tokens, cache_create_tokens) when the stream completes.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else config.llm.temperature,
            "system": system,
            "messages": self._convert_messages(messages),
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        t0 = time.monotonic()
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_blocks: dict[int, dict[str, Any]] = {}
        # Pre-estimate: MiMo proxy may not report input_tokens in streaming
        # message_start events.  Use our own estimate as a fallback so that
        # token budget tracking still works in streaming mode.
        _est_input = self._estimate_tokens(messages, system=system)
        input_tokens = _est_input
        output_tokens = 0
        _cache_read: int | None = None
        _cache_create: int | None = None
        picked = self._pick_client(messages)

        def _stream_once():
            nonlocal input_tokens, output_tokens, _cache_read, _cache_create
            # Clear accumulators on each retry to avoid duplicating deltas
            text_parts.clear()
            thinking_parts.clear()
            tool_blocks.clear()
            tool_calls.clear()
            # Reset to estimate on retry
            input_tokens = _est_input
            with picked.messages.stream(**kwargs) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            idx = event.index
                            tool_blocks[idx] = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input": {},
                            }
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            chunk = event.delta.text
                            text_parts.append(chunk)
                            if on_text_delta:
                                try:
                                    on_text_delta(chunk)
                                except Exception:
                                    pass
                        elif event.delta.type == "thinking_delta":
                            thinking_parts.append(event.delta.thinking)
                        elif event.delta.type == "input_json_delta":
                            idx = event.index
                            if idx in tool_blocks:
                                tool_blocks[idx]["input_json"] = (
                                    tool_blocks[idx].get("input_json", "") + event.delta.partial_json
                                )
                    elif event.type == "content_block_stop":
                        idx = event.index
                        if idx in tool_blocks:
                            import json as _json
                            blk = tool_blocks[idx]
                            raw = blk.pop("input_json", "{}")
                            try:
                                blk["input"] = _json.loads(raw)
                            except _json.JSONDecodeError:
                                blk["input"] = {}
                            tool_calls.append({"id": blk["id"], "name": blk["name"], "input": blk["input"]})
                    elif event.type == "message_start":
                        if event.message.usage:
                            input_tokens = event.message.usage.input_tokens
                            if event.message.usage.cache_read_input_tokens:
                                _cache_read = event.message.usage.cache_read_input_tokens
                    elif event.type == "message_delta":
                        if event.usage:
                            output_tokens = event.usage.output_tokens
                            if event.usage.input_tokens:
                                input_tokens = event.usage.input_tokens
                            if event.usage.cache_read_input_tokens:
                                _cache_read = event.usage.cache_read_input_tokens
                            if event.usage.cache_creation_input_tokens:
                                _cache_create = event.usage.cache_creation_input_tokens
                    elif event.type == "message_stop":
                        # Anthropic sends cache usage in message_stop
                        pass

        _retry_call(_stream_once)

        elapsed = time.monotonic() - t0

        # Log cache hit/miss info when available
        _cache_info = ""
        if _cache_read is not None:
            _cache_info += f" cache_read={_cache_read}"
        if _cache_create is not None:
            _cache_info += f" cache_create={_cache_create}"

        logger.info(
            "API(stream): %.1fs input=%s output=%s%s",
            elapsed,
            input_tokens,
            output_tokens,
            _cache_info,
        )
        return "".join(text_parts), tool_calls, "".join(thinking_parts), input_tokens, output_tokens, _cache_read or 0, _cache_create or 0

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[MessageParam]:
        converted: list[MessageParam] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                converted.append(MessageParam(role=role, content=content))  # type: ignore[arg-type]
            elif isinstance(content, list):
                converted.append(MessageParam(role=role, content=content))  # type: ignore[arg-type]
            else:
                converted.append(MessageParam(role=role, content=str(content)))  # type: ignore[arg-type]
        return converted

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[ToolParam]:
        result: list[ToolParam] = []
        for t in tools:
            result.append(
                ToolParam(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("parameters", {}),
                )
            )
        return result


def extract_tool_calls(response: Message) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for block in response.content:
        if block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "input": block.input if isinstance(block.input, dict) else {},
            })
    return tool_calls


def extract_text(response: Message) -> str:
    """Extract assistant text from a response.

    Uses text blocks when present, thinking blocks as fallback (for models
    that put the answer in thinking/redacted blocks without a text block).
    Never concatenates both — doing so creates duplicate output when the
    model repeats thinking content in the text block (common with MiMo).
    """
    text_parts: list[str] = []
    think_parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "thinking":
            think_parts.append(block.thinking)
    # Prefer text blocks — they are the model's intended visible output.
    # Only use thinking blocks when the model didn't produce a text block
    # (MiMo extended thinking without a separate text response).
    if text_parts:
        return "\n".join(text_parts)
    return "\n".join(think_parts)


def extract_thinking(response: Message) -> str:
    parts: list[str] = []
    for block in response.content:
        if block.type == "thinking":
            parts.append(block.thinking)
    return "\n".join(parts)


# ── Client pool ───────────────────────────────────────────────────

import threading as _threading

_client_pool: list[MiMoClient] = []
_pool_lock = _threading.Lock()
_pool_max_size = 8  # 4 is too tight for 2 agents (each may need main + compression + rerank)
_pool_key_counter: int = 0   # Round-robin across API keys


def acquire_client(key_index: int | None = None) -> MiMoClient:
    """Get a MiMoClient from the pool or create a new one.

    When key_index is None, picks the next API key round-robin from
    config.llm.api_keys so multiple agents spread across different keys,
    bypassing per-key concurrency limits on the MiMo API server.

    Callers must call release_client() when done (or use the context manager).
    """
    global _pool_key_counter
    if key_index is None:
        with _pool_lock:
            key_index = _pool_key_counter
            _pool_key_counter = (_pool_key_counter + 1) % max(len(config.llm.api_keys), 1)
    with _pool_lock:
        # Try to find a pooled client with the requested key_index
        for i, c in enumerate(_client_pool):
            if getattr(c, '_api_key_index', 0) == key_index:
                client = _client_pool.pop(i)
                logger.debug("Reusing client from pool (key=%d, remaining: %d)",
                           key_index, len(_client_pool))
                return client
        # If pool is empty (any key), return the last one — key mismatch is
        # better than creating an unbounded number of clients
        if _client_pool:
            client = _client_pool.pop()
            logger.debug("Reusing client from pool (key mismatch, remaining: %d)", len(_client_pool))
            return client
    logger.debug("Creating new MiMoClient (key_index=%d)", key_index)
    return MiMoClient(api_key_index=key_index)


def release_client(client: MiMoClient) -> None:
    """Return a MiMoClient to the pool for reuse.

    If the pool is full or the client is closed, it is discarded.

    Prefer using the pooled_client() context manager instead of calling
    acquire/release manually — it guarantees release even on exceptions.
    """
    if client._closed:
        return
    with _pool_lock:
        if len(_client_pool) < _pool_max_size:
            _client_pool.append(client)
            logger.debug("Returned client to pool (size: %d)", len(_client_pool))
        else:
            client.close()


from contextlib import contextmanager


@contextmanager
def pooled_client():
    """Context manager that acquires + releases a MiMoClient from the pool.

    Usage:
        with pooled_client() as client:
            response = client.chat(...)

    Guarantees release_client() even if an exception occurs.
    Prefer this over manual acquire_client()/release_client() pairs.
    """
    client = acquire_client()
    try:
        yield client
    finally:
        release_client(client)

