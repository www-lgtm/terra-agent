"""CompressionService — async conversation history compression.

Extracted from TerraAgent._compress_async() and the compression trigger logic.

Manages a background compression thread: when history exceeds the threshold
(both by message count AND estimated token count), starts a daemon thread to
compress it; the main loop picks up the result on the next iteration.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _text_tokens(text: str) -> int:
    """Estimate token count for mixed CJK/ASCII text.

    Claude's tokenizer: ~1.5 CJK chars/token, ~4 ASCII chars/token.
    Using len()//4 underestimates CJK-heavy text by ~2.6x, which prevents
    compression from ever triggering in Chinese conversations.

    Returns an integer token estimate suitable for threshold gating.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if '一' <= ch <= '鿿'
              or '㐀' <= ch <= '䶿'
              or '豈' <= ch <= '﫿'
              or '　' <= ch <= '〿')  # CJK punctuation
    ascii_chars = len(text) - cjk
    # CJK: ~1.5 chars/token → 2/3 token per char
    # ASCII: ~4 chars/token → 1/4 token per char
    return cjk * 2 // 3 + ascii_chars // 4


def estimate_history_tokens(history: list[dict[str, Any]]) -> int:
    """Rough token count estimate for conversation history.

    Uses _text_tokens() which properly weights CJK (~1.5 chars/token) vs
    ASCII (~4 chars/token), so Chinese-heavy histories are no longer
    underestimated by 2-3x.

    Images: vision models encode screenshots as fixed-size tiles, not raw
    pixels.  A 800px-wide JPEG screenshot costs ~1500-2500 tokens regardless
    of compression level.  Using len(data)//4 would overestimate by 15-20x
    (a 150KB base64 string → 37,500 fake tokens vs ~2,000 real ones),
    causing premature compression after just 2-3 screenshots.

    Tool results: rough char/4 for JSON payloads.
    """
    # Per-image token estimate for a typical 800px-wide game screenshot.
    # Conservative: erring high prevents under-compression; erring too high
    # (old code) triggers compression when history is still small.
    _IMG_TOKEN_ESTIMATE = 2000

    total = 0
    for msg in history:
        content = msg.get("content")
        if isinstance(content, str):
            total += _text_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    total += _text_tokens(str(block.get("text", "")))
                elif block.get("type") == "image":
                    total += _IMG_TOKEN_ESTIMATE
                elif block.get("type") == "tool_use":
                    inp = block.get("input", {})
                    total += _text_tokens(str(inp)) + 20  # 20 for overhead
                elif block.get("type") == "tool_result":
                    result = block.get("content")
                    if isinstance(result, str):
                        total += _text_tokens(result)
                    elif isinstance(result, list):
                        # Extract actual text from text blocks — avoid Python
                        # repr inflation from str(list) which includes wrapping
                        # brackets, quotes, commas, and (critically) base64
                        # image payloads returned by vision tools.
                        for rb in result:
                            if isinstance(rb, dict) and rb.get("type") == "text":
                                total += _text_tokens(str(rb.get("text", "")))
    return total


class CompressionService:
    """Asynchronous conversation history compression manager."""

    MESSAGE_THRESHOLD = 40  # Messages before qualifying for compression
    TOKEN_THRESHOLD = 25_000  # Estimated tokens before triggering compression

    def __init__(self, client_pool: Any | None = None) -> None:
        self._pending: list[dict[str, Any]] | None = None
        self._running: bool = False
        self._client_pool = client_pool

    def check_and_swap(self, history: list[dict[str, Any]]) -> list | None:
        """Check if a compression result is ready, or fire a new one.

        Called on every iteration of the main loop.

        Only triggers compression when BOTH thresholds are crossed:
        - Message count ≥ MESSAGE_THRESHOLD (80)
        - Estimated tokens ≥ TOKEN_THRESHOLD (80K)

        This prevents unnecessary compression of short-text histories
        (many small messages) and ensures compression fires promptly
        for image-heavy histories (few messages but huge screenshots).

        Returns:
            The compressed history if a background result is ready (caller
            should swap it in), or None to continue with the current history.
        """
        if self._pending is not None:
            # Background thread finished — use its result
            compressed = self._pending
            self._pending = None
            logger.debug("Swapped in async-compressed history (%d msgs)", len(compressed))
            return compressed

        if not self._running:
            # Dual threshold: message count AND token estimate
            msg_count = len(history)
            if msg_count < self.MESSAGE_THRESHOLD:
                return None
            est_tokens = estimate_history_tokens(history)
            if est_tokens < self.TOKEN_THRESHOLD:
                return None
            logger.info(
                "Compression triggered: %d msgs, ~%d est tokens",
                msg_count, est_tokens,
            )

            # Fire async compression for next iteration
            self._running = True
            history_copy = list(history)
            t = threading.Thread(
                target=self._run_compress,
                args=(history_copy,),
                daemon=True,
            )
            t.start()
            logger.debug("Fired async compression (%d msgs)", len(history_copy))

        return None

    def _run_compress(self, history: list[dict[str, Any]]) -> None:
        """Background thread: compress conversation history via truncation."""
        from src.agent.compressor import compress_history
        try:
            self._pending = compress_history(history, keep_first=2, keep_last=25)
        except Exception as e:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", f"Async compression failed: {e}")
        finally:
            self._running = False

    def reset_for_task(self) -> None:
        """Reset per-task state."""
        self._pending = None
        self._running = False
