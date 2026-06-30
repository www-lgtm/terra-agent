"""LLM JSON extraction — robust parsing with verify-retry loop.

Three places in the codebase parse JSON from LLM output:
  1. Multi-game task splitting  (agent.py:_extract_sub_tasks)
  2. Schedule create parsing    (weixin.py:_handle_schedule_create)
  3. Concierge tool-call params (anthropic SDK handles this, but fallback)

All three had the same pattern: try json.loads → if fail, silent fallback.
This module adds a verify-retry loop so the LLM gets one more chance to
fix its output before we give up.
"""

from __future__ import annotations

import json as _json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


def extract_json_block(raw: str) -> str | None:
    """Extract the largest JSON object {...} from raw LLM output.

    Tries three strategies in order:
    1. Markdown code fence: ```json ... ```
    2. Largest balanced-brace block (handles LLM adding text around the JSON)
    3. Raw string as-is
    """
    raw = raw.strip()
    if not raw:
        return None

    # Strategy 1: markdown code fence
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if candidate:
            return candidate

    # Strategy 2: largest balanced-brace block
    braces: list[int] = []
    best_start, best_end = -1, -1
    for i, ch in enumerate(raw):
        if ch == '{':
            braces.append(i)
        elif ch == '}' and braces:
            start = braces.pop()
            if not braces and (i - start) > (best_end - best_start):
                best_start, best_end = start, i + 1

    if best_start >= 0:
        return raw[best_start:best_end]

    return None


def llm_json_call(
    system: str,
    user_text: str,
    *,
    max_retries: int = 2,
    max_tokens: int = 200,
    client_factory: Callable[[], Any] | None = None,
    **extra_options: Any,
) -> dict[str, Any]:
    """Call LLM, parse JSON response, retry with error feedback.

    Args:
        system: System prompt that instructs the LLM to output JSON.
        user_text: The user message to parse.
        max_retries: How many times to retry on JSON parse failure (default 2).
        max_tokens: Max tokens for the LLM response.
        client_factory: Optional factory for MiMoClient (defaults to MiMoClient()).
        extra_options: Passed through to client.chat() (e.g. temperature).

    Returns:
        Parsed dict.

    Raises:
        ValueError: If the LLM cannot produce valid JSON after all retries.
    """
    if client_factory is None:
        from src.llm.client import acquire_client as _acquire
        from src.llm.client import release_client as _release
        _use_pool = True
    else:
        _acquire = client_factory
        _release = lambda c: c.close()
        _use_pool = False

    from src.llm.client import extract_text as _extract_text

    # Prefix enforcement: make JSON-only requirement unambiguous even for
    # models that tend to output reasoning text (e.g. MiMo).  Wrap the
    # caller's system prompt so the last instruction is always "只输出JSON".
    _JSON_ONLY_SUFFIX = (
        "\n\n★★★ 关键：你的整个回复必须是纯 JSON。"
        "不要解释、不要分析、不要问候、不要 markdown 围栏。"
        "从 { 开始，到 } 结束。除此之外一个字符都别输出。★★★"
    )
    system = system.rstrip() + _JSON_ONLY_SUFFIX

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
    last_raw = ""

    for attempt in range(max_retries):
        client = _acquire()
        try:
            response = client.chat(
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                thinking={"type": "disabled"},
                **extra_options,
            )
            raw = _extract_text(response).strip()
        except Exception as exc:
            logger.warning("llm_json_call LLM error (attempt %d/%d): %s",
                          attempt + 1, max_retries, exc)
            if attempt == max_retries - 1:
                raise ValueError(
                    f"LLM call failed after {max_retries} attempts: {exc}"
                ) from exc
            messages.append({
                "role": "user",
                "content": "调用失败，请重试。只输出合法 JSON。",
            })
            continue
        finally:
            _release(client)

        last_raw = raw
        block = extract_json_block(raw)

        if block is None:
            logger.warning("llm_json_call: no JSON block found (attempt %d/%d): %s",
                          attempt + 1, max_retries, raw[:200])
            if attempt < max_retries - 1:
                messages.append({
                    "role": "user",
                    "content": (
                        "你的回复中没有找到 JSON。"
                        "重要：现在只输出 JSON。不要解释。从 { 开始写。"
                    ),
                })
                continue
            raise ValueError(f"No JSON block found in LLM response: {raw[:300]}")

        try:
            result = _json.loads(block)
        except _json.JSONDecodeError as exc:
            logger.warning("llm_json_call parse error (attempt %d/%d): %s — raw: %s",
                          attempt + 1, max_retries, exc, raw[:200])
            if attempt < max_retries - 1:
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user",
                     "content": f"JSON 格式错误: {exc}。重新输出合法 JSON。"},
                ])
                continue
            raise ValueError(
                f"LLM returned invalid JSON after {max_retries} attempts: {raw[:300]}"
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(f"LLM returned non-dict JSON: {type(result).__name__}")

        return result

    raise AssertionError("unreachable")  # pragma: no cover
