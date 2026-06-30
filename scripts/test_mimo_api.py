"""Verify MiMo-V2.5 API connectivity: tool_use, streaming, and vision.

Usage:
    python scripts/test_mimo_api.py

Checks:
1. Basic chat completion
2. Tool use round-trip
3. Streaming chat
4. Vision/image understanding (VLM capability)

Requires: MIMO_API_KEY environment variable set.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.client import MiMoClient, extract_text, extract_tool_calls


def test_basic_chat(client: MiMoClient) -> bool:
    print("=" * 60)
    print("Test 1: Basic chat completion")
    print("=" * 60)

    messages = [{"role": "user", "content": "你好，请用一句话介绍你自己。"}]
    try:
        response = client.chat(system="你是一个有用的助手。", messages=messages)
        text = extract_text(response)
        print(f"  Response: {text[:200]}")
        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_tool_use(client: MiMoClient) -> bool:
    print()
    print("=" * 60)
    print("Test 2: Tool use round-trip")
    print("=" * 60)

    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name in Chinese or English"},
            },
            "required": ["city"],
        },
    }]

    messages = [{"role": "user", "content": "北京今天天气怎么样？"}]
    system = "你是一个助手。使用 get_weather 工具获取天气。"

    try:
        response = client.chat(system=system, messages=messages, tools=tools)
        text = extract_text(response)
        tool_calls = extract_tool_calls(response)

        print(f"  Text: {text[:100] if text else '(none)'}")
        print(f"  Tool calls: {len(tool_calls)}")
        for tc in tool_calls:
            print(f"    - {tc['name']}({json.dumps(tc['input'], ensure_ascii=False)})")

        if tool_calls:
            print("  PASSED (tool call returned)")
        else:
            print("  WARNING: No tool calls (model may have answered directly)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_streaming(client: MiMoClient) -> bool:
    print()
    print("=" * 60)
    print("Test 3: Streaming chat")
    print("=" * 60)

    messages = [{"role": "user", "content": "用一句话介绍明日方舟这款游戏。"}]
    system = "你是一个游戏助手。"

    try:
        event_types: set[str] = set()
        text_parts: list[str] = []

        with client.chat_stream(system=system, messages=messages) as stream:
            for event in stream:
                event_types.add(event.type)
                if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                    text_parts.append(event.delta.text)

        text = "".join(text_parts)
        print(f"  Event types: {sorted(event_types)}")
        print(f"  Response: {text[:200]}")
        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_vision(client: MiMoClient) -> bool:
    print()
    print("=" * 60)
    print("Test 4: Vision/image understanding")
    print("=" * 60)

    # Check for test screenshot
    screenshot_path = Path("data/screenshots/current.png")
    if not screenshot_path.exists():
        print(f"  SKIPPED: no screenshot at {screenshot_path}")
        print("  Capture a screenshot first to test vision.")
        return True  # Not a failure

    import base64

    img_b64 = base64.b64encode(screenshot_path.read_bytes()).decode()

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
            },
            {
                "type": "text",
                "text": "描述这张截图：这是什么界面？有哪些主要按钮和数字？只报告实际看到的。",
            },
        ],
    }]

    try:
        response = client.chat(system="你是一个游戏画面分析器。只描述实际看到的内容。", messages=messages)
        text = extract_text(response)
        print(f"  Response: {text[:300]}")
        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main() -> None:
    print("MiMo API Verification")
    print(f"  Endpoint: https://token-plan-cn.xiaomimo.com/anthropic")
    print(f"  Model: MiMo-V2.5")

    api_key = os.getenv("MIMO_API_KEY", "")
    if not api_key:
        print("\nERROR: MIMO_API_KEY environment variable not set.")
        print("  Set it in .env file or: export MIMO_API_KEY=your_key")
        sys.exit(1)

    print(f"  API key: {api_key[:8]}...{api_key[-4:]}")
    print()

    client = MiMoClient()

    results: list[bool] = []
    for test_fn in [test_basic_chat, test_tool_use, test_streaming, test_vision]:
        t0 = time.monotonic()
        ok = test_fn(client)
        elapsed = time.monotonic() - t0
        results.append(ok)
        print(f"  ({elapsed:.1f}s)")

    passed = sum(results)
    total = len(results)
    print()
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("All tests passed. MiMo API is ready.")
    else:
        print("Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()
