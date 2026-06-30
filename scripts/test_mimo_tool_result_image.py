"""Test if MiMo endpoint supports image in tool_result mid-conversation.

This is the foundational POC for the VLM/LLM merge plan.
If this fails, the whole merge approach is blocked.

Scenarios tested:
  A. image in regular user message (baseline, known to work)
  B. image inside tool_result content block (the critical path for the merge plan)
  C. multi-turn: tool_result image → LLM responds → another tool_result image
"""

import base64
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import httpx
from anthropic import Anthropic
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

API_KEY = os.getenv("MIMO_API_KEY", "")
BASE_URL = "https://api.xiaomimimo.com/anthropic"
MODEL = "mimo-v2.5"

if not API_KEY:
    print("ERROR: MIMO_API_KEY not set in .env")
    sys.exit(1)

client = Anthropic(
    base_url=BASE_URL,
    api_key=API_KEY,
    default_headers={"api-key": API_KEY},
    http_client=httpx.Client(timeout=120.0),
)

# --- Generate a synthetic test image with text ---
def make_test_image(text: str) -> str:
    img = Image.new("RGB", (800, 400), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Draw a colored rectangle as a "button"
    draw.rectangle([100, 150, 300, 250], fill=(70, 130, 180), outline=(0, 0, 0), width=2)
    draw.rectangle([400, 150, 600, 250], fill=(60, 179, 113), outline=(0, 0, 0), width=2)
    draw.rectangle([100, 280, 300, 350], fill=(220, 140, 60), outline=(0, 0, 0), width=2)
    draw.text((150, 185), "终端", fill="white")
    draw.text((440, 185), "作战", fill="white")
    draw.text((160, 300), "物资筹备", fill="white")
    # Draw the text parameter at top
    draw.text((20, 20), text, fill=(0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=50)
    return base64.b64encode(buf.getvalue()).decode()


# --- Test A: Baseline - image in regular user message ---
def test_a_baseline():
    print("=" * 60)
    print("Test A: Image in regular user message (baseline)")
    print("=" * 60)
    img_b64 = make_test_image("Test A - regular user message")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": "这张图里有几个按钮？分别是什么文字？请简短回答。"},
            ],
        }
    ]

    try:
        t0 = time.monotonic()
        resp = client.messages.create(
            model=MODEL, max_tokens=512,
            system="你是一个助手。用中文简短回答。",
            messages=messages,
        )
        elapsed = time.monotonic() - t0
        text = "".join(b.text for b in resp.content if b.type == "text")
        print(f"  Latency: {elapsed:.1f}s")
        print(f"  Response: {text[:300]}")
        print(f"  PASS")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


# --- Test B: image inside tool_result ---
def test_b_tool_result_image():
    print("=" * 60)
    print("Test B: Image inside tool_result content (THE CRITICAL TEST)")
    print("=" * 60)
    img_b64 = make_test_image("Test B - tool_result image")

    messages = [
        # Turn 1: assistant calls a fake "look" tool
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "我需要先看一下屏幕。"},
                {"type": "tool_use", "id": "toolu_test_001", "name": "look", "input": {"purpose": "查看主界面"}},
            ],
        },
        # Turn 2: tool_result with image + OCR text
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_test_001",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                        {"type": "text", "text": '{"success":true,"screen_hash":"abc123","screen_texts":["终端","作战","物资筹备"]}'},
                    ],
                }
            ],
        },
    ]

    try:
        t0 = time.monotonic()
        resp = client.messages.create(
            model=MODEL, max_tokens=512,
            system="你是一个手机游戏助手，你可以看到游戏截图。根据截图内容选择下一步操作。用中文简短回答。",
            messages=messages,
        )
        elapsed = time.monotonic() - t0
        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_calls = []
        for b in resp.content:
            if b.type == "tool_use":
                tool_calls.append(f"{b.name}({b.input})")

        print(f"  Latency: {elapsed:.1f}s")
        print(f"  Text: {text[:300]}")
        if tool_calls:
            print(f"  Tool calls: {tool_calls}")
        print(f"  PASS - MiMo accepts image in tool_result!")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


# --- Test C: multi-turn with two look calls ---
def test_c_multiturn():
    print("=" * 60)
    print("Test C: Multi-turn - two tool_result images in conversation")
    print("=" * 60)
    img1_b64 = make_test_image("Test C - first screen")
    img2_b64 = make_test_image("Test C - second screen")

    messages = [
        {
            "role": "user",
            "content": "帮我在游戏主界面找到'作战'按钮并点击。",
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "先看看屏幕。"},
                {"type": "tool_use", "id": "toolu_test_c1", "name": "look", "input": {"purpose": "找作战按钮"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_test_c1",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img1_b64}},
                        {"type": "text", "text": '{"success":true,"screen_hash":"abc","screen_texts":["终端","作战","物资筹备"]}'},
                    ],
                }
            ],
        },
        # LLM should now tap "作战"
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "看到作战按钮了，点击。"},
                {"type": "tool_use", "id": "toolu_test_c2", "name": "adb_tap", "input": {"target": "作战"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_test_c2",
                    "content": '{"success":true,"method":"screen_cache"}',
                }
            ],
        },
        # Now look at new screen
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "确认一下当前屏幕。"},
                {"type": "tool_use", "id": "toolu_test_c3", "name": "look", "input": {"purpose": "确认进入作战界面"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_test_c3",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img2_b64}},
                        {"type": "text", "text": '{"success":true,"screen_hash":"def","screen_texts":["GT-6","1-7","开始行动"]}'},
                    ],
                }
            ],
        },
    ]

    try:
        t0 = time.monotonic()
        resp = client.messages.create(
            model=MODEL, max_tokens=512,
            system="你是一个手机游戏助手，可以看到游戏截图。根据截图选择下一步操作。用中文简短回答。",
            messages=messages,
        )
        elapsed = time.monotonic() - t0
        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_calls = []
        for b in resp.content:
            if b.type == "tool_use":
                tool_calls.append(f"{b.name}({b.input})")

        print(f"  Latency: {elapsed:.1f}s")
        print(f"  Text: {text[:300]}")
        if tool_calls:
            print(f"  Tool calls: {tool_calls}")
        print(f"  PASS - multi-turn with images works!")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


if __name__ == "__main__":
    print(f"Endpoint: {BASE_URL}")
    print(f"Model: {MODEL}")
    print()

    results = []
    results.append(("A. baseline", test_a_baseline()))
    results.append(("B. tool_result image", test_b_tool_result_image()))
    results.append(("C. multi-turn", test_c_multiturn()))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'OK' if ok else 'FAIL'}  {name}")
    all_ok = all(r[1] for r in results)
    print()
    if all_ok:
        print("ALL TESTS PASSED - MiMo supports images in tool_result! Merge plan is viable.")
    else:
        print("SOME TESTS FAILED - cannot proceed with the merge plan as designed.")
