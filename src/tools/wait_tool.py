"""Wait tool: wait_for(condition, timeout).

Self-registers with the tool registry at import time.
"""

from __future__ import annotations

import json
import logging
import time

from src.device.adb import get_adb
from src.tools.registry import registry, ToolOutput
from src.vision.ocr import ocr_engine

logger = logging.getLogger(__name__)


def _adb_available() -> bool:
    try:
        from src.device.emulator import emulator_manager
        return emulator_manager.first_online is not None
    except Exception:
        return False


def wait_for(condition: str, timeout: float = 10.0) -> ToolOutput:
    """Wait until a condition is met on screen.

    Args:
        condition: Text to wait for. E.g. "行动结束" to wait for battle completion.
                   Prepending "!" means wait until text DISAPPEARS. E.g. "!loading".
        timeout: Maximum seconds to wait. Default 10s, max 60s.

    Returns success when condition met, or timeout error.
    """
    adb = get_adb()
    timeout = min(timeout, 60.0)

    wait_for_disappear = condition.startswith("!")
    target = condition[1:] if wait_for_disappear else condition

    check_interval = 2.0
    elapsed = 0.0
    start = time.monotonic()

    while elapsed < timeout:
        img = adb.get_screenshot_image()
        detections = ocr_engine.read_text(img)
        texts = [d["text"] for d in detections]

        found = any(target in t for t in texts)

        if wait_for_disappear and not found:
            return ToolOutput(text=json.dumps({
                "success": True,
                "condition": condition,
                "elapsed_seconds": round(elapsed, 1),
                "message": f"'{target}'已消失(等待{elapsed:.0f}秒)",
            }, ensure_ascii=False))
        elif not wait_for_disappear and found:
            return ToolOutput(text=json.dumps({
                "success": True,
                "condition": condition,
                "elapsed_seconds": round(elapsed, 1),
                "message": f"'{target}'已出现(等待{elapsed:.0f}秒)",
            }, ensure_ascii=False))

        time.sleep(check_interval)
        elapsed = time.monotonic() - start

    return ToolOutput(text=json.dumps({
        "success": False,
        "condition": condition,
        "elapsed_seconds": round(elapsed, 1),
        "message": f"等待'{condition}'超时({timeout}秒)",
        "current_texts": texts[:15],
    }, ensure_ascii=False))


# wait_for removed — navigation confirmation is handled by vlm_describe,
# and battle waiting no longer needs text polling.
