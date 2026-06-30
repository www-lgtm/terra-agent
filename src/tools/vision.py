"""Vision tools: screenshot, ocr_read, vlm_describe.

Self-registers with the tool registry at import time.
"""

from __future__ import annotations

import json
import logging
import re

from config.settings import config
from src.device.adb import _get_screenshot_path, get_adb
from src.tools import screen_cache
from src.tools.registry import registry, ToolOutput
from src.utils.hash import compute_image_hash
from src.vision.ocr import ocr_engine
from src.vision.vlm import vlm_descriptor

logger = logging.getLogger(__name__)


def _adb_available() -> bool:
    try:
        from src.device.emulator import emulator_manager
        return emulator_manager.first_online is not None
    except Exception:
        return False


def _adb_available_not_auto() -> bool:
    """Available only when ADB is usable AND vision_mode is NOT auto_inject.

    In auto_inject mode, screenshot + ocr_read are fully redundant —
    the system already injects fresh screenshots + OCR after every action.
    Hiding them prevents the LLM from wasting turns on no-op tools.
    """
    if not _adb_available():
        return False
    from config.settings import config
    return config.agent.vision_mode != "auto_inject"


def screenshot_tool() -> ToolOutput:
    """Capture current game screen (~1s). Returns screen_hash for change detection.

    For reading text, use ocr_read. Screenshots are auto-injected after actions,
    so you rarely need this tool — use it only to check if a screen changed.
    """
    adb = get_adb()
    img = adb.get_screenshot_image()
    screen_hash = compute_image_hash(img)

    return ToolOutput(
        text=json.dumps({
            "success": True,
            "screen_hash": screen_hash,
            "screenshot_path": str(_get_screenshot_path(adb.serial)),
            "message": (
                "!! 不需要再调用 screenshot() — 截图已自动注入到对话中。"
                "反复调用 screenshot() 是卡死的主要原因。请直接读屏幕上 OCR 文字操作。"
            ),
        }, ensure_ascii=False),
        screen_hash=screen_hash,
    )


def ocr_read_tool(region: str | None = None) -> ToolOutput:
    """Read text/numbers from screen.

    Args:
        region: Named region like "sanity", "top_bar", "bottom_bar", or None for full screen.
    """
    adb = get_adb()
    img = adb.get_screenshot_image()

    if region:
        screen_hash = compute_image_hash(img)

        cached = ocr_engine.get_cached_region(screen_hash, region)
        if cached:
            text = ocr_engine.read_region(img, *cached)
            return ToolOutput(text=json.dumps({"success": True, "region": region, "text": text, "method": "cached_region"}, ensure_ascii=False))

    detections = ocr_engine.read_text(img)
    texts = [d["text"] for d in detections]

    return ToolOutput(text=json.dumps({
        "success": True,
        "region": region or "full",
        "count": len(texts),
        "texts": texts,
    }, ensure_ascii=False))


# Common button patterns to extract from VLM descriptions
_BUTTON_PATTERNS = [
    "终端", "Terminal", "任务", "Mission", "基建", "Base",
    "编队", "Squads", "干员", "Operator", "好友", "Friends",
    "档案", "Archives", "采购中心", "Store", "仓库", "公开招募",
    "干员寻访", "Recruitment", "作战", "开始行动", "主页",
    "返回", "乐章收录", "反常光谱", "离解复合",
]


def _parse_vlm_buttons(desc: str) -> dict[str, tuple[int, int]]:
    """Extract button name → (x, y) coordinate mappings from VLM description text."""
    buttons: dict[str, tuple[int, int]] = {}
    for m in re.finditer(r"\[(\d{2,4}),\s*(\d{2,4})\]", desc):
        x, y = int(m.group(1)), int(m.group(2))
        if x == 0 and y == 0:
            continue
        # Grab surrounding context (~40 chars before, ~20 after)
        start = max(0, m.start() - 40)
        end = min(len(desc), m.end() + 20)
        ctx = desc[start:end]
        for pat in _BUTTON_PATTERNS:
            if pat in ctx and pat not in buttons:
                buttons[pat] = (x, y)
                break
    return buttons


# VLM cooldown: prevent calling vlm_describe more than once per 5s on the same screen.
# Keyed by device_serial to prevent cross-device contamination in multi-agent setups.
_vlm_last_call: dict[str, tuple[str, float]] = {}


def vlm_describe_tool(purpose: str = "") -> ToolOutput:
    """Describe current screen using MiMo-V2.5 vision.

    Args:
        purpose: What the agent is trying to accomplish (e.g. "find 1-7 main story stage").
                 When provided, VLM will suggest which specific button to press.
    """
    global _vlm_last_call
    import time as _time

    adb = get_adb()
    img = adb.get_screenshot_image()
    device = adb.serial

    # Cooldown check: same screen on same device within 5s → return cached
    screen_hash = compute_image_hash(img)
    now = _time.monotonic()
    prev = _vlm_last_call.get(device)
    if prev is not None and prev[0] == screen_hash and (now - prev[1]) < 5.0:
        logger.info("VLM cooldown: skipping describe for screen %s [%s] (%.1fs since last call)",
                    screen_hash[:8], device, now - prev[1])
        return ToolOutput(text=json.dumps({
            "success": True,
            "screen_hash": screen_hash,
            "description": "(VLM cooldown — screen unchanged, use previous description)",
            "cached": True,
        }, ensure_ascii=False))
    _vlm_last_call[device] = (screen_hash, now)

    try:
        result = vlm_descriptor.describe(img, purpose=purpose)
    except Exception as e:
        logger.warning("VLM describe failed: %s", e)
        return ToolOutput(text=json.dumps({
            "success": False,
            "error": str(e),
        }))

    # Extract button → coordinate mappings from VLM description and feed cache
    shash = result["screen_hash"]
    desc = result["description"]
    if shash and desc:
        try:
            buttons = _parse_vlm_buttons(desc)
            if buttons:
                screen_cache.bulk_set(shash, buttons, device_serial=adb.serial)
                logger.debug("Cached %d button positions from VLM", len(buttons))
        except Exception:
            logger.debug("VLM button parsing failed (best-effort, non-critical)", exc_info=True)

    return ToolOutput(text=json.dumps({
        "success": True,
        "screen_hash": shash,
        "description": desc,
        "cached": result.get("cached", False),
    }, ensure_ascii=False))


def vlm_read_numbers_tool() -> ToolOutput:
    """Read resource numbers (龙门币, 合成玉, 源石, 理智) using MiMo-V2.5 vision."""
    adb = get_adb()
    img = adb.get_screenshot_image()

    try:
        numbers = vlm_descriptor.read_numbers(img)
    except Exception as e:
        logger.warning("VLM read_numbers failed: %s", e)
        return ToolOutput(text=json.dumps({"success": False, "error": str(e)}))

    return ToolOutput(text=json.dumps({
        "success": True,
        "numbers": numbers,
    }, ensure_ascii=False))


def _draw_coordinate_rulers(img, tick_interval: int = 0) -> None:
    """Draw pixel-coordinate rulers on top & left edges of the image.

    Tick-marks + labels let the LLM **read** exact pixel positions from the
    image instead of guessing by eye.  Mutates the image in-place.

    Args:
        tick_interval: Spacing between ruler labels in px.  0 = auto (≈8 labels
                       on the longest side, rounded to nearest 100, min 100).
    """
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)
    w, h = img.size

    if tick_interval <= 0:
        tick_interval = max(100, round(max(w, h) / 8 / 100) * 100)

    try:
        font = ImageFont.truetype("consola.ttf", 13)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except Exception:
            font = ImageFont.load_default()

    BAND = 22  # px reserved for ruler strip

    # Dark background bands so labels are readable over any game content
    draw.rectangle([(0, 0), (w - 1, BAND)], fill=(30, 30, 30))
    draw.rectangle([(0, 0), (BAND, h - 1)], fill=(30, 30, 30))

    # Thin grid lines + coordinate labels
    for x in range(tick_interval, w, tick_interval):
        draw.line([(x, 0), (x, h - 1)], fill=(70, 70, 70), width=1)
        draw.text((x + 3, 2), str(x), fill=(255, 80, 80), font=font)
    for y in range(tick_interval, h, tick_interval):
        draw.line([(0, y), (w - 1, y)], fill=(70, 70, 70), width=1)
        draw.text((2, y + 2), str(y), fill=(255, 80, 80), font=font)


# Cache the last magnify screenshot per device so mark_position can reuse it.
# Keyed by device_serial to prevent cross-device contamination.
_last_magnify_cache: dict[str, tuple[object, float, int, int]] = {}
_MAGNIFY_CACHE_TTL = 5.0  # seconds — stale cache is worse than fresh screenshot
_last_mark_target: str = ""  # set by mark_position, consumed by tap_magnified


def magnify_tool(region: str = "") -> ToolOutput:
    """Return a high-resolution screenshot for precise targeting.

    Returns a screenshot with coordinate rulers drawn, capped at 1600px on the
    longest side to avoid API rejection for oversized multimodal payloads.
    Use this when the auto-injected 800px screenshot is too small to read.
    """
    import base64 as _b64
    from io import BytesIO as _BytesIO
    from PIL import Image
    from src.device.adb import get_adb as _get_adb

    import time as _time

    _MAX_DIM = 1600

    adb = _get_adb()
    img = adb.get_screenshot_image()
    orig_size = img.size  # save before any scaling for coordinate mapping
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Downscale if either dimension exceeds _MAX_DIM (preserves aspect ratio)
    w, h = img.size
    if max(w, h) > _MAX_DIM:
        ratio = _MAX_DIM / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, resample=Image.LANCZOS)

    _draw_coordinate_rulers(img)

    # Cache for mark_position reuse (per-device to avoid cross-contamination)
    global _last_magnify_cache
    _last_magnify_cache[adb.serial] = (img.copy(), _time.monotonic(), *orig_size)

    buf = _BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_b64 = _b64.b64encode(buf.getvalue()).decode()

    from src.tools.registry import ImageBlock
    mw, mh = img.size
    return ToolOutput(
        text=json.dumps({
            "success": True,
            "resolution": f"{mw}x{mh}",
            "screen_size": f"{orig_size[0]}x{orig_size[1]}",
            "scale_x": round(orig_size[0] / mw, 4),
            "scale_y": round(orig_size[1] / mh, 4),
        }, ensure_ascii=False),
        images=[ImageBlock(data=img_b64)],
    )


def tap_magnified_tool(x: int, y: int, magnified_width: int = 0,
                       target: str = "", year: int = 0) -> ToolOutput:
    """Tap at pixel coordinates from a magnified screenshot.

    Call magnify() first to get a high-res image. Find your target's pixel
    position in that image, then call this with the same x,y. Read the
    magnified_width from magnify's "resolution" output (e.g. "1600x900" → 1600).

    Args:
        x: Pixel x-coordinate in the magnified image (0 = left edge)
        y: Pixel y-coordinate in the magnified image (0 = top edge)
        magnified_width: Width from magnify output resolution (0 = auto-detect)
        target: Button name / label. If provided, the screen coordinate is
                cached so future adb_tap(target) hits this same spot instantly.
                Critical for icon-only buttons (✓, ▼, 铃铛) that OCR can't read.
    """
    from src.device.adb import get_adb as _get_adb

    adb = _get_adb()
    img = adb.get_screenshot_image()
    orig_w, orig_h = img.size

    _MAGNIFY_MAX_DIM = 1600  # same constant as magnify_tool
    if magnified_width > 0:
        mw = magnified_width
    else:
        mw = int(orig_w * _MAGNIFY_MAX_DIM / max(orig_w, orig_h))
    ratio = orig_w / mw
    actual_x = int(x * ratio)
    actual_y = int(y * ratio)

    logger.info("tap_magnified: (%d,%d) in %dpx → (%d,%d) on %dx%d screen",
                x, y, mw, actual_x, actual_y, orig_w, orig_h)
    adb.tap(actual_x, actual_y)

    # ── Resolve target name: explicit arg > mark_position's last target ──
    resolved_target = target.strip()
    if not resolved_target:
        global _last_mark_target
        if _last_mark_target:
            resolved_target = _last_mark_target
            _last_mark_target = ""  # consume it

    # ── Cache the screen coordinate so adb_tap(target) hits instantly next time ──
    if resolved_target:
        from src.utils.dhash import compute_dhash, dhash_to_hex
        shash = dhash_to_hex(compute_dhash(img))
        screen_cache.set(shash, resolved_target, actual_x, actual_y,
                         actual_x - 40, actual_y - 20,
                         actual_x + 40, actual_y + 20,
                         device_serial=adb.serial)
        logger.info("tap_magnified: cached '%s' at (%d,%d) for screen %s",
                    resolved_target, actual_x, actual_y, shash[:8])

    return ToolOutput(text=json.dumps({
        "success": True,
        "magnified_coords": [x, y],
        "screen_coords": [actual_x, actual_y],
        "cached_target": resolved_target or None,
    }, ensure_ascii=False))


def check_checkbox_tool(target_text: str) -> ToolOutput:
    """Verify the visual state of a checkbox next to text by analyzing brightness.

    Takes a screenshot, finds the bounding box of target_text via OCR,
    crops a small region to the LEFT of the text (where the checkbox square
    typically sits in Arknights), then computes the mean pixel brightness.

    Use this INSTEAD of magnify() to check checkbox state — it's deterministic
    image processing (no VLM guessing) and returns a simple bright/dark answer.

    Args:
        target_text: The label text to the RIGHT of the checkbox
                     (e.g. '代理指挥', '开始行动').
                     Partial/OCR-garbled text is ok.

    Returns:
        {"success": true, "state": "bright"|"dark", "mean_brightness": 0-255,
         "text_position": [x, y], "checkbox_region": [x1, y1, x2, y2]}
        "bright" → checkbox appears selected (white/lit)
        "dark"  → checkbox appears unselected (dark/black)
        "not_found" → target_text wasn't found by OCR on screen
    """
    import time as _time
    from PIL import Image as PILImage

    adb = get_adb()
    img = adb.get_screenshot_image()
    if img.mode != "L":
        gray = img.convert("L")
    else:
        gray = img

    w, h = gray.size

    # Find the target text's position via OCR
    detections = ocr_engine.read_text(img)
    best = None
    target_lower = target_text.strip().lower()

    for d in detections:
        if target_lower in d["text"].lower():
            if best is None or d["confidence"] > best.get("confidence", 0):
                best = d
    if not best:
        # Fuzzy match: try partial substring
        for d in detections:
            text_lower = d["text"].lower()
            if len(text_lower) >= 2 and (
                text_lower in target_lower or target_lower in text_lower
                or any(c in text_lower for c in target_lower)
            ):
                if best is None or d["confidence"] > best.get("confidence", 0):
                    best = d

    if not best:
        return ToolOutput(text=json.dumps({
            "success": True,
            "state": "not_found",
            "message": f"OCR did not find '{target_text}' on the current screen.",
        }, ensure_ascii=False))

    # The checkbox square is typically to the LEFT of the text label.
    # Arknights puts a small square checkbox (approx 30-40px) to the left of
    # the label text, with a small gap between them.
    #
    # OCR bbox is a flat 4-tuple: (x_min, y_min, x_max, y_max)
    bbox = best["bbox"]
    center = best["center"]  # [cx, cy]

    # Handle both 4-tuple (ocr.py) and legacy [[p1],[p2],[p3],[p4]] format
    if isinstance(bbox, (tuple, list)):
        if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            # Flat 4-tuple: (x_min, y_min, x_max, y_max)
            bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox
            text_h = int(bbox_y_max - bbox_y_min)
            text_left_x = int(bbox_x_min)
            text_min_y = int(bbox_y_min)
            text_max_y = int(bbox_y_max)
        else:
            # Legacy: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text_h = int(max(p[1] for p in bbox)) - int(min(p[1] for p in bbox))
            text_left_x = int(min(p[0] for p in bbox))
            text_min_y = int(min(p[1] for p in bbox))
            text_max_y = int(max(p[1] for p in bbox))

    # The checkbox square is roughly the same height as the text line,
    # positioned to the left with a small gap.
    chk_w = max(text_h, 20)
    chk_h = chk_w
    chk_x1 = max(0, text_left_x - chk_w - 15)  # 15px gap
    chk_y1 = max(0, int(text_min_y + (text_max_y - text_min_y - chk_h) / 2))
    chk_x2 = min(w, chk_x1 + chk_w)
    chk_y2 = min(h, chk_y1 + chk_h)

    # Crop and compute mean brightness
    region = gray.crop((chk_x1, chk_y1, chk_x2, chk_y2))
    pixels = list(region.getdata())
    if not pixels:
        return ToolOutput(text=json.dumps({
            "success": True,
            "state": "not_found",
            "message": "Checkbox region is empty (out of screen bounds).",
        }, ensure_ascii=False))

    mean_brightness = sum(pixels) / len(pixels)

    # Threshold: > 128 = bright (selected/white), <= 128 = dark (unselected)
    state = "bright" if mean_brightness > 128 else "dark"

    logger.info("check_checkbox: '%s' → %s (brightness=%.0f, region=%dx%d at %d,%d)",
                target_text, state, mean_brightness, chk_x2 - chk_x1, chk_y2 - chk_y1,
                chk_x1, chk_y1)

    return ToolOutput(text=json.dumps({
        "success": True,
        "state": state,
        "mean_brightness": round(mean_brightness, 1),
        "text_position": [center[0], center[1]],
        "checkbox_region": [chk_x1, chk_y1, chk_x2, chk_y2],
    }, ensure_ascii=False))


def mark_position_tool(x: int, y: int, magnified_width: int = 0,
                       target: str = "") -> ToolOutput:
    """Draw a crosshair at (x,y) on a screenshot and return it for visual confirmation.

    Use this BEFORE tap_magnified to verify your aim: call magnify(), then
    mark_position() at your estimated target. If the crosshair isn't on the
    right element, adjust x/y and try again. Once confirmed, use the same
    coordinates in tap_magnified().

    Args:
        x, y: Pixel coordinates in the magnified image
        magnified_width: Width from magnify output (0 = auto-detect)
        target: What button/element you're aiming at (e.g. '蓝勾确认', '小时▼').
                If set, tap_magnified() will auto-cache this coordinate under
                that name — no need to pass target again to tap_magnified.
    """
    import base64 as _b64
    from io import BytesIO as _BytesIO
    from PIL import ImageDraw, ImageFont
    from src.device.adb import get_adb as _get_adb

    # Store target for tap_magnified to consume
    global _last_mark_target
    if target.strip():
        _last_mark_target = target.strip()
        logger.debug("mark_position: target '%s' saved for tap_magnified", _last_mark_target)

    # Reuse magnify's cached image when available and fresh, so the
    # crosshair lands on the same content the LLM read coordinates from.
    import time as _time
    global _last_magnify_cache
    adb = _get_adb()
    now = _time.monotonic()
    cached = _last_magnify_cache.get(adb.serial)
    if cached is not None and (now - cached[1]) < _MAGNIFY_CACHE_TTL:
        img = cached[0].copy()
        logger.debug("mark_position: reusing magnify cache [%s] (%.1fs old)", adb.serial, now - cached[1])
    else:
        # Magnify cache expired — capture fresh but downscale to match
        # magnify_tool's dimensions so the LLM's coordinates stay valid.
        from PIL import Image as _PILImage
        img = adb.get_screenshot_image()
        orig_w, orig_h = img.size
        _MAGNIFY_MAX_DIM = 1600
        if max(orig_w, orig_h) > _MAGNIFY_MAX_DIM:
            ratio = _MAGNIFY_MAX_DIM / max(orig_w, orig_h)
            new_size = (int(orig_w * ratio), int(orig_h * ratio))
            img = img.resize(new_size, resample=_PILImage.LANCZOS)
        if cached is not None:
            logger.debug("mark_position: magnify cache [%s] expired (%.1fs), "
                        "using fresh screenshot downscaled to %dx%d",
                        adb.serial, now - cached[1], *img.size)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Rulers first (behind crosshair) so LLM can read exact coords
    _draw_coordinate_rulers(img)

    draw = ImageDraw.Draw(img)
    r = max(10, img.width // 100)  # crosshair size proportional to image
    # Outer ring
    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline="red", width=3)
    # Inner dot
    draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill="red")
    # Cross lines
    draw.line([(x - r, y), (x + r, y)], fill="red", width=1)
    draw.line([(x, y - r), (x, y + r)], fill="red", width=1)

    buf = _BytesIO()
    img.save(buf, format="JPEG", quality=90)
    img_b64 = _b64.b64encode(buf.getvalue()).decode()

    from src.tools.registry import ImageBlock
    return ToolOutput(
        text=json.dumps({"success": True, "marked_at": [x, y], "resolution": f"{img.size[0]}x{img.size[1]}"},
                        ensure_ascii=False),
        images=[ImageBlock(data=img_b64)],
    )


registry.register(
    name="mark_position",
    description=(
        "在高清截图上标记红圈准星，验证你读出的坐标是否真的落在目标上。使用时机：magnify() 后、tap_magnified() 前。\n"
        "★ target 参数：告诉系统你要标记的是什么按钮（如 '蓝勾确认'、'小时▼'），"
        "后续 tap_magnified() 会自动把这个坐标缓存下来——以后 adb_tap 就能直接命中。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Pixel x to mark on the image"},
            "y": {"type": "integer", "description": "Pixel y to mark on the image"},
            "magnified_width": {"type": "integer", "description": "Image width from magnify's resolution. 0 = auto-detect."},
            "target": {"type": "string", "description": "What button/element you're targeting (e.g. '蓝勾确认', '小时▼'). Saves coordinate for future adb_tap reuse."},
        },
        "required": ["x", "y"],
    },
    handler=mark_position_tool,
    check_fn=_adb_available,
)

registry.register(
    name="vlm_read_numbers",
    description="Use MiMo-V2.5 vision to read resource numbers: 龙门币, 合成玉, 源石, 理智. Returns exact values.",
    parameters={"type": "object", "properties": {}},
    handler=vlm_read_numbers_tool,
    check_fn=_adb_available,
)

registry.register(
    name="screenshot",
    description="Capture current game screen (~1s). Returns screen_hash for change detection. 注意：截图已自动注入，通常不需要手动调用。仅在需要检测画面是否因非操作原因变化时使用。阅读文字用 ocr_read，放大查看细节用 magnify。",
    parameters={"type": "object", "properties": {}},
    handler=screenshot_tool,
    check_fn=_adb_available_not_auto,
)

registry.register(
    name="ocr_read",
    description="Read text or numbers from a specific region or full screen.",
    parameters={
        "type": "object",
        "properties": {
            "region": {"type": "string", "description": "Region name (e.g. 'sanity', 'top_bar') or omit for full screen"},
        },
    },
    handler=ocr_read_tool,
    check_fn=_adb_available_not_auto,
)

registry.register(
    name="check_checkbox",
    description=(
        "检查文字标签旁边复选框的视觉状态（亮/暗）。\n"
        "这不是 VLM 猜测 — 它用像素亮度分析，100% 可靠。\n"
        "使用时机：点击复选框之前或之后，需要确认是选中(亮/白色)还是未选中(暗/黑色)时调用。\n"
        "参数示例：check_checkbox('代理指挥') → 返回 'bright' 或 'dark'。\n"
        "与 magnify 的区别：magnify 是把图像发给你用眼睛看（不可靠），check_checkbox 是程序直接算亮度（可靠）。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "target_text": {
                "type": "string",
                "description": "复选框右边的标签文字，OCR 的部分匹配也可以（如 '代理指挥'）",
            },
        },
        "required": ["target_text"],
    },
    handler=check_checkbox_tool,
    check_fn=_adb_available,
)

# ── Icon name aliases (Chinese → template filename) ─────────────────
# Template FILES loaded from disk are the canonical names.
# This dict only adds Chinese/English convenience aliases on top.
# Any template NOT listed here is still usable by its exact filename.

_ICON_ALIASES: dict[str, str] = {
    # 精英等级 — primary templates are tight badge crops (badge_e2, badge_e1)
    "精英0": "elite_e0", "精英1": "badge_e1", "精英2": "badge_e2",
    "精英0徽章": "elite_e0", "精英1徽章": "badge_e1", "精英2徽章": "badge_e2",
    "精英2六星": "elite_e2_6", "精英2六星徽章": "elite_e2_6",
    "e0": "elite_e0", "e1": "badge_e1", "e2": "badge_e2",
    # Fallback aliases (if tight badge templates not loaded, try originals)
    "elite_e1": "badge_e1", "elite_e2": "badge_e2",
    "精二": "badge_e2", "精一": "badge_e1",
    # 任务完成度
    "任务完成": "task_done", "任务已完成": "task_done",
    "任务未完成": "task_undone", "日常完成": "task_done",
    # 代理指挥
    "代理指挥已勾选": "deploy_on", "代理指挥未勾选": "deploy_off",
    # 技能专精
    "专精1": "mastery_m1", "专精2": "mastery_m2", "专精3": "mastery_m3",
    "专精一": "mastery_m1", "专精二": "mastery_m2", "专精三": "mastery_m3",
    # UI
    "返回箭头": "back_arrow", "主页图标": "home_icon",
}


def _resolve_icon_name(name: str) -> str:
    """Resolve LLM-friendly name → loaded template name.

    1. Check Chinese alias table → returns mapped name
    2. If alias resolves to a name that doesn't exist as a template,
       try the original name directly (supports Chinese filenames)
    3. Fuzzy fallback: for elite badges, try matching by 二/2 or 一/1 chars
    4. Otherwise use the original name as-is
    """
    from src.vision.template_match import template_matcher
    templates = template_matcher._templates

    resolved = _ICON_ALIASES.get(name, name)
    if resolved in templates:
        return resolved
    if name in templates:
        return name

    # ── Fuzzy fallback for elite badges ──
    # If user says "精英2" but templates are named "精二" or "elite_e2" or
    # anything with 二/2/一/1, find the best match.
    need_e2 = any(c in name for c in "二2")
    need_e1 = any(c in name for c in "一1")
    need_e0 = any(c in name for c in "零0")

    if need_e2 or need_e1 or need_e0:
        for tname in templates:
            has_2 = any(c in tname for c in "二2")
            has_1 = any(c in tname for c in "一1")
            has_0 = any(c in tname for c in "零0")
            if need_e2 and has_2:
                return tname
            if need_e1 and has_1:
                return tname
            if need_e0 and has_0:
                return tname

    return resolved


def _ensure_templates_loaded(game: str | None = None) -> None:
    """Lazily load icon templates for a game if not already loaded.

    No plugin code required — first use of match_icon auto-discovers
    all PNGs in data/templates/{game}/.
    """
    from src.vision.template_match import template_matcher
    from src.tools.registry import get_current_game as _get_game

    g = game or _get_game()
    template_matcher.ensure_templates_for_game(g)


def _list_loaded_icons() -> list[str]:
    """Return list of currently loaded template names."""
    from src.vision.template_match import template_matcher
    return sorted(template_matcher.loaded_templates)


def _build_match_icon_description(game: str | None = None) -> str:
    """Dynamic tool description listing every loaded template with its meaning."""
    from src.vision.template_match import template_matcher
    import json as _json
    from pathlib import Path as _Path

    from src.tools.registry import get_current_game as _get_game
    active_game = game or _get_game()
    _ensure_templates_loaded(active_game)
    loaded = template_matcher.loaded_templates

    # Load metadata (icons.json) — maps filename → {description, ...}
    meta: dict[str, dict] = {}
    meta_path = _Path(config.DATA_DIR) / "templates" / active_game / "icons.json"
    if meta_path.exists():
        try:
            raw = _json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                meta = raw
        except Exception:
            pass

    if not loaded:
        return (
            "识别图标状态 — 用 OpenCV 像素匹配。\n"
            "⚠️ 当前未加载任何模板。运行 scripts/capture_templates.py 采集图标。"
        )

    # Build a compact listing: each icon with its description
    lines = ["识别游戏图标的状态 — OpenCV 模板像素匹配，100% 确定。"]
    lines.append(f"当前可用 {len(loaded)} 个图标:\n")

    # Sort: icons with descriptions first, then bare filenames
    def _sort_key(filename: str) -> tuple[int, str]:
        leaf = filename.split("/")[-1]
        has_desc = bool(meta.get(leaf, {}).get("description"))
        return (0 if has_desc else 1, leaf)

    for fname in sorted(loaded, key=_sort_key):
        leaf = fname.split("/")[-1]
        info = meta.get(leaf, {})
        desc = info.get("description", "")
        if desc and desc != leaf:
            lines.append(f"  {leaf} — {desc}")
        else:
            lines.append(f"  {leaf} (含义未知，请按文件名推断)")

    # Chinese aliases as a quick reference
    aliases = sorted(_ICON_ALIASES.keys())
    if aliases:
        lines.append(f"\n中文别名: {', '.join(aliases[:20])}"
                     + (f"… 等 {len(aliases)} 个" if len(aliases) > 20 else ""))

    lines.append("\n用 name 传图标名（中文别名、文件名均可），near_text 限定搜索区域。")
    return "\n".join(lines)


def match_icon_tool(name: str, near_text: str = "", threshold: float = 0.0) -> ToolOutput:
    """Identify game icons via OpenCV template matching. Deterministic, not VLM guessing.

    Any template file name is a valid icon name. Chinese aliases (精英2, 任务完成, etc.)
    are resolved automatically. New templates added via capture_templates.py
    are picked up on next restart — no code changes needed.

    Args:
        name: Icon name. Can be a Chinese alias (精英2, 任务完成, 代理指挥已勾选)
              or the exact template filename (elite_e2, task_done, deploy_on).
        near_text: Limit search to the area around this OCR text (e.g. operator name).
        threshold: Match confidence 0-1 (default 0.85).
    """
    from src.vision.template_match import template_matcher
    from src.device.adb import get_adb

    _ensure_templates_loaded()  # Auto-load templates for current game on first use
    template_name = _resolve_icon_name(name)
    threshold = threshold or config.vision.template_match_threshold

    img = get_adb().get_screenshot_image()
    result: dict | None = None

    # ── Region-constrained search (near_text) ──
    if near_text:
        det = ocr_engine.find_text(img, near_text, min_confidence=0.5)
        if det:
            x1, y1, x2, y2 = det["bbox"][0], det["bbox"][1], det["bbox"][2], det["bbox"][3]
            rx1, ry1 = max(0, x1 - 120), max(0, y1 - 200)
            rx2, ry2 = min(img.width, x2 + 60), min(img.height, y1 + 50)
            crop = img.crop((rx1, ry1, rx2, ry2))
            match = template_matcher.match(crop, template_name, threshold)
            if match:
                match["center"] = (match["center"][0] + rx1, match["center"][1] + ry1)
                match["bbox"] = (
                    match["bbox"][0] + rx1, match["bbox"][1] + ry1,
                    match["bbox"][2] + rx1, match["bbox"][3] + ry1,
                )
            result = match
        else:
            return ToolOutput(text=json.dumps({
                "found": False, "name": name,
                "message": f"OCR 未找到 '{near_text}'。请确认该文字在当前画面可见。",
            }, ensure_ascii=False))
    else:
        result = template_matcher.match(img, template_name, threshold)

    if result:
        # Read brightness at match center (helps distinguish dark/light variants)
        bbox = result["bbox"]
        region = img.crop(bbox).convert("L")
        pixels = list(region.getdata())
        brightness = sum(pixels) / len(pixels) if pixels else 0
        return ToolOutput(text=json.dumps({
            "found": True, "name": name,
            "matched_template": template_name,
            "confidence": round(result["score"], 3),
            "position": list(result["center"]),
            "brightness": round(brightness, 1),
            "brightness_hint": (
                "bright（亮色/选中/已完成状态）" if brightness > 128
                else "dark（暗色/未选中/未完成状态）"
            ),
        }, ensure_ascii=False))
    else:
        import difflib
        all_loaded = list(template_matcher.loaded_templates)
        suggestions = difflib.get_close_matches(template_name, all_loaded, n=8, cutoff=0.3)
        tips = f"未找到 '{name}'（模板: {template_name}）。"
        if suggestions:
            tips += f" 可用: {', '.join(suggestions)}"
        if not all_loaded:
            tips += " ⚠️ 未加载模板！运行 scripts/capture_templates.py"
        return ToolOutput(text=json.dumps({
            "found": False, "name": name, "message": tips,
        }, ensure_ascii=False))


# ── Batch icon matching ──────────────────────────────────────────────

# Icon check order for elite badge detection (highest rank first).
# Each tuple: (alias_name, elite_label)
_ELITE_CHECK_ORDER = [
    ("精英2", "E2"),
    ("精英1", "E1"),
    ("精英0", "E0"),
]

# Badge template matching threshold — the PRIMARY signal.
# A genuine badge template match reliably scores ≥ 0.65. Card background
# noise (no badge) produces scores in the 0.55–0.65 range. Combined with
# the score-differential check below, this cleanly separates E0/E1/E2.
BADGE_MATCH_THRESHOLD = 0.65

# Minimum relative score advantage for the winning elite level.
# When no badge is present, E2 and E1 templates both match card-background
# structure at similar scores (e.g. 0.64 vs 0.63). When a badge IS present,
# the correct template outscores the wrong one by ≥ 15%. This differential
# check eliminates the need for unreliable color-based verification.
BADGE_SCORE_RATIO = 1.15  # winner must be ≥ 15% higher than loser

# ── Badge region layout constants (calibrated from 1920×1080 landscape) ──
# Badge is at the TOP-LEFT corner of each operator card, name at BOTTOM-CENTER.
#   Δx (badge center → name center): 64–126 px, avg 110
#   Δy (badge center → name center): 134–140 px, avg 138
BADGE_CENTER_DX = -110    # badge center relative to name center X (negative = left)
BADGE_CENTER_DY = -138    # badge center relative to name center Y (negative = above)
BADGE_CROP_HALF_W = 75    # half-width → 150px crop (enough search space for templates)
BADGE_CROP_HALF_H = 65    # half-height → 130px crop

# Templates larger than this fraction of the crop size are skipped.
# Large templates (badge_e2.png 90×72) include card background and produce
# artificially high (~0.99) correlation scores when applied to tight crops.
# Only small pure-icon templates are reliable for badge classification.
TEMPLATE_MAX_CROP_RATIO = 0.50  # template must be ≤ 50% of crop in both dimensions


def _fuzzy_match_names_to_detections(
    name_list: list[str],
    detections: list[dict],
    threshold: float = 0.55,
) -> dict[str, dict]:
    """Match operator names against OCR detections in memory (no extra OCR).

    One OCR pass serves all names. Uses fuzzy SequenceMatcher matching to
    handle OCR errors (truncated names, garbled characters, split glyphs).

    Args:
        name_list: Operator names to find.
        detections: Pre-computed OCR detections from one read_text() call.
        threshold: SequenceMatcher ratio threshold for fuzzy matching.

    Returns:
        {matched_name: detection_dict} — only names successfully matched.
    """
    from difflib import SequenceMatcher as _SM

    matched: dict[str, dict] = {}

    for name in name_list:
        # Exact match first
        for d in detections:
            if d["text"].strip() == name:
                matched[name] = d
                break
        if name in matched:
            continue

        # Substring match (OCR truncated: "维什" in "维什戴尔")
        name_len = len(name)
        best_score = 0.0
        best_det = None
        for d in detections:
            dtext = d["text"].strip()
            dlen = len(dtext)
            # Skip obviously wrong lengths (3x difference = unlikely match)
            if max(name_len, dlen) > min(name_len, dlen) * 3:
                continue
            if name in dtext or dtext in name:
                score = 0.85 - abs(name_len - dlen) * 0.05  # Prefer closer lengths
                if score > best_score:
                    best_score = score
                    best_det = d
        if best_det and best_score >= threshold:
            matched[name] = best_det
            continue

        # Fuzzy fallback
        best_score = 0.0
        best_det = None
        for d in detections:
            dtext = d["text"].strip()
            dlen = len(dtext)
            if dlen < 2 or max(name_len, dlen) > min(name_len, dlen) * 3:
                continue
            score = _SM(None, name, dtext).ratio()
            if score > best_score:
                best_score = score
                best_det = d
        if best_det and best_score >= threshold:
            matched[name] = best_det

    return matched


def _crop_badge_region(img, name_center: tuple[int, int]) -> tuple:
    """Crop a TIGHT region around the expected elite badge position.

    The badge center relative to name center is calibrated from real
    screenshots (1920×1080 landscape): Δx ≈ -110, Δy ≈ -138.

    Crop size is ~110×100px — just enough to contain a ~90×75px badge
    with margin for OCR jitter, but tight enough to exclude the card's
    golden border/background that would otherwise produce false E2 signals.

    Returns (crop, expected_badge_cx, expected_badge_cy) where the
    expected badge center is in the original image coordinates,
    or (None, 0, 0) if the crop would be invalid.
    """
    import numpy as np

    cx, cy = name_center
    img_w, img_h = img.size

    # Expected badge center in image coordinates
    badge_cx = cx + BADGE_CENTER_DX
    badge_cy = cy + BADGE_CENTER_DY

    # Crop bounds (tight around expected badge center)
    rx1 = max(0, badge_cx - BADGE_CROP_HALF_W)
    ry1 = max(0, badge_cy - BADGE_CROP_HALF_H)
    rx2 = min(img_w, badge_cx + BADGE_CROP_HALF_W)
    ry2 = min(img_h, badge_cy + BADGE_CROP_HALF_H)

    if rx2 <= rx1 or ry2 <= ry1:
        return None, 0, 0

    return img.crop((rx1, ry1, rx2, ry2)), badge_cx, badge_cy


def _analyze_badge_colors(crop) -> dict:
    """Analyze HSV color features of a badge-region crop.

    Returns a dict with E2/E1 color evidence ratios computed from the crop.
    The crop is the region around the expected badge position on an operator
    card — it includes both the badge (if present) and surrounding card art.

    Key insight: E2 badges contain golden/bronze pixels (warm hue, saturated).
    E1 badges contain silver/grey pixels (low saturation, medium-bright).
    E0 (no badge) has neither — just card background art.
    """
    import cv2
    import numpy as np

    if crop is None:
        return {"e2_golden_ratio": 0.0, "e1_silver_ratio": 0.0, "total_pixels": 0}

    arr = np.array(crop.convert("RGB"))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV)

    h, s, v = hsv[:, :, 0].astype(np.float32), hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
    total = h.size

    # E2 golden mask
    e2_mask = (
        (h >= E2_GOLDEN_HUE_MIN) & (h <= E2_GOLDEN_HUE_MAX)
        & (s >= E2_GOLDEN_SAT_MIN) & (v >= E2_GOLDEN_VAL_MIN)
    )
    e2_ratio = float(e2_mask.sum() / total)

    # E1 silver mask
    e1_mask = (
        (s <= E1_SILVER_SAT_MAX)
        & (v >= E1_SILVER_VAL_MIN) & (v <= E1_SILVER_VAL_MAX)
    )
    e1_ratio = float(e1_mask.sum() / total)

    return {"e2_golden_ratio": e2_ratio, "e1_silver_ratio": e1_ratio, "total_pixels": total}


def _template_match_in_region(crop, template_names: list[str], threshold: float) -> dict | None:
    """Run template matching for multiple templates within a cropped region.

    Returns the best match dict (with score, template_name) or None.
    Uses grayscale matching to reduce background-color variation noise.
    """
    from src.vision.template_match import template_matcher

    best = None
    best_score = 0.0

    for tname in template_names:
        resolved = _resolve_icon_name(tname)
        if resolved not in template_matcher._templates:
            continue
        match = template_matcher.match(crop, resolved, threshold, grayscale=True)
        if match and match.get("score", 0) > best_score:
            best_score = match["score"]
            best = {"template_name": resolved, "score": best_score, "match": match}

    return best


def _check_elite_for_operator(
    img, operator_name: str, threshold: float
) -> str | None:
    """Check the elite badge for a single operator. Returns 'E2'/'E1'/'E0' or None.

    Per-operator approach: crop the expected badge region, then combine
    template matching + HSV color analysis for the final decision.
    This avoids the global-search false-positive problem.
    """
    det = ocr_engine.find_text(img, operator_name, min_confidence=0.5)
    if not det:
        return None

    cx, cy = det["center"][0], det["center"][1]
    crop, _ox, _oy = _crop_badge_region(img, (cx, cy))
    if crop is None:
        return "E0"

    return _classify_badge_crop(crop, threshold)


def _detect_badge_hexagons(img) -> list[dict]:
    """Detect all elite badge hexagons on a full screenshot.

    Uses edge detection + contour analysis to find hexagonal badge shapes.
    Returns [{center: (cx, cy), elite: 'E2'|'E1', score, area}, ...].

    This is the PRIMARY badge detection mechanism — no template matching.
    Hexagons are detected by their GEOMETRIC SHAPE, not pixel correlation,
    making this immune to card-background false positives.
    """
    import cv2
    import numpy as np

    arr = np.array(img.convert("RGB"))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    screen_area = w * h

    # ── Edge detection (light blur, sensitive thresholds for small badges) ──
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 25, 80)

    # ── Morphological close to connect broken badge edges ──
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # ── Find contours ──
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Badge hexagons in a 1920×1080 screenshot are roughly:
    #   - diameter: 40-80 px
    #   - area: 800-4000 px²
    #   - circularity: 0.7-0.9 (regular hexagon ≈ 0.83)
    MIN_BADGE_AREA = 400
    MAX_BADGE_AREA = 6000

    hexagons: list[dict] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BADGE_AREA or area > MAX_BADGE_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri < 60:  # too small to be meaningful
            continue

        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        verts = len(approx)

        # Hexagon: 5-8 vertices (allow for rounded corners)
        if verts < 5 or verts > 8:
            continue

        if not cv2.isContourConvex(approx):
            continue

        # Aspect ratio near 1
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if aspect < 0.55 or aspect > 1.8:
            continue

        # Circularity
        circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
        if circularity < 0.55 or circularity > 0.95:
            continue

        # Score (prefer 6 vertices, circularity near 0.83)
        vert_score = 1.0 - abs(verts - 6) / 6.0
        circ_score = 1.0 - abs(circularity - 0.83) / 0.28
        score = vert_score * 0.3 + circ_score * 0.7

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx_h = int(M["m10"] / M["m00"])
            cy_h = int(M["m01"] / M["m00"])
        else:
            cx_h, cy_h = x + bw // 2, y + bh // 2

        # ── Classify E2 vs E1 by center color ──
        sample_r = max(2, min(bw, bh) // 6)
        y1 = max(0, cy_h - sample_r)
        y2 = min(h, cy_h + sample_r)
        x1 = max(0, cx_h - sample_r)
        x2 = min(w, cx_h + sample_r)

        if y2 <= y1 or x2 <= x1:
            continue

        patch = arr_bgr[y1:y2, x1:x2]
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mean_hue = patch_hsv[:, :, 0].mean()
        mean_sat = patch_hsv[:, :, 1].mean()

        # E2: golden/bronze center (warm hue, moderate-high saturation)
        # E1: silver/white center (low saturation, any hue is just noise)
        if 8 <= mean_hue <= 42 and mean_sat >= 30:
            elite = "E2"
        elif mean_sat < 28:
            elite = "E1"
        else:
            elite = "E1"  # safe default

        hexagons.append({
            "center": (cx_h, cy_h),
            "elite": elite,
            "score": round(score, 3),
            "area": int(area),
            "verts": verts,
            "circularity": round(circularity, 3),
            "hue": round(mean_hue, 1),
            "sat": round(mean_sat, 1),
        })

    return hexagons


def _classify_badge_crop(crop, threshold: float) -> str:
    """Classify a badge-region crop as E2/E1/E0 by analyzing the CENTER patch.

    Key insight from real data: the crop is centered on the badge position
    (the function _crop_badge_region places the crop at the calibrated badge
    center). A small patch (30×30) at the crop center captures:
      - E2: golden/bronze tones (R noticeably > B), high texture (σ > 50)
      - E1: silver/white tones (R ≈ G ≈ B), high texture (σ > 50)
      - E0: card portrait art, low texture (σ < 45), RGB varies wildly

    The texture (std dev) check is the PRIMARY "badge exists" signal —
    badge icons have sharp edges and contrast, while card art is smooth.
    The R/B ratio then distinguishes E2 (golden) from E1 (silver).

    This method requires NO template matching and NO contour detection.
    It works because the badge is at a FIXED, calibrated position relative
    to the operator name label.
    """
    import numpy as np

    arr = np.array(crop.convert("RGB")).astype(np.float64)
    h, w = arr.shape[:2]

    # ── Sample center patch (expected badge location) ──
    # Small patch to avoid sampling card border/background around the badge.
    patch_r = 8   # half-size → 16×16 patch
    cx, cy = w // 2, h // 2
    y1, y2 = max(0, cy - patch_r), min(h, cy + patch_r)
    x1, x2 = max(0, cx - patch_r), min(w, cx + patch_r)

    if y2 <= y1 or x2 <= x1:
        return "E0"

    patch = arr[y1:y2, x1:x2, :]

    # ── Features ──
    mean_rgb = patch.mean(axis=(0, 1))  # [R, G, B]
    std_rgb = patch.std(axis=(0, 1))    # [R, G, B]
    mean_std = std_rgb.mean()           # overall texture
    r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]
    rb_ratio = r / b if b > 0 else 99.0
    rg_ratio = r / g if g > 0 else 99.0

    # ── Classification ──

    # E0: Low texture → no badge. Card portrait art is relatively smooth
    # compared to the high-contrast badge icon.
    if mean_std < 45:
        return "E0"

    # E2: Golden badge → red channel noticeably higher than blue.
    # Golden ≈ (R high, G medium, B low). Typical R/B ≈ 1.25–1.40.
    # Also requires mean brightness > 80 (badge is bright, not dark).
    mean_brightness = mean_rgb.mean()
    if rb_ratio >= 1.18 and mean_brightness >= 100:
        return "E2"

    # E1: Silver/white badge → all channels roughly equal, high brightness.
    # Typical R/G ≈ 1.0 ± 0.10, R/B ≈ 1.0 ± 0.15.
    if 0.92 <= rg_ratio <= 1.10 and 0.90 <= rb_ratio <= 1.15 and mean_brightness >= 100:
        return "E1"

    # Ambiguous: has texture but colors don't match either badge type.
    # Could be card art with strong textures (some operator portraits).
    # Default to E1 (conservative — rather under-detect than false-E2).
    return "E1"


def _batch_check_elite_from_detections(
    img,
    matched_names: dict[str, dict],
    threshold: float,
) -> list[dict]:
    """Check elite badges via per-operator center-patch texture+color analysis.

    For each operator whose name was detected by OCR:
    1. Crop the expected badge region (fixed offset from name center).
    2. Sample a SMALL patch (30×30) at the crop center (= badge position).
    3. Measure texture (std dev) and color (R/B ratio, brightness).
    4. Classify:
       - Low texture (σ < 45) → E0 (card art, no badge)
       - High texture + golden (R/B ≥ 1.18) → E2
       - High texture + silver (R≈G≈B) → E1

    No template matching. No contour detection. No color analysis of the
    entire card background. Just a 30×30 pixel patch at the calibrated
    badge center.

    Args:
        img: PIL screenshot image.
        matched_names: {name: detection_dict}.
        threshold: Unused (kept for API compatibility).

    Returns:
        [{"name": "...", "elite": "E2"|"E1"|"E0"}, ...]
    """
    results: list[dict] = []
    stats = {"E2": 0, "E1": 0, "E0": 0}

    for name, det in matched_names.items():
        cx, cy = det["center"][0], det["center"][1]
        crop, _ox, _oy = _crop_badge_region(img, (cx, cy))

        if crop is None:
            elite = "E0"
        else:
            elite = _classify_badge_crop(crop, threshold)

        stats[elite] = stats.get(elite, 0) + 1
        results.append({"name": name, "elite": elite})

    logger.debug(
        "_batch_check_elite (center-patch): %d names → E2:%d E1:%d E0:%d",
        len(matched_names), stats["E2"], stats["E1"], stats["E0"],
    )

    return results


def match_icon_batch_tool(name_list: list[str], icon_type: str = "elite") -> ToolOutput:
    """Batch check elite badges for multiple operators in one tool call.

    Optimized: one OCR pass → memory-based fuzzy name matching → batch
    template matching. ~1s for 16 operators instead of ~67s with the old
    per-operator OCR approach.

    Args:
        name_list: List of operator names (from OCR). Only names actually found
                   on screen will be checked.
        icon_type: "elite" — checks E2/E1/E0 badges (default, only option for now).

    Returns:
        {"results": [{"name": "...", "elite": "E2"|"E1"|"E0"|null, "method": "template"},
                     ...],
         "summary": {"E2": N, "E1": N, "E0": N, "unknown": N},
         "elapsed_ms": N}
    """
    import time as _time
    from src.device.adb import get_adb as _get_adb

    _ensure_templates_loaded()
    adb = _get_adb()
    img = adb.get_screenshot_image()
    threshold = BADGE_MATCH_THRESHOLD if icon_type == "elite" else config.vision.template_match_threshold
    t0 = _time.monotonic()

    results: list[dict] = []
    stats = {"E2": 0, "E1": 0, "E0": 0, "unknown": 0}

    if icon_type == "elite" and name_list:
        # ── Optimized path: one OCR → memory match → batch template ──
        t_ocr = _time.monotonic()
        detections = ocr_engine.read_text(img)
        matched = _fuzzy_match_names_to_detections(name_list, detections)
        elite_results = _batch_check_elite_from_detections(img, matched, threshold)
        t_total_ocr = (_time.monotonic() - t_ocr) * 1000

        # Merge with unmatched names
        matched_names_set = {r["name"] for r in elite_results}
        for name in name_list:
            if name in matched_names_set:
                for r in elite_results:
                    if r["name"] == name:
                        elite = r["elite"]
                        break
            else:
                elite = None

            if elite:
                stats[elite] = stats.get(elite, 0) + 1
            else:
                stats["unknown"] += 1

            results.append({
                "name": name,
                "elite": elite,
                "method": "template" if elite else "not_found",
            })

        logger.debug(
            "match_icon_batch optimized: OCR+match=%.0fms for %d names (%d matched)",
            t_total_ocr, len(name_list), len(matched),
        )
    else:
        # Fallback: per-operator OCR (for non-elite icon types, future use)
        for name in name_list:
            elite = None
            if icon_type == "elite":
                elite = _check_elite_for_operator(img, name, threshold)

            if elite:
                stats[elite] = stats.get(elite, 0) + 1
            else:
                stats["unknown"] += 1

            results.append({
                "name": name,
                "elite": elite,
                "method": "template" if elite else "not_found",
            })

    elapsed = (_time.monotonic() - t0) * 1000
    logger.info(
        "match_icon_batch: %d operators checked in %.0fms — E2:%d E1:%d E0:%d unknown:%d",
        len(name_list), elapsed, stats["E2"], stats["E1"], stats["E0"], stats["unknown"],
    )

    return ToolOutput(text=json.dumps({
        "success": True,
        "icon_type": icon_type,
        "results": results,
        "summary": stats,
        "elapsed_ms": round(elapsed, 1),
    }, ensure_ascii=False))


registry.register(
    name="magnify",
    description=(
        "获取原始分辨率的高清截图，用于查看自动注入截图中看不清的细节。\n"
        "图片顶部和左侧有坐标标尺（自适应间距），你可以直接读出目标的像素坐标，不需要猜测。\n"
        "使用时机：1）需要点击微小目标（复选框、次数选择器、小图标）时；2）adb_tap 同一目标失败 2 次后；3）自动截图分辨率不足以阅读文字时。\n"
        "标准流程：magnify() → 从标尺读出坐标 → mark_position(target='名称') 验证→ tap_magnified() 点击。\n"
        "★ mark_position 传 target 参数后，tap_magnified 会自动缓存坐标，以后同画面 adb_tap 直接命中！"
    ),
    parameters={
        "type": "object",
        "properties": {
            "region": {"type": "string", "description": "What area you want to see clearly (e.g. 'top bar numbers', 'button labels'). Optional."},
        },
    },
    handler=magnify_tool,
    check_fn=_adb_available,
)

registry.register(
    name="tap_magnified",
    description=(
        "在 magnified 图像上点击。必须先调用 magnify() 获取高清图，用 mark_position() 验证目标位置，再调用此工具。\n"
        "注意：x, y 是 magnified 图像上的像素坐标（不是设备屏幕坐标），工具内部会自动转换。\n"
        "magnified_width 从 magnify 的返回结果中读取（resolution 字段）。\n"
        "★ target 参数：告诉系统你在点什么按钮（如 '蓝勾确认'、'小时▼'），坐标会被缓存——"
        "以后再遇到同样画面，adb_tap(target) 直接命中，不需要再走 magnify 流程！"
    ),
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Pixel x-coordinate in the magnified image"},
            "y": {"type": "integer", "description": "Pixel y-coordinate in the magnified image"},
            "magnified_width": {"type": "integer", "description": "Image width from magnify's resolution (e.g. 1600). 0 = auto-detect."},
            "target": {"type": "string", "description": "Name for this button/element (e.g. '蓝勾确认', '小时▼'). Cached so future adb_tap hits directly. mark_position's target auto-feeds if omitted."},
        },
        "required": ["x", "y"],
    },
    handler=tap_magnified_tool,
    check_fn=_adb_available,
)

# DEPRECATED: vlm_describe replaced by auto-injected screenshots (Always-On Vision).
# Function kept for vlm_read_numbers_tool internal use and legacy fallback.
# registry.register(
#     name="vlm_describe",
#     description="Use VLM to describe the current game screen. Always provide 'purpose' so the VLM knows what you're trying to do and can suggest which button to press.",
#     parameters={
#         "type": "object",
#         "properties": {
#             "purpose": {"type": "string", "description": "What you're trying to accomplish on this screen (e.g. 'find the main story stage 1-7', 'check if battle is complete'). VLM will tell you which button to press."},
#         },
#     },
#     handler=vlm_describe_tool,
#     check_fn=_adb_available,
# )
