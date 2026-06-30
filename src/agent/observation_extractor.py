"""Observation Extractor — LLM-based guide generation from observed gameplay.

Supports two extraction modes:

  TEXT mode (default, backward-compatible):
    Reads the manifest, runs OCR on significant frames, builds a rich text
    prompt with inline frame images + click coordinates, and feeds it to
    the LLM.  The LLM calls create_guide() to save the resulting skill.

  VISION mode (vision_mode=True):
    Annotates each significant frame with frame# / timestamp / click markers
    + OCR text overlay, then feeds the full image sequence directly to the
    vision LLM.  No intermediate OCR pipeline — the model reads UI text
    straight from the pixels.  Higher accuracy, fewer pipeline stages,
    and the model cannot "invent" button names that don't appear on screen.

Usage:
    extractor = ObservationExtractor()
    skill_name = extractor.extract(
        manifest_path="/path/to/manifest.json",
        game="reverse1999",
        task_name="1999日常",
        vision_mode=True,        # NEW: use vision-native pipeline
        on_done=callback,
    )
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from config.settings import config as app_config

logger = logging.getLogger(__name__)

# Maximum number of significant frames to include in the LLM prompt.
# Context windows are large (100K+ tokens), but each frame ~30-50KB image.
_MAX_PROMPT_FRAMES = 20       # max images to embed (text mode)
_MAX_TEXT_FRAMES = 200       # text-only frames we can include (text mode)

# ── Vision-mode constants ────────────────────────────────────────────────

# How many annotated frames to feed the vision model.  Each frame takes
# ~3-5s of model processing time; 15 frames ≈ 45-75s, fitting under the
# 90s slow-client timeout with margin.
_MAX_VISION_FRAMES = 15

# Frame downscale width.  UI text remains readable at 480px; the smaller
# JPEGs (25KB vs 80KB at 800px) make the difference between a reliable API
# call and a timeout.
_VISION_FRAME_MAX_WIDTH = 480

# Click-target magnifier settings
_CLICK_CROP_SIZE_DEV = 200   # device-pixel square to crop around click, full-res
_CLICK_CROP_DISPLAY_W = 220  # display width for the magnified crop

# Frame annotation: click-marker radius in pixels.
_CLICK_MARKER_RADIUS = 18
# Frame annotation: font size for overlay text (frame#, timestamp).
_OVERLAY_FONT_SIZE = 16

# ── Vision-mode system prompt ────────────────────────────────────────────

_VISION_SYSTEM = """你正在观看一段 {game_name} 的屏幕录像。画面上的红色圆点是玩家的点击位置，旁边标注了坐标。

附近有**放大裁剪图**专门展示点击区域的 UI 文字/图标，帮你看清按钮内容。

阅读你输出的，是另一个能识别 UI 元素但没玩过这个游戏的 LLM agent。它需要**精确坐标**来点击。

## 🔴 硬性规则

### 1. 每个有红圈的关键帧 → 必须写 adb_tap_position
红圈旁边标注的坐标就是设备像素坐标。格式：
`adb_tap_position(x_pct, y_pct) # [x, y] 按钮文字/图标描述`
- x_pct = x / 设备宽度（画面已标注），y_pct = y / 设备高度，保留两位小数
- **禁止编造坐标**。数字必须和画面标注完全一致
- 放大裁剪图展示的是红圈位置的 UI 元素——用它写"按钮文字/图标描述"
- 如果两帧坐标相同，说明用户在同一位置点了两次（如双击确认），写两遍

### 2. 一一对应：每个关键帧 ≥ 一步
帧序列按时间排列。对照每个关键帧写出步骤，不跳帧不合并。F0000 是录像起点。

### 3. 禁止纯自然语言步骤
下面这种是**废品**，agent 无法执行：
  ❌ `adb_tap('底部绿色按钮')` —— OCR 匹配不到这种文字
  ❌ `点击右上角关闭按钮` —— 没有坐标
  ✅ `adb_tap_position(0.96, 0.12) # [1037, 230] 右上角"X"关闭`

### 4. 步骤里写出具体的 UI 文字
"点击'活动'入口" 比 "进入活动" 好。按钮上的文字从放大裁剪图读。

### 5. 无红圈的关键帧
如果该帧没有红圈（用户在看/等），写等待或描述：`# 等待约 X 秒（加载/战斗）`

## 自检（提交前）
1. 有红圈的帧，我都用了 adb_tap_position 且坐标正确？
2. 步骤数 ≥ 关键帧数？
3. 没有跳过任何帧？

如果任何答案是"否"，补全。

调用 create_guide(name, description, steps, game, pitfalls, tags)，然后 task_complete。
name: 简洁英文/拼音。description: 达成什么目标（不是"用户做了什么"）。
pitfalls: UI 陷阱（易混淆的画面、隐晦按钮等）。tags: 如 daily, multi-stage。"""


# ── Refinement-mode system prompt (diff against existing guide) ──────

_REFINE_VISION_SYSTEM = """你正在比较两样东西：
1. 一份**已有的操作手册**（用户消息里有全文）
2. 一段 **{game_name}** 的屏幕录像（标注了红圈坐标和放大裁剪图）

你的任务不是重写整份手册。只找出**录像里有但手册里没有/不对/不够精确的地方**，调用 append_to_guide 追加到手册末尾。

画面上的红色圆点是玩家点击位置，旁边标注了设备像素坐标 (x,y)。附近有**放大裁剪图**展示红圈位置的 UI 元素。

## 🔴 直接调用工具，不要描述画面

看完录像后，如果你发现了手册遗漏的内容，**立刻调用 append_to_guide**。不要在回复里描述你看到了什么——把发现直接写进工具参数。

## 三类可补充的内容

### A. 缺失的步骤 (new_steps)
手册里完全没提到的操作。每条必须：
- 引用帧号（如"F0311"）
- 给出坐标 `adb_tap_position(x_pct, y_pct) # [x, y] 按钮文字`
- 插入位置说明（插在手册哪两个阶段之间）

### B. 缺失的注意事项 (new_pitfalls)
手册没提到的 UI 陷阱/加载时间。引用帧号。

### C. 修正建议 (corrections)
手册里有但描述不准确的。

## 如果没有发现遗漏
直接 task_complete，summary 写"手册已完整覆盖"。

不要为了凑数编造内容。"""


# ── Extraction prompt template ───────────────────────────────────────

_EXTRACT_SYSTEM = """[系统 — 观察学习提取]
你是游戏操作流程分析器。你观看了用户在 **{game_name}** 中的手动操作录像。
系统同时记录了：① 每步画面变化和 OCR 文字 ② 用户鼠标点击的精确设备坐标。

## 你的任务
严格还原用户操作流程，生成一份 **另一个 LLM agent 可以直接执行** 的操作手册。

## 🔴 硬性规则 — 违反任何一条就是失败

### 1. 一一对应：每个画面变化 = 至少一步
帧序列中的每个关键帧代表一次画面变化。你必须为**每一个**关键帧写出对应步骤。
如果用户在基建界面点了 5 个设施，就写 5 步，一步都不能少。

### 2. 坐标必须来自数据
- 如果某帧标注了 `🖱 用户点击坐标`，那一步**必须**包含 `adb_tap_position(x_pct, y_pct) # [{x}, {y}]`，坐标精确取自我提供的数据
- **禁止**编造坐标。如果帧数据里没有点击坐标，就不要写坐标
- pct 的计算：x_pct = x / 设备分辨率_w, y_pct = y / 设备分辨率_h，保留两位小数

### 3. 文字必须来自 OCR
- 步骤描述中的按钮名、界面名**必须**取自该帧 OCR 文字
- OCR 显示的是 "碳" → 写 "点击碳"，不是 "购买所需物品"
- OCR 显示的是 "制造站" → 写 "制造站"，不是 "设施"
- **禁止概括、抽象、美化 OCR 内容**

### 4. 禁止跳步、禁止合并
- 不要因为"操作很简单"就合并两步
- 不要因为"这不重要"就跳过某帧
- 如果我操作了 10 个不同画面，你必须有 10+ 个步骤

### 5. 帧间等待
- 如果两帧间隔超过 3 秒且没有点击，说明是加载等待
- 写为 "等待约 X 秒（加载中）"

## Steps 格式
每行一步，格式:
- 有坐标: `adb_tap_position(x_pct, y_pct) # [{x}, {y}] OCR文字`
- 无坐标: `adb_tap('OCR文字')`
- 返回: `adb_back() # 描述`
- 等待: `等待约 X 秒`
- 滚动: `adb_scroll('方向')`

## Pitfalls 格式
- 加载画面大约需要 X 秒
- OCR 分割/识别错误的按钮名（例如 OCR 把 "开始行动" 拆成 "开始"+"行动"）
- 某步容易点错/混淆

## 完成前自检
你调用 create_guide 之前，必须在心里自检：
1. 步骤数 ≥ 关键帧数？（每个关键帧至少一步）
2. 有坐标的步骤，坐标都来自数据吗？（不能自己编）
3. 步骤描述中的按钮名都能在 OCR 里找到吗？
4. 有没有跳过的帧？

如果任何一个答案是"否"，补全后再提交。

## 提取流程
1. 逐帧阅读，不要跳
2. 对每个关键帧写一个步骤
3. 有坐标的帧直接用坐标，无坐标的帧描述画面变化
4. 调用 create_guide(name, description, steps, game, pitfalls, tags)
5. 调用 task_complete()

开始。"""


# ── Frame sampling ───────────────────────────────────────────────────

def _sample_for_images(frames: list[Any], max_images: int = 10) -> list[Any]:
    """Pick evenly-spaced frames for image embedding (visual reference only)."""
    if not frames:
        return []
    if len(frames) <= max_images:
        return frames
    result = [frames[0]]
    step = (len(frames) - 1) / (max_images - 1)
    for i in range(1, max_images - 1):
        result.append(frames[int(i * step)])
    result.append(frames[-1])
    return result


# ── Vision-mode helpers ────────────────────────────────────────────────

def _sample_for_vision(frames: list[Any], max_frames: int = _MAX_VISION_FRAMES) -> list[Any]:
    """Pick frames for vision-mode image feed, prioritizing action frames.

    Strategy:
      1. Always keep first + last frame.
      2. Fill the remaining budget preferring frames WITH clicks (user actions
         are higher signal than animated transitions).
      3. If click-rich frames fill or exceed the budget, sample evenly among
         them.  If not, pad with the highest-hamming non-click frames.
    """
    if len(frames) <= max_frames:
        return frames

    click_frames = [f for f in frames if f.clicks_before]
    non_click = [f for f in frames if not f.clicks_before]

    # Always keep first and last
    result: list[Any] = [frames[0]]
    budget = max_frames - 2  # reserve first + last

    if len(click_frames) >= budget:
        # Sample evenly from click frames
        step = (len(click_frames) - 1) / max(budget - 1, 1)
        for i in range(budget):
            result.append(click_frames[int(i * step)])
    else:
        # Include all clicks, then pad with evenly-spaced non-click frames
        result.extend(click_frames)
        remaining = budget - len(click_frames)
        if remaining > 0 and non_click:
            # Sort by hamming distance (high = more visual change = more useful)
            sorted_non = sorted(non_click, key=lambda f: f.hamming_from_prev or 0, reverse=True)
            step = (len(sorted_non) - 1) / max(remaining - 1, 1)
            seen = set()
            for i in range(remaining):
                idx = int(i * step)
                if idx not in seen:
                    result.append(sorted_non[idx])
                    seen.add(idx)

    result.append(frames[-1])
    # Deduplicate while preserving order
    seen_idx: set[int] = set()
    filtered: list[Any] = []
    for f in result:
        if f.index not in seen_idx:
            filtered.append(f)
            seen_idx.add(f.index)
    return filtered


def _annotate_frame(
    frame: Any,
    img: Image.Image,
    device_w: int,
    device_h: int,
    total_sig_frames: int = 0,
) -> Image.Image:
    """Draw frame number, timestamp, click markers, and OCR snippets on a frame.

    Overlay design:
      - Top-left:    gold-colored frame# + timestamp (e.g. "F0042 +0:27.8")
      - Click dots:  red hollow circle at each click position + (x,y) label
      - Bottom strip: first few OCR texts in white on dark background
      - Right edge:   thin vertical progress bar showing position in session

    Mutates and returns the image.  Safe to call on RGBA images.
    """
    from PIL import ImageDraw, ImageFont

    if img.mode == "RGBA":
        img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Scale factors — annotation coords are in device pixels, but the saved
    # frame may have been downscaled.
    sx = w / max(device_w, 1)
    sy = h / max(device_h, 1)

    # ── Font ──
    font = None
    for font_name in ("consola.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(font_name, _OVERLAY_FONT_SIZE)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    # ── Top-left: frame# + timestamp ──
    elapsed = frame.timestamp_s
    t_label = f"F{frame.index:04d}  +{int(elapsed//60)}:{elapsed%60:04.1f}"
    # Dark background strip for readability
    tb = draw.textbbox((0, 0), t_label, font=font)
    draw.rectangle([tb[0] - 4, tb[1] - 2, tb[2] + 4, tb[3] + 2], fill=(20, 20, 20))
    draw.text((2, 0), t_label, fill=(255, 215, 0), font=font)  # gold

    # ── Click markers ──
    for click in frame.clicks_before:
        cx = int(click.device_x * sx)
        cy = int(click.device_y * sy)
        r = _CLICK_MARKER_RADIUS
        # Outer ring (thick, red)
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=(255, 40, 40), width=3,
        )
        # Inner dot
        draw.ellipse(
            [cx - 3, cy - 3, cx + 3, cy + 3],
            fill=(255, 40, 40),
        )
        # Coordinate label
        coord_label = f"({click.device_x},{click.device_y})"
        label_x = min(cx + r + 4, w - 100)
        label_y = max(cy - 10, 2)
        tb2 = draw.textbbox((0, 0), coord_label, font=font)
        lw, lh = tb2[2] - tb2[0], tb2[3] - tb2[1]
        draw.rectangle(
            [label_x - 2, label_y - 2, label_x + lw + 2, label_y + lh + 2],
            fill=(40, 20, 20),
        )
        draw.text((label_x, label_y), coord_label, fill=(255, 255, 255), font=font)

    # ── Bottom strip: OCR snippets ──
    if frame.ocr_texts:
        ocr_line = " | ".join(frame.ocr_texts[:8])
        if len(ocr_line) > 120:
            ocr_line = ocr_line[:117] + "…"
        tb3 = draw.textbbox((0, 0), ocr_line, font=font)
        lh3 = tb3[3] - tb3[1]
        draw.rectangle([0, h - lh3 - 8, w, h], fill=(20, 20, 20))
        draw.text((4, h - lh3 - 4), ocr_line, fill=(180, 180, 180), font=font)

    # ── Right edge: progress thin bar ──
    bar_w = 3
    if frame.index > 0 and total_sig_frames > 0:
        progress_h = int(h * min(frame.index / max(total_sig_frames, 1), 1.0))
        if progress_h > 0:
            draw.rectangle(
                [w - bar_w, h - progress_h, w, h],
                fill=(255, 215, 0),  # gold
            )

    return img


def _build_user_message_vision(
    manifest: Any,
    sig_frames: list[Any],
    game_name: str,
    task_name: str,
    duration: str,
) -> list[dict[str, Any]]:
    """Build a pure-image user message: annotated frames fed directly to a vision LLM.

    No OCR pipeline, no text timeline, no coordinate parsing.  The model gets
    every frame as an image with frame# / time / click dots / OCR text
    overlaid.  It reads UI text straight from the pixels.
    """
    content: list[dict[str, Any]] = []
    device_w, device_h = manifest.resolution

    # ── Header text ──
    # Pre-compute vision frame count for accurate header
    _vision_frames = _sample_for_vision(sig_frames, max_frames=_MAX_VISION_FRAMES)
    _shown = len(_vision_frames)
    _total = len(sig_frames)
    header = (
        f"## 观察记录: {game_name} / {task_name}\n"
        f"- 总帧数: {manifest.frame_count} / 关键帧: {_total} / 时长: {duration}\n"
        f"- 设备分辨率: {device_w} × {device_h}\n"
        f"- 模式: 视觉直读（模型直接看图，未经过OCR管道）\n"
        f"\n"
        f"**下面展示了 {_total} 个关键帧中的 {_shown} 个采样帧（优先保留有点击操作的帧）。"
        f"请仔细看完每一帧。**\n"
        f"每个画面左上角金色字是帧号，红色圆点是用户点击位置。\n"
        f"有点击的帧后面附有 **🔍 放大图**——展示红圈位置的 UI 元素原图，看清按钮文字。\n"
        f"你必须为全部 {_total} 个关键帧写出至少 {_total} 个步骤。\n"
        f"\n图例：⭕ 红圈=用户点这里 | (x,y)=设备像素坐标 | 🔍放大图=点击目标 | 右边金线=进度\n"
        "---"
    )
    content.append({"type": "text", "text": header})

    # ── All frames as annotated images ──
    vision_frames = _vision_frames  # Reuse from header computation above
    total = len(vision_frames)

    for idx, frame in enumerate(vision_frames):
        elapsed = frame.timestamp_s
        t_str = f"{int(elapsed//60)}:{elapsed%60:04.1f}"
        ocr_snip = ", ".join(frame.ocr_texts[:5]) if frame.ocr_texts else "(no OCR)"
        has_clicks = " 🖱" if frame.clicks_before else ""

        # Source frame image
        try:
            from src.agent.observation_store import get_frame_path
            from PIL import Image
            path = get_frame_path(manifest, frame)
            full_img = Image.open(path)
            if full_img.mode == "RGBA":
                full_img = full_img.convert("RGB")
        except Exception:
            # Frame file missing — skip with a text note
            content.append({
                "type": "text",
                "text": (
                    f"F{frame.index:04d} +{t_str} [图片缺失]{has_clicks}\n"
                    f"  OCR: {ocr_snip}"
                ),
            })
            continue

        # ── Click-target magnifier: crop full-res region BEFORE downscale ──
        # Magnified crops show the LLM exactly what the user clicked on —
        # button text, icon details, etc. at full device resolution.
        click_crops: list[Image.Image] = []
        click_px_labels: list[str] = []
        if frame.clicks_before:
            for c in frame.clicks_before:
                half = _CLICK_CROP_SIZE_DEV // 2
                left = max(0, c.device_x - half)
                top = max(0, c.device_y - half)
                right = min(full_img.width, c.device_x + half)
                bottom = min(full_img.height, c.device_y + half)
                if right > left and bottom > top:
                    crop = full_img.crop((left, top, right, bottom))
                    # Upscale to readable size
                    crop_w, crop_h = crop.size
                    scale = _CLICK_CROP_DISPLAY_W / max(crop_w, 1)
                    crop = crop.resize(
                        (_CLICK_CROP_DISPLAY_W, max(int(crop_h * scale), 40)),
                        Image.LANCZOS,
                    )
                    # Draw crosshair at click point
                    from PIL import ImageDraw as _IDraw2
                    _cd = _IDraw2.Draw(crop)
                    cx_crop = int((c.device_x - left) * scale)
                    cy_crop = int((c.device_y - top) * scale)
                    _r = 12
                    _cd.ellipse([cx_crop - _r, cy_crop - _r, cx_crop + _r, cy_crop + _r],
                               outline=(255, 40, 40), width=2)
                    _cd.line([(cx_crop - 8, cy_crop), (cx_crop + 8, cy_crop)],
                            fill=(255, 40, 40), width=2)
                    _cd.line([(cx_crop, cy_crop - 8), (cx_crop, cy_crop + 8)],
                            fill=(255, 40, 40), width=2)
                    click_crops.append(crop)
                    click_px_labels.append(f"({c.device_x},{c.device_y})")

        # Downscale before annotation
        img = full_img.copy()
        if img.width > _VISION_FRAME_MAX_WIDTH:
            ratio = _VISION_FRAME_MAX_WIDTH / img.width
            img = img.resize(
                (_VISION_FRAME_MAX_WIDTH, int(img.height * ratio)),
                Image.LANCZOS,
            )

        # Annotate
        annotated = _annotate_frame(frame, img, device_w, device_h,
                                    total_sig_frames=len(sig_frames))

        # Compress
        from io import BytesIO
        buf = BytesIO()
        annotated.save(buf, format="JPEG", quality=55, optimize=True)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_b64,
            },
        })

        # ── Magnified click-target crops ──
        # One small image per click, showing the UI element at the
        # click position in full resolution so the LLM can read button text.
        for ci, crop in enumerate(click_crops):
            crop_buf = BytesIO()
            crop.save(crop_buf, format="JPEG", quality=70, optimize=True)
            crop_b64 = base64.b64encode(crop_buf.getvalue()).decode()
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": crop_b64,
                },
            })
            content.append({
                "type": "text",
                "text": f"🔍 F{frame.index:04d} 点击#{ci+1} 放大: 坐标 {click_px_labels[ci]} — 此图显示用户点击的具体 UI 元素",
            })

        # Progress line every 10 frames
        if (idx + 1) % 10 == 0:
            content.append({
                "type": "text",
                "text": f"↑ F{frame.index:04d} / {idx+1}/{total} frames shown above. Continue below ↓",
            })

    # ── Footer: final instruction ──
    content.append({"type": "text", "text": (
        f"\n---\n\n"
        f"以上共 {len(sig_frames)} 个关键帧（展示了 {total} 个采样）。"
        f"请在 create_guide 中写出至少 {len(sig_frames)} 个步骤，"
        f"完成所有帧后再 task_complete。\n\n"
        f"**快速检查**：\n"
        f"  1. 有红圈🔍放大图的帧 → 用了 adb_tap_position 且坐标正确？\n"
        f"  2. 步骤数 ≥ {len(sig_frames)}？\n"
        f"  3. 没有跳过任何帧？\n"
        f"如果任何答案是'否'，补全。"
    )})

    return content


# ── Refinement: build user message with existing guide + frames ───────

def _build_user_message_refine(
    manifest: Any,
    sig_frames: list[Any],
    game_name: str,
    task_name: str,
    duration: str,
    existing_guide: str,
) -> list[dict[str, Any]]:
    """Build a vision-mode user message that feeds the existing guide + observation
    frames to the LLM, instructing it to produce a complete merged guide.
    """
    content: list[dict[str, Any]] = []
    device_w, device_h = manifest.resolution

    # ── Header: existing guide + merge instructions ──
    vision_frames = _sample_for_vision(sig_frames, max_frames=_MAX_VISION_FRAMES)
    shown = len(vision_frames)
    total = len(sig_frames)

    header = (
        f"## 任务：完善这份操作手册\n\n"
        f"### 已有手册（全文 — 请仔细阅读结构）\n"
        f"```markdown\n{existing_guide}\n```\n\n"
        f"---\n\n"
        f"### 录像信息\n"
        f"- 游戏: {game_name} / 任务: {task_name}\n"
        f"- 总帧数: {manifest.frame_count} / 关键帧: {total} / 时长: {duration}\n"
        f"- 设备分辨率: {device_w} × {device_h}\n"
        f"- 展示了 {shown}/{total} 个关键帧（优先保留有点击操作的帧）\n"
        f"\n"
        f"**你的任务：把录像里有但手册遗漏的内容插入手册对应位置，输出完整的合并版。**\n"
        f"画面左上角金色字是帧号(Fxxxx)，红色圆点是用户点击，旁边是坐标(x,y)。\n"
        f"有点击的帧附有 **🔍 放大图**——展示点击目标 UI 元素的详细内容。\n"
        f"保留手册原有内容，新步骤插在正确阶段，不要删改已有的正确步骤。\n"
        f"\n---\n"
    )
    content.append({"type": "text", "text": header})

    # ── Annotated frames (same pipeline as vision-mode extract) ──
    for idx, frame in enumerate(vision_frames):
        elapsed = frame.timestamp_s
        t_str = f"{int(elapsed // 60)}:{elapsed % 60:04.1f}"
        ocr_snip = ", ".join(frame.ocr_texts[:5]) if frame.ocr_texts else "(no OCR)"
        has_clicks = " 🖱" if frame.clicks_before else ""

        try:
            from src.agent.observation_store import get_frame_path
            from PIL import Image
            path = get_frame_path(manifest, frame)
            full_img = Image.open(path)
            if full_img.mode == "RGBA":
                full_img = full_img.convert("RGB")
        except Exception:
            content.append({
                "type": "text",
                "text": (
                    f"F{frame.index:04d} +{t_str} [图片缺失]{has_clicks}\n"
                    f"  OCR: {ocr_snip}"
                ),
            })
            continue

        # Click-target magnifier (same as vision-mode)
        click_crops: list[Image.Image] = []
        click_px_labels: list[str] = []
        if frame.clicks_before:
            for c in frame.clicks_before:
                half = _CLICK_CROP_SIZE_DEV // 2
                left = max(0, c.device_x - half)
                top = max(0, c.device_y - half)
                right = min(full_img.width, c.device_x + half)
                bottom = min(full_img.height, c.device_y + half)
                if right > left and bottom > top:
                    crop = full_img.crop((left, top, right, bottom))
                    crop_w, crop_h = crop.size
                    scale = _CLICK_CROP_DISPLAY_W / max(crop_w, 1)
                    crop = crop.resize(
                        (_CLICK_CROP_DISPLAY_W, max(int(crop_h * scale), 40)),
                        Image.LANCZOS,
                    )
                    from PIL import ImageDraw as _IDraw2
                    _cd = _IDraw2.Draw(crop)
                    cx_crop = int((c.device_x - left) * scale)
                    cy_crop = int((c.device_y - top) * scale)
                    _r = 12
                    _cd.ellipse(
                        [cx_crop - _r, cy_crop - _r, cx_crop + _r, cy_crop + _r],
                        outline=(255, 40, 40), width=2,
                    )
                    _cd.line(
                        [(cx_crop - 8, cy_crop), (cx_crop + 8, cy_crop)],
                        fill=(255, 40, 40), width=2,
                    )
                    _cd.line(
                        [(cx_crop, cy_crop - 8), (cx_crop, cy_crop + 8)],
                        fill=(255, 40, 40), width=2,
                    )
                    click_crops.append(crop)
                    click_px_labels.append(f"({c.device_x},{c.device_y})")

        # Downscale before annotation
        img = full_img.copy()
        if img.width > _VISION_FRAME_MAX_WIDTH:
            ratio = _VISION_FRAME_MAX_WIDTH / img.width
            img = img.resize(
                (_VISION_FRAME_MAX_WIDTH, int(img.height * ratio)),
                Image.LANCZOS,
            )

        annotated = _annotate_frame(frame, img, device_w, device_h,
                                    total_sig_frames=len(sig_frames))

        from io import BytesIO
        buf = BytesIO()
        annotated.save(buf, format="JPEG", quality=55, optimize=True)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_b64,
            },
        })

        # Magnified click-target crops
        for ci, crop in enumerate(click_crops):
            crop_buf = BytesIO()
            crop.save(crop_buf, format="JPEG", quality=70, optimize=True)
            crop_b64 = base64.b64encode(crop_buf.getvalue()).decode()
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": crop_b64,
                },
            })
            content.append({
                "type": "text",
                "text": (
                    f"🔍 F{frame.index:04d} 点击#{ci + 1} 放大: "
                    f"坐标 {click_px_labels[ci]}"
                ),
            })

        if (idx + 1) % 10 == 0:
            content.append({
                "type": "text",
                "text": (
                    f"↑ F{frame.index:04d} / {idx + 1}/{shown} "
                    f"frames shown above. Continue below ↓"
                ),
            })

    # ── Footer ──
    content.append({
        "type": "text",
        "text": (
            f"\n---\n\n"
            f"以上共 {len(sig_frames)} 个关键帧（展示了 {shown}）。\n"
            f"发现遗漏 → 调用 append_to_guide 追加到手册末尾。\n"
            f"没有遗漏 → 直接 task_complete('手册已完整覆盖')。\n"
            f"不要描述画面——把发现直接写进 append_to_guide 参数。"
        ),
    })

    return content


# ── Public API ───────────────────────────────────────────────────────

class ObservationExtractor:
    """LLM-based guide extraction from an observation session.

    Reads the manifest, runs OCR on frames, builds a prompt with inline
    images + click data, and delegates to the LLM to produce a guide skill.
    """

    def __init__(self) -> None:
        pass

    def extract(
        self,
        manifest_path: str,
        game: str = "",
        task_name: str = "",
        vision_mode: bool = True,
        on_done: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        """Extract a guide skill from an observation session.

        Runs in the current thread — caller should spawn a background thread.

        Args:
            manifest_path: Path to manifest.json of the recorded session.
            game: Game ID (auto-detected from manifest path if empty).
            task_name: Human-readable task label.
            vision_mode: If True (default), feed annotated frame images
                directly to a vision LLM.  No OCR pipeline needed.
                Set False for the legacy OCR+text pipeline.
            on_done: Optional callback receiving the result message.

        Returns:
            (skill_name, skill_path) — name is the saved filename stem,
            path is the absolute filesystem path to the .md file.
            Both empty on failure.
        """
        from src.agent.observation_store import (
            load_manifest, get_frame_path, mark_completed,
        )

        manifest = load_manifest(
            game or _guess_game(manifest_path),
            _guess_session_id(manifest_path),
        )
        if manifest is None:
            msg = "无法加载观察记录，请重试。"
            logger.warning(msg)
            if on_done:
                on_done(msg)
            return "", ""

        game = game or manifest.game
        task_name = task_name or manifest.task_name

        # ── Phase 1: Run OCR on significant frames ──
        logger.info("Extracting guide: game=%s task=%s frames=%d sig=%d",
                    game, task_name, manifest.frame_count, manifest.significant_count)

        # Get all significant frames (text timeline) up to MAX_TEXT_FRAMES
        all_sig = [f for f in manifest.frames if f.is_significant]
        if len(all_sig) > _MAX_TEXT_FRAMES:
            # Keep first + last, evenly sample middle
            keep = [all_sig[0]]
            step = (len(all_sig) - 1) / (_MAX_TEXT_FRAMES - 1)
            for i in range(1, _MAX_TEXT_FRAMES - 1):
                keep.append(all_sig[int(i * step)])
            keep.append(all_sig[-1])
            sig_frames = keep
        else:
            sig_frames = all_sig

        if len(sig_frames) < getattr(
            getattr(app_config, 'observation', None),
            'significant_frames_min', 3,
        ):
            msg = "画面变化太少（可能游戏不在前台操作），请重新录制。"
            logger.warning(msg)
            if on_done:
                on_done(msg)
            return "", ""

        # ── Vision mode: skip OCR, build image-only prompt ──
        game_name = _resolve_game_name(game)
        duration = self._calc_duration(manifest)
        device_w, device_h = manifest.resolution

        if vision_mode:
            # No OCR at all — the vision model reads UI text from pixels.
            logger.info(
                "Vision-mode extraction: game=%s task=%s frames=%d sig=%d",
                game, task_name, manifest.frame_count, manifest.significant_count,
            )

            system = _VISION_SYSTEM.format(
                game_name=game_name,
            )
            user_content = _build_user_message_vision(
                manifest=manifest,
                sig_frames=sig_frames,
                game_name=game_name,
                task_name=task_name,
                duration=duration,
            )
        else:
            # ── Text mode: run OCR, build text+sampled-image prompt ──
            ocr_ok = self._run_ocr_on_frames(manifest, sig_frames)

            system = _EXTRACT_SYSTEM.format(
                game_name=game_name,
            )

            user_content = self._build_user_message(
                manifest=manifest,
                sig_frames=sig_frames,
                game_name=game_name,
                task_name=task_name,
                duration=duration,
                ocr_ok=ocr_ok,
            )

        # ── Phase 3: Run LLM extraction ──
        skill_name, skill_path = self._run_llm_extraction(
            system=system,
            user_content=user_content,
            game=game,
            task_name=task_name,
        )

        # ── Fallback: if vision mode produced no result, try text mode ──
        if not skill_name and vision_mode:
            logger.info(
                "Vision-mode extraction produced no result — "
                "falling back to text mode (OCR pipeline)",
            )
            try:
                # Run OCR on frames (this populates frame.ocr_texts in place)
                ocr_ok = self._run_ocr_on_frames(manifest, sig_frames)

                fallback_system = _EXTRACT_SYSTEM.format(
                    game_name=game_name,
                )
                fallback_content = self._build_user_message(
                    manifest=manifest,
                    sig_frames=sig_frames,
                    game_name=game_name,
                    task_name=task_name,
                    duration=duration,
                    ocr_ok=ocr_ok,
                )

                fb_name, fb_path = self._run_llm_extraction(
                    system=fallback_system,
                    user_content=fallback_content,
                    game=game,
                    task_name=task_name,
                )
                if fb_name:
                    skill_name, skill_path = fb_name, fb_path
                    logger.info(
                        "Text-mode fallback succeeded: skill='%s'", fb_name,
                    )
                else:
                    logger.warning(
                        "Text-mode fallback also failed — session data preserved "
                        "for manual review",
                    )
            except Exception as e:
                logger.error("Text-mode fallback extraction error: %s", e)

        # ── Phase 4: Mark completed ──
        try:
            mark_completed(manifest)
        except Exception:
            pass

        if on_done and skill_name:
            on_done(
                f"✅ 已生成指引: {skill_name}\n"
                f"文件: {skill_path}\n"
                f"可直接编辑，运行时说「{task_name}」即可自动匹配到此指引。"
            )

        return skill_name, skill_path

    # ── Internal ─────────────────────────────────────────────────

    def _run_ocr_on_frames(
        self, manifest: Any, frames: list[Any],
    ) -> bool:
        """Run OCR on each significant frame, updating frame metadata in place.

        Also writes updated manifest so OCR results persist on disk.

        Returns True if OCR ran successfully on at least one frame.
        """
        try:
            from src.vision.ocr import ocr_engine
            ocr_engine.preload()
        except Exception as e:
            logger.warning("OCR engine unavailable: %s", e)
            return False

        ok_count = 0
        for frame in frames:
            if frame.ocr_texts:
                ok_count += 1
                continue  # Already has OCR

            try:
                from src.agent.observation_store import get_frame_path
                path = get_frame_path(manifest, frame)
                from PIL import Image
                img = Image.open(path)
                detections = ocr_engine.read_text(img)
                texts = [d["text"] for d in detections if d.get("confidence", 0) >= 0.5]
                frame.ocr_texts = texts
                ok_count += 1
            except Exception as e:
                logger.debug("OCR failed for frame %d: %s", frame.index, e)

        # Persist OCR results to manifest
        if ok_count > 0:
            try:
                from src.agent.observation_store import update_manifest
                update_manifest(manifest)
            except Exception:
                pass

        logger.info("OCR: %d/%d frames processed", ok_count, len(frames))
        return ok_count > 0

    def _build_user_message(
        self,
        manifest: Any,
        sig_frames: list[Any],
        game_name: str,
        task_name: str,
        duration: str,
        ocr_ok: bool,
    ) -> list[dict[str, Any]]:
        """Build the user message content blocks for the LLM.

        ALL significant frames are included as text (OCR + clicks).
        Only a sampled subset get embedded images to stay under token limits.
        """
        content: list[dict[str, Any]] = []
        device_w, device_h = manifest.resolution

        # ── Header ──
        header = (
            f"## 观察记录: {game_name} / {task_name}\n"
            f"- 总帧 {manifest.frame_count} / 关键帧 {len(sig_frames)} / 时长 {duration}\n"
            f"- 设备分辨率: {device_w}x{device_h}\n"
            f"- OCR: {'已提取' if ocr_ok else '⚠️ 未提取'}\n"
            f"\n"
            f"**你必须为每个关键帧写出至少一个步骤。共 {len(sig_frames)} 个关键帧，"
            f"至少 {len(sig_frames)} 步。**\n"
            f"\n---\n"
        )
        content.append({"type": "text", "text": header})

        # ── Text timeline: ALL frames ──
        # Build a compact text table of every frame so the LLM can't miss anything.
        timeline_lines = ["## 完整帧时间线（每一步都不能少）", ""]
        for frame in sig_frames:
            elapsed = frame.timestamp_s
            t_str = f"{int(elapsed//60)}:{elapsed%60:04.1f}"
            ocr = ", ".join(frame.ocr_texts[:20]) if frame.ocr_texts else "(无OCR)"
            click_str = ""
            if frame.clicks_before:
                xy_list = [f"({c.device_x},{c.device_y})" for c in frame.clicks_before]
                click_str = f" 🖱{','.join(xy_list)}"
            gap = ""
            if frame.hamming_from_prev is not None and frame.hamming_from_prev < 8:
                gap = " [微小变化]"
            timeline_lines.append(
                f"F{frame.index:04d} | +{t_str}{gap} | {ocr}{click_str}"
            )
        content.append({"type": "text", "text": "\n".join(timeline_lines)})

        # ── Image gallery: sampled frames only ──
        # Embed ~10 evenly-spaced images as visual reference.
        image_frames = _sample_for_images(sig_frames, max_images=10)

        gallery_lines = ["\n---\n", "## 关键画面截图（采样参考）", ""]
        content.append({"type": "text", "text": "\n".join(gallery_lines)})

        for frame in image_frames:
            elapsed = frame.timestamp_s
            t_str = f"{int(elapsed//60)}:{elapsed%60:04.1f}"
            ocr_snip = ", ".join(frame.ocr_texts[:10]) if frame.ocr_texts else ""
            label = f"[F{frame.index:04d} +{t_str}] {ocr_snip}"
            content.append({"type": "text", "text": label})

            try:
                from src.agent.observation_store import get_frame_path
                path = get_frame_path(manifest, frame)
                from PIL import Image
                img = Image.open(path)
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                w, h = img.size
                pw = getattr(getattr(app_config, 'observation', None), 'frame_max_width', 800)
                if w > pw:
                    ratio = pw / w
                    img = img.resize((pw, int(h * ratio)), Image.LANCZOS)
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=50, optimize=True)
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    },
                })
            except Exception:
                pass

        # ── Final instruction ──
        content.append({"type": "text", "text": (
            f"\n---\n\n"
            f"以上共 {len(sig_frames)} 个关键帧。"
            f"请在 create_guide 中写出至少 {len(sig_frames)} 个步骤，完成所有帧后再 task_complete。"
        )})

        return content

    def _run_llm_extraction(
        self,
        system: str,
        user_content: list[dict[str, Any]],
        game: str,
        task_name: str,
    ) -> tuple[str, str]:
        """Run the LLM extraction call with create_guide + task_complete tools.

        Retries up to 2 additional times if the LLM returns text without
        calling create_guide (common with vision models that "describe"
        frames instead of acting on them).

        Returns (skill_name, skill_path) on success, ("", "") on failure.
        """
        from src.llm.client import MiMoClient, extract_text, extract_tool_calls

        # ── Build minimal tool registry ──
        from src.tools.registry import ToolRegistry
        registry = ToolRegistry()

        # create_guide — reuse the real implementation
        from src.tools.guide_tool import create_guide
        registry.register(
            name="create_guide",
            description=(
                "创建或更新一个技能文件（skill）。接受自然语言描述，自动格式化。\n"
                "示例：create_guide(name='base-collect', description='基建收菜', game='arknights', "
                "steps='1. 点击基建\\n2. 点击通知铃铛', "
                "pitfalls='- 加载画面需要等待', tags='基建,收菜,daily')"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "技能名称（如 base-collect, farm-ce-5），英文/拼音",
                    },
                    "description": {
                        "type": "string",
                        "description": "一句话描述",
                    },
                    "steps": {
                        "type": "string",
                        "description": (
                            "操作步骤，每行一步。有坐标必须用 adb_tap_position。\n"
                            "有红圈→ adb_tap_position(x_pct, y_pct) # [x, y] 按钮文字\n"
                            "无红圈但按钮清晰→ adb_tap('按钮OCR文字')\n"
                            "返回→ adb_back() # 描述\n"
                            "等待→ # 等待约X秒（加载/战斗）\n"
                            "严禁: adb_tap('自然语言描述') —— OCR永远匹配不到"
                        ),
                    },
                    "game": {
                        "type": "string",
                        "description": "游戏 ID，如 arknights, reverse1999",
                    },
                    "pitfalls": {
                        "type": "string",
                        "description": "注意事项，每行一个（可选）",
                    },
                    "tags": {
                        "type": "string",
                        "description": "逗号分隔的关键词",
                    },
                    "skill_type": {
                        "type": "string",
                        "description": "'guide' 或 'script'，默认 guide",
                    },
                    "verified": {
                        "type": "boolean",
                        "description": "坐标是否已验证，默认 false",
                    },
                },
                "required": ["name", "description", "steps", "game"],
            },
            handler=create_guide,
        )

        # task_complete handler
        skill_result: list[tuple[str, str]] = []  # [(name, path), ...]

        def _task_complete(summary: str = "") -> Any:
            from src.tools.registry import ToolOutput
            return ToolOutput(
                text=json.dumps({"success": True, "summary": summary, "task_done": True}),
                task_done=True,
            )

        registry.register(
            name="task_complete",
            description="Mark the current task as complete.",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of what was accomplished.",
                    },
                },
            },
            handler=_task_complete,
        )

        tools = registry.get_definitions()

        # ── Retry loop: up to 3 total attempts ──
        _MAX_ATTEMPTS = 3
        _RETRY_COOLDOWN = 2.0  # seconds between retries

        last_text = ""
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            client = MiMoClient()
            try:
                # On retry, prepend a STRONGER reminder to call create_guide
                messages = [{"role": "user", "content": user_content}]
                if attempt > 1:
                    retry_nudge = {
                        "type": "text",
                        "text": (
                            f"\n\n⚠️ **重试第 {attempt} 次** — "
                            f"你上一轮没有调用 create_guide！\n"
                            f"你必须调用 create_guide(name=..., steps=..., ...) "
                            f"来保存技能文件，否则提取失败。\n"
                            f"看完所有帧后立即调用 create_guide，然后 task_complete。\n"
                            f"不要再描述画面内容——直接生成操作步骤并调用 create_guide。"
                        ),
                    }
                    if isinstance(messages[0]["content"], list):
                        messages[0]["content"].append(retry_nudge)
                    else:
                        messages[0]["content"] = [messages[0]["content"], retry_nudge]

                response = client.chat(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=8192,
                    temperature=0.3 + (attempt - 1) * 0.1,  # Slightly higher temp on retry
                )

                text = extract_text(response)
                tool_calls = extract_tool_calls(response)
                last_text = text or ""

                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get("name", "")
                        inp = tc.get("input", {})
                        if name == "create_guide":
                            try:
                                out = registry.dispatch(name, **inp)
                                data = json.loads(out.text)
                                if data.get("success") and data.get("name"):
                                    skill_result.append((data["name"], data.get("path", "")))
                                    logger.info(
                                        "Guide created via extraction (attempt %d): %s -> %s",
                                        attempt, data["name"], data.get("path", ""),
                                    )
                            except Exception as e:
                                logger.error("create_guide failed in extraction: %s", e)
                        elif name == "task_complete":
                            try:
                                registry.dispatch(name, **inp)
                            except Exception:
                                pass

                if skill_result:
                    return skill_result[0]

                # LLM returned text but didn't call create_guide
                if attempt < _MAX_ATTEMPTS:
                    logger.warning(
                        "LLM extraction attempt %d/%d: no create_guide call. "
                        "Text preview: %s",
                        attempt, _MAX_ATTEMPTS, (text or "(empty)")[:300],
                    )
                    import time as _time
                    _time.sleep(_RETRY_COOLDOWN)

            except Exception as e:
                logger.error(
                    "LLM extraction attempt %d/%d failed: %s",
                    attempt, _MAX_ATTEMPTS, e,
                )
                if attempt < _MAX_ATTEMPTS:
                    import time as _time
                    _time.sleep(_RETRY_COOLDOWN)
            finally:
                client.close()

        # All attempts exhausted
        if last_text:
            logger.warning(
                "LLM extraction: all %d attempts returned text but no create_guide. "
                "Last response: %s",
                _MAX_ATTEMPTS, last_text[:500],
            )

        return "", ""

    # ── Refinement mode ─────────────────────────────────────────────

    def refine_guide(
        self,
        manifest_path: str,
        guide_path: str,
        game: str = "",
        task_name: str = "",
        vision_mode: bool = True,
        on_done: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        """Refine an existing guide by merging observation findings directly into it.

        Shows the LLM the existing guide + annotated observation frames, asks it
        to produce the full merged version, and saves it **directly to the same
        file** (the old version is backed up as ``<name>.v{n}.md.bak``).

        Args:
            manifest_path: Path to manifest.json of the recorded session.
            guide_path: Path to the existing guide .md file (will be overwritten).
            game: Game ID (auto-detected from manifest path if empty).
            task_name: Human-readable task label.
            vision_mode: If True, feed annotated frame images directly.
            on_done: Optional callback receiving the result message.

        Returns:
            (guide_name, guide_path) — the updated guide. Empty on failure.
        """
        from src.agent.observation_store import (
            load_manifest, mark_completed,
        )

        manifest = load_manifest(
            game or _guess_game(manifest_path),
            _guess_session_id(manifest_path),
        )
        if manifest is None:
            msg = "无法加载观察记录。"
            logger.warning(msg)
            if on_done:
                on_done(msg)
            return "", ""

        game = game or manifest.game
        task_name = task_name or manifest.task_name

        # Load existing guide
        guide_content = ""
        guide_name = ""
        try:
            guide_p = Path(guide_path)
            if guide_p.exists():
                guide_content = guide_p.read_text(encoding="utf-8")
                guide_name = guide_p.stem
        except Exception as e:
            msg = f"无法读取已有指引: {e}"
            logger.warning(msg)
            if on_done:
                on_done(msg)
            return "", ""

        if not guide_content.strip():
            msg = "已有指引内容为空，请使用完整提取模式（extract）代替。"
            logger.warning(msg)
            if on_done:
                on_done(msg)
            return "", ""

        # Get significant frames
        all_sig = [f for f in manifest.frames if f.is_significant]
        if len(all_sig) > _MAX_TEXT_FRAMES:
            keep = [all_sig[0]]
            step = (len(all_sig) - 1) / (_MAX_TEXT_FRAMES - 1)
            for i in range(1, _MAX_TEXT_FRAMES - 1):
                keep.append(all_sig[int(i * step)])
            keep.append(all_sig[-1])
            sig_frames = keep
        else:
            sig_frames = all_sig

        if len(sig_frames) < 3:
            msg = "画面变化太少，请重新录制。"
            logger.warning(msg)
            if on_done:
                on_done(msg)
            return "", ""

        game_name = _resolve_game_name(game)
        duration = self._calc_duration(manifest)

        logger.info(
            "Refine mode: guide=%s game=%s frames=%d sig=%d",
            guide_name, game, manifest.frame_count, len(sig_frames),
        )

        # Parse existing guide to extract frontmatter fields for create_guide
        from src.skills.parser import parse_skill_md
        parsed = parse_skill_md(guide_content) if guide_content else {}

        system = _REFINE_VISION_SYSTEM.format(game_name=game_name)
        user_content = _build_user_message_refine(
            manifest=manifest,
            sig_frames=sig_frames,
            game_name=game_name,
            task_name=task_name,
            duration=duration,
            existing_guide=guide_content[:8000],
        )

        # Run LLM — it calls create_guide which overwrites the guide file.
        # create_guide already handles .bak backup via skill_mgr.save().
        updated = self._run_refine_merge(
            system=system,
            user_content=user_content,
            game=game,
            guide_name=guide_name,
            guide_path=guide_path,
            parsed=parsed,
        )

        try:
            mark_completed(manifest)
        except Exception:
            pass

        if updated:
            msg = f"已完善指引: {guide_name}\n文件: {guide_path}"
            if on_done:
                on_done(msg)
            return guide_name, str(guide_p)
        else:
            if on_done:
                on_done("手册已完整覆盖录像中的操作，没有发现遗漏。")
            return "", ""

    def _run_refine_merge(
        self,
        system: str,
        user_content: list[dict[str, Any]],
        game: str,
        guide_name: str,
        guide_path: str,
        parsed: dict[str, Any],
    ) -> bool:
        """Run the refine LLM call with append_to_guide tool.
        The LLM appends new steps/pitfalls/corrections directly to the guide file.
        Returns True if the guide was modified, False otherwise.
        """
        from src.llm.client import MiMoClient, extract_text, extract_tool_calls
        from src.tools.registry import ToolRegistry

        modified: list[bool] = [False]
        guide_file = Path(guide_path)

        # ── append_to_guide handler ──
        def _append_to_guide(
            new_steps: str = "",
            new_pitfalls: str = "",
            corrections: str = "",
        ) -> Any:
            if not new_steps.strip() and not new_pitfalls.strip() and not corrections.strip():
                from src.tools.registry import ToolOutput
                return ToolOutput(text=json.dumps(
                    {"success": False, "error": "至少需要一个参数有内容"},
                    ensure_ascii=False,
                ))

            content = guide_file.read_text(encoding="utf-8")
            now = datetime.now(tz=timezone.utc).isoformat()

            if new_steps.strip():
                # Insert before "## 完整执行注意事项" or "## Pitfalls", or at end
                marker = "## 完整执行注意事项"
                insert = f"\n## 补充步骤（录像对比 — {now}）\n\n{new_steps.strip()}\n"
                if marker in content:
                    content = content.replace(marker, insert + "\n" + marker)
                else:
                    content += "\n" + insert

            if new_pitfalls.strip():
                pitfalls_section = "## Pitfalls"
                pt_lines = new_pitfalls.strip().split('\n')
                pt_formatted = '\n'.join(
                    line if line.strip().startswith('-') else f"- {line}"
                    for line in pt_lines
                )
                if pitfalls_section in content:
                    content = content.rstrip() + "\n" + pt_formatted + "\n"
                else:
                    content += f"\n\n## Pitfalls\n{pt_formatted}\n"

            if corrections.strip():
                marker2 = "## 完整执行注意事项"
                insert_c = f"\n## 修正记录（录像对比 — {now}）\n\n{corrections.strip()}\n"
                if marker2 in content:
                    content = content.replace(marker2, insert_c + "\n" + marker2)
                else:
                    content += "\n" + insert_c

            guide_file.write_text(content, encoding="utf-8")
            modified[0] = True
            logger.info("Refine: appended to '%s'", guide_file.name)

            from src.tools.registry import ToolOutput
            return ToolOutput(text=json.dumps(
                {"success": True, "message": f"已追加到 {guide_file.name}"},
                ensure_ascii=False,
            ))

        # ── Tool registry ──
        registry = ToolRegistry()

        registry.register(
            name="append_to_guide",
            description=(
                "把录像中发现的遗漏内容追加到手册。三个参数选填，有就填。\n"
                "不要描述画面——直接把发现写进参数。没有遗漏就不要调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "new_steps": {
                        "type": "string",
                        "description": (
                            "手册缺失的操作步骤。格式：\n"
                            "### Fxxxx 步骤名\n"
                            "帧 Fxxxx | 坐标(x,y) | 按钮文字 | 插在手册的[阶段X]和[阶段Y]之间\n"
                            "adb_tap_position(x, y) # [x, y] 描述\n"
                        ),
                    },
                    "new_pitfalls": {
                        "type": "string",
                        "description": "手册缺失的陷阱。格式：- 描述（帧 Fxxxx），每行一条。",
                    },
                    "corrections": {
                        "type": "string",
                        "description": "手册描述不准确的地方。格式：- 原文'...'→应为'...'（帧 Fxxxx）",
                    },
                },
                "required": [],
            },
            handler=_append_to_guide,
        )

        def _task_complete(summary: str = "") -> Any:
            from src.tools.registry import ToolOutput
            return ToolOutput(
                text=json.dumps({"success": True, "summary": summary, "task_done": True}),
                task_done=True,
            )

        registry.register(
            name="task_complete",
            description="完成任务。summary 写'手册已完整覆盖'或简述追加了什么。",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "任务总结"},
                },
            },
            handler=_task_complete,
        )

        tools = registry.get_definitions()

        # ── Single attempt (diff mode is lightweight, no retry needed) ──
        client = MiMoClient()
        try:
            messages = [{"role": "user", "content": user_content}]
            response = client.chat(
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=4096,
                temperature=0.3,
            )

            text = extract_text(response)
            tool_calls = extract_tool_calls(response)

            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "")
                    inp = tc.get("input", {})
                    if name == "append_to_guide":
                        try:
                            registry.dispatch(name, **inp)
                        except Exception as e:
                            logger.error("append_to_guide failed: %s", e)
                    elif name == "task_complete":
                        try:
                            registry.dispatch(name, **inp)
                        except Exception:
                            pass

                return modified[0]

            # No tool calls — LLM returned text only
            logger.info(
                "Refine: LLM returned text without tool call: %s",
                (text or "(empty)")[:200],
            )
            return False
        finally:
            client.close()

    @staticmethod
    def _calc_duration(manifest: Any) -> str:
        """Calculate duration string from manifest timestamps."""
        try:
            from datetime import datetime as _dt
            start = _dt.fromisoformat(manifest.started_at)
            end_str = manifest.stopped_at
            if end_str:
                end = _dt.fromisoformat(end_str)
            else:
                end = datetime.now(tz=timezone.utc)
            delta = end - start
            mins = int(delta.total_seconds() // 60)
            secs = delta.total_seconds() % 60
            return f"{mins}分{secs:.0f}秒"
        except Exception:
            return "未知"


# ── Helpers ──────────────────────────────────────────────────────────

def _resolve_game_name(game: str) -> str:
    """Convert game ID to human-readable name."""
    try:
        from src.games.registry import get_game_registry
        return get_game_registry().get_game_name(game) or game
    except Exception:
        return game


def _guess_game(manifest_path: str) -> str:
    """Guess game from manifest path parent directory name."""
    try:
        return Path(manifest_path).parent.parent.name
    except Exception:
        return "arknights"


def _guess_session_id(manifest_path: str) -> str:
    """Guess session_id from manifest path parent directory name."""
    try:
        return Path(manifest_path).parent.name
    except Exception:
        return ""
