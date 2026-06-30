"""VLM screen reading via MiMo-V2.5 (Anthropic endpoint).

MiMo-V2.5 supports image input through the Anthropic Messages API.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from config.settings import config
from src.llm.client import MiMoClient
from src.utils.hash import compute_image_hash

logger = logging.getLogger(__name__)

class VLMDescriptor:
    """Use MiMo-V2.5 to read numbers and describe game screens."""

    _MAX_CACHE_SIZE = 200

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._client: MiMoClient | None = None

    def _get_client(self) -> MiMoClient:
        if self._client is None:
            self._client = MiMoClient()
        return self._client

    def _reset_client(self) -> None:
        """Close and recreate the underlying HTTP client (e.g. after a network error).

        Only resets the HTTP client — does NOT clear the VLM result cache.
        Cache entries are keyed by screen_hash (pixel content) and remain
        valid across client resets.  Clearing would degrade multi-agent
        performance by discarding valid cached results from other agents.
        """
        if self._client is not None:
            self._client.close()
            self._client = None
        logger.info("VLM client reset (cache preserved)")

    def describe(
        self,
        image: Image.Image | str | Path,
        purpose: str = "",
    ) -> dict[str, Any]:
        img = self._to_image(image).convert("RGB")
        screen_hash = compute_image_hash(img)

        # Cache by (screen_hash, purpose) — same screen + same question = same answer.
        # Purpose matters: same screen with different goals needs fresh analysis.
        # Cooldown in vlm_describe_tool prevents rapid re-calls on the same screen.
        cache_key = f"{screen_hash}:{purpose}" if purpose else screen_hash
        if cache_key in self._cache:
            logger.info("VLM cache hit: %s (purpose=%s)", screen_hash[:8], purpose[:40] if purpose else "none")
            return {**self._cache[cache_key], "screen_hash": screen_hash, "cached": True}

        w, h = img.size
        if w > 800:
            ratio = 800 / w
            img = img.resize((800, int(h * ratio)))

        img_b64 = self._encode(img)

        # Build the question based on whether we have a purpose
        if purpose:
            question = (
                f"我需要完成这个任务：{purpose}\n\n"
                "请帮我分析：\n"
                "1. 这是什么界面？画面中有哪些按钮？\n"
                "2. 要完成我的任务，现在应该点哪个按钮？为什么选这个？\n"
                "3. 点了之后预计会进入什么界面？\n"
                "4. 如果当前界面有其他看起来相关的选项，它们的区别是什么？\n"
                "只报告实际看到的，不要编造。"
            )
        else:
            question = (
                "逐项列出（只报告实际看到的，不要编造）：\n"
                "1. 这是什么界面？\n"
                "2. 资源数字：龙门币、合成玉、源石、理智各多少？\n"
                "3. 主要按钮（按钮上写的中文或英文标签，逐个列出）\n"
                "4. 如果是章节/关卡选择界面：当前显示的是哪一章节？能看到哪些关卡编号？\n"
                "5. 页面上有没有左右箭头按钮（◀ ▶ ＜ ＞）或翻页控件？\n"
                "   在什么位置（顶部/右边/底部）？显示的中文/英文标签是什么？\n"
                "6. 如果需要滑动或翻页才能看到更多内容：往哪个方向？在哪个区域滑？\n"
                "7. 有没有弹窗或提示？"
            )

        try:
            response = self._get_client().chat(
                system=(
                    "你是明日方舟UI分析器。列出按钮时附带像素坐标[x,y]。只报告实际看到的，不要编造元素。"
                    "UI约定：代理指挥复选框——出现X+数字（如X1、X2）表示已开启选中；空白方框表示未开启。"
                    "代理作战次数选择器——是一个可展开的控件，点击X1可以展开选择X2/X4/X6等次数。"
                    "关卡节点编号格式为X-Y（如1-7、1-8），TR-X为教学关卡。"
                ),
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                }],
            )
        except Exception as e:
            logger.warning("VLM failed (attempt 1): %s — retrying with fresh client", e)
            self._reset_client()
            try:
                response = self._get_client().chat(
                    system=(
                        "你是明日方舟UI分析器。列出按钮时附带像素坐标[x,y]。只报告实际看到的，不要编造元素。"
                        "UI约定：代理指挥复选框——出现X+数字（如X1、X2）表示已开启选中；空白方框表示未开启。"
                        "代理作战次数选择器——是一个可展开的控件，点击X1可以展开选择X2/X4/X6等次数。"
                        "关卡节点编号格式为X-Y（如1-7、1-8），TR-X为教学关卡。"
                    ),
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }],
                )
            except Exception as e2:
                logger.warning("VLM retry also failed: %s", e2)
                return {"screen_hash": screen_hash, "description": "", "numbers": {}, "cached": False}

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        text = text.strip()
        if not text:
            logger.warning("VLM returned empty description (output may have been truncated)")

        result = {
            "screen_hash": screen_hash,
            "description": text,
            "numbers": {},
            "cached": False,
        }
        if text:
            # LRU eviction: dicts preserve insertion order in Python 3.7+
            if len(self._cache) >= self._MAX_CACHE_SIZE:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                logger.debug("VLM cache evicted: %s (size=%d)", oldest[:8], len(self._cache))
            self._cache[cache_key] = result
        return result

    def read_numbers(self, image: Image.Image | str | Path) -> dict[str, str]:
        img = self._to_image(image).convert("RGB")
        w, h = img.size
        if w > 800:
            ratio = 800 / w
            img = img.resize((800, int(h * ratio)))
        img_b64 = self._encode(img)

        try:
            response = self._get_client().chat(
                system="只输出数字，不要任何解释。",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                        },
                        {
                            "type": "text",
                            "text": "龙门币=?\n合成玉=?\n源石=?\n理智=?\n只输出4行，每行格式：资源名=数字。",
                        },
                    ],
                }],
                max_tokens=200,
            )
        except Exception as e:
            logger.warning("VLM read_numbers failed (attempt 1): %s — retrying with fresh client", e)
            self._reset_client()
            try:
                response = self._get_client().chat(
                    system="只输出数字，不要任何解释。",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                            },
                            {
                                "type": "text",
                                "text": "龙门币=?\n合成玉=?\n源石=?\n理智=?\n只输出4行，每行格式：资源名=数字。",
                            },
                        ],
                    }],
                    max_tokens=200,
                )
            except Exception as e2:
                logger.warning("VLM read_numbers retry also failed: %s", e2)
                return {}

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        return self._parse_numbers(text)

    def invalidate_cache(self, screen_hash: str) -> None:
        """Remove cached VLM results for a given screen hash."""
        self._cache.pop(screen_hash, None)

    @staticmethod
    def _parse_numbers(text: str) -> dict[str, str]:
        import re
        result: dict[str, str] = {}
        patterns = [
            (r"龙门币[=:：\s]*(\d[\d,]*)", "龙门币"),
            (r"合成玉[=:：\s]*(\d[\d,]*)", "合成玉"),
            (r"源石[=:：\s]*(\d[\d,]*)", "源石"),
            (r"理智[=:：\s]*(\d[\d,/]*)", "理智"),
        ]
        for pattern, key in patterns:
            match = re.search(pattern, text)
            if match:
                result[key] = match.group(1)
        return result

    @staticmethod
    def _extract_text(response) -> str:
        """Extract text from VLM response, including thinking blocks.

        MiMo-V2.5 extended thinking puts answers in 'thinking' blocks instead
        of 'text' blocks. This handles both.
        """
        parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
            elif block.type == "thinking":
                parts.append(block.thinking)
        return "".join(parts).strip()

    @staticmethod
    def _encode(image: Image.Image) -> str:
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=50)
        return base64.b64encode(buffered.getvalue()).decode()

    @staticmethod
    def _to_image(image: Image.Image | str | Path) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        return Image.open(image)

    def match_material(
        self,
        screenshot: Image.Image,
        material_name: str,
        template_image: Image.Image | None = None,
    ) -> dict | None:
        """Use VLM to find a material icon in a warehouse screenshot.

        Sends the screenshot + reference description to the VLM and asks it
        to locate the material by visual appearance (shape, color, pattern).
        This works where OpenCV template matching fails — different background
        colors, resolution differences, similar-looking family variants.

        If template_image is provided, sends both the reference icon and the
        screenshot to the VLM for visual comparison. Otherwise, relies on the
        VLM's internal knowledge of Arknights material iconography.

        Args:
            screenshot: PIL Image of the warehouse page.
            material_name: Chinese material name (e.g. '赤金', '源石碎片').
            template_image: Optional reference icon image for visual comparison.

        Returns:
            {"name": str, "position": [x, y], "confidence": float, "reasoning": str}
            or None if not found.
        """
        import re as _re

        w, h = screenshot.size
        # Resize for VLM — keep enough detail for icon recognition
        if w > 1200:
            ratio = 1200 / w
            screen_resized = screenshot.resize((1200, int(h * ratio)))
        else:
            screen_resized = screenshot.copy()
        sw, sh = screen_resized.size

        screen_b64 = self._encode(screen_resized)

        # Build content blocks
        content_blocks: list[dict] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": screen_b64},
            },
        ]

        # If we have a reference template image, include it for visual comparison
        if template_image is not None:
            # Template images are typically small (~60-80px). Don't resize too small.
            tw, th = template_image.size
            if max(tw, th) < 80:
                template_image = template_image.resize((tw * 2, th * 2), Image.NEAREST)
            tpl_b64 = self._encode(template_image.convert("RGB"))
            content_blocks.insert(0, {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": tpl_b64},
            })

        # Build the prompt
        if template_image is not None:
            prompt = (
                f"图A 是明日方舟材料「{material_name}」的参考图标。\n"
                f"图B 是仓库页面的截图。\n\n"
                f"请在图B中寻找与图A图标相同的材料「{material_name}」。\n"
                f"注意：图标可能大小略有不同、背景颜色可能不同（稀有度底色 vs 纯黑底），"
                f"但图标的形状、图案、颜色是相同的。\n\n"
                f"如果找到了，严格按以下格式输出一行：\n"
                f"FOUND|{material_name}|x坐标|y坐标|置信度(0-1)|数量\n"
                f"例如: FOUND|赤金|320|150|0.92|568\n\n"
                f"如果没找到，输出: NOT_FOUND|{material_name}\n\n"
                f"只输出结果行，不要加任何解释。坐标是图标中心在图B中的像素位置。"
            )
        else:
            prompt = (
                f"这是明日方舟的仓库/仓库页面截图。\n"
                f"请寻找材料「{material_name}」的图标。\n"
                f"图标旁通常有一个数字（数量）。\n"
                f"明日方舟材料图标的特点：小方块（~60-80px），中间有材料的图案，"
                f"背景可能是黑色或稀有度颜色（金/紫/蓝/绿/白）。\n\n"
                f"如果找到了，严格按以下格式输出一行：\n"
                f"FOUND|{material_name}|x坐标|y坐标|置信度(0-1)|数量\n"
                f"例如: FOUND|赤金|320|150|0.92|568\n\n"
                f"如果没找到，输出: NOT_FOUND|{material_name}\n\n"
                f"只输出结果行，不要加任何解释。"
            )

        content_blocks.append({"type": "text", "text": prompt})

        try:
            response = self._get_client().chat(
                system=(
                    "你是明日方舟材料图标识别器。你的任务是在仓库截图中定位指定材料的图标。"
                    "你要对比图标的形状、图案、颜色特征来做判断，而不是逐像素对比。"
                    "只输出要求的格式，不要额外解释。"
                ),
                messages=[{"role": "user", "content": content_blocks}],
                max_tokens=200,
            )
        except Exception as e:
            logger.warning("VLM match_material failed (attempt 1): %s — retrying", e)
            self._reset_client()
            try:
                response = self._get_client().chat(
                    system=(
                        "你是明日方舟材料图标识别器。你的任务是在仓库截图中定位指定材料的图标。"
                        "只输出要求的格式，不要额外解释。"
                    ),
                    messages=[{"role": "user", "content": content_blocks}],
                    max_tokens=200,
                )
            except Exception as e2:
                logger.warning("VLM match_material retry also failed: %s", e2)
                return None

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        text = text.strip()
        logger.info("VLM match_material('%s'): %s", material_name, text[:200])

        # Parse the response (with optional quantity field)
        found_match = _re.match(
            r"FOUND\|(.+?)\|(\d+)\|(\d+)\|([\d.]+)(?:\|(\d+))?",
            text, _re.IGNORECASE,
        )
        if found_match:
            name = found_match.group(1)
            vlm_x = int(found_match.group(2))
            vlm_y = int(found_match.group(3))
            confidence = float(found_match.group(4))
            quantity = int(found_match.group(5)) if found_match.group(5) else 0

            # Map coordinates back to original screenshot size
            scale_x = w / sw
            scale_y = h / sh
            orig_x = int(vlm_x * scale_x)
            orig_y = int(vlm_y * scale_y)

            return {
                "name": name,
                "position": [orig_x, orig_y],
                "confidence": confidence,
                "quantity": quantity,
                "vlm_coords": [vlm_x, vlm_y],
                "reasoning": text,
            }

        logger.debug("VLM match_material('%s'): not found (response: %s)", material_name, text[:100])
        return None

    def identify_icon(
        self,
        icon_crop: Image.Image,
        candidate_names: list[str] | None = None,
    ) -> dict | None:
        """Ask VLM to identify which Arknights material a cropped icon represents.

        Use this when template matching finds an icon but can't confidently
        determine which material it is (e.g., family conflicts like 固源岩 vs 固源岩组).

        Args:
            icon_crop: PIL Image of a single material icon (cropped from screenshot).
            candidate_names: Optional list of candidate material names to choose from.
                            If omitted, the VLM identifies from its own knowledge.

        Returns:
            {"name": str, "confidence": float} or None.
        """
        import re as _re

        # Ensure the icon is large enough for the VLM to see detail
        iw, ih = icon_crop.size
        if iw < 40 or ih < 40:
            icon_crop = icon_crop.resize(
                (max(iw * 3, 120), max(ih * 3, 120)),
                Image.LANCZOS,
            )
        elif iw < 80:
            icon_crop = icon_crop.resize((iw * 2, ih * 2), Image.LANCZOS)

        icon_b64 = self._encode(icon_crop.convert("RGB"))

        if candidate_names:
            candidates_str = "、".join(candidate_names)
            prompt = (
                f"这是从明日方舟仓库页面裁剪出的一个材料图标。\n"
                f"候选材料（从中选择一个最匹配的）：{candidates_str}\n\n"
                f"请根据图标的图案、颜色、形状判断这是哪种材料。\n"
                f"严格按以下格式输出一行：\n"
                f"IDENTIFIED|材料名|置信度(0-1)\n"
                f"例如: IDENTIFIED|固源岩组|0.88\n\n"
                f"只输出结果行，不要加任何解释。"
            )
        else:
            prompt = (
                f"这是从明日方舟仓库页面裁剪出的一个材料图标。\n"
                f"请根据图标的图案、颜色、形状判断这是哪种材料。\n"
                f"严格按以下格式输出一行：\n"
                f"IDENTIFIED|材料中文名|置信度(0-1)\n"
                f"例如: IDENTIFIED|固源岩|0.92\n\n"
                f"只输出结果行，不要加任何解释。"
            )

        try:
            response = self._get_client().chat(
                system=(
                    "你是明日方舟材料图标识别专家。你熟悉所有材料的图标外观。"
                    "通过图标的颜色、形状、内部图案来判断材料种类。"
                    "T1材料通常颜色单一、图案简单；T2/T3材料颜色更丰富、图案更复杂。"
                    "只输出要求的格式，不要额外解释。"
                ),
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": icon_b64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=100,
            )
        except Exception as e:
            logger.warning("VLM identify_icon failed: %s", e)
            return None

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        text = text.strip()

        m = _re.match(r"IDENTIFIED\|(.+?)\|([\d.]+)", text, _re.IGNORECASE)
        if m:
            return {
                "name": m.group(1),
                "confidence": float(m.group(2)),
            }

        logger.debug("VLM identify_icon: could not parse response: %s", text[:100])
        return None

    def scan_warehouse_materials(
        self,
        screenshot: Image.Image,
        known_materials: list[str] | None = None,
    ) -> list[dict]:
        """Use VLM to identify ALL material icons visible on a warehouse page.

        Sends the full warehouse page to VLM with a grid coordinate system.
        The VLM identifies materials by their visual appearance and returns
        names + grid positions.

        This is the VLM equivalent of MaterialMatcher.scan_warehouse_page(),
        but uses semantic visual understanding instead of pixel correlation.

        Args:
            screenshot: PIL Image of the warehouse page.
            known_materials: Optional list of material names the VLM should
                            consider. Limits hallucination risk.

        Returns:
            List of {"name": str, "position": [x, y], "confidence": float}.
        """
        import re as _re

        w, h = screenshot.size
        # Resize for VLM
        if w > 1200:
            ratio = 1200 / w
            img = screenshot.resize((1200, int(h * ratio)))
        else:
            img = screenshot.copy()
        sw, sh = img.size

        img_b64 = self._encode(img)

        if known_materials:
            materials_hint = (
                "以下是已知可能出现的材料列表（你只能从这里面选）：\n"
                + "、".join(known_materials[:100])
                + (f"\n… 等共 {len(known_materials)} 种" if len(known_materials) > 100 else "")
            )
        else:
            materials_hint = ""

        prompt = (
            f"这是明日方舟仓库（仓库）页面的截图。画面中有多个材料图标。\n"
            f"每个材料图标是一个小方块（60-80px），背景为黑色或稀有度颜色，"
            f"中间有材料的图案。图标旁通常有数量数字。\n\n"
            f"{materials_hint}\n\n"
            f"请识别屏幕上能看到的**所有材料图标**。对每个识别出的材料，"
            f"输出一行：\n"
            f"MATERIAL|材料中文名|x坐标|y坐标|置信度(0-1)\n"
            f"例如: MATERIAL|固源岩|120|300|0.95\n\n"
            f"按从左到右、从上到下的顺序列出。每个材料一行。"
            f"不要输出重复的材料。"
            f"如果不确定，置信度写低一些（如0.5），不要瞎猜。"
            f"完成后输出 DONE"
        )

        try:
            response = self._get_client().chat(
                system=(
                    "你是明日方舟仓库材料扫描器。你的任务是从仓库截图中识别所有可见的材料图标。"
                    "通过图标的形状、颜色、内部图案来区分不同材料。"
                    "T1材料图标简单（如纯色方块），T2/T3更复杂（多层颜色/图案）。"
                    "只输出要求的格式，完成后输出DONE。不要额外解释。"
                ),
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=2000,
            )
        except Exception as e:
            logger.warning("VLM scan_warehouse_materials failed: %s", e)
            self._reset_client()
            return []

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        materials: list[dict] = []
        scale_x = w / sw
        scale_y = h / sh

        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("DONE"):
                break
            m = _re.match(r"MATERIAL\|(.+?)\|(\d+)\|(\d+)\|([\d.]+)", line, _re.IGNORECASE)
            if m:
                name = m.group(1)
                vlm_x = int(m.group(2))
                vlm_y = int(m.group(3))
                confidence = float(m.group(4))

                materials.append({
                    "name": name,
                    "position": [int(vlm_x * scale_x), int(vlm_y * scale_y)],
                    "confidence": confidence,
                })

        logger.info(
            "VLM scan_warehouse_materials: identified %d materials on page",
            len(materials),
        )
        return materials

vlm_descriptor = VLMDescriptor()
