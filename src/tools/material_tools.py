"""VLM material identification tools — single-material lookup with MAA icons.

vlm_match_material: find ONE named material in warehouse, return position + quantity
vlm_identify_icon: crop icon at (x,y), ask VLM what material it is

Neither writes to depot cache — use scan_depot or save_depot_resources for that.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image

from src.tools.registry import registry, ToolOutput

logger = logging.getLogger(__name__)

# ── MAA icon lookup ──────────────────────────────────────────────────

_ITEM_INDEX: dict[str, str] = {}
_LOADED = False

def _build_icon_dirs() -> list[Path]:
    """Build MAA icon search paths from config → env → fallback."""
    dirs: list[Path] = []
    try:
        from config.settings import config as _cfg
        _maa = getattr(_cfg, 'maa', None)
        if _maa:
            if _maa.resource_dir:
                dirs.append(Path(_maa.resource_dir) / "template" / "items")
            if _maa.root:
                dirs.append(Path(_maa.root) / "resource" / "template" / "items")
    except Exception:
        pass
    import os as _os
    _env_maa = _os.environ.get("MAA_ROOT") or _os.environ.get("MAA_DATA_DIR")
    if _env_maa:
        dirs.append(Path(_env_maa) / "resource" / "template" / "items")
    # Hard-coded fallbacks (developer machine)
    dirs.extend([
        Path("d:/edgedownload/MAA-v6.11.1-win-x64/resource/template/items"),
        Path("d:/MAA-v6.11.1-win-x64/resource/template/items"),
        Path("d:/MAA/resource/template/items"),
    ])
    return dirs


_ICON_DIRS = _build_icon_dirs()


def _load_index():
    global _ITEM_INDEX, _LOADED
    if _LOADED:
        return
    for d in _ICON_DIRS:
        p = d.parent.parent / "item_index.json"
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                for v in raw.values():
                    n, i = v.get("name", ""), v.get("icon", "")
                    if n and i:
                        _ITEM_INDEX[n] = i
                logger.info("Loaded %d items from MAA item_index", len(_ITEM_INDEX))
                break
            except Exception as e:
                logger.warning("MAA item_index load failed: %s", e)
    _LOADED = True


def _icon_for(name: str) -> Image.Image | None:
    _load_index()
    fn = _ITEM_INDEX.get(name)
    if not fn:
        return None
    for d in _ICON_DIRS:
        fp = d / fn
        if fp.exists():
            return Image.open(fp).convert("RGB")
    return None


def _adb_ok() -> bool:
    try:
        from src.device.emulator import emulator_manager
        return emulator_manager.first_online is not None
    except Exception:
        return False


def _maa_icons_ok() -> bool:
    """Check if MAA icon templates exist."""
    for d in _ICON_DIRS:
        if d.exists() and any(d.iterdir()):
            return True
    return False


# ── vlm_match_material — single-material VLM+icon lookup ────────────

def vlm_match_material_tool(name: str) -> ToolOutput:
    """Find ONE named material in the warehouse screenshot.

    Loads the MAA icon template for this material, sends it to VLM
    alongside the current screenshot for visual matching.
    Returns position and quantity. Does NOT write to depot cache.
    """
    from src.device.adb import get_adb
    from src.vision.vlm import vlm_descriptor

    adb = get_adb()
    img = adb.get_screenshot_image()
    icon = _icon_for(name)

    result = vlm_descriptor.match_material(img, name, template_image=icon)

    if result:
        px, py = result["position"]
        qty = result.get("quantity", 0)
        return ToolOutput(text=json.dumps({
            "found": True,
            "material": name,
            "quantity": qty,
            "position": [px, py],
        }, ensure_ascii=False))
    else:
        return ToolOutput(text=json.dumps({
            "found": False,
            "material": name,
            "message": f"VLM 未找到「{name}」——不在当前页面就调 scan_depot",
        }, ensure_ascii=False))


# ── vlm_identify_icon — crop + VLM asks "what is this?" ─────────────

def vlm_identify_icon_tool(x: int, y: int) -> ToolOutput:
    """Crop the icon at (x,y) and ask VLM what material it is."""
    from src.device.adb import get_adb
    from src.vision.vlm import vlm_descriptor

    adb = get_adb()
    img = adb.get_screenshot_image()

    r = 55
    crop = img.crop((
        max(0, x - r), max(0, y - r),
        min(img.width, x + r), min(img.height, y + r),
    ))
    result = vlm_descriptor.identify_icon(crop)

    if result:
        return ToolOutput(text=json.dumps({
            "success": True,
            "material": result,
            "crop_center": [x, y],
        }, ensure_ascii=False))
    else:
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": "VLM 无法识别该图标，换个位置或放大再试。",
        }, ensure_ascii=False))


# ── Register ────────────────────────────────────────────────────────

registry.register(
    name="vlm_match_material",
    description=(
        "★ [VLM 材料定位] 在当前仓库截图中找**一个**指定材料。\n"
        "使用 MAA 官方图标模板做视觉对比。只读不写缓存。\n"
        "参数 name=材料中文名（如'赤金'、'源石碎片'）。"
    ),
    handler=vlm_match_material_tool,
    check_fn=lambda: _adb_ok() and _maa_icons_ok(),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "材料中文名"},
        },
        "required": ["name"],
    },
)

registry.register(
    name="vlm_identify_icon",
    description=(
        "★ [VLM 图标识别] 裁剪坐标(x,y)处的图标，问 VLM 这是什么材料。"
        "参数 x,y = 图标中心像素坐标。"
    ),
    handler=vlm_identify_icon_tool,
    check_fn=_adb_ok,
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "图标中心 x"},
            "y": {"type": "integer", "description": "图标中心 y"},
        },
        "required": ["x", "y"],
    },
)
