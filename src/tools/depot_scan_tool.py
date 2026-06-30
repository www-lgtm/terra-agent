"""Depot resource scanning for Arknights base scheduling.

Two tools:
  scan_depot  — auto-scan: VLM reads HUD + warehouse with MAA icon templates
  save_depot_resources — LLM manually reads numbers and injects them (fallback)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from PIL import Image

from src.tools.registry import registry, ToolOutput

logger = logging.getLogger(__name__)

# MAA icon template lookup.
_ITEM_INDEX: dict[str, str] = {}
_ITEM_INDEX_LOADED = False

_MAA_ICON_DIRS: list[Path] = []

def _build_maa_icon_dirs() -> list[Path]:
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
    # Env var fallback
    import os as _os
    _env_maa = _os.environ.get("MAA_DATA_DIR") or _os.environ.get("MAA_ROOT")
    if _env_maa:
        dirs.append(Path(_env_maa) / "resource" / "template" / "items")
    # Hard-coded fallbacks (developer machine)
    dirs.extend([
        Path("d:/edgedownload/MAA-v6.11.1-win-x64/resource/template/items"),
        Path("d:/MAA-v6.11.1-win-x64/resource/template/items"),
        Path("d:/MAA/resource/template/items"),
    ])
    return dirs


_MAA_ICON_DIRS = _build_maa_icon_dirs()


def _load_item_index() -> dict[str, str]:
    global _ITEM_INDEX, _ITEM_INDEX_LOADED
    if _ITEM_INDEX_LOADED:
        return _ITEM_INDEX
    for item_dir in _MAA_ICON_DIRS:
        idx_path = item_dir.parent.parent / "item_index.json"
        if idx_path.exists():
            try:
                raw = json.loads(idx_path.read_text(encoding="utf-8"))
                for _k, v in raw.items():
                    name = v.get("name", "")
                    icon = v.get("icon", "")
                    if name and icon:
                        _ITEM_INDEX[name] = icon
                logger.info("Loaded %d items from MAA item_index.json", len(_ITEM_INDEX))
                break
            except Exception as e:
                logger.warning("Failed to load MAA item_index.json: %s", e)
    _ITEM_INDEX_LOADED = True
    return _ITEM_INDEX


def _load_icon_for(name: str) -> Image.Image | None:
    idx = _load_item_index()
    icon_file = idx.get(name)
    if not icon_file:
        return None
    for d in _MAA_ICON_DIRS:
        p = d / icon_file
        if p.exists():
            return Image.open(p).convert("RGB")
    return None


def _persist_depot_stock(stock) -> None:
    try:
        from src.intelligence.arknights.base_chain import SESSION_DIR
        if not SESSION_DIR.exists():
            return
        import json as _json
        sessions = sorted(
            [s for s in SESSION_DIR.iterdir()
             if s.is_dir() and (s / "state.json").exists()],
            key=lambda p: p.stat().st_mtime, reverse=True)
        if sessions:
            target = sessions[0]
            wh_path = target / "warehouse.json"
            data = _json.loads(wh_path.read_text(encoding="utf-8")) if wh_path.exists() else {"items": {}, "lmd": 0}
            data["lmd"] = stock.lmd
            items = data.setdefault("items", {})
            items["龙门币"] = stock.lmd
            items["赤金"] = stock.items.get("赤金", items.get("赤金", 0))
            items["固源岩"] = stock.items.get("固源岩", items.get("固源岩", 0))
            items["源石碎片"] = stock.items.get("源石碎片", items.get("源石碎片", 0))
            items["合成玉"] = stock.items.get("合成玉", items.get("合成玉", 0))
            wh_path.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("scan_depot: persisted to %s", wh_path)
    except Exception as e:
        logger.debug("scan_depot: persist failed: %s", e)


def _write_cache(lmd, puregold, orirock, origin_stone, orundum):
    from src.games.arknights.operators import MaterialStock
    from src.intelligence.arknights.base_scheduler import BaseScheduler
    stock = MaterialStock(items={}, lmd=lmd, scanned_at="vlm+template")
    if lmd > 0: stock.lmd = lmd
    if puregold > 0: stock.items["赤金"] = puregold
    if orirock > 0: stock.items["固源岩"] = orirock
    if origin_stone > 0: stock.items["源石碎片"] = origin_stone
    if orundum > 0: stock.items["合成玉"] = orundum
    cache = BaseScheduler._get_cache() or {}
    cache["depot_stock"] = stock
    BaseScheduler._set_cache(cache)
    _persist_depot_stock(stock)
    logger.info("scan_depot: cached lmd=%d gold=%d rock=%d stone=%d",
                lmd, puregold, orirock, origin_stone)


# ═══════════════════════════════════════════════════════════════
# scan_depot — auto-scan with MAA icon templates
# ═══════════════════════════════════════════════════════════════

_SCAN_MATERIALS = ["赤金", "源石碎片"]


def scan_depot_tool() -> ToolOutput:
    """Auto-scan main screen + warehouse with VLM and MAA icon templates.

    Phase 0: VLM reads 龙门币 / 合成玉 from main screen HUD.
    Phase 1: Navigates to warehouse, passes MAA icon PNGs to VLM for
             visual matching → gets position + quantity of each material.
    """
    from src.device.adb import get_adb
    adb = get_adb()
    t0 = time.monotonic()

    lmd = orundum = 0
    img_main = None
    w_main = h_main = 0

    try:
        img_main = adb.get_screenshot_image()
        w_main, h_main = img_main.size
    except Exception as e:
        logger.warning("scan_depot: main screenshot failed: %s", e)

    if img_main is not None:
        try:
            from src.vision.vlm import vlm_descriptor
            nums = vlm_descriptor.read_numbers(img_main)
            lmd = int(nums.get("龙门币", "0").replace(",", ""))
            orundum = int(nums.get("合成玉", "0").replace(",", ""))
            logger.info("scan_depot: HUD LMD=%d Orundum=%d", lmd, orundum)
        except Exception as e:
            logger.warning("scan_depot: VLM HUD failed: %s", e)

    notes: list[str] = []
    results: dict[str, int] = {}

    if img_main is not None:
        try:
            from src.vision.template_match import template_matcher
            template_matcher.ensure_templates_for_game("arknights")
            mp = template_matcher.match(img_main, "warehouse", 0.6)
            if mp:
                adb.tap(mp["center"][0], mp["center"][1])
            else:
                adb.tap(int(w_main * 0.80), int(h_main * 0.95))
            time.sleep(2.0)
        except Exception as e:
            logger.warning("scan_depot: tap failed: %s", e)
            notes.append(f"进仓库失败: {e}")

        try:
            img_wh = adb.get_screenshot_image()
        except Exception as e:
            logger.warning("scan_depot: warehouse screenshot failed: %s", e)
            img_wh = None

        if img_wh is not None:
            from src.vision.vlm import vlm_descriptor
            for name in _SCAN_MATERIALS:
                try:
                    icon = _load_icon_for(name)
                    r = vlm_descriptor.match_material(img_wh, name, template_image=icon)
                    if r:
                        qty = r.get("quantity", 0)
                        x, y = r["position"]
                        results[name] = qty
                        notes.append(f"{name}: {qty}个 (VLM+图标@{x},{y})")
                    else:
                        notes.append(f"{name}: 未找到")
                except Exception as e:
                    logger.warning("scan_depot: VLM %s failed: %s", name, e)
                    notes.append(f"{name}: VLM失败({e})")

    puregold = results.get("赤金", 0)
    orirock = results.get("固源岩", 0)
    origin_stone = results.get("源石碎片", 0)

    _write_cache(lmd, puregold, orirock, origin_stone, orundum)

    elapsed = (time.monotonic() - t0) * 1000
    detail = "\n".join(f"  {n}" for n in notes) if notes else "  未执行"
    msg = (f"仓库扫描完成 ({elapsed/1000:.0f}s):\n"
           f"  龙门币: {lmd}\n"
           f"  合成玉: {orundum}\n{detail}\n\n排班缓存已就绪。")

    return ToolOutput(text=json.dumps({
        "success": True,
        "lmd": lmd, "orundum": orundum,
        "puregold": puregold, "orirock": orirock,
        "origin_stone": origin_stone,
        "vlm_notes": notes,
        "elapsed_ms": round(elapsed, 1),
        "engine": "VLM + MAA icon templates",
        "message": msg,
    }, ensure_ascii=False, indent=2))


# ═══════════════════════════════════════════════════════════════
# save_depot_resources — LLM manual injection (fallback)
# ═══════════════════════════════════════════════════════════════

def save_depot_resources(
    lmd: int = 0,
    puregold: int = -1,
    orirock: int = -1,
    origin_stone: int = -1,
    orundum: int = -1,
) -> ToolOutput:
    """LLM reads resource numbers from the game and injects them here.
    -1 = leave unchanged.  0 = confirmed zero.
    """
    from src.games.arknights.operators import MaterialStock
    from src.intelligence.arknights.base_scheduler import BaseScheduler

    cache = BaseScheduler._get_cache()
    existing = cache.get("depot_stock") if cache else None
    stock = (MaterialStock(items=dict(existing.items), lmd=existing.lmd, scanned_at=existing.scanned_at)
             if existing and isinstance(existing, MaterialStock)
             else MaterialStock(items={}, lmd=0, scanned_at="llm"))

    updated: list[str] = []
    if lmd > 0:
        stock.lmd = lmd
        updated.append(f"龙门币={lmd}")
    if puregold >= 0:
        stock.items["赤金"] = puregold
        updated.append(f"赤金={puregold}")
    if orirock >= 0:
        stock.items["固源岩"] = orirock
        updated.append(f"固源岩={orirock}")
    if origin_stone >= 0:
        stock.items["源石碎片"] = origin_stone
        updated.append(f"源石碎片={origin_stone}")
    if orundum >= 0:
        stock.items["合成玉"] = orundum
        updated.append(f"合成玉={orundum}")

    BaseScheduler._set_cache({"depot_stock": stock} if cache is None else cache)
    _persist_depot_stock(stock)

    return ToolOutput(text=json.dumps({
        "success": True,
        "message": f"已更新: {', '.join(updated)}" if updated else "未更新任何值",
        "lmd": stock.lmd,
        "puregold": stock.items.get("赤金", 0),
        "orirock": stock.items.get("固源岩", 0),
        "origin_stone": stock.items.get("源石碎片", 0),
        "orundum": stock.items.get("合成玉", 0),
    }, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════
def _maa_icons_available() -> bool:
    """Check if MAA item icon templates exist before offering to LLM."""
    for d in _MAA_ICON_DIRS:
        if d.exists() and any(d.iterdir()):
            return True
    return False


# Register both tools
# ═══════════════════════════════════════════════════════════════

registry.register(
    name="scan_depot",
    check_fn=_maa_icons_available,
    description=(
        "★★ [仓库关键资源扫描] 一键扫描基建排班所需资源。\n"
        "VLM 读取主界面龙门币/合成玉 + 用 MAA 官方图标模板在仓库里匹配\n"
        "赤金、固源岩、源石碎片并读取数量。\n"
        "结果直接写入排班缓存，扫描完即可进行基建排班。\n"
        "\n"
        "【前置条件】必须在明日方舟主界面\n"
        "【用时】约 10-15 秒"
    ),
    handler=scan_depot_tool,
    game="arknights",
    parameters={"type": "object", "properties": {}},
)

registry.register(
    name="save_depot_resources",
    description=(
        "【注入仓库资源数据】将你从游戏画面中读取的资源数值写入排班缓存。\n"
        "scan_depot 失败或需要手动补充时使用。\n"
        "\n"
        "参数：\n"
        "  - lmd: 龙门币（主界面顶部）\n"
        "  - puregold: 赤金（仓库）\n"
        "  - orirock: 固源岩（仓库）\n"
        "  - origin_stone: 源石碎片（仓库）\n"
        "  - orundum: 合成玉（主界面顶部）\n"
        "-1 = 不改变该值"
    ),
    parameters={
        "type": "object",
        "properties": {
            "lmd": {"type": "integer", "description": "龙门币数量"},
            "puregold": {"type": "integer", "description": "赤金数量"},
            "orirock": {"type": "integer", "description": "固源岩数量"},
            "origin_stone": {"type": "integer", "description": "源石碎片数量"},
            "orundum": {"type": "integer", "description": "合成玉数量"},
        },
    },
    handler=save_depot_resources,
    game="arknights",
)
