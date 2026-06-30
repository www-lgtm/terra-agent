"""通用材料数据库加载器。

自动检测游戏，从对应 knowledge/{game}/materials.json 加载。
"""

from __future__ import annotations

import json
from pathlib import Path


def load_materials_index(game: str = "arknights") -> list[dict]:
    """从 materials.json 加载材料列表。

    返回扁平列表，每个元素包含 name, tier, description, drop_stages, craft_from, item_ids。
    兼容多种 JSON 结构: dict of dicts, list, 嵌套 materials 字段。
    """
    file_path = Path(__file__).parent / game / "materials.json"
    if not file_path.exists():
        return []

    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    # 解包常见结构
    if isinstance(raw, dict):
        # {"materials": {name: info, ...}}
        materials_dict = raw.get("materials", raw)
        if isinstance(materials_dict, dict):
            return [
                {"name": name, **info}
                for name, info in materials_dict.items()
                if isinstance(info, dict)
            ]
    if isinstance(raw, list):
        return raw
    return []


def load_materials_by_name(game: str = "arknights") -> dict[str, dict]:
    """加载材料名 → 信息快速查找字典。"""
    index = load_materials_index(game)
    return {m["name"]: m for m in index}


def load_material_families(game: str = "arknights") -> dict[str, list[str]]:
    """通过 craft_from 链计算材料族系。

    返回 {family_base: [T1_name, T2_name, T3_name, ...]}。
    """
    by_name = load_materials_by_name(game)

    # 反向索引: 谁可以合成什么
    derived_from: dict[str, list[str]] = {}
    for name, info in by_name.items():
        craft = info.get("craft_from")
        if craft and isinstance(craft, dict):
            base = craft.get("material")
            if base and base in by_name:
                derived_from.setdefault(base, []).append(name)

    families: dict[str, list[str]] = {}
    for name, info in by_name.items():
        craft = info.get("craft_from")
        if craft is None or not isinstance(craft, dict):
            family = [name]
            current = name
            seen = {name}
            while current in derived_from and len(family) <= 10:
                children = derived_from[current]
                for child in children:
                    if child not in seen:
                        family.append(child)
                        seen.add(child)
                current = children[0] if children else current
            if len(family) > 1:
                families[name] = family

    return families


def get_family_for_material(material_name: str, game: str = "arknights") -> list[str]:
    """获取指定材料所属的族系链，按 tier 升序排列。"""
    families = load_material_families(game)
    by_name = load_materials_by_name(game)
    for family in families.values():
        if material_name in family:
            return sorted(
                family,
                key=lambda n: by_name.get(n, {}).get("tier", 99),
            )
    return [material_name]
