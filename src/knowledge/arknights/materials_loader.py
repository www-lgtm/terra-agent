"""明日方舟材料加载器 —— 委托通用模块，绑定 game='arknights'。"""

from functools import partial

from src.knowledge.materials_loader import (
    load_materials_index as _load_index,
    load_materials_by_name as _load_by_name,
    load_material_families as _load_families,
    get_family_for_material as _get_family,
)

load_materials_index = partial(_load_index, "arknights")
load_materials_by_name = partial(_load_by_name, "arknights")
load_material_families = partial(_load_families, "arknights")
get_family_for_material = partial(_get_family, game="arknights")
