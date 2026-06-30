"""Arknights operator data — promotion requirements and rarity-based cost tables.

Standard E1/E2 promotion costs by rarity (community-verified).
These are the fixed LMD + chip costs; variable materials vary by operator
but follow approximate patterns that can be estimated.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ── Rarity-based elite promotion costs ────────────────────────────────
# LMD costs are exact. Material counts are approximate (typical for the rarity tier).
# For precise costs, operator-specific overrides go in PROMOTION_COSTS_SPECIFIC.

@dataclass
class EliteCost:
    lmd: int                          # LMD cost
    chip_type: str = ""               # chip prefix (e.g. "近卫", "狙击", ...)
    chip_count: int = 0               # number of chips (3 for E1, 4 for E2)
    chip_pack_count: int = 0          # chip packs (0 for E1, 3 for E2 dualchip)
    t4_materials: int = 0             # approximate T4 (purple) material count
    t3_materials: int = 0             # approximate T3 (blue) material count
    # "level up from 1 to max" cost
    level_lmd: int = 0                # LMD for leveling from 1 to E1max or E2max
    level_battle_records: int = 0     # approximate battle records needed


# Elite promotion costs: {rarity: {target_elite: EliteCost}}
ELITE_COSTS_BY_RARITY: dict[int, dict[int, EliteCost]] = {
    1: {  # 1★ — can only reach E0 Lv30
    },
    2: {  # 2★ — can only reach E0 Lv30
    },
    3: {  # 3★ — max E1 Lv55
        1: EliteCost(lmd=10000, chip_type="", chip_count=0, t3_materials=2,
                     level_lmd=4000, level_battle_records=20),
    },
    4: {  # 4★
        1: EliteCost(lmd=15000, chip_type="", chip_count=3, t3_materials=3,
                     level_lmd=8000, level_battle_records=40),
        2: EliteCost(lmd=60000, chip_type="", chip_count=4, chip_pack_count=3,
                     t4_materials=3, t3_materials=5,
                     level_lmd=30000, level_battle_records=120),
    },
    5: {  # 5★
        1: EliteCost(lmd=20000, chip_type="", chip_count=4, t3_materials=4,
                     level_lmd=12000, level_battle_records=60),
        2: EliteCost(lmd=120000, chip_type="", chip_count=4, chip_pack_count=4,
                     t4_materials=4, t3_materials=6,
                     level_lmd=60000, level_battle_records=240),
    },
    6: {  # 6★
        1: EliteCost(lmd=30000, chip_type="", chip_count=5, t3_materials=5,
                     level_lmd=20000, level_battle_records=80),
        2: EliteCost(lmd=180000, chip_type="", chip_count=4, chip_pack_count=4,
                     t4_materials=5, t3_materials=8,
                     level_lmd=120000, level_battle_records=400),
    },
}

# Chip type by operator class (approximate — maps to chip prefix)
_CLASS_CHIP_MAP: dict[str, str] = {
    "近卫": "近卫", "狙击": "狙击", "术师": "术师", "先锋": "先锋",
    "重装": "重装", "医疗": "医疗", "辅助": "辅助", "特种": "特种",
}


def get_elite_cost(
    rarity: int, target_elite: int, operator_class: str = "",
) -> EliteCost | None:
    """Get the standard elite promotion cost for an operator by rarity.

    Returns a COPY (not the shared dataclass) so callers can safely
    mutate chip_type without corrupting the global template.

    Returns None if the target elite is not achievable for this rarity.
    """
    import copy
    costs = ELITE_COSTS_BY_RARITY.get(rarity, {})
    ec = costs.get(target_elite)
    if ec is None:
        return None
    # Deep copy to prevent mutation of the shared global template
    ec = copy.deepcopy(ec)
    if operator_class and ec.chip_count > 0:
        chip = _CLASS_CHIP_MAP.get(operator_class, "")
        if chip:
            if target_elite == 1:
                ec.chip_type = f"{chip}芯片"
            else:
                ec.chip_type = f"{chip}芯片组"
    return ec


def estimate_total_lmd(
    rarity: int, from_elite: int, from_level: int,
    to_elite: int, to_level: int = 1,
) -> int:
    """Estimate total LMD needed to go from (E_from, Lv_from) to (E_to, Lv_to).

    This is a rough estimate — actual costs depend on the operator.
    """
    total = 0
    for target in range(from_elite + 1, to_elite + 1):
        cost = get_elite_cost(rarity, target)
        if cost:
            total += cost.lmd + cost.level_lmd
    return total


# ── Operator-specific promotion cost overrides ────────────────────────

PROMOTION_COSTS_SPECIFIC: dict[str, dict[int, dict[str, int]]] = {
    "银灰": {
        2: {"近卫芯片组": 4, "装置": 8, "糖聚块": 7},
    },
    "能天使": {
        2: {"狙击芯片组": 4, "固源岩组": 10, "糖聚块": 5},
    },
}


def get_promotion_cost(operator: str, elite_level: int = 2) -> dict[str, int]:
    """Get materials needed for promotion (operator-specific)."""
    op_costs = PROMOTION_COSTS_SPECIFIC.get(operator, {})
    return op_costs.get(elite_level, {})


def get_full_material_tree(operator: str, elite_level: int = 2) -> dict[str, int]:
    """Get the full material tree for an operator promotion, including crafted sub-materials."""
    from src.games.arknights.materials import get_craft_requirements

    direct = get_promotion_cost(operator, elite_level)
    result: dict[str, int] = {}
    for material, count in direct.items():
        sub = get_craft_requirements(material, count)
        for mat, qty in sub.items():
            result[mat] = result.get(mat, 0) + qty
    return result


# ── Material inventory (from depot scan) ──────────────────────────────

@dataclass
class MaterialStock:
    """Snapshot of the player's material warehouse."""
    items: dict[str, int] = field(default_factory=dict)  # material_name -> quantity
    lmd: int = 0
    scanned_at: str = ""

    @staticmethod
    def _maa_aliases(material: str) -> list[str]:
        """Return MAA item names that are TRUE synonyms of the given material.

        Only includes 1:1 naming variants (same item, different names).
        Does NOT include tier conversions (固源岩 ≠ 固源岩组, etc.).
        """
        _ALIASES: dict[str, list[str]] = {
            "赤金":       ["赤金", "金条", "纯金"],
            "源石碎片":   ["源石碎片"],
            "固源岩":     ["固源岩", "源岩"],
            "固源岩组":   ["固源岩组"],
            "装置":       ["装置"],
            "异铁":       ["异铁"],
            "异铁组":     ["异铁组"],
            "糖":         ["糖"],
            "聚酸酯":     ["聚酸酯"],
            "酮凝集组":   ["酮凝集组"],
            "合成玉":     ["合成玉"],
            "龙门币":     ["龙门币", "LMD"],
            "无人机":     ["无人机"],
        }
        return _ALIASES.get(material, [material])

    def get_any(self, material: str, *extra_names: str) -> int:
        """Look up a material by name, trying multiple naming variants.

        Returns the FIRST match — does NOT sum across aliases.
        Use this to find items that might be stored under different names.
        """
        aliases = list(self._maa_aliases(material))
        if extra_names:
            aliases.extend(extra_names)
        for name in aliases:
            qty = self.items.get(name, 0)
            if qty > 0:
                return qty
        return 0

    def get_all(self, material: str) -> int:
        """Get the exact quantity stored under the given name.

        No alias lookup — returns 0 if the exact key doesn't exist.
        """
        return self.items.get(material, 0)

    def has(self, material: str, count: int = 1) -> bool:
        """Check if ANY single alias has at least `count` quantity.

        Uses first-match semantics — appropriate for exact item checks
        (e.g. "do I have 4 近卫芯片组?").
        """
        return self.get_any(material) >= count

    def missing(self, required: dict[str, int]) -> dict[str, int]:
        """Return {material: deficit} for what's missing."""
        return {
            m: c - self.get_any(m)
            for m, c in required.items()
            if c > self.get_any(m)
        }

    def can_afford(self, lmd_cost: int = 0, materials: dict[str, int] | None = None) -> bool:
        """Check if the player has enough LMD and materials."""
        if lmd_cost > self.lmd > 0:  # if lmd > 0 (we know their balance)
            return False
        if materials:
            return not self.missing(materials)
        return True

    def is_empty(self) -> bool:
        return not self.items and self.lmd == 0


def parse_depot_scan_result(scan_output: dict) -> MaterialStock:
    """Convert a depot scan result into MaterialStock.

    depot scan returns: {"items": [{"name": "固源岩", "quantity": 42}, ...]}
    """
    stock = MaterialStock()
    items = scan_output.get("items", scan_output.get("materials", []))
    for entry in items:
        name = entry.get("name", "")
        qty = entry.get("quantity", entry.get("count", 0))
        if name and qty > 0:
            stock.items[name] = qty
            if name in ("龙门币", "LMD", "钱"):
                stock.lmd = max(stock.lmd, qty)
    # Also try top-level LMD field
    if stock.lmd == 0:
        stock.lmd = scan_output.get("lmd", scan_output.get("龙门币", 0))
    return stock
