"""Arknights material recipes and drop data.

Phase 1: manual data for core materials.
Phase 2: scrape from Penguin Logistics API / PRTS Wiki.
"""

from __future__ import annotations

# Material recipes: {material: {ingredient: count}}
RECIPES: dict[str, dict[str, int]] = {
    "糖聚块": {"糖": 4, "异铁组": 1},
    "固源岩组": {"固源岩": 5},
    "装置": {},  # Cannot craft, must farm
    "异铁组": {"异铁": 4},
}

# Stage drop rates: {stage: {material: drop_rate}}
DROP_RATES: dict[str, dict[str, float]] = {
    "GT-6": {"糖": 0.52},
    "CE-5": {"龙门币": 1.0},
    "S3-4": {"装置": 0.48},
    "S4-1": {"异铁组": 0.45},
}

# Stage sanity costs
STAGE_COSTS: dict[str, int] = {
    "GT-6": 15,
    "CE-5": 30,
    "S3-4": 18,
    "S4-1": 21,
}


def get_craft_requirements(material: str, quantity: int = 1) -> dict[str, int]:
    """Get the base materials needed to craft a given quantity of material."""
    if material not in RECIPES:
        return {material: quantity}

    requirements: dict[str, int] = {}
    recipe = RECIPES[material]
    for ingredient, count in recipe.items():
        sub_reqs = get_craft_requirements(ingredient, count * quantity)
        for sub_mat, sub_count in sub_reqs.items():
            requirements[sub_mat] = requirements.get(sub_mat, 0) + sub_count
    return requirements


def estimate_runs(material: str, quantity: int) -> list[dict]:
    """Estimate stage runs needed to farm a material. Returns list of {stage, runs, sanity}."""
    results: list[dict] = []
    for stage, drops in DROP_RATES.items():
        if material in drops:
            rate = drops[material]
            runs = int(quantity / rate) + 1
            sanity = runs * STAGE_COSTS.get(stage, 0)
            results.append({"stage": stage, "runs": runs, "sanity": sanity, "drop_rate": rate})
    return sorted(results, key=lambda x: x["sanity"])
