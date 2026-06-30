"""Extract game data from MAA resources into TerraAgent knowledge base JSON files.

Usage: python scripts/extract_maa_data.py

Reads from: d:/vsworkspace/MaaAssistantArknights/resource/
Writes to: src/knowledge/arknights/
"""

import json
import os
from pathlib import Path

MAA_RESOURCE = Path("d:/vsworkspace/MaaAssistantArknights/resource")
OUT_DIR = Path("d:/vsworkspace/terra-agent/src/knowledge/arknights")

# ── material name mapping (item ID → Chinese name) ──
# From MAA item_index.json
def load_item_names():
    """Load item ID → name mapping from MAA item_index.json."""
    with open(MAA_RESOURCE / "item_index.json", "r", encoding="utf-8") as f:
        items = json.load(f)
    return {item_id: info["name"] for item_id, info in items.items()}


def extract_recruit_tags():
    """Convert MAA recruitment.json → TerraAgent recruit_tags.json.

    Groups operators by tag combination → rarity → possible operators.
    """
    with open(MAA_RESOURCE / "recruitment.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build tag → set of (name, rarity) entries
    tag_index: dict[str, list[dict]] = {}
    for op in data["operators"]:
        for tag in op["tags"]:
            if tag not in tag_index:
                tag_index[tag] = []
            tag_index[tag].append({"name": op["name"], "rarity": op["rarity"]})

    # Sort each tag's operators by rarity desc
    for tag in tag_index:
        tag_index[tag].sort(key=lambda x: -x["rarity"])

    # Build combined tag → operators lookup (for multi-tag recruitment)
    combined_index: dict[str, list[dict]] = {}
    for op in data["operators"]:
        tags_key = "|".join(sorted(op["tags"]))
        if tags_key not in combined_index:
            combined_index[tags_key] = []
        combined_index[tags_key].append({"name": op["name"], "rarity": op["rarity"]})

    result = {
        "description": "公招标签→干员映射。用于OCR识别标签后查表判断高星干员。",
        "operators": data["operators"],
        "tag_index": tag_index,
        "tag_combination_index": combined_index,
    }

    out_path = OUT_DIR / "recruit_tags.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  → {out_path} ({len(data['operators'])} operators, {len(tag_index)} tags)")


def extract_base_skills():
    """Convert MAA infrast.json → TerraAgent base_skills.json.

    Flattens the nested facility→skills structure into a searchable list.
    """
    with open(MAA_RESOURCE / "infrast.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    skills = []
    for facility, facility_data in data.items():
        if facility == "maxNumOfOpers":
            continue  # skip duplicate key issue
        if isinstance(facility_data, dict) and "skills" in facility_data:
            for skill_id, skill in facility_data["skills"].items():
                skills.append({
                    "id": skill_id,
                    "facility": facility,
                    "name": skill.get("name", [skill_id]),
                    "description": skill.get("desc", []),
                    "efficiency": skill.get("efficient", {}),
                    "template": skill.get("template", ""),
                })

    result = {
        "description": "基建技能数据库。按facility分组，包含技能名称、效果、效率值。",
        "facilities": {
            facility: {
                "maxOpers": info.get("maxNumOfOpers", 0) if isinstance(info, dict) else 0,
                "products": info.get("products", []) if isinstance(info, dict) else [],
            }
            for facility, info in data.items()
            if isinstance(info, dict)
        },
        "skills": skills,
        "skills_by_facility": {
            facility: [s for s in skills if s["facility"] == facility]
            for facility in data
            if isinstance(data[facility], dict)
        },
    }

    out_path = OUT_DIR / "base_skills.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  → {out_path} ({len(skills)} skills across {len(result['skills_by_facility'])} facilities)")


def extract_stages():
    """Convert MAA stages.json → TerraAgent stages.json with item names.

    Adds item names from item_index.json for readability.
    """
    item_names = load_item_names()

    with open(MAA_RESOURCE / "stages.json", "r", encoding="utf-8") as f:
        stages = json.load(f)

    # Enrich with item names
    for stage in stages:
        for drop in stage.get("dropInfos", []):
            item_id = drop.get("itemId", "")
            if item_id in item_names:
                drop["itemName"] = item_names[item_id]

    # Organize by stage type
    main_stages = []
    supply_stages = []
    event_stages = []

    for stage in stages:
        code = stage.get("code", "")
        # Main story: 1-7, 2-3 format
        if "-" in code and code.split("-")[0].isdigit():
            main_stages.append(stage)
        # Supply: CE-5, LS-5, etc.
        elif any(code.startswith(p) for p in ("CE", "LS", "SK", "AP", "CA", "PR")):
            supply_stages.append(stage)
        else:
            event_stages.append(stage)

    result = {
        "description": "关卡数据库。包含AP消耗、掉落信息。",
        "total": len(stages),
        "main_story": main_stages,
        "supply": supply_stages,
        "event": event_stages,
    }

    out_path = OUT_DIR / "stages.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  → {out_path} ({len(stages)} stages: {len(main_stages)} main, {len(supply_stages)} supply, {len(event_stages)} event)")


def extract_materials():
    """Generate materials.json combining MAA item data + existing materials.py knowledge.

    Material recipes for crafting/farming planning.
    """
    item_names = load_item_names()

    # Material recipes from existing game knowledge + PRTS Wiki
    # Format: material_name → {recipe, drop_stages, tier}
    materials = {
        "固源岩": {
            "tier": 1,
            "description": "基础建材，1-7 高效掉落",
            "drop_stages": [
                {"code": "1-7", "ap": 6, "drop_rate": 1.2, "note": "单件理智效率最高"},
                {"code": "S2-12", "ap": 15, "drop_rate": 0.85, "note": "副产物更优"},
            ],
            "craft_from": None,  # Cannot be crafted upward
        },
        "固源岩组": {
            "tier": 2,
            "description": "中级建材",
            "drop_stages": [
                {"code": "2-4", "ap": 12, "drop_rate": 0.58},
                {"code": "4-6", "ap": 18, "drop_rate": 0.62},
                {"code": "S3-4", "ap": 18, "drop_rate": 0.72, "note": "推荐"},
            ],
            "craft_from": {"material": "固源岩", "count": 5, "lmd_cost": 200},
        },
        "提纯源岩": {
            "tier": 3,
            "description": "高级建材",
            "drop_stages": [
                {"code": "4-6", "ap": 18, "drop_rate": 0.08},
            ],
            "craft_from": {"material": "固源岩组", "count": 4, "lmd_cost": 400},
        },
        "装置": {
            "tier": 1,
            "description": "基础电子元件",
            "drop_stages": [
                {"code": "S3-4", "ap": 18, "drop_rate": 0.48},
            ],
            "craft_from": None,
        },
        "全新装置": {
            "tier": 2,
            "description": "中级电子元件",
            "drop_stages": [
                {"code": "3-4", "ap": 15, "drop_rate": 0.22},
            ],
            "craft_from": {"material": "装置", "count": 4, "lmd_cost": 300},
        },
        "异铁": {
            "tier": 1,
            "description": "基础金属材料",
            "drop_stages": [
                {"code": "S4-1", "ap": 21, "drop_rate": 0.45},
            ],
            "craft_from": None,
        },
        "异铁组": {
            "tier": 2,
            "description": "中级金属材料",
            "drop_stages": [
                {"code": "S4-1", "ap": 21, "drop_rate": 0.24},
            ],
            "craft_from": {"material": "异铁", "count": 4, "lmd_cost": 300},
        },
        "糖": {
            "tier": 1,
            "description": "基础糖类",
            "drop_stages": [
                {"code": "GT-6", "ap": 15, "drop_rate": 0.52},
            ],
            "craft_from": None,
        },
        "糖组": {
            "tier": 2,
            "description": "中级糖类",
            "drop_stages": [
                {"code": "2-5", "ap": 12, "drop_rate": 0.36},
            ],
            "craft_from": {"material": "糖", "count": 4, "lmd_cost": 300},
        },
        "糖聚块": {
            "tier": 3,
            "description": "高级糖类",
            "drop_stages": [],
            "craft_from": {"material": "糖组", "count": 4, "lmd_cost": 500},
        },
        "扭转醇": {
            "tier": 2,
            "description": "中级化工材料",
            "drop_stages": [
                {"code": "4-4", "ap": 18, "drop_rate": 0.46},
            ],
            "craft_from": None,
        },
        "龙门币": {
            "tier": 0,
            "description": "游戏通用货币",
            "drop_stages": [
                {"code": "CE-5", "ap": 30, "drop_rate": 7500, "note": "龙门币本（固定掉落）"},
                {"code": "CE-6", "ap": 36, "drop_rate": 10000, "note": "龙门币本（固定掉落）"},
            ],
            "craft_from": None,
        },
    }

    # Add item IDs from MAA item_index where we have names
    for item_id, name in item_names.items():
        # Only include materials, not furniture/etc
        if any(keyword in name for keyword in ["固源岩", "装置", "铁", "糖", "酮", "酯", "醇", "锰", "凝胶", "晶体", "溶剂", "切削液", "芯片"]):
            # Add item_id to existing material if name matches
            matched = False
            for mat_name in materials:
                if mat_name in name or name in mat_name:
                    if "item_ids" not in materials[mat_name]:
                        materials[mat_name]["item_ids"] = []
                    materials[mat_name]["item_ids"].append(item_id)
                    matched = True
                    break
            if not matched:
                # Only add if not already present
                pass

    result = {
        "description": "材料数据库。包含配方、推荐刷取关卡、掉落率。基于PRTS Wiki数据。",
        "materials": materials,
        "item_id_to_name": {k: v for k, v in item_names.items() if v in materials or any(m in v for m in materials)},
    }

    out_path = OUT_DIR / "materials.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  → {out_path} ({len(materials)} materials)")


def create_chip_schedule():
    """Create chip schedule data (fixed game rule)."""
    schedule = {
        "description": "芯片关卡排班表。每日轮换，周日全开。",
        "schedule": {
            "周一": {
                "day": 1,
                "chips": ["近卫芯片", "特种芯片"],
                "stages": [{"code": "PR-C-1", "ap": 18}, {"code": "PR-C-2", "ap": 36}],
            },
            "周二": {
                "day": 2,
                "chips": ["狙击芯片", "术师芯片"],
                "stages": [{"code": "PR-A-1", "ap": 18}, {"code": "PR-A-2", "ap": 36}],
            },
            "周三": {
                "day": 3,
                "chips": ["重装芯片", "医疗芯片"],
                "stages": [{"code": "PR-B-1", "ap": 18}, {"code": "PR-B-2", "ap": 36}],
            },
            "周四": {
                "day": 4,
                "chips": ["先锋芯片", "辅助芯片"],
                "stages": [{"code": "PR-D-1", "ap": 18}, {"code": "PR-D-2", "ap": 36}],
            },
            "周五": {
                "day": 5,
                "chips": ["近卫芯片", "特种芯片"],
                "stages": [{"code": "PR-C-1", "ap": 18}, {"code": "PR-C-2", "ap": 36}],
            },
            "周六": {
                "day": 6,
                "chips": ["狙击芯片", "术师芯片"],
                "stages": [{"code": "PR-A-1", "ap": 18}, {"code": "PR-A-2", "ap": 36}],
            },
            "周日": {
                "day": 0,
                "chips": ["全部芯片"],
                "stages": [],
                "note": "所有芯片关卡开放",
            },
        },
    }

    out_path = OUT_DIR / "chip_schedule.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schedule, f, ensure_ascii=False, indent=2)
    print(f"  → {out_path} (7-day schedule)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Extracting MAA game data → TerraAgent knowledge base...")
    print()

    extract_recruit_tags()
    extract_base_skills()
    extract_stages()
    extract_materials()
    create_chip_schedule()

    print()
    print("Done! Files written to src/knowledge/arknights/")


if __name__ == "__main__":
    main()
