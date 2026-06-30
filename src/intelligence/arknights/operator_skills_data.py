"""Curated operator -> base skill mapping for Arknights base scheduling.

Bootstrapping dataset compiled from:
  1. MAA custom_infrast reference layouts
  2. base_skills.json skill definitions
  3. Skill ID naming conventions

Data structure per operator:
  {"skill_id": "bskill_xxx", "facility": "Trade|Mfg|...",
   "products": ["PureGold", ...], "efficiency": {"PureGold": 30.0}, "name": "..."}
"""

OPERATOR_SKILLS: dict[str, list[dict]] = {
    "德克萨斯": [
        {"skill_id": "bskill_tra_texas", "facility": "Trade",
         "products": ["LMD", "Orundum"],
         "efficiency": {"LMD": 65.0, "Orundum": 65.0}, "name": "德克萨斯"},
    ],
    "拉普兰德": [
        {"skill_id": "bskill_tra_Lappland2", "facility": "Trade",
         "products": ["LMD", "Orundum"],
         "efficiency": {"LMD": 0.0, "Orundum": 0.0}, "name": "醉翁之意·β"},
    ],
    "能天使": [
        {"skill_id": "bskill_tra_exusiai", "facility": "Trade",
         "products": ["LMD", "Orundum"],
         "efficiency": {"LMD": 35.0, "Orundum": 35.0}, "name": "能天使"},
    ],
    "银灰": [
        {"skill_id": "bskill_tra_silverash", "facility": "Trade",
         "products": ["LMD", "Orundum"],
         "efficiency": {"LMD": 20.0, "Orundum": 20.0}, "name": "银灰(贸易站)"},
    ],
    "孑": [
        {"skill_id": "bskill_tra_jie", "facility": "Trade",
         "products": ["LMD", "Orundum"],
         "efficiency": {"LMD": 25.0, "Orundum": 25.0}, "name": "孑"},
    ],
    "琳琅诗怀雅": [
        {"skill_id": "bskill_tra_swirealt", "facility": "Trade",
         "products": ["LMD", "Orundum"],
         "efficiency": {"LMD": 25.0, "Orundum": 25.0}, "name": "琳琅诗怀雅"},
    ],
    "巫恋": [
        {"skill_id": "bskill_tra_witch", "facility": "Trade",
         "products": ["LMD"],
         "efficiency": {"LMD": 45.0}, "name": "巫恋"},
    ],
    "黑键": [
        {"skill_id": "bskill_tra_blackkey", "facility": "Trade",
         "products": ["LMD"],
         "efficiency": {"LMD": 35.0}, "name": "黑键"},
    ],
    "可露希尔": [
        {"skill_id": "bskill_tra_closure", "facility": "Trade",
         "products": ["LMD"],
         "efficiency": {"LMD": 35.0}, "name": "可露希尔"},
    ],
    "乌有": [
        {"skill_id": "bskill_tra_wuyou", "facility": "Trade",
         "products": ["LMD"],
         "efficiency": {"LMD": 40.0}, "name": "乌有"},
    ],
    "但书": [
        {"skill_id": "bskill_tra_danshu", "facility": "Trade",
         "products": ["LMD"],
         "efficiency": {"LMD": 45.0}, "name": "但书"},
    ],
    "伺夜": [
        {"skill_id": "bskill_tra_siye", "facility": "Trade",
         "products": ["LMD"],
         "efficiency": {"LMD": 25.0}, "name": "伺夜"},
    ],
    "卡夫卡": [
        {"skill_id": "bskill_tra_kafka", "facility": "Trade",
         "products": ["LMD"],
         "efficiency": {"LMD": 30.0}, "name": "卡夫卡"},
    ],
    "龙舌兰": [
        {"skill_id": "bskill_tra_dragontongue", "facility": "Trade",
         "products": ["LMD"],
         "efficiency": {"LMD": 25.0}, "name": "龙舌兰"},
    ],
    "推进之王": [
        {"skill_id": "bskill_tra_siege", "facility": "Trade",
         "products": ["LMD", "Orundum"],
         "efficiency": {"LMD": 30.0, "Orundum": 30.0}, "name": "推进之王"},
    ],
    "摩根": [
        {"skill_id": "bskill_tra_morgan", "facility": "Trade",
         "products": ["LMD"],
         "efficiency": {"LMD": 20.0}, "name": "摩根"},
    ],
    # Manufacturing - PureGold
    "清流": [
        {"skill_id": "bskill_man_qingliu", "facility": "Mfg",
         "products": ["PureGold"],
         "efficiency": {"PureGold": 45.0}, "name": "清流"},
    ],
    "温蒂": [
        {"skill_id": "bskill_man_weedy", "facility": "Mfg",
         "products": ["PureGold"],
         "efficiency": {"PureGold": 40.0}, "name": "温蒂"},
    ],
    "森蚺": [
        {"skill_id": "bskill_man_eunectes", "facility": "Mfg",
         "products": ["PureGold"],
         "efficiency": {"PureGold": 30.0}, "name": "森蚺"},
    ],
    "阿罗玛": [
        {"skill_id": "bskill_man_aroma", "facility": "Mfg",
         "products": ["PureGold"],
         "efficiency": {"PureGold": 35.0}, "name": "阿罗玛"},
    ],
    "槐琥": [
        {"skill_id": "bskill_man_waaifu", "facility": "Mfg",
         "products": ["PureGold"],
         "efficiency": {"PureGold": 35.0}, "name": "槐琥"},
    ],
    "迷迭香": [
        {"skill_id": "bskill_man_rosemary", "facility": "Mfg",
         "products": ["PureGold"],
         "efficiency": {"PureGold": 30.0}, "name": "迷迭香"},
    ],
    "苍苔": [
        {"skill_id": "bskill_man_cangtai", "facility": "Mfg",
         "products": ["PureGold"],
         "efficiency": {"PureGold": 35.0}, "name": "苍苔"},
    ],
    "砾": [
        {"skill_id": "bskill_man_gravel", "facility": "Mfg",
         "products": ["PureGold"],
         "efficiency": {"PureGold": 35.0}, "name": "砾"},
    ],
    "引星棘刺": [
        {"skill_id": "bskill_man_thorns", "facility": "Mfg",
         "products": ["PureGold"],
         "efficiency": {"PureGold": 35.0}, "name": "引星棘刺"},
    ],
    # Manufacturing - OriginStone
    "泡泡": [
        {"skill_id": "bskill_man_bubble", "facility": "Mfg",
         "products": ["OriginStone"],
         "efficiency": {"OriginStone": 40.0}, "name": "泡泡"},
    ],
    "火神": [
        {"skill_id": "bskill_man_vulcan", "facility": "Mfg",
         "products": ["OriginStone"],
         "efficiency": {"OriginStone": 35.0}, "name": "火神"},
    ],
    "褐果": [
        {"skill_id": "bskill_man_brownsugar", "facility": "Mfg",
         "products": ["OriginStone"],
         "efficiency": {"OriginStone": 30.0}, "name": "褐果"},
    ],
    # Manufacturing - Generic
    "斑点": [
        {"skill_id": "bskill_man_spot", "facility": "Mfg",
         "products": ["CombatRecord", "PureGold", "OriginStone"],
         "efficiency": {"all": 25.0}, "name": "斑点(通用)"},
    ],
    "末药": [
        {"skill_id": "bskill_man_myrrh", "facility": "Mfg",
         "products": ["CombatRecord", "PureGold"],
         "efficiency": {"all": 20.0}, "name": "末药(通用)"},
    ],
    "食铁兽": [
        {"skill_id": "bskill_man_feater", "facility": "Mfg",
         "products": ["CombatRecord", "PureGold", "OriginStone", "Chip"],
         "efficiency": {"all": 30.0}, "name": "食铁兽(通用)"},
    ],
    "Castle-3": [
        {"skill_id": "bskill_man_castle3", "facility": "Mfg",
         "products": ["CombatRecord", "PureGold", "OriginStone"],
         "efficiency": {"all": 30.0}, "name": "Castle-3",
         "level_required": 30},
    ],
    # Power Plant
    "承曦格雷伊": [
        {"skill_id": "bskill_pwr_greyyalter", "facility": "Power",
         "products": ["Drone"],
         "efficiency": {"Drone": 20.0}, "name": "承曦格雷伊"},
    ],
    "格雷伊": [
        {"skill_id": "bskill_pwr_greyy", "facility": "Power",
         "products": ["Drone"],
         "efficiency": {"Drone": 15.0}, "name": "格雷伊"},
    ],
    "烛煌": [
        {"skill_id": "bskill_pwr_candle", "facility": "Power",
         "products": ["Drone"],
         "efficiency": {"Drone": 15.0}, "name": "烛煌"},
    ],
    "澄闪": [
        {"skill_id": "bskill_pwr_goldenglow", "facility": "Power",
         "products": ["Drone"],
         "efficiency": {"Drone": 15.0}, "name": "澄闪"},
    ],
    # Control Center
    "重岳": [
        {"skill_id": "bskill_ctrl_zhongyue", "facility": "Control",
         "products": [], "efficiency": {"all": 0.07}, "name": "重岳"},
    ],
    "令": [
        {"skill_id": "bskill_ctrl_ling", "facility": "Control",
         "products": [], "efficiency": {"all": 0.05}, "name": "令"},
    ],
    "夕": [
        {"skill_id": "bskill_ctrl_xi", "facility": "Control",
         "products": [], "efficiency": {"all": 0.05}, "name": "夕"},
    ],
    "望": [
        {"skill_id": "bskill_ctrl_wang", "facility": "Control",
         "products": [], "efficiency": {"all": 0.05}, "name": "望"},
    ],
    "八幡海铃": [
        {"skill_id": "bskill_ctrl_hachiman", "facility": "Control",
         "products": [], "efficiency": {"all": 0.05}, "name": "八幡海铃"},
    ],
    "诗怀雅": [
        {"skill_id": "bskill_ctrl_swire", "facility": "Control",
         "products": [], "efficiency": {"all": 0.05}, "name": "诗怀雅"},
    ],
    "斩业星熊": [
        {"skill_id": "bskill_ctrl_hoshiguma", "facility": "Control",
         "products": [], "efficiency": {"all": 0.05}, "name": "斩业星熊"},
    ],
    "戴菲恩": [
        {"skill_id": "bskill_ctrl_delphine", "facility": "Control",
         "products": [], "efficiency": {"all": 0.05}, "name": "戴菲恩"},
    ],
    "灵知": [
        {"skill_id": "bskill_ctrl_gnosis", "facility": "Control",
         "products": [], "efficiency": {"all": 0.05}, "name": "灵知"},
    ],
    "阿米娅": [
        {"skill_id": "bskill_ctrl_amiya", "facility": "Control",
         "products": [], "efficiency": {"all": 0.05}, "name": "魔王传承"},
    ],
    "凯尔希": [
        {"skill_id": "bskill_ctrl_kaltsit", "facility": "Control",
         "products": [], "efficiency": {"all": 0.05}, "name": "凯尔希"},
    ],
    # Other facilities
    "凯尔希·思衡托": [
        {"skill_id": "bskill_hire_kaltsit2", "facility": "Office",
         "products": [], "efficiency": {"all": 40.0}, "name": "凯尔希·思衡托"},
    ],
    "斥罪": [
        {"skill_id": "bskill_hire_penance", "facility": "Office",
         "products": [], "efficiency": {"all": 35.0}, "name": "斥罪"},
    ],
    "真言": [
        {"skill_id": "bskill_meeting_shingon", "facility": "Reception",
         "products": [], "efficiency": {"all": 25.0}, "name": "真言"},
    ],
    "爱丽丝": [
        {"skill_id": "bskill_dorm_alice", "facility": "Dorm",
         "products": [], "efficiency": {"all": 0.0}, "name": "爱丽丝"},
    ],
    "车尔尼": [
        {"skill_id": "bskill_dorm_chopin", "facility": "Dorm",
         "products": [], "efficiency": {"all": 0.0}, "name": "车尔尼"},
    ],
    "塑心": [
        {"skill_id": "bskill_dorm_plastic", "facility": "Dorm",
         "products": [], "efficiency": {"all": 0.0}, "name": "塑心"},
    ],
    "煌": [
        {"skill_id": "bskill_dorm_blaze", "facility": "Dorm",
         "products": [], "efficiency": {"all": 0.0}, "name": "煌"},
    ],
    "电弧": [
        {"skill_id": "bskill_dorm_arc", "facility": "Dorm",
         "products": [], "efficiency": {"all": 0.0}, "name": "电弧"},
    ],
    "年": [
        {"skill_id": "bskill_proc_nian", "facility": "Processing",
         "products": [], "efficiency": {"all": 100.0}, "name": "年(加工站)"},
    ],
}


def get_operator_facilities(name: str) -> set[str]:
    """Return set of facility types this operator can work in."""
    skills = OPERATOR_SKILLS.get(name, [])
    return {s["facility"] for s in skills}


def get_operator_products(name: str, facility: str) -> dict[str, float]:
    """Return {product: efficiency%} for an operator in a given facility."""
    skills = OPERATOR_SKILLS.get(name, [])
    result: dict[str, float] = {}
    for s in skills:
        if s["facility"] != facility:
            continue
        eff = s.get("efficiency", {})
        for product, value in eff.items():
            if product == "all":
                continue
            result[product] = result.get(product, 0.0) + value
        if "all" in eff:
            result["_universal"] = result.get("_universal", 0.0) + eff["all"]
    return result
