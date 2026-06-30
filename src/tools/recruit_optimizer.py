"""Recruitment tag optimizer for Arknights.

Given OCR'd recruitment tags from each slot, computes the optimal tag
combination and recommended time limit.  Supports two strategies:

- "collection" (图鉴优先): prioritize guaranteed new operators.
- "yellow_cert" (黄票优先): prioritize guaranteed duplicate 5★/6★ operators.

Guarantee detection — 5 tiers:
  5: 高级资深干员 → guaranteed 6★ (9:00)
  4: 资深干员 → guaranteed 5★ (9:00)
  3: natural 5★ lock — pool has no 2/3★, min ≥ 5, has actual 5★ (9:00)
  2: natural 4★+ lock — pool has no 2/3★, has obtainable 4★/5★/1★ (9:00)
  1: 支援机械 → guaranteed 1★ robot (1:00)
  0: 新手 / no guarantee → 3★/4★ default (1:00)

Core principle: 2★/3★ are assumed max-potential and worthless.
1★ robots are rare → valuable as new 5★.
Without 高级资深干员, 6★ are impossible.
Pools of only 6★ with no 高资 are empty → demoted.

Self-registers with the tool registry at import time.
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

from config.settings import config
from src.tools.registry import registry, ToolOutput

logger = logging.getLogger(__name__)

# ── Tag data cache ────────────────────────────────────────────────────

_tag_index: dict | None = None
_name_to_op: dict | None = None
_all_tags: set[str] | None = None


def _load_tag_data() -> tuple[dict, dict, set]:
    global _tag_index, _name_to_op, _all_tags
    if _tag_index is not None:
        return _tag_index, _name_to_op, _all_tags
    tag_path = Path(__file__).parent.parent / "knowledge" / "arknights" / "recruit_tags.json"
    raw = json.loads(tag_path.read_text("utf-8"))
    _tag_index = raw.get("tag_index", {})
    _name_to_op = {op["name"]: op for op in raw.get("operators", [])}
    _all_tags = set(_tag_index.keys())
    logger.info("Loaded recruit tag data: %d tags, %d operators", len(_all_tags), len(_name_to_op))
    return _tag_index, _name_to_op, _all_tags


# ── Owned operator cache ──────────────────────────────────────────────

def _load_owned_operators() -> set[str]:
    path = Path(config.DATA_DIR) / "arknights" / "box_scan_result.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text("utf-8"))
        return {op["name"] for op in data.get("operators", [])}
    except Exception:
        logger.warning("Failed to load owned operators cache", exc_info=True)
        return set()


# ── Valuable-operator helpers ─────────────────────────────────────────

def _is_valuable_new(name: str, rarity: int, owned: set[str]) -> bool:
    """Only 1★/5★/6★ count as valuable. 2/3/4★ assumed max-potential."""
    if rarity not in (1, 5, 6):
        return False
    return name not in owned


def _is_valuable_dupe(name: str, rarity: int, owned: set[str]) -> bool:
    return rarity >= 5 and name in owned


# ── Fuzzy tag matching ────────────────────────────────────────────────

def _fuzzy_match_tag(input_tag: str, known_tags: set[str],
                     threshold: float = 0.7) -> str | None:
    if input_tag in known_tags:
        return input_tag
    from difflib import SequenceMatcher
    best_score, best_match = 0.0, None
    for tag in known_tags:
        score = SequenceMatcher(None, input_tag, tag).ratio()
        if score > best_score:
            best_score, best_match = score, tag
    if best_score >= threshold and best_match is not None:
        logger.debug("Fuzzy tag match: '%s' -> '%s' (%.2f)", input_tag, best_match, best_score)
        return best_match
    return None


# ── Set intersection engine ───────────────────────────────────────────

def _intersect_operators(selected_tags: list[str], tag_index: dict) -> list[dict]:
    if not selected_tags:
        return []
    sorted_tags = sorted(selected_tags, key=lambda t: len(tag_index.get(t, [])))
    candidates = tag_index.get(sorted_tags[0], [])
    if not candidates:
        return []
    other_name_sets = [{op["name"] for op in tag_index.get(t, [])} for t in sorted_tags[1:]]
    return [op for op in candidates if all(op["name"] in ns for ns in other_name_sets)]


# ── Guarantee detection ───────────────────────────────────────────────

def _detect_guarantee(
    selected_tags: list[str],
    matched_ops: list[dict],
) -> tuple[int, str, list[dict]]:
    """Returns (tier, label, effective_ops).

    Effective ops: the subset of matched_ops that can actually appear.
    Without 高资, 6★ are impossible.  2★/3★ are assumed worthless.
    1★ robots are rare & valuable.
    """
    rarities = set(op["rarity"] for op in matched_ops)
    min_r = min(rarities) if rarities else 0
    # "No trash" = no 2★ or 3★ in the actual pool
    no_trash = all(r not in (2, 3) for r in rarities)

    if "高级资深干员" in selected_tags:
        six = [op for op in matched_ops if op["rarity"] == 6]
        return (5, "高级资深干员 -> 必出6★", six)

    if "资深干员" in selected_tags:
        five = [op for op in matched_ops if op["rarity"] == 5]
        return (4, "资深干员 -> 必出5★", five)

    # Tier 1: 支援机械 guarantee (must come before natural locks —
    # robots are excluded at 9:00 so pool-of-only-robots shouldn't get tier ≥2)
    if "支援机械" in selected_tags:
        return (1, "支援机械 -> 必出1★", matched_ops)

    # ── Natural rarity locks (Tier 2 / 3) ──
    # Common: no trash (2/3★) in pool, min_r acceptable, all rarities in allowed set.
    def _natural_lock(min_allowed: int, allowed_rarities: set[int],
                      tier: int, label: str,
                      target_rarities: set[int]) -> tuple[int, str, list[dict]] | None:
        if not no_trash:
            return None
        # Acceptable rarity composition?
        if not (min_r >= min_allowed or
                (min_r == 1 and all(r in allowed_rarities for r in rarities))):
            return None
        effective = [op for op in matched_ops if op["rarity"] in target_rarities]
        if effective:
            return (tier, label, effective)
        return None

    # Tier 3: natural 5★ lock — pool has only {5,6} or {1,5,6}
    result = _natural_lock(5, {1, 5, 6}, 3, "必出5★（标签锁定）", {5})
    if result:
        return result

    # Tier 2: natural 4★+ lock — pool has {4,5,6} or {1,4,5,6}
    # NOTE: at 9:00 robots are excluded; target 4★/5★ only for accuracy.
    result = _natural_lock(4, {1, 4, 5, 6}, 2, "必出4★+（标签锁定）", {4, 5})
    if result:
        return result

    # Tier 0: 新手 / no guarantee
    if "新手" in selected_tags:
        return (0, "新手 -> 必出2★", matched_ops)

    return (0, "无保底 -> 3★/4★", matched_ops)


# ── Per-strategy scoring ──────────────────────────────────────────────

def _annotate_ops(ops: list[dict], owned: set[str]) -> tuple:
    annotated, val_new, val_dupes = [], [], []
    for op in ops:
        vn = _is_valuable_new(op["name"], op["rarity"], owned)
        vd = _is_valuable_dupe(op["name"], op["rarity"], owned)
        annotated.append({
            "name": op["name"], "rarity": op["rarity"],
            "is_valuable_new": vn, "is_valuable_dupe": vd,
        })
        if vn:
            val_new.append(annotated[-1])
        if vd:
            val_dupes.append(annotated[-1])
    return annotated, val_new, val_dupes


def _score_collection(selected_tags, all_annotated, val_new, matched_ops):
    tier, _, effective_ops = _detect_guarantee(selected_tags, matched_ops)
    has_new = 1 if val_new else 0
    pool = len(effective_ops)
    max_new = max((op["rarity"] for op in val_new), default=0)
    return (tier, has_new, -pool, max_new)


def _score_yellow_cert(selected_tags, all_annotated, val_dupes, matched_ops):
    tier, _, effective_ops = _detect_guarantee(selected_tags, matched_ops)
    has_dupe = 1 if (tier >= 3 and val_dupes) else 0
    max_dupe = max((op["rarity"] for op in val_dupes), default=0)
    pool = len(effective_ops)
    return (tier, has_dupe, max_dupe, -pool)


# ── Combination generator ─────────────────────────────────────────────

def _generate_combos(tags, tag_index, owned, strategy):
    score_fn = _score_collection if strategy == "collection" else _score_yellow_cert
    results = []
    max_k = min(3, len(tags))

    for k in range(1, max_k + 1):
        for combo in combinations(tags, k):
            selected = list(combo)
            ops = _intersect_operators(selected, tag_index)
            if not ops:
                continue

            annotated, val_new, val_dupes = _annotate_ops(ops, owned)
            tier, label, effective_ops = _detect_guarantee(selected, ops)
            score = score_fn(selected, annotated,
                             val_new if strategy == "collection" else val_dupes, ops)
            theoretical_max = max(op["rarity"] for op in ops)

            results.append({
                "selected_tags": selected,
                "matched_operators": sorted(
                    annotated, key=lambda o: (-o["is_valuable_new"], -o["rarity"], o["name"])),
                "max_rarity": theoretical_max,
                "guarantee_tier": tier,
                "guarantee_label": label,
                "effective_pool_size": len(effective_ops),
                "valuable_new_count": len(val_new),
                "valuable_dupe_count": len(val_dupes),
                "score": score,
            })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ── Time limit logic ──────────────────────────────────────────────────

def _recommend_time(guarantee_tier: int) -> str:
    if guarantee_tier >= 2:
        return "9:00"
    return "1:00"


# ── Alternative builder ────────────────────────────────────────────────

def _build_alternative(combo: dict, best: dict) -> dict:
    """Build an alternative entry with delta info relative to best combo."""
    best_set = set(best["selected_tags"])
    alt_set = set(combo["selected_tags"])

    removed = sorted(best_set - alt_set)
    added = sorted(alt_set - best_set)

    delta_parts = []
    if removed:
        delta_parts.append(f"去掉 {'、'.join(removed)}")
    if added:
        delta_parts.append(f"加上 {'、'.join(added)}")

    tier_diff = combo["guarantee_tier"] - best["guarantee_tier"]
    new_diff = combo["valuable_new_count"] - best["valuable_new_count"]
    dupe_diff = combo["valuable_dupe_count"] - best["valuable_dupe_count"]
    pool_diff = combo["effective_pool_size"] - best["effective_pool_size"]

    summary_parts = []
    if tier_diff != 0:
        summary_parts.append(f"保底等级 {'↑' if tier_diff > 0 else '↓'}{abs(tier_diff)}")
    if new_diff != 0:
        summary_parts.append(f"有价值新 {'+' if new_diff > 0 else ''}{new_diff}")
    if dupe_diff != 0:
        summary_parts.append(f"有价值重复 {'+' if dupe_diff > 0 else ''}{dupe_diff}")

    return {
        "selected_tags": combo["selected_tags"],
        "guarantee_tier": combo["guarantee_tier"],
        "guarantee_label": combo["guarantee_label"],
        "effective_pool_size": combo["effective_pool_size"],
        "valuable_new_count": combo["valuable_new_count"],
        "valuable_dupe_count": combo["valuable_dupe_count"],
        "delta": {
            "tag_change": "；".join(delta_parts) if delta_parts else (
                "标签相同（更少标签的组合）"
                if len(combo["selected_tags"]) < len(best["selected_tags"])
                else "标签相同"
            ),
            "tier_change": tier_diff if tier_diff != 0 else None,
            "valuable_new_change": new_diff if new_diff != 0 else None,
            "valuable_dupe_change": dupe_diff if dupe_diff != 0 else None,
            "pool_size_change": pool_diff if pool_diff != 0 else None,
            "summary": "；".join(summary_parts) if summary_parts else "评分接近，差异不大",
        },
    }


# ── Main tool handler ─────────────────────────────────────────────────

def optimize_recruit_tags(
    tags: list[list[str]],
    strategy: str = "collection",
) -> ToolOutput:
    """Optimize recruitment tag combinations for each slot.

    Two strategies:
      - "collection" (default): maximize guaranteed new 1★/5★/6★.
        2★/3★/4★ assumed already owned.
      - "yellow_cert": maximize guaranteed dupe 5★/6★ for yellow certs.

    Guarantee detection:
      5. 高级资深干员 -> 6★       9:00
      4. 资深干员 -> 5★           9:00
      3. natural 5★ lock          9:00
      2. natural 4★+ lock         9:00
      1. 支援机械 -> 1★ robot     1:00
      0. no guarantee             1:00
    """
    if not tags:
        return ToolOutput(text=json.dumps({"success": False, "error": "tags不能为空"}, ensure_ascii=False))
    if strategy not in ("collection", "yellow_cert"):
        return ToolOutput(text=json.dumps({
            "success": False,
            "error": f"strategy must be 'collection' or 'yellow_cert', got '{strategy}'",
        }, ensure_ascii=False))

    tag_index, _name_to_op, all_tags = _load_tag_data()
    owned = _load_owned_operators()
    logger.info("Recruit optimizer: strategy=%s, %d owned ops", strategy, len(owned))

    slot_results: list[dict] = []
    for idx, slot_tags in enumerate(tags):
        matched_tags: list[str] = []
        seen: set[str] = set()
        for raw_tag in slot_tags:
            matched = _fuzzy_match_tag(raw_tag.strip(), all_tags)
            if matched and matched not in seen:
                matched_tags.append(matched)
                seen.add(matched)

        if not matched_tags:
            slot_results.append({"slot_index": idx, "input_tags": slot_tags,
                                 "error": "没有识别到有效标签"})
            continue

        combos = _generate_combos(matched_tags, tag_index, owned, strategy)
        if not combos:
            slot_results.append({"slot_index": idx, "input_tags": slot_tags,
                                 "matched_tags": matched_tags, "error": "无有效标签组合"})
            continue

        best = combos[0]
        runner_up = combos[1] if len(combos) > 1 else None

        # ── 解释为什么最优 ──
        if runner_up is None:
            why_best = "唯一有效组合"
        elif best["guarantee_tier"] > runner_up["guarantee_tier"]:
            why_best = (
                f"保底等级更高（T{best['guarantee_tier']} vs T{runner_up['guarantee_tier']}），"
                f"本组合 {best['guarantee_label']}"
            )
        elif strategy == "collection":
            if best["valuable_new_count"] > runner_up["valuable_new_count"]:
                new_names = [o["name"] for o in best["matched_operators"] if o["is_valuable_new"]]
                why_best = (
                    f"有价值新干员更多（{best['valuable_new_count']} vs "
                    f"{runner_up['valuable_new_count']}），"
                    f"新增 {'/'.join(new_names[:3])}{'等' if len(new_names) > 3 else ''}"
                )
            elif best["effective_pool_size"] < runner_up["effective_pool_size"]:
                why_best = (
                    f"有效池更小（{best['effective_pool_size']} vs "
                    f"{runner_up['effective_pool_size']}），目标更精准"
                )
            elif best["max_rarity"] > runner_up["max_rarity"]:
                why_best = f"最高稀有度更高（{best['max_rarity']}★ vs {runner_up['max_rarity']}★）"
            else:
                why_best = "与其他组合评分相同，选标签更少/更简洁的组合"
        else:  # yellow_cert
            if best["valuable_dupe_count"] > runner_up["valuable_dupe_count"]:
                why_best = (
                    f"有价值重复干员更多（{best['valuable_dupe_count']} vs "
                    f"{runner_up['valuable_dupe_count']}）"
                )
            elif best["effective_pool_size"] < runner_up["effective_pool_size"]:
                why_best = (
                    f"有效池更小（{best['effective_pool_size']} vs "
                    f"{runner_up['effective_pool_size']}），目标更精准"
                )
            else:
                why_best = "与其他组合评分相同，选标签更少/更简洁的组合"

        selected = best["selected_tags"]
        tier = best["guarantee_tier"]
        time_str = _recommend_time(tier)

        if strategy == "collection":
            is_locked = best["valuable_new_count"] >= 1 and tier >= 3
        else:
            is_locked = best["valuable_dupe_count"] >= 1 and tier >= 3

        # ── Build actionable instruction ──
        # tier ≥ 2: 标签锁4★+ → 9小时才能生效
        # tier = 1: 支援机械 → 1:00 即可（机器人不靠时限）
        # tier = 0: 无保底 → 浪费公招券，优先刷新
        needs_nine_hours = tier >= 2
        is_robot = tier == 1 and "支援机械" in selected
        should_refresh = tier <= 1 and not is_robot

        # Timer shortcut: 点小时▼箭头从 01 下翻直接到 09（环绕），比点▲快很多
        if needs_nine_hours:
            timer_instruction = (
                "⚠️ 必须调成9:00！保底由时限赋予，标签自身不锁稀有度。"
                "**快速调时：点小时▼箭头（不是▲！），01 下翻直接到 09，点1次就行。**"
            )
            slot_action = "run"
            slot_action_reason = "标签有保底，拉满9小时执行"
        elif is_robot:
            timer_instruction = "1:00 即可（支援机械不靠时限）"
            slot_action = "run"
            slot_action_reason = "支援机械保底，1小时执行"
        elif should_refresh:
            timer_instruction = (
                "🔄 不建议执行！标签无保底，出3★浪费公招券。"
                "如有联络次数，点「刷新标签」换一组。"
                "没有联络次数了才用1:00执行。"
            )
            slot_action = "refresh"
            slot_action_reason = "标签无保底，刷新优先"
        else:
            timer_instruction = "1:00 即可"
            slot_action = "run"
            slot_action_reason = "无保底但不可刷新"

        slot_results.append({
            "slot_index": idx,
            "input_tags": slot_tags,
            "matched_tags": matched_tags,
            "best_combo": {
                "selected_tags": selected,
                "guarantee_tier": tier,
                "guarantee_label": best["guarantee_label"],
                "max_rarity": best["max_rarity"],
                "matched_operators": best["matched_operators"][:20],
                "total_operators": len(best["matched_operators"]),
                "effective_pool_size": best["effective_pool_size"],
                "valuable_new_count": best["valuable_new_count"],
                "valuable_dupe_count": best["valuable_dupe_count"],
                "is_locked": is_locked,
                "recommended_time": time_str,
                "timer_instruction": timer_instruction,
                "slot_action": slot_action,
                "slot_action_reason": slot_action_reason,
                "has_senior": "资深干员" in selected,
                "has_top_senior": "高级资深干员" in selected,
                "strategy": strategy,
                "why_best": why_best,
            },
            "alternatives": [
                _build_alternative(c, best)
                for c in combos[1:6]
            ],
        })

    has_top = any(s.get("best_combo", {}).get("has_top_senior") for s in slot_results)
    has_senior = any(s.get("best_combo", {}).get("has_senior") for s in slot_results)

    # ── 为每个有效slot添加优先级标签，方便LLM引导用户 ──
    valid_slots = [s for s in slot_results if "best_combo" in s]
    if valid_slots:
        def _priority_key(s):
            b = s.get("best_combo", {})
            if not b:
                return (0, 0)
            return (b.get("guarantee_tier", 0), b.get("valuable_new_count", 0))

        sorted_slots = sorted(valid_slots, key=_priority_key, reverse=True)
        priority_tiers = ["★★★ 最优先", "★★ 高优先", "★ 优先", "可选", "可选"]
        for i, s in enumerate(sorted_slots):
            s["slot_priority_label"] = priority_tiers[min(i, len(priority_tiers) - 1)]
        top_slot = sorted_slots[0]["slot_index"] if sorted_slots else None
    else:
        top_slot = None

    return ToolOutput(text=json.dumps({
        "success": True,
        "strategy": strategy,
        "note": "2/3/4★已假定满潜；无高资时6★不可出；1★机器人稀有度等同5★新干员",
        "owned_count": len(owned) if owned else None,
        "slots": slot_results,
        "summary": {
            "has_top_operator_slot": has_top,
            "has_senior_operator_slot": has_senior,
            "total_slots": len(tags),
            "top_priority_slot": top_slot,
        },
    }, ensure_ascii=False, indent=2))


# ── Tool registration ─────────────────────────────────────────────────

registry.register(
    name="optimize_recruit_tags",
    description=(
        "★ [公招标签优选] 传入各栏位OCR识别的公招标签、返回最优标签组合与推荐时限。\n"
        "保底检测：高资->6★ | 资深->5★ | 标签锁定5★ | 标签锁定4★+ | 支援机械->1★\n"
        "两种策略：collection(图鉴优先) / yellow_cert(黄票优先)\n"
        "⚠️ **时限规则（必须遵守）：**\n"
        "  - slot_action='run' + recommended_time='9:00'\n"
        "    → **快速调时：点小时▼箭头，01下翻直接到09（环绕，1次搞定）**\n"
        "      9小时是保底生效的必要条件，标签自身不提供稀有度保证\n"
        "  - slot_action='run' + recommended_time='1:00' → 保持01（默认）\n"
        "  - slot_action='refresh'\n"
        "    → 这个slot不值得执行，无保底=浪费公招券。\n"
        "      看屏幕OCR里的「联络次数」，如果显示「联络次数3/3」「联络次数2/3」「联络次数1/3」→有剩余次数→点击「刷新标签」。\n"
        "      如果显示「人脉联络已达上限」或「联络次数0/3」→次数已用完→才用1:00执行。\n"
        "2/3/4★假定已满潜，仅1/5/6★视为有价值新干员\n"
        "示例：optimize_recruit_tags(tags=[[\"输出\",\"治疗\",\"资深干员\"], ...])"
    ),
    handler=optimize_recruit_tags,
    parameters={
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "string"}},
                "description": "每个栏位的标签列表（最多5个）。",
            },
            "strategy": {
                "type": "string",
                "enum": ["collection", "yellow_cert"],
                "description": "优化策略：collection=图鉴优先（默认）、yellow_cert=黄票优先",
            },
        },
        "required": ["tags"],
    },
    game="arknights",
)
