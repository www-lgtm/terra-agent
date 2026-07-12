"""Arknights base layout optimizer -- config enumeration + Pareto frontier + SA.

Given an operator box + a layout, enumerates ALL valid product configurations,
runs simulated-annealing assignment for each, computes the Pareto frontier across
multi-objective outputs (synthetic jade, LMD, combat records), and returns both
the frontier and the best schedule for any user-chosen weighting.

LLM-free solver engine -- pure Python, deterministic given a fixed random seed.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Layout definitions -- facility COUNTS only, no fixed products
# ═══════════════════════════════════════════════════════════════════

# Facility count per layout.  Products are decided by the optimizer.
LAYOUT_FACILITY_COUNTS: dict[str, dict[str, int]] = {
    "333": {"Trade": 3, "Mfg": 3, "Power": 3},
    "243": {"Trade": 2, "Mfg": 4, "Power": 3},
    "252": {"Trade": 2, "Mfg": 5, "Power": 2},
    "153": {"Trade": 1, "Mfg": 5, "Power": 3},
}

# Valid products per facility type
FACILITY_PRODUCTS: dict[str, list[str]] = {
    "Trade": ["LMD", "Orundum"],
    "Mfg":   ["PureGold", "CombatRecord", "OriginStone"],
    "Power": ["Drone"],
}

# Dorm counts by layout (community consensus). Used for dorm operator assignment
# and rest group scheduling. Typical 243/333 has 4 dorms; 252 has 3.
LAYOUT_DORM_COUNTS: dict[str, int] = {
    "333": 4, "243": 4, "252": 3, "153": 4,
}


def _get_fixed_facilities(layout: str) -> list[tuple[str, str, int]]:
    """Return fixed (non-product) facility rooms for a layout.

    Production rooms come from ProductConfig. Fixed rooms have no product choices
    and are assigned once (in the first shift). Dorm count varies by layout.
    """
    num_dorms = LAYOUT_DORM_COUNTS.get(layout, 4)
    rooms: list[tuple[str, str, int]] = [
        ("Control", "Control", 5),
        ("Office", "Office", 1),
        ("Reception", "Reception", 2),
    ]
    for i in range(num_dorms):
        rooms.append(("Dorm", "Rest", 5))
    return rooms

FACILITY_MAX_OPERS: dict[str, int] = {
    "Mfg": 3, "Trade": 3, "Power": 1,
    "Control": 5, "Office": 1, "Reception": 2, "Dorm": 5,
}

PRODUCT_ALIASES: dict[str, str] = {
    "LMD": "LMD", "龙门币": "LMD", "Money": "LMD",
    "PureGold": "PureGold", "赤金": "PureGold",
    "CombatRecord": "CombatRecord", "作战记录": "CombatRecord",
    "OriginStone": "OriginStone", "源石碎片": "OriginStone",
    "Orundum": "Orundum", "合成玉": "Orundum", "SyntheticJade": "Orundum",
    "Drone": "Drone", "无人机": "Drone",
}

# Elite unlock mapping: PRTS name -> canonical
ELITE_UNLOCK_MAP: dict[str, int] = {
    "精英0": 0, "精英1": 1, "精英2": 2,
    "E0": 0, "E1": 1, "E2": 2,
    "潜能1": 0, "潜能2": 1, "潜能3": 2,
}

# MAA efficiency key -> canonical optimizer key.
# MAA infrast.json uses different key names than what the optimizer expects.
# "Money" encodes order-count bonuses (e.g. 违约索赔) -> maps to LMD product.
EFFICIENCY_KEY_MAP: dict[str, str] = {
    "Money": "LMD",
    "SyntheticJade": "Orundum",  # MAA's base_skills.json key for orundum trade
    "PureGold_reg": "PureGold",  # 再生能源 -- template skill key
}

# Facilities where decimal encoding should be cleaned.
# MAA encodes modifiers (mood cost, capacity) in the fractional part:
#   30.1 -> 30% efficiency + mood-cost flag
#   20.1 -> 20% efficiency + mood-cost flag
# Only Trade / Mfg / Power have percentage-based efficiency; Control / Dorm /
# Office / Reception / Processing use raw fractional values (0.05 = 5%).
EFFICIENCY_CLEAN_FACILITIES: set[str] = {"Trade", "Mfg", "Power"}

# Keys from MAA's efficient dict that are NOT product efficiency and should
# be filtered out (debug comments, template references, non-product data).
EFFICIENCY_SKIP_KEYS: set[str] = {
    "Doc", "General", "No1", "No2", "No3", "No4", "No5", "No6", "No7",
}


def _normalize_skill_efficiency(
    raw: dict[str, float | str],
    facility: str,
) -> dict[str, float]:
    """Normalize MAA efficiency dict for the optimizer.

    1. Remap MAA key names -> canonical optimizer product keys (Money -> LMD)
    2. Drop non-numeric values (debug strings, template references)
    3. Strip MAA decimal modifier encoding for Trade / Mfg (30.1 -> 30)
    4. Drop irrelevant keys (Doc, General, No1-No7)
    """
    out: dict[str, float] = {}
    for key, val in raw.items():
        # Skip comment / template / non-product keys
        if key in EFFICIENCY_SKIP_KEYS:
            continue
        # Drop non-numeric values (strings like '[NumOfTrade] * 20')
        if not isinstance(val, (int, float)):
            continue
        # Remap MAA key -> canonical key
        canonical_key = EFFICIENCY_KEY_MAP.get(key, key)
        # Strip decimal modifier encoding for percentage-based facilities.
        # Values ≥ 1 are percentages with modifier flags (30.1 -> 30).
        # Values < 1 are raw fractions (Control 0.05 = 5%) -- keep as-is.
        if facility in EFFICIENCY_CLEAN_FACILITIES and val >= 1.0:
            val = float(int(val))
        # Accumulate (multiple MAA keys may map to same canonical key).
        out[canonical_key] = out.get(canonical_key, 0.0) + val
    return out


def _infer_elite_for_facility(
    kb_op_data: dict | None,
    facility: str,
) -> int:
    """Infer elite level required for curated fallback skills.

    Looks up the KB base_skills for this operator at the given facility
    and returns the MAX unlock level found. This is conservative: curated
    data typically represents the operator's full potential (all skills
    unlocked), so we require the highest elite level among the KB skills.

    Falls back to 0 when there is no KB data for this operator/facility.
    """
    if not kb_op_data:
        return 0
    base_skills = kb_op_data.get("base_skills", [])
    max_elite = 0
    for bs in base_skills:
        if bs.get("facility") == facility:
            unlock_raw = bs.get("unlock_condition", "精英0")
            elite = ELITE_UNLOCK_MAP.get(unlock_raw, 0)
            if elite > max_elite:
                max_elite = elite
    return max_elite


# ── Combo skill parsing ─────────────────────────────────────────────

# Specific pairing: "当与{partner}在同一个{facility}时，订单获取效率+{bonus}%"
_PAIRING_RE = re.compile(
    r'当与(.+?)在同一个(?:贸易站|制造站)时.*?订单获取效率\+(\d+)%',
)

# Per-other-operator: "除自身以外每名处于工作状态的干员(?:订单获取效率)?\+(\d+)%"
_PER_OTHER_RE = re.compile(
    r'除自身以外每名(?:处于工作状态的)?干员(?:订单获取效率)?\+(\d+)%',
)

# Faction-based: "每有1名(.+?)干员(?:，)?(?:订单获取效率)?\+(\d+)%"
_FACTION_RE = re.compile(
    r'每有1名(.+?)干员.*?(?:订单获取效率)?\+(\d+)%',
)

# Alternative per-member: "同个贸易站中每有1名(.+?)干员.*?\+(\d+)%"
_FACTION_RE2 = re.compile(
    r'每有1名(.+?)干员.*?当前贸易站订单获取效率\+(\d+)%',
)

# Warehouse patterns
# Capacity: "仓库容量上限+8" or "仓库容量+10"
_WAREHOUSE_CAP_RE = re.compile(r'仓库容量[上限]*\+(\d+)')
# Warehouse->efficiency conversion (红云): "每1格仓库容量提供2%生产力"
_WAREHOUSE_PER_SLOT_RE = re.compile(r'每.?格仓库容量.*?(\d+)%')

# Morale drain modifier: "心情每小时消耗+0.3" or "心情每小时消耗-0.25"
_MORALE_MOD_RE = re.compile(r'心情每小时消耗([+-]\d+\.?\d*)')


def parse_morale_modifier(desc: str) -> float:
    """Extract morale drain modifier from skill description.

    e.g. "心情每小时消耗-0.25" -> -0.25, "心情每小时消耗+0.3" -> 0.3
    Returns 0.0 if no morale modifier found.
    """
    if not desc:
        return 0.0
    m = _MORALE_MOD_RE.search(desc)
    if m:
        return float(m.group(1))
    return 0.0


# ── Dorm recovery parsing ──────────────────────────────────────────────

_DORM_RECOVERY_SELF_RE = re.compile(
    r'自身心情每小时恢复([+\-]?\d+\.?\d*)',
)
# Room-wide: "该宿舍内"/"宿舍内" + "所有"/"其他"/"除自身以外"
_DORM_RECOVERY_ALL_RE = re.compile(
    r'(?:该宿舍内|宿舍内).*?(?:所有|其他|除自身以外).*?心情每小时恢复([+\-]?\d+\.?\d*)',
)
# Single-target heal: "使...某个干员每小时恢复+X" (闪灵 type)
_DORM_RECOVERY_SINGLE_RE = re.compile(
    r'(?:使|让).*?(?:该宿舍内|宿舍内).*?(?:某个|一名).*?干员.*?每小时恢复([+\-]?\d+\.?\d*)',
)
_DORM_RECOVERY_COND_RE = re.compile(
    r'心情(\d+)以下.*?恢复\+(\d+\.?\d*)',
)


def parse_dorm_recovery(desc: str) -> tuple[float, float, float]:
    """Extract dorm recovery bonuses from a skill description.

    Returns (self_bonus, room_bonus, single_target_bonus).
    self_bonus: applied only to this operator (e.g. "自身心情每小时恢复+0.55")
    room_bonus: applied to ALL operators in the same dorm
    single_target_bonus: applied to one random operator in the dorm
      (e.g. 闪灵's "使某个干员每小时恢复+0.55").
      In aggregate across multiple dorm slots, treated as room-wide * 0.2.
    """
    if not desc:
        return (0.0, 0.0, 0.0)
    self_bonus = 0.0
    room_bonus = 0.0
    single_bonus = 0.0
    m = _DORM_RECOVERY_SELF_RE.search(desc)
    if m:
        self_bonus = float(m.group(1))
    m = _DORM_RECOVERY_ALL_RE.search(desc)
    if m:
        room_bonus = float(m.group(1))
    m = _DORM_RECOVERY_SINGLE_RE.search(desc)
    if m and not room_bonus:  # only use single if room-wide not already matched
        single_bonus = float(m.group(1))
    m = _DORM_RECOVERY_COND_RE.search(desc)
    if m:
        # Conditional — apply as partial bonus (half of conditional value)
        return (self_bonus, room_bonus, float(m.group(2)))
    # Single-target bonus distributed across dorm: effective ~0.2 * single * N_slots
    # But for simplicity, treat the best single-target as a modest room bonus.
    effective_room = room_bonus + single_bonus * 0.25
    return (self_bonus, effective_room, 0.0)


# ── Operator dorm recovery rate ────────────────────────────────────────

def _op_dorm_recovery_rate(op: 'Operator') -> float:
    """Compute an operator's base dorm recovery rate from their skills.

    Accounts for self-recovery skills (e.g. "自身心情每小时恢复+0.55")
    and room-wide auras. Fiammetta's special recovery of 2.0/h overrides
    all other sources (her skill says "无法获得其他来源提供的心情恢复效果").
    """
    base = 0.75
    self_bonus = 0.0
    for sk in op.skills:
        if sk.facility != "Dorm":
            continue
        if sk.elite_required > op.elite_level:
            continue
        if op.level > 0 and sk.level_required > op.level:
            continue
        # Fiammetta override: "自身心情每小时恢复+2，同时无法获得其他来源"
        # Note: base 0.75 is innate (not from "other sources"), so total = 2.75
        if sk.dorm_self_recovery >= 2.0:
            return base + sk.dorm_self_recovery
        self_bonus += sk.dorm_self_recovery
    return base + self_bonus


# ── Morale-driven scheduling algorithms ───────────────────────────────

def _make_rest_group(
    gid: int, facility: str, product: str, rooms: list['Room'],
    group_ops: list['Operator'],
) -> RestGroup:
    """Build a single RestGroup from a pre-computed (facility, product) room group."""
    work_morale_drain = max(
        op.morale_drain_per_hour(facility) for op in group_ops
    )
    work_efficiency = sum(
        op.efficiency_for(product, facility) for op in group_ops
    )
    return RestGroup(
        group_id=gid,
        facility=facility,
        room_indices=[r.index for r in rooms],
        operators=group_ops,
        product=product,
        work_morale_drain=work_morale_drain,
        work_efficiency=work_efficiency,
    )


def _build_rest_groups(
    config: 'ProductConfig',
    operators: list['Operator'],
    ready_teams: list['TeamInstance'],
) -> list[RestGroup]:
    """Group operators by (facility, product) into synchronized work/rest units.

    Each RestGroup corresponds to one set of rooms producing the same product.
    Operators in the same group work and rest together, preserving combo synergies.
    """
    # Group rooms by (facility, product)
    rooms_by_fp: dict[tuple[str, str], list[Room]] = {}
    for room in config.to_rooms():
        key = (room.facility, room.product)
        rooms_by_fp.setdefault(key, []).append(room)

    groups: list[RestGroup] = []
    gid = 0
    used_names: set[str] = set()  # operators already assigned to a group

    for (facility, product), rooms in rooms_by_fp.items():
        total_slots = sum(r.max_slots for r in rooms)
        room_indices = [r.index for r in rooms]

        # Collect operators already placed by team warm-start (skip used)
        team_ops: list[Operator] = []
        team_names: set[str] = set()
        if ready_teams:
            for team in ready_teams:
                if team.template.facility != facility:
                    continue
                if team.template.product_hint and team.template.product_hint != product:
                    continue
                for op in team.operators:
                    if op.name not in team_names and op.name not in used_names:
                        team_ops.append(op)
                        team_names.add(op.name)

        # Fill remaining slots with best-fit operators (skip used)
        available = [op for op in operators
                    if op.name not in team_names
                    and op.name not in used_names
                    and op.efficiency_for(product, facility) > 0]
        available.sort(key=lambda o: -o.efficiency_for(product, facility))

        group_ops = team_ops[:total_slots]
        remaining = total_slots - len(group_ops)
        if remaining > 0:
            group_ops.extend(available[:remaining])

        if not group_ops:
            continue

        # Mark as used
        for op in group_ops:
            used_names.add(op.name)

        groups.append(_make_rest_group(gid, facility, product, rooms, group_ops))
        gid += 1

    return groups


def _build_rest_groups_from_shifts(
    shifts: list['Shift'],
    operators: list['Operator'],
) -> list[RestGroup]:
    """Build RestGroups from SA's actual room assignments in the first shift.

    Only production facilities (Trade/Mfg/Power) form rest groups — fixed
    facilities (Control/Office/Reception/Dorm) run 24/7 and don't rotate.
    """
    if not shifts:
        return []
    op_map = {op.name: op for op in operators}
    shift = shifts[0]  # primary (A队) shift determines the work groups
    _PROD_FACS = {"Trade", "Mfg", "Power"}

    # Group rooms by (facility, product)
    rooms_by_fp: dict[tuple[str, str], list['Room']] = {}
    for room in shift.rooms:
        if room.facility not in _PROD_FACS:
            continue
        key = (room.facility, room.product)
        rooms_by_fp.setdefault(key, []).append(room)

    groups: list[RestGroup] = []
    gid = 0
    for (facility, product), rooms in rooms_by_fp.items():
        group_ops: list[Operator] = []
        seen: set[str] = set()
        for room in rooms:
            for op in room.operators:
                if op.name not in seen:
                    group_ops.append(op)
                    seen.add(op.name)

        if not group_ops:
            continue

        groups.append(_make_rest_group(gid, facility, product, rooms, group_ops))
        gid += 1

    return groups


def _compute_work_duration(group: RestGroup, mood_threshold: float = 0.5) -> float:
    """Hours the group can work before any member hits the morale threshold.

    Returns hours until the fastest-draining member reaches mood_threshold * 24.

    Formula: starting morale = 24, morale_threshold = fraction to trigger rest.
    Work hours until morale drops to threshold:
        hours = 24 * (1.0 - threshold) / drain_rate

    Example: drain=0.75/h, threshold=0.35 → 24*0.65/0.75 = 20.8h
             drain=0.75/h, threshold=0.50 → 24*0.50/0.75 = 16.0h
    """
    max_work = 999.0
    for op in group.operators:
        drain = op.morale_drain_per_hour(group.facility)
        if drain <= 0:
            continue
        hours = 24.0 * (1.0 - mood_threshold) / drain
        if hours < max_work:
            max_work = hours
    return min(max_work, 999.0)


def _compute_rest_duration(
    group: RestGroup,
    dorm_index: int,
    all_operators: dict[str, 'Operator'],
    dorm_snapshots: list[DormSnapshot],
    work_duration: float,
) -> float:
    """Hours the group needs to rest to fully recover morale.

    Accounts for dorm skills and room-wide recovery auras in the assigned dorm.
    """
    if dorm_index < 0 or dorm_index >= len(dorm_snapshots):
        return 0.0
    snap = dorm_snapshots[dorm_index]

    # Morale after working
    max_rest_needed = 0.0
    for op in group.operators:
        morale_after = 24.0 - op.morale_drain_per_hour(group.facility) * work_duration
        morale_after = max(0.0, morale_after)
        deficit = 24.0 - morale_after

        # Recovery rate: base + self bonus + room aura
        recovery = _op_dorm_recovery_rate(op)
        # Add room-wide auras from OTHER operators in the same dorm
        for other_name in snap.operators:
            if other_name == op.name:
                continue  # skip self — room auras only apply to others
            other = all_operators.get(other_name)
            if other:
                for sk in other.skills:
                    if sk.facility == "Dorm" and sk.dorm_room_recovery > 0:
                        if sk.elite_required <= other.elite_level:
                            recovery += sk.dorm_room_recovery

        hours = deficit / max(recovery, 0.1)
        if hours > max_rest_needed:
            max_rest_needed = hours
    return max_rest_needed


def _assign_dorms(
    resting_group: RestGroup,
    other_resting_ops: list[str],
    all_operators: dict[str, 'Operator'],
    num_dorms: int,
    dorm_capacity: int = 5,
    fiammetta_name: str | None = None,
) -> list[DormSnapshot]:
    """Assign resting operators to dorms by priority.

    Lower-morale operators go to higher-priority dorms (lower index).
    Fiammetta gets a dedicated dorm slot (her recovery overrides all auras).
    Returns dorm snapshots with per-operator recovery rates.
    """
    snapshots: list[DormSnapshot] = []
    for i in range(num_dorms):
        snapshots.append(DormSnapshot(dorm_index=i))

    # Collect all resting operators with their dorm recovery rates
    resting: list[tuple[str, float]] = []
    for op_name in other_resting_ops:
        op = all_operators.get(op_name)
        if op:
            rate = _op_dorm_recovery_rate(op)
            resting.append((op_name, rate))
    # Add the resting group members
    for op in resting_group.operators:
        rate = _op_dorm_recovery_rate(op)
        resting.append((op.name, rate))

    # Sort: operators with lowest recovery first (recovery potential is scored
    # by recovery rate * hours — lower rate needs more time)
    resting.sort(key=lambda x: x[1])  # ascending recovery rate

    # Assign to dorms
    dorm_idx = 0
    for op_name, _ in resting:
        if fiammetta_name and op_name == fiammetta_name:
            # Fiammetta goes to her own slot (dorm 0 if available)
            snap = snapshots[0]
            snap.operators.append(op_name)
            snap.per_op_recovery[op_name] = 2.0
            continue
        # Find a dorm with space
        assigned = False
        for attempt in range(num_dorms):
            di = (dorm_idx + attempt) % num_dorms
            if len(snapshots[di].operators) < dorm_capacity:
                snapshots[di].operators.append(op_name)
                op = all_operators.get(op_name)
                if op:
                    snapshots[di].per_op_recovery[op_name] = _op_dorm_recovery_rate(op)
                assigned = True
                break
        if not assigned:
            # All dorms full — append to last dorm anyway (overflow)
            snapshots[-1].operators.append(op_name)
            op = all_operators.get(op_name)
            if op:
                snapshots[-1].per_op_recovery[op_name] = _op_dorm_recovery_rate(op)
        dorm_idx = (dorm_idx + 1) % num_dorms

    # Apply room-wide auras (add dorm_room_recovery to all ops in same dorm)
    for snap in snapshots:
        for name in snap.operators:
            op = all_operators.get(name)
            if not op:
                continue
            for sk in op.skills:
                if sk.facility == "Dorm" and sk.dorm_room_recovery > 0:
                    if sk.elite_required <= op.elite_level:
                        for other in snap.operators:
                            if other != name:
                                old = snap.per_op_recovery.get(other, 0.75)
                                snap.per_op_recovery[other] = old + sk.dorm_room_recovery

    return snapshots


def _compute_fia_charges(
    groups: list[RestGroup],
    all_operators: dict[str, 'Operator'],
    fia_name: str,
    fia_threshold: float,
    min_interval: float = 1.2,
    fia_morale_cost: float = 0.65,
    total_duration: float = 24.0,
) -> list[FiaChargeTarget]:
    """Compute optimal Fiammetta charging schedule.

    Algorithm:
    1. Identify chargeable operators (working, below threshold, not recently
       swapped, lowest morale in their group).
    2. Compute Fia coefficient = efficiency / morale_cost for each.
    3. Sort by coefficient descending, assign charges until Fia exhausted.
    4. Enforce minimum interval between charges.
    """
    fia_op = all_operators.get(fia_name)
    if not fia_op:
        return []

    # Collect charge candidates
    candidates: list[tuple[str, float, int, float]] = []  # (name, coeff, gid, morale)
    for group in groups:
        # Find lowest-morale operator in this group (after work_duration)
        lowest_morale = 24.0
        lowest_name = ""
        for op in group.operators:
            morale_after_work = 24.0 - op.morale_drain_per_hour(
                group.facility) * group.work_duration
            morale_after_work = max(0.0, morale_after_work)
            if morale_after_work < lowest_morale:
                lowest_morale = morale_after_work
                lowest_name = op.name

        if (lowest_name and
                lowest_morale < fia_threshold * 24.0 and
                group.work_duration > 1.0):  # not just returned
            eff = next(
                (op.efficiency_for(group.product, group.facility)
                 for op in group.operators if op.name == lowest_name),
                0.0,
            )
            coeff = eff / max(fia_morale_cost, 0.01)
            candidates.append((lowest_name, coeff, group.group_id, lowest_morale))

    # Sort by coefficient descending
    candidates.sort(key=lambda x: -x[1])

    # Assign charges with realistic timing
    charges: list[FiaChargeTarget] = []
    fia_remaining = 24.0  # Fiammetta's morale
    last_charge_time = -min_interval

    for name, coeff, gid, morale in candidates:
        if fia_remaining < fia_morale_cost:
            break
        # Find this operator's group to get the optimal charge window
        op_group = next((g for g in groups if g.group_id == gid), None)
        if not op_group:
            continue
        # Charge should happen when morale is low but operator can still benefit.
        # Target: 30-50% of work duration, when morale has dropped enough.
        charge_window_start = op_group.work_duration * 0.3
        charge_time = max(charge_window_start, last_charge_time + min_interval)
        if charge_time >= min(total_duration, op_group.work_duration * 0.9):
            continue
        # Throttle based on morale at CHARGE TIME, not at end of work
        morale_at_charge = 24.0 - (
            op.morale_drain_per_hour(op_group.facility) * charge_time
        ) if (op := next((o for o in op_group.operators if o.name == name), None)) else morale
        morale_at_charge = max(0.0, morale_at_charge)
        if morale_at_charge < 0.3 * 24.0:
            throttle = 1
        elif morale_at_charge < 0.5 * 24.0:
            throttle = 2
        else:
            throttle = 3
        charges.append(FiaChargeTarget(
            operator_name=name,
            coefficient=round(coeff, 2),
            charge_time_h=round(charge_time, 2),
            parent_group_id=gid,
            throttle=throttle,
        ))
        fia_remaining -= fia_morale_cost / throttle
        last_charge_time = charge_time

    return charges


def _get_operator_dorm_recovery(op: 'Operator') -> float:
    """Get an operator's dorm recovery rate (per hour).

    Used by solve_pareto to compute group rest durations.
    """
    return _op_dorm_recovery_rate(op)


def parse_warehouse_capacity(desc: str) -> int:
    """Extract warehouse capacity bonus from a skill description.

    e.g. "仓库容量上限+8" -> 8, "仓库容量上限+10" -> 10
    """
    if not desc:
        return 0
    m = _WAREHOUSE_CAP_RE.search(desc)
    if m:
        return int(m.group(1))
    return 0


def parse_combo_from_desc(desc: str) -> ComboDescriptor | None:
    """Parse a skill description for combo/synergy effects.

    Recognises five patterns:
      A. "当与{partner}在同一贸易站时，订单获取效率+X%" -> specific pairing
      B. "除自身以外每名工作干员+X%" -> per-other-operator bonus
      C. "每有1名{faction}干员+X%" -> faction-based bonus
      D. "每格仓库容量提供X%生产力" -> warehouse conversion (红云)
      E. Threshold warehouse: two different rates (泡泡)

    Returns None if no combo effect is recognised.
    """
    if not desc:
        return None

    # Pattern A: specific operator pairing
    m = _PAIRING_RE.search(desc)
    if m:
        partner = m.group(1).strip()
        bonus = float(m.group(2))
        return ComboDescriptor(
            partner=partner,
            partner_bonus={"all": bonus},
        )

    # Pattern B: per-other-operator in same room
    m = _PER_OTHER_RE.search(desc)
    if m:
        bonus = float(m.group(1))
        return ComboDescriptor(per_other_op=bonus)

    # Pattern C: faction-based
    for regex in (_FACTION_RE2, _FACTION_RE):
        m = regex.search(desc)
        if m:
            faction = m.group(1).strip()
            bonus = float(m.group(2))
            return ComboDescriptor(faction=faction, faction_bonus=bonus)

    # Pattern E: threshold-based warehouse conversion (泡泡 -- check first)
    # "16格以下的，每格提供1%生产力；大于16格的，每格提供3%生产力"
    low_match = re.search(r'(\d+)格以[下内].*?每.?格.*?(\d+)%', desc)
    high_match = re.search(r'[大高多超]于\s*(\d+)格.*?每.?格.*?(\d+)%', desc)
    if low_match and high_match:
        threshold = int(low_match.group(1))
        low_bonus = float(low_match.group(2))
        high_bonus = float(high_match.group(2))
        return ComboDescriptor(
            warehouse_threshold=threshold,
            warehouse_low_bonus=low_bonus,
            warehouse_high_bonus=high_bonus,
        )

    # Pattern D: simple warehouse per-slot conversion (红云)
    m = _WAREHOUSE_PER_SLOT_RE.search(desc)
    if m:
        bonus = float(m.group(1))
        return ComboDescriptor(warehouse_per_slot=bonus)

    # Pattern F/G/H: special trade mechanics -- accumulate flags
    trade_breach = False
    trade_fixed = False
    trade_gap = 0.0
    trade_cap = 0

    if "视为违约" in desc:
        trade_breach = True
    if "固定获取" in desc and "不受任何" in desc:
        trade_fixed = True

    gap_match = re.search(r'每差.?笔订单.*?订单.*?\+(\d+)%', desc)
    if gap_match:
        trade_gap = float(gap_match.group(1))

    cap_match = re.search(r'订单上限([+-]\d)', desc)
    if cap_match:
        trade_cap = int(cap_match.group(1))

    if trade_breach or trade_fixed or trade_gap > 0 or trade_cap != 0:
        return ComboDescriptor(
            trade_breach=trade_breach,
            trade_fixed_order=trade_fixed,
            trade_gap_bonus=trade_gap,
            trade_order_cap_modifier=trade_cap,
        )

    return None

def compute_room_combo_bonus(room: 'Room') -> dict[str, float]:
    """Calculate additional efficiency from combo effects in a room.

    Returns {product_key: bonus_efficiency} from all combo interactions.
    Does NOT double-count base skills -- only returns the EXTRA efficiency
    from synergy effects (pairings, per-other-op, faction, warehouse).
    """
    bonus: dict[str, float] = {}
    op_names = {op.name for op in room.operators}
    op_factions = {op.name: op.faction for op in room.operators}

    # Pre-compute total warehouse capacity (all operators, before combo loop)
    warehouse_total: int = sum(
        s.warehouse_capacity
        for op in room.operators
        for s in op.skills
        if s.elite_required <= op.elite_level and s.facility == room.facility
    )

    # Track the best warehouse conversion -- Red Cloud and Bubble
    # are mutually exclusive (only the best conversion applies).
    best_warehouse_bonus: float = 0.0

    for op in room.operators:
        for combo in op.active_combos(room.facility):
            # Type A: specific pairing
            if combo.partner and combo.partner in op_names:
                for product, val in combo.partner_bonus.items():
                    bonus[product] = bonus.get(product, 0.0) + val

            # Type B: per-other-operator
            if combo.per_other_op > 0:
                other_count = len(room.operators) - 1
                if other_count > 0:
                    bonus["all"] = bonus.get("all", 0.0) + combo.per_other_op * other_count

            # Type C: faction-based
            if combo.faction and combo.faction_bonus > 0:
                faction_count = sum(
                    1 for n, f in op_factions.items()
                    if f and combo.faction in f
                )
                if faction_count > 0:
                    bonus["all"] = bonus.get("all", 0.0) + combo.faction_bonus * faction_count

            # Type D: warehouse per-slot (红云)
            if combo.warehouse_per_slot > 0 and warehouse_total > 0:
                w_bonus = combo.warehouse_per_slot * warehouse_total
                if w_bonus > best_warehouse_bonus:
                    best_warehouse_bonus = w_bonus

            # Type E: threshold warehouse (泡泡)
            if combo.warehouse_threshold > 0 and warehouse_total > 0:
                if warehouse_total < combo.warehouse_threshold:
                    w_bonus = combo.warehouse_low_bonus * warehouse_total
                else:
                    w_bonus = combo.warehouse_high_bonus * warehouse_total
                if w_bonus > best_warehouse_bonus:
                    best_warehouse_bonus = w_bonus

    # Apply the best warehouse conversion (mutual exclusion resolved)
    if best_warehouse_bonus > 0:
        bonus["all"] = bonus.get("all", 0.0) + best_warehouse_bonus

    return bonus


# ═══════════════════════════════════════════════════════════════════
# Shared box parser -- used by both scheduler and optimizer
# ═══════════════════════════════════════════════════════════════════

def parse_operator_box(text: str) -> dict[str, int | dict]:
    """Parse operator list text -> {name: elite_level | {elite, level}}.

    Supports: "德克萨斯(E2)、清流(精英1)、能天使(E1)、斑点"
    Supports: "Castle-3 LV30(E0)" for level-qualified operators
    Operators without an explicit elite tag default to E0 (no unlocked skills
    beyond base) so the solver won't produce schedules that fail in-game.
    Returns {name: elite_int} for backward compat, or {name: dict} with level.
    """
    result: dict[str, int | dict] = {}
    tokens = re.split(r'[、,，\s]+', text.strip())
    for token in tokens:
        token = token.strip()
        if not token or len(token) < 2:
            continue
        elite = 0  # default E0 -- conservative: only count base skills
        level = 0  # 0 = unknown
        # Extract LV marker before parenthesized elite: "Castle-3 LV30(E0)"
        lv_match = re.search(r'\bLV\s*(\d+)\b', token, re.IGNORECASE)
        if lv_match:
            level = int(lv_match.group(1))
            token = re.sub(r'\bLV\s*\d+\b\s*', '', token, flags=re.IGNORECASE).strip()
        match = re.match(r'^(.+?)\s*[（(]\s*(?:精英)?\s*([Ee]?\d)\s*[）)]$', token)
        if match:
            name = match.group(1).strip()
            elite_str = match.group(2).strip().upper()
            if elite_str.startswith("E"):
                elite = min(2, max(0, int(elite_str[1])))
            else:
                elite = min(2, max(0, int(elite_str)))
            token = name
        if level > 0:
            result[token] = {"elite": elite, "level": level}
        else:
            result[token] = elite
    return result


# ═══════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ComboDescriptor:
    """Pre-parsed combo/synergy effect for a single skill."""
    # Type A: specific operator pairing
    #   e.g. Texas + Lappland in same Trade -> Texas +65%
    partner: str = ""
    partner_bonus: dict[str, float] = field(default_factory=dict)

    # Type B: per-other-operator in same room
    #   e.g. 火哨 "每名工作干员+15%" -- bonus × (room_size - 1)
    per_other_op: float = 0.0

    # Type C: faction-based
    #   e.g. 摩根 "每有1名格拉斯哥帮干员+20%"
    faction: str = ""
    faction_bonus: float = 0.0

    # Type D: warehouse capacity -> efficiency conversion
    #   e.g. 红云 "每格仓库容量提供2%生产力"
    warehouse_per_slot: float = 0.0

    # Type E: threshold-based warehouse conversion (泡泡)
    #   "容量<16格: 1%/格, ≥16格: 3%/格" -- incompatible with Type D
    warehouse_threshold: int = 0
    warehouse_low_bonus: float = 0.0
    warehouse_high_bonus: float = 0.0

    # Type F: special trade mechanics (但书/佩佩/孑)
    trade_breach: bool = False
    trade_fixed_order: bool = False
    trade_gap_bonus: float = 0.0
    trade_order_cap_modifier: int = 0

    @property
    def is_active(self) -> bool:
        return bool(
            self.partner or self.per_other_op > 0 or self.faction
            or self.warehouse_per_slot > 0 or self.warehouse_threshold > 0
            or self.trade_breach or self.trade_fixed_order
            or self.trade_gap_bonus > 0 or self.trade_order_cap_modifier != 0
        )


@dataclass
class OperatorSkill:
    skill_id: str
    facility: str
    name: str
    efficiency: dict[str, float] = field(default_factory=dict)
    elite_required: int = 0          # 0 = base, 1 = E1, 2 = E2
    level_required: int = 1          # minimum operator level (1 = always unlocked)
    combo: ComboDescriptor | None = None  # parsed combo effect
    warehouse_capacity: int = 0      # +N warehouse capacity provided
    morale_mod: float = 0.0          # modifier to base 0.75 morale drain per hour
    dorm_self_recovery: float = 0.0  # self recovery bonus in dorm (per hour)
    dorm_room_recovery: float = 0.0  # room-wide recovery bonus (per hour)


@dataclass
class Operator:
    name: str
    elite_level: int = 2             # User's actual elite level for this operator
    level: int = 0                   # Operator level (0 = unknown, check is skipped)
    rarity: int = 1
    faction: str = ""
    skills: list[OperatorSkill] = field(default_factory=list)

    def _skill_unlocked(self, skill: OperatorSkill) -> bool:
        """Check both elite and level requirements for skill unlock."""
        if skill.elite_required > self.elite_level:
            return False
        if self.level > 0 and skill.level_required > self.level:
            return False  # known level too low
        return True

    def __post_init__(self):
        """Initialize per-operator caches (skills are immutable after _resolve_operators)."""
        object.__setattr__(self, '_eff_cache', {})

    def efficiency_for(self, product: str, facility: str) -> float:
        """Efficiency for a product, only counting unlocked skills.

        Results are cached since operator skills don't change during a single
        solve_pareto() call — this is the hottest path in the SA inner loop.
        """
        key = (product, facility)
        cached = self._eff_cache.get(key)
        if cached is not None:
            return cached
        total = 0.0
        for skill in self.skills:
            if not self._skill_unlocked(skill):
                continue
            if skill.facility != facility:
                continue
            if product in skill.efficiency:
                total += skill.efficiency[product]
            elif "all" in skill.efficiency:
                total += skill.efficiency["all"]
        self._eff_cache[key] = total
        return total

    def active_combos(self, facility: str) -> list[ComboDescriptor]:
        """Return combo descriptors for unlocked skills."""
        combos: list[ComboDescriptor] = []
        for skill in self.skills:
            if not self._skill_unlocked(skill):
                continue
            if skill.facility != facility:
                continue
            if skill.combo and skill.combo.is_active:
                combos.append(skill.combo)
        return combos

    def morale_drain_per_hour(self, facility: str) -> float:
        """Morale drain rate at a given facility, including skill modifiers."""
        base = 0.75
        total_mod = 0.0
        for skill in self.skills:
            if not self._skill_unlocked(skill):
                continue
            if skill.facility == facility:
                total_mod += skill.morale_mod
        return max(0.1, base + total_mod)

    def max_work_hours(self, facility: str, morale_cap: float = 24.0) -> float:
        """Maximum continuous work hours before morale exhaustion."""
        return morale_cap / self.morale_drain_per_hour(facility)


@dataclass
class Room:
    facility: str
    product: str
    index: int
    max_slots: int
    operators: list[Operator] = field(default_factory=list)
    trade_post_level: int = 3   # 1/2/3, only relevant for Trade rooms
    rest_duration: float = 0.0   # hours to rest in dorm (morale-driven mode)
    dorm_index: int = -1         # which dorm this room's ops rest in (-1 = not assigned)

    def total_efficiency(self, include_combos: bool = True) -> float:
        base = sum(op.efficiency_for(self.product, self.facility) for op in self.operators)
        if include_combos:
            combo_bonus = compute_room_combo_bonus(self)
            for product_key, bonus in combo_bonus.items():
                if product_key == "all" or product_key == self.product:
                    base += bonus
        # Special trade mechanics: apply AFTER combo bonuses
        if self.facility == "Trade":
            base = self._apply_special_trade_mechanics(base)
        return base

    def _apply_special_trade_mechanics(self, base_eff: float) -> float:
        """Apply special trade skill effects.

        佩佩: fixed order output, ignores all efficiency at L1.
        孑: +X% per gap between current orders and order cap.
        但书: breach flag is metadata — efficiency already captured by normal skills.
        """
        has_fixed = False
        gap_bonus_total = 0.0
        order_cap_delta = 0

        for op in self.operators:
            for combo in op.active_combos(self.facility):
                if combo.trade_fixed_order:
                    has_fixed = True
                if combo.trade_gap_bonus > 0:
                    gap_bonus_total += combo.trade_gap_bonus
                order_cap_delta += combo.trade_order_cap_modifier

        if has_fixed and self.trade_post_level <= 1:
            return 0.0

        if gap_bonus_total > 0:
            base_cap = 10 + order_cap_delta
            active_orders = max(1, min(7, 4))
            gap = max(0, base_cap - active_orders)
            base_eff += gap_bonus_total * gap

        return base_eff

    @property
    def warnings(self) -> list[str]:
        """Warn if any assigned operator lacks elite level for assigned skills."""
        w: list[str] = []
        for op in self.operators:
            for sk in op.skills:
                if sk.facility == self.facility and sk.elite_required > op.elite_level:
                    w.append(f"{op.name}需E{sk.elite_required}才解锁「{sk.name}」(当前E{op.elite_level})")
        return w


@dataclass
class Shift:
    name: str
    duration_hours: float
    rooms: list[Room] = field(default_factory=list)

    def used_operators(self) -> set[str]:
        return {op.name for room in self.rooms for op in room.operators}


@dataclass
class ProductConfig:
    """A specific assignment of products to rooms (no operators yet)."""
    layout: str
    rooms: list[tuple[str, str]]   # [(facility, product), ...]

    def to_rooms(self) -> list[Room]:
        counters: dict[str, int] = defaultdict(int)
        out: list[Room] = []
        for facility, product in self.rooms:
            idx = counters[facility]
            counters[facility] += 1
            out.append(Room(
                facility=facility,
                product=product,
                index=idx,
                max_slots=FACILITY_MAX_OPERS.get(facility, 3),
            ))
        return out

    @property
    def output_vector(self) -> tuple[float, float, float]:
        """Return (orundum_share, lmd_share, combat_record_share) as room counts."""
        orundum = sum(1 for f, p in self.rooms if p == "Orundum")
        lmd = sum(1 for f, p in self.rooms if p == "LMD") + sum(1 for f, p in self.rooms if p == "PureGold")
        cr = sum(1 for f, p in self.rooms if p == "CombatRecord")
        return (orundum, lmd, cr)


@dataclass
class Inventory:
    """Player's current resource stockpile (from in-game storage)."""
    puregold: int = 0       # 赤金 count
    origin_stone: int = 0   # 源石碎片 count
    lmd: int = 0            # 龙门币
    orundum: int = 0        # 合成玉

    def is_empty(self) -> bool:
        return self.puregold == 0 and self.origin_stone == 0


_INV_PUREGOLD_RE = re.compile(r'(?:赤金|金条|金塊)\s*[:：]?\s*(\d+)\s*(?:个|枚|块)?')
_INV_ORIGIN_STONE_RE = re.compile(r'源石碎片\s*[:：]?\s*(\d+)\s*(?:个|枚|片)?')
_INV_LMD_RE = re.compile(r'(?:龙门币|钱)\s*[:：]?\s*(\d+)\s*(?:万|w|W)?')
_INV_ORUNDUM_RE = re.compile(r'(?:合成玉|玉)\s*[:：]?\s*(\d+)\s*(?:个|颗)?')


def parse_inventory(text: str) -> Inventory:
    """Extract resource stockpile quantities from user text."""
    inv = Inventory()
    m = _INV_PUREGOLD_RE.search(text)
    if m: inv.puregold = int(m.group(1))
    m = _INV_ORIGIN_STONE_RE.search(text)
    if m: inv.origin_stone = int(m.group(1))
    m = _INV_LMD_RE.search(text)
    if m:
        val = int(m.group(1))
        inv.lmd = val * 10000 if any(x in m.group(0) for x in ('万','w','W')) else val
    m = _INV_ORUNDUM_RE.search(text)
    if m: inv.orundum = int(m.group(1))
    return inv



@dataclass
class RestGroup:
    """A coordinated work/rest unit — operators that swap together (整组).

    When any member's morale drops below the threshold, the entire group
    moves to the dorm. When the last member recovers, the group returns.
    This preserves combo synergies (e.g. Texas+Lapland) that would break
    if members rotated individually.
    """
    group_id: int
    facility: str                  # "Trade" | "Mfg"
    room_indices: list[int]        # which rooms this group fills
    operators: list['Operator']     # member operators
    product: str                   # "LMD" | "Orundum" | "PureGold" | ...
    work_morale_drain: float = 0.0  # MAX morale drain per hour across members
    work_efficiency: float = 0.0    # total efficiency (sum of all members)
    work_duration: float = 0.0      # computed hours before rest needed
    rest_duration: float = 0.0      # computed hours to recover in dorm
    rest_dorm_index: int = 0        # assigned dorm (0 = highest priority)


@dataclass
class DormSnapshot:
    """Snapshot of one dorm's occupancy and recovery rates."""
    dorm_index: int
    operators: list[str] = field(default_factory=list)
    base_recovery: float = 0.75
    per_op_recovery: dict[str, float] = field(default_factory=dict)


@dataclass
class FiaChargeTarget:
    """One operator that Fiammetta should charge."""
    operator_name: str
    coefficient: float             # production_gain / morale_cost
    charge_time_h: float           # relative hour in schedule
    parent_group_id: int
    throttle: int = 1              # 1=full, 2=half, 3=light


@dataclass
class ParetoSolution:
    """One point on the Pareto frontier -- a complete schedule + its output metrics."""
    config: ProductConfig
    shifts: list[Shift]
    # Output metrics (normalized 0..1 relative to theoretical max)
    orundum_eff: float = 0.0
    lmd_eff: float = 0.0
    combat_record_eff: float = 0.0
    total_score: float = 0.0
    coverage: float = 0.0
    dominated_by: int = 0           # How many other solutions dominate this one
    # Morale-driven mode
    schedule_mode: str = "fixed_shift"  # "fixed_shift" | "morale_driven"
    rest_groups: list[RestGroup] = field(default_factory=list)
    dorm_snapshots: list[DormSnapshot] = field(default_factory=list)
    fia_charges: list[FiaChargeTarget] = field(default_factory=list)
    work_to_rest_ratio: float = 0.0
    operator_names: list[str] = field(default_factory=list)  # for dorm assignment
    box_data: dict = field(default_factory=dict)             # for level/elite checks
    # ── Concrete daily output estimates (not normalized percentages) ──
    daily_orundum: float = 0.0        # 合成玉/天
    daily_lmd: float = 0.0            # 龙门币/天
    daily_combat_record: float = 0.0  # 作战记录/天
    daily_puregold_net: float = 0.0   # 赤金净产出/天 (产出-消耗)
    # ── Closed-loop sustainability analysis ──
    sustain_lmd_balance: float = 0.0     # LMD净收支/天 (收入 - 搓石成本)
    sustain_rock_demand: float = 0.0     # 固源岩需求/天
    sustain_sanity_cost: float = 0.0     # 1-7刷石理智成本/天
    sustain_verdict: str = "ok"          # "ok" | "lmd_deficit" | "sanity_impossible" | "both"
    sustain_detail: str = ""             # Human-readable explanation

    def dominates(self, other: "ParetoSolution") -> bool:
        """True if self is strictly better on all non-zero dimensions."""
        better = False
        for a, b in [(self.orundum_eff, other.orundum_eff),
                     (self.lmd_eff, other.lmd_eff),
                     (self.combat_record_eff, other.combat_record_eff)]:
            if a < b:
                return False
            if a > b:
                better = True
        return better


# ═══════════════════════════════════════════════════════════════════
# Product config enumerator
# ═══════════════════════════════════════════════════════════════════

def enumerate_product_configs(layout: str) -> list[ProductConfig]:
    """Enumerate all valid product assignments for a layout.

    A 243 layout has Trade{2} × Mfg{4}:
      Trade: 2 rooms × 2 products (LMD, Orundum) = 2² = 4
      Mfg:   4 rooms × 3 products (Gold, CR, Stone) = 3⁴ = 81
      Total: 4 × 81 = 324 configurations
    """
    counts = LAYOUT_FACILITY_COUNTS.get(layout)
    if not counts:
        return []

    # Generate product lists per facility
    product_lists: dict[str, list[list[str]]] = {}
    for facility, room_count in counts.items():
        products = FACILITY_PRODUCTS.get(facility, [facility])
        # All combinations with repetition: cartesian product of [products] × room_count
        if room_count == 0:
            continue
        # Generate all product assignments for this facility's rooms
        assignments = list(itertools.product(products, repeat=room_count))
        product_lists[facility] = [list(a) for a in assignments]

    # Cross-product across facilities
    configs: list[ProductConfig] = []
    facility_order = sorted(product_lists.keys())
    iterables = [product_lists[f] for f in facility_order]
    for combo in itertools.product(*iterables):
        rooms: list[tuple[str, str]] = []
        for fi, facility in enumerate(facility_order):
            for product in combo[fi]:
                rooms.append((facility, product))
        configs.append(ProductConfig(layout=layout, rooms=rooms))

    return configs


# ═══════════════════════════════════════════════════════════════════
# Simulated Annealing
# ═══════════════════════════════════════════════════════════════════

def _sa_temperature(step: int, total: int, T0: float = 2.0) -> float:
    """Exponential cooling schedule — slightly faster cooling for quicker convergence."""
    return T0 * (0.005 / T0) ** (step / total)


def _sa_accept_prob(delta: float, T: float) -> float:
    """Metropolis criterion."""
    if delta >= 0:
        return 1.0
    return math.exp(delta / max(T, 1e-6))


def simulated_annealing_assignment(
    rooms: list[Room],
    operators: list[Operator],
    weights: dict[str, float],
    steps: int = 300,
    seed: int = 42,
    locked_names: set[str] | None = None,
    shift_hours: float = 24.0,
) -> list[Room]:
    """Simulated-annealing operator assignment for a fixed product config.

    1. Greedy initial solution (morale-constrained)
    2. SA: swap (same-facility, cross-facility) + replace (unassigned pool)
       -> Metropolis acceptance
    3. Return best solution found
    """
    rng = random.Random(seed)
    if not operators or not rooms:
        return rooms

    # ── Morale constraint helper ──
    def _can_sustain(op: 'Operator', facility: str) -> bool:
        """True if the operator can work a full shift at this facility."""
        return op.max_work_hours(facility, 24.0) >= shift_hours * 0.8

    # ── Identify pre-filled rooms (team-locked) ──
    # Rooms that already have operators assigned (e.g. community teams)
    # must be preserved -- the SA only optimizes remaining empty slots.
    locked_op_names: set[str] = set(locked_names) if locked_names else set()
    for room in rooms:
        for op in room.operators:
            locked_op_names.add(op.name)

    # ── Greedy initial solution (regret-based, morale-constrained) ──
    # Only fill rooms that don't already have team-locked operators.
    # Locked operators are excluded from the available pool.
    available = [op for op in operators
                 if op.name not in locked_op_names]
    empty_rooms = [r for r in rooms if not r.operators]

    # Pre-compute {operator: [(room_idx, weighted_eff), ...]} sorted desc
    # Only consider empty rooms (locked rooms are already filled)
    op_room_scores: dict[int, list[tuple[int, float]]] = {}
    for oi, op in enumerate(available):
        scores = []
        for ri, room in enumerate(empty_rooms):
            eff = op.efficiency_for(room.product, room.facility)
            if eff > 0 and _can_sustain(op, room.facility):
                w = weights.get(f"{room.facility}:{room.product}", 0.1)
                scores.append((ri, w * eff))
        scores.sort(key=lambda x: -x[1])  # best room first
        op_room_scores[oi] = scores

    # Compute regret = best_score - second_best_score
    op_regret: list[tuple[int, float]] = []
    for oi, scores in op_room_scores.items():
        if not scores:
            continue
        best = scores[0][1]
        second = scores[1][1] if len(scores) > 1 else 0.0
        op_regret.append((oi, best - second, best))

    # Assign operators with largest regret first
    op_regret.sort(key=lambda x: -x[1])
    room_slots_used: dict[int, int] = {ri: 0 for ri in range(len(empty_rooms))}

    for oi, _, _ in op_regret:
        scores = op_room_scores[oi]
        for ri, score in scores:
            if room_slots_used[ri] < empty_rooms[ri].max_slots:
                empty_rooms[ri].operators.append(available[oi])
                room_slots_used[ri] += 1
                break  # assigned -- move to next operator

    # Track unassigned operators for the replace move.
    assigned_names = {op.name for room in rooms for op in room.operators}
    unassigned = [op for op in operators if op.name not in assigned_names]

    def _total_score(rm: list[Room]) -> float:
        return sum(
            weights.get(f"{r.facility}:{r.product}", 0.1) * r.total_efficiency()
            for r in rm
        )

    best_rooms = [Room(facility=r.facility, product=r.product, index=r.index,
                       max_slots=r.max_slots, operators=list(r.operators)) for r in rooms]
    best_score = _total_score(best_rooms)

    # ── Group rooms by facility for swap candidates ──
    fac_groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rooms):
        fac_groups[r.facility].append(i)

    # ── SA loop ──
    for step in range(steps):
        T = _sa_temperature(step, steps)
        rand = rng.random()

        # Move type distribution: 25% cross-facility, 15% replace, 60% same-facility
        if rand < 0.25:
            # ── Cross-facility swap ──
            if len(rooms) < 2:
                continue
            a_idx, b_idx = rng.sample(range(len(rooms)), 2)
            ra, rb = rooms[a_idx], rooms[b_idx]
            if not ra.operators or not rb.operators:
                continue
            if ra.facility == rb.facility:
                continue  # same-facility handled below

            op_a = rng.choice(ra.operators)
            op_b = rng.choice(rb.operators)

            # Never touch team-locked operators
            if op_a.name in locked_op_names or op_b.name in locked_op_names:
                continue

            # Only allow if both operators can work in the target room
            eff_a_in_b = op_a.efficiency_for(rb.product, rb.facility)
            eff_b_in_a = op_b.efficiency_for(ra.product, ra.facility)
            if eff_a_in_b <= 0 or eff_b_in_a <= 0:
                continue
            # Morale: both operators must sustain the target room's shift
            if not _can_sustain(op_a, rb.facility) or not _can_sustain(op_b, ra.facility):
                continue

            old_score = (
                weights.get(f"{ra.facility}:{ra.product}", 0.1) *
                op_a.efficiency_for(ra.product, ra.facility) +
                weights.get(f"{rb.facility}:{rb.product}", 0.1) *
                op_b.efficiency_for(rb.product, rb.facility)
            )
            new_score = (
                weights.get(f"{ra.facility}:{ra.product}", 0.1) * eff_b_in_a +
                weights.get(f"{rb.facility}:{rb.product}", 0.1) * eff_a_in_b
            )
            delta = new_score - old_score

            if rng.random() < _sa_accept_prob(delta, T):
                ra.operators.remove(op_a)
                rb.operators.remove(op_b)
                ra.operators.append(op_b)
                rb.operators.append(op_a)
        elif rand < 0.40:
            # ── Replace: swap an assigned operator with an unassigned one ──
            # This escape hatch allows the SA to recover from greedy-init traps
            # where a multi-talented operator is assigned to sub-optimal rooms.
            if not unassigned:
                continue
            room = rng.choice(rooms)
            if not room.operators:
                continue
            old_op = rng.choice(room.operators)
            # Never touch team-locked operators
            if old_op.name in locked_op_names:
                continue
            old_eff = old_op.efficiency_for(room.product, room.facility)

            # Find unassigned operators that can work in this room
            candidate_indices = [
                i for i, op in enumerate(unassigned)
                if op.efficiency_for(room.product, room.facility) > 0
                and _can_sustain(op, room.facility)
            ]
            if not candidate_indices:
                continue
            idx = rng.choice(candidate_indices)
            new_op = unassigned[idx]
            new_eff = new_op.efficiency_for(room.product, room.facility)

            delta = weights.get(f"{room.facility}:{room.product}", 0.1) * (new_eff - old_eff)

            if rng.random() < _sa_accept_prob(delta, T):
                room.operators.remove(old_op)
                room.operators.append(new_op)
                unassigned[idx] = old_op  # old operator becomes unassigned
        else:
            # ── Same-facility swap ──
            swappable = {f: idxs for f, idxs in fac_groups.items() if len(idxs) >= 2}
            if not swappable:
                continue

            facility = rng.choice(list(swappable.keys()))
            a_idx, b_idx = rng.sample(swappable[facility], 2)
            ra, rb = rooms[a_idx], rooms[b_idx]

            if not ra.operators or not rb.operators:
                continue

            op_a = rng.choice(ra.operators)
            op_b = rng.choice(rb.operators)

            # Never touch team-locked operators
            if op_a.name in locked_op_names or op_b.name in locked_op_names:
                continue

            delta = (
                weights.get(f"{ra.facility}:{ra.product}", 0.1) *
                (op_b.efficiency_for(ra.product, ra.facility) -
                 op_a.efficiency_for(ra.product, ra.facility))
                +
                weights.get(f"{rb.facility}:{rb.product}", 0.1) *
                (op_a.efficiency_for(rb.product, rb.facility) -
                 op_b.efficiency_for(rb.product, rb.facility))
            )

            if rng.random() < _sa_accept_prob(delta, T):
                ra.operators.remove(op_a)
                rb.operators.remove(op_b)
                ra.operators.append(op_b)
                rb.operators.append(op_a)

        # ── Track best solution (after either swap type) ──
        current_score = _total_score(rooms)
        if current_score > best_score:
            best_score = current_score
            best_rooms = [Room(facility=r.facility, product=r.product, index=r.index,
                               max_slots=r.max_slots, operators=list(r.operators))
                          for r in rooms]

    return best_rooms


# ═══════════════════════════════════════════════════════════════════
# Pareto frontier
# ═══════════════════════════════════════════════════════════════════

def _product_composition_key(config) -> tuple[tuple[int, ...], ...]:
    """Canonical key: sorted product counts per facility."""
    from collections import Counter
    fac_counters: dict[str, Counter] = {}
    for facility, product in config.rooms:
        fac_counters.setdefault(facility, Counter())
        fac_counters[facility][product] += 1
    items = []
    for fac in sorted(fac_counters):
        items.append(tuple(sorted(fac_counters[fac].items())))
    return tuple(items)


def _dedup_by_composition(solutions: list[ParetoSolution]) -> list[ParetoSolution]:
    """Keep only the best-scoring solution for each unique product composition."""
    best: dict[tuple, ParetoSolution] = {}
    for s in solutions:
        key = _product_composition_key(s.config)
        if key not in best or s.total_score > best[key].total_score:
            best[key] = s
    return list(best.values())

def compute_pareto_frontier(
    solutions: list[ParetoSolution],
    sort_by: tuple[float, float, float] | None = None,
) -> list[ParetoSolution]:
    """Filter list to non-dominated solutions (Pareto frontier).

    A dominates B if A is ≥ B in all dimensions AND > B in at least one.
    sort_by: optional (orundum_w, lmd_w, combat_record_w) weights for ordering.
             Defaults to orundum-first for backward compatibility.
    """
    for i, sol in enumerate(solutions):
        sol.dominated_by = 0
        for j, other in enumerate(solutions):
            if i == j:
                continue
            if other.dominates(sol):
                sol.dominated_by += 1

    frontier = [s for s in solutions if s.dominated_by == 0]
    if sort_by:
        w_o, w_l, w_c = sort_by
        frontier.sort(
            key=lambda s: w_o * s.orundum_eff + w_l * s.lmd_eff + w_c * s.combat_record_eff,
            reverse=True,
        )
    else:
        frontier.sort(key=lambda s: (s.orundum_eff, s.lmd_eff), reverse=True)
    return frontier


# ═══════════════════════════════════════════════════════════════════
# Optimizer
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# Team templates -- community-vetted combo teams as optimization units
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TeamTemplate:
    """A community-vetted operator combo team that should be optimized as a unit."""
    name: str
    facility: str
    product_hint: str = ""       # Preferred product (PureGold/CombatRecord/OriginStone/Orundum/LMD)
    members: list[str] = field(default_factory=list)
    equiv_eff: float = 0.0       # Community-verified equivalent efficiency (%)
    tier: str = ""               # T0/T1/T2
    desc: str = ""
    requires_elite: dict[str, int] = field(default_factory=dict)  # member->min elite


# Community-vetted team templates from 公孙长乐 2025.06
COMMUNITY_TEAM_TEMPLATES: list[TeamTemplate] = [
    # ── Trade teams ──
    TeamTemplate("巫恋顶配组", "Trade", "Orundum",
                 ["巫恋", "龙舌兰", "柏喙"], 172, "T0",
                 "等效172%贸易+40%赤金，无人机最高优先级",
                 {"巫恋": 2, "龙舌兰": 2, "柏喙": 2}),
    TeamTemplate("巫恋裁缝组", "Trade", "Orundum",
                 ["巫恋", "龙舌兰", "卡夫卡"], 172, "T0",
                 "与柏喙等价",
                 {"巫恋": 2, "龙舌兰": 2, "卡夫卡": 2}),
    TeamTemplate("企鹅物流", "Trade", "Orundum",
                 ["德克萨斯", "拉普兰德", "能天使"], 100, "T1",
                 "Texas+Lapland combo+65%",
                 {"德克萨斯": 0, "拉普兰德": 0, "能天使": 0}),
    TeamTemplate("交际花续航组", "Trade", "",
                 ["古米", "月见夜", "空爆"], 90, "T1",
                 "-0.75心情，超长续航，适合长班",
                 {"古米": 0, "月见夜": 0, "空爆": 0}),
    TeamTemplate("龙舌兰单核", "Trade", "LMD",
                 ["龙舌兰"], 25, "T2",
                 "单人放贸易站，低配回本",
                 {"龙舌兰": 2}),
    TeamTemplate("黑键德狗组", "Trade", "LMD",
                 ["德克萨斯", "黑键", "古米"], 130, "T1",
                 "德狗65+黑键35+古米30=130%",
                 {"德克萨斯": 0, "黑键": 0, "古米": 0}),

    # ── Mfg teams ──
    TeamTemplate("红云仓库平民组", "Mfg", "",
                 ["红云", "蛇屠箱", "黑角"], 76, "T0",
                 "每格仓库+2%效率，零成本76%通用",
                 {"红云": 1, "蛇屠箱": 0, "黑角": 0}),
    TeamTemplate("红云仓库完全体", "Mfg", "",
                 ["红云", "火神", "蛇屠箱"], 84, "T1",
                 "火神19仓×2%=38%额外",
                 {"红云": 1, "火神": 2, "蛇屠箱": 0}),
    TeamTemplate("自动化赤金组", "Mfg", "PureGold",
                 ["清流", "温蒂", "森蚺"], 125, "T0",
                 "发电站越多效率越高，赤金1.25等效",
                 {"清流": 1, "温蒂": 0, "森蚺": 0}),
    TeamTemplate("赤金三人组", "Mfg", "PureGold",
                 ["砾", "斑点", "夜烟"], 95, "T1",
                 "95%赤金，不减心情",
                 {"砾": 1, "斑点": 1, "夜烟": 0}),
    TeamTemplate("泡泡火神组", "Mfg", "",
                 ["泡泡", "火神"], 52, "T1",
                 "大库存，一天一收必备",
                 {"泡泡": 1, "火神": 2}),
    TeamTemplate("经验三人组", "Mfg", "CombatRecord",
                 ["食铁兽", "断罪者"], 70, "T1",
                 "35+35作战记录",
                 {"食铁兽": 2, "断罪者": 1}),
    TeamTemplate("经验平民组", "Mfg", "CombatRecord",
                 ["白雪", "霜叶", "红豆"], 90, "T2",
                 "30+30+30作战记录",
                 {"白雪": 1, "霜叶": 1, "红豆": 0}),
    TeamTemplate("酒神食铁组", "Mfg", "CombatRecord",
                 ["酒神", "食铁兽", "Castle-3"], 115, "T1",
                 "55+30+30作战记录",
                 {"酒神": 2, "食铁兽": 0, "Castle-3": 0}),
    TeamTemplate("苍苔夜烟斑点", "Mfg", "PureGold",
                 ["苍苔", "夜烟", "斑点"], 90, "T2",
                 "30+30+30赤金",
                 {"苍苔": 0, "夜烟": 0, "斑点": 1}),
]


@dataclass
class TeamInstance:
    """A discovered team instance from the player's box."""
    template: TeamTemplate
    operators: list['Operator'] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # members not in box
    elite_blocked: list[str] = field(default_factory=list)  # members too low elite
    is_ready: bool = False

    @property
    def combined_eff(self) -> float:
        """Estimated combined efficiency including dynamic combo bonuses."""
        if not self.operators:
            return 0.0
        # Use template reference value as baseline
        base = self.template.equiv_eff
        # If we have real operators, compute actual combo bonus on top
        if len(self.operators) >= 2:
            dummy_room = Room(
                facility=self.template.facility,
                product=self.template.product_hint or "PureGold",
                index=0, max_slots=3,
                operators=list(self.operators),
            )
            combo = compute_room_combo_bonus(dummy_room)
            extra = sum(v for v in combo.values())
            base += extra
        return base


def _box_elite(val: int | dict | tuple) -> int:
    """Extract elite level from a box value (supports int, dict, tuple)."""
    if isinstance(val, dict):
        return val.get("elite", val.get("elite_level", 0))
    if isinstance(val, (tuple, list)):
        return int(val[0])
    return int(val)


def _box_level(val: int | dict | tuple) -> int:
    """Extract operator level from a box value (0 = unknown)."""
    if isinstance(val, dict):
        return val.get("level", 0)
    if isinstance(val, (tuple, list)) and len(val) >= 2:
        return int(val[1])
    return 0


def discover_teams(
    operators: list['Operator'],
    box: dict[str, int],
) -> list[TeamInstance]:
    """Scan the player's box against community team templates.

    Returns a list of TeamInstance sorted by tier and efficiency (T0 first).
    Full teams (all members available at required elite) come before partial ones.
    """
    results: list[TeamInstance] = []
    op_map = {op.name: op for op in operators}

    for template in COMMUNITY_TEAM_TEMPLATES:
        missing = []
        elite_blocked = []
        team_ops = []

        for member_name in template.members:
            if member_name not in box:
                missing.append(member_name)
                continue
            op = op_map.get(member_name)
            if op is None:
                missing.append(member_name)
                continue
            member_elite = _box_elite(box[member_name])
            member_level = _box_level(box[member_name])
            min_elite = template.requires_elite.get(member_name, 0)
            if member_elite < min_elite:
                elite_blocked.append(
                    f"{member_name}(需E{min_elite},当前E{member_elite})"
                )
            # Check level if box has level info and operator object has a skill
            # with explicit level_required
            if op and member_level > 0:
                for sk in op.skills:
                    if sk.level_required > 1 and member_level < sk.level_required:
                        elite_blocked.append(
                            f"{member_name}(需Lv{sk.level_required},当前Lv{member_level})"
                        )
                        break
            team_ops.append(op)

        is_ready = len(missing) == 0 and len(elite_blocked) == 0
        results.append(TeamInstance(
            template=template,
            operators=team_ops,
            missing=missing,
            elite_blocked=elite_blocked,
            is_ready=is_ready,
        ))

    # Sort: ready teams first, then by tier, then by efficiency
    tier_order = {"T0": 0, "T1": 1, "T2": 2, "": 3}
    results.sort(key=lambda t: (
        not t.is_ready,
        tier_order.get(t.template.tier, 3),
        -t.combined_eff,
    ))
    return results


class BaseOptimizer:
    """Multi-objective base scheduler with Pareto frontier optimization.

    Usage:
        opt = BaseOptimizer(knowledge_base)
        frontier = opt.solve_pareto(
            operator_box={"德克萨斯": 2, "清流": 2, ...},  # name -> elite_level
            layout="243",
            num_shifts=1,
        )
        best = opt.solve_with_weights(frontier, orundum=0.50, lmd=0.30, combat_record=0.20)
    """

    SA_STEPS = 300      # Simulated annealing iterations per config (cooled faster)
    SA_SEEDS = 3         # Multi-start: run SA from N random seeds, take best

    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base
        self._op_skills_cache: dict[str, list[OperatorSkill]] = {}

    # ── Public API ────────────────────────────────────────────────

    def solve_pareto(
        self,
        operator_box: dict[str, int | dict],   # {name: elite_level | {elite,level,...}}
        layout: str = "243",
        num_shifts: int = 1,
        sort_weights: tuple[float, float, float] | None = None,
        inventory: Inventory | None = None,
        mood_threshold: float = 0.5,           # 0.35-0.80, lower = longer work hours
        material_stock: 'MaterialStock | None' = None,
    ) -> list[ParetoSolution]:
        """Enumerate all product configs, run SA assignment, return Pareto frontier.

        Args:
            operator_box: {干员名: 精英等级}
            layout: "333", "243", "252", or "153"
            num_shifts: Number of shifts per day (1-4)
            sort_weights: optional (orundum, lmd, combat_record) for frontier ordering
            inventory: optional resource stockpile — relaxes chain constraints
                when the user has enough gold/stone to cover deficits
            mood_threshold: morale fraction (0.35-0.80) to trigger group rest.
                0.5 = rest at 50% morale → ~16h work for base drain.
                0.35 = rest at 35% morale → ~20h work for base drain.
                Lower values produce longer work shifts but higher burnout risk.
            material_stock: optional full MaterialStock from depot scan.
                Used to adjust product weights: low-stock items get priority,
                surplus items get deprioritized.
        """
        if layout not in LAYOUT_FACILITY_COUNTS:
            logger.warning("Unknown layout '%s', falling back to 243", layout)
            layout = "243"

        # Resolve operators
        operators = self._resolve_operators(operator_box)
        op_map = {op.name: op for op in operators}  # for team instantiation
        if not operators:
            logger.warning("No operators resolved from box")
            return []

        # ── Team discovery ──
        # Two-stage: (1) community-vetted templates, (2) auto-discovered synerigies.
        # Both are locked as atomic units during optimization.
        static_teams = discover_teams(operators, operator_box)
        dynamic_team_templates = discover_dynamic_teams(operators, operator_box)
        # Merge: dynamic teams first (they represent actual combo triggers),
        # then static templates that don't overlap with dynamic teams.
        dynamic_member_sets = [set(t.members) for t in dynamic_team_templates]
        all_ready_teams: list[TeamInstance] = []

        # Instantiate dynamic teams
        for dt in dynamic_team_templates:
            team_ops = [op_map.get(m) for m in dt.members if m in op_map]
            if len(team_ops) == len(dt.members):
                all_ready_teams.append(TeamInstance(
                    template=dt, operators=team_ops,
                    is_ready=True,
                ))

        # Add static teams that don't overlap with dynamic ones
        for st in static_teams:
            if not st.is_ready:
                continue
            st_members = set(st.template.members)
            if not any(st_members == ds for ds in dynamic_member_sets):
                all_ready_teams.append(st)

        # Sort: auto-discovered first (they represent actual combo triggers),
        # then community templates by tier
        tier_order = {"auto": 0, "T0": 1, "T1": 2, "T2": 3, "": 4}
        all_ready_teams.sort(key=lambda t: (
            tier_order.get(t.template.tier, 4),
            -t.combined_eff,
        ))

        ready_teams = all_ready_teams
        if ready_teams:
            logger.info(
                "Discovered %d ready teams (%d dynamic): %s",
                len(ready_teams),
                sum(1 for t in ready_teams if t.template.tier == "auto"),
                ", ".join(t.template.name for t in ready_teams[:10]),
            )

        # ── Auto-detect number of shifts ──
        # Single shift needs total_slots operators. With num_shifts>1,
        # operators rotate. But shift duration is also constrained by
        # operator morale -- a typical operator sustains ~32h (0.75/h drain),
        # so 2-3 shifts of 8-12h are optimal for most boxes.
        total_needed = sum(
            FACILITY_MAX_OPERS.get(f, 3) * c
            for f, c in LAYOUT_FACILITY_COUNTS.get(layout, {}).items()
        )
        if num_shifts <= 0:
            # Auto-detect: morale-driven groups work simultaneously
            # (整组同进同出), so only 1 set of rooms is needed.
            # Multiple shifts would split operators across time slots,
            # which contradicts the group rest model.
            num_shifts = 1

        # ── Fixed-shift mode: morale-constrained rotation ──
        # Only used when num_shifts > 1 (legacy fixed-schedule mode).
        # Morale-driven mode (num_shifts=1 from auto-detect) skips this
        # because groups work simultaneously, not in rotating shifts.
        if num_shifts > 1:
            max_hours_per_op = sorted([
                min(op.max_work_hours(fac, 24.0) for fac in ("Trade", "Mfg"))
                for op in operators
            ])
            median_max_hours = max_hours_per_op[len(max_hours_per_op) // 2] if max_hours_per_op else 24.0
            max_shift_hours = max(4.0, median_max_hours / 2.0)
            num_shifts = max(num_shifts, math.ceil(24.0 / max_shift_hours))
            num_shifts = max(1, min(num_shifts, 4))

        # Enumerate all product configurations and filter infeasible ones.
        # A config is infeasible if it demands more gold/stone than it produces.
        all_configs = enumerate_product_configs(layout)
        configs: list[ProductConfig] = []
        skipped_infeasible = 0
        for cfg in all_configs:
            num_lmd_trade = sum(1 for f, p in cfg.rooms if f == "Trade" and p == "LMD")
            num_puregold = sum(1 for f, p in cfg.rooms if f == "Mfg" and p == "PureGold")
            num_originium = sum(1 for f, p in cfg.rooms if f == "Mfg" and p == "OriginStone")
            num_orundum = sum(1 for f, p in cfg.rooms if f == "Trade" and p == "Orundum")
            num_combat = sum(1 for f, p in cfg.rooms if f == "Mfg" and p == "CombatRecord")

            # ── Structural chain viability ──
            # Each trade order consumes 2 raw materials (2 gold bars or
            # 2 source stones). At minimum, a trade room needs at least
            # 1 manufacturing room to supply it (50% output), ideally 2
            # (100% output). Hard-filter 0:1 configs; soft-penalize 1:1.
            inv = inventory or Inventory()
            _BASE_DAILY_PER_ROOM = 20

            # Gold chain: must have at least 1 PureGold per LMD trade room
            if num_lmd_trade > 0 and num_puregold < num_lmd_trade:
                skipped_infeasible += 1
                continue

            # Stone chain: must have at least 1 OriginStone per Orundum trade room
            if num_orundum > 0 and num_originium < num_orundum:
                skipped_infeasible += 1
                continue

            configs.append(cfg)
        if skipped_infeasible:
            logger.info(
                "Filtered %d infeasible configs (resource chain), %d remain",
                skipped_infeasible, len(configs),
            )
        shift_names = ["A组", "B组", "C组", "D组"][:num_shifts]
        shift_hours = 24.0 / num_shifts
        logger.info(
            "Enumerated %d product configs for layout %s with %d operators, %d shifts",
            len(configs), layout, len(operators), num_shifts,
        )

        # Score each config via SA assignment
        solutions: list[ParetoSolution] = []
        base_weights = self._default_weights()

        # ── Inventory-aware weight adjustment ──────────────────────────
        # When depot scan data is available, adjust base weights to favor
        # production of materials the player is low on and de-emphasize
        # materials they have in surplus.
        inv_adj = self._inventory_weight_adjustment(inventory, material_stock)
        base_weights = {k: v * inv_adj.get(k, 1.0) for k, v in base_weights.items()}

        _skip_logged: set[tuple[str, int]] = set()  # dedup warm-start skip logs across configs

        for ci, config in enumerate(configs):
            # ── Config-aware weight adjustment ──
            # When the config has both OriginStone manufacturing and Orundum
            # trade rooms, the two form a production chain (源石碎片->合成玉).
            # Boost Mfg:OriginStone weight to match Trade:Orundum so SA doesn't
            # bias operator swaps toward trade rooms at the expense of the
            # upstream supply chain.
            has_orundum_trade = any(
                f == "Trade" and p == "Orundum" for f, p in config.rooms
            )
            has_originium_mfg = any(
                f == "Mfg" and p == "OriginStone" for f, p in config.rooms
            )
            if has_orundum_trade and has_originium_mfg:
                weights = dict(base_weights)
                orundum_w = base_weights.get("Trade:Orundum", 10.0)
                weights["Mfg:OriginStone"] = max(
                    weights.get("Mfg:OriginStone", 3.0), orundum_w,
                )
            else:
                weights = base_weights

            # ── Multi-shift: A-first sequential SA ──
            # Two-pass for num_shifts>=2: A队 gets top pick of ALL operators,
            # B队 gets best-of-remaining.  This avoids one-pass global SA
            # splitting top operators across both shifts (mediocre efficiency
            # for both) and matches how real players schedule their base.
            prod_rooms = config.to_rooms()
            fixed_facs = _get_fixed_facilities(layout)

            def _clone_rooms(rooms):
                return [Room(facility=r.facility, product=r.product,
                             index=r.index, max_slots=r.max_slots)
                        for r in rooms]

            def _run_sa(rooms, ops, seed_offset: int):
                """Run multi-seed SA on given rooms+operators, return best assignment."""
                best = 0.0
                best_assigned = rooms
                for seed in range(self.SA_SEEDS):
                    fresh = _clone_rooms(rooms)
                    assigned = simulated_annealing_assignment(
                        rooms=fresh, operators=ops, weights=weights,
                        steps=self.SA_STEPS, seed=seed * 1000 + ci + seed_offset,
                        shift_hours=shift_hours,
                    )
                    score = sum(
                        weights.get(f"{r.facility}:{r.product}", 0.1) * r.total_efficiency()
                        for r in assigned
                    )
                    if score > best:
                        best = score
                        best_assigned = assigned
                return best_assigned

            if num_shifts >= 2:
                # ── Pass 1: A队 — top priority, ALL operators ──
                a_rooms = _clone_rooms(prod_rooms)
                for fac, prod, slots in fixed_facs:
                    a_rooms.append(Room(facility=fac, product=prod, index=0, max_slots=slots))
                best_a = _run_sa(a_rooms, operators, 0)
                # Collect used operators
                a_used: set[str] = set()
                for r in best_a:
                    for op in r.operators:
                        a_used.add(op.name)
                # ── Pass 2: B队 — remaining operators only ──
                remaining = [op for op in operators if op.name not in a_used]
                b_rooms = _clone_rooms(prod_rooms)
                best_b = _run_sa(b_rooms, remaining, 1000)
                # Merge: A production rooms first, then B production rooms
                best_all_rooms = best_a + best_b
                fixed_start_idx = len(best_a)
                best_score = sum(
                    weights.get(f"{r.facility}:{r.product}", 0.1) * r.total_efficiency()
                    for r in best_all_rooms
                )
            else:
                # ── Single-shift: original SA ──
                all_rooms_single = []
                for room in prod_rooms:
                    all_rooms_single.append(Room(
                        facility=room.facility, product=room.product,
                        index=room.index, max_slots=room.max_slots))
                for fac, prod, slots in fixed_facs:
                    all_rooms_single.append(Room(
                        facility=fac, product=prod, index=0, max_slots=slots))
                fixed_start_idx = len(all_rooms_single) - len(fixed_facs)
                best_assigned = _run_sa(all_rooms_single, operators, 0)
                best_all_rooms = best_assigned
                best_score = sum(
                    weights.get(f"{r.facility}:{r.product}", 0.1) * r.total_efficiency()
                    for r in best_assigned
                )
            # ── Build shifts from best_all_rooms ──
            shifts: list[Shift] = []
            if num_shifts >= 2:
                # Two-pass: best_a = production + fixed, best_b = production only.
                # Fixed facilities (Control/Office/Reception/Dorm) don't rotate —
                # they keep the same operators 24/7, so copy A's fixed rooms to B.
                prod_count = len(best_a) - len(fixed_facs)
                a_prod = best_a[:prod_count]
                a_fixed = best_a[prod_count:]
                b_prod = best_b
                # Shallow-clone A's fixed rooms for B shift (same operators, same rooms)
                b_fixed = [Room(facility=r.facility, product=r.product,
                                index=r.index, max_slots=r.max_slots,
                                operators=list(r.operators))
                          for r in a_fixed]
                shifts = [
                    Shift(name="A组", duration_hours=shift_hours,
                          rooms=a_prod + a_fixed),
                    Shift(name="B组", duration_hours=shift_hours,
                          rooms=b_prod + b_fixed),
                ]
            else:
                # Single-shift: production rooms + fixed at end
                prod_count = len(best_all_rooms) - len(fixed_facs)
                shifts = [
                    Shift(name="A组", duration_hours=shift_hours,
                          rooms=best_all_rooms[:prod_count] + best_all_rooms[prod_count:]),
                ]

            # Compute daily-average output metrics
            orundum_eff = 0.0
            lmd_eff = 0.0
            cr_eff = 0.0
            stone_eff = 0.0
            total_slots = 0
            filled_slots = 0
            for shift_rooms_iter in (s.rooms for s in shifts):
                orundum_eff += sum(
                    r.total_efficiency() for r in shift_rooms_iter
                    if r.facility == "Trade" and r.product == "Orundum"
                )
                lmd_eff += sum(
                    r.total_efficiency() for r in shift_rooms_iter
                    if (r.facility == "Trade" and r.product == "LMD") or
                       (r.facility == "Mfg" and r.product == "PureGold")
                )
                cr_eff += sum(
                    r.total_efficiency() for r in shift_rooms_iter
                    if r.facility == "Mfg" and r.product == "CombatRecord"
                )
                stone_eff += sum(
                    r.total_efficiency() for r in shift_rooms_iter
                    if r.facility == "Mfg" and r.product == "OriginStone"
                )
                total_slots += sum(r.max_slots for r in shift_rooms_iter)
                filled_slots += sum(len(r.operators) for r in shift_rooms_iter)

            orundum_eff /= num_shifts
            lmd_eff /= num_shifts
            cr_eff /= num_shifts
            stone_eff /= num_shifts

            # ── Supply-chain throughput caps ──
            # 1 orundum order = 2 stones; 1 LMD order = 2 gold bars.
            # Manufacturing rooms feed trade rooms. If stone/gold production
            # is less than trade demand, the effective trade output is capped.
            stone_capacity = stone_eff * 0.5
            if stone_eff > 0 and orundum_eff > 0 and stone_capacity < orundum_eff:
                orundum_eff = stone_capacity

            pg_trade_eff = sum(
                r.total_efficiency() for s in shifts for r in s.rooms
                if r.facility == "Trade" and r.product == "LMD"
            ) / num_shifts if num_shifts else 0
            pg_mfg_eff = sum(
                r.total_efficiency() for s in shifts for r in s.rooms
                if r.facility == "Mfg" and r.product == "PureGold"
            ) / num_shifts if num_shifts else 0
            gold_capacity = pg_mfg_eff * 0.5
            if pg_mfg_eff > 0 and pg_trade_eff > 0 and gold_capacity < pg_trade_eff:
                lmd_eff -= (pg_trade_eff - gold_capacity)

            total_orundum = orundum_eff + 0.5 * stone_eff
            coverage = filled_slots / max(total_slots, 1)

            # ── Morale-driven scheduling ──────────────────────────
            # Build rest groups FROM SA's ACTUAL room assignments (not from config).
            # This ensures group composition matches what SA optimized for.
            rest_groups = _build_rest_groups_from_shifts(shifts, operators)
            operator_dict = {op.name: op for op in operators}
            dorm_snapshots: list[DormSnapshot] = []
            fia_charges: list[FiaChargeTarget] = []
            w2r_ratio = 0.0

            if rest_groups:
                num_dorms = LAYOUT_DORM_COUNTS.get(layout, 4)

                # Check for Fiammetta in box
                fia_name = next(
                    (n for n in operator_box if n == "菲亚梅塔"), None)

                # ── Phase 1: compute work durations ──
                for g in rest_groups:
                    g.work_duration = _compute_work_duration(g, mood_threshold)

                # ── Phase 2: compute per-dorm recovery quality ──
                # Build dorm snapshots with ALL operators (for aura computation).
                all_resting_names: list[str] = []
                for g in rest_groups:
                    all_resting_names.extend(op.name for op in g.operators)
                all_resting_names = list(dict.fromkeys(all_resting_names))

                dorm_snapshots = _assign_dorms(
                    resting_group=RestGroup(
                        group_id=-1, facility="", room_indices=[],
                        operators=[], product="",
                    ),
                    other_resting_ops=all_resting_names,
                    all_operators=operator_dict,
                    num_dorms=num_dorms,
                )

                # ── Phase 3: assign groups to dorms optimally ──
                # Strategy: compute rest_duration for every (group, dorm) pair,
                # then greedily assign: group with longest rest → dorm with
                # shortest rest time (best recovery).
                if len(rest_groups) <= num_dorms:
                    # Compute rest duration for each group in each dorm
                    group_dorm_times: list[tuple[int, int, float]] = []  # (gid, dorm, rest_hours)
                    for g in rest_groups:
                        for di in range(num_dorms):
                            rd = _compute_rest_duration(
                                g, di, operator_dict, dorm_snapshots, g.work_duration,
                            )
                            group_dorm_times.append((g.group_id, di, rd))

                    # Sort: groups with longest rest time get priority
                    group_dorm_times.sort(key=lambda x: -x[2])

                    assigned_dorms: set[int] = set()
                    group_assignments: dict[int, int] = {}  # gid -> dorm_index

                    for gid, di, rd in group_dorm_times:
                        if gid in group_assignments:
                            continue  # already assigned
                        if di in assigned_dorms:
                            continue  # dorm already taken
                        group_assignments[gid] = di
                        assigned_dorms.add(di)

                    # Any unassigned group gets remaining dorms
                    remaining_dorms = [di for di in range(num_dorms) if di not in assigned_dorms]
                    for g in rest_groups:
                        if g.group_id not in group_assignments:
                            if remaining_dorms:
                                group_assignments[g.group_id] = remaining_dorms.pop(0)
                            else:
                                # All dorms taken — share with the group that has
                                # the shortest rest time (least conflict)
                                group_assignments[g.group_id] = min(
                                    group_assignments.values(),
                                    key=lambda di: sum(
                                        rd for gid2, di2, rd in group_dorm_times
                                        if gid2 in group_assignments and group_assignments[gid2] == di
                                    ),
                                )
                else:
                    # More groups than dorms — round-robin fallback
                    group_assignments = {g.group_id: g.group_id % num_dorms for g in rest_groups}

                # Apply assignments and compute final rest durations
                for g in rest_groups:
                    g.rest_dorm_index = group_assignments.get(g.group_id, g.group_id % num_dorms)
                    g.rest_duration = _compute_rest_duration(
                        g, g.rest_dorm_index, operator_dict,
                        dorm_snapshots, g.work_duration,
                    )

                # Fia charges
                if fia_name:
                    fia_charges = _compute_fia_charges(
                        rest_groups, operator_dict, fia_name, 0.7,
                    )

                # Work-to-rest ratio
                total_work = sum(g.work_duration for g in rest_groups)
                total_rest = sum(g.rest_duration for g in rest_groups)
                w2r_ratio = total_work / max(total_rest, 0.1)

                # Annotate shifts with rest durations (shift 0 gets them)
                if shifts and rest_groups:
                    # Mark room rest_duration/dorm_index for display
                    pass

            # ── Inventory-aware score bonus/penalty ───────────────
            # Adjust total_score based on what the player already has.
            # Producing materials the player is low on → bonus.
            # Producing materials in surplus → penalty.
            # ── LMD income floor ─────────────────────────────────
            # Plans with zero LMD production drain the wallet. LMD
            # is consumed by daily operations (recruitment, crafting,
            # upgrading). Unless the player has confirmed deep reserves,
            # heavily penalize zero-LMD plans.
            _lmd_rooms = sum(1 for r in config.rooms
                           if r[0] == "Trade" and r[1] == "LMD")
            _known_lmd = 0
            if material_stock and not material_stock.is_empty():
                _known_lmd = material_stock.lmd
            elif inventory and not inventory.is_empty():
                _known_lmd = inventory.lmd

            lmd_penalty = 1.0
            if _lmd_rooms == 0:
                if _known_lmd > 0:
                    if _known_lmd < 50000:
                        lmd_penalty = 0.10   # near-broke: barely viable
                    elif _known_lmd < 100000:
                        lmd_penalty = 0.30
                    elif _known_lmd < 200000:
                        lmd_penalty = 0.50
                    else:
                        lmd_penalty = 0.80   # deep reserves, warning only
                else:
                    lmd_penalty = 0.30       # unknown stock, conservative
            inv_score = best_score * lmd_penalty

            # Room counts for inventory bonus below
            orundum_room_count = sum(1 for r in config.rooms if r[0]=="Trade" and r[1]=="Orundum")
            puregold_room_count = sum(1 for r in config.rooms if r[0]=="Mfg" and r[1]=="PureGold")
            cr_room_count = sum(1 for r in config.rooms if r[0]=="Mfg" and r[1]=="CombatRecord")
            os_room_count = sum(1 for r in config.rooms if r[0]=="Mfg" and r[1]=="OriginStone")

            inv = inventory or Inventory()
            if not inv.is_empty() or (material_stock and not material_stock.is_empty()):
                bonus = 0.0
                # Puregold surplus → penalize PureGold production
                if inv.puregold >= 200:
                    bonus -= 8.0 * puregold_room_count
                elif inv.puregold >= 100:
                    bonus -= 4.0 * puregold_room_count
                if inv.origin_stone <= 5:
                    bonus += 10.0 * os_room_count
                elif inv.origin_stone <= 15:
                    bonus += 5.0 * os_room_count
                if inv.orundum >= 300:
                    bonus -= 3.0 * orundum_room_count

                # From MaterialStock: full depot context
                if material_stock and not material_stock.is_empty():
                    if material_stock.lmd >= 500000:
                        bonus -= 5.0 * _lmd_rooms
                        bonus -= 3.0 * puregold_room_count
                    elif material_stock.lmd >= 200000:
                        bonus -= 2.0 * _lmd_rooms
                    elif material_stock.lmd <= 10000:
                        bonus += 8.0 * _lmd_rooms
                        bonus += 5.0 * puregold_room_count

                inv_score = inv_score + bonus

            # ── Concrete daily output estimates ──
            daily_out = BaseOptimizer._estimate_daily_output(shifts, num_shifts)

            # ── Closed-loop sustainability analysis ──
            # Build a temporary solution to pass to _analyze_sustainability
            _tmp_sol = ParetoSolution(
                config=config, shifts=shifts,
                daily_orundum=daily_out["daily_orundum"],
                daily_lmd=daily_out["daily_lmd"],
                daily_combat_record=daily_out["daily_combat_record"],
                daily_puregold_net=daily_out["daily_puregold_net"],
            )
            sustain = BaseOptimizer._analyze_sustainability(_tmp_sol, inventory, material_stock)

            # ── Sustainability penalty ──────────────────────────
            # Unsustainable plans (LMD deficit, sanity impossible) get a
            # heavy score penalty so they won't be recommended as "best".
            # They still appear in the frontier for transparency, but ranked
            # below sustainable alternatives.
            sustain_penalty = 1.0
            if sustain["verdict"] == "both":
                sustain_penalty = 0.05   # LMD drain + sanity impossible → near-zero
            elif sustain["verdict"] == "lmd_deficit":
                # Scale penalty by deficit magnitude
                deficit = abs(sustain["lmd_balance"])
                if deficit >= 10000:
                    sustain_penalty = 0.05  # >1万/天亏空 → severe
                elif deficit >= 5000:
                    sustain_penalty = 0.10
                else:
                    sustain_penalty = 0.30
            elif sustain["verdict"] == "sanity_impossible":
                sustain_penalty = 0.30  # sanity problem alone
            inv_score = inv_score * sustain_penalty

            solutions.append(ParetoSolution(
                config=config,
                shifts=shifts,
                orundum_eff=total_orundum,
                lmd_eff=lmd_eff,
                combat_record_eff=cr_eff,
                total_score=inv_score,
                coverage=coverage,
                schedule_mode="morale_driven" if rest_groups else "fixed_shift",
                rest_groups=rest_groups,
                dorm_snapshots=dorm_snapshots,
                fia_charges=fia_charges,
                work_to_rest_ratio=round(w2r_ratio, 2),
                operator_names=sorted(operator_box.keys()),
                box_data=operator_box,
                daily_orundum=daily_out["daily_orundum"],
                daily_lmd=daily_out["daily_lmd"],
                daily_combat_record=daily_out["daily_combat_record"],
                daily_puregold_net=daily_out["daily_puregold_net"],
                sustain_lmd_balance=sustain["lmd_balance"],
                sustain_rock_demand=sustain["rock_demand"],
                sustain_sanity_cost=sustain["sanity_cost"],
                sustain_verdict=sustain["verdict"],
                sustain_detail=sustain["detail"],
            ))

        # Normalize efficiencies to 0..1 for Pareto comparison
        if solutions:
            max_orundum = max(s.orundum_eff for s in solutions) or 1.0
            max_lmd = max(s.lmd_eff for s in solutions) or 1.0
            max_cr = max(s.combat_record_eff for s in solutions) or 1.0
            for s in solutions:
                s.orundum_eff /= max_orundum
                s.lmd_eff /= max_lmd
                s.combat_record_eff /= max_cr

        # Deduplicate: same product composition -> keep best-scoring
        solutions = _dedup_by_composition(solutions)

        # Compute Pareto frontier, sorted by user weights if provided
        frontier = compute_pareto_frontier(solutions, sort_by=sort_weights)
        logger.info(
            "Pareto frontier: %d non-dominated solutions out of %d unique compositions",
            len(frontier), len(solutions),
        )

        return frontier




    def solve_with_weights(
        self,
        frontier: list,
        orundum: float = 0.0,
        lmd: float = 0.0,
        combat_record: float = 0.0,
    ):
        """Select the best Pareto solution for a given weighting."""
        if not frontier:
            return None
        best = max(
            frontier,
            key=lambda s: (
                orundum * s.orundum_eff +
                lmd * s.lmd_eff +
                combat_record * s.combat_record_eff
            ),
        )
        return best

    def solve_balanced(
        self,
        frontier: list,
    ):
        """Select the most balanced point on the Pareto frontier (knee-point)."""
        if not frontier:
            return None
        import math
        best = min(
            frontier,
            key=lambda s: math.sqrt(
                (1.0 - s.orundum_eff) ** 2
                + (1.0 - s.lmd_eff) ** 2
                + (1.0 - s.combat_record_eff) ** 2
            ),
        )
        return best

    def solve_legacy(
        self,
        operator_box: list,
        goal: str = "orundum_max",
        layout: str = "243",
        num_shifts: int = 1,
    ):
        """Legacy API: solve with string names, return dict for backward compat."""
        from typing import Any
        box_dict = self._parse_box_legacy(operator_box)
        weight_map = {
            "orundum_max":     (0.60, 0.25, 0.15),
            "lmd_max":         (0.05, 0.80, 0.15),
            "combat_record_max": (0.05, 0.15, 0.80),
            "balanced":        (0.30, 0.40, 0.30),
            "mixed_orundum_upgrade": (0.50, 0.30, 0.20),
        }
        w = weight_map.get(goal, (0.60, 0.25, 0.15))
        frontier = self.solve_pareto(box_dict, layout, num_shifts, sort_weights=w)
        if not frontier:
            return {"error": "No feasible solution found", "frontier": [], "best": None}
        best = self.solve_with_weights(frontier, *w)
        return {
            "frontier": [
                {
                    "orundum_eff": round(s.orundum_eff, 3),
                    "lmd_eff": round(s.lmd_eff, 3),
                    "combat_record_eff": round(s.combat_record_eff, 3),
                    "coverage": round(s.coverage, 2),
                    "product_summary": self._describe_config(s.config),
                    "schedule": self._format_shift(s.shifts[0]) if s.shifts else [],
                }
                for s in frontier[:10]
            ],
            "best": {
                "orundum_eff": round(best.orundum_eff, 3),
                "lmd_eff": round(best.lmd_eff, 3),
                "combat_record_eff": round(best.combat_record_eff, 3),
                "coverage": round(best.coverage, 2),
                "product_summary": self._describe_config(best.config) if best else "",
                "schedule": self._format_shift(best.shifts[0]) if best and best.shifts else [],
            } if best else None,
        }

    def _resolve_operators(self, box: dict) -> list:
        """Resolve {name: elite_level} or {name: {elite, level}} to Operator objects."""
        operators = []
        for name, val in box.items():
            name = name.strip()
            if not name:
                continue
            # Support both old format (int) and new format (dict or tuple)
            if isinstance(val, (int, float)):
                elite, level = int(val), 0
            elif isinstance(val, dict):
                elite = val.get("elite", val.get("elite_level", 2))
                level = val.get("level", 0)
            elif isinstance(val, (tuple, list)) and len(val) >= 2:
                elite, level = int(val[0]), int(val[1])
            else:
                elite, level = int(val), 0
            op_skills = self._load_operator_skills(name)
            rarity = self._get_operator_rarity(name)
            faction = self._get_operator_faction(name)
            operators.append(Operator(
                name=name, elite_level=elite, level=level,
                rarity=rarity, faction=faction, skills=op_skills,
            ))
        return operators

    def _parse_box_legacy(self, box: list) -> dict:
        """Parse old-format box list into {name: elite_level}."""
        return parse_operator_box("、".join(box))

    def _load_operator_skills(self, operator_name: str) -> list:
        """Load skills with elite unlock requirements from KB + curated data."""
        if operator_name in self._op_skills_cache:
            return self._op_skills_cache[operator_name]

        skills = []
        kb_op_data = None

        if self.kb:
            try:
                op_data = self.kb.get(
                    "arknights", "operator_base_skills",
                    key=operator_name, key_field="name",
                )
                kb_op_data = op_data
                if op_data and "base_skills" in op_data:
                    for bs in op_data["base_skills"]:
                        skill_id = bs.get("skill_id", "")
                        desc = self._load_skill_description(skill_id) or bs.get("description", "")
                        facility = bs.get("facility", "")
                        eff_detail = _normalize_skill_efficiency(
                            self._load_skill_efficiency(skill_id, desc),
                            facility,
                        )
                        unlock_raw = bs.get("unlock_condition", "精英0")
                        elite_required = ELITE_UNLOCK_MAP.get(unlock_raw, 0)
                        combo = parse_combo_from_desc(desc)
                        wh_cap = parse_warehouse_capacity(desc)
                        dorm_self, dorm_room, _ = parse_dorm_recovery(desc)
                        skills.append(OperatorSkill(
                            skill_id=skill_id,
                            facility=facility,
                            name=bs.get("skill_name", skill_id),
                            efficiency=eff_detail,
                            elite_required=elite_required,
                            level_required=bs.get("level_required", 1),
                            combo=combo,
                            warehouse_capacity=wh_cap,
                            morale_mod=parse_morale_modifier(desc),
                            dorm_self_recovery=dorm_self,
                            dorm_room_recovery=dorm_room,
                        ))
            except Exception as e:
                logger.debug("KB lookup failed for %s: %s", operator_name, e)

        try:
            from src.intelligence.arknights.operator_skills_data import OPERATOR_SKILLS
            if operator_name in OPERATOR_SKILLS:
                # Build index of existing KB skills by both skill_id and facility
                # (skill_ids can differ between KB and curated data).
                kb_by_fac: dict[str, list[OperatorSkill]] = {}
                for s in skills:
                    kb_by_fac.setdefault(s.facility, []).append(s)
                for sd in OPERATOR_SKILLS[operator_name]:
                    cid = sd.get("skill_id", "")
                    facility = sd.get("facility", "")
                    # Try merging into an existing KB skill by facility
                    merged = False
                    if facility and facility in kb_by_fac:
                        for existing in kb_by_fac[facility]:
                            # Merge level_required if KB skill doesn't have it
                            if existing.level_required <= 1 and sd.get("level_required", 1) > 1:
                                existing.level_required = sd["level_required"]
                            if existing.dorm_self_recovery == 0.0:
                                existing.dorm_self_recovery = sd.get("dorm_self_recovery", 0.0)
                            if existing.dorm_room_recovery == 0.0:
                                existing.dorm_room_recovery = sd.get("dorm_room_recovery", 0.0)
                            merged = True
                    if merged:
                        continue
                    # No KB skill for this facility — add curated as new
                    curated_eff = sd.get("efficiency", {})
                    if not curated_eff:
                        continue
                    inferred_elite = _infer_elite_for_facility(kb_op_data, facility)
                    skills.append(OperatorSkill(
                        skill_id=cid,
                        facility=facility,
                        name=sd.get("name", ""),
                        efficiency=_normalize_skill_efficiency(curated_eff, facility),
                        elite_required=inferred_elite,
                        level_required=sd.get("level_required", 1),
                        dorm_self_recovery=sd.get("dorm_self_recovery", 0.0),
                        dorm_room_recovery=sd.get("dorm_room_recovery", 0.0),
                    ))
        except ImportError:
            pass

        self._op_skills_cache[operator_name] = skills
        return skills

    def _load_skill_efficiency(self, skill_id: str, desc: str = "") -> dict:
        """Load efficiency from base_skills.json, fall back to desc parsing."""
        eff = {}
        if self.kb and skill_id:
            try:
                skills_list = self.kb.query("arknights", "base_skills", filters={"id": skill_id}, limit=1)
                if skills_list:
                    eff = dict(skills_list[0].get("efficiency", {}))
            except Exception:
                pass
        has_positive = any(isinstance(v, (int, float)) and v > 0 for v in eff.values())
        if not has_positive and desc:
            eff = self._parse_efficiency_from_desc(desc)
        return eff

    def _load_skill_description(self, skill_id: str) -> str | None:
        """Load full skill description from base_skills.json."""
        if not self.kb or not skill_id:
            return None
        try:
            skills_list = self.kb.query("arknights", "base_skills", filters={"id": skill_id}, limit=1)
            if skills_list:
                raw_desc = skills_list[0].get("description", [])
                if isinstance(raw_desc, list) and raw_desc:
                    return raw_desc[0]
                if isinstance(raw_desc, str) and raw_desc:
                    return raw_desc
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_efficiency_from_desc(desc: str) -> dict:
        """Parse Chinese skill descriptions for efficiency numbers."""
        result = {}
        product_kw = [
            ("贵金属|赤金", "PureGold"),
            ("作战记录|录像带", "CombatRecord"),
            ("源石碎片", "OriginStone"),
            ("无人机", "Drone"),
            ("订单获取", "all"),
            ("线索搜集", "all"),
            ("联络速度", "all"),
        ]
        import re
        percentages = re.findall(r'(\+?\d+(?:\.\d+)?)\s*%', desc)
        if not percentages:
            return result
        for kw, product in product_kw:
            if re.search(kw, desc):
                val = float(percentages[0])
                result[product] = val
                return result
        val = float(percentages[0])
        result["all"] = val
        return result

    def _get_operator_rarity(self, name: str) -> int:
        if self.kb:
            try:
                op_data = self.kb.get("arknights", "operator_base_skills", key=name, key_field="name")
                if op_data:
                    return op_data.get("rarity", 1)
            except Exception:
                pass
        return 1

    def _get_operator_faction(self, name: str) -> str:
        if self.kb:
            try:
                op_data = self.kb.get("arknights", "operator_base_skills", key=name, key_field="name")
                if op_data:
                    return op_data.get("faction", "")
            except Exception:
                pass
        return ""

    @staticmethod
    def _describe_config(config) -> str:
        """Human-readable product configuration summary."""
        parts = []
        for facility, product in config.rooms:
            parts.append(f"{facility}:{product}")
        return ", ".join(parts)

    @staticmethod
    def _format_shift(shift) -> list:
        """Format a shift as a list of room dicts."""
        return [
            {
                "facility": r.facility,
                "product": r.product,
                "index": r.index,
                "operators": [op.name for op in r.operators],
                "warnings": r.warnings,
                "efficiency": round(r.total_efficiency(), 1),
            }
            for r in shift.rooms
        ]

    @staticmethod
    def _compute_production_costs(
        config: 'ProductConfig',
        shifts: list['Shift'],
        num_shifts: int,
    ) -> dict:
        """Compute sanity cost of rock farming to sustain stone production.

        PureGold and CombatRecord have zero input cost — base produces them
        autonomously. Only OriginStone (源石碎片) requires 固源岩 which must
        be farmed from 1-7 (~4 sanity per rock, 2 rocks per stone).

        daily_sanity: sanity needed per day for rock farming
        """
        stone_rooms = sum(1 for f, p in config.rooms
                         if f == "Mfg" and p == "OriginStone")
        orundum_trade_rooms = sum(1 for f, p in config.rooms
                                  if f == "Trade" and p == "Orundum")

        if stone_rooms == 0:
            return {
                "daily_sanity": 0,
                "daily_rock_demand": 0,
                "daily_stone_output": 0,
                "daily_stone_consumed": 0,
                "sustainable": "ok",
                "warnings": [],
            }

        # Stone room efficiency
        eff_sum = 0.0; n = 0
        for shift in shifts:
            for r in shift.rooms:
                if r.facility == "Mfg" and r.product == "OriginStone":
                    eff_sum += r.total_efficiency(); n += 1
        avg_eff = eff_sum / n if n > 0 else 0.0

        # ~20 stones/day/room base × (1 + efficiency/100) × 2 rocks/stone
        _BASE = 20.0
        daily_stones = stone_rooms * _BASE * (1.0 + avg_eff / 100.0)
        daily_rocks = daily_stones * 2.0
        daily_sanity = daily_rocks * 4.0  # 1-7: ~4 sanity/rock

        # Orundum trade consumption: 2 stones per order
        daily_stone_consumed = orundum_trade_rooms * _BASE * 2

        warnings = []
        if daily_sanity > 480:
            sustainable = "critical"
            warnings.append(f"需{daily_sanity:.0f}理智/天刷1-7")
        elif daily_sanity > 240:
            sustainable = "warning"
            warnings.append(f"需{daily_sanity:.0f}理智/天刷1-7")
        else:
            sustainable = "ok"

        return {
            "daily_sanity": round(daily_sanity, 0),
            "daily_rock_demand": round(daily_rocks, 0),
            "daily_stone_output": round(daily_stones, 0),
            "daily_stone_consumed": round(daily_stone_consumed, 0),
            "sustainable": sustainable,
            "warnings": warnings,
        }

    @staticmethod
    def _estimate_daily_output(
        shifts: list['Shift'],
        num_shifts: int,
    ) -> dict:
        """Estimate concrete daily production numbers for a schedule.

        Returns {daily_orundum, daily_lmd, daily_combat_record, daily_puregold_net}
        as actual in-game quantities per 24h day.

        Game mechanics reference:
          - Manufacturing: 1 item / 72 min → 20/day at 0% bonus
          - Trade: 1 order / 144 min → 10/day at 0% bonus
          - LMD order at Trade Lv3 = 1000 LMD each
          - Orundum order = 20 orundum each
          - Each trade order consumes 2 materials (2 gold or 2 stones)
        """
        BASE_MFG_PER_DAY = 20.0     # items/day per room at 0% efficiency
        BASE_TRADE_PER_DAY = 10.0   # orders/day per room at 0% efficiency
        LMD_PER_ORDER = 1000.0      # LMD per order at Trade Lv3
        ORUNDUM_PER_ORDER = 20.0    # orundum per order
        STONES_PER_ORDER = 2.0      # origin stones consumed per orundum order
        GOLD_PER_ORDER = 2.0        # gold bars consumed per LMD order

        # Aggregate efficiency by (facility, product)
        eff_sum: dict[tuple[str, str], tuple[float, int]] = {}  # (sum_eff, room_count)
        for shift in shifts:
            for r in shift.rooms:
                key = (r.facility, r.product)
                prev_sum, prev_count = eff_sum.get(key, (0.0, 0))
                eff_sum[key] = (prev_sum + r.total_efficiency(), prev_count + 1)

        def _daily(facility: str, product: str, base_rate: float) -> float:
            total_eff, count = eff_sum.get((facility, product), (0.0, 0))
            if count == 0:
                return 0.0
            avg_eff = total_eff / count  # count already spans all shifts
            return count * base_rate * (1.0 + avg_eff / 100.0)

        # Raw production (before supply-chain bottlenecks)
        stone_output = _daily("Mfg", "OriginStone", BASE_MFG_PER_DAY)
        gold_output = _daily("Mfg", "PureGold", BASE_MFG_PER_DAY)
        cr_output = _daily("Mfg", "CombatRecord", BASE_MFG_PER_DAY)
        orundum_trade_raw = _daily("Trade", "Orundum", BASE_TRADE_PER_DAY)
        lmd_trade_raw = _daily("Trade", "LMD", BASE_TRADE_PER_DAY)

        # Apply supply-chain bottlenecks:
        # Orundum trade is limited by stone supply (2 stones per order)
        stone_demand = orundum_trade_raw * STONES_PER_ORDER
        if stone_output < stone_demand and orundum_trade_raw > 0:
            orundum_orders = stone_output / STONES_PER_ORDER
        else:
            orundum_orders = orundum_trade_raw

        # LMD trade is limited by gold supply (2 gold per order)
        gold_demand = lmd_trade_raw * GOLD_PER_ORDER
        if gold_output < gold_demand and lmd_trade_raw > 0:
            lmd_orders = gold_output / GOLD_PER_ORDER
        else:
            lmd_orders = lmd_trade_raw

        daily_orundum = orundum_orders * ORUNDUM_PER_ORDER
        daily_lmd = lmd_orders * LMD_PER_ORDER
        daily_cr = cr_output
        daily_puregold_net = gold_output - lmd_orders * GOLD_PER_ORDER

        return {
            "daily_orundum": round(daily_orundum),
            "daily_lmd": round(daily_lmd),
            "daily_combat_record": round(daily_cr),
            "daily_puregold_net": round(daily_puregold_net),
            # Raw breakdown for debugging / detail view
            "_stone_output": round(stone_output),
            "_gold_output": round(gold_output),
            "_orundum_orders": round(orundum_orders, 1),
            "_lmd_orders": round(lmd_orders, 1),
            "_stone_demand": round(stone_demand),
            "_gold_demand": round(gold_demand),
        }

    @staticmethod
    def _analyze_sustainability(
        solution: 'ParetoSolution',
        inventory: 'Inventory | None' = None,
        material_stock: 'MaterialStock | None' = None,
    ) -> dict:
        """Analyze whether a plan is a self-sustaining closed loop.

        Key costs of 搓玉 (orundum farming):
          - Each 源石碎片 costs 200 LMD + 2 固源岩 to manufacture
          - Each 合成玉 trade order consumes 2 源石碎片 → 20 合成玉
          - 固源岩 must be farmed from 1-7 (~4 sanity per rock)

        Returns sustainability verdict with detailed breakdown.
        """
        # Daily orundum trade orders
        daily_orundum_orders = solution.daily_orundum / 20.0 if solution.daily_orundum > 0 else 0
        # Stone consumption: 2 stones per orundum order
        daily_stone_consumed = daily_orundum_orders * 2
        # LMD cost: 200 LMD per stone crafted
        daily_lmd_cost = daily_stone_consumed * 200
        # Net LMD balance
        lmd_balance = solution.daily_lmd - daily_lmd_cost
        # Rock demand: 2 rocks per stone
        daily_rock_demand = daily_stone_consumed * 2
        # Sanity cost: ~4 sanity per rock at 1-7
        daily_sanity = daily_rock_demand * 4

        # Natural sanity regen: 240/day (1 per 6 min)
        NATURAL_SANITY = 240
        # Sustainable sanity threshold (with potions/weekly, ~300 is comfortable)
        SUSTAINABLE_SANITY = 300

        verdicts: list[str] = []
        detail_parts: list[str] = []

        if daily_orundum_orders > 0:
            detail_parts.append(
                f"日搓合成玉 **{solution.daily_orundum:.0f}**"
                f"（{daily_orundum_orders:.0f}单 × 20玉/单）"
            )
            detail_parts.append(
                f"消耗源石碎片 **{daily_stone_consumed:.0f}个/天**"
                f"（每单2个）"
            )

        if daily_lmd_cost > 0:
            detail_parts.append(
                f"搓石LMD成本 **{daily_lmd_cost:.0f}/天**"
                f"（{daily_stone_consumed:.0f}个 × 200龙门币/个）"
            )

        if solution.daily_lmd > 0:
            detail_parts.append(f"龙门币产出 **{solution.daily_lmd:.0f}/天**")

        detail_parts.append(f"**LMD净收支: {lmd_balance:+.0f}/天**")

        # ── Stockpile-aware adjustment ──────────────────────────
        # A plan with a daily LMD deficit may still be viable for weeks
        # if the player has a deep wallet.  Don't scare them unnecessarily.
        if material_stock and not material_stock.is_empty() and material_stock.lmd > 0:
            if lmd_balance < 0:
                days_until_broke = material_stock.lmd / abs(lmd_balance)
                if days_until_broke >= 30:
                    # Deep pockets — downgrade from "deficit" to "ok" for verdict
                    detail_parts.append(
                        f"💡 你仓库有 **{material_stock.lmd/10000:.1f}万** 龙门币，"
                        f"按当前赤字可撑 **{days_until_broke:.0f}天**，短期无虞"
                    )
                    # Remove lmd_deficit from verdicts if present
                    if "lmd_deficit" in verdicts:
                        verdicts.remove("lmd_deficit")
                elif days_until_broke >= 7:
                    detail_parts.append(
                        f"💡 你仓库有 **{material_stock.lmd/10000:.1f}万** 龙门币，"
                        f"可撑 **{days_until_broke:.0f}天**，但需提前准备"
                    )
                else:
                    detail_parts.append(
                        f"🔴 你仓库仅 **{material_stock.lmd/10000:.1f}万** 龙门币，"
                        f"按当前赤字仅能撑 **{days_until_broke:.0f}天**！"
                    )

        # Adjust LMD deficit verdict severity based on known wallet
        if material_stock and not material_stock.is_empty() and material_stock.lmd > 0:
            known_lmd = material_stock.lmd
        elif inventory and not inventory.is_empty() and inventory.lmd > 0:
            known_lmd = inventory.lmd
        else:
            known_lmd = 0

        if lmd_balance < -5000:
            if known_lmd > 0:
                days = known_lmd / abs(lmd_balance)
                if days < 3:
                    verdicts.append("lmd_deficit")
                    detail_parts.append(
                        f"⚠️ 每天净亏 **{abs(lmd_balance):.0f}** 龙门币，"
                        f"仅能撑 **{days:.0f}天**！你的钱包会被搓玉榨干！"
                    )
                # else: stockpile is enough, already handled above
            else:
                verdicts.append("lmd_deficit")
                detail_parts.append(
                    f"⚠️ 每天净亏 **{abs(lmd_balance):.0f}** 龙门币，"
                    f"约 **{abs(lmd_balance)/10000:.1f}万/天**。"
                    f"你的钱包会被搓玉榨干！"
                )
        elif lmd_balance < 0 and known_lmd == 0:
            verdicts.append("lmd_deficit")
            detail_parts.append(
                f"⚠️ 每天净亏 {abs(lmd_balance):.0f} 龙门币，小亏但可接受"
            )
        elif lmd_balance >= 0 and "lmd_deficit" not in verdicts:
            pass  # positive balance is handled by the ok case below
        elif lmd_balance >= 0:
            detail_parts.append(
                f"✅ 龙门币净盈余 **+{lmd_balance:.0f}/天**，收支平衡"
            )

        if daily_rock_demand > 0:
            detail_parts.append(
                f"固源岩需求 **{daily_rock_demand:.0f}个/天**"
                f" → 1-7刷石理智: **{daily_sanity:.0f}/天**"
            )

        if daily_sanity > SUSTAINABLE_SANITY:
            verdicts.append("sanity_impossible")
            detail_parts.append(
                f"⚠️ 需 **{daily_sanity:.0f} 理智/天**刷1-7，"
                f"但自然回复仅 **{NATURAL_SANITY}/天**。"
                f"每天缺口 {daily_sanity - NATURAL_SANITY:.0f} 理智，"
                f"需要大量碎石/理智药才能维持"
            )
        elif daily_sanity > 200:
            detail_parts.append(
                f"⚠️ 理智压力较大（{daily_sanity:.0f}/天），"
                f"建议配合每周理智药使用"
            )
        elif daily_sanity > 0:
            detail_parts.append(
                f"✅ 理智需求 {daily_sanity:.0f}/天在自然回复范围内"
            )

        # Daily sanity budget analysis
        if daily_sanity > 0:
            sanity_remaining = max(0, NATURAL_SANITY - daily_sanity)
            detail_parts.append(
                f"刷完1-7后剩余 **{sanity_remaining:.0f} 理智/天**"
                f"可用于其他关卡"
            )

        # Final verdict
        if "sanity_impossible" in verdicts and "lmd_deficit" in verdicts:
            sustain_verdict = "both"
            detail_parts.insert(0, "## 🔴 方案不可持续 — LMD亏空 + 理智不足")
        elif "sanity_impossible" in verdicts:
            sustain_verdict = "sanity_impossible"
            detail_parts.insert(0, "## 🟡 方案理智压力大 — 需大量刷1-7")
        elif "lmd_deficit" in verdicts:
            sustain_verdict = "lmd_deficit"
            detail_parts.insert(0, "## 🔴 方案不可持续 — 龙门币净亏空")
        elif daily_orundum_orders > 0:
            sustain_verdict = "ok"
            detail_parts.insert(0, "## ✅ 方案可闭环循环")
        else:
            sustain_verdict = "ok"
            # No orundum → no craft costs → always sustainable for LMD/CR

        return {
            "verdict": sustain_verdict,
            "lmd_balance": round(lmd_balance),
            "rock_demand": round(daily_rock_demand),
            "sanity_cost": round(daily_sanity),
            "detail": "\n".join(f"- {p}" if not p.startswith("##") else p for p in detail_parts),
        }

    @staticmethod
    def _default_weights() -> dict:
        """Default room weights for solving."""
        return {
            "Trade:Orundum": 10.0,
            "Trade:LMD": 5.0,
            "Mfg:PureGold": 5.0,
            "Mfg:CombatRecord": 4.0,
            "Mfg:OriginStone": 3.0,
            "Power:Drone": 1.0,
            "Control:Control": 2.0,
            "Office:Office": 1.0,
            "Reception:Reception": 1.5,
        }

    @staticmethod
    def _inventory_weight_adjustment(
        inventory: 'Inventory | None',
        material_stock: 'MaterialStock | None',
    ) -> dict[str, float]:
        """Compute per-product weight multipliers from warehouse stock.

        Low stock → boost (×1.3-2.0). High stock → reduce (×0.5-0.8).
        No stock data → neutral (×1.0).

        This nudges the Pareto frontier toward products the player
        actually needs instead of blindly optimizing output.
        """
        adj: dict[str, float] = {
            "Trade:Orundum": 1.0, "Trade:LMD": 1.0,
            "Mfg:PureGold": 1.0, "Mfg:CombatRecord": 1.0,
            "Mfg:OriginStone": 1.0, "Power:Drone": 1.0,
            "Control:Control": 1.0, "Office:Office": 1.0,
            "Reception:Reception": 1.0,
        }
        # ── From old Inventory (赤金/源石碎片 counts) ──
        if inventory and not inventory.is_empty():
            # PureGold: if stock > 2 days (50/day), reduce weight
            if inventory.puregold >= 150:
                adj["Mfg:PureGold"] = 0.5
            elif inventory.puregold >= 80:
                adj["Mfg:PureGold"] = 0.7
            # OriginStone: if stock low, boost strongly
            if inventory.origin_stone <= 10:
                adj["Mfg:OriginStone"] = 2.0
            elif inventory.origin_stone <= 30:
                adj["Mfg:OriginStone"] = 1.5
            # If player has lots of Orundum, they may not need to craft more
            if inventory.orundum >= 300:
                adj["Trade:Orundum"] = 0.8

        # ── From MaterialStock (full depot scan) ──
        if material_stock and not material_stock.is_empty():
            # If LMD is abundant, deprioritize Trade:LMD
            if material_stock.lmd >= 500000:
                adj["Trade:LMD"] = 0.5
                adj["Mfg:PureGold"] = 0.5  # gold feeds LMD
            elif material_stock.lmd >= 200000:
                adj["Trade:LMD"] = 0.7
            elif material_stock.lmd <= 10000:
                adj["Trade:LMD"] = 1.5
                adj["Mfg:PureGold"] = 1.5

            # Check specific materials used in production chains
            _surplus = lambda name, threshold: material_stock.has(name, threshold)

            # If player has tons of devices (装置) → OriginStone chain pushes up
            if _surplus("装置", 20):
                adj["Mfg:OriginStone"] *= 1.2

            # If chip stock is healthy, all elite upgrades are feasible
            # → more operators can unlock skills → all production benefits
            chip_count = sum(
                material_stock.items.get(n, 0) for n in material_stock.items
            )
            if chip_count >= 30:
                adj["Trade:Orundum"] = 1.0  # neutral, let player choose

            # Low combat record stock → prioritize Mfg:CombatRecord
            if material_stock.has("作战记录", 20) or material_stock.has("战术演习券", 5):
                adj["Mfg:CombatRecord"] = 0.7  # already have some
            else:
                adj["Mfg:CombatRecord"] = max(adj["Mfg:CombatRecord"], 1.2)

        # Clamp all to reasonable range
        return {k: max(0.3, min(3.0, v)) for k, v in adj.items()}

    @staticmethod
    def _inventory_sort_adjustment(
        inventory: 'Inventory | None',
        material_stock: 'MaterialStock | None',
    ) -> tuple[float, float, float]:
        """Adjust Pareto sort weights (orundum, lmd, combat_record) by inventory.

        Returns multipliers for (orundum_w, lmd_w, combat_record_w).
        Default (1.0, 1.0, 1.0) = no change to user's goal.
        Surplus gold → boost orundum & combat record weight.
        Low LMD → boost LMD weight.
        """
        o_mul, l_mul, c_mul = 1.0, 1.0, 1.0

        if inventory and not inventory.is_empty():
            # Rich in puregold → deprioritize gold chain → boost other outputs
            if inventory.puregold >= 200:
                l_mul = 0.4   # LMD less important when gold is abundant
                o_mul = 1.5   # shift focus to Orundum
                c_mul = 1.3
            elif inventory.puregold >= 100:
                l_mul = 0.6
                o_mul = 1.3

            # Low stone → boost orundum (stone feeds orundum chain)
            if inventory.origin_stone <= 10:
                o_mul *= 1.5

            # High orundum → let other goals compete
            if inventory.orundum >= 300:
                o_mul = 0.8

        if material_stock and not material_stock.is_empty():
            # Low LMD → boost LMD weight
            if material_stock.lmd <= 10000:
                l_mul = 2.0
                c_mul = 0.7  # experience can wait
            elif material_stock.lmd >= 500000:
                l_mul = 0.3

        return (
            max(0.3, min(3.0, o_mul)),
            max(0.3, min(3.0, l_mul)),
            max(0.3, min(3.0, c_mul)),
        )

    @staticmethod
    def check_resource_balance(
        solution,
        inventory=None,
        depot_stock: Any = None,
    ) -> dict:
        """Verify gold/stone production can sustain trade consumption.

        Uses depot_stock (MAA scan) as primary data source when available,
        falling back to old-format inventory.
        """
        if inventory is None:
            inventory = Inventory()
        # ── Get puregold / origin_stone from best data source ──
        _inv_puregold = inventory.puregold
        _inv_origin_stone = inventory.origin_stone
        if depot_stock is not None and not (hasattr(depot_stock, 'is_empty') and depot_stock.is_empty()):
            # MAA scan is primary — use get_any for robust name matching
            if _inv_puregold == 0:
                pg = depot_stock.get_any("赤金") if hasattr(depot_stock, 'get_any') else 0
                if pg > 0:
                    _inv_puregold = pg
            if _inv_origin_stone == 0:
                os_ = depot_stock.get_any("源石碎片") if hasattr(depot_stock, 'get_any') else 0
                if os_ > 0:
                    _inv_origin_stone = os_

        if not solution or not solution.shifts or not solution.shifts[0].rooms:
            return {"feasible": True, "warnings": []}
        shift = solution.shifts[0]
        num_lmd = sum(1 for r in shift.rooms if r.facility == "Trade" and r.product == "LMD")
        num_pg = sum(1 for r in shift.rooms if r.facility == "Mfg" and r.product == "PureGold")
        num_od = sum(1 for r in shift.rooms if r.facility == "Trade" and r.product == "Orundum")
        num_os = sum(1 for r in shift.rooms if r.facility == "Mfg" and r.product == "OriginStone")
        tpe = tle = toe = tse = 0.0
        ns = len(solution.shifts)
        for s in solution.shifts:
            for r in s.rooms:
                e = r.total_efficiency()
                if r.facility == "Mfg" and r.product == "PureGold": tpe += e
                elif r.facility == "Mfg" and r.product == "OriginStone": tse += e
                elif r.facility == "Trade" and r.product == "LMD": tle += e
                elif r.facility == "Trade" and r.product == "Orundum": toe += e
        ape = tpe / max(1, num_pg * ns) if num_pg > 0 else 0
        ale = tle / max(1, num_lmd * ns) if num_lmd > 0 else 0
        aoe = toe / max(1, num_od * ns) if num_od > 0 else 0
        ase = tse / max(1, num_os * ns) if num_os > 0 else 0
        BD = 20.0
        gp = num_pg * (1.0 + ape / 100.0) * BD
        gd = num_lmd * (1.0 + ale / 100.0) * BD
        sp = num_os * (1.0 + ase / 100.0) * BD
        sd = num_od * (1.0 + aoe / 100.0) * BD * 2
        rg = gp / max(gd, 0.01); rs = sp / max(sd, 0.01)
        gdd = max(0.0, gd - gp); sdd = max(0.0, sd - sp)
        gsd = _inv_puregold / gdd if gdd > 0 else float("inf")
        ssd = _inv_origin_stone / sdd if sdd > 0 else float("inf")
        warnings = []; GD = 7
        if num_lmd > 0 and num_pg == 0:
            if _inv_puregold == 0:
                warnings.append("配置有LMD贸易站但无赤金制造站，且无赤金库存。建议配赤金制造站或改用合成玉贸易。")
            elif gsd < GD:
                warnings.append(f"无赤金制造站，库存{_inv_puregold}个赤金仅能支撑约{gsd:.0f}天LMD贸易。")
        elif num_pg > 0 and rg < 0.5 and gsd < 3:
            # Only warn if severely under-produced AND short stockpile (<3 days)
            si = f"库存{_inv_puregold}个赤金可撑{gsd:.0f}天" if _inv_puregold > 0 else "无赤金库存"
            warnings.append(f"赤金严重不足（产出/消耗={rg:.1%}），{si}。")
        if num_od > 0 and num_os == 0:
            # No stone production at all but orundum trade — needs stockpile
            if _inv_origin_stone == 0 and ssd < GD:
                warnings.append("配置有合成玉贸易但无源石碎片制造站，且无库存。")
            elif ssd < 3:
                warnings.append(f"无源石碎片制造站，库存{_inv_origin_stone}个碎片仅能支撑约{ssd:.0f}天。")
        elif num_os > 0 and rs < 0.5 and ssd < 3:
            # Only warn if severely under-produced AND no stockpile to cover
            si = f"库存{_inv_origin_stone}个碎片可撑{ssd:.0f}天" if _inv_origin_stone > 0 else "无源石碎片库存"
            warnings.append(f"源石碎片严重不足（产出/消耗={rs:.1%}），{si}。")
        return {
            "feasible": len(warnings) == 0,
            "gold_production": round(gp, 1), "gold_demand": round(gd, 1),
            "gold_ratio": round(rg, 2), "gold_daily_deficit": round(gdd, 1),
            "gold_stockpile_days": round(gsd, 1) if gsd != float("inf") else None,
            "stone_production": round(sp, 1), "stone_demand": round(sd, 1),
            "stone_ratio": round(rs, 2), "stone_daily_deficit": round(sdd, 1),
            "stone_stockpile_days": round(ssd, 1) if ssd != float("inf") else None,
            "warnings": warnings,
        }

    @staticmethod
    def analyze_morale_sustainability(
        shifts: list,
        operators: list,
    ) -> dict:
        """Check shift sustainability based on morale drain rates."""
        if not shifts:
            return {"sustainable": True, "risks": []}
        sh = shifts[0].duration_hours
        risks = []
        used = {}
        for s in shifts:
            for r in s.rooms:
                for op in r.operators:
                    if op.name not in used: used[op.name] = []
                    used[op.name].append(s.name)
                    d = op.morale_drain_per_hour(r.facility)
                    if d * sh > 12:
                        risks.append(f"{op.name}心情消耗较快（{d:.1f}/h），{sh:.0f}h消耗{d*sh:.0f}")
        ms = {n: ss for n, ss in used.items() if len(ss) > 1}
        if ms:
            for n, sl in ms.items():
                risks.append(f"{n}被分配到多个班次（{'、'.join(sl)}），心情可能不足")
        return {
            "sustainable": len(risks) == 0,
            "total_operators_used": len(used),
            "multi_shift_violations": len(ms),
            "risks": risks[:8],
        }


def discover_dynamic_teams(
    operators: list,
    box: dict,
    min_efficiency: float = 40.0,
) -> list:
    """Auto-discover operator teams by scanning for combo interactions."""
    discovered = []
    op_map = {op.name: op for op in operators}
    op_names = set(box.keys())
    for op in operators:
        for combo in op.active_combos("Trade") + op.active_combos("Mfg"):
            if combo.partner and combo.partner in op_names:
                po = op_map.get(combo.partner)
                if not po: continue
                fac = ""
                for s in op.skills:
                    if s.combo and s.combo.partner == combo.partner:
                        fac = s.facility; break
                if not fac: continue
                best = 0.0
                for p in FACILITY_PRODUCTS.get(fac, []):
                    ce = op.efficiency_for(p, fac) + po.efficiency_for(p, fac) + sum(combo.partner_bonus.get(k, 0) for k in (p, "all"))
                    if ce > best: best = ce
                if best >= min_efficiency:
                    discovered.append(TeamTemplate(
                        name=f"{op.name}+{po.name}(auto)", facility=fac,
                        members=[op.name, po.name], equiv_eff=best,
                        tier="auto", desc=f"自动发现: {op.name}的combo触发{po.name}",
                    ))
    for op in operators:
        for combo in op.active_combos("Trade") + op.active_combos("Mfg"):
            if combo.per_other_op > 0:
                fac = ""
                for s in op.skills:
                    if s.combo and s.combo.per_other_op > 0:
                        fac = s.facility; break
                if not fac: continue
                comps = sorted(
                    [o for o in operators if o.name != op.name],
                    key=lambda o: sum(o.efficiency_for(p, fac) for p in FACILITY_PRODUCTS.get(fac, [])),
                    reverse=True,
                )[:2]
                if len(comps) >= 2:
                    total = max(
                        op.efficiency_for(p, fac) + combo.per_other_op * 2
                        + sum(c.efficiency_for(p, fac) for c in comps)
                        for p in FACILITY_PRODUCTS.get(fac, ["PureGold"])
                    )
                    if total >= min_efficiency:
                        discovered.append(TeamTemplate(
                            name=f"{op.name}+2(auto)", facility=fac,
                            members=[op.name, comps[0].name, comps[1].name],
                            equiv_eff=total, tier="auto",
                            desc=f"自动发现: {op.name}每队友+{combo.per_other_op}%",
                        ))
    return discovered
