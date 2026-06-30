"""Test base scheduling with real box: multi-shift + combo effects."""

import json, sys, os
from pathlib import Path
from collections import Counter

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge import KnowledgeBase
from src.intelligence.arknights.base_optimizer import BaseOptimizer

BOX_PATH = Path(__file__).parent.parent / "data" / "session"
# Find most recent session box.json
sessions = sorted(BOX_PATH.glob("base_*/box.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not sessions:
    print("ERROR: No session box found. Run scan_operator_box first.")
    sys.exit(1)
BOX_PATH = sessions[0]
print(f"Using box: {BOX_PATH}")
with open(BOX_PATH, "r", encoding="utf-8") as f:
    box_raw = json.load(f)

operator_box = {op["name"]: op["elite"] for op in box_raw["operators"]}
print(f"BOX: {len(operator_box)} operators (E2={box_raw['summary']['E2']}, E1={box_raw['summary']['E1']}, E0={box_raw['summary']['E0']})")

kb = KnowledgeBase()
optimizer = BaseOptimizer(knowledge_base=kb)

# Combo check
ops = optimizer._resolve_operators(operator_box)
combo_ops = [(op.name, s) for op in ops for s in op.skills if s.combo and s.combo.is_active]
print(f"COMBO: {len(combo_ops)} combo skills found in box")
for name, s in combo_ops:
    c = s.combo
    parts = []
    if c.partner: parts.append(f"+{c.partner_bonus} with {c.partner}")
    if c.per_other_op > 0: parts.append(f"+{c.per_other_op}%/other")
    if c.faction: parts.append(f"+{c.faction_bonus}/per {c.faction}")
    print(f"  {name} [{s.facility}] {s.name}: {', '.join(parts)}")

GOALS = [
    ("全力搓玉", (0.70, 0.20, 0.10)),
    ("搓玉+练级", (0.50, 0.30, 0.20)),
    ("均衡发展", (0.30, 0.40, 0.30)),
    ("最大化龙门币", (0.05, 0.80, 0.15)),
    ("最大化作战记录", (0.05, 0.15, 0.80)),
]

for goal_desc, weights in GOALS:
    print(f"\n{'='*70}")
    print(f"GOAL: {goal_desc} | weights={weights}")
    print(f"{'='*70}")

    frontier = optimizer.solve_pareto(operator_box, layout="243", num_shifts=0, sort_weights=weights)
    if not frontier:
        print("  NO SOLUTION")
        continue

    best = optimizer.solve_with_weights(frontier, *weights)
    if not best:
        print("  NO BEST")
        continue

    ns = len(best.shifts)
    total_ops = len({op.name for s in best.shifts for r in s.rooms for op in r.operators})
    print(f"  Frontier: {len(frontier)} points | Shifts: {ns} | Operators used: {total_ops}")
    print(f"  Config: ", end="")
    fc = Counter()
    for f, p in best.config.rooms:
        fc.setdefault(f, Counter())
        fc[f][p] += 1
    parts = []
    for fac in sorted(fc):
        items = [f"{p}x{c}" for p, c in sorted(fc[fac].items())]
        parts.append(f"{fac}({'+'.join(items)})")
    print(" ".join(parts))
    print(f"  Orundum={best.orundum_eff:.0%} LMD={best.lmd_eff:.0%} CR={best.combat_record_eff:.0%} Coverage={best.coverage:.0%}")

    for shift in best.shifts:
        print(f"\n  -- {shift.name} ({shift.duration_hours:.0f}h) --")
        for room in shift.rooms:
            ops_str = "、".join(op.name for op in room.operators) if room.operators else "-"
            base = sum(op.efficiency_for(room.product, room.facility) for op in room.operators)
            total = room.total_efficiency()
            combo = total - base
            combo_str = f" [+{combo:.0f}]" if combo > 0 else ""
            print(f"    {room.facility}{room.index+1} {room.product:<14} {ops_str:<30} {total:.0f}%{combo_str}")

# Layout comparison
print(f"\n{'='*70}")
print(f"LAYOUT COMPARISON (goal: 搓玉为主)")
print(f"{'='*70}")
for layout in ["243", "333", "252", "153"]:
    frontier = optimizer.solve_pareto(operator_box, layout=layout, num_shifts=0, sort_weights=(0.60, 0.25, 0.15))
    if not frontier:
        print(f"  {layout}: NO SOLUTION")
        continue
    best = optimizer.solve_with_weights(frontier, 0.60, 0.25, 0.15)
    if best:
        ns = len(best.shifts)
        total_ops = len({op.name for s in best.shifts for r in s.rooms for op in r.operators})
        print(f"  {layout}: frontier={len(frontier)} shifts={ns} ops={total_ops} O={best.orundum_eff:.0%} L={best.lmd_eff:.0%} C={best.combat_record_eff:.0%}")

print("\nDONE!")
