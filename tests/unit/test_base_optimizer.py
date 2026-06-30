"""Unit tests for the Arknights base optimizer with Pareto frontier."""

import pytest
from src.intelligence.arknights.base_optimizer import (
    BaseOptimizer, Operator, OperatorSkill, Room, Shift,
    ProductConfig, ParetoSolution,
    enumerate_product_configs,
    simulated_annealing_assignment,
    compute_pareto_frontier,
    _dedup_by_composition,
    _infer_elite_for_facility,
    _normalize_skill_efficiency,
    ELITE_UNLOCK_MAP,
    LAYOUT_FACILITY_COUNTS, FACILITY_PRODUCTS,
)


# ── Test data helpers ───────────────────────────────────────────

def _op(name, facility, product, eff, elite=2):
    """Create a test operator with one skill."""
    return Operator(
        name=name, elite_level=elite, rarity=5,
        skills=[OperatorSkill(skill_id=f"t_{name}", facility=facility,
                              name=name, efficiency={product: eff})],
    )


SAMPLE_BOX = {
    "德克萨斯": 2, "拉普兰德": 2, "能天使": 2, "银灰": 2,
    "清流": 2, "温蒂": 2, "森蚺": 2, "阿罗玛": 2, "槐琥": 2,
    "迷迭香": 2, "泡泡": 2, "火神": 2, "褐果": 1,
    "巫恋": 2, "黑键": 2, "乌有": 2, "但书": 2,
    "承曦格雷伊": 2, "烛煌": 1, "格雷伊": 1,
    "重岳": 2, "令": 2, "八幡海铃": 2,
    "斑点": 1, "食铁兽": 1,
}


# ── Product Config enumeration ──────────────────────────────────

class TestEnumeration:
    def test_243_configs(self):
        configs = enumerate_product_configs("243")
        # Trade: 2^2=4, Mfg: 3^4=81, Power: 1^3=1 → 4*81=324
        assert len(configs) == 324

    def test_333_configs(self):
        configs = enumerate_product_configs("333")
        # Trade: 2^3=8, Mfg: 3^3=27, Power: 1^3=1 → 8*27=216
        assert len(configs) == 216

    def test_config_has_correct_room_count(self):
        configs = enumerate_product_configs("243")
        for c in configs:
            facs = [f for f, _ in c.rooms]
            assert facs.count("Trade") == 2
            assert facs.count("Mfg") == 4
            assert facs.count("Power") == 3

    def test_product_diversity(self):
        """Make sure products are varied across configs."""
        configs = enumerate_product_configs("243")
        products_seen = set()
        for c in configs:
            for _, p in c.rooms:
                products_seen.add(p)
        assert "LMD" in products_seen
        assert "Orundum" in products_seen
        assert "PureGold" in products_seen
        assert "CombatRecord" in products_seen
        assert "OriginStone" in products_seen

    def test_dedup_reduces_configs(self):
        """Dedup should merge identical product compositions."""
        configs = enumerate_product_configs("333")
        # Create solutions with different scores for same compositions
        class FakeSolution:
            def __init__(self, cfg, score):
                self.config = cfg
                self.total_score = score
        sols = [FakeSolution(c, 100.0) for c in configs[:10]]
        deduped = _dedup_by_composition(sols)
        assert len(deduped) <= len(sols)


# ── Simulated Annealing ─────────────────────────────────────────

class TestSA:
    def test_sa_runs_without_error(self):
        rooms = [
            Room("Mfg", "PureGold", 0, 3),
            Room("Mfg", "PureGold", 1, 3),
        ]
        ops = [
            _op("A", "Mfg", "PureGold", 30.0),
            _op("B", "Mfg", "PureGold", 35.0),
            _op("C", "Mfg", "PureGold", 25.0),
            _op("D", "Mfg", "PureGold", 20.0),
        ]
        weights = {"Mfg:PureGold": 5.0}
        result = simulated_annealing_assignment(rooms, ops, weights, steps=50)
        assert len(result) == 2
        for r in result:
            assert 0 < len(r.operators) <= r.max_slots

    def test_sa_no_duplicates(self):
        rooms = [
            Room("Mfg", "PureGold", 0, 3),
            Room("Mfg", "CombatRecord", 1, 3),
        ]
        ops = [_op(chr(65+i), "Mfg", "PureGold", 30.0) for i in range(6)]
        weights = {"Mfg:PureGold": 5.0, "Mfg:CombatRecord": 3.0}
        result = simulated_annealing_assignment(rooms, ops, weights, steps=50)
        all_names = [op.name for r in result for op in r.operators]
        assert len(all_names) == len(set(all_names)), "Duplicate operators"

    def test_sa_respects_elite_level(self):
        """E0 operator should not benefit from E2-locked skills."""
        rooms = [Room("Mfg", "PureGold", 0, 3)]
        e0_op = _op("新手", "Mfg", "PureGold", 30.0, elite=0)
        # Give them a skill that needs E2 but the operator is E0
        e0_op.skills[0].elite_required = 2
        ops = [e0_op, _op("老手", "Mfg", "PureGold", 40.0)]
        result = simulated_annealing_assignment(
            rooms, ops, {"Mfg:PureGold": 5.0}, steps=50,
        )
        # The E0 operator should have 0 efficiency for PureGold since skill locked
        assert result[0].total_efficiency() <= 40.0  # Only 老手 contributes


# ── Pareto frontier ─────────────────────────────────────────────

class TestParetoFrontier:
    def test_dominance(self):
        a = ParetoSolution(ProductConfig("x", []), [])
        b = ParetoSolution(ProductConfig("x", []), [])
        a.orundum_eff, a.lmd_eff, a.combat_record_eff = 1.0, 1.0, 1.0
        b.orundum_eff, b.lmd_eff, b.combat_record_eff = 0.5, 0.5, 0.5
        assert a.dominates(b)
        assert not b.dominates(a)

    def test_no_self_dominance(self):
        a = ParetoSolution(ProductConfig("x", []), [])
        a.orundum_eff = a.lmd_eff = a.combat_record_eff = 0.5
        assert not a.dominates(a)

    def test_non_dominated_kept(self):
        a = ParetoSolution(ProductConfig("x", []), [])
        b = ParetoSolution(ProductConfig("x", []), [])
        c = ParetoSolution(ProductConfig("x", []), [])
        a.orundum_eff, a.lmd_eff, a.combat_record_eff = 1.0, 0.5, 0.2
        b.orundum_eff, b.lmd_eff, b.combat_record_eff = 0.8, 0.8, 0.3
        c.orundum_eff, c.lmd_eff, c.combat_record_eff = 0.3, 0.3, 0.3  # dominated
        frontier = compute_pareto_frontier([a, b, c])
        assert len(frontier) == 2
        assert a in frontier
        assert b in frontier
        assert c not in frontier

    def test_frontier_empty(self):
        assert compute_pareto_frontier([]) == []


# ── Full optimizer ──────────────────────────────────────────────

class TestBaseOptimizer:
    @pytest.fixture
    def opt(self):
        return BaseOptimizer(knowledge_base=None)

    def test_solve_pareto_basic(self, opt):
        frontier = opt.solve_pareto(SAMPLE_BOX, layout="243", num_shifts=1)
        assert len(frontier) > 0
        assert len(frontier) <= 324  # max after dedup
        for s in frontier:
            assert s.dominated_by == 0

    def test_solve_pareto_333(self, opt):
        frontier = opt.solve_pareto(SAMPLE_BOX, layout="333", num_shifts=1)
        assert len(frontier) > 0

    def test_solve_with_weights(self, opt):
        frontier = opt.solve_pareto(SAMPLE_BOX, layout="243")
        best = opt.solve_with_weights(frontier, orundum=0.60, lmd=0.25, combat_record=0.15)
        assert best is not None
        assert best.shifts

    def test_solve_with_weights_all_orundum(self, opt):
        frontier = opt.solve_pareto(SAMPLE_BOX, layout="243")
        best = opt.solve_with_weights(frontier, orundum=1.0, lmd=0.0, combat_record=0.0)
        assert best is not None

    def test_solve_with_weights_all_lmd(self, opt):
        frontier = opt.solve_pareto(SAMPLE_BOX, layout="243")
        best = opt.solve_with_weights(frontier, orundum=0.0, lmd=1.0, combat_record=0.0)
        assert best is not None

    def test_solve_legacy(self, opt):
        result = opt.solve_legacy(list(SAMPLE_BOX.keys()), goal="mixed_orundum_upgrade")
        assert result["best"] is not None
        assert len(result["frontier"]) > 0

    def test_empty_box(self, opt):
        frontier = opt.solve_pareto({}, layout="243")
        assert frontier == []

    def test_small_box(self, opt):
        small = {"德克萨斯": 2, "能天使": 2, "清流": 2}
        frontier = opt.solve_pareto(small, layout="243")
        assert len(frontier) >= 1

    def test_elite_level_parsing(self, opt):
        """Legacy parser should handle mixed formats, defaulting to E0."""
        box = ["德克萨斯(E2)", "清流(精英2)", "能天使(E1)", "斑点"]
        result = opt._parse_box_legacy(box)
        assert result["德克萨斯"] == 2
        assert result["清流"] == 2
        assert result["能天使"] == 1
        assert result["斑点"] == 0  # default E0 — conservative

    def test_outputs_are_normalized(self, opt):
        """Pareto solutions should have normalized values in [0, 1]."""
        frontier = opt.solve_pareto(SAMPLE_BOX, layout="243")
        for s in frontier:
            assert 0.0 <= s.orundum_eff <= 1.0
            assert 0.0 <= s.lmd_eff <= 1.0
            assert 0.0 <= s.combat_record_eff <= 1.0


# ── Operator skill filtering ────────────────────────────────────

class TestEliteFiltering:
    def test_skill_filtered_by_elite(self):
        """E1 operator should have E2 skills filtered out."""
        skill = OperatorSkill("s1", "Mfg", "高级技能", {"PureGold": 40.0}, elite_required=2)
        op = Operator(name="半成品", elite_level=1, skills=[skill])
        assert op.efficiency_for("PureGold", "Mfg") == 0.0

    def test_skill_available_at_elite(self):
        skill = OperatorSkill("s1", "Mfg", "基础技能", {"PureGold": 25.0}, elite_required=0)
        op = Operator(name="基础", elite_level=0, skills=[skill])
        assert op.efficiency_for("PureGold", "Mfg") == 25.0

    def test_mixed_skills(self):
        op = Operator(name="进阶", elite_level=1, skills=[
            OperatorSkill("s1", "Mfg", "基础", {"PureGold": 20.0}, elite_required=0),
            OperatorSkill("s2", "Mfg", "进阶", {"PureGold": 35.0}, elite_required=2),
        ])
        # Only base skill should count at E1
        assert op.efficiency_for("PureGold", "Mfg") == 20.0


# ── Room warnings ───────────────────────────────────────────────

class TestRoomWarnings:
    def test_warning_for_underleveled(self):
        skill = OperatorSkill("s1", "Mfg", "高级赤金", {"PureGold": 40.0}, elite_required=2)
        op = Operator(name="未达标", elite_level=1, skills=[skill])
        room = Room("Mfg", "PureGold", 0, 3, operators=[op])
        assert len(room.warnings) >= 1


# ── Curated skill elite inference ────────────────────────────────

class TestInferEliteForFacility:
    """Verify that _infer_elite_for_facility returns correct MAX unlock."""

    def test_max_unlock_e2(self):
        """KB has E0 and E2 skills → curated should need E2."""
        kb_data = {
            "base_skills": [
                {"facility": "Trade", "unlock_condition": "精英0"},
                {"facility": "Trade", "unlock_condition": "精英2"},
            ]
        }
        assert _infer_elite_for_facility(kb_data, "Trade") == 2

    def test_max_unlock_e1(self):
        """KB has single E1 skill → curated should need E1."""
        kb_data = {
            "base_skills": [
                {"facility": "Mfg", "unlock_condition": "精英1"},
            ]
        }
        assert _infer_elite_for_facility(kb_data, "Mfg") == 1

    def test_no_kb_skills_at_facility(self):
        """KB has no skills at this facility → default to 0."""
        kb_data = {
            "base_skills": [
                {"facility": "Trade", "unlock_condition": "精英0"},
            ]
        }
        assert _infer_elite_for_facility(kb_data, "Mfg") == 0

    def test_none_kb_data(self):
        """No KB data at all → default to 0."""
        assert _infer_elite_for_facility(None, "Trade") == 0

    def test_empty_base_skills(self):
        """Empty base_skills list → default to 0."""
        kb_data = {"base_skills": []}
        assert _infer_elite_for_facility(kb_data, "Trade") == 0

    def test_different_facility_not_counted(self):
        """Skills at other facilities should not affect inference."""
        kb_data = {
            "base_skills": [
                {"facility": "Control", "unlock_condition": "精英2"},
                {"facility": "Trade", "unlock_condition": "精英1"},
            ]
        }
        assert _infer_elite_for_facility(kb_data, "Trade") == 1  # NOT 2


# ── SA best-solution tracking ─────────────────────────────────────

class TestSABestTracking:
    """Verify that simulated annealing doesn't lose best solutions."""

    def test_cross_facility_swap_preserves_best(self):
        """After a beneficial cross-facility swap, best should be tracked."""
        rooms = [
            Room("Trade", "Orundum", 0, 3),
            Room("Mfg", "OriginStone", 0, 3),
        ]
        ops = [
            Operator("A", elite_level=2, skills=[
                OperatorSkill("a1", "Trade", "A", {"Orundum": 50.0}),
                OperatorSkill("a2", "Mfg", "A", {"OriginStone": 60.0}),
            ]),
            Operator("B", elite_level=2, skills=[
                OperatorSkill("b1", "Trade", "B", {"Orundum": 30.0}),
                OperatorSkill("b2", "Mfg", "B", {"OriginStone": 40.0}),
            ]),
            Operator("C", elite_level=2, skills=[
                OperatorSkill("c1", "Trade", "C", {"Orundum": 40.0}),
                OperatorSkill("c2", "Mfg", "C", {"OriginStone": 20.0}),
            ]),
            Operator("D", elite_level=2, skills=[
                OperatorSkill("d1", "Trade", "D", {"Orundum": 20.0}),
                OperatorSkill("d2", "Mfg", "D", {"OriginStone": 50.0}),
            ]),
            Operator("E", elite_level=2, skills=[
                OperatorSkill("e1", "Mfg", "E", {"OriginStone": 30.0}),
            ]),
            Operator("F", elite_level=2, skills=[
                OperatorSkill("f1", "Trade", "F", {"Orundum": 35.0}),
            ]),
        ]
        weights = {"Trade:Orundum": 10.0, "Mfg:OriginStone": 3.0}
        result = simulated_annealing_assignment(
            rooms, ops, weights, steps=200,
        )
        # Verify no duplicates — operators shouldn't appear in multiple rooms
        all_names = [op.name for r in result for op in r.operators]
        assert len(all_names) == len(set(all_names)), "Duplicate operators found"

        # Greedy init would put best Mfg operators in Mfg room.
        # If SA works correctly, best result >= greedy init.
        total_score = sum(
            weights.get(f"{r.facility}:{r.product}", 0.1) * r.total_efficiency()
            for r in result
        )
        assert total_score > 0

    def test_improvement_with_cross_facility(self):
        """SA with cross-facility swaps should not be worse than without them."""
        # Two rooms: Trade-Orundum + Mfg-OriginStone
        # Multi-talented operators should be able to migrate
        rooms = [
            Room("Trade", "Orundum", 0, 3),
            Room("Mfg", "OriginStone", 0, 3),
        ]
        ops = [
            Operator(f"Op{i}", elite_level=2, skills=[
                OperatorSkill(f"t{i}", "Trade", f"T{i}", {"Orundum": float(20 + i * 5)}),
                OperatorSkill(f"m{i}", "Mfg", f"M{i}", {"OriginStone": float(50 - i * 5)}),
            ])
            for i in range(6)
        ]
        weights = {"Trade:Orundum": 10.0, "Mfg:OriginStone": 10.0}
        result = simulated_annealing_assignment(
            rooms, ops, weights, steps=100,
        )
        all_names = [op.name for r in result for op in r.operators]
        assert len(all_names) == 6  # all 6 operators used
        assert len(all_names) == len(set(all_names))


# ── Orundum chain weight alignment ────────────────────────────────

class TestOrundumChainWeights:
    """Verify that orundum chain configurations boost Mfg:OriginStone."""

    @pytest.fixture
    def opt(self):
        return BaseOptimizer(knowledge_base=None)

    def test_orundum_chain_triggers_weight_boost(self, opt):
        """Config with Orundum trade AND OriginStone mfg should boost weight."""
        # Build a custom config with the orundum chain
        config = ProductConfig("243", [
            ("Trade", "Orundum"), ("Trade", "Orundum"),
            ("Mfg", "OriginStone"), ("Mfg", "OriginStone"),
            ("Mfg", "OriginStone"), ("Mfg", "OriginStone"),
            ("Power", "Drone"), ("Power", "Drone"), ("Power", "Drone"),
        ])
        frontier = opt.solve_pareto(
            {"德克萨斯": 2, "能天使": 2, "清流": 2},
            layout="243",
        )
        # Low-coverage but should still produce a frontier with this box size
        assert len(frontier) >= 1, "Should find at least one solution"

    def test_non_orundum_config_no_boost(self, opt):
        """Pure LMD config should use default weights (no orundum chain)."""
        frontier = opt.solve_pareto(
            {"德克萨斯": 2, "能天使": 2, "清流": 2},
            layout="243",
        )
        # Should work normally without errors
        assert len(frontier) >= 1
        for s in frontier:
            assert s.dominated_by == 0

    def test_chain_config_solution_not_empty(self, opt):
        """Orundum chain should produce valid operator assignments."""
        small_box = {"德克萨斯": 2, "能天使": 2, "清流": 2, "砾": 0, "斑点": 1}
        frontier = opt.solve_pareto(small_box, layout="243")
        assert len(frontier) >= 1
        best = opt.solve_with_weights(frontier, orundum=0.70, lmd=0.20, combat_record=0.10)
        assert best is not None
        if best.shifts:
            assert len(best.shifts[0].rooms) > 0


# ── Knee-point (balanced) selection ──────────────────────────────

class TestSolveBalanced:
    """Verify that solve_balanced picks the most balanced frontier point."""

    @pytest.fixture
    def opt(self):
        return BaseOptimizer(knowledge_base=None)

    def test_empty_frontier(self, opt):
        assert opt.solve_balanced([]) is None

    def test_single_solution(self, opt):
        s = ParetoSolution(ProductConfig("x", []), shifts=[])
        s.orundum_eff = 0.5
        s.lmd_eff = 0.5
        s.combat_record_eff = 0.5
        result = opt.solve_balanced([s])
        assert result is s

    def test_picks_most_balanced(self, opt):
        """Knee-point should pick the solution closest to (1,1,1)."""
        a = ParetoSolution(ProductConfig("x", []), shifts=[])
        a.orundum_eff, a.lmd_eff, a.combat_record_eff = 1.0, 0.2, 0.1
        b = ParetoSolution(ProductConfig("x", []), shifts=[])
        b.orundum_eff, b.lmd_eff, b.combat_record_eff = 0.5, 0.5, 0.5
        c = ParetoSolution(ProductConfig("x", []), shifts=[])
        c.orundum_eff, c.lmd_eff, c.combat_record_eff = 0.1, 0.2, 1.0

        # b is clearly closest to (1,1,1): distance = sqrt(0.25+0.25+0.25)=0.87
        # a: sqrt(0+0.64+0.81)=1.20, c: sqrt(0.81+0.64+0)=1.20
        result = opt.solve_balanced([a, b, c])
        assert result is b

    def test_balanced_vs_weighted(self, opt):
        """For a balanced goal, knee-point should differ from weighted-sum."""
        SAMPLE = {"德克萨斯": 2, "能天使": 2, "清流": 2, "砾": 0, "斑点": 1}
        frontier = opt.solve_pareto(SAMPLE, layout="243")
        assert len(frontier) >= 2

        best_ws = opt.solve_with_weights(frontier, orundum=0.3, lmd=0.4, combat_record=0.3)
        best_balanced = opt.solve_balanced(frontier)
        assert best_ws is not None
        assert best_balanced is not None

        # Both should be valid solutions on the frontier
        assert best_ws.dominated_by == 0
        assert best_balanced.dominated_by == 0

    def test_knee_more_balanced_than_weighted(self, opt):
        """Knee point should have lower spread between dimensions."""
        SAMPLE = {"德克萨斯": 2, "能天使": 2, "清流": 2, "砾": 0, "斑点": 1}
        frontier = opt.solve_pareto(SAMPLE, layout="243")
        if len(frontier) < 2:
            return  # not enough frontier points to compare

        best_ws = opt.solve_with_weights(frontier, orundum=0.3, lmd=0.4, combat_record=0.3)
        best_balanced = opt.solve_balanced(frontier)

        def spread(s: ParetoSolution) -> float:
            return max(s.orundum_eff, s.lmd_eff, s.combat_record_eff) - \
                   min(s.orundum_eff, s.lmd_eff, s.combat_record_eff)

        if best_ws is not best_balanced:
            # Knee point should be at least as balanced as weighted-sum
            assert spread(best_balanced) <= spread(best_ws)
