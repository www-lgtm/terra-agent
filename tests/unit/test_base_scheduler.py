"""Unit tests for the BaseScheduler intelligence tool with Pareto frontier."""

import pytest
from src.intelligence.base import IntelligenceContext, IntelligenceResult
from src.intelligence.arknights.base_scheduler import BaseScheduler


def _ctx(**kw):
    return IntelligenceContext(game="arknights", knowledge=None, skills=[], memories=[], **kw)


@pytest.fixture(autouse=True)
def clean_sessions(tmp_path, monkeypatch):
    """Redirect session I/O to a temp dir so disk state doesn't leak into tests."""
    import src.intelligence.arknights.base_chain as chain
    monkeypatch.setattr(chain, "SESSION_DIR", tmp_path / "session")


class TestCanHandle:
    @pytest.fixture
    def tool(self):
        return BaseScheduler()

    def test_handles_orundum(self, tool):
        assert tool.can_handle("全力搓玉")
        assert tool.can_handle("搓玉为主也要练级")

    def test_handles_base(self, tool):
        assert tool.can_handle("基建换班")
        assert tool.can_handle("制造站排班")

    def test_handles_lmd_and_cr(self, tool):
        assert tool.can_handle("龙门币最大化")
        assert tool.can_handle("作战记录排班")

    def test_handles_balanced(self, tool):
        assert tool.can_handle("平衡龙门币和作战记录")

    def test_rejects_unrelated(self, tool):
        assert not tool.can_handle("刷1-7")
        assert not tool.can_handle("招募")


class TestParseGoal:
    @pytest.fixture
    def tool(self):
        return BaseScheduler()

    def test_pure_orundum(self, tool):
        key, _, w, _ = tool._parse_goal("全力搓玉")
        assert key == "orundum_max"

    def test_mixed_orundum_upgrade(self, tool):
        key, _, w, _ = tool._parse_goal("搓玉为主也要练级")
        assert key == "mixed_orundum_upgrade"
        assert w[0] > w[2]  # orundum > combat_record

    def test_lmd(self, tool):
        key, _, w, _ = tool._parse_goal("最大化龙门币")
        assert key == "lmd_max"
        assert w[1] > w[0]

    def test_combat_record(self, tool):
        key, _, w, _ = tool._parse_goal("作战记录")
        assert key == "combat_record_max"

    def test_balanced(self, tool):
        key, _, w, _ = tool._parse_goal("均衡发展")
        assert key == "balanced"


class TestParseBox:
    @pytest.fixture
    def tool(self):
        return BaseScheduler()

    def test_elite_levels(self, tool):
        box = tool._parse_box_text("德克萨斯(E2)、清流(E1)、斑点")
        assert box["德克萨斯"] == 2
        assert box["清流"] == 1
        assert box["斑点"] == 0  # default E0

    def test_chinese_elite(self, tool):
        box = tool._parse_box_text("德克萨斯(精英2)、清流(精英1)")
        assert box["德克萨斯"] == 2
        assert box["清流"] == 1

    def test_mixed_separators(self, tool):
        box = tool._parse_box_text("德克萨斯(E2),清流(E1)、银灰")
        assert len(box) >= 3

    def test_extract_with_colon_prefix(self, tool):
        box = tool._extract_box(
            "全力搓玉，我的干员：德克萨斯(E2)、清流(E2)、斑点", _ctx()
        )
        assert "德克萨斯" in box
        assert "清流" in box
        assert box["德克萨斯"] == 2

    def test_empty(self, tool, clean_sessions):
        box = tool._extract_box("搓玉", _ctx())
        assert box == {}



class TestAnalyze:
    @pytest.fixture
    def tool(self):
        return BaseScheduler()

    def test_no_box_prompts_user(self, tool):
        result = tool.analyze(_ctx(), "全力搓玉")
        assert result is not None
        assert result.confidence <= 0.5
        assert "干员" in result.recommendation

    def test_with_box_returns_schedule(self, tool):
        task = (
            "搓玉为主也要练级，我的干员："
            "德克萨斯(E2)、能天使(E2)、清流(E2)、温蒂(E2)、森蚺(E2)、"
            "阿罗玛(E2)、泡泡(E2)、火神(E2)、巫恋(E2)、但书(E2)、"
            "承曦格雷伊(E2)、斑点(E1)、食铁兽(E1)"
        )
        result = tool.analyze(_ctx(), task)
        assert result is not None
        assert result.confidence > 0.0

    def test_mixed_goal_with_box(self, tool):
        task = (
            "又要搓玉又要练级，我的干员：德克萨斯(E2)、能天使(E2)、清流(E2)"
        )
        result = tool.analyze(_ctx(), task)
        assert result is not None

    def test_orundum_only_goal(self, tool):
        task = "全力搓玉，我的干员：德克萨斯(E2)、能天使(E2)、清流(E2)"
        result = tool.analyze(_ctx(), task)
        assert result is not None
        assert "方案对比" in result.recommendation or "Pareto" in result.recommendation

    def test_output_includes_tables(self, tool):
        task = "全力搓玉，我的干员：德克萨斯(E2)、能天使(E2)、清流(E2)"
        result = tool.analyze(_ctx(), task)
        assert result is not None
        assert "|" in result.recommendation  # Markdown table
