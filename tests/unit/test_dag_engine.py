"""Unit tests for material DAG scheduler."""

from src.intelligence.arknights.material_scheduler import SchedulerEngine
from src.games.arknights.materials import get_craft_requirements, estimate_runs


def test_get_craft_requirements_simple():
    reqs = get_craft_requirements("糖聚块", 1)
    assert "糖" in reqs
    assert reqs["糖"] == 4
    assert "异铁" in reqs  # 异铁组 further decomposes to 异铁
    assert reqs["异铁"] == 4


def test_get_craft_requirements_no_recipe():
    reqs = get_craft_requirements("龙门币", 5)
    assert reqs == {"龙门币": 5}


def test_estimate_runs():
    results = estimate_runs("糖", 10)
    assert len(results) > 0
    assert results[0]["stage"] == "GT-6"
    assert results[0]["runs"] > 0


def test_scheduler_plan():
    engine = SchedulerEngine()
    plan = engine.plan("银灰", 2)
    assert plan["operator"] == "银灰"
    assert plan["elite_level"] == 2
    assert plan["total_sanity"] > 0
    assert len(plan["plan"]) > 0
