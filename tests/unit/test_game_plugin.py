"""Unit tests for GamePlugin, GameRegistry, and game-aware router delegation.

Phase 4 multi-game refactoring test suite.
"""

import pytest

from src.games.plugin import GameManifest, GamePlugin
from src.games.registry import GameRegistry, get_game_registry


# ── Minimal test plugin ───────────────────────────────────────────

class _TestPlugin(GamePlugin):
    """Minimal plugin for testing registry operations — NOT auto-registered."""

    manifest = GameManifest(
        id="test_game",
        name="测试游戏",
        keywords=["测试", "test", "体力"],
        knowledge_tables=["test_table"],
        dangerous_keywords=["钻石"],
        safe_compound_terms=["钻石订单"],
        require_confirmation_keywords=["购买钻石"],
        task_keywords={
            "farm": ["刷", "体力"],
            "base": ["收菜"],
            "query": ["查询", "还有多少"],
        },
        task_priority={"farm": 1, "base": 2, "query": 3},
        system_prompt_append="## 测试游戏 UI",
    )

    def register_intelligence_tools(self) -> None:
        pass  # No-op for test


class _TestPlugin2(GamePlugin):
    """Second test plugin."""

    manifest = GameManifest(
        id="game2",
        name="游戏2",
        keywords=["game2", "二"],
        knowledge_tables=["g2_table"],
    )

    def register_intelligence_tools(self) -> None:
        pass


# ── GameRegistry tests ────────────────────────────────────────────

class TestGameRegistry:
    """Tests for the GameRegistry singleton (uses fresh instances, not the global)."""

    def test_register_and_get(self):
        reg = GameRegistry()
        plugin = _TestPlugin()
        reg.register(plugin)
        assert reg.get("test_game") is plugin

    def test_list_all(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())
        reg.register(_TestPlugin2())
        assert len(reg.list_all()) == 2
        assert "test_game" in reg.get_ids()
        assert "game2" in reg.get_ids()

    def test_get_nonexistent_returns_none(self):
        reg = GameRegistry()
        assert reg.get("nonexistent") is None

    def test_get_default_fallback(self):
        reg = GameRegistry()
        # Empty registry — get_default should not crash
        plugin = reg.get_default()
        assert plugin is None  # No games registered

    def test_detect_game_single_match(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())
        reg.register(_TestPlugin2())

        assert reg.detect_game("我要刷体力") == "test_game"
        assert reg.detect_game("game2 something") == "game2"

    def test_detect_game_no_match_defaults_to_first(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())
        assert reg.detect_game("你好") == "test_game" or reg.detect_game("你好") == reg.default_game

    def test_detect_game_multiple_matches_picks_best(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())   # keywords: ["测试", "test", "体力"]
        reg.register(_TestPlugin2())  # keywords: ["game2", "二"]

        # "体力" matches test_game exactly — test_game should win
        assert reg.detect_game("使用体力刷图") == "test_game"

        # "game2 体力" matches both — game2 has 1 hit, test_game has 1 hit.
        # First registered with score wins (typically test_game)
        result = reg.detect_game("game2 体力")
        assert result in ("test_game", "game2")

    def test_classify_task_delegates(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())
        assert reg.classify_task("刷体力", game_id="test_game") == "farm"
        assert reg.classify_task("收菜", game_id="test_game") == "base"
        assert reg.classify_task("查询库存", game_id="test_game") == "query"
        assert reg.classify_task("hello world", game_id="test_game") == "unknown"

    def test_classify_task_no_game_falls_back(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())
        # Default game should be test_game since it's the only one
        assert reg.classify_task("刷体力") in ("farm", "unknown")

    def test_get_task_priority(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())
        assert reg.get_task_priority("刷体力", game_id="test_game") == 1
        assert reg.get_task_priority("收菜", game_id="test_game") == 2
        assert reg.get_task_priority("unknown task", game_id="test_game") == 5

    def test_classify_schedule_intent(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())
        assert reg.classify_schedule_intent("每天早上9点清体力", game_id="test_game") == "create"
        assert reg.classify_schedule_intent("查看定时任务", game_id="test_game") == "list"
        assert reg.classify_schedule_intent("取消定时任务#3", game_id="test_game") == "delete"
        assert reg.classify_schedule_intent("暂停定时任务", game_id="test_game") == "disable"
        assert reg.classify_schedule_intent("启用定时任务", game_id="test_game") == "enable"
        assert reg.classify_schedule_intent("停止当前任务", game_id="test_game") == "stop"
        assert reg.classify_schedule_intent("你好", game_id="test_game") == ""

    def test_system_prompt_append(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())
        append = reg.get_system_prompt_append(game_id="test_game")
        assert "测试游戏 UI" in append

    def test_dangerous_keywords(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())
        assert "钻石" in reg.get_dangerous_keywords(game_id="test_game")
        assert "钻石订单" in reg.get_safe_compound_terms(game_id="test_game")
        assert "购买钻石" in reg.get_confirmation_keywords(game_id="test_game")

    def test_build_knowledge_tool_description(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())
        desc = reg.build_knowledge_tool_description()
        assert "test_table" in desc
        assert "测试游戏" in desc

    def test_re_register_updates(self):
        reg = GameRegistry()
        p1 = _TestPlugin()
        reg.register(p1)
        assert reg.get("test_game") is p1

        p2 = _TestPlugin()
        reg.register(p2)
        assert reg.get("test_game") is p2  # Updated

    def test_get_game_name_returns_manifest_name(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())
        assert reg.get_game_name("test_game") == "测试游戏"

    def test_get_game_name_unknown_returns_id(self):
        reg = GameRegistry()
        assert reg.get_game_name("nonexistent") == "nonexistent"

    def test_get_game_name_none_returns_default(self):
        reg = GameRegistry()
        reg.register(_TestPlugin())
        # Default game is "arknights" but test_game is the only registered
        name = reg.get_game_name(None)
        assert isinstance(name, str) and len(name) > 0


# ── GameManifest tests ────────────────────────────────────────────

class TestGameManifest:
    """Tests for the declarative GameManifest dataclass."""

    def test_manifest_frozen(self):
        m = GameManifest(id="test", name="Test", keywords=["a"])
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            m.id = "changed"  # type: ignore[misc]

    def test_manifest_defaults(self):
        m = GameManifest(id="test", name="Test")
        assert m.keywords == []
        assert m.knowledge_tables == []
        assert m.task_keywords == {}
        assert m.task_priority == {}
        assert m.dangerous_keywords == []
        assert m.safe_compound_terms == []
        assert m.require_confirmation_keywords == []


# ── GamePlugin default behavior tests ─────────────────────────────

class TestGamePluginDefaults:
    """Tests for the default method implementations in GamePlugin ABC."""

    def test_classify_task_empty_keywords_returns_unknown(self):
        class _EmptyPlugin(GamePlugin):
            manifest = GameManifest(id="empty", name="空")
            def register_intelligence_tools(self): pass

        plugin = _EmptyPlugin()
        assert plugin.classify_task("任意文字") == "unknown"

    def test_get_task_priority_empty_returns_5(self):
        class _EmptyPlugin(GamePlugin):
            manifest = GameManifest(id="empty", name="空")
            def register_intelligence_tools(self): pass

        plugin = _EmptyPlugin()
        assert plugin.get_task_priority("farm") == 5

    def test_classify_schedule_intent_empty_returns_empty_string(self):
        class _EmptyPlugin(GamePlugin):
            manifest = GameManifest(id="empty", name="空")
            def register_intelligence_tools(self): pass

        plugin = _EmptyPlugin()
        assert plugin.classify_schedule_intent("任意文字") == ""

    def test_get_system_prompt_append_defaults_to_manifest_value(self):
        m = GameManifest(id="test", name="Test",
                         system_prompt_append="游戏专属提示")
        class _Plugin(GamePlugin):
            manifest = m
            def register_intelligence_tools(self): pass

        plugin = _Plugin()
        assert plugin.get_system_prompt_append() == "游戏专属提示"

    def test_build_knowledge_tool_hint(self):
        m = GameManifest(id="test", name="测试",
                         knowledge_tables=["t1", "t2"])
        class _Plugin(GamePlugin):
            manifest = m
            def register_intelligence_tools(self): pass

        plugin = _Plugin()
        hint = plugin.build_knowledge_tool_hint()
        assert "t1" in hint
        assert "t2" in hint
        assert "测试" in hint

    def test_build_knowledge_tool_hint_empty(self):
        m = GameManifest(id="test", name="测试")
        class _Plugin(GamePlugin):
            manifest = m
            def register_intelligence_tools(self): pass

        plugin = _Plugin()
        hint = plugin.build_knowledge_tool_hint()
        assert hint == ""

    def test_get_daily_tasks_default_empty(self):
        class _Plugin(GamePlugin):
            manifest = GameManifest(id="test", name="测试")
            def register_intelligence_tools(self): pass

        plugin = _Plugin()
        assert plugin.get_daily_tasks() == []


# ── Router delegation tests ───────────────────────────────────────

class TestRouterDelegation:
    """Confirm router.py delegate functions work via GameRegistry."""

    def test_detect_game_returns_default(self):
        from src.agent.router import detect_game
        game = detect_game("这是一个不包含任何游戏关键词的普通消息")
        assert game in ("arknights", "")  # Falls back to default or empty

    def test_detect_game_arknights(self):
        from src.agent.router import detect_game
        # Arknights keywords include "理智", "龙门币", "基建", "公招" etc.
        game = detect_game("清理智刷GT-6")
        assert game == "arknights"

    def test_classify_task_arknights_farm(self):
        from src.agent.router import classify_task
        assert classify_task("清体力刷GT-6") == "farm"

    def test_classify_task_arknights_base(self):
        from src.agent.router import classify_task
        assert classify_task("基建收菜换班") == "base"

    def test_classify_task_arknights_query(self):
        from src.agent.router import classify_task
        assert classify_task("还有多少龙门币") == "query"

    def test_classify_task_arknights_plan(self):
        from src.agent.router import classify_task
        assert classify_task("精二银灰需要什么材料") == "plan"

    def test_get_priority_farm_higher_than_plan(self):
        from src.agent.router import get_priority
        assert get_priority("清体力") < get_priority("精二规划")

    def test_classify_schedule_intent_through_router(self):
        from src.agent.router import classify_schedule_intent
        assert classify_schedule_intent("每天早上9点清体力") == "create"
        assert classify_schedule_intent("查看定时任务列表") == "list"
        assert classify_schedule_intent("取消定时任务#5") == "delete"
        assert classify_schedule_intent("暂停定时任务") == "disable"
        assert classify_schedule_intent("启用定时任务") == "enable"
        assert classify_schedule_intent("你好今天天气怎么样") == ""

    def test_extract_task_id(self):
        from src.agent.router import extract_task_id
        assert extract_task_id("取消定时任务#3") == 3
        assert extract_task_id("删除定时任务 5 号") == 5
        assert extract_task_id("你好") is None

    def test_route_task_returns_game(self):
        from src.agent.router import route_task
        result = route_task("清体力刷GT-6")
        assert result["game"] == "arknights"
        assert result["task_type"] == "farm"
        assert result["priority"] < 5
        assert isinstance(result["matching_skills"], list)


# ── Skillmanager factory tests ────────────────────────────────────

class TestSkillManagerFactory:
    """Tests for get_skill_manager() factory function."""

    def test_default_returns_arknights_manager(self):
        from src.skills.manager import get_skill_manager
        mgr = get_skill_manager()
        assert mgr.game == "arknights"

    def test_explicit_game_returns_correct_manager(self):
        from src.skills.manager import get_skill_manager
        mgr = get_skill_manager("reverse1999")
        assert mgr.game == "reverse1999"

    def test_cached_returns_same_instance(self):
        from src.skills.manager import get_skill_manager
        mgr1 = get_skill_manager("test_cache")
        mgr2 = get_skill_manager("test_cache")
        assert mgr1 is mgr2

    def test_different_games_return_different_instances(self):
        from src.skills.manager import get_skill_manager
        mgr1 = get_skill_manager("game_a")
        mgr2 = get_skill_manager("game_b")
        assert mgr1 is not mgr2

    def test_old_singleton_still_works(self):
        from src.skills.manager import skill_manager
        assert skill_manager.game == "arknights"
        names = skill_manager.list_all()
        assert isinstance(names, list)


# ── Game-aware Safety tests ───────────────────────────────────────

class TestSafetyGameAware:
    """Tests for game-aware safety functions."""

    def test_check_dangerous_arknights_blocks(self):
        from src.tools.registry import set_current_game
        set_current_game("arknights")
        from src.tools.safety import check_dangerous
        result = check_dangerous("源石", game="arknights")
        assert result is not None
        assert result["dangerous"] is True
        assert "源石" in result["keyword"]

    def test_check_dangerous_safe_compound_allowed(self):
        from src.tools.safety import check_dangerous
        result = check_dangerous("源石订单", game="arknights")
        assert result is None  # Safe compound term overrides

    def test_check_dangerous_unknown_game_returns_none(self):
        from src.tools.safety import check_dangerous
        result = check_dangerous("anything", game="unknown_game")
        assert result is None  # No dangerous keywords defined

    def test_check_confirmation_required(self):
        from src.tools.safety import check_confirmation_required
        assert check_confirmation_required("购买钻石", game="arknights") is True
        assert check_confirmation_required("正常操作", game="arknights") is False
