"""Unit tests for ToolRegistry game grouping and thread-local context."""

import json
import pytest

from src.tools.registry import (
    ToolRegistry, ToolEntry, ToolOutput, registry,
    set_current_game, get_current_game,
)


# ── Helper ────────────────────────────────────────────────────────

def _dummy_handler(**kw) -> ToolOutput:
    return ToolOutput(text=json.dumps({"ok": True}))


# ── Tests ─────────────────────────────────────────────────────────

class TestToolRegistryGameGrouping:
    """Tests for game-filtered tool definitions."""

    def setup_method(self):
        """Use a fresh registry for each test to avoid global state pollution."""
        self.reg = ToolRegistry()
        self.reg.register("universal_tool", "A shared tool",
                         {"type": "object", "properties": {}},
                         _dummy_handler, game=None)
        self.reg.register("ark_tool", "Arknights-only tool",
                         {"type": "object", "properties": {}},
                         _dummy_handler, game="arknights")
        self.reg.register("r99_tool", "Reverse1999-only tool",
                         {"type": "object", "properties": {}},
                         _dummy_handler, game="reverse1999")

    def test_get_definitions_none_returns_universal_only(self):
        defs = self.reg.get_definitions(game=None)
        names = [d["name"] for d in defs]
        assert "universal_tool" in names
        assert "ark_tool" not in names
        assert "r99_tool" not in names

    def test_get_definitions_arknights_returns_universal_plus_ark(self):
        defs = self.reg.get_definitions(game="arknights")
        names = [d["name"] for d in defs]
        assert "universal_tool" in names
        assert "ark_tool" in names
        assert "r99_tool" not in names

    def test_get_definitions_r99_returns_universal_plus_r99(self):
        defs = self.reg.get_definitions(game="reverse1999")
        names = [d["name"] for d in defs]
        assert "universal_tool" in names
        assert "ark_tool" not in names
        assert "r99_tool" in names

    def test_get_names_game_filtered(self):
        names = self.reg.get_names(game="arknights")
        assert "universal_tool" in names
        assert "ark_tool" in names
        assert "r99_tool" not in names

    def test_unavailable_tool_excluded(self):
        """Tools with failing check_fn should be excluded from definitions."""
        self.reg.register(
            "flaky_tool", "Sometimes unavailable",
            {"type": "object", "properties": {}},
            _dummy_handler,
            game=None,
            check_fn=lambda: False,
        )
        defs = self.reg.get_definitions(game="arknights")
        names = [d["name"] for d in defs]
        assert "flaky_tool" not in names

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError):
            self.reg.register("universal_tool", "Duplicate",
                            {"type": "object", "properties": {}},
                            _dummy_handler)

    def test_register_override_allows_duplicate(self):
        self.reg.register("universal_tool", "Updated desc",
                         {"type": "object", "properties": {}},
                         _dummy_handler, override=True)
        # Should not raise

    def test_deregister(self):
        self.reg.deregister("ark_tool")
        defs = self.reg.get_definitions(game="arknights")
        names = [d["name"] for d in defs]
        assert "ark_tool" not in names

    def test_dispatch_unknown_tool(self):
        output = self.reg.dispatch("nonexistent_tool")
        data = json.loads(output.text)
        assert data.get("error")

    def test_dispatch_unavailable_tool(self):
        self.reg.register(
            "offline_tool", "Offline",
            {"type": "object", "properties": {}},
            _dummy_handler,
            check_fn=lambda: False,
        )
        output = self.reg.dispatch("offline_tool")
        data = json.loads(output.text)
        assert "unavailable" in data.get("error", "").lower()

    def test_dispatch_available_tool(self):
        output = self.reg.dispatch("universal_tool")
        data = json.loads(output.text)
        assert data.get("ok") is True

    def test_dynamic_description(self):
        """Tools with description_fn should get dynamic descriptions."""
        call_count = [0]

        def _desc_fn():
            call_count[0] += 1
            return f"Dynamic description v{call_count[0]}"

        self.reg.register(
            "dynamic_tool", "Static fallback",
            {"type": "object", "properties": {}},
            _dummy_handler,
            description_fn=_desc_fn,
        )
        defs = self.reg.get_definitions()
        dyn = [d for d in defs if d["name"] == "dynamic_tool"]
        assert len(dyn) == 1
        assert "Dynamic description v1" in dyn[0]["description"]

    def test_update_description(self):
        self.reg.update_description("universal_tool", "Updated description")
        entry = self.reg.get("universal_tool")
        assert entry is not None
        assert entry.description == "Updated description"

    def test_tool_count(self):
        assert self.reg.tool_count == 3


class TestGameContext:
    """Tests for thread-local game context."""

    def test_default_game(self):
        # Need to simulate a fresh thread-local — just test default
        from src.tools.registry import get_current_game
        # In test runner, no game has been set yet
        assert get_current_game() in ("arknights", "")  # Default may vary

    def test_set_and_get(self):
        set_current_game("reverse1999")
        assert get_current_game() == "reverse1999"
        set_current_game("arknights")  # Restore for other tests
        assert get_current_game() == "arknights"

    def test_dispatch_passes_handler_error(self):
        reg = ToolRegistry()

        def _failing(**kw):
            raise RuntimeError("Test error")

        reg.register("fail_tool", "Will fail",
                    {"type": "object", "properties": {}},
                    _failing)
        output = reg.dispatch("fail_tool")
        data = json.loads(output.text)
        assert "Test error" in data.get("error", "")

    def test_handler_returns_string_backward_compat(self):
        reg = ToolRegistry()

        def _str_handler(**kw):
            return json.dumps({"old_style": True})

        reg.register("old_tool", "Old-style",
                    {"type": "object", "properties": {}},
                    _str_handler)
        output = reg.dispatch("old_tool")
        assert isinstance(output, ToolOutput)
        data = json.loads(output.text)
        assert data.get("old_style") is True


class TestToolOutput:
    """Tests for the ToolOutput dataclass."""

    def test_defaults(self):
        to = ToolOutput(text="{}")
        assert to.images == []
        assert to.needs_user is False
        assert to.task_done is False
        assert to.screen_hash is None
        assert to.screen_texts == []

    def test_with_data(self):
        from src.tools.registry import ImageBlock
        img = ImageBlock(data="base64data")
        to = ToolOutput(
            text='{"ok": true}',
            images=[img],
            needs_user=True,
            task_done=True,
            screen_hash="abc123",
            screen_texts=["button1", "button2"],
        )
        assert len(to.images) == 1
        assert to.needs_user is True
        assert to.task_done is True
        assert to.screen_hash == "abc123"
        assert len(to.screen_texts) == 2
