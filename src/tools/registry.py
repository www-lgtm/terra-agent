"""Tool registry with check_fn support, TTL caching, and game grouping.

Pattern: tools self-register at module import time. check_fn results are cached
for 30 seconds to avoid repeated expensive checks (ADB status, etc.).

Game grouping: tools can be tagged as universal (game=None) or game-specific.
get_definitions(game="arknights") returns universal + arknights-specific tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ImageBlock:
    """Image content block matching Anthropic API format."""
    data: str           # base64 encoded image
    media_type: str = "image/jpeg"


@dataclass
class ToolOutput:
    """Structured tool result, replacing JSON string magic-field coupling."""
    text: str                                    # JSON text (backward compatible)
    images: list[ImageBlock] = field(default_factory=list)
    needs_user: bool = False                     # was: needs_confirmation
    task_done: bool = False                      # was: completed
    subtask_done: bool = False                 # subtask completed → clean history
    subtask_name: str = ""                     # subtask label (e.g. 'recruit')
    subtask_result: str = ""                   # human-readable outcome
    screen_hash: str | None = None               # perceptual hash for dedup
    screen_texts: list[str] = field(default_factory=list)  # OCR texts


class ToolEntry:
    __slots__ = ("name", "description", "parameters", "handler", "check_fn",
                 "game", "_check_cache", "_check_cache_time", "_description_fn")

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., ToolOutput],
        check_fn: Callable[[], bool] | None = None,
        game: str | None = None,
        description_fn: Callable[[], str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.check_fn = check_fn
        self.game = game  # None = universal, str = game-specific
        self._check_cache: bool | None = None
        self._check_cache_time: float = 0.0
        self._description_fn = description_fn  # Dynamic description (lazy)

    def is_available(self) -> bool:
        """Check if the tool is available. Results cached for 30s."""
        if self.check_fn is None:
            return True
        now = time.monotonic()
        if self._check_cache is not None and (now - self._check_cache_time) < 30.0:
            return self._check_cache
        try:
            self._check_cache = self.check_fn()
        except Exception:
            self._check_cache = False
        self._check_cache_time = now
        return self._check_cache

    def to_schema(self, game: str | None = None) -> dict[str, Any]:
        """Build the tool schema, optionally updating description dynamically."""
        desc = self.description
        if self._description_fn:
            try:
                dynamic = self._description_fn(game=game)
                if dynamic:
                    desc = dynamic
            except Exception:
                pass
        return {
            "name": self.name,
            "description": desc,
            "parameters": self.parameters,
        }

    def update_description(self, new_desc: str) -> None:
        """Replace the static description (used for dynamic tools like knowledge_query)."""
        self.description = new_desc


class ToolRegistry:
    """Thread-safe singleton registry for all agent tools.

    Supports game grouping so the LLM only sees tools relevant to the
    currently active game.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., ToolOutput],
        check_fn: Callable[[], bool] | None = None,
        override: bool = False,
        game: str | None = None,
        description_fn: Callable[[], str] | None = None,
    ) -> ToolEntry:
        """Register a tool.

        Args:
            game: None = universal (all games), str = game-specific.
            description_fn: Callable returning dynamic description (lazy).
        """
        with self._lock:
            if name in self._tools and not override:
                raise ValueError(f"Tool '{name}' already registered")
            entry = ToolEntry(name, description, parameters, handler, check_fn,
                            game=game, description_fn=description_fn)
            self._tools[name] = entry
            return entry

    def deregister(self, name: str) -> None:
        with self._lock:
            self._tools.pop(name, None)

    def get(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)

    def get_definitions(self, game: str | None = None, skill_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Return tool schemas filtered by game and optionally by matched skills.

        game=None: return only universal tools.
        game="arknights": return universal + arknights-specific tools.

        When skill_names is provided, heavy conditional tools (base_shift_maa,
        recruit_optimizer, box_scan, etc.) are excluded unless a matching skill
        is present — saving ~1500-2000 tokens per LLM call.
        """
        with self._lock:
            schemas = [
                t.to_schema(game) for t in self._tools.values()
                if t.is_available() and (t.game is None or t.game == game)
            ]
        if skill_names is not None:
            schemas = self._filter_by_skills(schemas, skill_names)
        return schemas

    @staticmethod
    def _filter_by_skills(schemas: list[dict[str, Any]], skill_names: list[str]) -> list[dict[str, Any]]:
        """Exclude conditional tools whose trigger skills are absent.

        Each conditional tool is only included when at least one matched skill
        name contains the trigger keyword.  Core tools (adb_tap, ask_user, etc.)
        are always included.
        """
        # Conditional tools: (trigger_keywords, tool_names)
        # Trigger fires if ANY keyword appears in ANY matched skill name.
        _CONDITIONAL: list[tuple[list[str], list[str]]] = [
            (["base"],        ["base_shift_maa"]),
            (["recruit"],     ["optimize_recruit_tags"]),
            (["box", "warehouse", "depot"], ["scan_operator_box", "scan_depot", "save_depot_resources"]),
            (["material"],    ["vlm_match_material", "vlm_identify_icon"]),
            (["schedule", "cron"], ["schedule_create", "schedule_list", "schedule_delete", "schedule_toggle"]),
        ]

        if not skill_names:
            return schemas

        # Build exclusion set: tools whose triggers don't match any skill
        skills_lower = " ".join(skill_names).lower()
        exclude: set[str] = set()
        for triggers, tools in _CONDITIONAL:
            if not any(kw in skills_lower for kw in triggers):
                exclude.update(tools)

        if exclude:
            return [s for s in schemas if s["name"] not in exclude]
        return schemas

    def get_names(self, game: str | None = None) -> list[str]:
        """Return tool names filtered by game."""
        with self._lock:
            return [
                name for name, t in self._tools.items()
                if t.is_available() and (t.game is None or t.game == game)
            ]

    def update_description(self, name: str, new_desc: str) -> None:
        """Update a tool's description (for dynamic tools)."""
        with self._lock:
            entry = self._tools.get(name)
            if entry:
                entry.update_description(new_desc)

    def dispatch(self, tool_name: str, ctx: Any = None, **kwargs: Any) -> ToolOutput:
        """Execute a tool by name. Returns structured ToolOutput.

        Args:
            tool_name: Name of the registered tool.
            ctx: Optional ToolContext for explicit context passing.
                 When provided, thread-local game is set so handlers
                 calling get_current_game() get the correct value.
            **kwargs: Tool-specific arguments forwarded to the handler.
        """
        entry = self._tools.get(tool_name)
        if entry is None:
            return ToolOutput(text=json.dumps({"error": f"Unknown tool: {tool_name}"}))
        if not entry.is_available():
            return ToolOutput(text=json.dumps({"error": f"Tool '{tool_name}' is currently unavailable"}))
        # If explicit context is provided, sync thread-local so existing
        # handlers that call get_current_game() / _get_current_game() work.
        prev_game = None
        if ctx is not None and hasattr(ctx, 'game'):
            prev_game = getattr(_game_context, 'game', None)
            _game_context.game = ctx.game
        # Also set agent_ctx if available (for ask_user reply routing)
        prev_agent = None
        if ctx is not None and hasattr(ctx, 'agent_ref'):
            import threading
            prev_agent = getattr(threading.current_thread(), '_terra_agent_ctx', None)
            threading.current_thread()._terra_agent_ctx = ctx.agent_ref
        try:
            result = entry.handler(**kwargs)
            if isinstance(result, ToolOutput):
                return result
            return ToolOutput(text=result)
        except Exception as e:
            logger.exception("Tool '%s' failed", tool_name)
            return ToolOutput(text=json.dumps({"error": str(e)}))
        finally:
            # Restore previous thread-local state
            if prev_game is not None:
                _game_context.game = prev_game
            if prev_agent is not None:
                import threading
                threading.current_thread()._terra_agent_ctx = prev_agent

    @property
    def tool_count(self) -> int:
        return len(self._tools)


registry = ToolRegistry()


# ── Thread-local game context ─────────────────────────────────────

_game_context: threading.local = threading.local()


def set_current_game(game: str) -> None:
    """Set the active game for the current thread.

    Called by TerraAgent at task start.  Tools that need game context
    (e.g. knowledge_query) read this.
    """
    _game_context.game = game


def get_current_game() -> str:
    """Get the active game for the current thread. Default: arknights."""
    return getattr(_game_context, "game", "arknights")


# ── Helpers ───────────────────────────────────────────────────────

def tool_error(message: str, **extra: Any) -> ToolOutput:
    return ToolOutput(text=json.dumps({"error": message, **extra}))


def tool_result(**kwargs: Any) -> ToolOutput:
    return ToolOutput(text=json.dumps(kwargs))
