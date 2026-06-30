"""Game plugin interface — the stable contract for multi-game support.

Every game implements a GamePlugin subclass.  The agent loop, router, and
tool system interact with games exclusively through this interface — never
through hardcoded keywords, table names, or task types.

Lifecycle:
    1. GamePlugin subclass is instantiated at import time
    2. GameRegistry.register(plugin) is called
    3. Agent loop queries registry for detection / routing / system prompts
    4. Tool descriptions are dynamically generated from registered manifests
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GameManifest:
    """Declarative game metadata — replaces hardcoded keywords in router.py.

    All fields are read-only by convention.  The manifest is the single
    source of truth for game identity, detection, and configuration.
    """

    id: str                          # "arknights", "reverse1999"
    name: str                        # "明日方舟", "重返未来1999"
    keywords: list[str] = field(default_factory=list)       # Intent detection
    knowledge_tables: list[str] = field(default_factory=list)  # Available tables
    skill_dir: str = ""              # Relative to data/skills/
    memory_dir: str = ""             # Relative to data/memories/
    knowledge_dir: str = ""          # Relative to src/knowledge/

    # UI / game-specific system prompt additions (appended after common rules)
    system_prompt_append: str = ""

    # Safety — dangerous keywords that should trigger confirmation or blocking
    dangerous_keywords: list[str] = field(default_factory=list)
    safe_compound_terms: list[str] = field(default_factory=list)
    require_confirmation_keywords: list[str] = field(default_factory=list)

    # Task classification — keyword → task_type mapping
    task_keywords: dict[str, list[str]] = field(default_factory=dict)
    task_priority: dict[str, int] = field(default_factory=dict)
    # Simple task verbs for deterministic dispatch (avoids LLM call for
    # obvious single-agent scenarios like "刷1-7", "收菜", "公招")
    task_verbs: list[str] = field(default_factory=list)

    # ADB auto-discovery — Android package names for detecting installed games.
    # Used by EmulatorInventory.auto_discover_device() via adb shell pm list packages.
    # Multiple packages per game supported (e.g. CN + EN versions).
    android_packages: list[str] = field(default_factory=list)


class GamePlugin(ABC):
    """Abstract base for game-specific logic.

    Subclass this, fill out the manifest, implement abstract methods,
    and register with GameRegistry.  The agent loop picks it up automatically.

    Subclasses live in src/games/<game_id>/plugin.py and register themselves
    at module import time.
    """

    manifest: GameManifest  # Set by subclass as a class attribute

    # ── Task classification (may be overridden by subclass) ──────

    def classify_task(self, text: str) -> str:
        """Classify a user message into a task type using manifest keywords.

        Scores each task type by total keyword hit count (handles overlapping
        keywords like "龙门币" appearing in both farm and query).  The type
        with the most keyword matches wins.
        """
        text_lower = text.lower()
        best_type = "unknown"
        best_hits = 0
        for task_type, keywords in self.manifest.task_keywords.items():
            hits = sum(1 for kw in keywords if kw.lower() in text_lower)
            if hits > best_hits:
                best_hits = hits
                best_type = task_type
        return best_type

    def get_task_priority(self, task_type: str) -> int:
        """Get numeric priority for a task type (lower = higher priority)."""
        return self.manifest.task_priority.get(task_type, 5)

    # ── System prompt ────────────────────────────────────────────

    def get_system_prompt_append(self) -> str:
        """Game-specific additions appended after the common stable layer.

        Override for richer per-game UI guidance.
        """
        return self.manifest.system_prompt_append

    # ── Safety overrides ─────────────────────────────────────────

    def get_safety_overrides(self) -> dict[str, list[str]]:
        """Return per-game safety keyword overrides.

        Returns dict with keys: 'dangerous_keywords', 'safe_compound_terms',
        'require_confirmation_keywords'.  Each maps to a list of strings.
        """
        return {
            "dangerous_keywords": self.manifest.dangerous_keywords,
            "safe_compound_terms": self.manifest.safe_compound_terms,
            "require_confirmation_keywords": self.manifest.require_confirmation_keywords,
        }

    def get_task_verbs(self) -> list[str]:
        """Return task action verbs for deterministic dispatch."""
        return self.manifest.task_verbs

    # ── Schedule intent classification ───────────────────────────

    # Overridable keyword sets for schedule intent parsing
    schedule_create_keywords: list[str] = [
        "定时", "每天", "每周", "每隔", "每个小时", "每分钟",
    ]
    schedule_manage_keywords: dict[str, list[str]] = {
        "list": ["查看定时任务", "定时任务列表", "有哪些定时任务"],
        "delete": ["取消定时任务", "删除定时任务", "移除定时任务"],
        "disable": ["暂停定时任务"],
        "enable": ["启用定时任务"],
        "stop": ["停止当前任务", "停止任务", "停止运行"],
    }

    def classify_schedule_intent(self, text: str) -> str:
        """Classify a message into a schedule management intent.

        Returns 'create', 'list', 'delete', 'disable', 'enable', 'stop', or ''.
        """
        import re

        for intent, keywords in self.schedule_manage_keywords.items():
            for kw in keywords:
                if kw in text:
                    return intent

        # "create" intent: require a time expression alongside the schedule keyword.
        # Without time info, words like "每天"/"定时" are just conversational
        # ("每天上号清体力" = I want to run dailies NOW, not schedule a cron).
        has_schedule_kw = any(kw in text for kw in self.schedule_create_keywords)
        if has_schedule_kw:
            has_time = bool(re.search(
                r'\d+点|\d+[：:]\d+|早上|上午|中午|下午|晚上|凌晨|'
                r'每小时|每分钟|每隔?\d+[分时天秒]|'
                r'明天|后天|大后天|下周|下个月|周[一二三四五六日]',
                text,
            ))
            if has_time:
                return "create"
        return ""

    # ── Daily tasks ──────────────────────────────────────────────

    def get_daily_tasks(self) -> list[dict[str, Any]]:
        """Return a list of recommended daily tasks for this game.

        Each entry: {"description": str, "priority": int}
        Override in subclass for game-specific daily checklists.
        """
        return []

    # ── Knowledge tool description (dynamic) ─────────────────────

    def build_knowledge_tool_hint(self) -> str:
        """Build the table-list portion of the knowledge_query tool description."""
        if not self.manifest.knowledge_tables:
            return ""
        tables = ", ".join(self.manifest.knowledge_tables)
        return f"- {self.manifest.name} ({self.manifest.id}): {tables}"

    # ── VLM adapter (Phase A) ────────────────────────────────────

    def get_vlm_adapter(self) -> dict[str, str]:
        """Return VLM→OCR term mapping for this game. Default: empty.

        VLM outputs English button names, but OCR can't read game art fonts.
        Each game can provide a mapping from English VLM terms to target-language
        terms that OCR can actually detect on screen.

        Override in game-specific plugins.
        """
        return {}

    # ── Tool registration (called when game is activated) ────────

    @abstractmethod
    def register_intelligence_tools(self) -> None:
        """Register game-specific intelligence tools with IntelligenceRegistry."""
        ...

    def register_game_tools(self) -> None:
        """Register game-specific tools with ToolRegistry.

        Override if the game needs custom tools beyond the universal set.
        Default: no-op (universal tools are already registered).
        """
        pass
