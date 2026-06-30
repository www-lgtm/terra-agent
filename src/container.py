"""Dependency Injection container for Terra Agent.

Provides a single immutable AppContainer dataclass that holds all service
instances.  Replaces scattered module-level singletons with a centralized
registry that is test-friendly (swap instances via set_container).

Design:
- AppContainer is frozen=True — constructed once, never mutated.
- get_container() lazy-builds on first call (double-check locking).
- set_container() / reset_container() are test hooks.
- Phase 1: registers existing singletons. Phase 2: adds extracted services.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AppContainer:
    """Immutable DI container holding all infrastructure and service instances.

    Constructed via build_container().  Tests use set_container() to inject
    mocks.  Never modified after construction — all fields are read-only.
    """

    # ── Configuration ──
    config: Any

    # ── Infrastructure singletons (Phase 1) ──
    tool_registry: Any         # ToolRegistry instance
    memory_db: Any             # MemoryDB instance
    skill_db: Any              # SkillDB instance
    game_registry: Any         # GameRegistry instance
    schedule_db: Any           # ScheduleDB instance
    emulator_manager: Any      # EmulatorManager instance

    # ── Service layer (Phase 2 — set when available) ──
    memory_hint_service: Any = None   # MemoryHintService
    compression_service: Any = None   # CompressionService
    execution_logger: Any = None      # ExecutionLogger
    review_trigger: Any = None        # ReviewTrigger

    # ── Client pool ──
    client_pool: Any = None

    # ── Factory method ──

    def build_terra_agent(self, device_serial: str, game: str = "arknights",
                          ask_fn: Any = None, ocr_engine: Any = None) -> Any:
        """Construct a TerraAgent with all container services injected.

        This is the preferred way to create a TerraAgent — it automatically
        receives memory_hint_service, compression_service, execution_logger,
        and review_trigger from the container.
        """
        from src.agent.loop import TerraAgent
        return TerraAgent(
            device_serial=device_serial,
            game=game,
            ask_fn=ask_fn,
            ocr_engine=ocr_engine,
            container=self,
        )


# ── Singleton management ──────────────────────────────────────────

_container: AppContainer | None = None
_lock = threading.Lock()


def build_container() -> AppContainer:
    """Construct the container, registering all existing singletons.

    Idempotent — repeated calls return the same result.  Imported lazily
    inside the function to avoid circular imports at module level.
    """
    from config.settings import config
    from src.tools.registry import registry as _tool_registry
    from src.memory.history import history_db as _history_db
    from src.memory.memory_db import memory_db as _memory_db
    from src.memory.skill_db import skill_db as _skill_db
    from src.games.registry import get_game_registry
    from src.scheduler.schedule_db import schedule_db as _schedule_db
    from src.device.emulator import emulator_manager as _emu_mgr
    from src.agent.memory_hint_service import MemoryHintService
    from src.agent.compression_service import CompressionService
    from src.agent.execution_logger import ExecutionLogger
    from src.agent.review_trigger import ReviewTrigger

    return AppContainer(
        config=config,
        tool_registry=_tool_registry,
        memory_db=_memory_db,
        skill_db=_skill_db,
        game_registry=get_game_registry(),
        schedule_db=_schedule_db,
        emulator_manager=_emu_mgr,
        client_pool=None,
        memory_hint_service=MemoryHintService(_memory_db, None, config),
        compression_service=CompressionService(None),
        execution_logger=ExecutionLogger(_history_db, _memory_db, config),
        review_trigger=ReviewTrigger(_memory_db, config),
    )


def get_container() -> AppContainer:
    """Get the global container.  Builds it on first call (lazy init).

    Thread-safe via double-check locking.
    """
    global _container
    if _container is None:
        with _lock:
            if _container is None:
                _container = build_container()
    return _container


# ── Test hooks ─────────────────────────────────────────────────────

def set_container(c: AppContainer) -> None:
    """Replace the global container (for testing)."""
    global _container
    _container = c


def reset_container() -> None:
    """Reset the global container to None (for test teardown)."""
    global _container
    _container = None
