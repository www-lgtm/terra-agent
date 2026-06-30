"""Per-slot task queue — defer tasks when account is busy.

Phase 3: all mutations persist to SQLite for crash recovery.
Memory is primary (fast), DB is backup.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PendingTask:
    """A task waiting in a slot's queue."""
    task_description: str
    task_type: str = "custom"      # farm / base / query / plan / custom
    game: str = "arknights"
    queued_at: float = 0.0
    priority: int = 5              # lower = more urgent
    db_id: int | None = None       # task_queue row id for DB sync


PRIORITY_MAP: dict[str, int] = {
    "user_explicit": 0,
    "farm": 1,
    "base": 2,
    "query": 3,
    "plan": 4,
    "custom": 5,
}


def _save_queue(slot: Any) -> None:
    """Sync a slot's in-memory queue to the DB.

    Wraps DELETE + INSERT in a transaction so a crash between the two
    operations doesn't permanently lose the queue.
    """
    try:
        from src.scheduler.schedule_db import schedule_db
        conn = schedule_db.conn
        conn.execute("BEGIN")
        try:
            conn.execute(
                "DELETE FROM task_queue WHERE slot_id = ?", (slot.slot_id,)
            )
            for pt in slot.pending_tasks:
                pt.db_id = schedule_db.enqueue_task_db(
                    slot_id=slot.slot_id,
                    task_description=pt.task_description,
                    game=pt.game,
                    task_type=pt.task_type,
                    priority=pt.priority,
                )
            conn.commit()
        except Exception:
            conn.execute("ROLLBACK")
            raise
    except (sqlite3.OperationalError, sqlite3.ProgrammingError) as e:
        # DB unavailable or locked — non-critical (queue survives in memory)
        logger.warning("Failed to save queue for slot %s: %s", slot.slot_id, e)
    except Exception as e:
        logger.error("Unexpected error saving queue for slot %s: %s", slot.slot_id, e)


def enqueue_task(slot: Any, task_desc: str, game: str = "arknights",
                 task_type: str = "custom", chained_from: str = "") -> PendingTask:
    """Add a task to a slot's queue. Returns the PendingTask.

    chained_from: if set (e.g. "multi_game"), the task is part of a chain
    and should auto-start on the same device after the current task completes.
    """
    import time as _time

    priority = PRIORITY_MAP.get(task_type, 5)
    pt = PendingTask(
        task_description=task_desc,
        task_type=task_type,
        game=game,
        queued_at=_time.time(),
        priority=priority,
    )
    slot.pending_tasks.append(pt)
    # Sort by priority (ascending)
    slot.pending_tasks.sort(key=lambda t: t.priority)
    _save_queue(slot)
    return pt


def dequeue_next(slot: Any) -> PendingTask | None:
    """Pop the highest-priority task from a slot's queue. Returns None if empty."""
    if not slot.pending_tasks:
        return None
    task = slot.pending_tasks.pop(0)
    _save_queue(slot)
    return task


def cancel_queued(slot: Any, index: int = 0) -> PendingTask | None:
    """Remove a queued task by position (0-based). Returns the removed task or None."""
    if 0 <= index < len(slot.pending_tasks):
        task = slot.pending_tasks.pop(index)
        _save_queue(slot)
        return task
    return None


def clear_queue(slot: Any) -> list[PendingTask]:
    """Clear all queued tasks. Returns the removed list."""
    removed = list(slot.pending_tasks)
    slot.pending_tasks.clear()
    _save_queue(slot)
    return removed


def queue_size(slot: Any) -> int:
    return len(slot.pending_tasks)


def load_queue(slot: Any) -> int:
    """Hydrate a slot's pending_tasks from DB on startup. Returns count loaded.

    Only clears the in-memory queue if it is currently empty (startup state).
    Mid-lifecycle calls are no-ops to avoid losing in-flight tasks.
    """
    import time as _time
    # Guard: only load into empty queues (startup recovery)
    if slot.pending_tasks:
        return 0
    try:
        from src.scheduler.schedule_db import schedule_db
        rows = schedule_db.get_queued_tasks(slot.slot_id)
        recovered = 0
        for r in rows:
            task_game = r.get("game", "")
            slot_game = getattr(slot, 'game', '')
            # Stale recovery guard: skip tasks whose game doesn't match the slot.
            if slot_game and task_game and slot_game != task_game:
                logger.warning(
                    "Dropping stale recovered task '%s' (game=%s) — slot %s is game=%s",
                    str(r["task_description"])[:60], task_game, slot.slot_id, slot_game,
                )
                try:
                    schedule_db.conn.execute(
                        "DELETE FROM task_queue WHERE id = ?", (r["id"],)
                    )
                    schedule_db.conn.commit()
                except Exception as e:
                    logger.warning("Failed to delete stale task %s: %s", r["id"], e)
                continue
            slot.pending_tasks.append(PendingTask(
                task_description=r["task_description"],
                task_type=r.get("task_type", "custom"),
                game=task_game or "arknights",
                priority=r.get("priority", 5),
                db_id=r["id"],
            ))
            recovered += 1
        slot.pending_tasks.sort(key=lambda t: t.priority)
        return recovered
    except Exception as e:
        logger.warning("Failed to load queue for slot %s: %s", slot.slot_id, e)
        return 0


def load_all_queues(slots: list[Any]) -> int:
    """Hydrate all slots' queues from DB on startup. Returns total count."""
    total = 0
    for s in slots:
        total += load_queue(s)
    return total
