"""ScheduleDB: CRUD for scheduled tasks.

Now uses its own SQLite file (data/memory/schedule.db) instead of sharing
the monolithic history_db.conn.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from config.settings import config
from src.memory.base_db import BaseDB

logger = logging.getLogger(__name__)

DB_PATH = Path(config.DATA_DIR) / "memory" / "schedule.db"


class ScheduleDB(BaseDB):
    """CRUD operations for scheduled_tasks and schedule_history tables."""

    def __init__(self, path: Path | None = None) -> None:
        super().__init__(path or DB_PATH)
        self._tables_ensured = False
        self._ensure_tables()
        self._migrate_from_history_db()

    def _migrate_from_history_db(self) -> None:
        """One-time migration: copy scheduled_tasks + schedule_history from old terra.db.

        The old terra.db may still contain schedule tables even after the
        skills/memories tables were migrated.  This copies any rows and then
        drops the legacy tables from terra.db.
        """
        import sqlite3 as _sqlite3
        from config.settings import config as _config

        old_path = Path(_config.DATA_DIR) / "memory" / "terra.db"
        if not old_path.exists():
            return

        # Check if we've already migrated (our tables have data)
        row = self.conn.execute(
            "SELECT COUNT(*) FROM scheduled_tasks"
        ).fetchone()
        if row and row[0] > 0:
            return  # Already migrated

        old_conn = _sqlite3.connect(str(old_path))
        old_conn.row_factory = _sqlite3.Row
        try:
            # Check old tables exist
            tables = [r[0] for r in old_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "scheduled_tasks" not in tables:
                return  # Nothing to migrate

            tasks = old_conn.execute("SELECT * FROM scheduled_tasks").fetchall()
            history = old_conn.execute("SELECT * FROM schedule_history").fetchall()

            for t in tasks:
                t_dict = dict(t)
                # Check if this task already exists in new DB
                existing = self.conn.execute(
                    "SELECT id FROM scheduled_tasks WHERE id = ?", (t_dict["id"],)
                ).fetchone()
                if existing:
                    continue
                self.conn.execute(
                    """INSERT INTO scheduled_tasks
                       (id, name, description, game, task_type, task_payload,
                        schedule_type, schedule_value, enabled, one_shot,
                        next_run, last_run, last_result, run_count, max_runs,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        t_dict["id"], t_dict["name"], t_dict.get("description", ""),
                        t_dict.get("game", "arknights"), t_dict.get("task_type", "custom"),
                        t_dict["task_payload"],
                        t_dict["schedule_type"], t_dict["schedule_value"],
                        t_dict["enabled"], t_dict["one_shot"],
                        t_dict["next_run"], t_dict["last_run"],
                        t_dict.get("last_result", ""), t_dict.get("run_count", 0),
                        t_dict.get("max_runs"), t_dict["created_at"], t_dict["updated_at"],
                    ),
                )

            for h in history:
                h_dict = dict(h)
                self.conn.execute(
                    """INSERT INTO schedule_history
                       (id, task_id, started_at, finished_at, success, result_summary,
                        error_message, iterations, duration_seconds, device_serial)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        h_dict["id"], h_dict["task_id"], h_dict["started_at"],
                        h_dict["finished_at"], h_dict["success"],
                        h_dict.get("result_summary", ""), h_dict.get("error_message", ""),
                        h_dict.get("iterations", 0), h_dict.get("duration_seconds", 0),
                        h_dict.get("device_serial", ""),
                    ),
                )

            self.conn.commit()
            if tasks or history:
                logger.info("Schedule migration: %d tasks, %d history rows → %s",
                           len(tasks), len(history), self.path)
        except _sqlite3.OperationalError:
            logger.debug("Schedule migration: no legacy data to copy")
        except Exception as e:
            logger.warning("Schedule migration failed: %s", e)
        finally:
            old_conn.close()

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist (idempotent)."""
        if self._tables_ensured:
            return
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL,
                description     TEXT DEFAULT '',
                game            TEXT NOT NULL DEFAULT 'arknights',
                task_type       TEXT NOT NULL DEFAULT 'custom',
                task_payload    TEXT NOT NULL,
                schedule_type   TEXT NOT NULL DEFAULT 'cron',
                schedule_value  TEXT NOT NULL,
                enabled         INTEGER NOT NULL DEFAULT 1,
                one_shot        INTEGER NOT NULL DEFAULT 0,
                next_run        REAL,
                last_run        REAL,
                last_result     TEXT DEFAULT '',
                run_count       INTEGER NOT NULL DEFAULT 0,
                max_runs        INTEGER DEFAULT NULL,
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_enabled
                ON scheduled_tasks(enabled, next_run);

            CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_type
                ON scheduled_tasks(task_type);

            CREATE TABLE IF NOT EXISTS schedule_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id         INTEGER NOT NULL,
                started_at      REAL NOT NULL,
                finished_at     REAL,
                success         INTEGER NOT NULL DEFAULT 0,
                result_summary  TEXT DEFAULT '',
                error_message   TEXT DEFAULT '',
                iterations      INTEGER DEFAULT 0,
                duration_seconds REAL DEFAULT 0,
                device_serial   TEXT DEFAULT '',
                FOREIGN KEY (task_id) REFERENCES scheduled_tasks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_schedule_history_task
                ON schedule_history(task_id, started_at);
        """)
        self.conn.commit()

        # Phase 2 migration: add slot_id column if not exists
        for table in ("scheduled_tasks", "schedule_history"):
            try:
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN slot_id TEXT DEFAULT ''"
                )
                logger.debug("Added slot_id column to %s", table)
            except Exception:
                pass  # Column already exists

        # Phase 3: persistent task queues
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS task_queue (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                slot_id         TEXT NOT NULL,
                task_description TEXT NOT NULL,
                task_type       TEXT NOT NULL DEFAULT 'custom',
                game            TEXT NOT NULL,
                priority        INTEGER NOT NULL DEFAULT 5,
                queued_at       REAL NOT NULL,
                status          TEXT NOT NULL DEFAULT 'queued'
            );
            CREATE INDEX IF NOT EXISTS idx_task_queue_slot
                ON task_queue(slot_id, status, priority);
        """)

        self._tables_ensured = True

    # ---- Scheduled Tasks CRUD ----

    def create(self, name: str, task_payload: dict, *,
               schedule_type: str = "cron",
               schedule_value: str = "",
               description: str = "",
               game: str = "arknights",
               task_type: str = "custom",
               one_shot: bool = False,
               max_runs: int | None = None,
               next_run: float | None = None,
               slot_id: str = "") -> int:
        """Insert a new scheduled task. Returns the new row id."""
        self._ensure_tables()
        now = time.time()
        if next_run is None:
            from src.scheduler.time_parser import calculate_next_run
            next_run = calculate_next_run(schedule_type, schedule_value).timestamp()

        cursor = self.conn.execute(
            """INSERT INTO scheduled_tasks
               (name, description, game, task_type, task_payload,
                schedule_type, schedule_value, enabled, one_shot,
                next_run, last_run, last_result, run_count, max_runs,
                created_at, updated_at, slot_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, NULL, '', 0, ?, ?, ?, ?)""",
            (name, description, game, task_type,
             json.dumps(task_payload, ensure_ascii=False),
             schedule_type, schedule_value, int(one_shot),
             next_run, max_runs, now, now, slot_id),
        )
        self.conn.commit()
        logger.info("Created scheduled task #%d: %s (%s=%s, next=%s)",
                     cursor.lastrowid, name, schedule_type, schedule_value,
                     time.strftime("%Y-%m-%d %H:%M", time.localtime(next_run)))
        return cursor.lastrowid

    def get_all_enabled(self) -> list[dict[str, Any]]:
        """Return all enabled tasks, ordered by next_run."""
        self._ensure_tables()
        rows = self.conn.execute(
            "SELECT * FROM scheduled_tasks WHERE enabled = 1 ORDER BY next_run ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self) -> list[dict[str, Any]]:
        """Return all tasks (including disabled)."""
        self._ensure_tables()
        rows = self.conn.execute(
            "SELECT * FROM scheduled_tasks ORDER BY id ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_id(self, task_id: int) -> dict[str, Any] | None:
        """Return a single task by id, or None."""
        self._ensure_tables()
        row = self.conn.execute(
            "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_due_tasks(self, now_ts: float | None = None) -> list[dict[str, Any]]:
        """Return enabled tasks whose next_run is due (<= now + 1s grace)."""
        self._ensure_tables()
        now_ts = now_ts or time.time()
        # +1s grace window for clock skew
        rows = self.conn.execute(
            "SELECT * FROM scheduled_tasks WHERE enabled = 1 AND next_run <= ? ORDER BY next_run ASC",
            (now_ts + 1.0,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_next_run(self, task_id: int, next_run: float) -> None:
        """Update next_run for a task."""
        now = time.time()
        self.conn.execute(
            "UPDATE scheduled_tasks SET next_run = ?, updated_at = ? WHERE id = ?",
            (next_run, now, task_id),
        )
        self.conn.commit()

    def record_execution(self, task_id: int, started_at: float) -> None:
        """Mark a task as having been launched (increment run_count, set last_run)."""
        now = time.time()
        self.conn.execute(
            "UPDATE scheduled_tasks SET last_run = ?, run_count = run_count + 1, updated_at = ? WHERE id = ?",
            (started_at, now, task_id),
        )
        self.conn.commit()

    def record_completion(self, task_id: int, success: bool, *,
                          result_summary: str = "",
                          error_message: str = "",
                          next_run: float | None = None,
                          delete_one_shot: bool = False) -> None:
        """Update task after execution completes.

        If delete_one_shot is True, the task row is deleted.
        Otherwise next_run is updated (if provided).
        """
        now = time.time()
        result_json = json.dumps({"success": success, "summary": result_summary}, ensure_ascii=False)

        if delete_one_shot:
            self.conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
            logger.info("One-shot task #%d deleted after completion", task_id)
        else:
            if next_run is not None:
                self.conn.execute(
                    "UPDATE scheduled_tasks SET last_result = ?, next_run = ?, updated_at = ? WHERE id = ?",
                    (result_json, next_run, now, task_id),
                )
            else:
                self.conn.execute(
                    "UPDATE scheduled_tasks SET last_result = ?, updated_at = ? WHERE id = ?",
                    (result_json, now, task_id),
                )
        self.conn.commit()

    def set_enabled(self, task_id: int, enabled: bool) -> bool:
        """Enable or disable a task. Returns True if the task was found."""
        now = time.time()
        cursor = self.conn.execute(
            "UPDATE scheduled_tasks SET enabled = ?, updated_at = ? WHERE id = ?",
            (int(enabled), now, task_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def delete(self, task_id: int) -> bool:
        """Delete a task and its history. Returns True if found."""
        # Delete history first (belt-and-suspenders: also relies on CASCADE)
        self.conn.execute(
            "DELETE FROM schedule_history WHERE task_id = ?", (task_id,)
        )
        cursor = self.conn.execute(
            "DELETE FROM scheduled_tasks WHERE id = ?", (task_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ---- Schedule History ----

    def log_history(self, task_id: int, started_at: float, finished_at: float, *,
                    success: bool = False,
                    result_summary: str = "",
                    error_message: str = "",
                    iterations: int = 0,
                    device_serial: str = "",
                    slot_id: str = "") -> int:
        """Insert an execution history row. Returns the new row id."""
        duration = finished_at - started_at
        cursor = self.conn.execute(
            """INSERT INTO schedule_history
               (task_id, started_at, finished_at, success, result_summary,
                error_message, iterations, duration_seconds, device_serial, slot_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (task_id, started_at, finished_at, int(success),
             result_summary, error_message, iterations, duration, device_serial, slot_id),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_history(self, task_id: int, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent execution history for a task."""
        self._ensure_tables()
        rows = self.conn.execute(
            "SELECT * FROM schedule_history WHERE task_id = ? ORDER BY started_at DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def cleanup_old_history(self, retention_days: int | None = None) -> int:
        """Delete schedule_history rows older than retention_days.

        Args:
            retention_days: Days of history to keep. Defaults to config value.

        Returns:
            Number of rows deleted.
        """
        from config.settings import config as _cfg
        if retention_days is None:
            retention_days = _cfg.scheduler.history_retention_days

        cutoff = time.time() - (retention_days * 86400)
        cursor = self.conn.execute(
            "DELETE FROM schedule_history WHERE started_at < ?", (cutoff,)
        )
        self.conn.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.info("Cleaned up %d old schedule_history rows (retention=%dd)",
                       deleted, retention_days)
        return deleted

    def vacuum(self) -> None:
        """Run PRAGMA optimize for query planner statistics (was: VACUUM).

        Note: this does NOT reclaim disk space.  Use conn.execute('VACUUM')
        if disk space reclamation is needed (requires ~2x free space).
        """
        self.conn.execute("PRAGMA optimize")
        logger.debug("Schedule DB optimized")

    def optimize(self) -> None:
        """Alias for vacuum() — correct name for what this actually does."""
        return self.vacuum()

    def cancel_all_queued(self) -> int:
        """Cancel ALL queued tasks left over from a previous session.

        A bot restart means in-memory scheduling state is gone, so any
        ``queued`` row in task_queue is an orphan from a prior crash or
        unclean shutdown — it should not silently resurrect on the next
        startup.
        """
        self._ensure_tables()
        cursor = self.conn.execute(
            "UPDATE task_queue SET status = 'cancelled' WHERE status = 'queued'"
        )
        self.conn.commit()
        n = cursor.rowcount
        if n:
            logger.info("Cleaned up %d stale queued task(s) from previous session", n)
        return n

    # ---- Persistent Task Queue (Phase 3) ----

    def enqueue_task_db(self, slot_id: str, task_description: str,
                        game: str = "arknights", task_type: str = "custom",
                        priority: int = 5) -> int:
        """Persist a queued task. Returns the new row id."""
        self._ensure_tables()
        now = time.time()
        cursor = self.conn.execute(
            """INSERT INTO task_queue
               (slot_id, task_description, task_type, game, priority, queued_at, status)
               VALUES (?, ?, ?, ?, ?, ?, 'queued')""",
            (slot_id, task_description, task_type, game, priority, now),
        )
        self.conn.commit()
        return cursor.lastrowid

    def dequeue_task_db(self, slot_id: str) -> dict | None:
        """Pop the highest-priority queued task for a slot. Returns dict or None."""
        self._ensure_tables()
        row = self.conn.execute(
            """SELECT id, slot_id, task_description, task_type, game, priority, queued_at
               FROM task_queue
               WHERE slot_id = ? AND status = 'queued'
               ORDER BY priority ASC, queued_at ASC
               LIMIT 1""",
            (slot_id,),
        ).fetchone()
        if row is None:
            return None
        # Mark as done
        self.conn.execute(
            "UPDATE task_queue SET status = 'done' WHERE id = ?", (row["id"],)
        )
        self.conn.commit()
        return dict(row)

    def get_queued_tasks(self, slot_id: str) -> list[dict]:
        """Return all queued tasks for a slot (ordered by priority)."""
        self._ensure_tables()
        rows = self.conn.execute(
            """SELECT * FROM task_queue
               WHERE slot_id = ? AND status = 'queued'
               ORDER BY priority ASC, queued_at ASC""",
            (slot_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_slot_queue(self, slot_id: str) -> int:
        """Mark all queued tasks for a slot as cancelled. Returns count."""
        self._ensure_tables()
        cursor = self.conn.execute(
            "UPDATE task_queue SET status = 'cancelled' WHERE slot_id = ? AND status = 'queued'",
            (slot_id,),
        )
        self.conn.commit()
        return cursor.rowcount

    def recover_orphaned_tasks(self, slot_ids: list[str]) -> list[dict]:
        """Recover tasks left in 'queued' state after a crash.
        Only returns tasks for slot_ids that are currently available.
        """
        self._ensure_tables()
        if not slot_ids:
            return []
        placeholders = ",".join("?" * len(slot_ids))
        rows = self.conn.execute(
            f"""SELECT * FROM task_queue
               WHERE slot_id IN ({placeholders}) AND status = 'queued'
               ORDER BY priority ASC, queued_at ASC""",
            slot_ids,
        ).fetchall()
        return [dict(r) for r in rows]


schedule_db = ScheduleDB()
