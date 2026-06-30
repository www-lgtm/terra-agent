"""Execution history DB: execution_history + game_state tables only.

Skills and memories have been migrated to their own databases
(skill_db.py and memory_db.py).  This module now focuses solely on
task execution logging and game state persistence.
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

DB_PATH = Path(config.DATA_DIR) / "memory" / "terra.db"

# One-time migration guard
_migration_done: bool = False


class HistoryDB(BaseDB):
    """SQLite database for execution history and game state."""

    def __init__(self, path: Path | None = None) -> None:
        super().__init__(path or DB_PATH)
        self._init_tables()
        self._run_one_time_migration()
        # Release connection so sibling databases can acquire write locks
        self.close()

    def _init_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS execution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                game TEXT NOT NULL,
                task_type TEXT NOT NULL,
                task_description TEXT,
                success INTEGER NOT NULL DEFAULT 0,
                iterations INTEGER,
                duration_seconds REAL,
                details TEXT
            );

            CREATE TABLE IF NOT EXISTS game_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        self.conn.commit()

    def _run_one_time_migration(self) -> None:
        """Migration from old monolithic DB to split DBs — DISABLED.

        The architecture has permanently moved back to a unified terra.db
        (MemoryDB, SkillDB, and HistoryDB all share the same SQLite file).
        The old migration logic would copy from terra.db → terra.db (no-op)
        and then DROP the tables — destroying live data.

        Kept as a no-op marker so the _migration_done guard still prevents
        the now-disabled logic from ever executing.
        """
        global _migration_done
        _migration_done = True

    # ---- Execution history ----

    def log_execution(
        self,
        game: str,
        task_type: str,
        task_description: str = "",
        success: bool = False,
        iterations: int = 0,
        duration_seconds: float = 0.0,
        details: dict | None = None,
    ) -> int:
        cursor = self.conn.execute(
            """INSERT INTO execution_history
               (timestamp, game, task_type, task_description, success, iterations, duration_seconds, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), game, task_type, task_description, int(success), iterations, duration_seconds,
             json.dumps(details or {}, ensure_ascii=False)),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_state(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM game_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO game_state (key, value, updated_at)
               VALUES (?, ?, ?)""",
            (key, value, time.time()),
        )
        self.conn.commit()

    def get_recent_history(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM execution_history ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_state_summary(self, game: str) -> str:
        """Build a text summary of recent state for LLM context."""
        recent = self.get_recent_history(5)
        if not recent:
            return "No recent activity."

        lines = [f"Recent {game} activity:"]
        for r in recent:
            status = "OK" if r["success"] else "FAIL"
            lines.append(f"  - [{status}] {r['task_type']}: {r['task_description'][:80]}")
        return "\n".join(lines)


history_db = HistoryDB()
