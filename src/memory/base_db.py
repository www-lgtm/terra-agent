"""Shared SQLite connection management for domain-specific databases.

Each domain (execution, skills, memories, schedules) owns its own SQLite
file, with WAL mode and consistent PRAGMA settings applied via BaseDB.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseDB:
    """Base class for SQLite databases with connection management.

    Subclasses define their own DB file path and table schemas.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(":memory:")
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def migrate_column(self, table: str, column: str, col_type: str = "INTEGER DEFAULT 0") -> None:
        """Safely add a column if it doesn't already exist.

        Deprecated: prefer using versioned migrations via _run_migrations().
        """
        try:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists
        except Exception:
            logger.warning(
                "Unexpected error during migration: %s.%s", table, column, exc_info=True
            )

    def _run_migrations(self, migrations: list[tuple[int, Callable]]) -> None:
        """Run versioned migrations using PRAGMA user_version.

        Each migration is a (target_version, callable(conn)) tuple.
        Migrations are run in order by version.  The user_version is
        updated after each successful migration.

        Idempotent — if current version >= target, the migration is skipped.

        Example:
            MIGRATIONS = [
                (1, _migrate_v1_add_columns),
                (2, _migrate_v2_add_learning_state),
            ]
            self._run_migrations(MIGRATIONS)
        """
        current = self.conn.execute("PRAGMA user_version").fetchone()[0]
        for target_version, migrate_fn in sorted(migrations, key=lambda m: m[0]):
            if current < target_version:
                try:
                    migrate_fn(self.conn)
                    self.conn.execute(f"PRAGMA user_version = {target_version}")
                    self.conn.commit()
                    logger.info(
                        "Migration to v%d complete (%s)",
                        target_version,
                        migrate_fn.__name__,
                    )
                except Exception:
                    logger.error(
                        "Migration v%d (%s) failed — aborting migration chain",
                        target_version,
                        migrate_fn.__name__,
                        exc_info=True,
                    )
                    raise
        if current < migrations[-1][0] if migrations else current:
            final = self.conn.execute("PRAGMA user_version").fetchone()[0]
            logger.info("DB at %s: user_version %d (was %d)", self.path.name, final, current)
