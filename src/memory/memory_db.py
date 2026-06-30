"""Memory DB: memories_data + memories_fts tables.

Manages the memory index independently from execution history and skills.
Migrates data from the old monolithic terra.db on first startup.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from config.settings import config
from src.memory.base_db import BaseDB

logger = logging.getLogger(__name__)

# ── Versioned migrations (Phase 7) ────────────────────────────────


def _migrate_v1_add_columns(conn) -> None:
    """Add Phase 1 columns to memories_data for databases created earlier."""
    import sqlite3
    for col, col_type in [
        ("screen_hash", "TEXT DEFAULT NULL"),
        ("injected_count", "INTEGER DEFAULT 0"),
        ("injected_success_count", "INTEGER DEFAULT 0"),
        ("merge_count", "INTEGER DEFAULT 0"),
        ("help_count", "INTEGER DEFAULT 0"),
        ("harm_count", "INTEGER DEFAULT 0"),
        ("last_helpful_at", "REAL DEFAULT NULL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE memories_data ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists


def _migrate_v2_add_confidence_updated_at(conn) -> None:
    """Add confidence and updated_at columns to memories_data."""
    import sqlite3
    for col, col_type in [
        ("confidence", "REAL DEFAULT NULL"),
        ("updated_at", "REAL DEFAULT NULL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE memories_data ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists


def _migrate_v3_add_embedding(conn) -> None:
    """Add embedding BLOB column for vector search."""
    import sqlite3
    try:
        conn.execute("ALTER TABLE memories_data ADD COLUMN embedding BLOB DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Column already exists


def _migrate_v4_add_soft_delete_and_audit(conn) -> None:
    """Add deleted_at column for soft delete and memory_audit_log table."""
    import sqlite3
    try:
        conn.execute("ALTER TABLE memories_data ADD COLUMN deleted_at REAL DEFAULT NULL")
    except sqlite3.OperationalError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memory_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            memory_name TEXT NOT NULL,
            game TEXT NOT NULL DEFAULT 'arknights',
            action TEXT NOT NULL,
            changed_at REAL NOT NULL,
            FOREIGN KEY (memory_id) REFERENCES memories_data(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_audit_memory
            ON memory_audit_log(memory_id, changed_at);
    """)


def _migrate_v5_fix_audit_log_fk(conn) -> None:
    """Rebuild memory_audit_log with ON DELETE CASCADE to fix FK violations.

    If the table was created by the old v4 migration (no CASCADE), this
    recreates it with proper cascading deletes.  The operation is wrapped
    in an explicit transaction so partial failures roll back cleanly.
    """
    try:
        # Check if the table exists
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_audit_log'"
        ).fetchone()
        if not exists:
            return

        # Check if CASCADE is already applied (new v4 or prior v5 success)
        fk_info = conn.execute(
            "PRAGMA foreign_key_list('memory_audit_log')"
        ).fetchall()
        if fk_info and any(row["on_delete"] == "CASCADE" for row in fk_info):
            return  # Already fixed

        # Rebuild: all-or-nothing in an explicit transaction.
        # Individual execute() calls share the same transaction because
        # auto-commit is suppressed after the explicit BEGIN.
        conn.execute("BEGIN")
        try:
            conn.execute("""
                CREATE TABLE memory_audit_log_tmp (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL,
                    memory_name TEXT NOT NULL,
                    game TEXT NOT NULL DEFAULT 'arknights',
                    action TEXT NOT NULL,
                    changed_at REAL NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories_data(id) ON DELETE CASCADE
                )
            """)
            conn.execute(
                "INSERT INTO memory_audit_log_tmp SELECT * FROM memory_audit_log"
            )
            conn.execute("DROP TABLE memory_audit_log")
            conn.execute(
                "ALTER TABLE memory_audit_log_tmp RENAME TO memory_audit_log"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_memory"
                " ON memory_audit_log(memory_id, changed_at)"
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    except Exception as e:
        logger.debug("Migration v5 (audit FK cascade) skipped or failed: %s", e)


MEMORY_DB_MIGRATIONS: list[tuple[int, callable]] = [
    (1, _migrate_v1_add_columns),
    (2, _migrate_v2_add_confidence_updated_at),
    (3, _migrate_v3_add_embedding),
    (4, _migrate_v4_add_soft_delete_and_audit),
    (5, _migrate_v5_fix_audit_log_fk),
]

DB_PATH = Path(config.DATA_DIR) / "memory" / "terra.db"


class MemoryDB(BaseDB):
    """SQLite database for memory (experience/insight) indexing."""

    _startup_done = False

    def __init__(self, path: Path | None = None) -> None:
        super().__init__(path or DB_PATH)
        self._init_tables()
        self._fts_dirty = False
        self._fts_mutex = threading.Lock()
        self._run_startup_maintenance()
        # Release the connection so sibling databases (SkillDB) can acquire
        # a write lock during their own initialization. The connection is
        # recreated lazily on next access via the conn property.
        self.close()

    def mark_fts_dirty(self) -> None:
        """Mark FTS5 index as dirty — rebuild deferred to next search/task-end."""
        with self._fts_mutex:
            self._fts_dirty = True

    def rebuild_fts_if_dirty(self) -> None:
        """Rebuild FTS5 index only if marked dirty (batched rebuild)."""
        with self._fts_mutex:
            if self._fts_dirty:
                self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                self.conn.commit()
                self._fts_dirty = False

    def cleanup_old_logs(self, keep_days: int = 30) -> int:
        """Delete old injection_log and task_executions rows."""
        import time as _time
        cutoff = _time.time() - keep_days * 86400
        deleted_inj = self.conn.execute(
            "DELETE FROM injection_log WHERE injected_at < ?", (cutoff,),
        ).rowcount
        deleted_tasks = self.conn.execute(
            "DELETE FROM task_executions WHERE started_at < ?", (cutoff,),
        ).rowcount
        self.conn.commit()
        total = deleted_inj + deleted_tasks
        if total:
            logger.info("Cleaned up %d old log rows (%d injection_log, %d task_executions)",
                       total, deleted_inj, deleted_tasks)
        return total

    def _run_startup_maintenance(self) -> None:
        """One-time startup cleanup (once per process)."""
        if MemoryDB._startup_done:
            return
        MemoryDB._startup_done = True
        try:
            self.merge_orphan_databases()
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "Orphan DB merge skipped")
        try:
            self.clear_legacy_llm_memories("arknights")
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "Legacy memory cleanup skipped")
        try:
            self.cleanup_orphans()
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "Orphan cleanup skipped")
        try:
            self.ensure_indexed_from_disk()
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "Disk re-index skipped")
        try:
            self._index_pending_embeddings()
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "Embedding index skipped")

    def _init_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories_data (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                game TEXT NOT NULL DEFAULT 'arknights',
                tags TEXT,
                body TEXT,
                source TEXT DEFAULT 'llm_discovery',
                created TEXT NOT NULL,
                updated_at REAL DEFAULT NULL,
                hits INTEGER DEFAULT 0,
                screen_hash TEXT DEFAULT NULL,
                injected_count INTEGER DEFAULT 0,
                injected_success_count INTEGER DEFAULT 0,
                merge_count INTEGER DEFAULT 0,
                help_count INTEGER DEFAULT 0,
                harm_count INTEGER DEFAULT 0,
                last_helpful_at REAL DEFAULT NULL,
                confidence REAL DEFAULT NULL,
                embedding BLOB DEFAULT NULL,
                deleted_at REAL DEFAULT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                name, tags, body,
                content=memories_data,
                content_rowid=id
            );

            CREATE TABLE IF NOT EXISTS injection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                task_id INTEGER NOT NULL,
                injected_at REAL NOT NULL,
                screen_dhash TEXT,
                was_helpful INTEGER DEFAULT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories_data(id)
            );
            CREATE INDEX IF NOT EXISTS idx_injection_log_memory
                ON injection_log(memory_id, injected_at);

            CREATE TABLE IF NOT EXISTS task_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at REAL,
                finished_at REAL,
                game TEXT NOT NULL DEFAULT 'arknights',
                task_type TEXT NOT NULL DEFAULT 'unknown',
                task_description TEXT,
                success INTEGER DEFAULT 0,
                iterations INTEGER DEFAULT 0,
                duration_seconds REAL DEFAULT 0,
                failure_signal_types TEXT,
                skill_name TEXT,
                skill_fast_chain_success INTEGER DEFAULT NULL,
                user_interrupted INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS learning_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                memory_name TEXT NOT NULL,
                game TEXT NOT NULL DEFAULT 'arknights',
                action TEXT NOT NULL,
                changed_at REAL NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories_data(id)
            );
            CREATE INDEX IF NOT EXISTS idx_audit_memory
                ON memory_audit_log(memory_id, changed_at);
        """)

        # ── Versioned migrations (Phase 7) ──
        self._run_migrations(MEMORY_DB_MIGRATIONS)
        self.conn.commit()

    def run_migration_from_old_db(self, old_db_path: Path) -> int:
        """Copy data from old monolithic terra.db into this DB.

        Returns the number of rows migrated, or 0 if no old data exists.
        """
        import sqlite3 as _sqlite3

        if not old_db_path.exists():
            logger.debug("No old DB at %s — skipping memory migration", old_db_path)
            return 0

        old_conn = _sqlite3.connect(str(old_db_path))
        old_conn.row_factory = _sqlite3.Row
        try:
            rows = old_conn.execute("SELECT * FROM memories_data").fetchall()
        except _sqlite3.OperationalError:
            logger.debug("No memories_data table in old DB — nothing to migrate")
            return 0
        finally:
            old_conn.close()

        if not rows:
            return 0

        count = 0
        for r in rows:
            try:
                r_dict = dict(r)
                existing = self.conn.execute(
                    "SELECT id FROM memories_data WHERE name = ? AND game = ?",
                    (r_dict["name"], r_dict.get("game", "arknights")),
                ).fetchone()
                if existing:
                    continue  # Already migrated or manually created

                self.conn.execute(
                    """INSERT INTO memories_data (name, game, tags, body, source, created, hits,
                       screen_hash, injected_count, injected_success_count, merge_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        r_dict["name"], r_dict.get("game", "arknights"), r_dict.get("tags", ""), r_dict.get("body", ""),
                        r_dict.get("source", "llm_discovery"), r_dict.get("created", ""), r_dict.get("hits", 0),
                        r_dict.get("screen_hash"), r_dict.get("injected_count", 0),
                        r_dict.get("injected_success_count", 0), r_dict.get("merge_count", 0),
                    ),
                )
                count += 1
            except Exception as e:
                logger.debug("Skipping memory migration row: %s", e)

        self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        self.conn.commit()
        logger.info("Memory migration: %d rows from old DB → %s", count, self.path)
        return count

    def merge_orphan_databases(self) -> int:
        """One-time migration: merge data from orphan memories.db/skills.db
        into terra.db, then delete the orphan files.

        Background: the DB architecture underwent v1(monolithic terra.db) →
        v2(split: memories.db, skills.db, terra.db for history) → v3(all back
        to terra.db).  Orphan databases from v2 contain valuable historical
        data (injection_log, task_executions, learning_state) that must be
        recovered for the learning feedback loop.

        This runs exactly ONCE, guarded by learning_state key
        'orphan_migration_complete'.

        Returns total rows merged.
        """
        already_done = self.get_learning_state("orphan_migration_complete")
        if already_done == "1":
            logger.debug("Orphan migration already complete — skipping")
            return 0

        mem_dir = self.path.parent
        import sqlite3 as _sqlite3

        total_merged = 0

        # ── Migrate memories.db → terra.db ──
        orphan_mem = mem_dir / "memories.db"
        if orphan_mem.exists():
            try:
                orphan_conn = _sqlite3.connect(str(orphan_mem))
                orphan_conn.row_factory = _sqlite3.Row
                try:
                    total_merged += self._merge_table(
                        orphan_conn, "injection_log",
                        conflict_cols=("memory_id", "task_id", "injected_at"),
                    )
                    total_merged += self._merge_table(
                        orphan_conn, "task_executions",
                        conflict_cols=None,  # Auto-increment IDs
                    )
                    total_merged += self._merge_learning_state(orphan_conn)
                finally:
                    orphan_conn.close()
                # Delete orphan after successful merge
                try:
                    orphan_mem.unlink()
                    logger.info("Deleted orphan database: memories.db")
                except OSError:
                    pass
            except Exception as e:
                logger.warning("Failed to merge memories.db: %s", e)

        # ── Migrate skills.db → terra.db ──
        # Deferred: skills_data/skills_fts tables are created by SkillDB.__init__(),
        # which runs AFTER MemoryDB.__init__() in the import order.  We record a
        # learning_state marker here and SkillDB._run_startup_maintenance() picks it up.
        orphan_sk = mem_dir / "skills.db"
        if orphan_sk.exists():
            self.set_learning_state("orphan_skills_pending", "1")

        # Always set flag, even if 0 merged (nothing to migrate or files missing)
        self.set_learning_state("orphan_migration_complete", "1")
        if total_merged:
            self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self.conn.commit()
            logger.info("Orphan migration: %d total rows merged into terra.db", total_merged)
        return total_merged

    def _merge_table(
        self,
        orphan_conn: "sqlite3.Connection",
        table: str,
        conflict_cols: tuple[str, ...] | None = None,
    ) -> int:
        """Merge rows from an orphan database table into the current terra.db.

        For tables with conflict_cols, checks existence before INSERT to avoid
        duplicates (INSERT OR IGNORE is unreliable without a UNIQUE index).
        For auto-increment tables (conflict_cols=None), just inserts.

        Returns count of inserted rows.
        """
        try:
            rows = orphan_conn.execute(f"SELECT * FROM {table}").fetchall()
        except Exception:
            return 0
        if not rows:
            return 0

        columns = [d[0] for d in rows[0].keys()]
        col_list = ", ".join(columns)
        placeholders = ", ".join("?" * len(columns))

        count = 0
        for row in rows:
            try:
                # Dedup: check existence via conflict columns first
                if conflict_cols:
                    where_clauses: list[str] = []
                    where_values: list = []
                    for col in conflict_cols:
                        if col in columns:
                            idx = columns.index(col)
                            where_clauses.append(f"{col} = ?")
                            where_values.append(row[idx])
                    if where_clauses:
                        existing = self.conn.execute(
                            f"SELECT 1 FROM {table} WHERE {' AND '.join(where_clauses)} LIMIT 1",
                            where_values,
                        ).fetchone()
                        if existing:
                            continue  # Already exists — skip

                self.conn.execute(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
                    tuple(row),
                )
                if self.conn.lastrowid:
                    count += 1
            except Exception as e:
                logger.debug("Skipping %s row (conflict or schema mismatch): %s", table, e)
        self.conn.commit()
        logger.info("Merged %d rows into %s from orphan DB", count, table)
        return count

    def _merge_learning_state(self, orphan_conn: "sqlite3.Connection") -> int:
        """Merge learning_state rows, preserving existing values (newer wins)."""
        try:
            rows = orphan_conn.execute(
                "SELECT key, value, updated_at FROM learning_state"
            ).fetchall()
        except Exception:
            return 0
        if not rows:
            return 0

        count = 0
        for row in rows:
            existing = self.conn.execute(
                "SELECT updated_at FROM learning_state WHERE key=?", (row["key"],)
            ).fetchone()
            if existing is not None:
                continue  # Key already exists in target — skip
            try:
                self.conn.execute(
                    "INSERT INTO learning_state (key, value, updated_at) VALUES (?, ?, ?)",
                    (row["key"], row["value"], row["updated_at"]),
                )
                count += 1
            except Exception as e:
                logger.debug("Skipping learning_state '%s': %s", row["key"], e)
        self.conn.commit()
        logger.info("Merged %d learning_state rows from orphan DB", count)
        return count

    def cleanup_orphans(self) -> int:
        """Delete DB rows whose .md files no longer exist on disk.

        Handles the case where a user manually deletes a memory file —
        the DB row becomes an orphan and must be cleaned up so FTS5
        search doesn't return dead entries.

        Returns count of deleted rows.
        """
        from config.settings import config as _config

        mem_dir = _config.DATA_DIR / "memories"
        rows = self.conn.execute(
            "SELECT id, name, game FROM memories_data WHERE deleted_at IS NULL"
        ).fetchall()

        orphan_ids: list[int] = []
        for r in rows:
            file_path = mem_dir / (r["game"] or "arknights") / f"{r['name']}.md"
            if not file_path.exists():
                orphan_ids.append(r["id"])
                logger.debug("Orphan memory DB row: %s/%s.md (file missing)",
                            r["game"], r["name"])

        if not orphan_ids:
            return 0

        placeholders = ",".join("?" * len(orphan_ids))
        # Rebuild FTS5 before deleting — content table and FTS index may be
        # out of sync from a previous crash or unclean shutdown.
        try:
            self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        except Exception:
            pass

        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                f"DELETE FROM memories_data WHERE id IN ({placeholders})", orphan_ids
            )
            self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self.conn.commit()
            logger.info("Cleaned up %d orphan memory DB rows (file missing)", len(orphan_ids))
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            try:
                self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                self.conn.commit()
            except Exception:
                pass
            logger.info("Orphan cleanup: %d rows identified, FTS rebuilt", len(orphan_ids))
        return len(orphan_ids)

    def ensure_indexed_from_disk(self) -> int:
        """Scan memories directory and re-index any .md files not yet in the DB.

        Mirrors skill_db.ensure_indexed_from_disk().  Handles the reverse
        case: a user manually copies a .md file into the memories directory
        and it needs to appear in FTS5 search.

        Returns the number of memories newly indexed.
        """
        from config.settings import config as _config
        from datetime import datetime, timezone

        mem_dir = _config.DATA_DIR / "memories"
        if not mem_dir.exists():
            return 0

        count = 0
        for md_path in mem_dir.rglob("*.md"):
            name = md_path.stem
            existing = self.conn.execute(
                "SELECT id FROM memories_data WHERE name = ?", (name,)
            ).fetchone()
            if existing:
                continue
            try:
                content = md_path.read_text(encoding="utf-8")
                # Parse YAML frontmatter
                meta: dict[str, str] = {}
                body = content
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        for line in parts[1].strip().split("\n"):
                            line = line.strip()
                            if ":" in line:
                                k, _, v = line.partition(":")
                                meta[k.strip()] = v.strip()
                        body = parts[2].strip()

                game = meta.get("game", "arknights")
                tags = meta.get("tags", "").strip("[] ")
                created = meta.get("created", datetime.now(tz=timezone.utc).isoformat())
                screen_hash = meta.get("screen_hash", "")

                self.conn.execute(
                    """INSERT INTO memories_data
                       (name, game, tags, body, source, created, screen_hash)
                       VALUES (?, ?, ?, ?, 'manual', ?, ?)""",
                    (name, game, tags, body, created, screen_hash or None),
                )
                count += 1
                logger.debug("Re-indexed memory from disk: %s/%s.md", game, name)
            except Exception as e:
                logger.debug("Failed to index memory from disk: %s (%s)", name, e)

        if count:
            self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self.conn.commit()
            logger.info("Re-indexed %d memories from disk", count)
        return count

    def _index_pending_embeddings(self) -> int:
        """Generate vector embeddings for memories that don't have them yet.

        Delegates to VectorStore which handles the sentence-transformers
        model.  No-op when the model is unavailable.
        """
        from src.memory.vector_store import get_vector_store
        vs = get_vector_store()
        return vs.index_all_pending(self)

    # ---- Audit log ----

    def log_audit(self, memory_id: int, memory_name: str, game: str,
                  action: str) -> None:
        """Record a memory lifecycle event (create/update/merge/delete/soft_delete)."""
        import time as _time
        try:
            self.conn.execute(
                """INSERT INTO memory_audit_log (memory_id, memory_name, game, action, changed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (memory_id, memory_name, game, action, _time.time()),
            )
            self.conn.commit()
        except Exception as e:
            logger.debug("Audit log write failed: %s", e)

    def soft_delete_memory(self, memory_id: int) -> bool:
        """Mark a memory as deleted (soft delete). Returns True on success."""
        import time as _time
        try:
            row = self.conn.execute(
                "SELECT name, game FROM memories_data WHERE id = ? AND deleted_at IS NULL",
                (memory_id,),
            ).fetchone()
            if not row:
                return False
            now = _time.time()
            self.conn.execute(
                "UPDATE memories_data SET deleted_at = ? WHERE id = ?",
                (now, memory_id),
            )
            self.log_audit(memory_id, row["name"], row["game"], "soft_delete")
            self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self.conn.commit()
            return True
        except Exception as e:
            logger.warning("Soft delete failed for memory %d: %s", memory_id, e)
            return False

    def purge_soft_deleted(self, older_than_days: int = 30) -> int:
        """Permanently delete soft-deleted memories older than N days.

        This is the hard-delete step — called periodically from
        cleanup_stale_memories.  Also deletes the .md files.

        Returns count of purged memories.
        """
        import time as _time
        cutoff = _time.time() - older_than_days * 86400
        rows = self.conn.execute(
            """SELECT id, name, game FROM memories_data
               WHERE deleted_at IS NOT NULL AND deleted_at < ?""",
            (cutoff,),
        ).fetchall()

        if not rows:
            return 0

        mem_dir = self._get_memories_dir("")
        for r in rows:
            try:
                file_path = mem_dir / (r["game"] or "arknights") / f"{r['name']}.md"
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass

        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(ids))
        # Delete from audit log first to avoid FK constraint violation
        self.conn.execute(
            f"DELETE FROM memory_audit_log WHERE memory_id IN ({placeholders})", ids,
        )
        self.conn.execute(
            f"DELETE FROM memories_data WHERE id IN ({placeholders})", ids,
        )
        self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        self.conn.commit()
        logger.info("Purged %d soft-deleted memories (older than %d days)",
                   len(ids), older_than_days)
        return len(ids)

    # ---- Task execution tracking (Phase 1 Learning Engine) ----

    def create_task_execution(self, game: str, task_type: str,
                             task_description: str = "") -> int:
        """Create a task_executions row. Returns the new row id."""
        import time as _time
        cursor = self.conn.execute(
            """INSERT INTO task_executions (started_at, game, task_type, task_description)
               VALUES (?, ?, ?, ?)""",
            (_time.time(), game, task_type, task_description),
        )
        self.conn.commit()
        return cursor.lastrowid

    def complete_task_execution(self, task_id: int, success: bool,
                                iterations: int, duration: float,
                                failure_signal_types: str = "",
                                skill_name: str = "",
                                skill_fast_chain_success: bool | None = None,
                                user_interrupted: bool = False) -> None:
        """Update task_executions with completion data."""
        import time as _time
        self.conn.execute(
            """UPDATE task_executions SET finished_at=?, success=?, iterations=?,
               duration_seconds=?, failure_signal_types=?, skill_name=?,
               skill_fast_chain_success=?, user_interrupted=?
               WHERE id=?""",
            (_time.time(), int(success), iterations, duration,
             failure_signal_types, skill_name,
             int(skill_fast_chain_success) if skill_fast_chain_success is not None else None,
             int(user_interrupted), task_id),
        )
        self.conn.commit()

    def log_injection(self, memory_id: int, task_id: int,
                      screen_dhash: str | None = None) -> int:
        """Log a memory injection event. Returns the injection_log row id."""
        import time as _time
        cursor = self.conn.execute(
            """INSERT INTO injection_log (memory_id, task_id, injected_at, screen_dhash)
               VALUES (?, ?, ?, ?)""",
            (memory_id, task_id, _time.time(), screen_dhash),
        )
        self.conn.commit()
        return cursor.lastrowid

    def score_injection(self, injection_id: int, was_helpful: bool) -> None:
        """Update injection_log with helpfulness verdict."""
        self.conn.execute(
            "UPDATE injection_log SET was_helpful=? WHERE id=?",
            (1 if was_helpful else -1, injection_id),
        )
        self.conn.commit()

    def update_memory_help_stats(self, memory_id: int) -> None:
        """Recompute help_count/harm_count for a memory from injection_log."""
        import time as _time
        row = self.conn.execute(
            """SELECT COUNT(*) as hc FROM injection_log
               WHERE memory_id=? AND was_helpful=1""",
            (memory_id,),
        ).fetchone()
        help_count = row["hc"] if row else 0

        row = self.conn.execute(
            """SELECT COUNT(*) as hc FROM injection_log
               WHERE memory_id=? AND was_helpful=-1""",
            (memory_id,),
        ).fetchone()
        harm_count = row["hc"] if row else 0

        # Get latest helpful timestamp
        row = self.conn.execute(
            """SELECT injected_at FROM injection_log
               WHERE memory_id=? AND was_helpful=1
               ORDER BY injected_at DESC LIMIT 1""",
            (memory_id,),
        ).fetchone()
        last_helpful = row["injected_at"] if row else None

        self.conn.execute(
            """UPDATE memories_data SET help_count=?, harm_count=?, last_helpful_at=?
               WHERE id=?""",
            (help_count, harm_count, last_helpful, memory_id),
        )
        self.conn.commit()

    # ---- Learning state (Phase 2 Pattern Miner) ----

    def get_learning_state(self, key: str) -> str | None:
        """Read a learning_state value."""
        row = self.conn.execute(
            "SELECT value FROM learning_state WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_learning_state(self, key: str, value: str) -> None:
        """Write a learning_state value."""
        import time as _time
        self.conn.execute(
            "INSERT OR REPLACE INTO learning_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, _time.time()),
        )
        self.conn.commit()

    def get_task_count_since(self, game: str, since_timestamp: float) -> int:
        """Count completed tasks since a timestamp."""
        row = self.conn.execute(
            """SELECT COUNT(*) as cnt FROM task_executions
               WHERE game=? AND finished_at >= ?""",
            (game, since_timestamp),
        ).fetchone()
        return row["cnt"] if row else 0

    def get_recent_task_executions(self, game: str = "arknights",
                                   limit: int = 50) -> list[dict]:
        """Get recent task execution records for pattern mining."""
        rows = self.conn.execute(
            """SELECT * FROM task_executions WHERE game=?
               ORDER BY finished_at DESC LIMIT ?""",
            (game, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Cold-start cleanup (Phase 1) ----

    def clear_legacy_llm_memories(self, game: str = "arknights") -> int:
        """One-time migration: delete pre-Phase-1 memories that have NO engagement
        data (never injected, no help/harm scores, never manually searched).

        This runs exactly ONCE per database, tracked via learning_state key
        'cold_start_migration_complete'.  Once the flag is set, subsequent
        calls are a no-op — the system must NOT delete its own learned memories.

        Preserves:
        - user-manual memories (source='manual')
        - action patterns (source='action_pattern')
        - pattern miner findings (source='pattern_miner')
        - memories tagged with '永久' or 'permanent'
        - memories with any engagement: injected_count > 0 OR hits > 0 OR
          help_count > 0 (these have been used and may be valuable)

        Only deletes truly stale legacy memories (injected=0, hits=0,
        help=0, harm=0) that were created before the Phase 1 system existed.

        Also deletes the corresponding .md files.

        Returns count of deleted memories.
        """
        # Guard: this is a one-time migration, not a recurring cleanup.
        # Once the flag is set, never run again — we must not delete learned
        # memories that the system itself created during normal operation.
        already_done = self.get_learning_state("cold_start_migration_complete")
        if already_done == "1":
            logger.debug("Cold-start migration already complete — skipping")
            return 0

        rows = self.conn.execute(
            """SELECT id, name, game, tags, source, injected_count, hits,
               help_count, harm_count FROM memories_data WHERE game=? AND deleted_at IS NULL""",
            (game,),
        ).fetchall()

        delete_ids: list[int] = []
        for r in rows:
            source = (r["source"] or "").lower()
            tags = (r["tags"] or "").lower()
            # Preserve: manual, action_pattern, pattern_miner, and permanently-tagged memories
            if source in ("manual", "action_pattern", "pattern_miner"):
                continue
            if "永久" in tags or "permanent" in tags:
                continue
            # CRITICAL: preserve memories with any engagement signal.
            # Memories injected at least once, searched manually, or with
            # help/harm feedback are real learned knowledge — not stale legacy.
            injected = r["injected_count"] or 0
            hits = r["hits"] or 0
            help_count = r["help_count"] or 0
            if injected > 0 or hits > 0 or help_count > 0:
                logger.debug(
                    "Preserving engaged memory '%s' (injected=%d hits=%d help=%d)",
                    r["name"], injected, hits, help_count,
                )
                continue
            delete_ids.append(r["id"])

        # Always set the flag so we never run this again, even if 0 deleted.
        self.set_learning_state("cold_start_migration_complete", "1")

        if not delete_ids:
            logger.info("Cold-start cleanup: no legacy memories to clear")
            return 0

        # Delete .md files
        mem_dir = self._get_memories_dir(game)
        for r in rows:
            if r["id"] in delete_ids:
                try:
                    file_path = mem_dir / f"{r['name']}.md"
                    if file_path.exists():
                        file_path.unlink()
                except Exception:
                    pass

        # Delete DB rows
        placeholders = ",".join("?" * len(delete_ids))
        self.conn.execute(
            f"DELETE FROM memories_data WHERE id IN ({placeholders})", delete_ids
        )
        self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        self.conn.commit()
        logger.info("Cold-start cleanup: deleted %d legacy memories (game=%s)",
                    len(delete_ids), game)
        return len(delete_ids)

    @staticmethod
    def _get_memories_dir(game: str) -> Path:
        from config.settings import config as _config
        return _config.DATA_DIR / "memories" / game


memory_db = MemoryDB()
