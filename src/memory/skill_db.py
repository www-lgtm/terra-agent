"""Skill DB: skills_data + skills_fts tables.

Manages the skill index independently from execution history and memories.
Moves index_skill_fts() and _cleanup_stale_skills() from router.py to
resolve the circular import between SkillManager and router.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import config
from src.memory.base_db import BaseDB

logger = logging.getLogger(__name__)

DB_PATH = Path(config.DATA_DIR) / "memory" / "terra.db"


class SkillDB(BaseDB):
    """SQLite database for skill indexing (FTS5 search)."""

    _startup_done = False

    def __init__(self, path: Path | None = None) -> None:
        super().__init__(path or DB_PATH)
        self._init_tables()
        self._run_startup_maintenance()
        # Release connection so subsequent modules can acquire write locks
        self.close()

    def _run_startup_maintenance(self) -> None:
        """One-time startup cleanup (once per process)."""
        if SkillDB._startup_done:
            return
        SkillDB._startup_done = True
        try:
            self._migrate_orphan_skills_db()
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "Orphan skills DB merge skipped")
        try:
            self.cleanup_stale_skills()
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "Stale skill cleanup skipped")
        try:
            self.ensure_indexed_from_disk()
        except Exception:
            from src.utils.errors import safe_log
            safe_log(logger, "warning", "Skill re-index skipped")

    def _init_tables(self) -> None:
        # Retry on "database is locked" — sibling databases (MemoryDB) may hold
        # a transient write lock during their startup maintenance.
        import time as _time
        last_err = None
        for attempt in range(5):
            try:
                self._init_tables_impl()
                return
            except Exception as e:
                last_err = e
                if "locked" in str(e).lower() and attempt < 4:
                    wait = 0.5 * (2 ** attempt)
                    logger.debug("SkillDB _init_tables locked, retrying in %.1fs (%d/5)",
                               wait, attempt + 1)
                    _time.sleep(wait)
                    # Reopen connection on retry — the old one may be in a bad state
                    self.close()
                else:
                    raise
        raise last_err  # type: ignore[misc]

    def _init_tables_impl(self) -> None:
        self.conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
                name, description, tags, body,
                content=skills_data,
                content_rowid=id
            );

            CREATE TABLE IF NOT EXISTS skills_data (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                body TEXT,
                game TEXT DEFAULT 'arknights',
                verified INTEGER DEFAULT 0,
                type TEXT DEFAULT 'guide',
                file_mtime REAL DEFAULT 0.0
            );
        """)
        self.migrate_column("skills_data", "verified", "INTEGER DEFAULT 0")
        self.migrate_column("skills_data", "type", "TEXT DEFAULT 'guide'")
        self.migrate_column("skills_data", "game", "TEXT DEFAULT 'arknights'")
        self.migrate_column("skills_data", "file_mtime", "REAL DEFAULT 0.0")
        # Backfill type for existing rows (one-time)
        self.conn.execute(
            "UPDATE skills_data SET type='script' WHERE type IS NULL AND verified=1"
        )
        self.conn.execute(
            "UPDATE skills_data SET type='guide' WHERE type IS NULL OR type=''"
        )
        # Backfill game for existing rows (one-time — all pre-migration skills are arknights)
        self.conn.execute(
            "UPDATE skills_data SET game='arknights' WHERE game IS NULL OR game=''"
        )
        self.conn.commit()

    def run_migration_from_old_db(self, old_db_path: Path) -> int:
        """Copy data from old monolithic terra.db into this DB."""
        import sqlite3 as _sqlite3

        if not old_db_path.exists():
            return 0

        old_conn = _sqlite3.connect(str(old_db_path))
        old_conn.row_factory = _sqlite3.Row
        try:
            rows = old_conn.execute("SELECT * FROM skills_data").fetchall()
        except _sqlite3.OperationalError:
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
                    "SELECT id FROM skills_data WHERE name = ?", (r_dict["name"],)
                ).fetchone()
                if existing:
                    continue
                self.conn.execute(
                    """INSERT INTO skills_data (name, description, tags, body, verified)
                       VALUES (?, ?, ?, ?, ?)""",
                    (r_dict["name"], r_dict.get("description", ""), r_dict.get("tags", ""),
                     r_dict.get("body", ""), r_dict.get("verified", 0)),
                )
                count += 1
            except Exception as e:
                logger.debug("Skipping skill migration row: %s", e)

        self.conn.execute("INSERT INTO skills_fts(skills_fts) VALUES('rebuild')")
        self.conn.commit()
        logger.info("Skill migration: %d rows from old DB → %s", count, self.path)
        return count

    def _migrate_orphan_skills_db(self) -> int:
        """One-time migration: merge data from orphan skills.db into terra.db.

        This runs AFTER skills_data and skills_fts tables are created
        (unlike MemoryDB.merge_orphan_databases which runs earlier in the
        import chain).  After migration, deletes the orphan file.

        Guarded by learning_state key 'orphan_migration_complete' (set by
        MemoryDB.merge_orphan_databases).  Returns count of rows merged.
        """
        orphan_path = self.path.parent / "skills.db"
        if not orphan_path.exists():
            return 0

        # Check if MemoryDB already handled the migration
        from src.memory.memory_db import memory_db
        already_done = memory_db.get_learning_state("orphan_migration_complete")
        if already_done != "1":
            logger.debug("Orphan skills migration deferred — waiting for MemoryDB marker")
            return 0

        import sqlite3 as _sqlite3
        try:
            orphan_conn = _sqlite3.connect(str(orphan_path))
            orphan_conn.row_factory = _sqlite3.Row
            try:
                rows = orphan_conn.execute("SELECT * FROM skills_data").fetchall()
            except _sqlite3.OperationalError:
                return 0
            finally:
                orphan_conn.close()
        except Exception as e:
            logger.debug("Failed to read orphan skills.db: %s", e)
            return 0

        if not rows:
            try:
                orphan_path.unlink()
                logger.info("Deleted empty orphan database: skills.db")
            except OSError:
                pass
            return 0

        columns = [d[0] for d in rows[0].keys()]
        col_list = ", ".join(columns)
        placeholders = ", ".join("?" * len(columns))

        count = 0
        for row in rows:
            try:
                r_dict = dict(row)
                existing = self.conn.execute(
                    "SELECT id FROM skills_data WHERE name = ?", (r_dict["name"],)
                ).fetchone()
                if existing:
                    continue
                self.conn.execute(
                    f"INSERT INTO skills_data ({col_list}) VALUES ({placeholders})",
                    tuple(row),
                )
                count += 1
            except Exception as e:
                logger.debug("Skipping skills.db row: %s", e)

        if count:
            self.conn.execute("INSERT INTO skills_fts(skills_fts) VALUES('rebuild')")
            self.conn.commit()
            logger.info("Skills migration: %d rows from orphan skills.db → terra.db", count)

        try:
            orphan_path.unlink()
            logger.info("Deleted orphan database: skills.db")
        except OSError:
            pass

        return count

    # ---- Index operations (moved from router.py) ----

    def index_skill(self, name: str, description: str, tags: str, body: str,
                    verified: bool = False, skill_type: str = "guide",
                    game: str = "arknights") -> None:
        """Index or update a skill in the FTS5 table."""
        try:
            existing = self.conn.execute(
                "SELECT id FROM skills_data WHERE name = ?", (name,)
            ).fetchone()

            # Read actual file mtime so ensure_indexed_from_disk can detect
            # manual edits made outside of SkillManager.save()
            file_mtime = 0.0
            try:
                from pathlib import Path
                skills_dir = Path("data/skills") / game
                md_path = skills_dir / f"{name}.md"
                if md_path.exists():
                    file_mtime = md_path.stat().st_mtime
            except Exception:
                pass

            if existing:
                self.conn.execute(
                    """UPDATE skills_data SET description=?, tags=?, body=?, verified=?,
                       type=?, game=?, file_mtime=? WHERE name=?""",
                    (description, tags, body, int(verified), skill_type, game, file_mtime, name),
                )
                self.conn.execute("INSERT INTO skills_fts(skills_fts) VALUES('rebuild')")
            else:
                self.conn.execute(
                    """INSERT INTO skills_data (name, description, tags, body, verified, type, game, file_mtime)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (name, description, tags, body, int(verified), skill_type, game, file_mtime),
                )
                self.conn.execute("INSERT INTO skills_fts(skills_fts) VALUES('rebuild')")
            self.conn.commit()
            logger.info("Indexed skill '%s' in FTS5 (game=%s, verified=%s, type=%s, mtime=%.0f)",
                       name, game, verified, skill_type, file_mtime)
        except Exception as e:
            logger.warning("Failed to index skill '%s' in FTS5: %s", name, e)

    def cleanup_stale_skills(self) -> None:
        """Delete skills_data rows whose .md files no longer exist on disk."""
        from config.settings import config as _config

        skills_dir = _config.DATA_DIR / "skills"
        try:
            rows = self.conn.execute("SELECT id, name FROM skills_data").fetchall()
            stale_ids: list[int] = []
            for r in rows:
                # Search all game subdirectories + _shared for the .md file
                found = False
                if skills_dir.exists():
                    for md_path in skills_dir.rglob(f"{r['name']}.md"):
                        found = True
                        break
                if not found:
                    stale_ids.append(r["id"])
            if stale_ids:
                placeholders = ",".join("?" * len(stale_ids))
                # Delete stale rows, then rebuild FTS to sync the index.
                # FTS5 content tables have NO foreign key constraints —
                # only a single rebuild is needed AFTER deletion.
                self.conn.execute(f"DELETE FROM skills_data WHERE id IN ({placeholders})", stale_ids)
                self.conn.execute("INSERT INTO skills_fts(skills_fts) VALUES('rebuild')")
                self.conn.commit()
                logger.info("Cleaned %d stale skill DB entries", len(stale_ids))
        except Exception as e:
            logger.debug("Stale skill cleanup skipped: %s", e)

    def remove_skill(self, name: str) -> bool:
        """Delete a skill from the index. Returns True if deleted."""
        try:
            cur = self.conn.execute("DELETE FROM skills_data WHERE name = ?", (name,))
            if cur.rowcount:
                self.conn.execute("INSERT INTO skills_fts(skills_fts) VALUES('rebuild')")
                self.conn.commit()
                logger.info("Removed skill '%s' from index", name)
                return True
            return False
        except Exception as e:
            logger.warning("Failed to remove skill '%s': %s", name, e)
            return False

    def ensure_indexed_from_disk(self) -> int:
        """Scan skills directory and re-index files not yet in DB, plus files
        whose mtime has changed (manually edited outside SkillManager.save).

        Extracts game from the file path: data/skills/<game>/<name>.md

        Returns the number of skills (re-)indexed.
        """
        from config.settings import config as _config
        from src.skills.parser import parse_skill_md
        import time as _time

        skills_dir = _config.DATA_DIR / "skills"
        if not skills_dir.exists():
            return 0

        count = 0
        for md_path in skills_dir.rglob("*.md"):
            if md_path.name.startswith("_"):     # Internal files like _explore_graph.json
                continue
            if not md_path.name.endswith(".md"):
                continue

            name = md_path.stem
            file_mtime = md_path.stat().st_mtime
            existing = self.conn.execute(
                "SELECT id, file_mtime FROM skills_data WHERE name = ?", (name,)
            ).fetchone()

            # Skip if already indexed AND mtime hasn't changed
            if existing and abs(existing["file_mtime"] - file_mtime) < 1.0:
                continue

            # Extract game from path: data/skills/<game>/<name>.md
            game = md_path.parent.name if md_path.parent != skills_dir else "arknights"
            try:
                content = md_path.read_text(encoding="utf-8")
                skill = parse_skill_md(content)
                # Frontmatter game overrides path-derived game
                resolved_game = skill.get("game", game) or game
                self.index_skill(
                    name=skill.get("name", name),
                    description=skill.get("description", ""),
                    tags=", ".join(skill.get("tags", [])),
                    body=skill.get("body", ""),
                    verified=skill.get("verified", False),
                    skill_type=skill.get("type", "script" if skill.get("verified") else "guide"),
                    game=resolved_game,
                )
                count += 1
                logger.info("Re-indexed skill '%s' from disk (mtime changed)", name)
            except Exception as e:
                logger.debug("Failed to index skill from disk: %s (%s)", name, e)

        if count:
            logger.info("Re-indexed %d skills from disk", count)
        return count


skill_db = SkillDB()
# Auto-reindex on startup: new files + files modified since last index.
# The mtime check in ensure_indexed_from_disk skips unchanged files,
# so this is cheap for the common case (re-reads only edited files).
try:
    skill_db.ensure_indexed_from_disk()
except Exception:
    pass
