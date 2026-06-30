"""MEMORY.md file management for explicit user preferences only.

Boundary: MEMORY.md stores ONLY user-declared preferences.
Derived preferences are computed from SQLite history, not persisted to MEMORY.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_PATH = Path("MEMORY.md")


class MemFile:
    """Read/write MEMORY.md for persistent user preferences."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or MEMORY_PATH

    def read(self) -> str:
        """Read the full MEMORY.md content."""
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8")

    def write(self, content: str) -> None:
        """Replace MEMORY.md content."""
        self.path.write_text(content, encoding="utf-8")
        logger.info("MEMORY.md updated")

    def append(self, line: str) -> None:
        """Append a line to MEMORY.md."""
        current = self.read()
        if current and not current.endswith("\n"):
            current += "\n"
        current += f"- {line}\n"
        self.write(current)

    @property
    def exists(self) -> bool:
        return self.path.exists()


memfile = MemFile()
