"""Observation session storage — manifest + frame I/O.

Each observation session is a directory under data/observations/{game}/{session_id}/
containing a manifest.json and numbered JPEG frames.  Click events are stored
inline in the manifest so the extraction LLM sees click coordinates alongside
each frame's OCR text.

Session lifecycle:
  - created/       — recording in progress
  - completed/     — stopped cleanly, guide extracted (sentinel: .completed)
  - interrupted/   — ADB disconnected or error (sentinel: .interrupted)
  - cancelled/     — user sent /stop (directory deleted immediately)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config.settings import config

logger = logging.getLogger(__name__)

_OBSERVATIONS_DIR = config.DATA_DIR / "observations"


@dataclass
class ClickRecord:
    """A single mouse click captured between two frames."""
    timestamp_s: float         # seconds since recording start
    desktop_x: int             # Windows desktop absolute X
    desktop_y: int             # Windows desktop absolute Y
    device_x: int              # mapped device pixel X
    device_y: int              # mapped device pixel Y

    def to_dict(self) -> dict:
        return {
            "timestamp_s": round(self.timestamp_s, 3),
            "desktop_xy": [self.desktop_x, self.desktop_y],
            "device_xy": [self.device_x, self.device_y],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClickRecord:
        return cls(
            timestamp_s=d["timestamp_s"],
            desktop_x=d["desktop_xy"][0],
            desktop_y=d["desktop_xy"][1],
            device_x=d["device_xy"][0],
            device_y=d["device_xy"][1],
        )


@dataclass
class ObsFrame:
    """One captured frame in an observation session."""
    index: int                  # 0-based, monotonically increasing
    filename: str               # e.g. "00001.jpg"
    timestamp_s: float          # seconds since recording start
    dhash: str                  # 16-char hex dHash
    ocr_texts: list[str] = field(default_factory=list)
    is_significant: bool = False
    hamming_from_prev: int | None = None  # None for first frame
    clicks_before: list[ClickRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {
            "index": self.index,
            "filename": self.filename,
            "timestamp_s": round(self.timestamp_s, 3),
            "dhash": self.dhash,
            "ocr_texts": self.ocr_texts,
            "is_significant": self.is_significant,
            "hamming_from_prev": self.hamming_from_prev,
            "clicks_before": [c.to_dict() for c in self.clicks_before],
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ObsFrame:
        return cls(
            index=d["index"],
            filename=d["filename"],
            timestamp_s=d["timestamp_s"],
            dhash=d["dhash"],
            ocr_texts=d.get("ocr_texts", []),
            is_significant=d.get("is_significant", False),
            hamming_from_prev=d.get("hamming_from_prev"),
            clicks_before=[ClickRecord.from_dict(c) for c in d.get("clicks_before", [])],
        )


@dataclass
class ObservationManifest:
    """Top-level metadata for one observation session."""
    session_id: str
    game: str
    device_serial: str
    task_name: str
    started_at: str           # ISO8601
    stopped_at: str = ""      # ISO8601, filled on stop()
    frame_count: int = 0
    significant_count: int = 0
    resolution: tuple[int, int] = (0, 0)
    frames: list[ObsFrame] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "game": self.game,
            "device_serial": self.device_serial,
            "task_name": self.task_name,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "frame_count": self.frame_count,
            "significant_count": self.significant_count,
            "resolution": {"w": self.resolution[0], "h": self.resolution[1]},
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ObservationManifest:
        return cls(
            session_id=d["session_id"],
            game=d["game"],
            device_serial=d["device_serial"],
            task_name=d["task_name"],
            started_at=d["started_at"],
            stopped_at=d.get("stopped_at", ""),
            frame_count=d.get("frame_count", 0),
            significant_count=d.get("significant_count", 0),
            resolution=(d.get("resolution", {}).get("w", 0),
                        d.get("resolution", {}).get("h", 0)),
            frames=[ObsFrame.from_dict(f) for f in d.get("frames", [])],
        )


# ── Public API ───────────────────────────────────────────────────────

def create_session(game: str, device_serial: str, task_name: str) -> ObservationManifest:
    """Create a new observation session directory and return its manifest.

    The session_id is derived from timestamp + random suffix.
    """
    _OBSERVATIONS_DIR.mkdir(parents=True, exist_ok=True)

    session_id = _generate_session_id()
    started_at = datetime.now(tz=timezone.utc).isoformat()

    manifest = ObservationManifest(
        session_id=session_id,
        game=game,
        device_serial=device_serial,
        task_name=task_name,
        started_at=started_at,
    )

    _save_manifest(manifest)
    _ensure_frames_dir(manifest)
    logger.info("Observation session created: %s/%s", game, session_id)
    return manifest


def load_manifest(game: str, session_id: str) -> ObservationManifest | None:
    """Load an existing observation session manifest."""
    manifest_path = _manifest_path(game, session_id)
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ObservationManifest.from_dict(data)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load manifest %s: %s", session_id, e)
        return None


def save_frame(manifest: ObservationManifest, frame: ObsFrame,
               image_bytes: bytes, update_disk: bool = False) -> Path:
    """Save a frame image and append its metadata to the manifest.

    Writes the JPEG image to disk.  Only rewrites manifest.json if
    update_disk=True (call every N frames and on stop to minimize IO).

    Returns the path to the saved frame file.
    """
    session_dir = _session_dir(manifest.game, manifest.session_id)
    frame_path = session_dir / "frames" / frame.filename

    # Write image
    frame_path.write_bytes(image_bytes)

    # Update manifest counts in memory
    manifest.frames.append(frame)
    manifest.frame_count = len(manifest.frames)
    if frame.is_significant:
        manifest.significant_count += 1

    # Persist manifest periodically or explicitly
    if update_disk:
        _save_manifest(manifest)

    return frame_path


def update_manifest(manifest: ObservationManifest) -> None:
    """Re-save manifest (e.g. after updating frame metadata like OCR texts)."""
    _save_manifest(manifest)


def mark_stopped(manifest: ObservationManifest) -> None:
    """Mark the session as stopped and write final manifest."""
    manifest.stopped_at = datetime.now(tz=timezone.utc).isoformat()
    _save_manifest(manifest)
    logger.info("Observation session stopped: %s/%s (%d frames, %d significant)",
                manifest.game, manifest.session_id,
                manifest.frame_count, manifest.significant_count)


def mark_completed(manifest: ObservationManifest) -> None:
    """Write a .completed sentinel file so the cleanup cycle knows this session is done."""
    sentinel = _session_dir(manifest.game, manifest.session_id) / ".completed"
    sentinel.touch()
    logger.debug("Session %s marked completed", manifest.session_id)


def mark_interrupted(manifest: ObservationManifest) -> None:
    """Write a .interrupted sentinel (ADB disconnect, error, etc.)."""
    sentinel = _session_dir(manifest.game, manifest.session_id) / ".interrupted"
    sentinel.touch()
    logger.info("Session %s marked interrupted", manifest.session_id)


def delete_session(manifest: ObservationManifest) -> None:
    """Delete the entire session directory (for /stop cancellation)."""
    import shutil
    session_dir = _session_dir(manifest.game, manifest.session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
        logger.info("Observation session deleted: %s/%s",
                    manifest.game, manifest.session_id)


# ── Internal helpers ─────────────────────────────────────────────────

def _generate_session_id() -> str:
    """Generate a unique session ID: obs_20260614_153045_a1b2c3."""
    import uuid
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"obs_{date_str}_{suffix}"


def _session_dir(game: str, session_id: str) -> Path:
    return _OBSERVATIONS_DIR / game / session_id


def _manifest_path(game: str, session_id: str) -> Path:
    return _session_dir(game, session_id) / "manifest.json"


def _ensure_frames_dir(manifest: ObservationManifest) -> Path:
    frames_dir = _session_dir(manifest.game, manifest.session_id) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir


def _save_manifest(manifest: ObservationManifest) -> None:
    """Write manifest.json (atomic via temp file)."""
    import os
    manifest_path = _manifest_path(manifest.game, manifest.session_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    data = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2)
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, manifest_path)


def list_sessions(game: str) -> list[str]:
    """List all session IDs for a game."""
    game_dir = _OBSERVATIONS_DIR / game
    if not game_dir.exists():
        return []
    return sorted(
        d.name for d in game_dir.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    )


def get_significant_frames(manifest: ObservationManifest) -> list[ObsFrame]:
    """Return only significant frames (screen changed enough)."""
    return [f for f in manifest.frames if f.is_significant]


def get_frame_path(manifest: ObservationManifest, frame: ObsFrame) -> Path:
    """Get the full filesystem path to a frame's image file."""
    return _session_dir(manifest.game, manifest.session_id) / "frames" / frame.filename


# ── Retention policy ─────────────────────────────────────────────────

def cleanup_old_sessions(game: str, keep_days: int = 30,
                         dry_run: bool = False) -> dict:
    """Remove observation sessions older than keep_days.

    Sessions that have a .completed sentinel file are eligible for cleanup.
    Sessions without .completed (recording was interrupted/cancelled) are
    also cleaned up if older than keep_days.

    Args:
        game: Game ID to clean up.
        keep_days: Keep sessions younger than this many days.
        dry_run: If True, only report what would be deleted.

    Returns:
        {"deleted": count, "freed_mb": approx_mb, "kept": count}
    """
    import shutil
    import time as _time

    game_dir = _OBSERVATIONS_DIR / game
    if not game_dir.exists():
        return {"deleted": 0, "freed_mb": 0, "kept": 0}

    cutoff = _time.time() - keep_days * 86400
    deleted = 0
    freed_bytes = 0
    kept = 0

    for session_dir in sorted(game_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        # Check age from manifest mtime or directory mtime
        try:
            mtime = manifest_path.stat().st_mtime
        except OSError:
            mtime = session_dir.stat().st_mtime

        if mtime < cutoff:
            # Count total size before deletion
            try:
                total_size = sum(
                    f.stat().st_size for f in session_dir.rglob("*")
                    if f.is_file()
                )
            except OSError:
                total_size = 0

            if not dry_run:
                try:
                    shutil.rmtree(session_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning("Failed to clean up session %s: %s",
                                 session_dir.name, e)
                    kept += 1
                    continue

            deleted += 1
            freed_bytes += total_size
        else:
            kept += 1

    freed_mb = round(freed_bytes / (1024 * 1024), 1)
    if deleted > 0:
        logger.info(
            "Observation cleanup (%s): %d sessions deleted, ~%s MB freed, %d kept%s",
            game, deleted, freed_mb, kept,
            " (dry run)" if dry_run else "",
        )

    return {"deleted": deleted, "freed_mb": freed_mb, "kept": kept}


def get_observation_disk_usage(game: str = "") -> dict:
    """Report disk usage for observation data.

    Args:
        game: Game ID to report on, or "" for all games.

    Returns:
        {"sessions": total_sessions, "size_mb": total_size_mb}
    """
    base = _OBSERVATIONS_DIR
    if game:
        base = base / game
    if not base.exists():
        return {"sessions": 0, "size_mb": 0.0}

    total_size = 0
    session_count = 0
    for session_dir in base.rglob("manifest.json"):
        try:
            session_count += 1
            parent = session_dir.parent
            total_size += sum(
                f.stat().st_size for f in parent.rglob("*") if f.is_file()
            )
        except OSError:
            pass

    return {
        "sessions": session_count,
        "size_mb": round(total_size / (1024 * 1024), 1),
    }
