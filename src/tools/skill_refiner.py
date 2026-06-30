"""Skill Refiner — auto-maintain stale skills (Phase 4).

When fast_chain reports "coordinates are likely stale" and the LLM manually
completes the task, this module compares the new success chain with the old
skill file and decides the right maintenance action.

Diff classifications:
  IDENTICAL    — nothing changed, likely false-positive (loading delay etc.)
  COORDS_ONLY  — steps identical, only coordinates drifted → patch coords
  STRUCTURAL   — steps inserted/removed but core flow preserved → replace body
  REWRITE      — flow completely changed → old deprecated, new file created

Lifecycle:
  1. Mark:    fast_chain failure → mark_skill_potentially_stale(name)
  2. Compare: background_review extracts new chain → compare with old skill
  3. Classify:  _classify_diff() determines the type of change
  4. Act:       _handle_identical / _handle_coords / _handle_structural / _handle_rewrite
  5. Backup:    old version saved as skill_name.v{n}.md.bak before any mutation
"""

from __future__ import annotations

import json
import logging
import re
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _safe_read_json(path: Path) -> dict | None:
    """Read a JSON file with encoding fallback (UTF-8 → GBK).

    P1 fix: external editors on Chinese Windows may save files in GBK.
    Byte 0xbb is a valid GBK lead byte but never valid UTF-8.
    """
    if not path.exists():
        return None
    for enc in ("utf-8", "gbk"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except OSError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Try next encoding before giving up
            continue
    return None


# Coordinate drift threshold: fraction of screen dimension
COORD_STALE_FRACTION = 0.04  # 4% (~43px on 1080p) — was 0.10, too coarse

# Sequence overlap threshold for STRUCTURAL vs REWRITE
OVERLAP_THRESHOLD = 0.5  # ≥50% steps overlap → STRUCTURAL, else REWRITE

# False-positive: max consecutive IDENTICAL results before escalating
FALSE_POSITIVE_MAX_RETRIES = 5  # was 3 — more tolerant with finer drift detection

_STALE_MARKER_FILE = "stale_skills.json"


# ── Diff types ─────────────────────────────────────────────────────

@dataclass
class SkillDiff:
    """Result of comparing old and new skill chains."""
    type: str = "IDENTICAL"          # IDENTICAL | COORDS_ONLY | STRUCTURAL | REWRITE
    changed_steps: list[int] = field(default_factory=list)
    coord_deltas: dict[int, tuple[float, float]] = field(default_factory=dict)
    overlap_ratio: float = 1.0
    detail: str = ""


# ── Stale marker persistence ───────────────────────────────────────

def _marker_path() -> Path:
    from config.settings import config as _cfg
    return Path(_cfg.DATA_DIR) / "memory" / _STALE_MARKER_FILE


def mark_skill_potentially_stale(skill_name: str, game: str = "arknights",
                                 reason: str = "fast_chain_failed") -> None:
    """Record that a skill's fast_chain failed, making it a refinement candidate."""
    marker_path = _marker_path()
    marker_path.parent.mkdir(parents=True, exist_ok=True)

    markers: dict[str, list[dict]] = _safe_read_json(marker_path) or {}

    key = f"{game}/{skill_name}"
    if key not in markers:
        markers[key] = []

    markers[key].append({
        "timestamp": _time.time(),
        "reason": reason,
    })
    markers[key] = markers[key][-10:]  # Keep last 10

    marker_path.write_text(json.dumps(markers, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Skill '%s' marked as potentially stale (count=%d)", skill_name, len(markers[key]))


def _get_stale_count(skill_name: str, game: str = "arknights") -> int:
    marker_path = _marker_path()
    if not marker_path.exists():
        return 0
    markers = _safe_read_json(marker_path)
    if markers is None:
        return 0
    key = f"{game}/{skill_name}"
    return len(markers.get(key, []))


def _clear_stale_marker(skill_name: str, game: str = "arknights") -> None:
    marker_path = _marker_path()
    if not marker_path.exists():
        return
    markers = _safe_read_json(marker_path)
    if markers is None:
        return
    key = f"{game}/{skill_name}"
    if key in markers:
        del markers[key]
        marker_path.write_text(json.dumps(markers, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Cleared stale marker for skill '%s'", skill_name)


# ── Diff classification ────────────────────────────────────────────

def _classify_diff(
    old_steps: list[dict],
    new_steps: list[dict],
    screen_w: int,
    screen_h: int,
) -> SkillDiff:
    """Compare old and new skill chains, classify the type of change."""
    # Build tool+arg signatures for structural comparison.
    # OCR text normalization: strip whitespace so "开始 行动" matches "开始行动".
    def _sig(steps: list[dict]) -> list[str]:
        out: list[str] = []
        for s in steps:
            tool = s.get("tool", "")
            args = s.get("args", [])
            # Normalize each arg: strip internal whitespace for OCR tolerance
            normalized = [re.sub(r'\s+', '', str(a)) for a in args]
            out.append(f"{tool}({','.join(normalized)})")
        return out

    old_sigs = _sig(old_steps)
    new_sigs = _sig(new_steps)

    # 1. Check: identical tool sequence?
    if old_sigs == new_sigs:
        # Same sequence — check coordinates
        changed: list[int] = []
        deltas: dict[int, tuple[float, float]] = {}
        for i, (old, new) in enumerate(zip(old_steps, new_steps)):
            oc = old.get("coords")
            nc = new.get("coords")
            if oc and nc and len(oc) == 2 and len(nc) == 2:
                dx = abs(oc[0] - nc[0]) / screen_w
                dy = abs(oc[1] - nc[1]) / screen_h
                if dx > COORD_STALE_FRACTION or dy > COORD_STALE_FRACTION:
                    changed.append(i + 1)
                    deltas[i + 1] = (round(dx, 3), round(dy, 3))
        if changed:
            return SkillDiff(
                type="COORDS_ONLY",
                changed_steps=changed,
                coord_deltas=deltas,
                detail=f"{len(changed)}/{len(old_steps)} steps have coordinate drift",
            )
        return SkillDiff(type="IDENTICAL", detail="all steps identical")

    # 2. Calculate sequence overlap
    old_set = set(old_sigs)
    new_set = set(new_sigs)
    if old_set and new_set:
        intersection = len(old_set & new_set)
        union = len(old_set | new_set)
        overlap = intersection / max(union, 1)
    else:
        overlap = 0.0

    # 3. Classify
    if overlap >= OVERLAP_THRESHOLD:
        # Same core flow but structure changed
        added = sum(1 for s in new_sigs if s not in old_set)
        removed = sum(1 for s in old_sigs if s not in new_set)
        return SkillDiff(
            type="STRUCTURAL",
            overlap_ratio=overlap,
            detail=f"overlap={overlap:.0%}, +{added}/-{removed} steps",
        )

    return SkillDiff(
        type="REWRITE",
        overlap_ratio=overlap,
        detail=f"overlap={overlap:.0%}, flow completely changed",
    )


# ── Extraction helpers ─────────────────────────────────────────────

def _normalize_chain_steps(chain: list[dict]) -> list[dict]:
    """Normalize steps from background_review's _build_skill_step format to
    parse_skill_steps format.  _build_skill_step uses 'target' for adb_tap
    and 'args' for adb_tap_position; parse_skill_steps uses 'args' for both.
    This function converts 'target' → 'args' so both sources produce
    comparable step dicts."""
    normalized: list[dict] = []
    for s in chain:
        step = dict(s)
        if "target" in step and "args" not in step:
            # _build_skill_step format: adb_tap has target
            step["args"] = [step.pop("target")]
        elif "target" in step and step.get("tool") in ("adb_swipe", "adb_scroll"):
            # adb_swipe/adb_scroll may have args from parse but target from builder
            if not step.get("args"):
                step["args"] = [step.pop("target")]
        normalized.append(step)
    return normalized


def _extract_pitfalls_section(body: str) -> str:
    """Extract the Pitfalls section from a skill body, preserving formatting."""
    m = re.search(r'(##\s*(?:Pitfalls|注意事项).*)', body, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


# ── Action handlers per diff type ──────────────────────────────────

def _handle_coords_only(
    skill_name: str, game: str, old_body: str,
    old_steps: list[dict], new_chain: list[dict],
    skill: dict[str, Any],
) -> dict | None:
    """Patch coordinates in place, keeping step structure and text intact."""
    version = skill.get("version", 0) + 1
    new_body = _update_skill_body_coords(old_body, old_steps, new_chain)
    _backup_and_save(skill_name, game, new_body, skill, version)
    _clear_stale_marker(skill_name, game)

    logger.info("Skill '%s': coords updated (v%d)", skill_name, version)
    # ── Cross-validate with observation data (background, but collect result) ──
    _xv_result: list[dict | None] = [None]

    def _xv_and_store() -> None:
        try:
            report = cross_validate_with_observations(skill_name, game)
            _xv_result[0] = report
            if report and report.get("drifting_steps"):
                logger.info(
                    "Cross-validation: skill '%s' has %d/%d steps with >5%% drift "
                    "vs observation data — consider manual review. Steps: %s",
                    skill_name,
                    len(report["drifting_steps"]),
                    report.get("total_steps", 0),
                    [d["step"] for d in report["drifting_steps"]],
                )
            elif report is None:
                logger.debug(
                    "Cross-validation: no observation data available for '%s'",
                    skill_name,
                )
        except Exception as e:
            logger.debug("Cross-validation thread error: %s", e)

    import threading as _thr
    _thr.Thread(target=_xv_and_store, daemon=True).start()
    return {
        "action": "coords_updated",
        "skill_name": skill_name,
        "version": version,
    }


def _handle_structural(
    skill_name: str, game: str, old_body: str,
    new_chain: list[dict], skill: dict[str, Any],
) -> dict | None:
    """Replace entire body, preserving the original Pitfalls section."""
    pitfalls = _extract_pitfalls_section(old_body)
    version = skill.get("version", 0) + 1

    new_body = _steps_chain_to_markdown(new_chain)
    if pitfalls:
        new_body += "\n\n" + pitfalls

    _backup_and_save(skill_name, game, new_body, skill, version)
    _clear_stale_marker(skill_name, game)

    logger.info("Skill '%s': body replaced (v%d, pitfalls preserved)", skill_name, version)
    return {"action": "body_replaced", "skill_name": skill_name, "version": version}


def _handle_rewrite(
    skill_name: str, game: str, old_body: str,
    new_chain: list[dict], skill: dict[str, Any],
    task_description: str,
) -> dict | None:
    """Deprecate old skill, create a new one with the new chain."""
    skill_mgr = _get_skill_manager(game)

    # 1. Mark old skill deprecated
    old_content = skill.get("raw", "")
    # Add deprecated to frontmatter if not already there
    if "deprecated:" not in old_content:
        old_content = old_content.replace(
            "---\n", "---\ndeprecated: true\n", 1,
        )
    skill_mgr.save(skill_name, old_content)
    logger.info("Skill '%s' marked deprecated", skill_name)

    # 2. Create new skill from the successful chain
    # Derive a new name
    new_name = f"{skill_name}-v2"
    # Check if it already exists, bump
    base = new_name
    idx = 2
    while skill_mgr.load(new_name):
        new_name = f"{base}-{idx}"
        idx += 1

    # Build new skill from chain
    pitfalls = _extract_pitfalls_section(old_body)
    body = _steps_chain_to_markdown(new_chain)
    if pitfalls:
        body += "\n\n" + pitfalls

    desc = skill.get("description", task_description)[:80]
    frontmatter = (
        "---\n"
        f"name: {new_name}\n"
        f'description: "{desc}"\n'
        f"tags: [{', '.join(skill.get('tags', []))}]\n"
        f"game: {game}\n"
        f"type: script\n"
        f"verified: true\n"
        f"version: 1\n"
        f"replaces: {skill_name}\n"
        f"coords_verified_at: {datetime.now(tz=timezone.utc).isoformat()}\n"
        "---"
    )
    content = f"{frontmatter}\n\n{body}\n"
    skill_mgr.save(new_name, content)

    _clear_stale_marker(skill_name, game)

    logger.info("Skill '%s': rewrite → deprecated, created '%s'", skill_name, new_name)
    return {
        "action": "rewritten",
        "deprecated": skill_name,
        "new_skill": new_name,
        "version": 1,
    }


def _handle_identical(
    skill_name: str, game: str,
) -> dict | None:
    """Likely false-positive — count retries, only clear if below threshold."""
    stale_count = _get_stale_count(skill_name, game)
    if stale_count >= FALSE_POSITIVE_MAX_RETRIES:
        logger.warning(
            "Skill '%s': %d IDENTICAL failures — possible timing issue, "
            "not clearing marker. Consider adding wait step.",
            skill_name, stale_count,
        )
        return {
            "action": "false_positive_escalated",
            "skill_name": skill_name,
            "retries": stale_count,
            "suggestion": "Consider adding a wait step or increasing poll timeout",
        }
    # Below threshold: clear the marker, it was a one-off
    _clear_stale_marker(skill_name, game)
    logger.info("Skill '%s': IDENTICAL (retry %d/%d) — cleared",
               stale_count, FALSE_POSITIVE_MAX_RETRIES)
    return {"action": "cleared_false_positive", "skill_name": skill_name}


# ── Body manipulation helpers ──────────────────────────────────────

def _update_skill_body_coords(
    old_body: str,
    old_steps: list[dict],
    new_chain: list[dict],
) -> str:
    """Replace stale coordinates in skill body with new ones. Line-by-line patch."""
    lines = old_body.split("\n")
    result_lines: list[str] = []

    for line in lines:
        updated = line
        m = re.match(r"^(\d+)\.\s+(\w+)\(.*\)\s*#\s*\[(\d+)\s*,\s*(\d+)\]", line)
        if m:
            step_num = int(m.group(1))
            if step_num - 1 < len(new_chain):
                nc = new_chain[step_num - 1].get("coords")
                if nc and len(nc) == 2:
                    old_coord = f"# [{m.group(3)}, {m.group(4)}]"
                    new_coord = f"# [{int(nc[0])}, {int(nc[1])}]"
                    updated = line.replace(old_coord, new_coord)
        result_lines.append(updated)

    return "\n".join(result_lines)


def _steps_chain_to_markdown(chain: list[dict]) -> str:
    """Convert a parsed steps chain back to markdown body format."""
    lines = ["## Steps", ""]
    for i, step in enumerate(chain):
        tool = step.get("tool", "")
        args = step.get("args", [])
        coords = step.get("coords")
        args_str = ", ".join(f"'{a}'" if not re.match(r'^[\d.]+$', str(a)) else str(a)
                            for a in args)
        line = f"{i + 1}. {tool}({args_str})"
        if coords and len(coords) == 2:
            line += f"  # [{int(coords[0])}, {int(coords[1])}]"
        lines.append(line)
    return "\n".join(lines)


def _backup_and_save(
    skill_name: str, game: str, new_body: str,
    skill: dict[str, Any], version: int,
) -> None:
    """Save new content and backup old to .v{n}.md.bak."""
    skill_mgr = _get_skill_manager(game)

    # Read old content BEFORE save overwrites
    old_path = skill_mgr.base_dir / f"{skill_name}.md"
    old_content = ""
    if old_path.exists():
        try:
            old_content = old_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # P1: fall back to GBK (Chinese Windows default)
            try:
                old_content = old_path.read_text(encoding="gbk")
            except Exception:
                old_content = ""

    desc = skill.get("description", "").replace('"', '\\"')
    tags = skill.get("tags", [])
    skill_type = skill.get("type", "script" if skill.get("verified") else "guide")

    frontmatter = (
        "---\n"
        f"name: {skill_name}\n"
        f'description: "{desc}"\n'
        f"tags: [{', '.join(tags)}]\n"
        f"game: {game}\n"
        f"type: {skill_type}\n"
        f"verified: true\n"
        f"version: {version}\n"
        f"coords_verified_at: {datetime.now(tz=timezone.utc).isoformat()}\n"
        "---"
    )
    content = f"{frontmatter}\n\n{new_body}\n"
    skill_mgr.save(skill_name, content)

    # Write backup
    if old_content and old_content != content:
        old_path.with_suffix(f".v{version - 1}.md.bak").write_text(
            old_content, encoding="utf-8",
        )
        logger.info("Skill '%s' v%d backup saved", skill_name, version - 1)


def cross_validate_with_observations(
    skill_name: str, game: str = "arknights",
    screen_w: int = 1600, screen_h: int = 900,
) -> dict | None:
    """Compare skill coordinates with user observation click data.

    When a skill's fast_chain repeatedly fails, loads the most recent
    observation session for the same game/task and compares click
    coordinates frame by frame.  Returns a report of which steps have
    drifted and by how much.

    Returns None if no observation data is available.
    """
    try:
        from src.agent.observation_store import list_sessions, load_manifest
        sessions = list_sessions(game)
        if not sessions:
            return None

        # Load the most recent completed observation
        manifest = None
        for sid in reversed(sessions[-5:]):  # Last 5 sessions
            m = load_manifest(game, sid)
            if m and m.stopped_at:
                manifest = m
                break
        if manifest is None:
            return None

        # Get all significant frames with clicks
        sig_frames = [f for f in manifest.frames if f.is_significant and f.clicks_before]
        if not sig_frames:
            return None

        # Load skill steps
        from src.tools.fast_chain import parse_skill_steps
        from src.skills.manager import get_skill_manager
        import re as _re
        skill = get_skill_manager(game).load(skill_name)
        if not skill:
            return None
        steps = parse_skill_steps(skill.get("body", ""))
        body = skill.get("body", "")

        # ── Path A: parsed steps exist, compare vs observation ──
        if steps and sig_frames:
            drifts: list[dict] = []
            total_steps = len(steps)
            total_frames = len(sig_frames)

            for i, step in enumerate(steps):
                skill_coords = step.get("coords")
                if not skill_coords or len(skill_coords) != 2:
                    continue

                frame_idx = min(int(i / max(total_steps, 1) * total_frames), total_frames - 1)
                frame = sig_frames[frame_idx]

                if not frame.clicks_before:
                    continue
                avg_x = sum(c.device_x for c in frame.clicks_before) / len(frame.clicks_before)
                avg_y = sum(c.device_y for c in frame.clicks_before) / len(frame.clicks_before)

                dx_pct = abs(skill_coords[0] - avg_x) / screen_w
                dy_pct = abs(skill_coords[1] - avg_y) / screen_h

                if dx_pct > 0.05 or dy_pct > 0.05:
                    drifts.append({
                        "step": i + 1,
                        "skill_coords": list(skill_coords),
                        "observed_coords": [round(avg_x, 1), round(avg_y, 1)],
                        "drift_pct": [round(dx_pct, 3), round(dy_pct, 3)],
                    })

            if drifts:
                logger.info(
                    "Cross-validation: skill '%s' has %d/%d steps with >5%% drift "
                    "(observation: %s, %d frames)",
                    skill_name, len(drifts), len(steps),
                    manifest.session_id, total_frames,
                )
                return {
                    "skill_name": skill_name,
                    "observation_session": manifest.session_id,
                    "task_name": manifest.task_name,
                    "drifting_steps": drifts,
                    "total_steps": len(steps),
                }

        # ── Path B: no parseable steps (guide skill) — generate from observation ──
        if not steps and sig_frames:
            # Extract target keywords from numbered text steps
            # Format: "N. 描述文字" or "N. 「关键词」..." or "-「关键词」..."
            text_steps: list[tuple[int, str, str]] = []  # (idx, raw_line, target_kw)
            for i, line in enumerate(body.split("\n")):
                m = _re.match(
                    r'^\s*(?:\d+\.|[-•])\s*(?:.*?[「『"]([^」』"]+)[」』"])',
                    line,
                )
                if not m:
                    m = _re.match(
                        r'^\s*(?:\d+\.|[-•])\s*(?:点击|进入|返回)?\s*(.+?)(?:\s*→|\s*$|。)',
                        line,
                    )
                if m:
                    text_steps.append((i, line.strip(), m.group(1).strip()))

            if not text_steps:
                return None

            generated: list[dict] = []
            for step_idx, raw_line, target_kw in text_steps:
                # Find observation frame whose OCR best matches this step's target
                best_frame = None
                best_overlap = 0
                kw_chars = set(target_kw)
                for f in sig_frames:
                    frame_ocr = " ".join(f.ocr_texts or [])
                    frame_chars = set(frame_ocr)
                    if not kw_chars:
                        continue
                    overlap = len(kw_chars & frame_chars) / len(kw_chars)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_frame = f

                if best_frame and best_frame.clicks_before and best_overlap >= 0.3:
                    avg_x = sum(c.device_x for c in best_frame.clicks_before) / len(best_frame.clicks_before)
                    avg_y = sum(c.device_y for c in best_frame.clicks_before) / len(best_frame.clicks_before)
                    generated.append({
                        "step": step_idx + 1,  # 1-based for consistency
                        "raw_line": raw_line,
                        "target_kw": target_kw,
                        "observed_coords": [round(avg_x, 1), round(avg_y, 1)],
                        "ocr_match": best_overlap,
                    })

            if generated:
                logger.info(
                    "Cross-validation: skill '%s' generated %d steps from observation "
                    "(%d text steps → %d matched frames, session=%s)",
                    skill_name, len(generated), len(text_steps),
                    len(sig_frames), manifest.session_id,
                )
                return {
                    "skill_name": skill_name,
                    "observation_session": manifest.session_id,
                    "task_name": manifest.task_name,
                    "generated_steps": generated,
                    "total_text_steps": len(text_steps),
                }

        return None
    except Exception as e:
        logger.debug("Observation cross-validation skipped: %s", e)
        return None


def _build_skill_from_observation(
    skill_name: str, game: str, body: str,
    generated: list[dict], skill: dict, version: int,
    screen_w: int, screen_h: int,
) -> dict | None:
    """Build a verified skill body from observation click data.

    When a guide skill has text steps but no parseable ADB commands,
    observation data provides the missing coordinates.  Rebuilds the
    Steps section with adb_tap_position commands.

    Preserves frontmatter and Pitfalls section; replaces Steps entirely.
    """
    import re as _re

    # Build new step lines with coordinates
    new_steps: list[str] = []
    step_count = 0
    for g in generated:
        step_count += 1
        x, y = g["observed_coords"]
        pct_x = round(x / screen_w, 4)
        pct_y = round(y / screen_h, 4)
        new_steps.append(
            f'{step_count}. adb_tap_position({pct_x}, {pct_y}) '
            f'# [{int(x)}, {int(y)}]  # {g.get("target_kw", "")}'
        )

    if not new_steps:
        return None

    # Rebuild body: keep frontmatter, replace Steps section, keep Pitfalls
    frontmatter_end = body.find("\n---\n")
    if frontmatter_end >= 0:
        frontmatter_end = body.find("\n", frontmatter_end + 4)  # skip ---\n
        frontmatter = body[:frontmatter_end].rstrip()
    else:
        frontmatter = ""

    pitfalls_start = body.rfind("\n## Pitfalls")
    if pitfalls_start < 0:
        pitfalls_start = body.rfind("\n# Pitfalls")
    pitfalls = body[pitfalls_start:].strip() if pitfalls_start >= 0 else ""

    new_body = frontmatter + "\n\n## Steps\n\n" + "\n".join(new_steps)
    if pitfalls:
        new_body += "\n\n" + pitfalls

    # Update verified flag in frontmatter
    new_body = _re.sub(r'\nverified:\s*false', '\nverified: true', new_body)
    new_body = _re.sub(r'\ntype:\s*guide', '\ntype: script', new_body)

    _backup_and_save(skill_name, game, new_body, skill, version)
    _clear_stale_marker(skill_name, game)

    logger.info(
        "Observation→Skill: '%s' built %d steps from observation (v%d)",
        skill_name, step_count, version,
    )
    return {
        "action": "observation_built",
        "skill_name": skill_name,
        "built_steps": step_count,
        "version": version,
    }


def auto_refine_from_observation(
    skill_name: str, game: str = "arknights",
    screen_w: int = 1080, screen_h: int = 900,
) -> dict | None:
    """Close the observation→skill loop: use user click data to patch skill coords.

    When a skill's fast_chain fails due to stale coordinates, this loads the
    most recent observation session for the same game, compares user click
    coordinates with the skill's step coordinates, and patches any step where
    the drift exceeds 5% of screen dimension.

    This is the missing link — previously cross_validate_with_observations()
    only reported drift but never fixed it. Now we can auto-heal.

    Returns a dict with action + patched_step_count, or None if no fix applied.
    """
    report = cross_validate_with_observations(
        skill_name, game, screen_w=screen_w, screen_h=screen_h,
    )
    if not report:
        return None

    skill_mgr = _get_skill_manager(game)
    skill = skill_mgr.load(skill_name)
    if not skill:
        return None
    body = skill.get("body", "")
    version = skill.get("version", 0) + 1

    # ── Path A: generated_steps — build skill body from observation ──
    generated = report.get("generated_steps")
    if generated:
        return _build_skill_from_observation(
            skill_name, game, body, generated, skill, version,
            screen_w, screen_h,
        )

    # ── Path B: drifting_steps — patch existing coordinates ──
    drifting = report.get("drifting_steps")
    if not drifting:
        return None
    logger.info(
        "Auto-refine: skill '%s' has %d drifting steps — patching from observation",
        skill_name, len(drifting),
    )

    from src.tools.fast_chain import parse_skill_steps
    steps = parse_skill_steps(body)
    old_body = body

    # Build a set of step indices to patch
    drift_map: dict[int, tuple[float, float]] = {}
    for d in drifting:
        step_idx = d["step"] - 1  # Convert 1-based to 0-based
        obs = d["observed_coords"]
        drift_map[step_idx] = (obs[0], obs[1])

    # Patch body: update coordinates in markdown lines
    lines = old_body.split("\n")
    step_count = 0
    patched = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_tap_pos = (
            stripped.startswith("adb_tap_position(") and "# [" in stripped
        )
        if is_tap_pos and step_count in drift_map:
            new_x, new_y = drift_map[step_count]
            new_pct_x = round(new_x / screen_w, 4)
            new_pct_y = round(new_y / screen_h, 4)
            # Replace pct args and coords comment
            import re as _re
            lines[i] = _re.sub(
                r'adb_tap_position\([^)]+\)\s*#\s*\[[^\]]+\]',
                f'adb_tap_position({new_pct_x}, {new_pct_y}) # [{int(new_x)}, {int(new_y)}]',
                stripped,
            )
            patched += 1
        if is_tap_pos or stripped.startswith("adb_tap("):
            step_count += 1

    if patched == 0:
        logger.debug("Auto-refine: no lines matched for patching")
        return None

    new_body = "\n".join(lines)
    _backup_and_save(skill_name, game, new_body, skill, version)
    _clear_stale_marker(skill_name, game)

    logger.info(
        "Auto-refine: skill '%s' patched %d/%d steps from observation (v%d)",
        skill_name, patched, len(drifting), version,
    )
    return {
        "action": "observation_patched",
        "skill_name": skill_name,
        "patched_steps": patched,
        "total_drifting": len(drifting),
        "version": version,
    }


def _get_skill_manager(game: str):
    from src.skills.manager import get_skill_manager
    return get_skill_manager(game)


# ── Main entry point ───────────────────────────────────────────────

def check_and_refine_skill(
    skill_name: str,
    new_chain: list[dict[str, Any]],
    game: str = "arknights",
    screen_w: int = 1080,
    screen_h: int = 1920,
    task_description: str = "",
) -> dict | None:
    """Compare new success chain with existing skill, decide and apply refinement.

    Called from background_review after Phase 1 skill extraction completes.

    Returns:
        Action result dict, or None if no refinement was needed.
    """
    from src.tools.fast_chain import parse_skill_steps

    skill_mgr = _get_skill_manager(game)
    skill = skill_mgr.load(skill_name)
    if not skill:
        logger.debug("Skill '%s' not found for refinement check", skill_name)
        _clear_stale_marker(skill_name, game)
        return None

    old_body = skill.get("body", "")
    old_steps = parse_skill_steps(old_body)
    # Normalize: _build_skill_step (background_review) uses 'target' for
    # adb_tap, parse_skill_steps uses 'args[0]'. Convert to a common format.
    new_chain_steps = _normalize_chain_steps([
        s for s in new_chain if s.get("tool") and (s.get("coords") or s.get("args") or s.get("target"))
    ])

    if not old_steps or not new_chain_steps:
        # P1: upgrade from debug→warning — silent skip masked refinement gaps.
        # Log both the step counts AND the raw inputs so we can debug WHY
        # parsing produced empty lists (format mismatch, missing coords, etc.).
        logger.warning(
            "Skill '%s': refinement skipped — empty steps after normalization "
            "(old=%d steps from %d chars, new=%d steps from %d chain items). "
            "This means either the old skill body has no parseable steps or "
            "the background review chain extraction failed.",
            skill_name,
            len(old_steps), len(old_body),
            len(new_chain_steps), len(new_chain),
        )
        return None

    # ── Classify the difference ──
    diff = _classify_diff(old_steps, new_chain_steps, screen_w, screen_h)
    logger.info("Skill '%s' diff: type=%s, %s", skill_name, diff.type, diff.detail)

    # ── Dispatch by type ──
    if diff.type == "IDENTICAL":
        return _handle_identical(skill_name, game)

    if diff.type == "COORDS_ONLY":
        return _handle_coords_only(skill_name, game, old_body, old_steps, new_chain_steps, skill)

    if diff.type == "STRUCTURAL":
        return _handle_structural(skill_name, game, old_body, new_chain_steps, skill)

    if diff.type == "REWRITE":
        return _handle_rewrite(skill_name, game, old_body, new_chain_steps, skill, task_description)

    return None
