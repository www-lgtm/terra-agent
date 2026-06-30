"""Base scheduling execution chain — session-based orchestration.

Links three stages:
  1. box-scan (skill)     → data/session/<id>/box.json
  2. base-schedule (intel) → data/session/<id>/schedule.json
  3. base-deploy (skill)  → reads schedule.json, executes in-game

Each stage's output is persisted to disk so the chain can be resumed
mid-flight, retried on failure, or re-run with different parameters.

The conductor is exposed as an IntelligenceTool so the agent loop
can discover pending stages and inject guidance into the conversation.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.intelligence.base import (
    IntelligenceContext,
    IntelligenceResult,
    IntelligenceTool,
)

logger = logging.getLogger(__name__)

SESSION_DIR = Path(__file__).parent.parent.parent.parent / "data" / "session"

# Chain stages — order matters
STAGES: list[dict[str, Any]] = [
    {
        "name": "box-scan",
        "label": "扫描干员Box",
        "type": "skill",
        "skill_name": "box-scan",
        "output_file": "box.json",
        "description": "从游戏内干员列表逐屏截图OCR，识别干员名和精英等级",
    },
    {
        "name": "base-schedule",
        "label": "计算最优排班",
        "type": "intelligence",
        "intel_tool": "BaseScheduler",
        "input_file": "box.json",
        "output_file": "schedule.json",
        "description": "基于干员Box和用户目标，枚举产品配置并计算帕累托最优排班",
    },
    {
        "name": "base-deploy",
        "label": "执行基建排班",
        "type": "skill",
        "skill_name": "base-deploy",
        "input_file": "schedule.json",
        "output_file": None,
        "description": "进入基建按排班表逐个设施入驻干员",
    },
]


# ── Session helpers ──────────────────────────────────────────────

def _session_path(session_id: str) -> Path:
    return SESSION_DIR / session_id


def _ensure_session_dir(session_id: str) -> Path:
    p = _session_path(session_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class SessionState:
    """Snapshot of a chain session's progress."""
    session_id: str
    current_stage: int = 0          # 0-based index into STAGES
    stages: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    goal_desc: str = ""
    layout: str = "243"
    errors: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.current_stage >= len(self.stages)

    @property
    def current_stage_info(self) -> dict | None:
        if 0 <= self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return None


def load_session(session_id: str) -> SessionState | None:
    """Load a session from disk. Returns None if it doesn't exist."""
    path = _session_path(session_id) / "state.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SessionState(**data)
    except Exception:
        return None


def save_session(state: SessionState) -> None:
    """Persist session state to disk."""
    d = _ensure_session_dir(state.session_id)
    path = d / "state.json"
    data = {
        "session_id": state.session_id,
        "current_stage": state.current_stage,
        "stages": state.stages,
        "created_at": state.created_at,
        "goal_desc": state.goal_desc,
        "layout": state.layout,
        "errors": state.errors[-20:],  # Cap error log
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def new_session_id() -> str:
    """Generate a new session ID based on current timestamp + random suffix."""
    import random
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=4))
    return f"base_{ts}_{suffix}"


def read_box_file(session_id: str) -> dict[str, int | dict] | None:
    """Read box.json from a session, returning {name: {elite,level,...}}.

    Accepts both old format ({name: elite_int}) and new format
    ({name: {elite, level, potential, rarity}}).
    """
    path = _session_path(session_id) / "box.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("operators", data)
    except Exception:
        return None


def write_box_file(session_id: str, operators: dict[str, int | dict]) -> None:
    """Write box.json to a session directory.

    Accepts both old format ({name: elite_int}) and new rich format
    ({name: {elite, level, potential, rarity}}).
    """
    d = _ensure_session_dir(session_id)
    payload = {
        "game": "arknights",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "total": len(operators),
        "operators": operators,
    }
    (d / "box.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_schedule_file(session_id: str) -> dict | None:
    """Read schedule.json from a session."""
    path = _session_path(session_id) / "schedule.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_schedule_file(session_id: str, schedule_data: dict) -> None:
    """Write schedule.json to a session directory."""
    d = _ensure_session_dir(session_id)
    (d / "schedule.json").write_text(
        json.dumps(schedule_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Conductor / Intelligence Tool ────────────────────────────────

class BaseChainConductor(IntelligenceTool):
    """Orchestration tool that guides the agent through the box→schedule→deploy chain.

    - Detects whether a session is in progress
    - Reads intermediate outputs (box.json, schedule.json)
    - Injects the next skill/action as [智能建议] for the agent loop
    """

    def can_handle(self, task: str) -> bool:
        keywords = [
            "排班链", "一站式", "一键排班", "全自动基建",
            "扫描box", "基建扫描", "box扫描", "box-scan",
            "base-deploy", "执行排班", "按排班表",
            "chain", "pipeline", "orchestrate",
        ]
        task_lower = task.lower()
        return any(kw in task_lower for kw in keywords)

    def analyze(self, ctx: IntelligenceContext, task: str) -> IntelligenceResult | None:
        # ── Check for existing session ──
        session_id = self._find_session(task, ctx)
        state = load_session(session_id) if session_id else None

        # ── New session requested ──
        if not state:
            return self._start_new_session(task, ctx)

        # ── Resume existing session ──
        return self._resume_session(state, task, ctx)

    # ── Internal ──────────────────────────────────────────────────

    def _find_session(self, task: str, ctx: IntelligenceContext) -> str | None:
        """Look for session ID in task text or context.

        Only auto-detects when the user explicitly asks to continue.
        Fresh chain starts always create a new session.
        """
        import re
        # Explicit session reference
        match = re.search(r'session[：:=]\s*(\S+)', task)
        if match:
            sid = match.group(1).strip().rstrip("/")
            if _session_path(sid).exists():
                return sid

        # Explicit "continue" keyword — find most recent INCOMPLETE session
        if re.search(r'继续|接上|恢复|resume|continue', task):
            if SESSION_DIR.exists():
                sessions = sorted(
                    SESSION_DIR.iterdir(),
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
                for s in sessions:
                    if s.is_dir() and (s / "state.json").exists():
                        state = load_session(s.name)
                        if state and not state.is_complete:
                            return s.name

        return None

    def _start_new_session(self, task: str, ctx: IntelligenceContext) -> IntelligenceResult:
        session_id = new_session_id()

        # Clean up all old sessions — each new chain replaces the last.
        try:
            for d in SESSION_DIR.iterdir():
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

        state = SessionState(
            session_id=session_id,
            current_stage=0,
            stages=[dict(s) for s in STAGES],
            created_at=datetime.now(timezone.utc).isoformat(),
            goal_desc=task,
        )

        # Parse goal/layout hints from task
        if "搓玉" in task:
            state.goal_desc = "搓玉为主"
            state.layout = "243"
        if "333" in task:
            state.layout = "333"
        elif "252" in task:
            state.layout = "252"

        save_session(state)

        stage = STAGES[0]
        recommendation = (
            f"## 🔗 基建排班执行链 — 已创建\n\n"
            f"- **Session**: `{session_id}`\n"
            f"- **目标**: {state.goal_desc}\n"
            f"- **布局**: {state.layout}\n\n"
            f"### 阶段 1/3: {stage['label']}\n\n"
            f"{stage['description']}\n\n"
            f"👉 执行 skill: `skill_run('{stage['skill_name']}')`\n\n"
            f"完成后说「继续排班链」进入下一阶段。\n"
            f"如果 box 扫描结果不准确，说「重试本阶段」重新来。"
        )

        return IntelligenceResult(
            recommendation=recommendation,
            confidence=0.9,
            source="hybrid",
        )

    def _resume_session(
        self, state: SessionState, task: str, ctx: IntelligenceContext,
    ) -> IntelligenceResult:
        stage = state.current_stage_info

        if state.is_complete:
            return IntelligenceResult(
                recommendation=(
                    f"## ✅ 基建排班链已全部完成\n\n"
                    f"- **Session**: `{state.session_id}`\n"
                    f"- **目标**: {state.goal_desc}\n"
                    f"- **布局**: {state.layout}\n\n"
                    f"如果需要对排班进行调整，可以说「重新计算排班」回到阶段 2。"
                ),
                confidence=0.95,
                source="hybrid",
            )

        # ── Stage 1: box-scan ──
        if state.current_stage == 0:
            # Check if box.json already exists
            box = read_box_file(state.session_id)
            if box and len(box) > 0:
                # Box already scanned — auto-advance
                state.current_stage = 1
                save_session(state)
                return self._resume_session(state, task, ctx)

            return IntelligenceResult(
                recommendation=(
                    f"## 🔗 排班链 — 阶段 {state.current_stage + 1}/3: {stage['label']}\n\n"
                    f"**Session**: `{state.session_id}`\n\n"
                    f"执行 `skill_run('box-scan')` 从游戏内扫描您的干员列表。\n"
                    f"扫描完成后会自动推进到排班计算阶段。"
                ),
                confidence=0.85,
                source="hybrid",
            )

        # ── Stage 2: base-schedule ──
        if state.current_stage == 1:
            box = read_box_file(state.session_id)
            schedule = read_schedule_file(state.session_id)

            if schedule:
                # Already computed — skip to deploy
                state.current_stage = 2
                save_session(state)
                return self._resume_session(state, task, ctx)

            if not box:
                # No box — go back to scan
                state.current_stage = 0
                save_session(state)
                return IntelligenceResult(
                    recommendation="未找到干员box数据，回到阶段1。说「扫描box」重新开始。",
                    confidence=0.5,
                    source="hybrid",
                )

            # Trigger the intelligence tool to compute schedule
            # Format box as inline text for display (handles both old int and new dict format)
            from src.intelligence.arknights.base_optimizer import _box_elite
            box_entries = [
                f"{name}(E{_box_elite(elite)})"
                for name, elite in box.items()
            ]
            box_text = "、".join(box_entries[:5])  # Just a sample
            if len(box_entries) > 5:
                box_text += f"... (共{len(box_entries)}名)"

            goal_text = state.goal_desc or task
            schedule_task = f"{goal_text}，box文件：data/session/{state.session_id}/box.json"

            # Write the box as a named dict so the scheduler can load it
            # (This re-uses the existing JSON file reading path)

            return IntelligenceResult(
                recommendation=(
                    f"## 🔗 排班链 — 阶段 {state.current_stage + 1}/3: {stage['label']}\n\n"
                    f"**Session**: `{state.session_id}`\n"
                    f"**干员数**: {len(box)} 名\n\n"
                    f"系统已自动从 `box.json` 读取干员数据。\n"
                    f"请确认目标：`{schedule_task}`\n\n"
                    f"调度器将计算帕累托最优排班并保存到 `schedule.json`。"
                ),
                confidence=0.85,
                source="hybrid",
            )

        # ── Stage 3: base-deploy ──
        if state.current_stage == 2:
            schedule = read_schedule_file(state.session_id)
            if not schedule:
                state.current_stage = 1
                save_session(state)
                return self._resume_session(state, task, ctx)

            return IntelligenceResult(
                recommendation=(
                    f"## 🔗 排班链 — 阶段 {state.current_stage + 1}/3: {stage['label']}\n\n"
                    f"**Session**: `{state.session_id}`\n\n"
                    f"排班表已生成（schedule.json）。\n"
                    f"执行 `skill_run('base-deploy')` 进入基建按排班表入驻干员。\n\n"
                    f"完成后说「完成」结束排班链。"
                ),
                confidence=0.90,
                source="hybrid",
            )

        return IntelligenceResult(
            recommendation=f"未知阶段: {state.current_stage}",
            confidence=0.0,
            source="hybrid",
        )

    # ── Programmatic API (callable from tools / tests) ────────────

    def advance_stage(self, session_id: str) -> SessionState | None:
        """Advance a session to the next stage. Returns updated state."""
        state = load_session(session_id)
        if not state:
            return None
        state.current_stage = min(state.current_stage + 1, len(STAGES))
        save_session(state)
        return state

    def mark_error(self, session_id: str, error: str) -> None:
        """Record an error in the session."""
        state = load_session(session_id)
        if state:
            state.errors.append(f"[{datetime.now(timezone.utc).isoformat()}] {error}")
            save_session(state)

    def get_box(self, session_id: str) -> dict[str, int] | None:
        return read_box_file(session_id)

    def get_schedule(self, session_id: str) -> dict | None:
        return read_schedule_file(session_id)
