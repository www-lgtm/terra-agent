"""Unit tests for the base chain conductor and session management."""

import json
import os
import pytest
from pathlib import Path
from src.intelligence.base import IntelligenceContext, IntelligenceResult
from src.intelligence.arknights.base_chain import (
    BaseChainConductor, SessionState, STAGES,
    new_session_id, save_session, load_session,
    write_box_file, read_box_file,
    write_schedule_file, read_schedule_file,
    SESSION_DIR,
)


class TestSessionIO:
    def test_new_session_id_unique(self):
        a = new_session_id()
        b = new_session_id()
        assert a != b

    def test_write_read_box(self):
        import shutil
        sid = "test_box_io"
        box = {"德克萨斯": 2, "清流": 1}
        write_box_file(sid, box)
        loaded = read_box_file(sid)
        assert loaded is not None
        assert loaded["德克萨斯"] == 2
        assert loaded["清流"] == 1
        shutil.rmtree(SESSION_DIR / sid, ignore_errors=True)

    def test_write_read_schedule(self):
        import shutil
        sid = "test_sched_io"
        data = {"best": {"orundum_eff": 0.95}}
        write_schedule_file(sid, data)
        loaded = read_schedule_file(sid)
        assert loaded is not None
        assert loaded["best"]["orundum_eff"] == 0.95
        shutil.rmtree(SESSION_DIR / sid, ignore_errors=True)

    def test_save_load_state(self):
        import shutil
        sid = "test_state_io"
        state = SessionState(
            session_id=sid, current_stage=1,
            stages=[dict(s) for s in STAGES],
            created_at="2026-01-01T00:00:00",
            goal_desc="搓玉",
        )
        save_session(state)
        loaded = load_session(sid)
        assert loaded is not None
        assert loaded.current_stage == 1
        assert loaded.goal_desc == "搓玉"
        shutil.rmtree(SESSION_DIR / sid, ignore_errors=True)

    def test_load_nonexistent(self):
        assert load_session("nonexistent_xyz") is None


class TestChainConductor:
    @pytest.fixture
    def conductor(self):
        return BaseChainConductor()

    @pytest.fixture
    def ctx(self):
        return IntelligenceContext(game="arknights", knowledge=None)

    def test_can_handle_chain_keywords(self, conductor):
        assert conductor.can_handle("全自动基建排班")
        assert conductor.can_handle("扫描box")
        assert conductor.can_handle("按排班表执行")
        assert conductor.can_handle("一键排班")

    def test_can_handle_rejects_unrelated(self, conductor):
        assert not conductor.can_handle("刷1-7")
        assert not conductor.can_handle("招募")

    def test_new_chain_creates_session(self, conductor, ctx):
        # Clean up any leftover session data from previous tests
        import shutil
        for d in SESSION_DIR.iterdir():
            if d.is_dir() and d.name.startswith("test_"):
                shutil.rmtree(d, ignore_errors=True)
            elif d.is_dir() and (d / "state.json").exists():
                state = load_session(d.name)
                if state and state.current_stage >= 2:  # Old completed sessions
                    shutil.rmtree(d, ignore_errors=True)

        result = conductor.analyze(ctx, "全自动基建排班 全新开始 不要继续session")
        assert result is not None
        assert result.confidence > 0.5
        # Should be stage 1 (box-scan) for a fresh start
        assert "阶段" in result.recommendation or "1" in result.recommendation

    def test_resume_chain_detects_existing(self, conductor, ctx):
        # Create a session with box data already done
        sid = "test_resume_chain"
        box = {"德克萨斯": 2, "能天使": 2}
        write_box_file(sid, box)
        state = SessionState(
            session_id=sid, current_stage=0,
            stages=[dict(s) for s in STAGES],
            created_at="2026-01-01T00:00:00",
            goal_desc="搓玉",
        )
        save_session(state)

        # Resume via session ID
        result = conductor.analyze(ctx, f"继续排班链 session:{sid}")
        loaded = load_session(sid)
        assert loaded is not None
        assert loaded.current_stage >= 1  # Advanced past box-scan since box exists

        # Cleanup
        import shutil
        shutil.rmtree(SESSION_DIR / sid, ignore_errors=True)

    def test_advance_stage(self, conductor):
        sid = "test_advance"
        state = SessionState(
            session_id=sid, current_stage=0,
            stages=[dict(s) for s in STAGES],
            created_at="2026-01-01T00:00:00",
        )
        save_session(state)

        updated = conductor.advance_stage(sid)
        assert updated is not None
        assert updated.current_stage == 1

        updated2 = conductor.advance_stage(sid)
        assert updated2.current_stage == 2

        updated3 = conductor.advance_stage(sid)
        assert updated3.current_stage == 3  # Complete

        # Cleanup
        (SESSION_DIR / sid / "state.json").unlink(missing_ok=True)

    def test_mark_error(self, conductor):
        sid = "test_error"
        state = SessionState(
            session_id=sid, current_stage=0,
            stages=[dict(s) for s in STAGES],
            created_at="2026-01-01T00:00:00",
        )
        save_session(state)

        conductor.mark_error(sid, "OCR failed on screen 3")
        loaded = load_session(sid)
        assert len(loaded.errors) == 1
        assert "OCR failed" in loaded.errors[0]

        (SESSION_DIR / sid / "state.json").unlink(missing_ok=True)
