"""Unit tests for src.scheduler.schedule_db."""

from __future__ import annotations

import time

import pytest

from src.scheduler.schedule_db import ScheduleDB, schedule_db


@pytest.fixture(autouse=True)
def _cleanup() -> None:
    """Clean up test tasks before each test to ensure isolation."""
    # We work on the shared DB; clean up any test-prefixed tasks
    for task in schedule_db.get_all():
        if task["name"].startswith("test-"):
            schedule_db.delete(task["id"])


class TestScheduleDBCreate:
    def test_create_minimal(self) -> None:
        task_id = schedule_db.create(
            name="test-create-minimal",
            task_payload={"custom_prompt": "do something"},
            schedule_type="interval",
            schedule_value="30m",
            one_shot=True,
        )
        assert task_id > 0

        task = schedule_db.get_by_id(task_id)
        assert task is not None
        assert task["name"] == "test-create-minimal"
        assert task["enabled"] == 1
        assert task["one_shot"] == 1
        assert task["run_count"] == 0

    def test_create_with_all_fields(self) -> None:
        task_id = schedule_db.create(
            name="test-create-full",
            task_payload={"custom_prompt": "清体力"},
            schedule_type="cron",
            schedule_value="0 9 * * *",
            description="早上9点清体力",
            game="arknights",
            task_type="custom",
            one_shot=False,
            max_runs=10,
        )
        task = schedule_db.get_by_id(task_id)
        assert task["schedule_type"] == "cron"
        assert task["schedule_value"] == "0 9 * * *"
        assert task["description"] == "早上9点清体力"
        assert task["max_runs"] == 10


class TestScheduleDBQuery:
    def test_get_all(self) -> None:
        # Create a test task
        tid = schedule_db.create(
            name="test-query-all",
            task_payload={"custom_prompt": "test"},
            schedule_type="interval",
            schedule_value="1h",
            one_shot=True,
        )
        tasks = schedule_db.get_all()
        assert len(tasks) >= 1
        ids = [t["id"] for t in tasks]
        assert tid in ids

    def test_get_all_enabled_only(self) -> None:
        tid = schedule_db.create(
            name="test-query-enabled",
            task_payload={"custom_prompt": "test"},
            schedule_type="interval",
            schedule_value="1h",
            one_shot=True,
        )
        # Initially enabled
        enabled = schedule_db.get_all_enabled()
        ids = [t["id"] for t in enabled]
        assert tid in ids

        # Disable it
        schedule_db.set_enabled(tid, False)
        enabled_after = schedule_db.get_all_enabled()
        ids_after = [t["id"] for t in enabled_after]
        assert tid not in ids_after

    def test_get_by_id_missing(self) -> None:
        assert schedule_db.get_by_id(999999) is None

    def test_get_due_tasks(self) -> None:
        # Create a task with next_run in the past
        past = time.time() - 60  # 1 minute ago
        tid = schedule_db.create(
            name="test-due-task",
            task_payload={"custom_prompt": "test"},
            schedule_type="interval",
            schedule_value="1h",
            one_shot=True,
            next_run=past,
        )
        due = schedule_db.get_due_tasks()
        ids = [t["id"] for t in due]
        assert tid in ids

        # Create a task with next_run far in the future
        future = time.time() + 86400  # 1 day from now
        schedule_db.create(
            name="test-future-task",
            task_payload={"custom_prompt": "test"},
            schedule_type="interval",
            schedule_value="1d",
            one_shot=True,
            next_run=future,
        )
        due_after = schedule_db.get_due_tasks()
        ids_after = [t["id"] for t in due_after]
        # future task should NOT be in due tasks
        assert tid in ids_after  # the past one is still there


class TestScheduleDBUpdate:
    def test_set_enabled(self) -> None:
        tid = schedule_db.create(
            name="test-toggle",
            task_payload={"custom_prompt": "test"},
            schedule_type="interval",
            schedule_value="1h",
            one_shot=True,
        )
        assert schedule_db.set_enabled(tid, False)
        task = schedule_db.get_by_id(tid)
        assert task["enabled"] == 0

        assert schedule_db.set_enabled(tid, True)
        task = schedule_db.get_by_id(tid)
        assert task["enabled"] == 1

    def test_set_enabled_missing(self) -> None:
        assert not schedule_db.set_enabled(999999, True)

    def test_record_execution(self) -> None:
        tid = schedule_db.create(
            name="test-execution",
            task_payload={"custom_prompt": "test"},
            schedule_type="interval",
            schedule_value="1h",
            one_shot=True,
        )
        started = time.time()
        schedule_db.record_execution(tid, started)
        task = schedule_db.get_by_id(tid)
        assert task["run_count"] == 1
        assert task["last_run"] is not None


class TestScheduleDBDelete:
    def test_delete(self) -> None:
        tid = schedule_db.create(
            name="test-delete",
            task_payload={"custom_prompt": "test"},
            schedule_type="interval",
            schedule_value="1h",
            one_shot=True,
        )
        assert schedule_db.delete(tid)
        assert schedule_db.get_by_id(tid) is None

    def test_delete_missing(self) -> None:
        assert not schedule_db.delete(999999)

    def test_record_completion_one_shot(self) -> None:
        tid = schedule_db.create(
            name="test-oneshot-complete",
            task_payload={"custom_prompt": "test"},
            schedule_type="interval",
            schedule_value="1h",
            one_shot=True,
        )
        schedule_db.record_completion(tid, success=True, delete_one_shot=True)
        assert schedule_db.get_by_id(tid) is None


class TestScheduleDBHistory:
    def test_log_and_get_history(self) -> None:
        tid = schedule_db.create(
            name="test-history",
            task_payload={"custom_prompt": "test"},
            schedule_type="interval",
            schedule_value="1h",
            one_shot=True,
        )
        started = time.time() - 10
        finished = time.time()

        schedule_db.log_history(
            tid, started, finished,
            success=True,
            result_summary="completed ok",
            iterations=5,
            device_serial="emulator-5554",
        )

        history = schedule_db.get_history(tid)
        assert len(history) >= 1
        entry = history[0]
        assert entry["success"] == 1
        assert entry["result_summary"] == "completed ok"
        assert entry["iterations"] == 5
        assert entry["device_serial"] == "emulator-5554"
        assert entry["duration_seconds"] > 0

    def test_history_limit(self) -> None:
        tid = schedule_db.create(
            name="test-history-limit",
            task_payload={"custom_prompt": "test"},
            schedule_type="interval",
            schedule_value="1h",
            one_shot=True,
        )
        for i in range(5):
            schedule_db.log_history(tid, time.time() - 1, time.time(), success=True)

        history = schedule_db.get_history(tid, limit=3)
        assert len(history) == 3
