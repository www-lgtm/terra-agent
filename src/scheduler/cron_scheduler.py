"""Time-based scheduler daemon for Terra Agent.

Polls the database for due tasks and executes them on available devices.
Multi-device safe: each device runs at most one task at a time via a semaphore.

Thread-safety architecture:
- ``_device_sem[serial]`` (BoundedSemaphore(1)) — at most 1 task per device
- ``_task_locks[task_id]`` (Lock) — prevents the same task from running twice
- ``_pending_queue`` — tasks that couldn't be dispatched yet, retried each poll
- ``_running_tasks`` — track running (thread, agent) for cancellation

External code (WeChat handler, CLI) that wants to run a task on a device
should coordinate via ``engine.is_device_busy()`` / ``engine.reserve_device()``
/ ``engine.release_device()`` so we don't run two agents on the same device.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from config.settings import config
from src.scheduler.schedule_db import schedule_db
from src.scheduler.time_parser import calculate_next_run

logger = logging.getLogger(__name__)


def _cleanup_old_conv_logs(data_dir: str, retention_days: int = 30) -> int:
    """Delete conversation JSON log files older than retention_days.

    Called periodically by the scheduler daemon to prevent unbounded
    disk growth from per-task conversation history files.
    """
    import os as _os
    logs_dir = Path(data_dir) / "logs"
    if not logs_dir.exists():
        return 0
    cutoff = time.time() - (retention_days * 86400)
    deleted = 0
    try:
        for entry in logs_dir.iterdir():
            if entry.is_file() and entry.name.startswith("conv_") and entry.suffix == ".json":
                try:
                    if entry.stat().st_mtime < cutoff:
                        entry.unlink()
                        deleted += 1
                except OSError:
                    pass
    except Exception as e:
        logger.warning("Conv log cleanup failed: %s", e)
    return deleted


class ScheduleEngine:
    """Background daemon that polls for and executes scheduled tasks.

    One instance manages one set of devices. Tasks are allocated to
    the first available device. If all devices are busy, the task
    remains in the pending queue and is retried on the next poll cycle.
    """

    def __init__(self, device_serials: list[str] | None = None,
                 poll_interval: float | None = None) -> None:
        self.device_serials: list[str] = device_serials or ["emulator-5554"]
        self.poll_interval: float = poll_interval or config.scheduler.poll_interval

        # Per-device semaphore: at most 1 task per device at a time
        self._device_sem: dict[str, threading.BoundedSemaphore] = {}
        for serial in self.device_serials:
            self._device_sem[serial] = threading.BoundedSemaphore(value=1)

        # Administratively paused devices (e.g. during emulator restart).
        # _find_free_device skips these even if the semaphore is available.
        self._paused_devices: set[str] = set()
        self._paused_lock = threading.Lock()

        # Per-task lock: prevents the same task from running concurrently
        self._task_locks: dict[int, threading.Lock] = {}
        self._task_locks_lock = threading.Lock()

        # Per-slot semaphores (Phase 2): at most 1 task per GameSlot
        self._slot_sem: dict[str, threading.BoundedSemaphore] = {}
        self._slot_to_device: dict[str, str] = {}  # slot_id → device_serial
        self._slot_lock = threading.Lock()

        # Pending queue: tasks that couldn't be dispatched yet
        self._pending_queue: list[dict[str, Any]] = []
        self._pending_lock = threading.Lock()

        # Running task tracking: task_id -> (thread, agent) for cancellation
        self._running_tasks: dict[int, tuple[threading.Thread, Any]] = {}
        self._running_lock = threading.Lock()

        # Housekeeping tracking
        self._last_history_cleanup = 0.0  # timestamp of last schedule_history cleanup
        self._last_db_optimize = 0.0  # timestamp of last DB optimize
        self._poll_counter: int = 0

        self._thread: threading.Thread | None = None
        self._running = False

    # ---- Public API ----

    def start(self) -> None:
        """Start the scheduler daemon in a background thread."""
        if self._running:
            logger.warning("ScheduleEngine already running")
            return

        # Cancel stale queued tasks from previous session — a bot restart
        # means in-memory state is gone, so any leftover queued rows are orphans.
        from src.scheduler.schedule_db import schedule_db as _sdb
        _sdb.cancel_all_queued()

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="schedule-daemon"
        )
        self._thread.start()
        logger.info("ScheduleEngine started (poll=%.1fs, devices=%s)",
                     self.poll_interval, self.device_serials)

    def add_device(self, serial: str) -> None:
        """Register a new device serial discovered after engine creation.

        Thread-safe.  Call this when a restart or dynamic launch produces
        a new ADB serial (e.g. MuMu 12 assigns a different port after restart).
        Idempotent — does nothing if the serial is already known.
        """
        if serial in self._device_sem:
            return
        self._device_sem[serial] = threading.BoundedSemaphore(value=1)
        if serial not in self.device_serials:
            self.device_serials.append(serial)
        logger.info("ScheduleEngine: device %s added (now %d devices)",
                     serial, len(self.device_serials))

    def remove_device(self, serial: str) -> None:
        """Remove a device serial from the scheduler."""
        self._device_sem.pop(serial, None)
        if serial in self.device_serials:
            self.device_serials.remove(serial)
        self._paused_devices.discard(serial)
        logger.info("ScheduleEngine: device %s removed (now %d devices)",
                     serial, len(self.device_serials))

    def stop(self) -> None:
        """Gracefully stop the scheduler. Waits for in-flight tasks."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30.0)
        logger.info("ScheduleEngine stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def trigger_now(self, task_id: int) -> bool:
        """Manually trigger a scheduled task immediately.

        Uses the same safe dispatch path as the poll loop — respects
        per-task lock and device semaphore. If all devices are busy the
        task is prepended to the pending queue.

        Returns True if dispatched immediately, False if queued.
        """
        task = schedule_db.get_by_id(task_id)
        if task is None:
            raise ValueError(f"Task #{task_id} not found")

        # Acquire task lock (same logic as _try_dispatch)
        with self._task_locks_lock:
            if task_id not in self._task_locks:
                self._task_locks[task_id] = threading.Lock()
            lock = self._task_locks[task_id]

        if not lock.acquire(blocking=False):
            logger.info("Task #%d already running — trigger_now skipped", task_id)
            return False

        # Try to find a free device
        serial = self._find_free_device()
        if serial is None:
            lock.release()
            # Queue at front for highest priority
            with self._pending_lock:
                self._pending_queue.insert(0, task)
            logger.info("All devices busy — task #%d queued", task_id)
            return False

        # Dispatch
        thread = threading.Thread(
            target=self._execute_and_release, args=(task, serial, lock),
            daemon=True, name=f"schedule-trigger-{task_id}",
        )
        thread.start()
        return True

    def cancel_task(self, task_id: int) -> bool:
        """Cancel a running or pending task. Returns True if cancelled."""
        # Check running tasks
        with self._running_lock:
            entry = self._running_tasks.pop(task_id, None)
        if entry is not None:
            _thread, agent = entry
            try:
                agent.state.running = False
                agent.state.inject_message("用户要求停止")
            except Exception:
                pass
            logger.info("Cancelled running task #%d", task_id)
            return True

        # Check pending queue
        with self._pending_lock:
            for i, t in enumerate(self._pending_queue):
                if t["id"] == task_id:
                    self._pending_queue.pop(i)
                    logger.info("Removed pending task #%d from queue", task_id)
                    return True
        return False

    def get_running_task_ids(self) -> list[int]:
        """Return IDs of currently running tasks."""
        with self._running_lock:
            return list(self._running_tasks.keys())

    # ---- Device coordination (for external callers like WeChat handler) ----

    def is_device_busy(self, serial: str) -> bool:
        """Check if a device is currently occupied (non-blocking).

        Returns False if the serial is unknown — an unknown device can't
        possibly have a cron task running on it, so it's free.
        """
        if serial not in self._device_sem:
            return False
        sem = self._device_sem[serial]
        # BoundedSemaphore: try acquire+release to check availability
        acquired = sem.acquire(blocking=False)
        if acquired:
            sem.release()
            return False
        return True

    def reserve_device(self, serial: str) -> bool:
        """Try to reserve a device for external use. Returns True on success.

        Auto-registers previously unknown serials (hot-plug support).
        The caller MUST call ``release_device(serial)`` when done.
        """
        if serial not in self._device_sem:
            self.add_device(serial)
        return self._device_sem[serial].acquire(blocking=False)

    def release_device(self, serial: str) -> None:
        """Release a device previously reserved via ``reserve_device()``."""
        if serial in self._device_sem:
            try:
                self._device_sem[serial].release()
            except ValueError:
                logger.warning("release_device: semaphore for %s was not acquired", serial)

    # ---- Slot-level coordination (Phase 2) ----
    # Slot semaphores are per-slot_id (not per-device-serial), allowing
    # multiple GameSlots on the same device to operate independently.
    # Slot_id → device_serial mapping is registered via configure_slots().

    def configure_slots(self, slots: list[Any]) -> None:
        """Register GameSlots for per-slot semaphore management.

        Called at startup when GameSlot config is available. Creates
        per-slot semaphores and records the slot_id → device_serial mapping.
        """
        with self._slot_lock:
            for s in slots:
                if s.slot_id not in self._slot_sem:
                    self._slot_sem[s.slot_id] = threading.BoundedSemaphore(value=1)
                    self._slot_to_device[s.slot_id] = s.device_serial
                    logger.debug("Registered slot semaphore: %s → %s",
                               s.slot_id, s.device_serial)

    def reserve_slot(self, slot_id: str) -> bool:
        """Reserve a GameSlot for exclusive use. Returns True on success.

        Creates a semaphore entry for unknown slots so reserve_slot() works
        even before configure_slots() is called (e.g. in tests or Phase 1).
        """
        with self._slot_lock:
            if slot_id not in self._slot_sem:
                self._slot_sem[slot_id] = threading.BoundedSemaphore(value=1)
            sem = self._slot_sem[slot_id]
        return sem.acquire(blocking=False)

    def release_slot(self, slot_id: str) -> None:
        """Release a previously reserved GameSlot."""
        with self._slot_lock:
            if slot_id in self._slot_sem:
                try:
                    self._slot_sem[slot_id].release()
                except ValueError:
                    logger.warning("release_slot: semaphore for %s was not acquired",
                                 slot_id)

    def is_slot_busy(self, slot_id: str) -> bool:
        """Check if a GameSlot is currently occupied (non-blocking)."""
        with self._slot_lock:
            if slot_id not in self._slot_sem:
                return False
            sem = self._slot_sem[slot_id]
        acquired = sem.acquire(blocking=False)
        if acquired:
            sem.release()
            return False
        return True

    # ---- Device pause/resume (for emulator restart coordination) ----

    def pause_device(self, serial: str) -> bool:
        """Mark a device as paused. No new tasks will be dispatched to it.

        If a task is currently running on this device, it is left alone —
        typically the running agent itself triggered the restart and is
        blocked inside restart_emulator() waiting for it to complete.
        Killing it would leave the device stranded post-restart.

        The pause flag survives semaphore release — even after the current
        task finishes and releases the semaphore, _find_free_device won't
        pick this device until ``resume_device()`` is called.

        Returns True if the device was found and paused.
        """
        if serial not in self._device_sem:
            return False

        # Persistently mark as paused (survives semaphore releases)
        with self._paused_lock:
            self._paused_devices.add(serial)

        # Try to grab the semaphore. If we succeed, no task is running —
        # we hold it until resume_device() releases it.
        # If we fail, a task is holding the semaphore (the restarting agent
        # itself, or a system-triggered restart while an agent is busy).
        # In either case, don't kill the agent — the running task will
        # release the semaphore when it finishes.
        acquired = self._device_sem[serial].acquire(blocking=False)
        if acquired:
            # No task running — we'll release in resume_device()
            logger.info("Device %s paused (semaphore held, paused_devices=%s)",
                         serial, sorted(self._paused_devices))
        else:
            # Task is running — it will release the semaphore itself
            logger.info("Device %s paused (task in progress, paused_devices=%s)",
                         serial, sorted(self._paused_devices))
        return True

    def resume_device(self, serial: str) -> bool:
        """Resume a previously paused device. New tasks may be dispatched again.

        Returns True if the device was found and resumed.
        """
        if serial not in self._device_sem:
            return False

        with self._paused_lock:
            self._paused_devices.discard(serial)

        # Release semaphore if we held it from pause_device.
        # ValueError = we didn't hold it (the running task held it).
        try:
            self._device_sem[serial].release()
        except ValueError:
            pass
        logger.info("Device %s resumed (paused_devices=%s)", serial, sorted(self._paused_devices))
        return True

    @property
    def paused_devices(self) -> set[str]:
        """Return a copy of the paused device set (for status display)."""
        with self._paused_lock:
            return set(self._paused_devices)

    # ---- Internal ----

    def _run_loop(self) -> None:
        """Main polling loop. Runs in a daemon thread."""
        # Bind the first device to this thread so any direct ADB calls work
        if self.device_serials:
            try:
                from src.device.adb import bind_device_to_thread
                bind_device_to_thread(self.device_serials[0])
            except Exception:
                pass

        while self._running:
            try:
                self._poll()
            except Exception:
                logger.exception("ScheduleEngine poll iteration failed — continuing")
            time.sleep(self.poll_interval)

    def _poll(self) -> None:
        """Check for due tasks and dispatch them to available devices."""
        now = time.time()

        # Collect newly due tasks (dedup by id)
        due = schedule_db.get_due_tasks(now)
        if due:
            logger.info("ScheduleEngine: %d task(s) due", len(due))
            with self._pending_lock:
                existing_ids = {t["id"] for t in self._pending_queue}

                # Stale-task guard:
                # - One-shot tasks overdue by >60s: auto-delete (past window, won't fire)
                # - Recurring tasks overdue by >10min: reschedule instead of executing
                _STALE_THRESHOLD = 600  # 10 minutes for recurring
                _ONE_SHOT_EXPIRY = 60   # 1 minute for one-shot
                for t in due:
                    tid = t["id"]
                    if tid in existing_ids:
                        continue
                    with self._running_lock:
                        if tid in self._running_tasks:
                            continue
                    next_run = t.get("next_run", 0)
                    if next_run > 0 and now - next_run > _STALE_THRESHOLD:
                        # Stale recurring task — recalculate and skip
                        try:
                            new_next = calculate_next_run(
                                t["schedule_type"], t["schedule_value"]
                            ).timestamp()
                            schedule_db.update_next_run(tid, new_next)
                            logger.warning(
                                "Skipping stale task #%d '%s' (overdue by %.0f min), "
                                "rescheduled to %s",
                                tid, t.get("name", ""), (now - next_run) / 60,
                                time.strftime("%H:%M", time.localtime(new_next)),
                            )
                        except Exception:
                            logger.error("Failed to reschedule stale task #%d", tid)
                        continue
                    # One-shot tasks that missed their window → auto-delete
                    if t.get("one_shot") and next_run > 0 and now - next_run > _ONE_SHOT_EXPIRY:
                        schedule_db.delete(tid)
                        logger.info(
                            "Auto-deleted expired one-shot task #%d '%s' "
                            "(was due at %s, now overdue by %.0f min)",
                            tid, t.get("name", ""),
                            time.strftime("%H:%M:%S", time.localtime(next_run)),
                            (now - next_run) / 60,
                        )
                        continue
                    self._pending_queue.append(t)

        # Clean up finished running tasks
        with self._running_lock:
            finished = [
                tid for tid, (_t, agent) in self._running_tasks.items()
                if not agent.state.running
            ]
            for tid in finished:
                del self._running_tasks[tid]

        # Periodically clean up stale task_locks (every ~2.5h at 30s interval)
        with self._task_locks_lock:
            if len(self._task_locks) > 100 or self._poll_counter % 300 == 0:
                # Collect all known task IDs from DB
                known_ids = {t["id"] for t in schedule_db.get_all()}
                stale = [tid for tid in self._task_locks if tid not in known_ids]
                for tid in stale:
                    del self._task_locks[tid]
                if stale:
                    logger.debug("Cleaned up %d stale task_locks entries", len(stale))

        # Periodically clean up old schedule_history (every hour)
        if now - self._last_history_cleanup > 3600:
            schedule_db.cleanup_old_history()
            deleted = _cleanup_old_conv_logs(str(config.DATA_DIR))
            if deleted:
                logger.info("Cleaned up %d old conversation log files", deleted)
            self._last_history_cleanup = now

        # Periodically optimize DB (every 6 hours)
        if now - self._last_db_optimize > 21600:
            schedule_db.vacuum()
            self._last_db_optimize = now

        # Try to dispatch pending tasks
        with self._pending_lock:
            remaining: list[dict[str, Any]] = []
            for task in self._pending_queue:
                task_id = task["id"]
                # Re-read from DB to check if still enabled/exists
                current = schedule_db.get_by_id(task_id)
                if current is None or not current["enabled"]:
                    continue  # Task was deleted or disabled — drop from queue

                # ── One-shot pending expiry ──
                # A one-shot task that sits in pending_queue because the device
                # was busy may fire long after the user has already manually
                # completed the same task.  Auto-delete when it's been waiting
                # > 2 min past its scheduled time.
                _PENDING_EXPIRY = 120  # seconds
                _nxt = current.get("next_run", 0)
                if (current.get("one_shot")
                        and _nxt > 0
                        and now - _nxt > _PENDING_EXPIRY):
                    schedule_db.delete(task_id)
                    logger.info(
                        "Auto-deleted pending one-shot task #%d '%s' "
                        "(overdue by %.0f min — device was busy)",
                        task_id, current.get("name", ""),
                        (now - _nxt) / 60,
                    )
                    continue

                if self._try_dispatch(task):
                    pass  # Dispatched successfully
                else:
                    remaining.append(task)
            self._pending_queue = remaining

        # Increment poll counter for periodic housekeeping
        self._poll_counter += 1

    def _try_dispatch(self, task: dict[str, Any]) -> bool:
        """Try to dispatch a task to a free device. Returns True if dispatched."""
        task_id = task["id"]

        # Check if already running (per-task lock)
        with self._task_locks_lock:
            if task_id not in self._task_locks:
                self._task_locks[task_id] = threading.Lock()
            lock = self._task_locks[task_id]

        if not lock.acquire(blocking=False):
            return False  # Already running — skip this tick

        # Find a free device
        serial = self._find_free_device()
        if serial is None:
            lock.release()
            return False  # All devices busy — stay in queue

        # Launch on a new thread
        thread = threading.Thread(
            target=self._execute_and_release, args=(task, serial, lock),
            daemon=True, name=f"schedule-task-{task_id}",
        )
        thread.start()
        return True

    def _find_free_device(self) -> str | None:
        """Return the first device with an available semaphore slot,
        acquiring the semaphore on success. Caller is responsible for
        releasing it via ``_device_sem[serial].release()``.

        Skips devices that are administratively paused (e.g. emulator restart).
        """
        with self._paused_lock:
            paused = set(self._paused_devices)
        for serial in self.device_serials:
            if serial in paused:
                continue
            if self._device_sem[serial].acquire(blocking=False):
                return serial
        return None

    def _execute_and_release(self, task: dict[str, Any], serial: str,
                             lock: threading.Lock) -> None:
        """Execute a task on the given device, then release both the lock
        and the device semaphore."""
        try:
            self._execute_task(task, serial)
        finally:
            try:
                self._device_sem[serial].release()
            except ValueError:
                logger.warning("_execute_and_release: semaphore for %s was not acquired", serial)
            try:
                lock.release()
            except RuntimeError:
                logger.warning("_execute_and_release: lock for task #%d was not acquired",
                              task.get("id", "?"))

    def _execute_task(self, task: dict[str, Any], serial: str) -> None:
        """Execute a single scheduled task via TerraAgent on the given device."""
        task_id = task["id"]
        started_at = time.time()

        # Mark as launched
        schedule_db.record_execution(task_id, started_at)

        # Build the user message
        user_message = self._build_message(task)

        logger.info("Executing scheduled task #%d (%s) on %s",
                     task_id, task["name"], serial)

        # ── One-shot redundancy guard ──
        # A one-shot created in a previous session (>4h ago) whose game has
        # had a successful execution within the last 2h is almost certainly
        # redundant — the user already did the work manually via WeChat.
        # Only auto-delete when BOTH conditions hold, so freshly created
        # one-shots and one-shots on inactive games are never affected.
        if task.get("one_shot"):
            _created = task.get("created_at", 0)
            if _created > 0 and time.time() - _created > 14400:  # > 4 hours old
                try:
                    import sqlite3 as _sqlite3
                    from config.settings import config as _cfg
                    _db = _cfg.DATA_DIR / "memory" / "terra.db"
                    _conn = _sqlite3.connect(str(_db))
                    _row = _conn.execute(
                        "SELECT 1 FROM execution_history "
                        "WHERE game = ? AND success = 1 AND timestamp > ? "
                        "LIMIT 1",
                        (task.get("game", "arknights"), time.time() - 7200),
                    ).fetchone()
                    _conn.close()
                    if _row is not None:
                        logger.info(
                            "Skipping one-shot task #%d '%s' — redundant "
                            "(created %.0fh ago, recent success found on %s)",
                            task_id, task.get("name", ""),
                            (time.time() - _created) / 3600,
                            task.get("game", ""),
                        )
                        schedule_db.delete(task_id)
                        # Record as "completed" so history shows it was skipped
                        schedule_db.log_history(
                            task_id, started_at, time.time(),
                            success=True,
                            result_summary="Auto-skipped: redundant (recent manual execution found)",
                            device_serial=serial,
                            slot_id=task.get("slot_id", ""),
                        )
                        return
                except Exception:
                    pass  # Non-critical — if DB query fails, just execute normally

        try:
            from src.agent.loop import TerraAgent
            agent = TerraAgent(device_serial=serial, game=task.get("game", "arknights"))

            # Track for cancellation
            with self._running_lock:
                self._running_tasks[task_id] = (threading.current_thread(), agent)

            result = agent.run(user_message)
            finished_at = time.time()

            success = bool(result.get("success", False))
            summary = str(result.get("final_response", ""))[:500]
            error_msg = "" if success else str(result.get("error", ""))[:500]
            iterations = int(result.get("iterations", 0))

            try:
                schedule_db.log_history(
                    task_id, started_at, finished_at,
                    success=success,
                    result_summary=summary,
                    error_message=error_msg,
                    iterations=iterations,
                    device_serial=serial,
                    slot_id=task.get("slot_id", ""),
                )
            except Exception as e:
                logger.warning("Failed to log history for task #%d (may have been deleted): %s",
                               task_id, e)

            # Calculate next run
            is_one_shot = bool(task.get("one_shot", 0))
            max_runs = task.get("max_runs")
            current_count = (task.get("run_count") or 0) + 1
            delete_after = is_one_shot or (max_runs is not None and current_count >= max_runs)

            if delete_after:
                next_run_val = None
            else:
                try:
                    next_run_val = calculate_next_run(
                        task["schedule_type"], task["schedule_value"]
                    ).timestamp()
                except Exception:
                    logger.exception("Failed to calculate next_run for task #%d", task_id)
                    next_run_val = None

            try:
                schedule_db.record_completion(
                    task_id, success=success,
                    result_summary=summary,
                    error_message=error_msg,
                    next_run=next_run_val,
                    delete_one_shot=delete_after,
                )
            except Exception as e:
                logger.warning("Failed to record completion for task #%d (may have been deleted): %s",
                               task_id, e)

            logger.info("Scheduled task #%d completed: success=%s, next=%s",
                         task_id, success,
                         time.strftime("%Y-%m-%d %H:%M",
                                       time.localtime(next_run_val)) if next_run_val else "(deleted)")

        except Exception as e:
            finished_at = time.time()
            error_msg = str(e)[:500]
            logger.error("Scheduled task #%d crashed: %s", task_id, error_msg, exc_info=True)
            try:
                schedule_db.log_history(
                    task_id, started_at, finished_at,
                    success=False,
                    error_message=error_msg,
                    device_serial=serial,
                )
            except Exception:
                pass

            # Still calculate next run so a transient failure doesn't break the schedule
            is_one_shot = bool(task.get("one_shot", 0))
            if not is_one_shot:
                try:
                    next_run_val = calculate_next_run(
                        task["schedule_type"], task["schedule_value"]
                    ).timestamp()
                except Exception:
                    next_run_val = None
            else:
                next_run_val = None
            try:
                schedule_db.record_completion(
                    task_id, success=False,
                    error_message=error_msg,
                    next_run=next_run_val,
                    delete_one_shot=is_one_shot,
                )
            except Exception:
                pass

        finally:
            with self._running_lock:
                self._running_tasks.pop(task_id, None)

    @staticmethod
    def _build_message(task: dict[str, Any]) -> str:
        """Build the user message string passed to TerraAgent.run()."""
        try:
            payload = json.loads(task.get("task_payload", "{}"))
        except (json.JSONDecodeError, TypeError):
            payload = {}

        task_type = task.get("task_type", "custom")

        if task_type == "skill":
            skill_name = payload.get("skill_name", "")
            if skill_name:
                return f"执行技能: {skill_name}，完成后 task_complete()"

        # Default: use the custom prompt from the payload
        custom_prompt = payload.get("custom_prompt", "")
        if custom_prompt:
            return custom_prompt

        # Ultimate fallback
        return task.get("description", task.get("name", "执行定时任务"))


# Module-level singleton (lazy, not auto-started)
_engine: ScheduleEngine | None = None


def get_engine(device_serials: list[str] | None = None,
               poll_interval: float | None = None) -> ScheduleEngine:
    """Return the module-level ScheduleEngine singleton, creating it on first call.

    Note: On subsequent calls, ``device_serials`` and ``poll_interval`` are
    ignored (the original singleton is returned). To change devices, stop the
    engine first and call ``reset_engine()``, then call ``get_engine()`` again.
    """
    global _engine
    if _engine is None:
        _engine = ScheduleEngine(device_serials=device_serials, poll_interval=poll_interval)
    elif device_serials and device_serials != _engine.device_serials:
        logger.warning("get_engine(): engine already exists with devices=%s; "
                        "ignoring requested devices=%s. Use reset_engine() first.",
                        _engine.device_serials, device_serials)
    return _engine


def reset_engine() -> None:
    """Stop and reset the engine singleton. Next ``get_engine()`` creates a fresh one."""
    global _engine
    if _engine is not None:
        if _engine.is_running:
            _engine.stop()
        _engine = None
