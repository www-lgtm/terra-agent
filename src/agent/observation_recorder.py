"""Observation Recorder — dual-thread engine for watching user gameplay.

Records the user's manual gameplay by:
  1. Polling the ADB device for screenshots at regular intervals
  2. Running a Windows mouse hook to capture click coordinates
  3. Detecting significant screen changes via dHash
  4. Saving frames + click data to disk as an ObservationSession

The recorder is NON-INVASIVE — it only captures screenshots; it never taps,
swipes, or otherwise controls the device.  The user plays freely while the
system passively observes.

Usage:
    recorder = ObservationRecorder(
        device_serial="127.0.0.1:16384",
        game="reverse1999",
        task_name="1999日常",
    )
    manifest = recorder.start()
    # ... user plays game ...
    manifest_path = recorder.stop()
    # → Extraction LLM reads manifest_path and generates a guide skill
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from config.settings import config as app_config

logger = logging.getLogger(__name__)


class ObservationRecorder:
    """Passive screen + click recorder for game observation learning.

    Manages two threads:
      - screenshot_thread: polls ADB screenshots at configurable interval
      - mouse_thread: runs MouseClickMonitor (event-driven, not polling)

    Thread safety: clicks are drained under a lock managed by MouseClickMonitor.
    Manifest writes are serialized via the screenshot thread (only one writer).
    """

    def __init__(
        self,
        device_serial: str,
        game: str = "arknights",
        task_name: str = "",
        notify_fn: Any = None,
    ) -> None:
        self.device_serial = device_serial
        self.game = game
        self.task_name = task_name
        self._notify_fn = notify_fn  # Called on timeout/errors, signature: fn(msg: str)

        # Config
        self._poll_interval = getattr(
            getattr(app_config, 'observation', None), 'poll_interval_ms', 1500,
        ) / 1000.0  # Convert ms → seconds
        self._max_duration = getattr(
            getattr(app_config, 'observation', None), 'max_duration_s', 600,
        )
        self._dhash_threshold = getattr(
            getattr(app_config, 'observation', None), 'dhash_change_threshold', 8,
        )
        self._frame_max_w = getattr(
            getattr(app_config, 'observation', None), 'frame_max_width', 800,
        )
        self._frame_quality = getattr(
            getattr(app_config, 'observation', None), 'frame_jpeg_quality', 50,
        )

        # State
        self._manifest: Any = None  # ObservationManifest
        self._monitor: Any = None  # MouseClickMonitor
        self._screenshot_thread: threading.Thread | None = None
        self._running = threading.Event()
        self._last_dhash: int | None = None
        self._frame_index: int = 0
        self._t0: float = 0.0
        self._emu_window: dict | None = None  # Cached for periodic refresh
        self._dev_w: int = 1600
        self._dev_h: int = 900

    # ── Public API ───────────────────────────────────────────────

    def start(self) -> Any:
        """Start recording. Returns the ObservationManifest.

        Spawns screenshot + mouse threads.  Non-blocking.
        """
        from src.agent.observation_store import create_session

        # ── Set DPI awareness BEFORE any window rect queries ──
        # GetWindowRect returns virtualized coords on scaled displays when the
        # process is DPI-unaware.  MSLLHOOKSTRUCT.pt always reports physical
        # coords.  If we query the window rect before setting DPI awareness,
        # the two coordinate spaces diverge and ALL clicks are silently filtered.
        self._ensure_dpi_awareness()

        # Create session directory and manifest
        self._manifest = create_session(
            game=self.game,
            device_serial=self.device_serial,
            task_name=self.task_name,
        )

        # Get device screen size
        self._dev_w, self._dev_h = self._get_device_resolution()
        self._manifest.resolution = (self._dev_w, self._dev_h)

        # Find emulator window for mouse hook
        self._emu_window = self._find_emu_window()

        # Start mouse click monitor
        self._monitor, clicks_ok = self._start_mouse_monitor(
            self._emu_window, self._dev_w, self._dev_h,
        )
        if not clicks_ok:
            logger.warning("Mouse hook unavailable — recording screenshots only")

        # Start screenshot polling thread
        self._running.set()
        self._t0 = time.monotonic()
        self._screenshot_thread = threading.Thread(
            target=self._screenshot_loop,
            daemon=True,
            name="obs-screenshot",
        )
        self._screenshot_thread.start()

        logger.info(
            "Observation recording started: game=%s task=%s dev=%s (%dx%d) interval=%.1fs",
            self.game, self.task_name, self.device_serial,
            self._dev_w, self._dev_h, self._poll_interval,
        )
        return self._manifest

    def stop(self) -> str:
        """Stop recording gracefully. Returns manifest path string.

        Stops both threads, drains any remaining clicks into the last
        frame, writes final manifest.
        """
        self._running.clear()

        # Join screenshot thread
        if self._screenshot_thread and self._screenshot_thread.is_alive():
            self._screenshot_thread.join(timeout=5.0)

        # Drain any clicks that happened after the last frame.
        # Append them to the last frame if one exists.
        trailing_clicks = self._drain_remaining_clicks()
        captured_clicks = len(trailing_clicks)
        if trailing_clicks and self._manifest and self._manifest.frames:
            from src.agent.observation_store import ClickRecord
            last = self._manifest.frames[-1]
            last.clicks_before.extend(
                ClickRecord(
                    timestamp_s=c.timestamp,
                    desktop_x=c.desktop_x,
                    desktop_y=c.desktop_y,
                    device_x=c.device_x,
                    device_y=c.device_y,
                )
                for c in trailing_clicks
            )

        # Stop mouse monitor — drain one final time to catch clicks
        # that landed between our drain and the unhook.  monitor.stop()
        # internally unhooks and returns any clicks captured after the
        # final drain; we must append those too.
        if self._monitor:
            final_batch = self._monitor.drain_clicks()
            captured_clicks += len(final_batch)
            try:
                # stop() unhooks and returns clicks that arrived between
                # our drain_clicks() and the WM_QUIT being processed.
                stop_batch = self._monitor.stop()
            except Exception:
                stop_batch = []
            captured_clicks += len(stop_batch)
            self._monitor = None

            all_remaining = final_batch + stop_batch
            if all_remaining and self._manifest and self._manifest.frames:
                from src.agent.observation_store import ClickRecord
                last2 = self._manifest.frames[-1]
                last2.clicks_before.extend(
                    ClickRecord(
                        timestamp_s=c.timestamp, desktop_x=c.desktop_x,
                        desktop_y=c.desktop_y, device_x=c.device_x,
                        device_y=c.device_y,
                    )
                    for c in all_remaining
                )

        # Write final manifest
        if self._manifest:
            from src.agent.observation_store import mark_stopped
            mark_stopped(self._manifest)

            duration = time.monotonic() - self._t0
            logger.info(
                "Observation recording stopped: %d frames (%d significant), "
                "%d clicks, %.1fs",
                self._manifest.frame_count,
                self._manifest.significant_count,
                captured_clicks,
                duration,
            )

            from src.agent.observation_store import _manifest_path
            return str(_manifest_path(self.game, self._manifest.session_id))

        return ""

    def cancel(self) -> None:
        """Cancel recording and delete all saved data."""
        self._running.clear()

        if self._screenshot_thread and self._screenshot_thread.is_alive():
            self._screenshot_thread.join(timeout=3.0)

        if self._monitor:
            try:
                self._monitor.stop()
            except Exception:
                pass
            self._monitor = None

        if self._manifest:
            from src.agent.observation_store import delete_session
            delete_session(self._manifest)
            self._manifest = None

        logger.info("Observation recording cancelled")

    @property
    def manifest(self) -> Any | None:
        return self._manifest

    @property
    def elapsed_seconds(self) -> float:
        if self._t0 == 0:
            return 0.0
        return time.monotonic() - self._t0

    # ── Internal ─────────────────────────────────────────────────

    def _screenshot_loop(self) -> None:
        """Background thread: poll screenshots, detect changes, save frames."""
        try:
            from src.device.adb import bind_device_to_thread, get_adb
            bind_device_to_thread(self.device_serial)
            adb = get_adb()
        except Exception as e:
            logger.error("Cannot access ADB for observation: %s", e)
            self._mark_interrupted()
            return

        # Preload OCR engine (expensive first-run, do it once upfront)
        try:
            from src.vision.ocr import ocr_engine
            ocr_engine.preload()
        except Exception:
            pass

        from src.utils.dhash import compute_dhash, hamming_distance, dhash_to_hex

        _total_clicks_captured = 0  # Cumulative count across all drains

        while self._running.is_set():
            loop_start = time.monotonic()

            # Check max duration — warn but don't auto-stop.
            # Let the user decide when to /done; they have 30 minutes.
            elapsed_total = loop_start - self._t0 if self._t0 > 0 else 0.0
            if elapsed_total > self._max_duration:
                if self._notify_fn and self._frame_index % 30 == 0:
                    mins = int(self._max_duration / 60)
                    self._notify_fn(
                        f"⏰ 观察已超过 {mins} 分钟，请尽快完成操作后发送 /done。"
                    )

            # Warn user 2 minutes before the 30-minute soft limit
            if elapsed_total > (self._max_duration - 120) and self._frame_index % 30 == 0:
                if self._notify_fn:
                    self._notify_fn("⏰ 观察即将达到时间上限，请尽快完成操作后发送 /done。")

            # Refresh emu_window rect every ~30 frames (handle window moves/resizes)
            if self._frame_index > 0 and self._frame_index % 30 == 0 and self._monitor:
                self._refresh_emu_window()

            try:
                # Capture screenshot
                img = adb.get_screenshot_image()
                img_hash = compute_dhash(img)

                # Compute significance vs previous frame
                hamming = None
                is_sig = True  # First frame is always significant
                if self._last_dhash is not None:
                    hamming = hamming_distance(self._last_dhash, img_hash)
                    is_sig = hamming >= self._dhash_threshold
                self._last_dhash = img_hash

                # Drain clicks that happened since the last frame
                click_events = self._drain_remaining_clicks()
                _total_clicks_captured += len(click_events)

                # Compress and save frame
                img_bytes = self._compress_frame(img)

                now_since_start = loop_start - self._t0
                filename = f"{self._frame_index + 1:05d}.jpg"

                from src.agent.observation_store import (
                    ObsFrame, ClickRecord, save_frame,
                )

                # Convert ClickEvent → ClickRecord
                click_records = [
                    ClickRecord(
                        timestamp_s=c.timestamp,
                        desktop_x=c.desktop_x,
                        desktop_y=c.desktop_y,
                        device_x=c.device_x,
                        device_y=c.device_y,
                    )
                    for c in click_events
                ]

                frame = ObsFrame(
                    index=self._frame_index,
                    filename=filename,
                    timestamp_s=now_since_start,
                    dhash=dhash_to_hex(img_hash),
                    is_significant=is_sig,
                    hamming_from_prev=hamming,
                    clicks_before=click_records,
                )

                # Persist manifest every 10 frames, including frame 0
                # so the manifest is always on disk from the start.
                persist = (self._frame_index % 10 == 0)
                save_frame(self._manifest, frame, img_bytes, update_disk=persist)
                self._frame_index += 1

                if self._frame_index % 20 == 0:
                    logger.debug(
                        "Observation: %d frames, %d clicks (total)",
                        self._frame_index, _total_clicks_captured,
                    )
                    # ── Click drought warning (once at ~3 min) ──
                    # After 120 frames (~3 min at 1.5s interval) with zero clicks
                    # AND the monitor is installed, the hook is likely not working.
                    if self._frame_index == 120 and _total_clicks_captured == 0 and self._monitor:
                        msg = (
                            "⚠️ 已录制 60 帧 (~90秒) 但未捕获到任何鼠标点击。\n"
                            "可能原因：\n"
                            "  1. 模拟器窗口检测失败（窗口标题是否包含 'mumu'?）\n"
                            "  2. Windows DPI 缩放导致坐标偏移\n"
                            "  3. pywin32 未安装\n"
                            "录制仍在继续，但建议发送 /stop 取消后排查重试。"
                        )
                        if self._notify_fn:
                            try:
                                self._notify_fn(msg)
                            except Exception:
                                pass

            except Exception as e:
                logger.error("Screenshot capture failed: %s", e)
                self._mark_interrupted()
                break

            # Sleep until next poll interval
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0, self._poll_interval - elapsed)
            if sleep_time > 0:
                self._running.wait(sleep_time)

    def _start_mouse_monitor(
        self, emu_window: dict | None, dev_w: int, dev_h: int,
    ) -> tuple[Any, bool]:
        """Start the mouse click monitor. Returns (monitor, ok)."""
        if emu_window is None or emu_window.get("hwnd", 0) == 0:
            msg = (
                "⚠️ 未检测到模拟器窗口，无法捕获鼠标点击。\n"
                "录制将继续（仅截屏），但生成的 skill 将缺少坐标信息。\n"
                "请确认：\n"
                "  1. 模拟器窗口可见且未被最小化\n"
                "  2. 模拟器类型已在 config 中正确配置\n"
                "  3. pywin32 已安装 (pip install pywin32)\n"
                "\n"
                "可运行 python scripts/diagnose_emu_window.py 排查。"
            )
            logger.warning(msg.replace("\n", " "))
            if self._notify_fn:
                try:
                    self._notify_fn(msg)
                except Exception:
                    pass
            # Dump all visible windows to log for diagnostics
            try:
                from src.agent.mouse_hook import _list_all_visible_windows
                logger.info("Visible windows for diagnostics:\n%s",
                          "\n".join(_list_all_visible_windows()))
            except Exception:
                pass
            return None, False

        try:
            from src.agent.mouse_hook import MouseClickMonitor
            monitor = MouseClickMonitor()
            monitor.start(
                emu_window=emu_window,
                device_width=dev_w,
                device_height=dev_h,
            )
            return monitor, True
        except ImportError:
            logger.warning("pywin32 not available — mouse hook disabled")
            return None, False
        except Exception as e:
            logger.warning("Failed to start mouse hook: %s", e)
            return None, False

    def _drain_remaining_clicks(self) -> list[Any]:
        """Drain and return any clicks buffered in the monitor."""
        if self._monitor is None:
            return []
        try:
            return self._monitor.drain_clicks()
        except Exception:
            return []

    def _get_device_resolution(self) -> tuple[int, int]:
        """Get device screen resolution via ADB."""
        try:
            from src.device.adb import get_adb
            adb = get_adb(serial=self.device_serial)
            return adb.get_screen_size()
        except Exception as e:
            logger.warning("Failed to get screen size: %s — using defaults", e)
            return 1600, 900

    @staticmethod
    def _ensure_dpi_awareness() -> None:
        """Set DPI awareness before any window rect queries or mouse hooks.

        Must be called BEFORE GetWindowRect or SetWindowsHookExW.
        """
        try:
            from src.agent.mouse_hook import MouseClickMonitor
            MouseClickMonitor._ensure_dpi_awareness()
        except Exception:
            pass

    def _find_emu_window(self) -> dict | None:
        """Find the emulator window for mouse click mapping.

        Tries emulator_manager.get_emulator_window() first (handles
        multiple emulator types), falls back to find_mumu_window().
        """
        try:
            from src.device.emulator import emulator_manager
            win = emulator_manager.get_emulator_window(self.device_serial)
            if win:
                logger.debug("Emulator window found via emulator_manager: %s", win.get("title"))
                return win
        except Exception:
            pass
        try:
            from src.agent.mouse_hook import find_mumu_window
            return find_mumu_window()
        except Exception as e:
            logger.warning("Failed to find emulator window: %s", e)
            return None

    def _compress_frame(self, img: Any) -> bytes:
        """JPEG compress a frame at full resolution for OCR accuracy.

        Downscaling for the LLM prompt happens later in the extractor.
        """
        from io import BytesIO
        from PIL import Image

        if img.mode == "RGBA":
            img = img.convert("RGB")

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=self._frame_quality, optimize=True)
        return buf.getvalue()

    def _mark_interrupted(self) -> None:
        """Mark the session as interrupted (ADB disconnect, error)."""
        self._running.clear()
        if self._manifest:
            try:
                from src.agent.observation_store import mark_interrupted
                mark_interrupted(self._manifest)
            except Exception:
                pass
        if self._notify_fn:
            self._notify_fn("⚠️ 观察已中断（ADB 连接断开）。已记录的数据已保留，可稍后 /done 尝试分析。")

    def _refresh_emu_window(self) -> None:
        """Periodically refresh the emulator window rect for mouse coords.

        Handles window moves/resizes during recording.
        Uses the same search strategy as _find_emu_window() to cover all
        emulator types, not just MuMu.
        """
        new_win: dict | None = None
        try:
            from src.device.emulator import emulator_manager
            new_win = emulator_manager.get_emulator_window(self.device_serial)
        except Exception:
            pass
        if not new_win:
            try:
                from src.agent.mouse_hook import find_mumu_window
                new_win = find_mumu_window()
            except Exception:
                pass
        if new_win and self._monitor:
            old_rect = self._monitor._emu_rect
            self._monitor.update_window_rect(new_win)
            new_rect = new_win["rect"]
            if old_rect != new_rect:
                logger.debug("Emulator window moved/resized: %s -> %s",
                           old_rect, new_rect)
            self._emu_window = new_win
