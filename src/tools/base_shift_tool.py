"""Base shift tool for Arknights — MAA MaaCore.dll Infrast task.

MAA's infrast automation reads a custom plan JSON (operator names per room)
and places operators — no box scanning, no AI decision-making.  This replaces
the ~100-step manual ADB procedure that LLM agents struggle with.

All modes (default, custom) now execute BaseScheduler plans.  Rotation mode
uses the in-game queue rotation feature for fast Mfg/Trade swap only.
"""

from __future__ import annotations

import ctypes as _ct
import json
import logging
import platform as _platform
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path

from config.settings import config
from src.tools.registry import ToolOutput, registry

logger = logging.getLogger(__name__)

# ── DLL discovery ──────────────────────────────────────────────────────

def _build_maa_dir_candidates() -> list[Path]:
    """Build MAA directory search paths from config → env → fallback."""
    dirs: list[Path] = []
    try:
        _maa = getattr(config, 'maa', None)
        if _maa:
            if _maa.root:
                dirs.append(Path(_maa.root))
            if _maa.resource_dir:
                dirs.append(Path(_maa.resource_dir))
    except Exception:
        pass
    import os as _os
    _env_maa = _os.environ.get("MAA_ROOT") or _os.environ.get("MAA_DATA_DIR")
    if _env_maa:
        dirs.append(Path(_env_maa))
    # Hard-coded fallbacks (developer machine)
    dirs.extend([
        Path("d:/edgedownload/MAA-v6.11.1-win-x64"),
        Path("d:/MAA-v6.11.1-win-x64"),
        Path("d:/MAA"),
    ])
    return dirs


_MAA_DIR_CANDIDATES = _build_maa_dir_candidates()


def _find_maa_dir() -> Path:
    for candidate in _MAA_DIR_CANDIDATES:
        dll = candidate / "MaaCore.dll"
        if dll.exists():
            return candidate
    raise FileNotFoundError(
        "MAA release not found. Download MAA from https://maa.plus"
    )


_DLL: _ct.WinDLL | _ct.CDLL | None = None
_DLL_LOADED = False
_DLL_LOCK = threading.Lock()


def _ensure_maa_dll():
    global _DLL, _DLL_LOADED
    if _DLL_LOADED:
        return
    with _DLL_LOCK:
        if _DLL_LOADED:
            return
        maa_dir = _find_maa_dir()
        dll_path = str(maa_dir / "MaaCore.dll")
        if _platform.system() == "Windows":
            _DLL = _ct.WinDLL(dll_path)
        else:
            _DLL = _ct.CDLL(dll_path)

        _CB_TYPE = _ct.WINFUNCTYPE(None, _ct.c_int, _ct.c_char_p, _ct.c_void_p)

        _DLL.AsstLoadResource.argtypes = (_ct.c_char_p,)
        _DLL.AsstLoadResource.restype = _ct.c_bool
        _DLL.AsstSetUserDir.argtypes = (_ct.c_char_p,)
        _DLL.AsstSetUserDir.restype = _ct.c_bool
        _DLL.AsstCreateEx.argtypes = (_CB_TYPE, _ct.c_void_p)
        _DLL.AsstCreateEx.restype = _ct.c_void_p
        _DLL.AsstDestroy.argtypes = (_ct.c_void_p,)
        _DLL.AsstConnect.argtypes = (_ct.c_void_p, _ct.c_char_p, _ct.c_char_p, _ct.c_char_p)
        _DLL.AsstConnect.restype = _ct.c_bool
        _DLL.AsstAppendTask.argtypes = (_ct.c_void_p, _ct.c_char_p, _ct.c_char_p)
        _DLL.AsstAppendTask.restype = _ct.c_int
        _DLL.AsstStart.argtypes = (_ct.c_void_p,)
        _DLL.AsstStart.restype = _ct.c_bool
        _DLL.AsstStop.argtypes = (_ct.c_void_p,)
        _DLL.AsstStop.restype = _ct.c_bool
        _DLL.AsstRunning.argtypes = (_ct.c_void_p,)
        _DLL.AsstRunning.restype = _ct.c_bool
        _DLL.AsstSetInstanceOption.argtypes = (_ct.c_void_p, _ct.c_int, _ct.c_char_p)
        _DLL.AsstSetInstanceOption.restype = _ct.c_bool

        maa_dir_bytes = str(maa_dir).encode("utf-8")
        _DLL.AsstSetUserDir(maa_dir_bytes)
        _DLL.AsstLoadResource(maa_dir_bytes)
        _DLL_LOADED = True
        logger.info("MaaCore.dll loaded for base-shift: %s", dll_path)


# ── ADB discovery ──────────────────────────────────────────────────────

def _find_adb_exe() -> str:
    import shutil
    adb_path = config.adb.path
    if adb_path and Path(adb_path).exists():
        return adb_path
    adb = shutil.which("adb")
    if adb:
        return adb
    for c in [
        Path("D:/platform-tools/adb.exe"),
        Path.home() / "AppData" / "Local" / "Android" / "Sdk" / "platform-tools" / "adb.exe",
        Path("C:/platform-tools/adb.exe"),
    ]:
        if c.exists():
            return str(c)
    raise RuntimeError("ADB not found. Install ADB or set ADB_PATH in .env")


# ── Helpers ────────────────────────────────────────────────────────────

def _error(text: str) -> ToolOutput:
    return ToolOutput(text=json.dumps({"success": False, "error": text}, ensure_ascii=False))


def _ok(data: dict) -> ToolOutput:
    data.setdefault("success", True)
    return ToolOutput(text=json.dumps(data, ensure_ascii=False))


# ── Custom plan writer ─────────────────────────────────────────────────

_CUSTOM_PLAN_DIR = Path(config.DATA_DIR) / "maa_plans"


def _write_custom_plan(
    operator_box: dict[str, int],
    frontier: list,
    plan_index: int,
    operators: list | None = None,
) -> tuple[Path, list[str], str] | tuple[None, None, str]:
    """Write ALL shifts of the optimizer's plan as separate MAA custom plans.

    Returns (path, facility_list, drone_setting).
    drone_setting is auto-detected from the solution's product configuration.
    """
    if not frontier or plan_index < 0 or plan_index >= len(frontier):
        return None, None, "_NotUse"
    solution = frontier[plan_index]

    if not solution.shifts:
        return None, None, "_NotUse"

    _maa_names_inv = {
        "trading": "Trade", "manufacture": "Mfg", "power": "Power",
        "control": "Control", "meeting": "Reception",
        "hire": "Office", "dormitory": "Dorm",
    }
    _fac_key_map = {
        "Trade": "trading", "Mfg": "manufacture", "Power": "power",
        "Control": "control", "Reception": "meeting",
        "Office": "hire", "Dorm": "dormitory",
    }
    _product_map = {
        "PureGold": "Pure Gold", "CombatRecord": "Battle Record",
        "OriginStone": "Originium Shard", "LMD": "LMD",
        "Orundum": "Orundum",
    }
    # Facilities that don't take a "product" field in MAA custom JSON
    _NO_PRODUCT_FACS: set[str] = {"Control", "Office", "Reception", "Power", "Dorm"}

    maa_plans: list[dict] = []
    all_maa_facilities: list[str] = []

    for shift in solution.shifts:
        rooms: dict[str, list[dict]] = {}
        for room in shift.rooms:
            fac = _fac_key_map.get(room.facility, room.facility.lower())
            product = ""
            if room.facility not in _NO_PRODUCT_FACS:
                product = _product_map.get(room.product, "")

            op_names = [op.name for op in room.operators]
            room_entry: dict = {"operators": op_names}
            if product:
                room_entry["product"] = product
            if not op_names:
                room_entry["autofill"] = True

            rooms.setdefault(fac, []).append(room_entry)
            mn = _maa_names_inv.get(fac)
            if mn and mn not in all_maa_facilities:
                all_maa_facilities.append(mn)

        maa_plans.append({"rooms": rooms})

    # Auto-detect best drone usage from product config
    drone = _auto_drone_from_solution(solution)

    _CUSTOM_PLAN_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up old plan files — keep only the latest.
    for old in sorted(_CUSTOM_PLAN_DIR.glob("base_plan_*.json")):
        try:
            old.unlink()
        except OSError:
            pass

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = _CUSTOM_PLAN_DIR / f"base_plan_{ts}.json"
    payload: dict = {
        "plans": maa_plans,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    # Add morale-driven schedule metadata if available
    if solution.schedule_mode == "morale_driven" and solution.rest_groups:
        payload["schedule_mode"] = "morale_driven"
        payload["groups"] = []
        for g in solution.rest_groups:
            payload["groups"].append({
                "facility": g.facility,
                "product": g.product,
                "operators": [op.name for op in g.operators],
                "work_duration_h": round(g.work_duration, 1),
                "rest_duration_h": round(g.rest_duration, 1),
                "rest_dorm": g.rest_dorm_index + 1,
            })
        if solution.fia_charges:
            payload["fia_charges"] = [
                {"target": fc.operator_name, "time_h": fc.charge_time_h,
                 "coefficient": fc.coefficient, "throttle": fc.throttle}
                for fc in solution.fia_charges
            ]
        payload["work_to_rest_ratio"] = solution.work_to_rest_ratio

        # ── Generate rotation checkpoint plans ──
        # plan[0] = initial full staff (already in maa_plans).
        # Each subsequent plan = one rotation checkpoint where some groups
        # are resting (autofill) and others are working (named operators).
        #
        # Index must match base_scheduler cron table: both iterate the SAME
        # sorted unique time points from facility_by_time (all events, not
        # just "rest" events).  Otherwise rotation_index and plan array
        # index are misaligned.
        #
        # Build group event timeline (same logic as BaseScheduler cron gen).
        facility_by_time: dict[float, set[str]] = {}
        for g in solution.rest_groups:
            if g.work_duration <= 0 and g.rest_duration <= 0:
                continue
            t = g.work_duration
            while t < 24.0:
                facility_by_time.setdefault(t, set()).add(g.facility)
                t += max(g.rest_duration, 0.01)
                if t < 24.0:
                    facility_by_time.setdefault(t, set()).add(g.facility)
                t += max(g.work_duration, 0.01)

        # For each unique time point, compute which groups are working/resting
        # using the same modulo logic as the scheduler display.
        def _group_is_resting(g, at_time: float) -> bool:
            cycle = g.work_duration + g.rest_duration
            if cycle <= 0:
                return False
            pos = at_time % cycle
            return pos >= g.work_duration

        rotation_plan_map: dict[int, dict] = {}
        initial_rooms = maa_plans[0]["rooms"]

        for rot_idx, t in enumerate(sorted(facility_by_time.keys()), start=1):
            # At this time point, which (fac, room_index) pairs are in rest?
            resting_rooms: set[tuple[str, int]] = set()
            for g in solution.rest_groups:
                if _group_is_resting(g, t):
                    _maa_fac = _fac_key_map.get(g.facility, g.facility.lower())
                    for ri in g.room_indices:
                        resting_rooms.add((_maa_fac, ri))

            # Build plan: resting rooms → autofill, working rooms → named ops
            rot_rooms: dict[str, list[dict]] = {}
            for fac, room_list in initial_rooms.items():
                rot_rooms[fac] = []
                for ri, room in enumerate(room_list):
                    if (fac, ri) in resting_rooms:
                        entry = {"operators": [], "autofill": True}
                        if room.get("product"):
                            entry["product"] = room["product"]
                        rot_rooms[fac].append(entry)
                    else:
                        rot_rooms[fac].append(dict(room))

            maa_plans.append({"rooms": rot_rooms})

            h = int(t)
            m = int((t - h) * 60)
            rotation_plan_map[rot_idx] = {"time_h": t, "time_str": f"+{h}h{m:02d}min"}

        if rotation_plan_map:
            payload["rotation_plan_map"] = rotation_plan_map
            logger.info(
                "MAA custom plan: added %d rotation checkpoints (total %d plans)",
                len(rotation_plan_map), len(maa_plans),
            )

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    total_rooms = sum(sum(len(v) for v in p["rooms"].values()) for p in maa_plans)
    mode_tag = "+dorm" if any("dormitory" in str(p) for p in maa_plans) else ""
    logger.info("Wrote MAA custom plan: %s (%d shifts, %d total rooms%s, facilities=%s, drone=%s)",
               path, len(maa_plans), total_rooms, mode_tag, all_maa_facilities, drone)
    return path, all_maa_facilities, drone


def _auto_drone_from_solution(solution) -> str:
    """Auto-detect best drone usage from the solution's product configuration."""
    if not solution.shifts:
        return "_NotUse"
    shift = solution.shifts[0]
    has_orundum = any(r.facility == "Trade" and r.product == "Orundum" for r in shift.rooms)
    has_lmd = any(r.facility == "Trade" and r.product == "LMD" for r in shift.rooms)
    has_combat = any(r.facility == "Mfg" and r.product == "CombatRecord" for r in shift.rooms)

    if has_orundum:
        return "SyntheticJade"
    if has_lmd:
        return "Money"
    if has_combat:
        return "CombatRecord"
    return "_NotUse"


# ── MAA task runner (shared by all modes) ──────────────────────────────

_MAA_INFRAST_TIMEOUT = 900  # 15 min — custom plans may touch all facilities


def _run_maa_infrast(handle, params: dict, mode: str, adb_addr: str) -> ToolOutput:
    """Append Infrast task, start, poll until done, return _ok or _error."""
    results: list[dict] = []
    task_status = ""         # "Success" | "Failed" from top-level callback
    failures: list[str] = []

    _facility_start: dict[str, float] = {}  # subtask → first event timestamp
    _facility_done: list[str] = []           # completed subtask names, in order
    _t0 = _time.time()

    def _callback(_msg_type: int, msg: bytes, _custom_arg) -> None:
        try:
            payload = json.loads(msg.decode("utf-8", errors="replace"))
        except Exception:
            return
        results.append(payload)
        nonlocal task_status
        # MAA v6.11.1 uses "taskchain" for the top-level status callback
        # (e.g. {"taskchain": "Infrast", "status": "Success"}).
        # Older versions used "task" — check both for compatibility.
        status = payload.get("status", "")
        task = payload.get("taskchain", payload.get("task", ""))
        subtask = payload.get("subtask", "")
        if status in ("Success", "Failed") and (
            task == "Infrast" or task.startswith("Infrast")
        ):
            task_status = status
            if status == "Failed":
                what = payload.get("what", payload.get("why", str(payload)[:120]))
                failures.append(what)

        # ── Per-facility timing — total duration from first event to finish ──
        if subtask:
            _now = _time.time()
            if subtask not in _facility_start:
                _facility_start[subtask] = _now
            if status in ("Success", "Failed"):
                _dur = _now - _facility_start[subtask]
                logger.info("MAA subtask %s: %s (%.1fs)", subtask, status, _dur)
                _facility_done.append(subtask)

    cb = _ct.WINFUNCTYPE(None, _ct.c_int, _ct.c_char_p, _ct.c_void_p)(_callback)
    handle = _DLL.AsstCreateEx(cb, None)
    if not handle:
        return _error("MAA AsstCreateEx 失败")

    _DLL.AsstSetInstanceOption(handle, 2, b"maatouch")

    try:
        adb_exe = _find_adb_exe()
        ok = _DLL.AsstConnect(
            handle, adb_exe.encode("utf-8"),
            adb_addr.encode("utf-8"), b"General",
        )
        if not ok:
            return _error(f"MAA 连接设备失败: {adb_addr}，adb: {adb_exe}")

        params_json = json.dumps(params, ensure_ascii=False)
        task_id = _DLL.AsstAppendTask(handle, b"Infrast", params_json.encode("utf-8"))

        if task_id == 0:
            return _error(
                f"MAA 拒绝任务（task_id=0），mode={mode}。"
                "参数格式可能不正确或必要字段缺失。"
            )

        logger.info(
            "base_shift_maa: mode=%s facilities=%s drones=%s threshold=%.2f task_id=%d",
            mode, params.get("facility", "custom"),
            params.get("drones", "_NotUse"), params.get("threshold", 0.3), task_id,
        )

        if not _DLL.AsstStart(handle):
            return _error("MAA AsstStart 失败")

        deadline = _time.time() + _MAA_INFRAST_TIMEOUT
        last_report = 0
        while _DLL.AsstRunning(handle) and _time.time() < deadline:
            _time.sleep(0.5)
            elapsed = _time.time() - (deadline - _MAA_INFRAST_TIMEOUT)
            if elapsed - last_report >= 10:
                last_report = elapsed
                logger.info("base_shift_maa: running... %.0fs (%d events)",
                           elapsed, len(results))

        _DLL.AsstStop(handle)
        ran_out = _time.time() >= deadline

        # ── Facility timing summary ──
        if _facility_done:
            logger.info("MAA facility order: %s", " → ".join(_facility_done))

        # count swap events — match MAA v6.11.1 callback patterns.
        # Custom plans (mode=10000) generate room-level completion events:
        #  {"taskchain": "Infrast", "subtask": "InfrastMfgTask", "status": "Success"}
        #  {"taskchain": "Infrast", "subtask": "InfrastTradeTask", "status": "Success"}
        # Also match "what" field (Chinese text) and general Infrast success.
        _FACILITY_KEYS = ("Mfg", "Trade", "Power", "Control", "Reception", "Office", "Dorm")
        swap_events = [r for r in results
                       if r.get("status") == "Success"
                       and ("Infrast" in str(r.get("subtask", ""))
                            or "Infrast" in str(r.get("task", ""))
                            or "干员" in str(r)
                            or "进驻" in str(r)
                            or any(fk in str(r.get("subtask", "")) for fk in _FACILITY_KEYS)
                            or any(fk in str(r.get("what", "")) for fk in _FACILITY_KEYS))]

        if ran_out and task_status != "Success":
            return _error(
                f"MAA 基建换班超时（{_MAA_INFRAST_TIMEOUT // 60}分钟），"
                f"已收到 {len(results)} 个事件，swap_events={len(swap_events)}，"
                f"status={task_status or 'none'}。"
            )

        return _ok({
            "mode": mode,
            "events": len(results),
            "swaps": len(swap_events),
            "status": task_status or "completed",
            "errors": failures[:5],
        })

    finally:
        _DLL.AsstDestroy(handle)


# ── Public tool ────────────────────────────────────────────────────────


def base_shift_maa(
    mode: str = "default",
    facility: str = "",
    drones: str = "_NotUse",
    threshold: float = 0.3,
    plan_index: int = 0,
    rotation_index: int = 0,
    adb_addr: str = "127.0.0.1:16384",
) -> ToolOutput:
    """Execute Arknights base shift via MAA's Infrast task engine."""
    try:
        _ensure_maa_dll()
    except FileNotFoundError as e:
        return _error(str(e))

    mode_int = {"default": 10000, "custom": 10000, "rotation": 20000}.get(mode, 10000)

    if mode in ("default", "custom"):
        from src.intelligence.arknights.base_scheduler import BaseScheduler
        cache = BaseScheduler._get_cache()
        if not cache or "frontier" not in cache:
            return _error(
                "没有已缓存的排班方案。"
                "请先 scan_depot() → scan_operator_box() 让优化器生成方案，再执行换班。"
            )

        # Resolve plan_index via display→frontier mapping if available.
        plan_map = cache.get("plan_index_map")
        if plan_map and isinstance(plan_map, dict) and plan_index in plan_map:
            frontier_index = plan_map[plan_index]
        else:
            frontier_index = plan_index

        plan_path, maa_fac, auto_drone = _write_custom_plan(
            cache["box"], cache["frontier"], frontier_index, cache.get("operators"),
        )
        if plan_path is None:
            return _error(
                f"方案 #{plan_index + 1} 不存在。"
                f"共 {len(cache['frontier'])} 个方案。"
            )

        # Reception (会客室) and Office (办公室) are kept in MAA.
        # With custom plans (no box scanning needed), MAA just places
        # named operators — fast and reliable. LLM verifies after.

        # Auto-detect drone if user hasn't explicitly set one.
        effective_drones = drones if drones != "_NotUse" else auto_drone

        if facility:
            user_facs = [f.strip() for f in facility.split(",") if f.strip()]
            effective_fac = [f for f in user_facs if f in maa_fac]
            if effective_fac:
                maa_fac = effective_fac
                logger.info("base_shift_maa: facility override → %s", maa_fac)

        params = {
            "mode": mode_int,
            "filename": plan_path.as_posix(),
            "plan_index": rotation_index,
            "facility": maa_fac,
            "drones": effective_drones,
        }

    elif mode == "rotation":
        params = {
            "mode": mode_int,
            "facility": ["Mfg", "Trade"],
            "drones": drones,
            "threshold": threshold,
        }

    else:
        return _error(f"不支持的模式: {mode}。可选: default, custom, rotation")

    return _run_maa_infrast(None, params, mode, adb_addr)


# ── Register ───────────────────────────────────────────────────────────

def _maa_available() -> bool:
    """Check if MAA DLL is available before offering to LLM."""
    for candidate in _MAA_DIR_CANDIDATES:
        if (candidate / "MaaCore.dll").exists():
            return True
    return False


try:
    from src.tools.adb_control import _adb_available
except ImportError:
    _adb_available = lambda: True

registry.register(
    name="base_shift_maa",
    game="arknights",
    check_fn=lambda: _maa_available() and _adb_available(),
    description=(
        "【基建自动换班 — MAA引擎】执行 BaseScheduler 最优排班方案。\n"
        "⚠️ 前置条件：必须先 scan_depot() + scan_operator_box() 让系统生成换班方案。\n"
        "无方案时调用会报错——先去出方案再换班。\n"
        "MAA 处理全部设施（制造站/贸易站/发电站/控制中枢/会客室/办公室/宿舍）。\n"
        "LLM 在 MAA 完成后验证满员即可，无需手动换人。\n"
        "\n"
        "参数：\n"
        "  - plan_index: 方案编号（默认 0=推荐方案）\n"
        "  - rotation_index: 轮换索引（0=初始全站，1+=定时轮换点），由 Cron 指令传递\n"
        "  - drones: '_NotUse' | 'Money' | 'SyntheticJade' | ...\n"
    ),
    parameters={
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["default", "custom", "rotation"],
                "description": "换班模式。default=日常换班，custom=优化器方案，rotation=队列轮换",
            },
            "facility": {
                "type": "string",
                "description": "逗号分隔的设备列表，如 'Mfg,Trade,Power'。空=全部。",
            },
            "drones": {
                "type": "string",
                "enum": ["_NotUse", "Money", "SyntheticJade", "CombatRecord", "PureGold", "OriginStone"],
                "description": "无人机用途。默认 _NotUse",
            },
            "threshold": {
                "type": "number",
                "description": "心情阈值 0.0-1.0。低于此值就换。默认 0.3",
            },
            "plan_index": {
                "type": "integer",
                "description": "方案编号(0=推荐方案)。",
            },
            "rotation_index": {
                "type": "integer",
                "description": "轮换索引。0=初始全站，1+=定时轮换点。由 CronCreate 指令传递，不要手动设。",
            },
        },
    },
    handler=base_shift_maa,
)
