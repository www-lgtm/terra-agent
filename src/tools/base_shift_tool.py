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
import os
import platform as _platform
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path

from config.settings import config
from src.tools.registry import ToolOutput, registry

logger = logging.getLogger(__name__)


def _notify_screenshot(message: str) -> None:
    """Send screenshot + message to user via WeChat notification."""
    try:
        from src.agent.screen_injector import capture_screen_jpeg
        ctx = getattr(threading.current_thread(), '_terra_agent_ctx', None)
        if ctx is None:
            return
        ctx._notify(message, notify_type="screenshot", image_b64=capture_screen_jpeg())
        logger.info("base_shift: notify — %s", message[:120])
    except Exception:
        logger.warning("base_shift: notify failed", exc_info=True)


def _notify_plan_image(message: str, image_b64: str) -> None:
    """Send a generated plan image + message to user via WeChat."""
    try:
        ctx = getattr(threading.current_thread(), '_terra_agent_ctx', None)
        if ctx is None:
            return
        ctx._notify(message, notify_type="screenshot", image_b64=image_b64)
        logger.info("base_shift: plan image notify — %s", message[:120])
    except Exception:
        logger.warning("base_shift: plan image notify failed", exc_info=True)


def _render_plan_image(
    frontier: list, layout_desc: str, operator_count: int,
    coverage: float, depot_stock,
) -> str:
    """Render ALL plans with A/B dual-shift tables + rotation, return base64 JPEG."""
    import io as _io
    import base64 as _b64
    from PIL import Image, ImageDraw, ImageFont

    _font_paths = []
    if _platform.system() == "Windows":
        _font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
    elif _platform.system() == "Linux":
        _font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
    elif _platform.system() == "Darwin":
        _font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ]
    _font = None
    for _fp in _font_paths:
        if Path(_fp).exists():
            try:
                _font = ImageFont.truetype(_fp, 14)
                _font_sm = ImageFont.truetype(_fp, 12)
                _font_bold = ImageFont.truetype(_fp, 16)
                _font_title = ImageFont.truetype(_fp, 19)
                _font_section = ImageFont.truetype(_fp, 17)
            except Exception:
                continue
            break
    if _font is None:
        _font = ImageFont.load_default()
        _font_sm = _font_bold = _font_title = _font_section = _font

    COL_W = [80, 55, 230, 230, 48, 48]
    ROW_H = 24
    PAD = 10
    CANVAS_W = sum(COL_W) + PAD * 2 + 14
    _HEADERS = ["设施", "产品", "A队（第一班）", "B队（A休息时顶上）", "AEff", "BEff"]

    # ── Summary classification ──
    _summary: list[str] = []
    for _pi, _p in enumerate(frontier[:6]):
        if _p.daily_orundum > 10:
            _cat = "🟡搓玉"
        elif _p.daily_lmd <= 0 and _p.daily_combat_record > 0:
            _cat = "📋作战记录"
        elif _p.daily_combat_record > _p.daily_lmd * 1.5 and _p.daily_lmd > 0:
            _cat = "📋作战记录"
        else:
            _cat = "💰龙门币"
        _summary.append(
            f"方案{_pi + 1} {_cat}  玉{_p.daily_orundum:.0f}  币{_p.daily_lmd:.0f}  记录{_p.daily_combat_record:.0f}"
        )

    _stock_note = "（未扫仓库，无法计算库存天数）"
    if depot_stock is not None:
        from src.intelligence.arknights.base_optimizer import BaseOptimizer
        _bal = BaseOptimizer.check_resource_balance(frontier[0], None, depot_stock=depot_stock)
        _gsd = _bal.get("gold_stockpile_days")
        if _gsd is not None:
            _stock_note = f"库存可支撑约{_gsd:.0f}天"

    ROT_ROW_H = 22
    SECTION_GAP = 28
    SECTION_HEADER_H = 44
    _total_h = 84
    _plan_meta: list[dict] = []

    _PROD_FACS = {"Trade", "Mfg", "Power"}
    for plan in frontier[:6]:
        _sh = getattr(plan, 'shifts', []) or []
        is_dual = len(_sh) >= 2

        # Build A/B rows: always show A room; show B only if dual & populated
        _rows: list[list[str]] = []
        _b_populated = 0
        if is_dual:
            _a_rooms = {f"{r.facility}{r.index}": r for r in _sh[0].rooms}
            _b_rooms = {f"{r.facility}{r.index}": r for r in _sh[1].rooms}
            for _key in sorted(_a_rooms):
                _ar = _a_rooms[_key]
                _br = _b_rooms.get(_key)
                _a_ops = _fmt_ops(_ar)
                _b_ops, _b_eff = "—", "—"
                if _br and _br.operators:
                    _b_ops = _fmt_ops(_br)
                    _b_eff = f"{_br.total_efficiency():.0f}%"
                    _b_populated += 1
                _rows.append([
                    f"{_ar.facility}{_ar.index + 1}", _ar.product or "—",
                    _a_ops, _b_ops,
                    f"{_ar.total_efficiency():.0f}%", _b_eff,
                ])
        else:
            for r in _sh[0].rooms:
                _rows.append([
                    f"{r.facility}{r.index + 1}", r.product or "—",
                    _fmt_ops(r), "—",
                    f"{r.total_efficiency():.0f}%", "—",
                ])

        # Rotation: compute from A队 production rooms
        _rot_lines: list[str] = []
        _rot_h = 0
        if is_dual and _sh[0].rooms:
            _a_prod = [r for r in _sh[0].rooms if r.facility in _PROD_FACS]
            # Group by (facility, product)
            _by_fp: dict = {}
            for r in _a_prod:
                k = (r.facility, r.product)
                _by_fp.setdefault(k, []).append(r)
            if _by_fp:
                # Compute work duration for each group: 24 * (1.0 - threshold) / max_drain
                _mood_th = 0.35
                _rot_lines.append("设施   产品       工作    休息    周期      替补")
                for (_fac, _prod), _rooms in sorted(_by_fp.items()):
                    _all_ops = []
                    for r in _rooms:
                        _all_ops.extend(r.operators)
                    if not _all_ops:
                        continue
                    _max_drain = max(op.morale_drain_per_hour(_fac) for op in _all_ops if hasattr(op, 'morale_drain_per_hour'))
                    if _max_drain <= 0:
                        continue
                    _work_h = 24.0 * (1.0 - _mood_th) / max(_max_drain, 0.01)
                    _work_h = min(_work_h, 24.0)
                    _rest_h = _work_h * (_max_drain / (24.0 - _max_drain * _work_h / 24.0))
                    # Simplified: rest = (mood_lost / recovery_rate), recovery ≈ 2.0/h base
                    _mood_lost = _work_h * _max_drain
                    _recovery_rate = 2.0
                    _rest_h = _mood_lost / _recovery_rate
                    _cycle = _work_h + _rest_h
                    # B队覆盖：same key in B shift
                    _b_covers = "✓" if is_dual else "—"
                    if is_dual and len(_sh) >= 2:
                        _br = [r for r in _sh[1].rooms if r.facility == _fac and r.product == _prod and r.operators]
                        _b_covers = f"{len(_br)}房" if _br else "✗"
                    _rot_lines.append(
                        f"{_fac:<6} {_prod:<8} {_work_h:.1f}h   {_rest_h:.1f}h   "
                        f"{_cycle:.1f}h     {_b_covers}"
                    )
                _rot_h = 20 + len(_rot_lines) * ROT_ROW_H + 8

        _section_h = (SECTION_HEADER_H + len(_rows) * ROW_H + ROW_H + _rot_h + 36
                      if _rot_lines else SECTION_HEADER_H + len(_rows) * ROW_H + ROW_H + 20)
        _plan_meta.append({
            "plan": plan, "rows": _rows, "section_h": _section_h,
            "rot_lines": _rot_lines, "rot_h": _rot_h,
            "is_dual": is_dual, "b_populated": _b_populated,
        })
        _total_h += _section_h + SECTION_GAP

    # ── Draw ──
    CANVAS_H = _total_h + PAD + 30
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "#1a1a2e")
    draw = ImageDraw.Draw(img)

    y = PAD + 6
    draw.text((PAD, y), "🏗️ 基建排班方案", fill="#e0e0e0", font=_font_title)
    y += 24
    draw.text((PAD, y),
              f"{layout_desc}  ·  {operator_count}名干员  ·  覆盖率{coverage:.0%}  ·  {len(frontier)}个方案",
              fill="#8899aa", font=_font_sm)
    y += 22
    for _i, _line in enumerate(_summary):
        draw.text((PAD, y), _line, fill="#ffcc66" if _i == 0 else "#8899aa", font=_font_sm)
        y += 20
    y += 4
    draw.text((PAD, y), _stock_note, fill="#88aacc", font=_font_sm)
    y += 24

    for _si, _pm in enumerate(_plan_meta):
        _is_rec = _si == 0
        _bg = "#1a1a30" if _is_rec else "#16162a"
        draw.rectangle([(PAD - 4, y), (CANVAS_W - PAD + 4, y + _pm["section_h"])], fill=_bg)

        _cat_label = _summary[_si] if _si < len(_summary) else f"方案{_si + 1}"
        _star = "⭐ " if _is_rec else ""
        draw.text((PAD, y + 4), f"{_star}{_cat_label}", fill="#ffcc66" if _is_rec else "#cccccc", font=_font_section)
        y += SECTION_HEADER_H - 8

        # Table header
        _hdr_x = PAD
        for _ci, _h in enumerate(_HEADERS):
            draw.text((_hdr_x, y), _h, fill="#7799bb", font=_font_bold)
            _hdr_x += COL_W[_ci]
        y += ROW_H
        draw.line([(PAD, y - 1), (CANVAS_W - PAD, y - 1)], fill="#334455", width=1)

        # Rows
        for _ri, _row in enumerate(_pm["rows"]):
            _row_x = PAD
            _rb = "#1e1e38" if _ri % 2 == 0 else "#252545"
            draw.rectangle([(PAD, y), (CANVAS_W - PAD, y + ROW_H)], fill=_rb)
            for _ci, _cell in enumerate(_row):
                if _ci == 2: _c = "#dddddd"
                elif _ci == 3: _c = "#ddcc88" if _cell != "—" else "#556677"
                elif _ci in (4, 5): _c = "#66cc88"
                else: _c = "#8899aa"
                draw.text((_row_x, y + 3), _cell, fill=_c, font=_font)
                _row_x += COL_W[_ci]
            y += ROW_H

        # Rotation schedule
        _rot = _pm["rot_lines"]
        if _rot:
            y += 6
            draw.text((PAD, y), "🔄 换班节奏（基于A队心情消耗，心情降至35%后 B队顶上）",
                      fill="#ffaa66", font=_font_sm)
            y += 20
            for _rl in _rot:
                draw.text((PAD, y), _rl, fill="#bbbbbb", font=_font_sm)
                y += ROT_ROW_H
            y += 4

        # B team summary — only warn for production facilities (Trade/Mfg)
        if _pm["is_dual"]:
            _bp = _pm["b_populated"]
            _total_a_rooms = len(_pm["rows"])
            # Count production rooms (Trade, Mfg) — only these need substitutes
            _prod_rooms = sum(1 for r in _pm["rows"]
                            if r[0].startswith(("Trade", "Mfg")))
            # Count B-team production room coverage
            _bp_prod = sum(1 for r in _pm["rows"]
                         if r[0].startswith(("Trade", "Mfg")) and r[5] != "—")
            if _bp_prod == 0 and _prod_rooms > 0:
                draw.text((PAD, y), "⚠️ B队无可用替补干员，A队休息时生产设施停产",
                          fill="#cc8866", font=_font_sm)
                y += 18
            elif _bp_prod < _prod_rooms:
                draw.text((PAD, y),
                          f"💡 B队覆盖 {_bp_prod}/{_prod_rooms} 个生产房间，空房在A队休息期间停产",
                          fill="#88aacc", font=_font_sm)
                y += 18

        y += SECTION_GAP - 8

    # Footer
    draw.text((PAD, y), "按上图设置进驻预设。A/B两队交替：A工作→A休息B顶上→A恢复后换回",
              fill="#667788", font=_font_sm)

    img_jpg = _io.BytesIO()
    img.convert("RGB").save(img_jpg, format="JPEG", quality=85)
    return _b64.b64encode(img_jpg.getvalue()).decode()


def _fmt_ops(room) -> str:
    """Format operator list for a room — compact with overflow count."""
    if not room.operators:
        return "（空缺）"
    ops = "、".join(op.name for op in room.operators[:6])
    if len(room.operators) > 6:
        ops += f"…({len(room.operators)}人)"
    return ops

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

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
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

    # Write to temp file first, then atomically rename (Windows os.replace is atomic).
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)

    # Clean up old plan files — keep only the latest.
    for old in sorted(_CUSTOM_PLAN_DIR.glob("base_plan_*.json")):
        if old == path:
            continue
        try:
            old.unlink()
        except OSError:
            pass

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


def _run_maa_infrast(params: dict, mode: str, adb_addr: str) -> ToolOutput:
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

    return _run_maa_infrast(params, mode, adb_addr)


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


# ═══════════════════════════════════════════════════════════════════════
# base_plan — show all facility×operator plans from optimizer
# ═══════════════════════════════════════════════════════════════════════

_FACILITY_NAMES: dict[str, str] = {
    "control": "控制中枢", "trading": "贸易站", "manufacture": "制造站",
    "power": "发电站", "dormitory": "宿舍", "meeting": "会客室",
    "hire": "办公室", "processing": "加工站",
}

_CONFIG_NAMES: dict[str, str] = {
    "orundum": "搓玉", "lmd": "龙门币", "combat_record": "作战记录",
}


def base_plan() -> ToolOutput:
    """Generate ALL feasible base scheduling plans for the user to choose from.

    Runs the SA optimizer on the latest operator box (default layout 243)
    and returns every plan on the Pareto frontier — from pure 搓玉 to pure
    练级.  No warehouse data required.

    The LLM reads the plan summaries and helps the user pick the best one.
    """
    from src.intelligence.arknights.base_optimizer import BaseOptimizer, parse_inventory
    from src.intelligence.arknights.base_scheduler import BaseScheduler

    # ── 1. Load operator box ───────────────────────────────────────
    operator_box: dict[str, int] = {}
    cache = BaseScheduler._get_cache()
    if cache and cache.get("box"):
        operator_box = cache["box"]
        logger.info("base_plan: loaded %d ops from scheduler cache", len(operator_box))

    if not operator_box:
        from src.intelligence.arknights.base_chain import SESSION_DIR, read_box_file
        if SESSION_DIR.exists():
            sessions = sorted(
                [s for s in SESSION_DIR.iterdir()
                 if s.is_dir() and (s / "box.json").exists()],
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if sessions:
                try:
                    box_data = read_box_file(sessions[0].name)
                    if box_data and len(box_data) > 2:
                        operator_box = box_data
                        logger.info("base_plan: loaded %d ops from session %s",
                                   len(operator_box), sessions[0].name)
                except Exception as e:
                    logger.warning("base_plan: failed to load box: %s", e)

    if not operator_box or len(operator_box) <= 2:
        return ToolOutput(text=json.dumps({
            "success": False,
            "error": "没有干员数据",
            "message": "请先 scan_operator_box() 扫描干员列表。",
        }, ensure_ascii=False))

    # ── 2. Load depot (optional) ───────────────────────────────────
    depot_stock = _load_depot_stock(cache)

    # ── 3. Knowledge context ───────────────────────────────────────
    from src.intelligence.base import IntelligenceContext
    ctx = IntelligenceContext(game="arknights", knowledge=None)
    try:
        from src.knowledge import KnowledgeBase
        ctx.knowledge = KnowledgeBase()
    except Exception:
        pass

    # ── 4. Run optimizer (read layout from cache, fallback to 243) ──
    inventory = parse_inventory("")
    NEUTRAL = (0.40, 0.35, 0.25)
    layout = cache.get("layout") if cache and cache.get("layout") else "243"

    try:
        optimizer = BaseOptimizer(ctx.knowledge)
        logger.info("base_plan: optimizing layout=%s operators=%d", layout, len(operator_box))
        frontier = optimizer.solve_pareto(
            operator_box, layout, num_shifts=2,
            sort_weights=NEUTRAL,
            inventory=inventory,
            mood_threshold=0.35,
            material_stock=depot_stock,
        )
        if not frontier:
            return ToolOutput(text=json.dumps({
                "success": False,
                "error": "无法生成方案",
                "message": "当前干员列表无法填满布局。",
            }, ensure_ascii=False))

        # ── Category-based dedup: one best plan per goal ───────────
        _best_per_cat: dict[str, tuple[int, float]] = {}
        for pi, plan in enumerate(frontier):
            if plan.daily_orundum > 10:
                cat, score = "搓玉", plan.daily_orundum
            elif plan.daily_lmd > 0 and plan.daily_combat_record > plan.daily_lmd * 1.5:
                cat, score = "作战记录", plan.daily_combat_record
            elif plan.daily_combat_record > 0 and plan.daily_lmd == 0:
                cat, score = "作战记录", plan.daily_combat_record
            else:
                cat, score = "龙门币", plan.daily_lmd
            if cat not in _best_per_cat or score > _best_per_cat[cat][1]:
                _best_per_cat[cat] = (pi, score)

        _keep_idx = sorted(idx for idx, _ in _best_per_cat.values())
        frontier = [frontier[i] for i in _keep_idx]
        logger.info("base_plan: category-deduped %d → %d plans (%s)",
                   len(_best_per_cat), len(frontier),
                   ", ".join(f"{c}" for c in _best_per_cat))
        coverage = frontier[0].coverage if frontier else 0
    except Exception as e:
        logger.error("base_plan: optimization failed: %s", e, exc_info=True)
        return ToolOutput(text=json.dumps({
            "success": False,
            "error": f"优化计算失败: {e}",
        }, ensure_ascii=False))

    layout_desc = {
        "243": "243（2贸4制3电）", "333": "333（3贸3制3电）",
        "252": "252（2贸5制2电）", "153": "153（1贸5制3电）",
    }.get(layout, layout)

    operators_resolved = optimizer._resolve_operators(operator_box)

    # ── 5. Format ALL plans ─────────────────────────────────────────
    lines = [
        f"## 基建排班方案（{len(operator_box)}名干员，{layout_desc}）",
        "",
        f"覆盖率 {coverage:.0%}，共 {len(frontier)} 个生产方案：",
        "",
    ]

    summaries: list[dict] = []

    for pi, plan in enumerate(frontier):
        # ── Quick classification ──
        if plan.daily_orundum > 10:
            category = "🟡 搓玉"
        elif plan.daily_lmd > plan.daily_combat_record * 2:
            category = "💰 龙门币"
        elif plan.daily_combat_record > plan.daily_lmd * 1.5:
            category = "📋 作战记录"
        else:
            category = "⚖️ 均衡"

        config_parts = []
        for shift in plan.shifts:
            for room in shift.rooms:
                if room.product:
                    label = _CONFIG_NAMES.get(room.product, room.product)
                    config_parts.append(f"{_FACILITY_NAMES.get(room.facility, room.facility)}{room.index+1}={label}")

        # ── Compact summary ──
        plan_lines = [f"### 方案{pi + 1}（{category}）"]
        plan_lines.append("")
        plan_lines.append(f"- 日产：合成玉 **{plan.daily_orundum:.0f}** | 龙门币 **{plan.daily_lmd:.0f}** | 作战记录 **{plan.daily_combat_record:.0f}**")

        # Show facility assignments
        if plan.shifts:
            plan_lines.append("")
            for shift in plan.shifts:
                for room in shift.rooms:
                    fac = f"{_FACILITY_NAMES.get(room.facility, room.facility)}{room.index+1}"
                    prod = f" ({room.product})" if room.product else ""
                    if room.operators:
                        ops = "、".join(
                            f"{op.name}" + (f"(E{op.elite})" if getattr(op, 'elite', 0) >= 2 else "")
                            for op in room.operators
                        )
                    elif room.facility in ("power", "dormitory"):
                        ops = "任意干员（不影响产出）"
                    else:
                        ops = "空缺"
                    plan_lines.append(f"**{fac}{prod}**：{ops}")

        # ── Rotation schedule ──
        if plan.schedule_mode == "morale_driven" and plan.rest_groups:
            plan_lines.append("")
            plan_lines.append(f"**🔄 换班节奏**（心情降到35%以下休息）：")
            for g in plan.rest_groups:
                cycle = g.work_duration + g.rest_duration
                plan_lines.append(
                    f"  - {chr(65 + g.group_id)}组（{g.facility} {g.product}）："
                    f"工作 {g.work_duration:.1f}h → 休息 {g.rest_duration:.1f}h → "
                    f"每 {cycle:.1f}h 换一次"
                )
            if plan.work_to_rest_ratio > 0:
                plan_lines.append(f"  - 工休比 {plan.work_to_rest_ratio:.1f}:1")

        # Sustainability check
        if depot_stock:
            from src.intelligence.arknights.base_optimizer import BaseOptimizer as BO
            bal = BO.check_resource_balance(plan, inventory, depot_stock=depot_stock)
            gsd = bal.get("gold_stockpile_days")
            if gsd is not None:
                plan_lines.append(f"  ↳ 库存可支撑约{gsd:.0f}天")
            if bal.get("warnings"):
                plan_lines.append(f"  ↳ ⚠️ {bal['warnings'][0]}")
            stockpile_days_str = f"{gsd:.0f}" if gsd is not None else None
        else:
            plan_lines.append(f"  ↳ 未扫描仓库，无法估算可持续天数")
            stockpile_days_str = None

        lines.extend(plan_lines)
        lines.append("")

        summary_entry = {
            "index": pi + 1,
            "category": category,
            "orundum": f"{plan.daily_orundum:.0f}",
            "lmd": f"{plan.daily_lmd:.0f}",
            "combat_record": f"{plan.daily_combat_record:.0f}",
        }
        if stockpile_days_str:
            summary_entry["stockpile_days"] = stockpile_days_str
        summaries.append(summary_entry)

    # ── Quick-reference table ──
    lines.append("---")
    lines.append("### 📊 速览")
    lines.append("")
    lines.append("| # | 方向 | 合成玉/天 | 龙门币/天 | 作战记录/天 |")
    lines.append("|---|------|-----------|------------|-------------|")
    for s in summaries:
        lines.append(f"| {s['index']} | {s['category']} | {s['orundum']} | {s['lmd']} | {s['combat_record']} |")
    lines.append("")

    if not depot_stock:
        lines.append("> ⚠️ 未扫描仓库，库存天数无法评估。说「扫描仓库」即可。")

    plan_text = "\n".join(lines)

    # ── 6. Cache ───────────────────────────────────────────────────
    BaseScheduler._set_cache({
        "frontier": frontier,
        "box": operator_box,
        "layout": layout,
        "layout_desc": layout_desc,
        "operators": operators_resolved,
        "inventory": inventory,
        "depot_stock": depot_stock,
        "plan_index_map": {i: fi for i, fi in enumerate(_keep_idx)},
    })

    logger.info("base_plan: %d plans generated for %d operators",
               len(frontier), len(operator_box))

    # ── 7. Auto-notify ──────────────────────────────────────────────
    # Send ALL plans as a single tall table image so the user can see
    # every room's operator assignments for every plan.
    summary_lines = [
        f"🏗️ 基建排班方案（{len(operator_box)}名干员，{layout_desc}，覆盖率{coverage:.0%}）",
        "",
        f"共 {len(frontier)} 个方案，详情见图片 ↓",
    ]

    # Render all plans as one tall PNG
    try:
        _img = _render_plan_image(frontier, layout_desc,
                                  len(operator_box), coverage, depot_stock)
        _notify_plan_image("\n".join(summary_lines), _img)
    except Exception:
        logger.warning("base_plan: failed to render plan image, fallback to screenshot", exc_info=True)
        _notify_screenshot("\n".join(summary_lines))

    # ── Warehouse note in tool output ──
    _wh_hint = ""
    if depot_stock is None:
        _wh_hint = (
            "\n\n⚠️ 没有仓库数据，各方案的库存可持续天数未计算。"
            "下次排班时，先按技能流程去仓库截图问用户材料数量，"
            "save_depot_resources 写入后再出方案，就能看到每个方案能撑几天。"
        )

    return ToolOutput(text=json.dumps({
        "success": True,
        "layout": layout_desc,
        "operator_count": len(operator_box),
        "plan_count": len(frontier),
        "has_depot": depot_stock is not None,
        "message": plan_text + _wh_hint,
    }, ensure_ascii=False))


def _load_depot_stock(cache: dict | None):
    """Load depot stock from cache or latest session. Thin wrapper over shared utility."""
    from src.intelligence.arknights.base_chain import load_depot_stock_from_session
    return load_depot_stock_from_session(cache)


registry.register(
    name="base_plan",
    game="arknights",
    description=(
        "【基建排班方案生成】全自动列出所有可行方案，内置截图通知发送给用户。\n"
        "🔴 工具已内置截图通知！调用后直接 subtask_done，不要手动排班、不要进基建、不要 notify_with_screen。\n"
        "前端：有 scan_operator_box() 缓存即可，仓库可选。"
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
    handler=base_plan,
)
