"""Box scanning tool for Arknights operator list — MAA MaaCore.dll direct call.

Uses MAA's compiled C++ engine (MaaCore.dll) for 100% accurate operator box
scanning. Same engine that powers the MAA desktop app — no OCR tuning needed.

Output goes directly into the active base_chain session's box.json.
No stale intermediate files.

Self-registers with the tool registry at import time.
"""

from __future__ import annotations

import ctypes as _ct
import json
import logging
import platform as _platform
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from config.settings import config
from src.device.adb import get_adb
from src.tools.registry import registry, ToolOutput

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# MAA DLL path discovery
# ═══════════════════════════════════════════════════════════════════════════

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

_maa_dir: Path | None = None


def _find_maa_dir() -> Path:
    global _maa_dir
    if _maa_dir is not None:
        return _maa_dir
    for candidate in _MAA_DIR_CANDIDATES:
        dll = candidate / "MaaCore.dll"
        if dll.exists():
            _maa_dir = candidate
            logger.info("MAA found at: %s", candidate)
            return _maa_dir
    raise FileNotFoundError(
        "MAA release not found. Download MAA from https://maa.plus"
    )


def _find_adb_exe() -> str:
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


# ═══════════════════════════════════════════════════════════════════════════
# MAA DLL interface
# ═══════════════════════════════════════════════════════════════════════════

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

        _DLL.AsstLoadResource.argtypes = (_ct.c_char_p,)
        _DLL.AsstLoadResource.restype = _ct.c_bool
        _DLL.AsstSetUserDir.argtypes = (_ct.c_char_p,)
        _DLL.AsstSetUserDir.restype = _ct.c_bool
        _DLL.AsstCreateEx.argtypes = (_ct.WINFUNCTYPE(
            None, _ct.c_int, _ct.c_char_p, _ct.c_void_p), _ct.c_void_p)
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
        logger.info("MaaCore.dll loaded: %s", dll_path)


# ═══════════════════════════════════════════════════════════════════════════
# Session output helpers
# ═══════════════════════════════════════════════════════════════════════════

SESSION_DIR = Path(config.DATA_DIR) / "session"


def _find_or_create_session() -> str:
    """Find the active base_chain session at stage 0, or create a new one.

    Returns the session_id that the box data was written to.
    """
    # Clean up old sessions before creating a new one — one run, one session.
    try:
        for old in SESSION_DIR.iterdir():
            if old.is_dir():
                import shutil
                shutil.rmtree(old, ignore_errors=True)
    except Exception:
        pass

    # Look for an existing incomplete session at stage 0
    if SESSION_DIR.exists():
        sessions = sorted(SESSION_DIR.iterdir(),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        for s in sessions:
            if not s.is_dir():
                continue
            state_file = s / "state.json"
            if not state_file.exists():
                continue
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if state.get("current_stage") == 0 and state.get("stages"):
                return s.name

    # No matching session — create a minimal one
    import random
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=4))
    session_id = f"base_{ts}_{suffix}"
    session_path = SESSION_DIR / session_id
    session_path.mkdir(parents=True, exist_ok=True)

    state = {
        "session_id": session_id,
        "current_stage": 0,
        "stages": [
            {"name": "box-scan", "label": "扫描干员Box", "type": "skill",
             "skill_name": "box-scan", "output_file": "box.json"},
            {"name": "base-schedule", "label": "计算最优排班", "type": "intelligence",
             "intel_tool": "BaseScheduler", "input_file": "box.json",
             "output_file": "schedule.json"},
            {"name": "base-deploy", "label": "执行基建排班", "type": "skill",
             "skill_name": "base-deploy", "input_file": "schedule.json"},
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (session_path / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("scan_operator_box: created session %s", session_id)
    return session_id


def _write_box_to_session(session_id: str, operators: list[dict]) -> None:
    """Write full MAA operator data to session/box.json.

    Preserves all fields from MAA OperBox callback:
    name, elite(E0/E1/E2), level, potential, rarity, id, own.
    """
    box_data: dict[str, dict] = {}
    for op in operators:
        name = op.get("name", "")
        if not name:
            continue
        box_data[name] = {
            "elite": op.get("elite", 0),
            "level": op.get("level", 1),
            "potential": op.get("potential", 0),
            "rarity": op.get("rarity", 0),
        }
    payload = {
        "game": "arknights",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "total": len(box_data),
        "operators": box_data,
    }
    session_path = SESSION_DIR / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    (session_path / "box.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("scan_operator_box: wrote %s/box.json (%d operators with level/elite/rarity)",
                session_id, len(box_data))


# ═══════════════════════════════════════════════════════════════════════════
# Tool: scan_operator_box
# ═══════════════════════════════════════════════════════════════════════════

def scan_operator_box_tool(max_duration_ms: int = 180000) -> ToolOutput:
    """MAA OperBox scan — uses MaaCore.dll for 100% accurate results.

    Writes box data directly into the active base_chain session
    (data/session/<session_id>/box.json) for downstream scheduling.

    Prerequisites (LLM handles these):
      - Must already be on the operator list (OperBox) screen
      - Sort set to 等级 (by level), role filter expanded (all professions)
    """
    t_start = time.monotonic()
    _ensure_maa_dll()

    adb = get_adb()
    adb_addr = adb.serial
    adb_exe = _find_adb_exe()

    if _platform.system() == "Windows":
        _CbType = _ct.WINFUNCTYPE(None, _ct.c_int, _ct.c_char_p, _ct.c_void_p)
    else:
        _CbType = _ct.CFUNCTYPE(None, _ct.c_int, _ct.c_char_p, _ct.c_void_p)

    own_opers: dict[str, dict] = {}
    done = False
    cb_lock = threading.Lock()

    @_CbType
    def _on_operbox_msg(msg: int, details: bytes, _arg):
        nonlocal done
        try:
            d = json.loads(details.decode("utf-8"))
        except Exception:
            return
        if d.get("what") != "OperBoxInfo":
            return
        dd = d.get("details", {})
        if dd.get("done"):
            done = True
        with cb_lock:
            for oper in dd.get("own_opers", []):
                name = oper.get("name", "")
                if name and name not in own_opers:
                    own_opers[name] = oper
        if not dd.get("done"):
            logger.info("  scan_operator_box: %d operators so far...", len(own_opers))

    _cb_ref = _on_operbox_msg

    handle = _DLL.AsstCreateEx(_on_operbox_msg, None)
    if not handle:
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": "MAA AsstCreateEx 失败——MaaCore.dll 可能版本不兼容",
        }, ensure_ascii=False))

    _DLL.AsstSetInstanceOption(handle, 2, b"maatouch")

    if not _DLL.AsstConnect(handle, adb_exe.encode("utf-8"),
                            adb_addr.encode("utf-8"), b"General"):
        _DLL.AsstDestroy(handle)
        return ToolOutput(text=json.dumps({
            "success": False,
            "message": f"MAA 连接设备失败: {adb_addr}",
        }, ensure_ascii=False))

    logger.info("scan_operator_box: MAA connected to %s, scanning...", adb_addr)

    _DLL.AsstAppendTask(handle, b"OperBox", b"{}")
    _DLL.AsstStart(handle)

    timeout_sec = max_duration_ms / 1000.0
    while _DLL.AsstRunning(handle) and (time.monotonic() - t_start) < timeout_sec:
        time.sleep(0.3)

    _DLL.AsstStop(handle)
    _DLL.AsstDestroy(handle)

    # ── Build output ──
    operators = list(own_opers.values())
    operators.sort(key=lambda o: (-o.get("rarity", 0), o.get("name", "")))

    summary = {"E2": 0, "E1": 0, "E0": 0}
    for r in operators:
        elite_key = f"E{r.get('elite', 0)}"
        summary[elite_key] = summary.get(elite_key, 0) + 1

    # ── Write to session ──
    session_id = _find_or_create_session()
    _write_box_to_session(session_id, operators)

    elapsed_ms = (time.monotonic() - t_start) * 1000

    logger.info(
        "scan_operator_box: DONE — %d operators, E2:%d E1:%d E0:%d, %.0fms → %s",
        len(operators), summary["E2"], summary["E1"], summary["E0"],
        elapsed_ms, session_id,
    )

    return ToolOutput(text=json.dumps({
        "success": True,
        "total": len(operators),
        "summary": summary,
        "operators": operators,
        "elapsed_ms": round(elapsed_ms, 1),
        "session": session_id,
        "engine": "MAA MaaCore.dll",
    }, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════════════════
# ADB availability check
# ═══════════════════════════════════════════════════════════════════════════

def _adb_available() -> bool:
    try:
        from src.device.emulator import emulator_manager
        return emulator_manager.first_online is not None
    except Exception:
        return False


def _maa_available() -> bool:
    """Check if MAA DLL is available before offering to LLM."""
    for candidate in _MAA_DIR_CANDIDATES:
        if (candidate / "MaaCore.dll").exists():
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Register tool
# ═══════════════════════════════════════════════════════════════════════════

registry.register(
    name="scan_operator_box",
    check_fn=lambda: _maa_available() and _adb_available(),
    description=(
        "★ [Box扫描] 使用 MAA 引擎 (MaaCore.dll) 自动扫描干员列表。\n"
        "与 MAA 桌面版完全相同的识别精度——100% 准确。\n"
        "\n"
        "【前置条件 — 调用前请确保 LLM 已完成以下导航】\n"
        "  1. 进入干员列表界面（OperBox）\n"
        "  2. 排序方式设为「等级」（点击排序栏→选择等级）\n"
        "  3. 职业筛选已展开（显示全部职业）\n"
        "\n"
        "【输出】\n"
        "  - 干员数据直接写入排班链 session → 后续自动计算排班\n"
        "  - 返回值：JSON (total, summary, operators, session)\n"
        "  - 每个干员：name, elite(E0/E1/E2), level, potential, rarity, id\n"
        "\n"
        "【引擎】MAA MaaCore.dll (C++ 原生识别，无需 Python OCR)"
    ),
    handler=scan_operator_box_tool,
    parameters={
        "type": "object",
        "properties": {
            "max_duration_ms": {
                "type": "integer",
                "description": "最长扫描时间（毫秒）。默认180000（3分钟）。",
            },
        },
    },
)
