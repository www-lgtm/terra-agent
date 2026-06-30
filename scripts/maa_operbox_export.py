"""使用 MAA (MaaAssistantArknights) 引擎导出干员 Box 完整数据。

MAA 的识别精度远高于模板匹配方案，能同时给出：
  干员名、精英化阶段(E0/E1/E2)、等级、潜能、稀有度

前提条件:
  1. 下载 MAA 发行版（约 150MB zip），解压到任意目录
     下载地址: https://maa.plus （或 GitHub Releases）
  2. 模拟器已启动，进入明日方舟

用法:
  python scripts/maa_operbox_export.py <MAA解压目录>

  # 示例（自动发现模拟器）:
  python scripts/maa_operbox_export.py D:/MAA-v6.11.1-win-x64

  # 指定 ADB 地址:
  python scripts/maa_operbox_export.py D:/MAA-v6.11.1-win-x64 --adb 127.0.0.1:5555

  # 指定输出文件:
  python scripts/maa_operbox_export.py D:/MAA-v6.11.1-win-x64 -o my_box.json

输出 JSON 示例:
  {
    "total": 55,
    "summary": {"E2": 12, "E1": 20, "E0": 23},
    "operators": [
      {"name": "银灰", "elite": 2, "level": 90, "potential": 3, "rarity": 6, "own": true},
      {"name": "史都华德", "elite": 0, "level": 30, "potential": 6, "rarity": 3, "own": true}
    ]
  }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAA_PYTHON_DIR = PROJECT_ROOT.parent / "MaaAssistantArknights" / "src" / "Python"


def _find_adb_address() -> str | None:
    """自动发现模拟器 ADB 地址。"""
    try:
        from src.device.emulator import emulator_manager
        return emulator_manager.first_online
    except Exception:
        return None


def _find_adb_exe() -> str | None:
    """找到 ADB 可执行文件路径。"""
    # 1. 环境变量 ADB_PATH
    import os
    adb = os.environ.get("ADB_PATH")
    if adb and Path(adb).exists():
        return adb

    # 2. 从 config 读
    try:
        from config.settings import config
        if config.device.adb_path and Path(config.device.adb_path).exists():
            return config.device.adb_path
    except Exception:
        pass

    # 3. 常见位置搜索
    candidates = [
        Path("D:/platform-tools/adb.exe"),
        Path.home() / "AppData" / "Local" / "Android" / "Sdk" / "platform-tools" / "adb.exe",
        Path("C:/platform-tools/adb.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # 4. PATH 中搜索
    import shutil
    adb = shutil.which("adb")
    if adb:
        return adb

    return None


def run_operbox_scan(maa_root: str | Path, adb_addr: str,
                     timeout_sec: int = 180) -> list[dict]:
    """执行 MAA OperBox 扫描，返回干员列表。

    Args:
        maa_root: MAA 发行版根目录（含 MaaCore.dll 和 resource/ 的目录）
        adb_addr: ADB 地址，如 "127.0.0.1:5555"
        timeout_sec: 超时秒数

    Returns:
        [{"name": "银灰", "elite": 2, "level": 90, "potential": 3,
          "rarity": 6, "own": true, "id": "char_..."}, ...]
    """
    maa_root = Path(maa_root)

    # 验证 MAA 目录
    if not (maa_root / "MaaCore.dll").exists():
        raise FileNotFoundError(
            f"在 {maa_root} 中未找到 MaaCore.dll。\n"
            f"请确认该目录是 MAA 发行版的根目录（包含 MaaCore.dll 和 resource/ 文件夹）。\n"
            f"下载地址: https://maa.plus"
        )

    # ── 加载 MAA DLL + 修复 Windows 回调 ──────────────────────────────
    # MAA 的 Python 绑定有 name-mangling 和调用约定问题。
    # 这里完全绕开 bindings，直接操作 DLL。
    import ctypes as _ct
    import platform as _platform

    _dll_path = str(maa_root / "MaaCore.dll")
    _dll = _ct.WinDLL(_dll_path) if _platform.system() == "Windows" else _ct.CDLL(_dll_path)

    # 声明函数签名
    _dll.AsstLoadResource.argtypes = (_ct.c_char_p,)
    _dll.AsstLoadResource.restype = _ct.c_bool
    _dll.AsstSetUserDir.argtypes = (_ct.c_char_p,)
    _dll.AsstSetUserDir.restype = _ct.c_bool

    # 加载资源
    _dll.AsstSetUserDir(str(maa_root).encode("utf-8"))
    _dll.AsstLoadResource(str(maa_root).encode("utf-8"))
    print(f"✓ 已加载 MAA: {maa_root}")

    # 回调类型: Windows 用 WINFUNCTYPE(__stdcall), 其他用 CFUNCTYPE(__cdecl)
    if _platform.system() == "Windows":
        _CbType = _ct.WINFUNCTYPE(None, _ct.c_int, _ct.c_char_p, _ct.c_void_p)
    else:
        _CbType = _ct.CFUNCTYPE(None, _ct.c_int, _ct.c_char_p, _ct.c_void_p)

    _dll.AsstCreateEx.argtypes = (_CbType, _ct.c_void_p)
    _dll.AsstCreateEx.restype = _ct.c_void_p
    _dll.AsstCreate.argtypes = ()
    _dll.AsstCreate.restype = _ct.c_void_p
    _dll.AsstDestroy.argtypes = (_ct.c_void_p,)
    _dll.AsstConnect.argtypes = (_ct.c_void_p, _ct.c_char_p, _ct.c_char_p, _ct.c_char_p)
    _dll.AsstConnect.restype = _ct.c_bool
    _dll.AsstAppendTask.argtypes = (_ct.c_void_p, _ct.c_char_p, _ct.c_char_p)
    _dll.AsstAppendTask.restype = _ct.c_int
    _dll.AsstStart.argtypes = (_ct.c_void_p,)
    _dll.AsstStart.restype = _ct.c_bool
    _dll.AsstStop.argtypes = (_ct.c_void_p,)
    _dll.AsstStop.restype = _ct.c_bool
    _dll.AsstRunning.argtypes = (_ct.c_void_p,)
    _dll.AsstRunning.restype = _ct.c_bool
    _dll.AsstSetInstanceOption.argtypes = (_ct.c_void_p, _ct.c_int, _ct.c_char_p)
    _dll.AsstSetInstanceOption.restype = _ct.c_bool

    # 结果收集
    own_opers: dict[str, dict] = {}
    done = False

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

        for oper in dd.get("own_opers", []):
            name = oper.get("name", "")
            if name and name not in own_opers:
                own_opers[name] = oper

        if not dd.get("done"):
            print(f"\r  已识别 {len(own_opers)} 名干员...", end="", flush=True)

    # 创建实例 + 连接
    _cb_ref = _on_operbox_msg  # 阻止 GC
    _handle = _dll.AsstCreateEx(_on_operbox_msg, None)
    if not _handle:
        raise RuntimeError("AsstCreateEx 失败")

    _dll.AsstSetInstanceOption(_handle, 2, b"maatouch")  # touch_type

    # 找到 ADB 可执行文件
    _adb_exe = _find_adb_exe()
    if not _adb_exe:
        _dll.AsstDestroy(_handle)
        raise RuntimeError("未找到 adb.exe，请安装 ADB 或设置 ADB_PATH 环境变量")

    print(f"ADB: {_adb_exe}")
    if not _dll.AsstConnect(_handle, _adb_exe.encode("utf-8"),
                            adb_addr.encode("utf-8"), b"General"):
        _dll.AsstDestroy(_handle)
        raise RuntimeError(f"连接失败: {adb_addr}")

    print(f"✓ 已连接: {adb_addr}")
    print(f"开始扫描（MAA 自动截图→识别→翻页）...")

    # 执行
    _dll.AsstAppendTask(_handle, b"OperBox", b"{}")
    _dll.AsstStart(_handle)

    t_start = time.time()
    while _dll.AsstRunning(_handle) and (time.time() - t_start) < timeout_sec:
        time.sleep(0.3)

    _dll.AsstStop(_handle)
    _dll.AsstDestroy(_handle)
    print()  # 换行

    result = list(own_opers.values())
    # 按稀有度降序 + 名称排序
    result.sort(key=lambda o: (-o.get("rarity", 0), o.get("name", "")))

    if not done:
        print("⚠ 扫描可能不完全（超时），已保存已识别的部分")
    else:
        print(f"✓ 扫描完成")

    return result


def build_summary(operators: list[dict]) -> dict:
    """统计精英化分布。"""
    summary = {"E2": 0, "E1": 0, "E0": 0}
    for oper in operators:
        elite = oper.get("elite", 0)
        key = f"E{elite}"
        summary[key] = summary.get(key, 0) + 1
    return summary


def print_table(operators: list[dict], top_n: int = 40):
    """打印格式化表格。"""
    summary = build_summary(operators)
    print()
    print("=" * 65)
    print(f"  总计: {len(operators)} 名已拥有干员")
    print(f"  精二(E2): {summary.get('E2', 0)}  "
          f"精一(E1): {summary.get('E1', 0)}  "
          f"未精英(E0): {summary.get('E0', 0)}")
    print("=" * 65)
    print(f"  {'干员':<6} {'精英':>4} {'等级':>4} {'潜能':>4}  {'稀有度'}")
    print(f"  {'─' * 30}")
    for oper in operators[:top_n]:
        name = oper.get("name", "?")
        elite = oper.get("elite", 0)
        level = oper.get("level", 0)
        pot = oper.get("potential", 0)
        stars = "★" * oper.get("rarity", 0)
        print(f"  {name:<6}  E{elite}  Lv{level:<4} 潜{pot}  {stars}")
    if len(operators) > top_n:
        print(f"  ... 还有 {len(operators) - top_n} 名（详见输出文件）")


def main():
    parser = argparse.ArgumentParser(
        description="MAA OperBox 干员数据导出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/maa_operbox_export.py D:/MAA-v6.11.1-win-x64
  python scripts/maa_operbox_export.py D:/MAA-v6.11.1-win-x64 --adb 127.0.0.1:5555 -o box.json
        """,
    )
    parser.add_argument(
        "maa_dir",
        help="MAA 发行版根目录（包含 MaaCore.dll 和 resource/ 的目录）",
    )
    parser.add_argument(
        "--adb", "-a",
        help="ADB 地址（默认自动检测）",
    )
    parser.add_argument(
        "--output", "-o",
        help="输出 JSON 文件路径（默认 data/operator_box_maa.json）",
    )
    parser.add_argument(
        "--timeout", "-t", type=int, default=180,
        help="扫描超时秒数（默认 180）",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="只输出 JSON，不打印表格",
    )

    args = parser.parse_args()

    # ADB 地址
    adb_addr = args.adb or _find_adb_address()
    if not adb_addr:
        print("错误: 未找到模拟器。请指定 --adb 参数，例如 --adb 127.0.0.1:5555")
        sys.exit(1)

    # 执行扫描
    operators = run_operbox_scan(args.maa_dir, adb_addr, args.timeout)

    # 构建结果
    result = {
        "total": len(operators),
        "summary": build_summary(operators),
        "operators": operators,
    }

    # 保存
    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "data" / "operator_box.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果已保存 → {output_path}")

    # 打印
    if not args.quiet:
        print_table(operators)

    return result


if __name__ == "__main__":
    main()
