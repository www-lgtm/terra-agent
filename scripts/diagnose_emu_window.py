#!/usr/bin/env python3
"""Emulator window detection diagnostic.  纯只读, 不安装钩子, 不修改系统状态.

Usage:
    python scripts/diagnose_emu_window.py
    python scripts/diagnose_emu_window.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _check_pywin32() -> bool:
    try:
        import win32gui  # noqa: F401
        return True
    except ImportError:
        return False


def _check_dpi_awareness() -> str:
    """Report current DPI awareness mode (read-only, does NOT change it)."""
    import ctypes

    try:
        shcore = ctypes.windll.shcore
        awareness = ctypes.c_int()
        shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
        modes = {
            0: "UNAWARE (虚拟化坐标 — 点击可能丢失)",
            1: "SYSTEM_DPI_AWARE",
            2: "PER_MONITOR_DPI_AWARE (推荐)",
        }
        return modes.get(awareness.value, f"Unknown ({awareness.value})")
    except Exception:
        pass

    try:
        user32 = ctypes.windll.user32
        if user32.IsProcessDPIAware():
            return "SYSTEM_DPI_AWARE (legacy)"
        return "UNAWARE (legacy detection)"
    except Exception:
        return "无法检测"


def _list_emulator_candidates() -> list[dict]:
    try:
        import win32gui
    except ImportError:
        return []

    patterns = [
        "mumuplayer", "mumu模拟器", "mumu",
        "雷电", "ldplayer", "ldconsole",
        "bluestacks",
        "nox", "noxplayer", "memu",
        "gameloop", "腾讯手游助手",
        "模拟器", "逍遥", "夜神",
    ]
    candidates: list[dict] = []

    def _cb(hwnd: int, _ctx) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        cls = win32gui.GetClassName(hwnd)
        try:
            rect = win32gui.GetWindowRect(hwnd)
            w, h = rect[2] - rect[0], rect[3] - rect[1]
        except Exception:
            return True

        matched_pattern = ""
        title_lower = title.lower()
        cls_lower = cls.lower()
        for p in patterns:
            if p in title_lower or p in cls_lower:
                matched_pattern = p
                break
        if not matched_pattern and "QWindowIcon" in cls:
            matched_pattern = "Qt (MuMu 12 candidate)"

        if w >= 200 and h >= 200:
            candidates.append({
                "hwnd": hwnd,
                "title": title,
                "class": cls,
                "rect": rect,
                "size": f"{w}x{h}",
                "area": w * h,
                "matched": matched_pattern,
            })
        return True

    win32gui.EnumWindows(_cb, None)
    # Sort: matched first, then by area descending
    candidates.sort(key=lambda c: (not c["matched"], -c["area"]))
    return candidates


def _check_coord_mapping_sanity(
    best: dict, dev_w: int, dev_h: int
) -> list[str]:
    """Check if the coordinate mapping from window rect → device pixels
    looks sane, WITHOUT installing any hooks.

    Returns a list of diagnostic messages.
    """
    import win32gui

    msgs: list[str] = []
    try:
        rect = win32gui.GetWindowRect(best["hwnd"])
        left, top, right, bottom = rect
        win_w = right - left
        win_h = bottom - top
    except Exception as e:
        return [f"无法获取窗口位置: {e}"]

    # Check 1: window size sanity
    if win_w <= 0 or win_h <= 0:
        msgs.append("❌ 窗口尺寸为 0 — 模拟器可能被最小化")
        return msgs
    msgs.append(f"模拟器窗口: ({left},{top}) → ({right},{bottom}), 尺寸 {win_w}x{win_h}")

    # Check 2: aspect ratio — most emulators are portrait or near-square
    aspect = win_h / max(win_w, 1)
    if 0.5 < aspect < 3.0:
        msgs.append(f"✅ 窗口宽高比 {aspect:.1f} 正常 (模拟器通常 0.8~2.5)")
    else:
        msgs.append(f"⚠️ 窗口宽高比 {aspect:.1f} 异常 — 可能不是模拟器窗口")

    # Check 3: coordinate mapping range
    scale_x = dev_w / max(win_w, 1)
    scale_y = dev_h / max(win_h, 1)
    msgs.append(f"坐标缩放: X轴 {scale_x:.2f}  Y轴 {scale_y:.2f}")

    # Check 4: DPI scaling mismatch detection
    if scale_x < 0.3 or scale_y < 0.3:
        msgs.append("⚠️ 缩放因子过小 — DPI 虚拟化可能在起作用")

    # Check 5: corner coordinate test (synthetic, no hook)
    center_dx = left + win_w // 2
    center_dy = top + win_h // 2
    center_dev_x = int((win_w // 2) * scale_x)
    center_dev_y = int((win_h // 2) * scale_y)
    msgs.append(
        f"坐标验证: 桌面中心点 ({center_dx},{center_dy}) "
        f"→ 映射到设备 ({center_dev_x},{center_dev_y})"
        f"  [预期接近 ({dev_w // 2},{dev_h // 2})]"
    )

    if abs(center_dev_x - dev_w // 2) > dev_w * 0.3:
        msgs.append(
            "⚠️ X 轴映射偏差过大 → DPI 虚拟化坐标不匹配\n"
            f"   生产代码会在启动时调用 SetProcessDpiAwareness(2) 自动修复此问题。"
        )
    if abs(center_dev_y - dev_h // 2) > dev_h * 0.3:
        msgs.append(
            "⚠️ Y 轴映射偏差过大 → DPI 虚拟化坐标不匹配\n"
            f"   生产代码会在启动时调用 SetProcessDpiAwareness(2) 自动修复此问题。"
        )

    if all("⚠️" not in m for m in msgs[1:]):
        msgs.append("✅ 坐标映射 sanity check 通过")

    return msgs


def _test_adb_resolution() -> tuple[int, int] | None:
    try:
        from src.device.adb import get_adb
        adb = get_adb()
        return adb.get_screen_size()
    except Exception as e:
        print(f"  ADB 分辨率检测失败: {e}")
        return None


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="诊断模拟器窗口检测（纯只读，不安装钩子）",
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示所有候选窗口")
    args = parser.parse_args()

    print("=" * 60)
    print("🔍 模拟器窗口诊断工具（只读模式）")
    print("=" * 60)

    # 1. pywin32
    print("\n1. pywin32 状态:")
    if _check_pywin32():
        print("  ✅ pywin32 已安装")
    else:
        print("  ❌ pywin32 未安装 — 运行: pip install pywin32")
        return

    # 2. DPI awareness
    print("\n2. DPI 感知状态:")
    dpi_status = _check_dpi_awareness()
    is_aware = "UNAWARE" not in dpi_status
    prefix = "✅" if is_aware else "⚠️"
    print(f"  {prefix} {dpi_status}")
    if not is_aware:
        print("  → 生产代码启动时会自动调用 SetProcessDpiAwareness(2) 修复")
        print("  → 本诊断脚本不会修改系统状态")

    # 3. Emulator candidates
    print("\n3. 候选模拟器窗口:")
    candidates = _list_emulator_candidates()
    if not candidates:
        print("  ❌ 未找到任何候选窗口!")
        print("  请确认模拟器已启动且窗口可见。")
        print("  如果模拟器确实在运行, 可能是窗口标题/类名不在已知列表中。")
        print("  运行 --verbose 查看所有可见窗口。")
        return

    shown = 0
    for c in candidates:
        matched = c["matched"]
        if matched:
            shown += 1
            print(f"  {'✅' if shown <= 3 else '  '} "
                  f"'{c['title']}' | class={c['class']} | size={c['size']} "
                  f"| matched='{matched}' | hwnd={c['hwnd']}")
        elif args.verbose or shown < 5:
            shown += 1
            marker = "→" if shown <= 5 else " "
            print(f"  {marker} '{c['title']}' | class={c['class']} | size={c['size']}")

    # 4. Best guess
    best = next((c for c in candidates if c["matched"]), None)
    if not best and candidates:
        best = candidates[0]

    if best:
        print(f"\n4. 最佳候选:")
        print(f"   标题: '{best['title']}'")
        print(f"   类名: {best['class']}")
        print(f"   句柄: {best['hwnd']}")
        print(f"   匹配规则: {best['matched'] or '(无 — 使用最大窗口启发式)'}")

    # 5. ADB resolution
    print(f"\n5. ADB 设备分辨率:")
    res = _test_adb_resolution()
    if res:
        dev_w, dev_h = res
        print(f"  ✅ {dev_w}x{dev_h}")
    else:
        dev_w, dev_h = 1600, 900

    # 6. Coordinate sanity check (synthetic, no hooks)
    if best:
        print(f"\n6. 坐标映射 sanity check:")
        msgs = _check_coord_mapping_sanity(best, dev_w, dev_h)
        for m in msgs:
            print(f"  {m}")

    # Summary
    print(f"\n{'=' * 60}")
    if best and best["matched"] and not any("⚠️" in m for m in msgs) and is_aware:
        print("✅ 所有检查通过 — 可以开始录制")
    else:
        print("⚠️ 存在问题，录制前请先解决")
        if not best or not best["matched"]:
            print("  → 未检测到匹配的模拟器窗口")
        if not is_aware:
            print("  → DPI 感知未开启")
    print("=" * 60)


if __name__ == "__main__":
    main()
