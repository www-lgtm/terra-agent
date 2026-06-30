"""Emulator lifecycle tools — restart, launch, app starter.

The emulator lifecycle engine (src/device/emulator.py) already has full
restart / launch / health-monitor capability.  These tools expose it so the
LLM can handle "重启模拟器" without needing the user to walk to the PC.
"""

from __future__ import annotations

import json
import logging
import time as _time

from src.tools.registry import ToolOutput, registry

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────

_ARKNIGHTS_PACKAGE = "com.hypergryph.arknights"


def _start_game_via_adb(serial: str) -> bool:
    """Launch app via `adb shell monkey`, falling back to `am start`.

    MuMu multi-drive clones may reject `monkey` but accept `am start`.
    """
    try:
        from src.device.adb import get_adb
        adb = get_adb(serial)
        for cmd in [
            ("monkey -p {pkg} -c android.intent.category.LAUNCHER 1",
             f"monkey -p {_ARKNIGHTS_PACKAGE} -c android.intent.category.LAUNCHER 1"),
            ("am start {pkg}",
             f"am start -a android.intent.action.MAIN -c android.intent.category.LAUNCHER {_ARKNIGHTS_PACKAGE}"),
        ]:
            try:
                out = adb.shell(*cmd[1].split())
                logger.info("ADB launched Arknights on %s via %s: %s", serial, cmd[0], out.strip()[:80])
                return True
            except RuntimeError:
                continue
        logger.warning("ADB launch failed for %s: both monkey and am start failed", serial)
        return False
    except Exception as e:
        logger.warning("ADB launch failed for %s: %s", serial, e)
        return False


def _reinit_adb(serial: str) -> None:
    try:
        from src.device.adb import init_adb, remove_device
        remove_device(serial)
        init_adb(serial)
    except Exception as e:
        logger.debug("ADB re-init: %s", e)


# ── Tools ─────────────────────────────────────────────────────────────────


def adb_input_text(text: str) -> ToolOutput:
    """Type text into the currently focused input field via ADB.

    Use this to fill in phone numbers, usernames, passwords, or any text
    input field. First tap the input field to focus it, then call this.

    Handles spaces and special characters automatically.

    Args:
        text: The text to type. Can include numbers, letters, spaces.
    """
    from src.device.adb import get_adb
    try:
        adb = get_adb()
        # Escape special shell characters: replace space with %s, escape quotes
        escaped = text.replace(" ", "%s").replace('"', '\\"')
        adb.shell("input", "text", escaped)
        return ToolOutput(text=json.dumps({
            "success": True,
            "message": f"已输入文字: {text}",
        }, ensure_ascii=False))
    except RuntimeError as e:
        return ToolOutput(text=json.dumps({
            "success": False,
            "error": f"输入失败: {e}",
        }, ensure_ascii=False))


def restart_emulator(adb_addr: str = "127.0.0.1:16384") -> ToolOutput:
    """Restart the emulator. Use when the game is stuck, frozen, or fails to load.

    After restart, the game is automatically re-launched. No need to navigate
    the Android desktop — just wait for the game loading screen and then
    continue your task.

    Args:
        adb_addr: ADB device address (default 127.0.0.1:16384).
    """
    from src.device.emulator import emulator_manager

    logger.warning("LLM-triggered emulator restart: %s", adb_addr)

    result = emulator_manager.restart_emulator(adb_addr)

    if result == "ok":
        _time.sleep(2.0)
        _reinit_adb(adb_addr)

        # Auto-launch the game so the agent doesn't land on the Android desktop
        game_ok = _start_game_via_adb(adb_addr)
        game_msg = (
            "已自动打开明日方舟，约30秒后进入登录界面。"
            if game_ok else "未能自动打开游戏，请手动点击明日方舟图标。"
        )

        return ToolOutput(text=json.dumps({
            "success": True,
            "message": f"模拟器 {adb_addr} 重启完成。{game_msg}",
        }, ensure_ascii=False))
    elif result == "already_running":
        return ToolOutput(text=json.dumps({
            "success": False,
            "error": "模拟器正在重启中，请等待上一次重启完成。",
        }, ensure_ascii=False))
    else:
        return ToolOutput(text=json.dumps({
            "success": False,
            "error": f"模拟器重启失败: {result}。请手动重启。",
        }, ensure_ascii=False))


def launch_emulator() -> ToolOutput:
    """Cold-start the emulator when it is completely closed (not just stuck).

    Also launches Arknights automatically. Returns the new device serial.
    """
    from src.device.emulator import emulator_manager

    logger.info("LLM-triggered emulator cold-launch")

    serial = emulator_manager.launch_emulator()

    if serial:
        _reinit_adb(serial)
        _start_game_via_adb(serial)

        return ToolOutput(text=json.dumps({
            "success": True,
            "serial": serial,
            "message": f"模拟器已启动（{serial}），明日方舟已自动打开。"
                       "等待30-60秒游戏进入主界面后再继续操作。",
        }, ensure_ascii=False))
    else:
        return ToolOutput(text=json.dumps({
            "success": False,
            "error": "模拟器启动失败。检查 EMULATOR_TYPE 和 EMULATOR_CONSOLE 配置。",
        }, ensure_ascii=False))


def adb_launch_app(package: str = _ARKNIGHTS_PACKAGE) -> ToolOutput:
    """Open an app via ADB. Use this when the game is not running but the
    emulator is online — for example after a restart dropped you on the
    Android desktop.

    Args:
        package: Android package name. Defaults to Arknights.
    """
    from src.device.adb import get_adb
    try:
        adb = get_adb()
    except RuntimeError as e:
        return ToolOutput(text=json.dumps({
            "success": False,
            "error": f"无可用ADB设备: {e}",
        }, ensure_ascii=False))

    # Try monkey first (standard), then am start (more compatible with
    # emulator clones / multi-drive instances where monkey may fail).
    for method_name, cmd_args in [
        ("monkey", ("monkey", "-p", package,
                    "-c", "android.intent.category.LAUNCHER", "1")),
        ("am_start", ("shell", "am", "start", "-a",
                      "android.intent.action.MAIN",
                      "-c", "android.intent.category.LAUNCHER",
                      package)),
    ]:
        try:
            out = adb.shell(*cmd_args)
            return ToolOutput(text=json.dumps({
                "success": True,
                "message": f"已启动 {package}（{method_name}）。等待游戏加载。",
                "output": out.strip()[:120],
            }, ensure_ascii=False))
        except RuntimeError:
            continue

    return ToolOutput(text=json.dumps({
        "success": False,
        "error": f"无法启动 {package}：monkey 和 am start 均失败。请尝试在桌面找到应用图标手动点击。",
    }, ensure_ascii=False))


# ── Register ───────────────────────────────────────────────────────────────

registry.register(
    name="restart_emulator",
    game="arknights",
    description=(
        "【重启模拟器】当游戏卡死、冻结、反复加载失败时使用。\n"
        "杀掉模拟器进程→断开ADB→重新启动→自动重连ADB→自动打开明日方舟。\n"
        "整个过程约60-90秒。调用后只需等待游戏进入主界面即可继续任务，\n"
        "无需手动找游戏图标——已经在执行时自动打开了。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "adb_addr": {
                "type": "string",
                "description": "ADB设备地址，默认 127.0.0.1:16384",
            },
        },
    },
    handler=restart_emulator,
)

registry.register(
    name="launch_emulator",
    game="arknights",
    description=(
        "【冷启动模拟器】当模拟器进程完全关闭时使用。\n"
        "打开模拟器→等待ADB连接→自动打开明日方舟。\n"
        "与 restart_emulator 的区别：launch 是冷启动（模拟器没开），"
        "restart 是热重启（模拟器卡住了但进程还在）。"
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    handler=launch_emulator,
)

registry.register(
    name="adb_launch_app",
    game=None,  # universal — all game agents may need to launch their app
    description=(
        "【启动APP】通过ADB命令打开一个应用。\n"
        "当你发现自己在安卓桌面而不是游戏里时，直接用这个工具拉起游戏，不用在桌面OCR找图标。\n"
        "常用包名：明日方舟 com.hypergryph.arknights，1999 com.bluepoch.bluepoch.reversenineninetynine"
    ),
    parameters={
        "type": "object",
        "properties": {
            "package": {
                "type": "string",
                "description": "应用包名。如 com.hypergryph.arknights",
            },
        },
    },
    handler=adb_launch_app,
)

registry.register(
    name="adb_input_text",
    game=None,  # universal — all games need text input
    description=(
        "【输入文字】在输入框中打字。\n"
        "先点击输入框使其获得焦点，然后调用此工具输入内容。\n"
        "适用场景：输入手机号、密码、验证码、搜索关键词等。\n"
        "支持中文、英文、数字和空格。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "要输入的文字内容",
            },
        },
        "required": ["text"],
    },
    handler=adb_input_text,
)
