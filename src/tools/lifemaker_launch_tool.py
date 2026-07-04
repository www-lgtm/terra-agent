"""Life Makeover full launch — biubiu → ads → accelerate → game auto-launches."""

from __future__ import annotations

import json
import logging
import re
import time

from src.device.adb import get_adb
from src.tools.registry import ToolOutput, registry
from src.vision.ocr import ocr_engine

logger = logging.getLogger(__name__)

_BIUBIU_PKG = "com.njh.biubiu"
_AD_TIMEOUT = 50.0
_AD_POLL = 1.0
_MAX_ADS = 3
_MID_AD_KW = ("浏览5秒", "提前领奖",)
_AD_REWARD_KW = ("已经获得",)
_AD_START_KW = ("获取时长", "继续领时长", "继续获取", "再看")


def _ok(data: dict) -> ToolOutput:
    data.setdefault("success", True)
    return ToolOutput(text=json.dumps(data, ensure_ascii=False))


def _error(msg: str) -> ToolOutput:
    return ToolOutput(text=json.dumps({"success": False, "error": msg}, ensure_ascii=False))


def _dets(adb):
    return ocr_engine.read_text(adb.get_screenshot_image())


def _has_any(dets, keywords: tuple[str, ...]) -> bool:
    return any(any(kw in d["text"] for d in dets) for kw in keywords)


def _find_tap(adb, dets, target: str) -> bool:
    for d in sorted(dets, key=lambda x: (x["center"][1], x["center"][0])):
        if target in d["text"]:
            adb.shell("input", "tap", str(d["center"][0]), str(d["center"][1]))
            return True
    return False


def lifemaker_launch() -> ToolOutput:
    adb = get_adb()

    # 1. Launch biubiu and wait for it to fully load
    adb.shell("monkey", "-p", _BIUBIU_PKG, "-c", "android.intent.category.LAUNCHER", "1")
    logger.info("lifemaker_launch: launched biubiu, waiting for it to load")

    for _ in range(60):  # up to 60s for biubiu to load
        time.sleep(1.0)
        dets = _dets(adb)
        # Dismiss splash ad
        if _has_any(dets, ("跳过",)):
            _find_tap(adb, dets, "跳过")
            time.sleep(0.5)
        # Check if biubiu home is ready
        if _has_any(dets, ("加速",)) and len([d for d in dets if len(d["text"].strip()) >= 2]) >= 8:
            logger.info("lifemaker_launch: biubiu loaded (%d texts)", len(dets))
            break
    else:
        logger.warning("lifemaker_launch: biubiu may still be loading after 60s")

    # 2. Enter ad page
    for attempt in range(2):
        dets = _dets(adb)
        for kw in ("立即获取", "今日福利已就绪", "今日福利"):
            if _find_tap(adb, dets, kw):
                logger.info("lifemaker_launch: tapped '%s'", kw)
                time.sleep(1.5)
                break
        if _has_any(_dets(adb), _AD_START_KW + ("看广告领时长",)):
            break
        time.sleep(0.5)

    # 3. Check time / watch ads
    dets = _dets(adb)
    time_text = " ".join(d["text"] for d in dets)
    dm = re.search(r"(\d+)\s*天", time_text)
    hm = re.search(r"(\d+)\s*(?:小时|小)", time_text)
    total_h = (int(dm.group(1)) if dm else 0) * 24 + (int(hm.group(1)) if hm else 0)

    if total_h >= 10:
        logger.info("lifemaker_launch: already %dh — skip ads", total_h)
    else:
        for rnd in range(_MAX_ADS):
            dets = _dets(adb)
            start_btn = None
            for kw in _AD_START_KW:
                if any(kw in d["text"] for d in dets):
                    start_btn = kw; break
            if not start_btn: break

            logger.info("lifemaker_launch: ad %d/%d (%s)", rnd + 1, _MAX_ADS, start_btn)
            _find_tap(adb, dets, start_btn)
            time.sleep(1.5)

            deadline = time.monotonic() + _AD_TIMEOUT
            while time.monotonic() < deadline:
                time.sleep(_AD_POLL)
                dets = _dets(adb)
                if _has_any(dets, _MID_AD_KW):
                    for kw in ("残忍离开", "坚持退出"):
                        if _find_tap(adb, dets, kw): break
                    adb.press_back(); time.sleep(0.3); continue
                if _has_any(dets, _AD_REWARD_KW):
                    _find_tap(adb, dets, "关闭"); time.sleep(0.3)
                    if _has_any(_dets(adb), ("关闭",)):
                        _find_tap(adb, _dets(adb), "关闭"); time.sleep(0.3)
                    break

            if time.monotonic() >= deadline:
                logger.warning("lifemaker_launch: ad %d timeout", rnd + 1)
                adb.press_back(); time.sleep(1.0)
            time.sleep(0.5)

    # 4. Back to biubiu home → tap accelerate for 新加坡服
    # We may be on the ad page — press back to leave it
    if not _has_any(_dets(adb), ("新加坡",)):
        adb.press_back(); time.sleep(0.5)
        adb.press_back(); time.sleep(0.5)

    dets = _dets(adb)
    sg_y = None
    for d in dets:
        if "新加坡" in d["text"]:
            sg_y = d["center"][1]; break

    if sg_y is not None:
        accs = sorted([d for d in dets if "加速" in d["text"]],
                      key=lambda d: abs(d["center"][1] - sg_y))
        if accs:
            d = accs[0]
            adb.shell("input", "tap", str(d["center"][0]), str(d["center"][1]))
    else:
        accs = sorted([d for d in dets if "加速" in d["text"]],
                      key=lambda x: (x["center"][1], x["center"][0]))
        if len(accs) >= 2:
            d = accs[1]
            adb.shell("input", "tap", str(d["center"][0]), str(d["center"][1]))
        elif accs:
            _find_tap(adb, dets, "加速")

    # 5. Wait for game title screen — LLM gets a ready-to-tap screen
    for _ in range(30):
        dets = _dets(adb)
        if any(any(kw in d["text"] for d in dets) for kw in ("点击开始", "Life Makeover", "LIFE MAKEOVER", "设计室", "家园", "衣橱")):
            logger.info("lifemaker_launch: game appeared")
            return _ok({"message": "游戏已到达标题画面"})
        time.sleep(1.0)

    logger.warning("lifemaker_launch: game not appeared within 30s")
    return _ok({"message": "游戏正在启动", "warning": "等待超时"})


def _adb_check() -> bool:
    try:
        from src.device.emulator import emulator_manager
        return emulator_manager.first_online is not None
    except Exception:
        return False


registry.register(
    name="lifemaker_launch",
    game="lifemakeover",
    check_fn=_adb_check,
    description=(
        "【以闪亮之名全自动启动】桌面直启biubiu → 看广告→ 点加速→ 游戏自行启动。"
        "无需前置条件，一键完成。游戏启动后LLM等标题画面→关弹窗。"
    ),
    parameters={"type": "object", "properties": {}},
    handler=lifemaker_launch,
)
