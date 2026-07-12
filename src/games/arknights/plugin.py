"""Arknights game plugin — all game-specific logic in one place.

Registers itself with GameRegistry at import time.
"""

from __future__ import annotations

import json
import logging
import time as _time
from pathlib import Path

from src.games.plugin import GameManifest, GamePlugin

logger = logging.getLogger(__name__)


# ── Annihilation status cache ──
# The game main screen doesn't show annihilation completion status.
# We cache it in a small JSON file, updated by the scheduler or by
# task_complete after a 剿灭 run.  Cache is per-device-serial.

_ANNIHILATION_CACHE_DIR = Path("data/arknights")
_ANNIHILATION_CACHE_TTL = 7 * 24 * 3600  # 7 days (weekly reset on Monday)


def _cache_path(device_serial: str) -> Path:
    return _ANNIHILATION_CACHE_DIR / f"annihilation_{device_serial.replace(':', '_')}.json"


def _load_annihilation_cache(device_serial: str) -> bool | None:
    """Load cached annihilation status. Returns True/False/None."""
    cp = _cache_path(device_serial)
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
    except Exception:
        return None
    ts = data.get("timestamp", 0)
    if _time.time() - ts > _ANNIHILATION_CACHE_TTL:
        return None  # Expired
    return data.get("done", None)


def save_annihilation_cache(device_serial: str, done: bool) -> None:
    """Save annihilation status to cache file. Thread-safe enough for
    single-writer use."""
    cp = _cache_path(device_serial)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({
        "done": done,
        "timestamp": _time.time(),
    }, ensure_ascii=False), encoding="utf-8")


class ArknightsGamePlugin(GamePlugin):
    """明日方舟 game plugin."""

    manifest = GameManifest(
        id="arknights",
        name="明日方舟",
        keywords=[
            "明日方舟", "方舟", "arknights", "银灰", "能天使", "理智", "源石",
            "龙门币", "芯片", "基建", "公招", "作战", "gt-", "ce-", "pr-",
        ],
        knowledge_tables=[
            "recruit_tags",   # 公招标签→干员
            "base_skills",    # 基建技能
            "stages",         # 关卡数据
            "materials",      # 材料配方
            "chip_schedule",  # 芯片排班
        ],
        skill_dir="arknights",
        memory_dir="arknights",
        knowledge_dir="arknights",

        system_prompt_append="""## 主界面锚点

每个任务从主界面开始、在主界面结束。主界面上有 基建、作战、终端 等按钮。

- **任务开始**：如果第一张截图不在主界面，先导航回主界面再开始任务。
- **任务结束**：调用 task_complete() 之前必须回到主界面。所有技能提取都假设在主界面开始和结束 —— 这保证了技能的可组合性和可复用性。
- **截图通知规则**：notify_with_screen 前必须确认画面干净——不能有「获得物品」「获得物资」「正在提交反馈至神经……」等弹窗/动画。用 adb_back 反复关闭所有弹窗后再截图，否则用户看到的是弹窗不是结果
- **快速返回**：用左上角主页图标或「首页」按钮一键返回。不要用 adb_back 一步步退 —— 这会造成屏幕 hash 污染，破坏技能提取。
- 公共招募等按钮在右侧面板。
- 体力叫「理智」，抽卡货币叫「合成玉」
	- 代理指挥次数选择器可选 1-6 次，最大 6 次。刷图时理智充足优先选 ×6，不够时点开选择器用 OCR 确认可选次数。

## 仓库资源读取（基建排班用）

基建排班需要的资源：**龙门币**、**赤金**、**固源岩**、**源石碎片**。

### scan_depot() — ★★ 首选（VLM + MAA图标模板）
- 一键完成：VLM 读龙门币/合成玉 → 进仓库 → MAA图标模板图对图匹配赤金/固源岩/源石碎片
- 约 10-15 秒
- **前置条件**：必须在明日方舟主界面

### save_depot_resources() — 手动补充（fallback）
scan_depot 不靠谱时手动读数注入。

## 明日方舟常识

### 精英等级图标外观
- E0: 等级数字旁无徽章，仅显示职业图标(剑/盾/十字等)
- E1: 银色/蓝灰色向上的箭头图标，底部有蚀刻数字 "1"
- E2: 橙色/金色翼形图标，底部有蚀刻数字 "2"（6★的徽章更华丽更大）
- 职业图标在卡片左上角，精英徽章在等级数字的正下方或正右方

### 等级上限
| 稀有度 | E0 满级 | E1 满级 | E2 满级 |
|--------|---------|---------|---------|
| 6★    | 50      | 80      | 90      |
| 5★    | 50      | 70      | 80      |
| 4★    | 45      | 60      | 70      |
| 3★    | 40      | 55      | —       |

推导: LV90→必然 E2; LV80 且 5★+→必然 E2; LV>50 且 6★→至少 E1; LV1→大概率 E0""",

        dangerous_keywords=["源石", "合成玉"],
        safe_compound_terms=["源石订单", "源石碎片"],
        require_confirmation_keywords=["购买", "消费", "消耗源石"],

        # Simple action verbs for deterministic dispatch (avoids LLM for single-agent)
        task_verbs=[
            "刷", "收菜", "公招", "换班", "清体力", "基建", "剿灭",
            "采购", "抽卡", "打", "闯关", "推图", "制造", "贸易",
        ],

        task_keywords={
            "create_guide": ["存操作", "/save", "/s", "加操作"],
            "schedule": ["定时", "每天", "每周", "每隔", "每个小时", "每分钟"],
            "farm": ["刷", "清体力", "farm", "理智", "龙门币", "糖", "装置", "芯片", "gt-", "ce-", "pr-"],
            "base": ["基建", "收菜", "收取", "制造站", "贸易站", "换班", "排班"],
            "query": ["多少", "进度", "查询", "还剩", "还有", "库存"],
            "plan": ["精二", "精英化", "升级", "材料", "规划"],
        },
        task_priority={
            "create_guide": 0,  # 优先匹配，避免被其他关键词误分类
            "schedule": 0,
            "farm": 1,
            "base": 2,
            "query": 3,
            "plan": 4,
        },
        android_packages=[
            "com.hypergryph.arknights",     # 官服
            "com.hypergryph.arknights.bili", # B服
            "com.YoStarEN.Arknights",        # 国际服
            "tw.txwy.and.arknights",         # 台服
            "com.YoStarKR.Arknights",        # 韩服
        ],
    )

    def get_vlm_adapter(self) -> dict[str, str]:
        """Return Arknights VLM→Chinese OCR term mapping."""
        from src.games.arknights.adapter import VLM_TO_CN
        return VLM_TO_CN

    def register_intelligence_tools(self) -> None:
        """Register Arknights-specific intelligence tools."""
        from src.intelligence.arknights.base_scheduler import BaseScheduler
        from src.intelligence.arknights.base_chain import BaseChainConductor
        from src.intelligence.base import get_intelligence_registry

        registry = get_intelligence_registry("arknights")
        registry.register(BaseScheduler())
        registry.register(BaseChainConductor())

    def register_game_tools(self) -> None:
        """Load operator box templates for Box scanning (operbox elite, potential, profession)."""
        from src.vision.template_match import template_matcher

        try:
            template_matcher.load_all_templates("arknights")
        except Exception:
            logger.warning("Failed to load templates for arknights", exc_info=True)

    def get_daily_tasks(self) -> list[dict]:
        return [
            {"description": "基建收菜（制造站+贸易站产物）", "priority": 1},
            {"description": "会客室线索处理（快捷置入→领NEW→传递重复）", "priority": 2},
            {"description": "无人机加速（控制中枢→无人机→加速生产设施）", "priority": 3},
            {"description": "公开招募（检查招募位+刷新标签）", "priority": 4},
            {"description": "信用商店购买", "priority": 5},
            {"description": "清体力（刷材料关卡）", "priority": 6},
            {"description": "剿灭作战（如未打满）", "priority": 7},
        ]

    def gather_runtime_context(self, device_serial: str = "") -> dict:
        """Gather Arknights-specific runtime state for context injection.

        Best-effort: tries to OCR the main screen for sanity value.
        Returns empty dict if the game is not on the main screen or ADB is
        unavailable.  The LLM can check itself during task execution.
        """
        result: dict = {}
        try:
            from src.device.adb import get_adb
            adb = get_adb()
            if not adb._heartbeat_ok:
                return result

            img = adb.get_screenshot_image()

            # Try OCR on the top bar where sanity is displayed.
            # The sanity number is near the top of the screen with a ⚡ icon.
            # We crop roughly the top 8% of a 1920px screen (~150px).
            from src.vision.ocr import ocr_engine as _ocr
            texts = _ocr.read_texts(img, region=[0, 0, 1080, 160])

            # Sanity format: a number like "120/135" or "120 / 135"
            import re
            for t in texts:
                m = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', t)
                if m:
                    result["sanity_current"] = int(m.group(1))
                    result["sanity_max"] = int(m.group(2))
                    break

            # P3: Check cached annihilation status from last known state.
            # Updated by the scheduler or by task_complete after a 剿灭 run.
            _ann_cache = _load_annihilation_cache(device_serial)
            result["annihilation_done"] = _ann_cache

        except Exception:
            pass  # Best-effort, non-critical

        return result
