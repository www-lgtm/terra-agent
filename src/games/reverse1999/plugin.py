"""Reverse:1999 game plugin — validates multi-game architecture.

Demonstrates that adding a new game = 1 plugin file + 1 knowledge JSON dir.
Registers itself with GameRegistry at import time.
"""

from __future__ import annotations

from src.games.plugin import GameManifest, GamePlugin


class Reverse1999GamePlugin(GamePlugin):
    """重返未来：1999 game plugin."""

    manifest = GameManifest(
        id="reverse1999",
        name="重返未来1999",
        keywords=[
            "1999", "重返未来", "雨滴", "共鸣", "荒原",
            "活性", "独一律", "六星", "五星",
        ],
        knowledge_tables=[
            "characters",     # 角色数据
            "materials",      # 材料配方
            "stages",         # 关卡数据
        ],
        skill_dir="reverse1999",
        memory_dir="reverse1999",
        knowledge_dir="reverse1999",

        system_prompt_append="""## 1999 游戏常识
- 主页底部 tab：荒原、战斗、角色、仓库等
- 战斗 → 主线/资源本，刷资源本使用"复现"（扫荡券）
- 体力叫「活性」，抽卡货币叫「独一律」
- 角色养成材料在「荒原」和「资源本」获取
- **自动战斗**：战斗默认自动进行。顶部右侧圆环形图标在转动 = 自动已开启。未开启则点击该图标。部分 BOSS 关圆环禁用需手动操作，其余一律自动
- **复现（扫荡）**：编队界面有沙漏形状的「四级复现 ×4」按钮，点击一次即可自动完成全部4场战斗，省去手动重复打关。日常刷活性必须用复现
- **四级复现正确流程**：点击四级复现按钮一次 → 战斗自动进行（不要干预）→ 每场胜利结算画面点击屏幕继续下一场 → 4场全部完成后回到编队界面。**绝对不要重复点击复现按钮**，重复点击会中断扫荡
- 返回用 adb_back

## 材料识别

1999 仓库/背包中材料**只显示图标，不显示名字**——OCR 无法读取材料名称。
要识别画面中的材料图标，使用:
- **vlm_match_material(name)** — VLM 视觉识别，检查指定材料是否在当前画面
- **vlm_identify_icon(x, y)** — 裁剪指定坐标的图标区域，让 VLM 识别是什么材料

## 主界面锚点

每个任务从主界面开始、在主界面结束。

- **任务开始**：如果第一张截图不在主界面，先导航回主界面再开始任务。
- **任务结束**：调用 task_complete() 之前必须回到主界面。
- **快速返回**：用左上角返回箭头或底部 tab 一键回到主界面。不要用 adb_back 一步步退。""",

        dangerous_keywords=["雨滴", "纯雨滴", "独一律"],
        safe_compound_terms=[],
        require_confirmation_keywords=["购买", "兑换", "消耗雨滴", "消耗独一律"],

        task_keywords={
            "farm": ["刷", "活性", "体力", "复现", "扫荡"],
            "base": ["荒原", "收菜", "收取"],
            "query": ["多少", "进度", "查询", "还剩", "还有", "库存"],
            "plan": ["升级", "材料", "规划", "洞悉"],
        },
        task_priority={
            "farm": 1,
            "base": 2,
            "query": 3,
            "plan": 4,
        },
        android_packages=[
            "com.bluepoch.bluepoch.reversenineninetynine",  # 国服
            "com.bluepoch.bluepoch.reverse1999",            # 国际服/台服
            "com.bluepoch.bluepoch.m.en.reverse1999",       # 国际服移动版
        ],
    )

    def register_game_tools(self) -> None:
        """Load game-specific templates when the game is activated."""
        from src.vision.template_match import template_matcher

        try:
            template_matcher.load_all_templates("reverse1999")
        except Exception:
            pass

    def register_intelligence_tools(self) -> None:
        """Register Reverse:1999-specific intelligence tools (stub)."""
        # No intelligence tools yet — placeholder for future expansion
        pass

    def get_daily_tasks(self) -> list[dict]:
        return [
            {"description": "荒原收菜", "priority": 1},
            {"description": "清活性（刷资源本/复现）", "priority": 2},
            {"description": "每日签到/商店", "priority": 3},
        ]

    def gather_runtime_context(self, device_serial: str = "") -> dict:
        """Gather Reverse:1999-specific runtime state for context injection.

        Best-effort: returns hints for the LLM to check material stock
        and decide which stage to farm.  Actual OCR is done by the agent
        during task execution.
        """
        return {
            "material_hints": [
                "啮合齿轮",
                "盐封秘银",
                "双头形骨架",
                "银矿原油",
                "长青剑",
            ],
            "note": "材料库存需在仓库中手动检查，优先刷数量最少的材料关。",
        }
