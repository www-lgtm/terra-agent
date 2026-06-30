"""Life Makeover (以闪亮之名) game plugin — validates multi-game architecture.

以闪亮之名 is a dress-up/life-sim game.  Daily tasks involve:
  - Logging in via biubiu accelerator (国际服/Singapore server)
  - Closing announcement/friend-recruit popups
  - Daily schedule quests (签到, 思绪漫步, story/hard stages, 心意之期)
  - Story stage farming with 搭配多次 (multi-clear)
  - 日常事件簿 dispatch (time-based missions)
  - Photo/dress-up tasks
  - 小镇代言 (town endorsement battles)
  - 协会应援 (guild support)
  - Weekly activity rewards

Registers itself with GameRegistry at import time.
"""

from __future__ import annotations

from src.games.plugin import GameManifest, GamePlugin


class LifemakerGamePlugin(GamePlugin):
    """Life Makeover (以闪亮之名) game plugin."""

    manifest = GameManifest(
        id="lifemakeover",
        name="以闪亮之名（新马服）",
        keywords=[
            "以闪", "闪亮之名", "lifemaker", "life makeover", "lifemakeover",
            "以闪亮之名", "以闪亮之名国际服", "以闪亮之名新马服", "biubiu",
            "搭配", "羁绊", "代言", "协会", "小镇",
        ],
        knowledge_tables=[],
        skill_dir="lifemaker",
        memory_dir="lifemaker",
        knowledge_dir="lifemaker",

        system_prompt_append="""## 以闪亮之名 (Life Makeover) 游戏常识
- 国际服需要通过 **biubiu加速器** 启动，不要直接启动游戏本体
- 主界面底部 tab：日程、搭配、羁绊、协会、小镇 等
- 体力叫「体力」，抽卡货币叫「钻石」
- 关卡分为：普通主线、困难主线、心意之期
- **困难主线刷材料**：新马服刷固定关卡 2-3, 2-4, 2-5, 2-7, 2-8, 2-9, 2-10, 2-12（免活复刻材料），跳过 2-6 和 2-11
- **心意之期**：优先选择未成套的套装关卡刷，进页面先看各套装收集进度（如3/6=缺3件），缺件多的优先刷
- **搭配多次**：关卡结算后点击"搭配多次"可快速刷多次掉落，比单次高效
- **日常事件簿**：派遣羁绊完成时间任务，需选择属性匹配的羁绊
- **小镇代言**：使用最佳搭配挑战代言，分数越高奖励越好
- **协会应援**：每日有次数上限，优先完成
- 弹窗（好友招募/活动公告/签到）用右上角X关闭

## 主界面锚点

- **任务开始**：如果不在主界面（能看到角色立绘+底部菜单栏），先导航回去
- **任务结束**：调用 task_complete() 之前必须回到主界面
- **快速返回**：左上角返回箭头，不要用 adb_back 一步步退""",

        dangerous_keywords=["购买钻石", "消耗钻石", "充值"],
        safe_compound_terms=["可获得钻石", "钻石等奖励", "钻石奖励", "获得钻石", "钻石*", "钻石 x",
                            "完成每日日程"],
        require_confirmation_keywords=["购买钻石", "消耗钻石", "充值"],

        task_keywords={
            "daily": ["日常", "每日", "日程", "任务"],
            "farm": ["刷", "体力", "搭配", "关卡", "掉落"],
            "dispatch": ["派遣", "事件簿", "事件"],
            "photo": ["拍照", "萌拍", "摄影"],
            "endorse": ["代言", "小镇"],
            "guild": ["协会", "应援", "公会"],
        },
        task_priority={
            "daily": 1,
            "dispatch": 2,
            "farm": 3,
            "guild": 4,
            "endorse": 5,
            "photo": 6,
        },
        android_packages=[
            "com.archosaur.lifemakeover",          # 国际服
            "com.archosaur.lifemakeover.gp",       # Google Play 版
        ],
    )

    def register_game_tools(self) -> None:
        """Load game-specific templates when the game is activated."""
        from src.vision.template_match import template_matcher

        try:
            template_matcher.load_all_templates("lifemaker")
        except Exception:
            pass

    def register_intelligence_tools(self) -> None:
        """Register lifemaker-specific intelligence tools (stub)."""
        pass

    def get_daily_tasks(self) -> list[dict]:
        return [
            {"description": "启动游戏（biubiu加速器→打开游戏）", "priority": 1},
            {"description": "关闭弹窗（公告/好友招募/签到）", "priority": 2},
            {"description": "每日日程任务（签到/思绪漫步/关卡/心意之期）", "priority": 3},
            {"description": "日常事件簿派遣", "priority": 4},
            {"description": "协会应援", "priority": 5},
            {"description": "小镇代言", "priority": 6},
            {"description": "周活跃奖励领取", "priority": 7},
        ]

    def gather_runtime_context(self, device_serial: str = "") -> dict:
        """Gather lifemaker-specific runtime state for context injection."""
        return {
            "note": "以闪亮之名国际服需要通过biubiu加速器启动，不要直接点游戏图标。",
        }
