"""Batch create initial skill files for bootstrapping.

Usage:
    python scripts/seed_skills.py

Creates initial skill files if they don't already exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.skills.manager import skill_manager

SEED_SKILLS = {
    "boot-to-main": """---
name: boot-to-main
description: "从模拟器桌面启动明日方舟并进入主界面"
tags: [boot, system]
game: arknights
---

# 启动到主界面

## When to Use
- 游戏不在前台时
- 刚启动模拟器后

## Steps
1. find_and_tap: "明日方舟"  # 桌面图标
2. wait_until_gone: "加载中" timeout=60
3. find_and_tap: "开始唤醒"  # 登录后点击
4. wait_until_gone: "loading" timeout=30

## Pitfalls
- 可能需要更新下载（大版本后）
- 登录过期需人工处理
""",

    "farm-ce-5": """---
name: farm-ce-5
description: "刷CE-5获取龙门币"
tags: [farm, lmd, sanity]
game: arknights
---

# Farm CE-5

## When to Use
- 需要龙门币
- 理智 >= 30

## Steps
1. find_and_tap: "作战"
2. find_and_tap: "物资筹备"
3. swipe_down_until: "CE-5"
4. find_and_tap: "CE-5"
5. find_and_tap: "代理指挥"
6. wait_until_gone: "行动结束" timeout=120
7. tap_anywhere  # 关闭结算

## Pitfalls
- 理智不足时会弹窗提示
- 周六日龙门币本全天开放
""",

    "collect-base": """---
name: collect-base
description: "基建一键收取制造站/贸易站/信赖"
tags: [base, daily, collect]
game: arknights
---

# 基建收菜

## When to Use
- 每日清体力后
- 制造站/贸易站已满时

## Steps
1. find_and_tap: "基建"
2. find_and_tap: "一键收取"
3. find_and_tap: "确认"
4. find_and_tap: "返回"
5. tap_anywhere  # 返回主界面

## Pitfalls
- 宿舍信任也需要单独点
- 控制中枢提示可以忽略
""",
}


def main() -> None:
    created = 0
    skipped = 0

    for name, content in SEED_SKILLS.items():
        existing = skill_manager.load(name)
        if existing:
            print(f"SKIP: {name} (already exists)")
            skipped += 1
        else:
            skill_manager.save(name, content)
            print(f"CREATE: {name}")
            created += 1

    print(f"\nCreated {created}, skipped {skipped}")


if __name__ == "__main__":
    main()
