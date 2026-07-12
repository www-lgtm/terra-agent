---
name: annihilation
description: "剿灭作战：从主界面→终端→剿灭（合成玉）→选择当前剿灭关卡→代理指挥→开始行动→等待战斗结束。"
tags: [剿灭, 作战, 周常, annihilation, battle]
game: arknights
type: guide
verified: false
---

## Steps

1. `adb_tap_position(0.88, 0.23)` 进入终端页面。
2. **不滑动，直接看 OCR**：在 OCR 文字里搜"合成玉"三个字。
   - 🔴 **没有"合成玉" → 已打完！** `notify_with_screen("本周剿灭已完成")` → `subtask_done('annihilation', '已打完')`，结束。
   - 有"合成玉" → `adb_tap('合成玉')` 进入剿灭关卡。
3. 进入关卡后：有PRTS代理卡 → 勾选「全权委托」；没有 → 勾选「代理指挥」复选框。
4. 点击「开始行动」，等待战斗完成。
5. 奖励拿满1800完成后 `notify_with_screen("剿灭已完成")` → `subtask_done('annihilation', ...)` → 点房子图标回主界面。

## Pitfalls

- 🔴 **终端页面是固定布局，不是轮播 banner。进去后直接 OCR 搜"合成玉"三个字，不要滑动！不要找"剿灭"按钮！**
- 没有"合成玉" = 本周已打完，不是页面错了，直接 skip 不要纠结。
- 优先用 PRTS 卡（全权委托）省时间；没有才手动勾代理指挥。
- 剿灭周上限 1800 合成玉，战斗约 15-20 分钟。

