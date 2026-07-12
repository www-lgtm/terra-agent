---
name: base-shift
description: "基建排班：base_plan() 出方案 → 用户手动设进驻预设 → 日常 base_collect 队列轮换"
tags: [基建, 换班, 轮换, 排班, 日常, base, shift]
game: arknights
type: guide
verified: true
---

## 基建排班 — 调用 base_plan() 即可

🔴 `base_plan()` 工具内置截图通知，直接 subtask_done。
🔴 **不要手动进基建排班、不要 notify_with_screen、不要做任何额外操作。**
🔴 **不要执行 visit-friend-base 和 base-reception-clue，那不是排班任务的一部分。**
🔴 **排班任务只有出方案这一步。subtask_done 后立即 task_complete，不要检查是否有其他子任务。**

### 第一步：收集数据

1. 确保已有干员 Box 数据（调用 `scan_operator_box()`）
2. 🔴 **收集仓库材料数据**（赤金、固源岩、源石碎片数量直接影响方案可持续性分析）：
   - 🔴 **这一步必须做，不能跳过。** 只有用户明确回复「跳过」或「直接出方案」才能省略。
   - 进入仓库：主界面右下角「仓库」按钮。OCR 匹配「仓库」→ adb_tap
   - 🔴 **进仓库后直接 ask_user，不要 magnify、不要自己识别、不要翻标签页。截图已经有了，直接问。**
   - 调用 `ask_user("请告诉我仓库里这三样材料各有多少个：赤金、固源岩、源石碎片？")`
   - 用户回复数量后，调用 `save_depot_resources(lmd=<主界面龙门币>, puregold=<赤金>, orirock=<固源岩>, origin_stone=<源石碎片>)`
   - 如果 ask_user 时用户说「跳过」或「直接出方案」，就跳过这步继续

### 第二步：出方案

3. 调用 `base_plan()` → 工具自动生成排班图片通知用户
4. subtask_done('base-shift', '已生成N个排班方案')
5. 🔴 **立即 task_complete()，不要管匹配列表里的其他技能。排班任务已经完成。**

### 第三步：手动设置进驻预设

用户根据方案图片，进入基建 → 进驻总览 → 逐个设施设预设队列。这是一次性操作。
