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
🔴 **不要执行 visit-friend-base，那不是排班任务的一部分。**

### 第一步：出方案

1. 确保已有干员 Box 数据（调用 `scan_operator_box()`）
2. 调用 `base_plan()` → 工具自动截图通知用户
3. subtask_done('base-shift', '已生成排班方案')

### 第二步：手动设置进驻预设

用户根据方案，进入基建 → 进驻总览 → 逐个设施设预设队列。这是一次性操作。
