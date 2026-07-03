---
name: daily
description: "明日方舟日常任务：基建收菜→信用商店→公招→会客室线索→（周六）剿灭→（理智>30）刷1-7"
tags: [日常, 任务, 每日, daily, routine, 编排]
game: arknights
type: orchestrator
verified: false
subskills:
  - credit-shop
  - recruit
  - annihilation
  - farm-1-7
verify: daily-reward-check
---

## Steps

1. 进入基建后调用 `base_collect()` 工具 → 内置截图通知 → subtask_done('base-collect', ...)
   - 如果 base_collect() 返回 success: false → subtask_done('base-collect', '收菜失败：<error>')，继续下一个子任务
2. skill_run('credit-shop')
3. skill_run('recruit')
4. 检查 [系统上下文] 中的星期和理智：
   - 如果是周六或周日且剿灭未完成 → skill_run('annihilation')
   - 如果理智 > 30 → **执行两轮**：先 skill_run('farm-1-7')，等待完成后再 skill_run('farm-1-7')（共两轮6倍，消耗72理智）
   - 🔴 **条件不满足直接跳过**：跳过前先调用 subtask_done('<skill_name>', '条件不满足跳过')，例如 subtask_done('annihilation', '周四跳过')。这样系统知道该子任务已处理完毕。
   - 🔴 两轮刷完后，先去 Step 5 检查日常奖励。如果奖励还有未领完的档位 → 继续刷 1-7 补充每日任务点数，直到所有档位领取完毕。
5. 完成任务前验证：
   - **进入「日常任务」标签页**，切换到日常 tab（最左侧，adb_tap_position(0.15,0.11)）
   - 🔴 **截图必须截到任务面板本身**，让用户能看到左侧奖励栏和右侧任务列表。
   - 🔴 **所有档次的奖励要全部领完**。
   - 🔴 **截图时画面上如果有"报酬已领取"半透明大字 → 那是任务完成水印，不是弹窗！直接用 notify_with_screen 截图，不要 adb_back。**
   - 🔴 **截图前清掉的是"获得物资""正在提交反馈至神经……"这类弹窗/动画，不是"报酬已领取"水印。**
   - 🔴 **不要滚动左侧奖励栏！** 明日方舟机制：已领取奖励自动沉底，**未领取的奖励永远留在顶部**。
   - 周日：日常领完后再切到「周常任务」tab（adb_tap_position(0.35,0.11)）→ 同样要求 → notify_with_screen("明日方舟本周奖励已领完")
   - 🔴 验证完成后调用 subtask_done('daily-reward-check', '已进入日常任务页确认所有奖励已领取并截图')
6. 🔴 调用 task_complete()。回到主界面后立即 task_complete，不要重新审视编排列表、不要检查理智、不要做任何额外操作。



## Pitfalls

- 剿灭每周六打，刷关是可选的，根据用户的理智和未完成任务需求决定
- 如果某个子技能执行失败（如基建加载超时），调用 notify_with_screen("子技能失败：<技能名>") 通知用户并继续执行后续技能，不要全部中断
- 🔴 **子技能成功完成后必须先 notify_with_screen 截图再 subtask_done**（base_collect 除外，截图通知由工具内部处理）。notify 截图 → subtask_done，顺序不能反
- 任务面板顶部 tab 从左到右：日常(adb_tap_position 0.15,0.11) | 周常(0.35,0.11) | 主线 | 特勤。不要误入特勤或主线！
- "报酬已领取"是任务行的完成状态标记，不是弹窗！看到它说明奖励已领过，绝对不要用 adb_back 关它
- 🔴 **notify_with_screen 截图必须干净**：绝对不能截到"获得物品""获得物资""正在提交反馈至神经……"等弹窗/动画。必须等动画结束、用 adb_back 清掉所有弹窗后，确认能看到任务列表本身再截图
- 🔴 "正在提交反馈至神经……"是服务器通信动画，领取还未完成。必须等它消失、画面稳定后再判断是否全部领取完毕
- 🔴 **不要滚动任务左侧奖励栏！** 已领取奖励自动沉底，只检查顶部就能判断是否全部领完。滚动到底部看到的全是已完成，会产生全部完成的错觉。
