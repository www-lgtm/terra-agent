---
name: daily
description: "明日方舟日常任务：基建收菜→信用商店→公招→会客室线索→（周六）剿灭→（理智>30）刷1-7"
tags: [日常, 任务, 每日, daily, routine, 编排]
game: arknights
type: orchestrator
verified: false
subskills:
  - base-collect
  - credit-shop
  - recruit
  - annihilation
  - farm-1-7
---

## Steps

1. skill_run('base-collect')
   - 🔴 **如果快速链执行成功 → 完成后保持通知面板打开，立即 notify_with_screen("基建产物已全部收取") → subtask_done('base-collect', ...)**
   - 如果快速链失败（无坐标/坐标过期）→ 手动按 base-collect guide 执行 → 在底部面板打开时 notify_with_screen → subtask_done
   - 🔴 **绝对不要在截图前关闭通知面板或按返回键！** 用户需要看到面板里的收取内容。
2. skill_run('credit-shop')
   - 完成后 notify_with_screen("信用商店已购买完毕") → subtask_done('credit-shop', ...)
3. skill_run('recruit')
   - 完成后 notify_with_screen("公招完成") → subtask_done('recruit', ...)
4. 检查 [系统上下文] 中的星期和理智：
   - 如果是周六或周日且剿灭未完成 → skill_run('annihilation')
   - 如果理智 > 30 → **执行两轮**：先 skill_run('farm-1-7')，等待完成后再 skill_run('farm-1-7')（共两轮6倍，消耗72理智）
5. 完成任务前验证：
   - **进入「日常任务」标签页**，切换到日常 tab（最左侧，adb_tap_position(0.15,0.11)）
   - 🔴 **不要滚动左侧奖励栏！** 明日方舟机制：已领取奖励自动沉底，**未领取的奖励永远留在顶部**。只看顶部即可判断是否全部领完，滚动到底部看到的全是已完成奖励反而误判。
   - 顶部所有奖励都显示"已完成" → notify_with_screen("明日方舟今日奖励已全部领完")
   - 🔴 **截图必须停留在任务面板上**，不能退回主界面再截。用户需要看到任务面板才能判断是否真的全部领完。
   - 周日：日常领完后再切到「周常任务」tab（adb_tap_position(0.35,0.11)）→ 同样只看顶部不要滚动 → notify_with_screen("明日方舟本周奖励已领完")
   - 如发现未领取的奖励（顶部有"点击领取"按钮）→ 点击领取 → 等动画消失 → 继续检查
6. 回到主界面 → task_complete()



## Pitfalls

- 剿灭每周六打，刷关是可选的，根据用户的理智和未完成任务需求决定
- 如果某个子技能执行失败（如基建加载超时），调用 notify_with_screen("子技能失败：<技能名>") 通知用户并继续执行后续技能，不要全部中断
- 🔴 **子技能成功完成后必须先 notify_with_screen 截图再 subtask_done**。notify 截图 → subtask_done，顺序不能反。subtask_done 调用后上下文被清理就不知道之前做了什么。
- 🔴 **基建收菜截图必须在通知面板打开时进行**。面板展示了收取内容（制造站产物/订单交易/干员信赖），关闭面板后截到的只是基建俯视图，用户什么都看不到。
- 任务面板顶部 tab 从左到右：日常(adb_tap_position 0.15,0.11) | 周常(0.35,0.11) | 主线 | 特勤。不要误入特勤或主线！
- "报酬已领取"是任务行的完成状态标记，不是弹窗！看到它说明奖励已领过，绝对不要用 adb_back 关它
- 🔴 **notify_with_screen 截图必须干净**：绝对不能截到"获得物品""获得物资""正在提交反馈至神经……"等弹窗/动画。必须等动画结束、用 adb_back 清掉所有弹窗后，确认能看到任务列表本身再截图
- 🔴 "正在提交反馈至神经……"是服务器通信动画，领取还未完成。必须等它消失、画面稳定后再判断是否全部领取完毕
- 🔴 **不要滚动任务左侧奖励栏！** 已领取奖励自动沉底，只检查顶部就能判断是否全部领完。滚动到底部看到的全是已完成，会产生全部完成的错觉。
- 🔴 **线索处理必须做**：MAA 换班后线索不会自动放置。每天做完基建收菜后必须进会客室处理线索（快捷置入→领NEW→传递重复），否则线索仓库满了就浪费了
- 🔴 **无人机由 MAA 自动处理**：base_shift_maa 执行时根据方案产品配置自动选择加速类型并加速，不需要 LLM 手动进基建操作
