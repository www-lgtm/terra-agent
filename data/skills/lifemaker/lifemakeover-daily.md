---
name: lifemakeover-daily
description: "以闪亮之名新马服(Life Makeover)每日日常任务全流程：启动游戏→关闭弹窗→签到+思绪漫步→刷固定材料关卡→事件派遣→小镇代言→协会应援→收尾"
tags: [日常, 任务, 每日, daily, routine, 编排]
game: lifemakeover
type: orchestrator
verified: false
subskills:
  - lifemakeover-launch
  - lifemakeover-schedule
  - lifemakeover-farm
  - lifemakeover-dispatch
  - lifemakeover-endorse
  - lifemakeover-guild
---

## Steps

1. skill_run('lifemakeover-launch')
   # 启动游戏（biubiu加速器 → 看广告获取时长 → 加速启动 → 标题画面进入 → 关闭弹窗）
2. skill_run('lifemakeover-schedule')
   # 日程任务：签到 → 思绪漫步 → 时尚对决（周一15次/周日5次/平时3次，打高分段对手）
3. skill_run('lifemakeover-farm')
   # 刷关卡（从日程页面进入）：普通主线10次 → 困难主线固定关卡10次(2-3~2-12) → 心意之期10次 → 一键领取活跃度
4. skill_run('lifemakeover-dispatch')
   # 闪亮之旅 → 日常事件簿派遣
5. skill_run('lifemakeover-endorse')
   # 闪亮之旅 → 代言女王（每日宣传 + 进阶代言四街区）+ 一键拾取
6. skill_run('lifemakeover-guild')
   # 协会应援（普通应援）+ 灵感碰撞（5次）+ 开启企划
7. 收尾检查：
   - 回到日程页面 → 检查活跃度进度条，确认达到 120 且所有档位奖励已领取
   - 🔴 活跃度未达 120 → 回日程页面看还有哪个任务没完成（0/X 或未满），继续补做
   - ⚠️ **确认 120/120 后在日程页面立刻 notify_with_screen**——趁画面还是满档奖励截图发给用户！
   - 检查完邮件/信箱后回到主界面，**不要再 notify**（日程截图已经在前面发了）
   - 检查周活跃奖励（"精彩活动/海量福利"入口，检查即可，不用领钻石档位）
8. notify_with_screen("以闪亮之名今日日常任务全部完成，活跃度120/120 ✓")
   # ⚠️ 这步必须在日程页面（120满档画面上），确认后就发，不要先跳到其他页面！
9. 检查邮件/信箱 → 回到主界面 → task_complete()

## Pitfalls

- 每个子技能执行失败不要中断全部任务——notify_with_screen 通知用户后继续下一个
- 困难主线必须刷固定关卡（2-3~2-12，跳过2-6和2-11），否则体力浪费在无用掉落上
- 协会应援只做普通应援，全力应援消耗钻石
- 游戏加载黑屏/进度条/初始化提示 = 正常过渡，不要操作，等待画面变化
