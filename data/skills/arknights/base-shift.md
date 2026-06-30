---
name: base-shift
description: "基建换班：scan 仓库+干员 → 系统自动出方案 → base_shift_maa() 执行最优方案 → 验证满员"
tags: [基建, 换班, 轮换, 排班, 日常, base, shift]
game: arknights
type: guide
verified: false
---

## 基建换班 — 统一流程

任何换班都走方案，没有例外：

### 第一步：出方案（无方案时）

如果已有排班方案缓存（系统通知过"方案已生成"），跳过此步。否则：

1. `scan_depot()` 读取仓库资源
2. `scan_operator_box()` 扫描干员列表（MAA 全自动，约 30 秒）
3. 系统自动触发排班计算 → 方案出现

### 第二步：执行换班

1. `base_shift_maa()` → MAA 执行方案中的全部设施（制造站/贸易站/发电站/控制中枢/会客室/办公室/宿舍）
   - custom plan 按干员名放置，无需扫描，通常 2-5 分钟完成
   - 多方案但无人选择 → 执行推荐方案（plan_index=0）
2. 等待 MAA 完成

### 第三步：验证

1. 进入基建总览 → 逐个点击每个设施确认全部满员（无心情 0/24 的干员）
2. MAA 已处理所有设施，LLM 只需核实，不需要手动换人。仅当发现遗漏时才手动补换
3. 全部设施满员 → notify_with_screen 截图通知 → subtask_done → task_complete

---

## 注意事项

- **不要手动 adb_tap 逐个换人** —— MAA custom plan 已指定干员，自动完成。
- 如果 base_shift_maa 返回"没有已缓存的排班方案"，回到第一步出方案。
- **无人机**：MAA 自动检测方案配置选择加速类型（SyntheticJade/Money/CombatRecord/PureGold/OriginStone），`base_shift_maa()` 调用时自动处理，**不需要 LLM 手动进基建加速**。如需指定：`base_shift_maa(drones='Money')`。

## Pitfalls

- 🔴 **没有方案就换班会被拒绝**：base_shift_maa 要求必须有方案缓存。如果报错"没有已缓存的排班方案"，先 scan_depot() → scan_operator_box() 出方案
- MAA 完成后务必进入基建总览→逐个点击每个设施检查心情。不要只扫一眼就 task_complete
- 如果 MAA 返回超时但 swap_events > 0 或事件数正常增长，说明已完成，按已完成处理
- 如果 base_shift_maa 失败，检查 MAA 路径
