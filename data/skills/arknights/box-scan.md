---
name: box-scan
description: "扫描干员列表 → scan_operator_box(MAA引擎)自动扫描 → 返回全量box JSON"
tags: [box, scan, 扫描, 干员列表, box-scan]
game: arknights
type: guide
verified: false
---

## Steps

### 阶段1：进入干员列表

1. 从主界面点击干员档案入口进入干员列表

### 阶段2：自动扫描

4. `scan_operator_box()` — 等待30-120秒，MAA自动向右逐屏扫描全量干员

### 阶段3：返回结果

5. `task_complete` 返回汇总：共N名，E2:X E1:Y E0:Z

## 输出

干员数据直接写入排班链 session → 后续自动计算最优排班。

## Pitfalls

- **★ 不需要手动滑动** — MAA 自动处理截图、翻页、边界检测
- **★ 不需要 magnify** — 工具内部用 MAA C++ 引擎识别
