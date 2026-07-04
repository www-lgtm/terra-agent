---
name: credit-shop
description: "信用商店：调用 credit_shop() 工具自动导航→收信用→扫描商品→按优先级购买→返回主界面。"
tags: [信用, 商店, 采购, 日常, credit, shop, store]
game: arknights
type: guide
verified: false
---

## Steps

1. 调用 `credit_shop()` 工具 — 全自动处理。
   - 工具内置：导航→收信用→扫描→购买→截图通知→返回主界面
   - 🔴 **必须调 credit_shop()，禁止手动点击采购中心/信用交易所！**
2. subtask_done('credit-shop', '购买了N件商品，花费X信用')
