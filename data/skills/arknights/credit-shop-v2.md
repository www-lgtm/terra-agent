---
name: credit-shop-v2
description: "信用商店：从主界面进入采购中心→信用交易所→购买信用商品→返回主界面。"
tags: [信用, 商店, 采购, 日常, credit, shop, store]
game: arknights
type: script
verified: true
version: 1
replaces: credit-shop
coords_verified_at: 2026-06-29T22:21:30.792752+00:00
---

## Steps

1. adb_tap('采购中心')  # [1458, 786] 主界面右侧面板
2. adb_tap('信用交易所')  # 采购中心顶部或可见区域
3. 收取信用 + 购买商品（招聘许可优先 > 材料折扣最低者）
4. 每次购买后 adb_back() 关"获得物品"弹窗
5. 购买完毕后 notify_with_screen("信用商店已购买完毕") → subtask_done('credit-shop') → adb_back() 回主界面

## Pitfalls

- ⛔ **截图前必须清除弹窗**：notify_with_screen 时画面不能有"获得物品"弹窗，用户什么都看不到
- 信用点上限300，每天尽量花完
- 同一物品不同折扣必须用 magnify 确认，不要用 adb_tap 靠文字匹配——会点错价格档位
- 信用交易所没有「购买物品」按钮，是直接点击商品然后点弹出的购买按钮
