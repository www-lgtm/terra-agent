---
deprecated: true
name: credit-shop
description: "信用商店：从主界面进入采购中心→信用交易所→购买信用商品→返回主界面。"
tags: [信用, 商店, 采购, 日常, credit, shop, store]
game: arknights
type: guide
verified: false
---

## Steps

1. 从主界面点击「采购中心」进入商店区域
2. 在采购中心顶部找到「信用交易所」入口并点击
3. 进入信用交易所后：
   - 先点「收取信用」领取当日信用点
   - 浏览可购买的商品列表（招聘许可、材料、家具零件等）
   - ⚠️ 同一商品可能有不同折扣（如"糖-75%" 50信用 vs "糖-50%" 72信用），OCR 都识别成"糖"
   - 优先购买：招聘许可 > 作战记录 > 技巧概要 > 材料（看哪个折扣最低）
   - 如遇到同名商品不同折扣 → magnify() 放大看清折扣数字 → tap_magnified() 精确点击
4. 每次购买后会弹出"获得物品"弹窗 → adb_back() 关闭弹窗
5. 信用点花完或无可购买商品后：
   - notify_with_screen("信用商店已购买完毕")
   - subtask_done('credit-shop', '购买了X件商品，剩余Y信用')
6. adb_back() 返回主界面

## Pitfalls

- ⛔ **截图前必须清除弹窗**：notify_with_screen 时画面不能有"获得物品"弹窗，用户什么都看不到
- 信用点上限300，每天尽量花完
- 同一物品不同折扣必须用 magnify 确认，不要用 adb_tap 靠文字匹配——会点错价格档位
- 信用交易所没有「购买物品」按钮，是直接点击商品然后点弹出的购买按钮
