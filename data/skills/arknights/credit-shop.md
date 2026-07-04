---
name: credit-shop
description: "信用商店：调用 credit_shop() 工具自动导航→收信用→扫描商品→按优先级购买→返回主界面。"
tags: [信用, 商店, 采购, 日常, credit, shop, store]
game: arknights
type: guide
verified: false
---

## Steps

1. 从主界面调用 `credit_shop()` 工具（确定性脚本，零 LLM 参与）：
   - 工具自动处理：导航到信用交易所 → 收信用 → OCR扫描商品 → 按优先级购买 → 截图通知 → 返回主界面
   - 购买策略：招聘许可 > 作战记录 > 技巧概要 > 材料，预算内全买
   - 工具返回 JSON：`{"bought": N, "spent": X, "remaining": Y, "items": [...]}`
2. subtask_done('credit-shop', '购买了N件商品，花费X信用')

## Pitfalls

### 手动执行方案（credit_shop() 不可用时的 fallback）

1. 从主界面点击「采购中心」进入商店区域
2. 在采购中心顶部找到「信用交易所」入口并点击
3. 先点「收取信用」领取当日信用点
4. 浏览商品列表，优先购买：招聘许可 > 作战记录 > 技巧概要 > 材料
   - 同名商品不同折扣用 magnify 确认 → tap_magnified 精确点击
5. 每次购买后弹出"获得物品"弹窗 → adb_back 关闭
6. notify_with_screen("信用商店已购买完毕") → subtask_done → 返回主界面
