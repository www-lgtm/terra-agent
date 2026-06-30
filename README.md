# Terra Agent

基于 LLM 的游戏自动化助手，通过大语言模型自主控制 Android 模拟器。

支持《明日方舟》《重返未来：1999》《以闪亮之名》多款游戏，提供微信远程控制和 CLI 交互两种使用方式。

## 快速开始

### 环境

- Python 3.11+
- Windows 10/11
- Android 模拟器（MuMu 12 / LDPlayer / BlueStacks）

### 安装

```bash
cd terra-agent
pip install -e .
pip install -e ".[ocr]"   # 可选：OCR 支持
```

### 配置

```bash
cp .env.example .env
```

编辑 `.env`：

```env
MIMO_API_KEY=sk-xxx          # LLM API Key（必填）
ADB_PATH=D:\platform-tools\adb.exe  # ADB 路径（必填）
EMULATOR_TYPE=mumu           # 模拟器类型
```

### 使用

```bash
python -m src.cli chat              # CLI 交互模式
python -m src.cli run "刷1-7"       # 单次任务
python -m src.cli weixin            # 微信 Bot
python -m src.cli skills            # 查看可用技能
python -m src.cli emulator          # 模拟器管理
```

## 主要功能

- **自然语言操控**：输入日常任务描述，LLM 自动理解并操作屏幕
- **微信远程控制**：扫码登录后通过微信随时随地操控
- **明日方舟日常**：基建收菜、换班、公开招募、刷材料、信用商店
- **模拟器管理**：自动启动、健康监控、定时重启
- **多游戏支持**：插件化架构，添加新游戏只需一个插件文件

## 目录结构

```
terra-agent/
├── src/
│   ├── agent/         # Agent 主循环、状态管理
│   ├── concierge/     # 消息路由、会话管理
│   ├── device/        # ADB 操作、模拟器管理
│   ├── gateway/       # 微信 iLink 协议
│   ├── games/         # 游戏插件
│   ├── intelligence/  # 游戏智能（排班优化等）
│   ├── llm/           # LLM API 客户端
│   ├── memory/        # 记忆系统
│   ├── scheduler/     # 定时任务
│   ├── skills/        # 技能管理
│   ├── tools/         # 工具系统（点击/滑动/OCR/截图）
│   ├── utils/         # 工具函数
│   └── vision/        # OCR、视觉处理
├── config/            # 配置文件
├── data/
│   ├── skills/        # 技能定义
│   └── templates/     # 模板图片
├── tests/
└── scripts/           # 开发工具脚本
```

## 支持的模拟器

MuMu Player 12（推荐）、LDPlayer、BlueStacks、通用 ADB

## FAQ

**模拟器没反应？** `adb devices` 检查连接，确保开启了 USB 调试。

**任务执行报错？** 查看 `data/logs/` 下的日志文件。API 连接错误通常等几分钟重试即可。
