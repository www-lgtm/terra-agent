# Terra Agent 架构文档

## 系统架构

```
 用户 (微信/CLI)
       │
       ▼
  ┌─────────────┐
  │  WeixinBot   │  (asyncio 事件循环, iLink 协议)
  │  CLI 入口    │  (python -m src.cli chat)
  └──────┬──────┘
         │ 消息
         ▼
  ┌─────────────┐
  │  Concierge   │  MessageRouter（确定性路由，无 LLM）
  │  (管家层)     │  多游戏检测 → 技能匹配 → 代理委派
  └──────┬──────┘
         │ 委派任务
         ▼
  ┌─────────────┐
  │ TerraAgent   │  主循环: LLM ↔ 工具执行 ↔ 屏幕注入
  │  (代理层)     │  最多 800 次迭代，30 分钟超时
  └──┬──┬──┬────┘
     │  │  │
     ▼  ▼  ▼
  ┌───┐ ┌───┐ ┌───┐
  │LLM│ │ADB│ │OCR│
  └───┘ └───┘ └───┘
```

## 核心设计：Agent-Concierge 分离

### Concierge（管家层）
- **职责**：消息路由、会话管理、任务队列、多游戏调度
- **路由策略**：确定性规则（无 LLM 调用），仅模糊场景触发 LLM
- **实现**：`src/concierge/router.py` — `MessageRouter` 类
- **每用户一个实例**，持有 `ConciergeSession`

### TerraAgent（代理层）
- **职责**：接收任务描述 → LLM 循环调用工具 → 完成任务
- **循环**：截图 → 构建提示词 → 调用 LLM → 解析工具调用 → 执行 → 注入反馈
- **实现**：`src/agent/loop.py` — `TerraAgent` 类
- **并发**：每个设备最多 1 个代理运行（由 `ScheduleEngine` 的信号量控制）

## 数据流

### 消息处理全链路

```
用户消息 → WeixinBot._handle_message()
           │
           ├─ 设置 trace_id
           ├─ 调度命令拦截（定时任务列表/删除/创建）
           ├─ 快速路径（聊天/状态/管理/回复）
           └─ MessageRouter.process_message()
                    │
                    ├─ 模拟器重启命令
                    ├─ 观察学习命令 (/record /done /stop)
                    ├─ 创建指南命令 (/save)
                    ├─ 简单单任务 → 直接委派
                    ├─ 多游戏 → 拆分批处理
                    ├─ 管家 LLM（模糊场景）
                    └─ 自动出队（队列任务启动）
                              │
                              ▼
                       TerraAgent.run()
                         │
                         ├─ _setup_task_context() → 技能搜索 + 意图分类
                         ├─ _run_loop() → LLM ↔ 工具 ↔ 屏幕
                         └─ execution_logger.log() → 持久化记录
```

### 三级提示词架构

```
[稳定层]  个性 + 通用规则 + 游戏特定追加  ← 可缓存（不随迭代变化）
[上下文层] 匹配的技能内容                   ← 随技能变化
[易变层]   状态快照 + 当前时间              ← 每轮更新（永不缓存）
```

### 屏幕注入流水线

```
screencap → dHash 去重 → OCR 文字提取 → 注入 LLM 上下文
              ↓
        缓存命中 → 跳过 OCR（节省 ~2s/轮）
```

## 游戏插件系统

### 插件接口

所有游戏通过 `GamePlugin` ABC 接入：

```python
class GamePlugin(ABC):
    manifest: GameManifest  # 游戏元数据
    classify_task(text) → str
    get_system_prompt_append() → str
    get_task_verbs() → list[str]
    get_safety_overrides() → dict
    get_daily_tasks() → list[dict]
    register_intelligence_tools()
    register_game_tools()
```

### GameManifest 结构

```python
@dataclass(frozen=True)
class GameManifest:
    id: str                      # "arknights"
    name: str                    # "明日方舟"
    keywords: list[str]          # 意图检测关键词
    dangerous_keywords: list[str] # 危险操作拦截
    task_verbs: list[str]        # 确定性分发的动作动词
    task_keywords: dict          # 关键词 → 任务类型
    android_packages: list[str]  # ADB 自动发现
    system_prompt_append: str    # 游戏特定 UI 指南
    ...
```

### GameRegistry

线程安全的全局注册表，支持：
- 游戏检测（通过关键词匹配）
- 多游戏消歧
- 调度意图分类
- 危险关键词查询

## DI 容器

`src/container.py` — `AppContainer` 不可变 dataclass：

```
AppContainer
├── config (Config)
├── memory_db (MemoryDB)
├── skill_db (SkillDB)
├── game_registry (GameRegistry)
├── emulator_manager (EmulatorManager)
├── schedule_db (ScheduleDB)
├── sched_engine (ScheduleEngine)
├── MemoryHintService
├── CompressionService
├── ExecutionLogger
└── ReviewTrigger
```

惰性初始化，通过模块级函数 `get_container()` 访问。

## 数据库设计

4 个 SQLite 数据库（WAL 模式，FTS5 全文搜索）：

| 数据库 | 文件 | 用途 |
|--------|------|------|
| memory_db | data/terra.db | 记忆存储 + 任务执行记录 + 注入反馈 |
| skill_db | data/terra.db | 技能索引（FTS5 搜索） |
| history_db | data/terra.db | 执行历史（日志级） |
| schedule_db | data/scheduler.db | 定时任务持久化 + 任务队列 |

### 记忆系统

```
用户操作 → 记忆记录（FTS5 中文分词）
         → dHash 屏幕指纹关联
         → 语义重排序
         → 智能注入到 LLM 上下文
         → 反馈追踪（有帮助/无帮助）
         → 自动淘汰（低成功率/过期）
```

## 工具系统

工具在导入时自注册到全局 `ToolRegistry`：

```python
# src/tools/adb_control.py
ToolRegistry.register(
    name="adb_tap",
    description="点击屏幕上的文字...",
    fn=_adb_tap_dispatch,
    game=None,  # None = 通用工具
)
```

工具执行通过 `dispatch()` 统一入口，包含：
- 线程上下文传递（game/agent 上下文）
- 安全检查（危险关键词 + 日限）
- 计数器提交（仅成功操作）

## 安全机制

### 运行时防护

| 防护层 | 说明 |
|--------|------|
| 危险操作拦截 | 源石/合成玉等付费货币操作需用户确认 |
| 日操作限制 | 每日最多 500 次操作，成功才计数 |
| 循环检测 | 重复/突发/滚动闭环检测 |
| 暗屏检测 | 多级亮度感知 + 加载中关键词 |
| 资源消耗检测 | 确认面板 + 资源关键词 + 安全屏排除 |
| 停滞检测 | 操作无效 → 自动切换策略 |
| 断路器 | LLM/ADB/OCR 故障 → 快速终止，防止雪崩 |

### 传输安全

- HTTPS 通信（SSL 证书验证）
- 微信令牌 AES-256-GCM 加密存储
- 环境变量隔离敏感配置

## 通信协议

### 微信 iLink

- 协议：iLink Bot API（`https://ilinkai.weixin.qq.com`）
- 认证：二维码扫码登录 + 令牌续期
- 消息格式：文本 / 图片（base64 JPEG）
- 会话：长轮询接收 + RESTful 发送
- 图片限制：500KB，超过用纯文本替代

## 可观测性

- **Trace ID**：全链路请求追踪（weixin → concierge → agent → LLM → tools）
- **Agent Tag**：多代理日志标签（游戏+设备标识）
- **日志级别**：标准 Python logging，支持 Rich 终端输出
- **执行记录**：每任务 JSON 快照保存到 `data/logs/`
