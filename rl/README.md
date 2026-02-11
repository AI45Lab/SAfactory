# Quick Start

bash run_slime_generator.sh
then bash run_buffer_server.sh

# AIEvoBox RL Rollout Buffer

基于 Slime 的 rollout buffer 实现，用于 AIEvoBox 环境的强化学习训练。

## 目录结构

```
rl/
├── slime_generator.py              # Slime 客户端，提供 generate_rollout()
├── buffer_server.py                # Buffer Server，管理数据分组和子进程
├── llm_proxy.py                    # LLM Proxy，代理请求并记录轨迹
├── run_buffer_server.sh            # Buffer Server 启动脚本
├── run_slime_generator.sh          # Slime 训练启动脚本
├── .env                            # 环境变量配置
├── .env.example                    # 环境变量模板
├── dummy.jsonl                     # 占位数据文件（Slime 需要）
├── mask/
│   ├── trajectory_mask_builder.py  # 轨迹 Mask 构建器
│   └── test_trajectory_mask_builder.py
└── utils/
    ├── env_utils.py                # 环境变量工具
    └── metrics.py                  # W&B 指标记录器
```

## 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Slime Training (Ray)                            │
│  slime_generator.py                                                     │
│  ├── generate_rollout()      # 主入口，被 Slime 框架调用                  │
│  ├── start_rollout()         # 初始化 rollout 流程                       │
│  ├── get_rollout_data()      # 从 Buffer Server 获取分组数据             │
│  └── query_trajectory()      # 从 LLM Proxy 查询 tokens 和 mask          │
└──────────┬──────────────────────────────────┬───────────────────────────┘
           │                                  │
           │ /start_rollout                   │ /get_tokens
           │ /get_rollout_data                │ /get_trajectory_mask
           ▼                                  ▼
┌─────────────────────┐              ┌─────────────────────┐
│   Buffer Server     │──启动──────▶ │    LLM Proxy        │
│   :18889            │              │    :18890           │
├─────────────────────┤              ├─────────────────────┤
│ • 启动 LLM Proxy    │              │ • 代理 LLM 请求      │
│ • 启动 AIEvoBox     │              │ • 记录轨迹 mask      │
│ • 数据分组返回      │              │ • Token 前缀复用     │
└─────────────────────┘              └─────────┬───────────┘
                                               │
                                               │ /generate
                                               ▼
                                     ┌─────────────────────┐
                                     │   sglang 引擎       │
                                     └─────────────────────┘
```

## 核心组件

### 1. Slime Generator (`slime_generator.py`)

Slime 训练框架的客户端接口，提供 `generate_rollout()` 函数。

**核心功能**：
- `generate_rollout()`: 主入口，被 Slime 框架调用获取训练样本
- `start_rollout()`: 首次调用时初始化 Buffer Server 和 LLM Proxy
- `get_rollout_data()`: 从 Buffer Server 获取按 group_id 分组的数据
- `query_trajectory()`: 从 LLM Proxy 获取 tokens 和 response_mask
- `build_loss_mask_from_response_mask()`: 将 response_mask 转换为 loss_mask

**特性**：
- 异步 I/O 操作，支持高并发
- 权重版本过滤（`RL_OFF_BY_N` 控制，防止使用过时数据）
- W&B 指标集成（通过 `MetricsRecorder`）

### 2. Buffer Server (`buffer_server.py`)

中心化服务，管理数据收集和子进程。

**核心功能**：
- `/start_rollout`: 启动 LLM Proxy 和 AIEvoBox Runner 子进程
- `/get_rollout_data`: 返回按 group_id 分组的完成数据
- `/health`: 健康检查

**特性**：
- 游标分页查询，避免重复处理
- 子进程生命周期管理

### 3. LLM Proxy (`llm_proxy.py`)

代理 LLM 请求，同时记录完整的对话轨迹用于训练。

**核心功能**：
- `/init`: 初始化 tokenizer 和远程 sglang 引擎 URL
- `/v1/{session_id}/chat/completions`: 代理 chat completion 请求
- `/get_tokens`: 获取 session 的 tokens 和 response_mask
- `/get_trajectory_mask`: 获取轨迹 mask（兼容接口）

**特性**：
- Token 前缀复用优化（减少 tokenization 开销）
- 多轮对话轨迹记录
- 支持采样参数（temperature, top_p 等）

### 4. Trajectory Mask Builder (`mask/trajectory_mask_builder.py`)

构建和管理多轮对话的轨迹 mask。

**核心功能**：
- `prepare_generate_input()`: 准备 /generate 输入，复用历史 token，并维护多模态 image_data
- `add_assistant_message()`: 记录生成轮次（tokens, response_mask, logprobs, image_data）
- `query_training_info()`: 查询训练所需信息（tokens/mask/logprobs/image_data）

**特性**：
- 支持 `<think>...</think>` 标签的可选前缀匹配
- 字符级匹配转 token 级 mask

## 服务端口

| 服务 | 默认端口 | 环境变量 |
|------|----------|----------|
| Buffer Server | 18889 | `BUFFER_SERVER_PORT` |
| LLM Proxy | 18890 | `LLM_PROXY_PORT` |

## 启动流程

### 1. 启动 Buffer Server（终端 1）

```bash
cd /root/AIEvoBox/rl
./run_buffer_server.sh
```

### 2. 启动 Slime 训练（终端 2）

```bash
cd /root/AIEvoBox/rl
./run_slime_generator.sh
```

训练开始后，Buffer Server 会自动：
1. 启动 LLM Proxy
2. 启动 AIEvoBox Runner

### 启动脚本说明

| 脚本 | 说明 |
|------|------|
| `run_buffer_server.sh` | 启动 Buffer Server |
| `run_slime_generator.sh` | 启动 Slime 训练（包含 Ray、sglang 等） |
| `.env` | 环境变量配置 |
| `dummy.jsonl` | 占位数据文件（Slime 需要但不使用） |

## 日志

日志文件存储在 `$AIEVOBOX_ROOT/logs/` 目录下：

| 文件 | 说明 |
|------|------|
| `buffer_server.log` | Buffer Server 日志 |
| `llm_proxy.log` | LLM Proxy 日志 |
| `aievobox_runner.log` | AIEvoBox Runner 日志 |

日志特性：
- 使用 `RotatingFileHandler`，单个文件最大 50MB，保留 5 个备份
- 文件日志级别：DEBUG（详细信息）
- 控制台日志级别：INFO（关键信息）

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AIEVOBOX_ROOT` | `/root/AIEvoBox` | AIEvoBox 根目录 |
| `AIEVOBOX_DB_URL` | `sqlite:///rl/rl.db` | 数据库 URL |
| `RL_ENV_NUM` | `5` | 每个样本的环境实例数 |
| `RL_MAX_STEPS` | `10` | 每个 episode 最大步数 |
| `RL_API_KEY` | - | LLM API Key |
| `RL_MODEL` | - | LLM 模型名称 |
| `ROLLOUT_MAX_WORKERS` | `200` | 并发 worker 数量 |
| `BUFFER_SERVER_HOST` | `127.0.0.1` | Buffer Server 连接地址（服务固定监听 0.0.0.0） |
| `BUFFER_SERVER_PORT` | `18889` | Buffer Server 端口 |
| `LLM_PROXY_HOST` | `127.0.0.1` | LLM Proxy 连接地址（服务固定监听 0.0.0.0） |
| `LLM_PROXY_PORT` | `18890` | LLM Proxy 端口 |
| `LLM_MAX_LENGTH` | `4608` | 最大 token 长度 |
| `LLM_TEMPERATURE` | `1.0` | 采样温度 |
| `RL_OFF_BY_N` | `0` | 允许的最大权重版本差（0=只用当前版本数据） |

## API 端点

### Buffer Server (:18889)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/start_rollout` | POST | 启动 rollout，触发 LLM Proxy 和 AIEvoBox |
| `/get_rollout_data` | POST | 获取已完成的分组数据 |

### LLM Proxy (:18890)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/init` | POST | 初始化 tokenizer 和远程引擎 URL |
| `/v1/{session_id}/chat/completions` | POST | 代理 LLM 请求 |
| `/get_trajectory_mask` | POST | 获取轨迹 mask |
| `/health` | GET | 健康检查 |

## 数据流

```
[1] Slime 首次调用 generate_rollout()
         │
         ▼
    start_rollout() ──▶ Buffer Server /start_rollout
         │                    │
         │                    ├──▶ 启动 LLM Proxy 子进程
         │                    │         └──▶ 初始化 TrajectoryMaskBuilder
         │                    │
         │                    └──▶ 启动 AIEvoBox launcher.py 子进程
         │                              └──▶ 运行 Interactor 环境
         │                                        │
         ▼                                        ▼
[2] Interactor 与环境交互              LLM Proxy 代理请求
         │                                        │
         │                                        ├──▶ 转发到 sglang 引擎
         │                                        └──▶ 记录 tokens + mask
         │
         ▼
[3] Slime 循环调用 get_rollout_data()
         │
         ├──▶ Buffer Server 查询已完成数据
         │         └──▶ 按 group_id 分组返回
         │
         ├──▶ 过滤权重版本（RL_OFF_BY_N）
         │
         └──▶ query_trajectory() 获取每个样本的 tokens 和 mask
                   │
                   └──▶ LLM Proxy /get_tokens
                            └──▶ 返回 (tokens, response_mask)

[4] 构建训练样本
         │
         ├──▶ build_loss_mask_from_response_mask()
         │         └──▶ 转换为 (token_ids, loss_mask, response_length)
         │
         └──▶ 返回 Slime Sample 对象用于 GRPO 训练
```

**关键说明**：
- reward 归一化在 Slime 训练端完成（GRPO 算法，可通过 `--disable-rewards-normalization` 控制）
- 权重版本过滤确保只使用当前或近期 checkpoint 产生的数据

## Mask 实现

`TrajectoryMaskBuilder` 维护 token 级的轨迹 mask：

- **response_mask**: 0=上下文 token，1=生成的 token
- **add_assistant_message()**: 记录每轮对话的完整 token 序列、response_mask、logprobs 和 image_data
- **query_training_info()**: 查询训练所需信息（tokens/mask/logprobs/image_data）
- **特性**: 支持 `<think>...</think>` 可选前缀匹配，多轮对话复用历史 token

## 性能优化

1. **Token 前缀复用**: LLM Proxy 通过前缀匹配复用历史 token，避免重复 tokenization
2. **异步 I/O**: Slime Generator 使用 async/await 模式支持高并发
3. **连接池**: httpx 客户端维护持久连接
4. **游标分页**: Buffer Server 使用自增 ID 游标分页，避免重复处理

## 与 Slime 原实现的差异

| 功能 | Slime 原实现 | 本实现 |
|------|-------------|--------|
| Generator | `BaseGenerator` 多进程 | `Interactor` 异步 |
| 数据存储 | 内存队列 | SQLite 数据库 |
| LLM 调用 | 直接调用 | 通过 LLM Proxy 代理 |
| Mask 计算 | `MultiTurnLossMaskGenerator` 本地 | `TrajectoryMaskBuilder` 服务端 |
| 进程模型 | Generator 内嵌 | 独立子进程 |

## 调试

**日志文件** (位于 `$AIEVOBOX_ROOT/logs/`)：
- `buffer_server.log`: Buffer Server 日志
- `llm_proxy.log`: LLM Proxy 日志
- `train_*.log`: 调试训练数据（tokens, masks, rewards）

**调试训练样本**：
Slime Generator 会将每个训练样本写入 `train_*.log`，包含：
- tokens 序列
- loss_mask
- reward 值
