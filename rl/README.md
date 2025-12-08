# AIEvoBox RL Rollout Buffer

基于 slime 的 rollout buffer 实现，用于 AIEvoBox 环境的强化学习训练。

## 架构

```
┌─────────────┐    start_rollout()    ┌─────────────────┐
│   Slime     │ ────────────────────▶ │  Buffer Server  │
│   训练端    │                       │  :8889          │
└──────┬──────┘                       └────────┬────────┘
       │                                       │
       │ get_rollout_data()                    │ 1. 启动 LLM Proxy
       │                                       │ 2. 启动 AIEvoBox Runner
       │                                       ▼
       │                              ┌─────────────────┐
       │                              │  LLM Proxy      │
       │ get_trajectory_mask()        │  :8890          │
       │◀─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│                 │
       │                              └────────┬────────┘
       │                                       │
       │                                       │ 转发到真实 LLM
       │                                       ▼
       │                              ┌─────────────────┐
       │                              │  sglang 引擎    │
       │                              └─────────────────┘
       │
       │         ┌───────────────┐
       └─────────│   Database    │◀──── AIEvoBox Runner
      查询 DB    │  (SQLite)     │     (Interactor 写入)
                 └───────────────┘
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `rollout_buffer_slime.py` | Slime 训练端客户端，提供 `generate_rollout()` 函数 |
| `buffer_server.py` | Buffer Server，管理数据分组和子进程启动 |
| `llm_proxy.py` | LLM Proxy，代理 LLM 请求并记录轨迹 mask |
| `aievobox_runner.py` | AIEvoBox 启动入口，运行 Interactor |
| `mask/trajectory_mask_builder.py` | 轨迹 Mask 构建器 |

## 服务端口

| 服务 | 默认端口 | 环境变量 |
|------|----------|----------|
| Buffer Server | 8889 | `ROLLBUF_PORT` |
| LLM Proxy | 8890 | `LLM_PROXY_PORT` |

## 启动流程

### 1. 启动 Buffer Server（终端 1）

```bash
cd /root/AIEvoBox/rl
./run_buffer_server.sh
```

### 2. 启动 Slime 训练（终端 2）

```bash
cd /root/AIEvoBox/rl
./run_slime.sh
```

训练开始后，Buffer Server 会自动：
1. 启动 LLM Proxy
2. 启动 AIEvoBox Runner

### 启动脚本说明

| 脚本 | 说明 |
|------|------|
| `run_buffer_server.sh` | 启动 Buffer Server |
| `run_slime.sh` | 启动 Slime 训练（包含 Ray、sglang 等） |
| `.env` | 环境变量配置 |
| `dummy.jsonl` | 占位数据文件（slime 需要但不使用） |

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
| `AIEVOBOX_DB_URL` | `sqlite:////root/AIEvoBox/rollout.db` | 数据库 URL |
| `LLM_PROXY_URL` | `http://127.0.0.1:8890` | LLM Proxy 地址 |
| `ROLLOUT_BUFFER_URL` | `http://127.0.0.1:8889` | Buffer Server 地址 |
| `ROLLOUT_BATCH_SIZE` | `128` | 并发 worker 数量 |
| `ROLLOUT_MAX_STEPS` | `10` | 每个 episode 最大步数 |
| `NUM_REPEAT_PER_SAMPLE` | `1` | 每个环境的 episode 数量 |

## API 端点

### Buffer Server (:8889)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/start_rollout` | POST | 启动 rollout，触发 LLM Proxy 和 AIEvoBox |
| `/get_rollout_data` | POST | 获取已完成的分组数据 |
| `/buffer/write` | POST | 写入数据（兼容性保留） |

### LLM Proxy (:8890)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/init` | POST | 初始化 tokenizer 和远程引擎 URL |
| `/v1/{session_id}/chat/completions` | POST | 代理 LLM 请求 |
| `/get_trajectory_mask` | POST | 获取轨迹 mask |
| `/health` | GET | 健康检查 |

## 数据流

1. **Slime 调用 `start_rollout()`**
   - Buffer Server 启动 LLM Proxy 和 AIEvoBox Runner

2. **AIEvoBox Runner 执行 rollout**
   - Interactor 通过 LLM Proxy 调用 LLM
   - LLM Proxy 记录每轮对话的 mask
   - Interactor 将完成的 step 写入数据库

3. **Slime 调用 `get_rollout_data()`**
   - Buffer Server 从数据库查询已完成的 steps
   - 按 `instance_id` 分组，达到 `group_size` 后返回
   - 返回前进行 reward 归一化（GRPO 风格）

4. **Slime 获取 mask**
   - 调用 LLM Proxy 的 `/get_trajectory_mask`
   - 使用 `tokenize_with_char_mask_ranges()` 转换为 token 级 mask

## Mask 实现

`TrajectoryMaskBuilder` 维护字符级的轨迹 mask：

- **save()**: 记录每轮对话，assistant 输出部分 mask=1
- **query()**: 查询给定对话字符串的 mask
- **特性**: 支持 `<think>...</think>` 可选前缀匹配，多轮对话复用历史 mask

## 与 Slime 原实现的差异

| 功能 | Slime 原实现 | 本实现 |
|------|-------------|--------|
| Generator | `BaseGenerator` 多进程 | `Interactor` 异步 |
| 数据存储 | 内存队列 | SQLite 数据库 |
| LLM 调用 | 直接调用 | 通过 LLM Proxy 代理 |
| Mask 计算 | `MultiTurnLossMaskGenerator` 本地 | `TrajectoryMaskBuilder` 服务端 |
| 进程模型 | Generator 内嵌 | 独立子进程 |
