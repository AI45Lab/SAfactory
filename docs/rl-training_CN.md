# 强化学习训练

Safactory 包含一个面向 Slime 风格训练的 RL bridge：

- `rl/buffer_server.py` 启动 `launcher.py`，读取已完成的 trainable 行，按 `group_id` 聚合样本，并通过 `/get_rollout_data` 输出 batch。
- `rl/llm_proxy.py` 由 `slime_generator` 托管，提供类 OpenAI 接口用于在线 rollout generation。
- `rl/examples/*` 中包含按任务划分的脚本和 `env.sh`。

RL bridge 仍使用历史的 `AIEVOBOX_*` 环境变量前缀。部分示例脚本还保留了旧变量名，使用 v2 launcher 前请先核对下面的变量。

## 架构

```text
Safactory runtime  <--- 由 launcher.py / rl/buffer_server.py 启动
  |
  | 带 session 的 rollout 模型请求
  v
Gateway
  |
  | X-Safactory-Session-Id
  v
LLM proxy  <--- 由 Slime generator 托管
  |
  | 生成请求
  v
SGLang rollout engine

Gateway / evaluator
  |
  | session_steps 中带 reward 的可训练记录
  v
Buffer Server /get_rollout_data
  |
  | grouped rollout samples
  v
Slime trainer
```

对于 v2 adapter，launcher 仍需要有效的 gateway 兼容模型路径和 runtime start config。扩容前请先在 `logs/buffer_server.log` 中确认生成的 launcher 命令。

## 当前集成状态

基于当前代码：

- `buffer_server.py` 会将 `AIEVOBOX_AGENT_CONFIG` 映射为 `launcher.py --agent-config`。
- 如果未设置 `AIEVOBOX_AGENT_CONFIG`，则将 `AIEVOBOX_AGENT_ROOT` 映射为 `--agent-root`。
- 将 `AIEVOBOX_AGENT_START_CONFIG` 映射为 `--agent-start-config`；未设置时会尽量推导同目录的 `_start.yaml`。
- 将 `RL_MODEL` 映射为 `launcher.py --llm-model`。
- 必须设置 `AIEVOBOX_GATEWAY_BASE_URL`，并映射为 `launcher.py --gateway-base-url`；不再使用 LLM proxy 作为 gateway 回退。
- `AIEVOBOX_ENABLE_EVALUATION=1` 会启用 evaluator flow；`AIEVOBOX_EVALUATION_CONFIG` 可指定 evaluator 配置。

因此，仓库中的 RL examples 更适合作为模板，而不是直接可用的 v2 命令。

## 示例脚本

脚本位于 `rl/examples/`：

| 目录 | 脚本 |
|------|------|
| `rl/examples/search` | `env.sh`, `run_buffer_server.sh`, `run_slime_generator.sh` |
| `rl/examples/deepeyes` | `env.sh`, `run_buffer_server.sh`, `run_slime_generator.sh` |
| `rl/examples/geo3k_vl` | `env.sh`, `run_buffer_server.sh`, `run_slime_generator.sh` |
| `rl/examples/math500` | `env.sh`, `run_buffer_server.sh`, `run_slime_generator_opd_sglang.sh` |

从示例目录运行：

```bash
cd rl/examples/math500
./run_buffer_server.sh
```

Slime generator 进程需要按对应示例脚本和 Slime 文档单独启动。

## 关键变量

Launcher 和数据：

| 变量 | 用作 | 说明 |
|------|------|------|
| `AIEVOBOX_ROOT` | launcher 路径和 Python path | Safactory 仓库根目录。 |
| `STORAGE_TYPE` | `--storage-type` | `sqlite` 或 `cloud`。 |
| `AIEVOBOX_DB_URL` | `--db-path` | Launcher 和 Buffer Server 使用的 SQLite URI。 |
| `AIEVOBOX_AGENT_CONFIG` | `--agent-config` | 单个 v2 agent config YAML。 |
| `AIEVOBOX_AGENT_ROOT` | `--agent-root` | 未设置 `AIEVOBOX_AGENT_CONFIG` 时的 agent config 根目录。 |
| `AIEVOBOX_GATEWAY_BASE_URL` | `--gateway-base-url` | 必填的 Gateway session root；不会回退到 LLM proxy。 |
| `AIEVOBOX_ENABLE_EVALUATION` | `--enable-evaluation` | 设为 `1`、`true`、`yes` 或 `on` 时运行 evaluator 并提交可训练 reward。 |
| `AIEVOBOX_EVALUATION_CONFIG` | `--evaluation-config` | 可选的 evaluator runtime YAML。 |
| `RL_MODEL` | `--llm-model` | Gateway route key 或 rollout model 标识。 |
| `LLM_TEMPERATURE` | `--llm-temperature` | 采样温度。 |
| `AIEVOBOX_MAX_STEPS` | `--max-steps` | Episode 步数限制。 |
| `AIEVOBOX_POOL_SIZE` | `--pool-size` | Launcher pool size。 |

RL grouping：

| 变量 | 说明 |
|------|------|
| `RL_GROUP_SIZE` | 每个 prompt 的采样数，会作为 `--rl-group-size` 并用于 Buffer Server grouping。 |
| `RL_EPOCH` | Rollout epoch 扩展，会作为 `--rl-epoch`。 |
| `RL_OFF_BY_N` | 允许的最大权重版本滞后，由训练侧脚本使用。 |
| `SLIME_GLOBAL_BATCH_SIZE` | Slime 全局 batch size。 |
| `SLIME_N_SAMPLES_PER_PROMPT` | 通常设置为 `RL_GROUP_SIZE`。 |

服务：

| 变量 | 说明 |
|------|------|
| `BUFFER_SERVER_HOST` | Slime 访问 Buffer Server 使用的 host。 |
| `BUFFER_SERVER_PORT` | Buffer Server 端口，常用 `18889`。 |
| `LLM_PROXY_HOST` | Gateway 访问 Slime-hosted LLM proxy 使用的 host。 |
| `LLM_PROXY_PORT` | LLM proxy 端口，常用 `18890`。 |

## Buffer Server API

| 端点 | 用途 |
|------|------|
| `POST /start_rollout` | 如果当前没有 launcher 子进程，则启动一次 rollout。 |
| `POST /get_rollout_data` | 返回从上次 cursor 之后读取到的 grouped rollout samples。 |
| `GET /health` | 返回 Buffer Server 健康状态、launcher 进程状态和 DataManager 初始化状态。 |

`/get_rollout_data` 返回给 Slime 的 item 形态：

```json
{
  "uid": "...",
  "instance_id": "<group_id>",
  "messages": [],
  "reward": 0.0,
  "extra_info": {
    "session_id": "...",
    "env_id": "...",
    "group_id": "...",
    "weight_version": 0,
    "truncated": false
  }
}
```

## 建议的 V2 流程

1. 先在 RL 外部用 `launcher.py` 和 gateway 跑通 adapter。
2. 确认 `session_steps` 中出现期望 `job_id` 的 trainable 行。
3. 设置 `AIEVOBOX_AGENT_CONFIG`、`AIEVOBOX_DB_URL`、`AIEVOBOX_GATEWAY_BASE_URL`、`RL_MODEL`、`RL_GROUP_SIZE` 和 `AIEVOBOX_POOL_SIZE`。
4. 设置 `AIEVOBOX_AGENT_START_CONFIG`；如果环境依赖 evaluator 提交 reward，同时启用 evaluation。
5. 启动 Slime generator 和 Buffer Server。
6. 由 trainer 或脚本调用 `/start_rollout`。
7. 观察 `logs/buffer_server.log`、`logs/main.log`、gateway 日志和 Slime 日志。

## 排错

| 现象 | 检查项 |
|------|--------|
| Buffer Server 启动 launcher 时仍像旧 `--env-config` 流程 | 使用 `AIEVOBOX_AGENT_CONFIG` 或 `AIEVOBOX_AGENT_ROOT`，更新旧示例 `env.sh`。 |
| Launcher 因缺少 start config 失败 | 设置 `AIEVOBOX_AGENT_START_CONFIG`，或采用同目录 `<name>_start.yaml` 命名供 Buffer Server 自动推导。 |
| Launcher 提示缺少 gateway URL | 将 `AIEVOBOX_GATEWAY_BASE_URL` 设置为 Gateway session root；不要指向 LLM proxy。 |
| `/get_rollout_data` 没有样本 | 检查 `session_steps` 是否有 `is_trainable = 1`、`job_id` 是否正确、`group_id` 数量是否匹配。 |
| Group 一直不 ready | `RL_GROUP_SIZE` 大于该 `group_id` 下已完成样本数。 |
| Weight-version 解析失败 | 如果训练策略依赖版本，请确保 `env_state.weight_version` 存在。 |
