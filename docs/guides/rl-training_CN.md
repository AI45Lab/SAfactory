# 强化学习训练

SAfactory 的 RL bridge 面向 Slime 风格训练，把环境 rollout、Gateway 轨迹记录、rule evaluator reward 和训练侧数据消费连接起来：

- `rl/buffer_server.py` 启动 `launcher.py`，读取已完成的 trainable 行，按 `group_id` 聚合样本，并通过 `/get_rollout_data` 输出 batch。
- `rl/llm_proxy.py` 由 `slime_generator` 托管，提供类 OpenAI 接口用于在线 rollout generation。
- `rl/run_slime_generator.sh` 启动 Slime generator、训练进程和 rollout engine。
- `rl/run_buffer_server.sh` 启动 Buffer Server，并由 Buffer Server 拉起 SAfactory rollout。

## 架构

```text
SAfactory runtime  <--- 由 launcher.py / rl/buffer_server.py 启动
  |
  | 带 session 的 rollout 模型请求
  v
Gateway  <--- 默认由 Buffer Server 自动启动
  |
  | route key 来自 RL_MODEL
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

## 前置检查

先在 RL 外部跑通 `my_env` 的最小评测。这一步用于确认环境镜像、dataset、Gateway route、存储和 `rule_evaluator.py` 都能正常工作：

```bash
python launcher.py \
  --mode docker \
  --agent-config env/my_env/my_env_config.yaml \
  --agent-start-config env/my_env/my_env_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model my_env_model \
  --enable-evaluation \
  --db-path sqlite://env_trajs.db \
  --job-id my-env-docker-smoke \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 10
```

如果你的环境不需要 evaluator，可以去掉 `--enable-evaluation`；但用于 RL 的轨迹通常需要最终 reward 和 `is_trainable` 行，因此建议明确实现 evaluator 并先验证 reward 写入。

## 创建 `env.sh`

为环境创建一个独立的 RL 配置文件。推荐路径：

```bash
mkdir -p rl/examples/my_env
touch rl/examples/my_env/env.sh
```

最小配置如下：

```bash
#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

export AIEVOBOX_EXAMPLE_NAME="${AIEVOBOX_EXAMPLE_NAME:-my_env}"
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-${REPO_ROOT}}"
export AIEVOBOX_MODE="${AIEVOBOX_MODE:-docker}"
export STORAGE_TYPE="${STORAGE_TYPE:-sqlite}"
export AIEVOBOX_DB_URL="${AIEVOBOX_DB_URL:-sqlite:///${AIEVOBOX_ROOT}/rl/examples/my_env/my_env.db}"

export AIEVOBOX_AGENT_CONFIG="${AIEVOBOX_AGENT_CONFIG:-${AIEVOBOX_ROOT}/env/my_env/my_env_config.yaml}"
export AIEVOBOX_AGENT_START_CONFIG="${AIEVOBOX_AGENT_START_CONFIG:-${AIEVOBOX_ROOT}/env/my_env/my_env_start.yaml}"
export AIEVOBOX_ENABLE_EVALUATION="${AIEVOBOX_ENABLE_EVALUATION:-1}"
export AIEVOBOX_MAX_STEPS="${AIEVOBOX_MAX_STEPS:-10}"
export AIEVOBOX_POOL_SIZE="${AIEVOBOX_POOL_SIZE:-2}"

export RL_MODEL="${RL_MODEL:-my_env_model}"
export RL_GROUP_SIZE="${RL_GROUP_SIZE:-2}"
export RL_GLOBAL_BATCH_SIZE="${RL_GLOBAL_BATCH_SIZE:-64}"
export RL_ROLLOUT_GROUP_BATCH_SIZE="${RL_ROLLOUT_GROUP_BATCH_SIZE:-32}"
export RL_EPOCH="${RL_EPOCH:-1}"
export RL_OFF_BY_N="${RL_OFF_BY_N:-0}"

export BUFFER_SERVER_HOST="${BUFFER_SERVER_HOST:-127.0.0.1}"
export BUFFER_SERVER_PORT="${BUFFER_SERVER_PORT:-18889}"
export LLM_PROXY_HOST="${LLM_PROXY_HOST:-127.0.0.1}"
export LLM_PROXY_PORT="${LLM_PROXY_PORT:-18890}"
export AIEVOBOX_GATEWAY_HOST="${AIEVOBOX_GATEWAY_HOST:-127.0.0.1}"
export AIEVOBOX_GATEWAY_PORT="${AIEVOBOX_GATEWAY_PORT:-8000}"
export AIEVOBOX_GATEWAY_BASE_URL="${AIEVOBOX_GATEWAY_BASE_URL:-http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions}"

export LOG_ROOT="${LOG_ROOT:-${AIEVOBOX_ROOT}/logs}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export RAY_BIN="${RAY_BIN:-ray}"

export SLIME_HOME="${SLIME_HOME:-/path/to/slime}"
export MEGATRON_HOME="${MEGATRON_HOME:-/path/to/Megatron-LM}"
export HF_CKPT_DIR="${HF_CKPT_DIR:-/path/to/hf-checkpoint}"
export LOAD_DIR="${LOAD_DIR:-${HF_CKPT_DIR}}"
export SAVE_DIR="${SAVE_DIR:-/path/to/save/checkpoints}"

export NUM_GPUS="${NUM_GPUS:-4}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-1}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-3}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"

export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-${SLIME_HOME}/train.py}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_HOME}/scripts/models/qwen3-1.7B.sh}"
export ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-rl.slime_generator.generate_rollout}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
```

`RL_MODEL` 是 Gateway route key，也是 Buffer Server 自动生成 Gateway 配置时使用的 route 名。使用自动 Gateway 时，它会被路由到 Slime generator 托管的 LLM proxy；使用外部 Gateway 时，外部配置也必须提供同名 route。

## 启动训练

按根目录 README 的方式，在仓库根目录启动两个进程。

终端 1 启动 Slime generator：

```bash
RL_ENV_SH=rl/examples/my_env/env.sh bash rl/run_slime_generator.sh
```

终端 2 启动 Buffer Server：

```bash
RL_ENV_SH=rl/examples/my_env/env.sh bash rl/run_buffer_server.sh
```

也可以使用 `--env` 参数：

```bash
bash rl/run_slime_generator.sh --env rl/examples/my_env/env.sh
bash rl/run_buffer_server.sh --env rl/examples/my_env/env.sh
```

Buffer Server 默认会自动启动 Gateway，生成 `logs/gateway.rl.generated.yaml`，把 `RL_MODEL` 路由到 Slime 托管的 LLM proxy，启动 Docker rollout 采集，并通过 `/get_rollout_data` 提供已完成的 group。使用自动启动时，请先停止同一端口上手动启动的 Gateway。

只有当你已经手动启动了外部 Gateway，且它同时满足以下条件时，才设置 `AIEVOBOX_GATEWAY_AUTOSTART=0`：

- `AIEVOBOX_GATEWAY_BASE_URL` 指向该 Gateway 的 session root。
- Gateway 的 `llm_routes` 中存在 `RL_MODEL` 对应 route。
- Gateway 的存储配置与 `AIEVOBOX_DB_URL` 一致。

## 当前集成状态

只有在你已经手动启动了外部 Gateway，并且它使用相同存储后端和 route key 时，才设置 `AIEVOBOX_GATEWAY_AUTOSTART=0`。

- `buffer_server.py` 会将 `AIEVOBOX_AGENT_CONFIG` 映射为 `launcher.py --agent-config`。
- 如果未设置 `AIEVOBOX_AGENT_CONFIG`，则将 `AIEVOBOX_AGENT_ROOT` 映射为 `--agent-root`。
- 将 `AIEVOBOX_AGENT_START_CONFIG` 映射为 `--agent-start-config`；未设置时会尽量推导同目录的 `_start.yaml`。
- 将 `AIEVOBOX_DB_URL` 映射为 `launcher.py --db-path`。
- 将 `AIEVOBOX_GATEWAY_BASE_URL` 映射为 `launcher.py --gateway-base-url`。
- 将 `RL_MODEL` 映射为 `launcher.py --llm-model`。
- `AIEVOBOX_ENABLE_EVALUATION=1` 会启用 evaluator flow。

这意味着 RL 训练不需要为每个环境复制启动脚本；通常只需要准备自己的 `rl/examples/my_env/env.sh`。

## 关键变量

Launcher 和数据：

| 变量 | 用作 | 说明 |
|------|------|------|
| `AIEVOBOX_ROOT` | launcher 路径和 Python path | SAfactory 仓库根目录。 |
| `AIEVOBOX_MODE` | `--mode` | `docker`、`rjob` 或 `sandbox`。 |
| `STORAGE_TYPE` | `--storage-type` | `sqlite` 或 `cloud`。 |
| `AIEVOBOX_DB_URL` | `--db-path` | Launcher、Gateway 和 Buffer Server 使用的 SQLite URI。 |
| `AIEVOBOX_AGENT_CONFIG` | `--agent-config` | 单个 v2 agent config YAML。 |
| `AIEVOBOX_AGENT_ROOT` | `--agent-root` | 未设置 `AIEVOBOX_AGENT_CONFIG` 时的 agent config 根目录。 |
| `AIEVOBOX_AGENT_START_CONFIG` | `--agent-start-config` | runtime 启动 YAML。 |
| `AIEVOBOX_GATEWAY_BASE_URL` | `--gateway-base-url` | Gateway session root；不会回退到 LLM proxy。 |
| `AIEVOBOX_ENABLE_EVALUATION` | `--enable-evaluation` | 设为 `1`、`true`、`yes` 或 `on` 时运行 evaluator 并提交可训练 reward。 |
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
| `RL_GLOBAL_BATCH_SIZE` | 训练全局 batch size。 |
| `RL_ROLLOUT_GROUP_BATCH_SIZE` | 单次 rollout batch 拉取的 group 数量。 |
| `SLIME_N_SAMPLES_PER_PROMPT` | 通常设置为 `RL_GROUP_SIZE`。 |

服务：

| 变量 | 说明 |
|------|------|
| `BUFFER_SERVER_HOST` | Slime 访问 Buffer Server 使用的 host。 |
| `BUFFER_SERVER_PORT` | Buffer Server 端口，常用 `18889`。 |
| `LLM_PROXY_HOST` | Gateway 访问 Slime-hosted LLM proxy 使用的 host。 |
| `LLM_PROXY_PORT` | LLM proxy 端口，常用 `18890`。 |
| `AIEVOBOX_GATEWAY_AUTOSTART` | 默认为启用；设为 `0` 时 Buffer Server 不自动启动 Gateway。 |

## Buffer Server API

| 端点 | 用途 |
| --- | --- |
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

## 建议流程

1. 先在 RL 外部用 `launcher.py`、Gateway 和 evaluator 跑通 `my_env`。
2. 确认 `session_steps` 中出现带 reward 的 trainable 行。
3. 创建 `rl/examples/my_env/env.sh`，设置环境配置、模型 route、DB URL、训练路径和并发参数。
4. 在仓库根目录启动 `rl/run_slime_generator.sh`。
5. 在另一个终端启动 `rl/run_buffer_server.sh`。
6. 观察 `logs/buffer_server.log`、`logs/main.log`、Gateway 日志和 Slime 日志。
7. 小规模稳定后，再扩大 `AIEVOBOX_POOL_SIZE`、`RL_GROUP_SIZE` 和训练 batch。

## 排错

| 现象 | 检查项 |
|------|--------|
| Buffer Server 启动 launcher 时仍像旧 `--env-config` 流程 | 使用 `AIEVOBOX_AGENT_CONFIG` 或 `AIEVOBOX_AGENT_ROOT`，不要使用旧变量。 |
| Launcher 因缺少 start config 失败 | 设置 `AIEVOBOX_AGENT_START_CONFIG`，或采用同目录 `<name>_start.yaml` 命名供 Buffer Server 自动推导。 |
| Launcher 提示缺少 gateway URL | 将 `AIEVOBOX_GATEWAY_BASE_URL` 设置为 Gateway session root；不要指向 LLM proxy。 |
| Gateway route 找不到模型 | 确认 `RL_MODEL` 与 Gateway `llm_routes` key 一致；自动 Gateway 会使用 `RL_MODEL` 生成 route。 |
| `/get_rollout_data` 没有样本 | 检查 `session_steps` 是否有 `is_trainable = 1`、reward 是否写入、`group_id` 数量是否匹配。 |
| Group 一直不 ready | `RL_GROUP_SIZE` 大于该 `group_id` 下已完成样本数。 |
| Gateway storage 与 launcher 不一致 | 确认自动 Gateway 生成配置中的 `storage_config.db_url` 与 `AIEVOBOX_DB_URL` 一致；外部 Gateway 也必须使用同一个 DB/backend。 |
