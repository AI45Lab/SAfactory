# Safactory RL 使用指南

本文说明如何在 Safactory 中使用 local mode 运行 RL 训练。local 工作流包括一个 Safactory 仓库、本地或可访问的环境服务、SQLite 轨迹数据库、Buffer Server，以及一个 Slime 训练进程。

对于复杂环境，local mode 仍然可以连接 HTTP 服务或 Docker 容器。RayJob 是独立的远端部署模式，不是 local RL 训练的必要条件。

## 组件关系

| 组件 | 作用 |
| --- | --- |
| `rl/examples/<env>/env.sh` | 实验配置。启动任意 RL 进程前都先 source 这个文件。 |
| `rl/run_buffer_server.sh` | 启动 Safactory Buffer Server 和 rollout runner。 |
| `rl/run_slime_generator.sh` | 启动 Slime 训练和 Safactory rollout function。 |
| `rl/buffer_server.py` | 启动 rollout 采集，从 SQLite 读取完成轨迹，聚合 sample group，并提供给 generator。 |
| `rl/slime_generator.py` | Slime rollout function。它会启动 LLM proxy，从 Buffer Server 获取轨迹 group，构造 mask/reward，并返回 Slime samples。 |
| Slime / Megatron / SGLang | 训练和推理栈。请按 Slime 环境或 Docker 说明单独安装。 |

## 工作流

### 配置 `env.sh`

每个示例的运行配置都集中在 `rl/examples/<env>/env.sh`，每个环境的实验配置应维护在自己的 `env.sh` 中。运行前至少检查以下参数：

```bash
export AIEVOBOX_ROOT=/path/to/Safactory
export AIEVOBOX_MODE=local
export AIEVOBOX_ENV_CONFIG=/path/to/env_config.yaml
export AIEVOBOX_DB_URL=sqlite:////path/to/rollout.db
export AIEVOBOX_ENV_TRANSPORT=inproc

export BUFFER_SERVER_HOST=127.0.0.1
export BUFFER_SERVER_PORT=18889
export LLM_PROXY_HOST=127.0.0.1
export LLM_PROXY_PORT=18890

export SLIME_HOME=/path/to/slime
export MEGATRON_HOME=/path/to/Megatron-LM
export HF_CKPT_DIR=/path/to/hf/checkpoint
export LOAD_DIR=
export SAVE_DIR=/path/to/save/checkpoints
```

配置建议：

- 使用 `AIEVOBOX_MODE=local`。
- 使用 `AIEVOBOX_ENV_CONFIG` 指定单个环境 YAML，或使用 `AIEVOBOX_ENV_ROOT` 指定一组 YAML。
- 使用 `AIEVOBOX_DB_URL=sqlite:////absolute/path/to/file.db` 保存轨迹。
- Python 环境可直接在 Safactory 进程内运行时，使用 `AIEVOBOX_ENV_TRANSPORT=inproc`。
- 环境由独立进程或容器提供服务时，使用 `AIEVOBOX_ENV_TRANSPORT=http`。

rollout 和训练并行度配置：

```bash
export AIEVOBOX_POOL_SIZE=2
export AIEVOBOX_LLM_MAX_CONCURRENCY=2
export AIEVOBOX_MAX_STEPS=10
export AIEVOBOX_MESSAGE_CUT=1

export RL_GROUP_SIZE=2
export RL_ROLLOUT_GROUP_BATCH_SIZE=1
export RL_GLOBAL_BATCH_SIZE=2
export RL_EPOCH=10
export NUM_ROLLOUT=8

export NUM_GPUS=2
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE=1
export ROLLOUT_NUM_GPUS=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=1
export TP_SIZE=1
```

配置建议：

- `AIEVOBOX_POOL_SIZE` 是并发环境实例数。
- `AIEVOBOX_LLM_MAX_CONCURRENCY` 限制访问 generator 内置 LLM proxy 的并发请求。
- `RL_GROUP_SIZE` 对应 Slime 的 `n_samples_per_prompt`。
- `RL_ROLLOUT_GROUP_BATCH_SIZE` 控制每个 rollout batch 请求多少个 completed group。
- `RL_GLOBAL_BATCH_SIZE` 是 Slime 训练 global batch size。
- `NUM_GPUS`、`ACTOR_NUM_GPUS_PER_NODE`、`ROLLOUT_NUM_GPUS`、`ROLLOUT_NUM_GPUS_PER_ENGINE` 必须与 Ray 和 SGLang 可见的 GPU 数匹配。

### 启动训练

从 Safactory 仓库根目录打开两个终端。

终端 1 启动 Slime 训练和 Safactory generator：

```bash
source rl/examples/geo3k_vl/env.sh
bash rl/run_slime_generator.sh
```

终端 2 启动 Buffer Server 和 rollout 采集：

```bash
source rl/examples/geo3k_vl/env.sh
bash rl/run_buffer_server.sh
```

Generator 会先启动 LLM proxy。Buffer Server 会启动 rollout runner，将轨迹写入 SQLite，聚合完成的 sessions，并把这些 group 提供给 generator。

## 关键参数

| 变量 | 说明 |
| --- | --- |
| `AIEVOBOX_ROOT` | Safactory 仓库路径。 |
| `AIEVOBOX_MODE` | local RL 使用 `local`。 |
| `AIEVOBOX_ENV_CONFIG` | 单个环境 YAML 文件。 |
| `AIEVOBOX_ENV_ROOT` | 包含多个环境 YAML 的目录。 |
| `AIEVOBOX_ENV_TRANSPORT` | `inproc` 或 `http`。 |
| `AIEVOBOX_DB_URL` | 轨迹数据库 URL。local mode 推荐 SQLite。 |
| `AIEVOBOX_POOL_SIZE` | 并发环境实例数。 |
| `AIEVOBOX_MAX_STEPS` | 每个 episode 的最大环境步数。 |
| `AIEVOBOX_MESSAGE_CUT` | prompt 中保留的最近轮数，`0` 表示保留全部。 |
| `RL_GROUP_SIZE` | 每个 prompt 的采样数。 |
| `RL_ROLLOUT_GROUP_BATCH_SIZE` | 每个 rollout batch 请求的 completed group 数。 |
| `RL_GLOBAL_BATCH_SIZE` | Slime global batch size。 |
| `RL_OFF_BY_N` | 允许的最大策略版本滞后。 |
| `DAPO_filter` | 是否丢弃 reward 全相同的 group。 |
| `BUFFER_SERVER_HOST` / `BUFFER_SERVER_PORT` | Generator 连接 Buffer Server 使用的地址。 |
| `LLM_PROXY_HOST` / `LLM_PROXY_PORT` | rollout workers 连接 generator 内置 LLM proxy 使用的地址。 |
| `AIEVOBOX_BUFFER_INCOMPLETE_GROUP_TTL_SECONDS` | Buffer Server 中 incomplete pending group 的超时时间。 |
