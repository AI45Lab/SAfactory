# Safactory RL 使用说明

本文说明 Safactory 中标准 RL 训练入口的使用方式。通用启动逻辑放在 `rl/` 目录下；每个示例目录只需要维护自己的 `env.sh` 配置文件。

## 组件

| 组件 | 作用 |
| --- | --- |
| `rl/examples/<env>/env.sh` | 环境和实验配置。切换环境时只需要 source 或传入不同的配置文件。 |
| `rl/run_buffer_server.sh` | 启动 Buffer Server 和 Safactory rollout runner。 |
| `rl/run_slime_generator.sh` | 启动 Ray、Slime 训练、SGLang rollout engine 和 Safactory rollout 函数。 |
| `rl/buffer_server.py` | 启动 rollout 采集，从存储中读取完成的轨迹，聚合样本组并提供给 Slime。 |
| `rl/slime_generator.py` | Slime rollout 函数。负责启动 LLM proxy、拉取轨迹组、构造 mask/reward，并返回训练样本。 |

`PYTHON_BIN` 和 `RAY_BIN` 对应的运行环境中需要提前安装 Slime、Megatron-LM 和 SGLang。

## 配置 `env.sh`

每个示例都应把 RL 相关配置集中放在 `rl/examples/<env>/env.sh`。例如：

```bash
export AIEVOBOX_ROOT=/path/to/SAfactory
export AIEVOBOX_MODE=docker
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.yaml
export AIEVOBOX_DB_URL=sqlite:///${AIEVOBOX_ROOT}/rl/examples/geo3k_vl/geo3k_vl.db

export BUFFER_SERVER_HOST=127.0.0.1
export BUFFER_SERVER_PORT=18889
export LLM_PROXY_HOST=127.0.0.1
export LLM_PROXY_PORT=18890

export SLIME_HOME=/path/to/slime
export MEGATRON_HOME=/path/to/Megatron-LM
export HF_CKPT_DIR=/path/to/hf/checkpoint
export LOAD_DIR=${HF_CKPT_DIR}
export SAVE_DIR=/path/to/save/checkpoints
```

rollout 和训练并行度也在同一个文件里配置：

```bash
export AIEVOBOX_POOL_SIZE=16
export AIEVOBOX_MAX_STEPS=10

export RL_GROUP_SIZE=8
export RL_GLOBAL_BATCH_SIZE=512
export RL_ROLLOUT_GROUP_BATCH_SIZE=64
export NUM_ROLLOUT=300

export NUM_GPUS=4
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE=1
export ROLLOUT_NUM_GPUS=3
export ROLLOUT_NUM_GPUS_PER_ENGINE=1
export TP_SIZE=1
```

`AIEVOBOX_MODE` 按 launcher 支持的后端选择：`docker`、`rjob` 或 `sandbox`。

## 启动训练

在 Safactory 仓库根目录打开两个终端。

终端 1 启动 Slime 训练和 rollout generator：

```bash
source rl/examples/geo3k_vl/env.sh
bash rl/run_slime_generator.sh
```

终端 2 启动 Buffer Server 和 rollout 采集：

```bash
source rl/examples/geo3k_vl/env.sh
bash rl/run_buffer_server.sh
```

也可以不把配置 source 到当前 shell，直接传入配置文件：

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_slime_generator.sh
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_buffer_server.sh
```

或使用 `--env`：

```bash
bash rl/run_slime_generator.sh --env rl/examples/geo3k_vl/env.sh
bash rl/run_buffer_server.sh --env rl/examples/geo3k_vl/env.sh
```

切换环境时，只需要换一个 `env.sh`，不需要复制或修改入口脚本。

## 关键变量

| 变量 | 说明 |
| --- | --- |
| `AIEVOBOX_ROOT` | Safactory 仓库路径。 |
| `AIEVOBOX_MODE` | launcher 后端：`docker`、`rjob` 或 `sandbox`。 |
| `AIEVOBOX_AGENT_CONFIG` | 单个 agent YAML 配置文件。 |
| `AIEVOBOX_AGENT_START_CONFIG` | Docker/RJob/Sandbox 的运行时启动配置。 |
| `AIEVOBOX_DB_URL` | rollout 轨迹存储 URL。本地运行推荐 SQLite。 |
| `AIEVOBOX_POOL_SIZE` | 并发 rollout 环境实例数。 |
| `AIEVOBOX_MAX_STEPS` | 每个 episode 的最大环境步数。 |
| `RL_GROUP_SIZE` | 每个 prompt 的采样数，对应 Slime 的 `n_samples_per_prompt`。 |
| `RL_ROLLOUT_GROUP_BATCH_SIZE` | 每个 rollout batch 拉取的完成样本组数量。 |
| `RL_GLOBAL_BATCH_SIZE` | Slime 训练 global batch size。 |
| `RL_OFF_BY_N` | 允许的最大策略版本滞后。 |
| `DAPO_filter` | 是否丢弃 reward 全部相同的组。 |
| `BUFFER_SERVER_HOST` / `BUFFER_SERVER_PORT` | generator 连接 Buffer Server 使用的地址。 |
| `LLM_PROXY_HOST` / `LLM_PROXY_PORT` | rollout worker 访问 generator 内置 LLM proxy 使用的地址。 |
| `SLIME_HOME` | Slime 仓库路径。 |
| `MEGATRON_HOME` | Megatron-LM 仓库路径。 |
| `HF_CKPT_DIR` | 用于初始化训练和 rollout engine 的 HuggingFace checkpoint。 |
| `LOAD_DIR` | 传给 Slime `--load` 的 checkpoint 路径。首次运行通常等于 HF checkpoint。 |
| `SAVE_DIR` | Megatron checkpoint 保存目录。 |
| `NUM_GPUS` | 注册给本地 Ray head 的 GPU 数。 |
| `ACTOR_NUM_GPUS_PER_NODE` | 训练 actor 使用的 GPU 数。 |
| `ROLLOUT_NUM_GPUS` | rollout engine 使用的 GPU 总数。 |
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | 每个 SGLang rollout engine 使用的 GPU 数。 |
| `SGLANG_MEM_FRACTION_STATIC` | SGLang 预留静态显存比例。在线权重更新 OOM 时可适当调低。 |

## 示例目录结构

示例目录应只保留环境相关配置。以 `geo3k_vl` 为例，RL 入口统一使用：

```bash
rl/run_buffer_server.sh
rl/run_slime_generator.sh
```

示例自己的配置文件是：

```bash
rl/examples/geo3k_vl/env.sh
```
