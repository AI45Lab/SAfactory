# 强化学习训练

本文是 Safactory RL 训练的唯一入口文档。通用启动脚本放在 `rl/`目录下；每个环境只保留自己的 `env.sh` 配置。切换环境或运行后端时，修改 `env.sh` 中的变量即可，不需要复制启动脚本。

## 组件

| 组件 | 作用 |
| --- | --- |
| `rl/examples/<env>/env.sh` | 环境、运行后端、服务和 Slime 训练配置。 |
| `rl/run_buffer_server.sh` | 启动 Buffer Server，按需自动启动 Gateway，并启动 Safactory rollout 采集。 |
| `rl/run_slime_generator.sh` | 启动 Ray、Slime 训练、SGLang rollout engine 和 Safactory rollout 函数。 |
| `rl/buffer_server.py` | 启动 `launcher.py`，读取存储中的 trainable 行，按 `group_id` 聚合样本，并通过 `/get_rollout_data` 提供给 Slime。 |
| `rl/slime_generator.py` | Slime rollout 函数。负责启动 LLM proxy、拉取轨迹组、构造 mask 和 reward，并返回训练样本。 |
| `rl/llm_proxy.py` | 由 Slime generator 托管的 OpenAI 兼容接口，Gateway 在线 rollout 时会转发到这里。 |

`PYTHON_BIN` 和 `RAY_BIN` 对应的运行环境需要提前安装 Slime、
Megatron-LM、SGLang、Ray 以及模型运行依赖。

## 架构

```text
Safactory runtime  <--- 由 launcher.py / rl/buffer_server.py 启动
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

对于 v2 adapter，launcher 仍需要有效的 Gateway 兼容模型 route 和 runtime start config。扩容前请先在 `logs/buffer_server.log` 中确认生成的 launcher 命令。

## 最小 Geo3K 训练路径

先在 RL 外部用 `launcher.py`、Gateway 和 `--enable-evaluation` 跑通 Geo3K 评测。这一步会验证 Docker 镜像、dataset、route key、存储和 `rule_evaluator.py`。

然后编辑或覆盖 `rl/examples/geo3k_vl/env.sh`：

```bash
export AIEVOBOX_ROOT=$(pwd)
export AIEVOBOX_MODE=docker
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.yaml
export AIEVOBOX_GATEWAY_HOST=127.0.0.1
export AIEVOBOX_GATEWAY_PORT=8000
export AIEVOBOX_GATEWAY_BASE_URL=http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions
export RL_MODEL=geo3k_model

export AIEVOBOX_ENABLE_EVALUATION=1
export AIEVOBOX_POOL_SIZE=2
export AIEVOBOX_MAX_STEPS=10
export RL_GROUP_SIZE=2
export RL_EPOCH=1

export HF_CKPT_DIR=/path/to/hf-checkpoint
export LOAD_DIR=${HF_CKPT_DIR}
export SAVE_DIR=/path/to/save/checkpoints
export SLIME_HOME=/path/to/slime
export MEGATRON_HOME=/path/to/Megatron-LM
```

如果默认 Geo3K 配置指向完整本地 parquet 数据集，smoke test 阶段请改用仓库自带样例数据。

在仓库根目录启动两个服务：

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_slime_generator.sh
```

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_buffer_server.sh
```

Buffer Server 默认会自动启动 Gateway，生成 `logs/gateway.rl.generated.yaml`，把 `RL_MODEL` 路由到 Slime 托管的 LLM proxy，启动 Docker rollout 采集，并通过 `/get_rollout_data` 提供已完成的 group。使用自动启动时，请先停止同一端口上手动启动的 Gateway。

- `AIEVOBOX_DB_URL` 作为存储 DB，与 `launcher.py --db-path` 保持一致。
- `RL_MODEL` 作为 Gateway route key。
- `http://${LLM_PROXY_HOST}:${LLM_PROXY_PORT}/v1` 作为 route 上游。
- `AIEVOBOX_GATEWAY_PORT` 作为监听端口。

只有在你已经手动启动了外部 Gateway，并且它使用相同存储后端和 route key 时，才设置 `AIEVOBOX_GATEWAY_AUTOSTART=0`。

## 启动 RL 训练

因此，仓库中的 RL examples 是围绕共享 v2 launcher 的配置模板。Geo3K 是当前维护的标准模板。

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

训练由 trainer 或客户端调用 Buffer Server 的 `POST /start_rollout` 后开始。随后 Slime generator 通过 `POST /get_rollout_data` 拉取完成的 grouped samples。

## 通用 `env.sh` 配置 —— Geo3K 为例

`rl/examples/geo3k_vl/env.sh` 是 Geo3K RL 通常唯一需要用户编辑的文件。训练参数、服务地址和运行后端都集中放在这里。

```bash
export AIEVOBOX_ROOT=/path/to/SAfactory
export STORAGE_TYPE=sqlite
export AIEVOBOX_DB_URL=sqlite:///${AIEVOBOX_ROOT}/rl/examples/geo3k_vl/geo3k_vl.db

export BUFFER_SERVER_HOST=127.0.0.1
export BUFFER_SERVER_PORT=18889
export LLM_PROXY_HOST=127.0.0.1
export LLM_PROXY_PORT=18890

# Docker 本地运行可用 127.0.0.1；RJob 需要换成集群可访问地址。
export AIEVOBOX_GATEWAY_HOST=127.0.0.1
export AIEVOBOX_GATEWAY_PORT=8000
export AIEVOBOX_GATEWAY_BASE_URL=http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions

export RL_MODEL=model
export RL_GROUP_SIZE=8
export RL_GLOBAL_BATCH_SIZE=512
export RL_ROLLOUT_GROUP_BATCH_SIZE=64
export RL_EPOCH=1000
export RL_OFF_BY_N=0
export DAPO_filter=true

export AIEVOBOX_POOL_SIZE=16
export AIEVOBOX_MAX_STEPS=10
export AIEVOBOX_ENABLE_EVALUATION=1

export SLIME_HOME=/path/to/slime
export MEGATRON_HOME=/path/to/Megatron-LM
export HF_CKPT_DIR=/path/to/hf/checkpoint
export LOAD_DIR=${HF_CKPT_DIR}
export SAVE_DIR=/path/to/save/checkpoints
```

Geo3K 建议开启 evaluation，因为 `env/geo3k/rule_evaluator.py` 会把 runner 产出的 `metrics.score` 转成 Safactory reward。如果关闭 evaluation，rollout 可以运行，但 reward 可能不会按可训练样本提交。

## Geo3K Docker 模式

当 launcher 所在机器可以直接运行本地 Docker 容器时，使用 Docker 模式。

在 `rl/examples/geo3k_vl/env.sh` 中配置：

```bash
export AIEVOBOX_MODE=docker
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.yaml

export AIEVOBOX_GATEWAY_HOST=127.0.0.1
export AIEVOBOX_GATEWAY_PORT=8000
export AIEVOBOX_GATEWAY_BASE_URL=http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions
```

Docker 相关文件：

| 文件 | 作用 |
| --- | --- |
| `env/geo3k/geo3k_config.yaml` | 指定本地 Docker image、Geo3K parquet 路径、需要读取的数据列，以及 `max_turns`、`max_images` 等 `env_params`。 |
| `env/geo3k/geo3k_start.yaml` | 定义本地 Docker runner 挂载、结果目录挂载、环境变量和 `host.docker.internal` 网络配置。 |

Docker 模式注意事项：

- launcher 环境必须有 `docker` CLI，并且当前用户有启动容器的权限。
- `geo3k_config.yaml` 中的 `env_image` 必须已经在本机存在，或能够按 Docker pull 策略拉取。
- `container.runner_entrypoint.source: ./` 会把整个 `env/geo3k` 目录挂到容器内，所以 `runner.py` 和 `math_utils.py` 不需要提前烘进镜像。
- 镜像仍然必须包含 Geo3K 运行依赖，例如 `requests`、`sympy` 和 `pylatexenc`。
- `geo3k_start.yaml` 会挂载 `./results` 用于 artifact。Docker 相对挂载路径按 launcher 工作目录解析。
- 访问本地 Gateway 时，Docker adapter 会注入容器可用的 session URL，`geo3k_start.yaml` 也配置了 `host.docker.internal`。

## Geo3K RJob 模式

当 rollout 环境需要运行在远端 RJob 集群上时，使用 RJob 模式。除非你另行部署，
Slime trainer、SGLang rollout engines、Buffer Server、Gateway 和 launcher
仍然运行在 RL launcher 环境中。

在 `rl/examples/geo3k_vl/env.sh` 中配置：

```bash
export AIEVOBOX_MODE=rjob
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.rjob.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.rjob.yaml

# 必须是 RJob 容器可访问的地址；不要使用 127.0.0.1 或 localhost。
export AIEVOBOX_GATEWAY_HOST=<launcher-or-gateway-ip-visible-to-rjob>
export AIEVOBOX_GATEWAY_PORT=8000
export AIEVOBOX_GATEWAY_BASE_URL=http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions
```

RJob 相关文件：

| 文件 | 作用 |
| --- | --- |
| `env/geo3k/geo3k_config.rjob.yaml` | 指定 RJob 集群可拉取的 registry image，以及 launcher 物化 parquet row 时可访问的数据集路径。 |
| `env/geo3k/geo3k_start.rjob.yaml` | 定义 RJob 资源、runner 嵌入文件、清理策略、结果 artifact 挂载和代理相关环境变量。 |
| `config.yaml` | `launcher.py` 默认读取的全局 RJob 连接和鉴权配置。当前 RL Buffer Server 调用 `launcher.py` 时没有透传 `--rjob-config`，因此 RL RJob 会使用默认 `config.yaml`。 |

launcher 的 Python 环境必须能导入 RJob SDK：

```bash
python -c "from brainpp.rjob import RJobClient; print(RJobClient)"
```

全局 RJob 配置放在 `config.yaml`：

```yaml
rjob:
  cluster_entry: "https://your-rjob-platform.example"
  namespace: "your-namespace"
  access_key: "replace-me"
  secret_key: "replace-me"
  charged_group: "your-quota-or-project"
  gateway_base_url: "http://<gateway-host-visible-to-rjob>:8000/v1/sessions"
  submit_concurrency: 1
  cleanup_on_finish: true
  no_packaging: true
```

RJob 配置要点：

- `access_key` 和 `secret_key` 用于 `RJobClient` 创建、轮询、读取日志和删除 RJob。它们也决定你是否有权限使用 namespace、charged group、镜像和挂载目录。尽量不要把真实 AK/SK 提交到仓库。
- `charged_group` 用于选择 RJob 消耗的配额或计费组。
- 全局或 per-agent RJob 配置里的 `gateway_base_url` 会覆盖 launcher request 中的 URL。如果设置了它，必须保证它是 RJob 容器可访问的 Gateway 地址。如果希望完全由 `env.sh` 控制，清理掉过期的 `rjob.gateway_base_url`。
- `mount_config` 会把集群可访问存储挂到 RJob 容器内。左侧必须是 RJob 集群能挂载的存储，而不是本机 Docker bind path。Geo3K 用它保存 result artifact，例如 `gpfs://gpfs1/evobox-share/chenxinquan/SAfactory/results:/app/results`。
- `container.runner_entrypoint.source: ./runner.py` 会由 RJob runtime 自动嵌入或分发到 `target`，RJob 不使用 Docker 那种本地 bind mount。
- runner 额外 import 的本地文件需要写到 `rjob.embedded_files`。Geo3K 中包含 `math_utils.py`。
- RJob 镜像必须包含环境依赖。嵌入 runner 文件只提供 adapter 代码，不会安装 Python 包。

- RJob 网络最常见的问题是 Gateway URL 写成本地地址。`AIEVOBOX_GATEWAY_BASE_URL=http://127.0.0.1:8000/v1/sessions` 只对 launcher 机器上的进程有效；RJob 容器需要能从集群路由到 Gateway 主机的地址。

## 关键变量

| 变量 | 说明 |
| --- | --- |
| `AIEVOBOX_ROOT` | Safactory 仓库路径。 |
| `AIEVOBOX_MODE` | 运行后端：`docker`、`rjob` 或 `sandbox`。 |
| `AIEVOBOX_AGENT_CONFIG` | Geo3K 单环境 YAML 配置。 |
| `AIEVOBOX_AGENT_START_CONFIG` | Geo3K Docker 或 RJob 启动配置。 |
| `STORAGE_TYPE` | `sqlite` 或 `cloud`；Geo3K RL 示例通常使用 `sqlite`。 |
| `AIEVOBOX_DB_URL` | Gateway、launcher、Buffer Server 和 evaluator 共享的 rollout 轨迹 DB URL。 |
| `AIEVOBOX_GATEWAY_HOST` / `AIEVOBOX_GATEWAY_PORT` | Buffer Server ready 检查和 `AIEVOBOX_GATEWAY_BASE_URL` 使用的 host/port。 |
| `AIEVOBOX_GATEWAY_BASE_URL` | 传给 launcher 和 runtime request 的 Gateway session root。必填。 |
| `AIEVOBOX_GATEWAY_AUTOSTART` | 默认启用。设为 `0` 表示使用手动管理的外部 Gateway。 |
| `AIEVOBOX_ENABLE_EVALUATION` | 设为 `1`、`true`、`yes` 或 `on` 时运行 evaluator 并提交可训练 reward。 |
| `AIEVOBOX_POOL_SIZE` | 并发 rollout 环境实例数。RJob 模式下表示目标并发 RJob 数。 |
| `AIEVOBOX_MAX_STEPS` | 每个 episode 的最大环境步数。 |
| `RL_MODEL` | Gateway route key。必须匹配 RL LLM proxy 生成的 route。 |
| `RL_GROUP_SIZE` | 每个 prompt 的采样数，对应 Slime `n_samples_per_prompt` 和 launcher `--rl-group-size`。 |
| `RL_ROLLOUT_GROUP_BATCH_SIZE` | 每个 rollout batch 拉取的完成样本组数量。为空时默认 `RL_GLOBAL_BATCH_SIZE / RL_GROUP_SIZE`。 |
| `RL_GLOBAL_BATCH_SIZE` | Slime 训练 global batch size。 |
| `RL_EPOCH` | 重复调度 RL 数据集行的 rollout epoch 数。 |
| `RL_OFF_BY_N` | 训练侧过滤使用的最大策略版本滞后。 |
| `DAPO_filter` | 是否丢弃 reward 全部相同的 group。 |
| `BUFFER_SERVER_HOST` / `BUFFER_SERVER_PORT` | generator 连接 Buffer Server 使用的地址。 |
| `LLM_PROXY_HOST` / `LLM_PROXY_PORT` | Gateway 连接 Slime generator 内置 LLM proxy 使用的地址。 |
| `SLIME_HOME` | Slime 仓库路径。 |
| `MEGATRON_HOME` | Megatron-LM 仓库路径。 |
| `HF_CKPT_DIR` | 初始化训练和 rollout engine 的 HuggingFace checkpoint。 |
| `LOAD_DIR` | 传给 Slime `--load` 的 checkpoint 路径。首次运行通常等于 HF checkpoint。 |
| `SAVE_DIR` | Megatron checkpoint 保存目录。 |
| `NUM_GPUS` | 注册给本地 Ray head 的 GPU 数。 |
| `ACTOR_NUM_GPUS_PER_NODE` | 训练 actor 使用的 GPU 数。 |
| `ROLLOUT_NUM_GPUS` | SGLang rollout engine 使用的 GPU 总数。 |
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | 每个 SGLang rollout engine 使用的 GPU 数。 |
| `SGLANG_MEM_FRACTION_STATIC` | SGLang 预留静态显存比例。在线权重更新 OOM 时可适当调低。 |

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

## 运行前检查

扩容前建议检查：

1. 先在 RL 外部使用同一套 Docker 或 RJob 配置跑通一个 Geo3K case。
2. 确认 launcher 能访问 Gateway `/readyz`。
3. RJob 模式下，确认 RJob 容器内也能访问 Gateway URL。
4. 确认 `session_steps` 中出现期望 `job_id` 和 `group_id` 的 trainable 行。
5. 确认每个 group 能产出 `RL_GROUP_SIZE` 条完成样本，否则 Buffer Server 会一直等待不完整 group。
6. 观察 `logs/<run>/main.log`、`logs/<run>/slime.log`、`logs/gateway.log`、`logs/gateway_requests.jsonl` 和 `logs/buffer_server.log`。

## 排错

| 现象 | 检查项 |
| --- | --- |
| `FileNotFoundError: docker` | Docker 模式要求 launcher 环境有 Docker CLI。没有 Docker 时切换到 RJob。 |
| `RJob mode requires brainpp.rjob / RJobClient` | 在 launcher 使用的 Python 环境中安装或激活 RJob SDK。 |
| 创建 RJob 返回 `403 Forbidden` | 检查 RJob AK/SK、namespace、charged group、镜像权限和挂载目录权限。 |
| RJob succeeded 但没有解析到 result JSON | runner 需要向 stdout 输出一条 `SimulationStartResult` JSON；artifact fallback 需要确保 `SAFACTORY_RESULT_PATH` 指向可写挂载路径，例如 `/app/results`。 |
| RJob runner 连接 Gateway 失败 | RJob 不要使用 `127.0.0.1` 或 `localhost`。改用 RJob 集群可访问的 Gateway 地址。 |
| Gateway 对 Geo3K 图片请求返回 `400` | `max_images > 0` 时，`RL_MODEL` 必须路由到支持多模态的 rollout model。 |
| evaluator 日志中 `total_rows=0 trainable_rows=0` | runtime 在记录模型调用前已经失败，或 Gateway 和 launcher 没有使用同一个 DB。先看 evaluator 之前的 worker 错误。 |
| group 一直不 ready | `RL_GROUP_SIZE` 大于同一 `group_id` 下已经完成的样本数。 |
| 在线权重更新 OOM | 调低 `SGLANG_MEM_FRACTION_STATIC`，降低 rollout engine 并发，或降低模型并行压力。 |
