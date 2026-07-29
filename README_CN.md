<div align="center">

# SAfactory

<p align="center">
    中文 &nbsp ｜ &nbsp <a href="README.md">English</a>
</p>

**测训一体的下一代智能体基础设施，支持 Agent 快速接入、社区 Benchmark 快速接入、并发 rollout 运行、轨迹采集，以及在 OS、Android、Minecraft、具身智能、QA、数据处理、科学发现等多类任务上的强化学习训练。首次验证智能体可信 Scaling Law，实现安全能力提升且无对齐税。**

**内置 Gateway 是具备会话感知能力的 OpenAI 兼容 API 层：它将模型请求路由到配置的上游 LLM 服务，负责并发和步数控制，并把轨迹写入指定的存储。**

[快速开始](#quick-start) |
[演示](#demo) |
[RL 训练](docs/rl-training_CN.md) |
[RJob 模式](docs/rjob-mode_CN.md) |
[Sandbox 模式](docs/sandbox-mode_CN.md) |
[自定义环境](docs/custom-environment_CN.md) |
[配置](docs/configuration_CN.md) |
[数据](docs/data-manager_CN.md) |
[报告](https://arxiv.org/pdf/2605.06230)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Execution](https://img.shields.io/badge/mode-local%20%7C%20remote-orange)
![LLM](https://img.shields.io/badge/LLM-OpenAI--compatible-purple)

</div>

---

## <a id="why-SAfactory"></a>✨ 为什么使用 SAfactory

SAfactory 是面向需要统一完成评测、数据生成和 RL 训练的团队的智能体沙箱。它帮助团队快速接入新的 Agent 和社区 Benchmark，通过可扩展的 rollout 池并发运行，经由 Gateway 统一路由 OpenAI 兼容模型流量，持久化轨迹数据，并将完成的数据桥接到 Slime / GRPO 训练。

| 需求 | SAfactory 提供                                   |
|------|------------------------------------------------|
| 评测 Agent 与 Benchmark | 在真实交互任务和社区 Benchmark 中运行 LLM 或 VLM Agent 并收集奖励。 |
| 构建轨迹数据 | 将消息、动作、观察、奖励和环境状态持久化到数据平台。                     |
| RL 训练 | 通过内置 Buffer Server 将 rollout 轨迹流式送入 Slime。     |
| 接入新 Agent 与 Bench | 快速接入智能体和Benchmark套件，并通过并发 rollout worker 扩展运行。 |

核心能力：

- 多领域 Agent 与 Benchmark adapter：OS、Android、Minecraft、RoboTrustBench、Embodied ALFRED、QA、DABStep、DiscoveryWorld、DeepEyes、Geo3K-VL 和 Math500。
- 通过池化管理和异步worker调度支持高并发运行。
- 支持 vLLM、SGLang、托管 API 和本地代理等 OpenAI 兼容模型服务。
- 支持本地 Docker 模式、远程 RJob 模式和 Brainbox Sandbox 模式。

## <a id="demo"></a>🎬 演示

<div align="center">

https://github.com/user-attachments/assets/4c551b27-ce4d-4fc8-8df6-d6dc8100cc88

*点击播放查看完整演示*

</div>

## <a id="quick-start"></a>🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/AI45Lab/SAfactory.git
cd SAfactory
pip install -r requirements.txt
```

Docker 模式需要安装 Docker，并准备与环境适配的镜像。


### 2. 配置 Gateway

复制示例配置，并新增目标LLM 相关参数：

```bash
cp gateway/config.example.yaml gateway/config.local.yaml
```

```yaml
listen_port: 8000
storage_type: sqlite
storage_config:
  db_url: sqlite://env_trajs.db

llm_routes:
  LLM_MODEL_NAME:
    base_url: http://YOUR_LLM_HOST/v1
    api_key: YOUR_API_KEY
    supports_stream: true
    max_concurrency: 64
```

启动 gateway：

```bash
python -m gateway --config gateway/config.local.yaml
```

另开一个终端检查 ready 状态：

```bash
curl http://127.0.0.1:8000/readyz
```

### 3. Docker 模式运行评测

```bash
python launcher.py \
  --mode docker \
  --agent-config env/geo3k/geo3k_config.yaml \
  --agent-start-config env/geo3k/geo3k_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --enable-evaluation \
  --job-id geo3k-docker-smoke \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 10
```

关键点：

- `--llm-model` 是 gateway `llm_routes` 中的 `LLM_MODEL_NAME`，不是任意上游模型名。
- `--agent-config` 定义任务和数据集。
- `--agent-start-config` 定义智能体运行时如何启动。
- `--gateway-base-url` 指向 gateway 的 session root。
- 使用 `sqlite` 时，`--db-path` 必须与 `gateway.storage_config.db_url` 一致。
- `--enable-evaluation` 会按约定发现 `rule_evaluator.py` 并提交 score。

### 4. Docker 模式运行 RL 训练

RL 训练复用同一套 Docker runtime，但 Gateway 通常由 Buffer Server 自动启动，并路由到 Slime generator 内置的 LLM proxy。先修改 `rl/examples/geo3k_vl/env.sh`：

```bash
export AIEVOBOX_MODE=docker
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.yaml
export AIEVOBOX_GATEWAY_HOST=127.0.0.1
export AIEVOBOX_GATEWAY_PORT=8000
export HF_CKPT_DIR=/path/to/qwen3-vl-checkpoint
export SLIME_HOME=/path/to/slime
export MEGATRON_HOME=/path/to/Megatron-LM
```

如果评测 smoke test 中手动启动的 Gateway 仍然占用同一个端口，先停止它再启动 Buffer Server。只有当外部 Gateway 已经把 `RL_MODEL` 路由到 Slime LLM proxy，并且使用同一个 `AIEVOBOX_DB_URL` 时，才设置 `AIEVOBOX_GATEWAY_AUTOSTART=0`。

然后在仓库根目录打开两个终端。

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_slime_generator.sh
```

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_buffer_server.sh
```

Slime generator 会托管 `rl/llm_proxy.py`；Buffer Server 会生成 `logs/gateway.rl.generated.yaml`，启动 Gateway，拉起 Docker rollout 采集，并通过 `/get_rollout_data` 提供完成的训练 group。完整 RL 参数见 [RL 训练](docs/rl-training_CN.md)。

远程 runtime 使用相同的配置概念，但资源分配后端不同。见 [RJob 模式](docs/rjob-mode_CN.md) 和 [Sandbox 模式](docs/sandbox-mode_CN.md)。

## 可选：S3 + LanceDB 存储

Safactory 可以通过 `wt-data-platform-sdk` 将轨迹和环境数据持久化到以 S3 为对象存储、LanceDB 为数据引擎的存储平台。SQLite 仍是默认的本地存储策略；云存储相关依赖单独维护在 `requirements-cloud.txt` 中。

可选的 LanceDB/cloud 依赖栈要求使用 Python 3.10-3.12，当前已验证的环境为 Python 3.12。

安装可选依赖：

```bash
pip install -r requirements-cloud.txt
```

创建本地 `.env` 文件并填写数据平台连接参数（请勿提交包含凭证的文件）：

```bash
# 可选值：production 或 test
WT_SDK_PROFILE=test
WT_SDK_DB_URI=s3://YOUR_DATA_DATABASE
WT_SDK_ENV_CONFIG_DB_URI=s3://YOUR_ENV_CONFIG_DATABASE
WT_SDK_S3_ENDPOINT=https://YOUR_S3_ENDPOINT
WT_SDK_S3_ALLOW_HTTP=true
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
AWS_EC2_METADATA_DISABLED=true
```

启动 Safactory 前，将配置加载到进程环境：

```bash
set -a
source .env
set +a
```

然后将 gateway 的 `storage_type` 设置为 `cloud`，并使用 `--storage-type cloud` 启动 Safactory。`production` profile 会选择生产 landing/serving 表，`test` profile 会选择对应的测试表。完整配置、表说明，以及如何查询和拉取数据，请参阅 [AI45Lab/wt-data-platform-sdk](https://github.com/AI45Lab/wt-data-platform-sdk)。

## 运行数据

本地运行默认将任务行和轨迹写入 `env_trajs.db`。建议启动时显式传入 `--job-id geo3k-docker-smoke`，便于后续查询、复现和训练过滤。

- `job_id`：一次 `launcher.py` 运行。
- `session_id`：一个环境实例/任务实例，对应 `job_environments.env_id` 和 `session_steps.session_id`。

查看最近运行：

```bash
sqlite3 env_trajs.db "
  SELECT id, job_id, env_id AS session_id, env_name, group_id, finished, created_at
  FROM job_environments
  ORDER BY id DESC
  LIMIT 20;"
```

查看某个 session 的 step、奖励和完成状态：

```bash
sqlite3 env_trajs.db "
  SELECT step_id, llm_model, step_reward, reward,
         is_terminal, is_session_completed, is_trainable, created_at
  FROM session_steps
  WHERE session_id = '<session-id>'
  ORDER BY step_id, id;"
```

默认本地产物：

| 产物 | 默认位置 |
|------|----------|
| SQLite 轨迹 DB | `env_trajs.db` |
| Launcher 日志 | `logs/<timestamp>/main.log` |
| Gateway 日志 | `logs/gateway.log` |
| Gateway 请求日志 | `logs/gateway_requests.jsonl` |
| Adapter 输出 | `results/` 或 adapter 挂载目录 |

完整表结构、行类型和更多查询见[数据管理器](docs/data-manager_CN.md)。


## 文档

| 指南                                     | 内容                                                                        |
|----------------------------------------|---------------------------------------------------------------------------|
| [Gateway](docs/gateway_CN.md)          | Gateway 端点、路由、Admission Control、telemetry、请求日志和存储一致性。                     |
| [配置](docs/configuration_CN.md)         | 当前 `launcher.py`、gateway、agent config、agent start config 和 RJob 字段。       |
| [RJob 模式](docs/rjob-mode_CN.md)       | 远程 RJob runtime 配置、鉴权、挂载、Gateway 可达性和 Geo3K 示例。                         |
| [Sandbox 模式](docs/sandbox-mode_CN.md) | Brainbox Sandbox Environment 配置、volume、生命周期和启动流程。                         |
| [支持的环境](docs/environments_CN.md)       | 当前仓库内置 adapter 及运行时依赖。                                                    |
| [评测](docs/evaluation_CN.md)            | Rule evaluator 配置和 reward commit 行为。 |
| [数据管理器](docs/data-manager_CN.md)       | SQLite/cloud 存储行为、表、事件类型和查询示例。                                            |
| [自定义环境](docs/custom-environment_CN.md) | 如何新增自定义环境。                                                                |
| [RL 训练](docs/rl-training_CN.md)        | Buffer Server 与 Slime 集成细节。                                               |

## <a id="architecture"></a>🏗️ 架构

![SAfactory architecture](fig/overview.png)

整体上，`launcher.py` 会加载环境 YAML 文件，启动或连接环境服务，将观察发送到 OpenAI 兼容模型端点，通过数据管理器记录每次交互，并可选择将完成的 rollout 转发给 RL 训练。

## 数据集

![tax](fig/tax.png)

SAfactory 可以生成可复用的轨迹数据集。公开 OS 轨迹发布在 Hugging Face：

- [AI45Research/SATraj-OS](https://huggingface.co/datasets/AI45Research/SATraj-OS)，由 SAfactory 生成、用于智能体训练和分析的 OS 轨迹数据集。

SATraj-OS 可用于 SFT。利用该数据集训练出的 SCOPE 模型，在 OSWorld 和 OS-BLIND 任务上实现了能力与安全性之间更好的平衡：

![SCOPE capability-safety joint scaling](fig/scope_capability_safety_aaai_trend.png)

## 贡献

欢迎贡献新的自定义环境、 bug 修复和可复现实例。

每一个环境都为`env/`下的一个子目录，新增环境步骤如下：
1. 在 `env/`下创建新的子目录，以环境名称命名
2. 提供 `dataset/`， dataset 文件以`jsonl` 呈现，每一行为一个独立的调度任务
3. 同时提供`<name>_config.yaml` 和 `<name>_start.yaml`，并包含必须的`docker image`。
3. 按照环境需求字段添加启动运行脚本，统一命名为`runner`如 （`runner.py`/`runner.mjs`）, 
4. 根据评测需求实现`rule_evaluator.py`，非必须
5. 使用 `launcher.py` 运行本地 smoke test。

完整步骤见 [自定义环境](docs/custom-environment_CN.md)。

## 引用

如果 SAfactory 或 SAfactory 生成的数据集对你的工作有帮助，请引用本仓库以及你使用的具体数据集或报告。

```bibtex
@misc{chen2026safactoryscalableagenticinfrastructure,
      title={SAfactory: A Scalable Agentic Infrastructure for Training Trustworthy Autonomous Intelligence},
      author={Shanghai AI Lab},
      year={2026},
      eprint={2605.06230},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.06230},
}
```
