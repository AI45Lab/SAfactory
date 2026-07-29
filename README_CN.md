<div align="center">

# Safactory

<p align="center">
    中文 &nbsp ｜ &nbsp <a href="README.md">English</a>
</p>

**测训一体的下一代智能体基础设施，支持 Agent 快速接入、社区 Benchmark 快速接入、并发 rollout 运行、轨迹采集，以及在 OS、Android、Minecraft、具身智能、QA、数据处理、科学发现等多类任务上的强化学习训练。首次验证智能体可信 Scaling Law，实现安全能力提升且无对齐税。**

**内置 Gateway 是具备会话感知能力的 OpenAI 兼容 API 层：它将模型请求路由到配置的上游 LLM 服务，负责并发和步数控制，并把轨迹写入指定的存储。**

[快速开始](#quick-start) |
[演示](#demo) |
[RL 训练](docs/rl-training_CN.md) |
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

## <a id="why-safactory"></a>✨ 为什么使用 Safactory

![tax](fig/tax.png)

Safactory 是面向需要统一完成评测、数据生成和 RL 训练的团队的智能体沙箱。它帮助团队快速接入新的 Agent 和社区 Benchmark，通过可扩展的 rollout 池并发运行，经由 Gateway 统一路由 OpenAI 兼容模型流量，持久化轨迹数据，并将完成的数据桥接到 Slime / GRPO 训练。

| 需求 | Safactory 提供                                   |
|------|------------------------------------------------|
| 评测 Agent 与 Benchmark | 在真实交互任务和社区 Benchmark 中运行 LLM 或 VLM Agent 并收集奖励。 |
| 构建轨迹数据 | 将消息、动作、观察、奖励和环境状态持久化到数据平台。                     |
| RL 训练 | 通过内置 Buffer Server 将 rollout 轨迹流式送入 Slime。     |
| 接入新 Agent 与 Bench | 快速接入智能体和Benchmark套件，并通过并发 rollout worker 扩展运行。 |

核心能力：

- 多领域 Agent 与 Benchmark adapter：OS、Android、Minecraft、RoboTrustBench、Embodied ALFRED、QA、DABStep、DiscoveryWorld、DeepEyes、Geo3K-VL 和 Math500。
- 通过池化管理和异步worker调度支持高并发运行。
- 支持 vLLM、SGLang、托管 API 和本地代理等 OpenAI 兼容模型服务。
- 支持本地单机模式和基于 RayJob 的远程集群模式。

## <a id="demo"></a>🎬 演示

<div align="center">

https://github.com/user-attachments/assets/4c551b27-ce4d-4fc8-8df6-d6dc8100cc88

*点击播放查看完整演示*

</div>

## <a id="quick-start"></a>🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/AI45Lab/Safactory.git
cd Safactory
pip install -r requirements.txt
```


### 2. 配置 Gateway

复制示例配置，并新增目标LLM 相关参数：

```bash
cp gateway/config.example.yaml gateway/config.local.yaml
```

在 `gateway/config.local.yaml` 中，确保 gateway 和 launcher 使用同一个 SQLite DB：

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

### 3. 启动运行


#### 本地 Docker 运行任务

对于并发数要求不高的任务，可以采用docker mode 启动；下面示例用 Docker 模式运行仓库内置OpenRT 环境：

```bash
python launcher.py \
  --mode docker \
  --agent-config env/openrt/openrt_config.yaml \
  --agent-start-config env/openrt/openrt_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --db-path sqlite://env_trajs.db \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 20
```

关键点：

- `--llm-model` 是 gateway `llm_routes` 中的 `LLM_MODEL_NAME`，不是任意上游模型名。
- `--agent-config` 定义任务和数据集。
- `--agent-start-config` 定义智能体运行时如何启动。
- `--gateway-base-url` 指向 gateway 的 session root。
- 使用 `sqlite` 时，`--db-path` 必须与 `gateway.storage_config.db_url` 一致。

#### 使用 RJob 扩展并发

如果本地 Docker 已跑通，并且需要更高并发或集群资源，可以切换到 RJob 模式。RJob 模式仍使用同一个 launcher，但运行时资源由 RJob 提交：

```bash
python launcher.py \
  --mode rjob \
  --rjob-config config.yaml \
  --agent-config env/openrt/openrt_config.rjob.yaml \
  --agent-start-config env/openrt/openrt_start.rjob.yaml \
  --gateway-base-url http://YOUR_GATEWAY_HOST:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --db-path sqlite://env_trajs.db\
  --pool-size 8
```

全局 RJob 鉴权放在 `config.yaml` 或 `--rjob-config`。每个 agent 的镜像、资源、挂载、嵌入文件和运行命令放在 `--agent-start-config`。`--pool-size` 控制并发规模，具体上限取决于集群资源和 agent start config。

#### 使用 Brainbox Sandbox

`--mode sandbox` 会从预先创建的 Brainbox Sandbox Environment 分配 rollout 实例。连接配置和 Environment ID 放在 `--sandbox-config`，runner 仍由 `--agent-start-config` 定义。

```bash
export OPEN_SANDBOX_API_KEY='<ak>:<sk>'
python launcher.py \
  --mode sandbox \
  --sandbox-config config.sandbox.example.yaml \
  --agent-config env/openrt/openrt_config.yaml \
  --agent-start-config env/openrt/openrt_start.yaml \
  --gateway-base-url http://YOUR_GATEWAY_HOST:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --pool-size 8
```

Environment、volume、生命周期和评测要求见 [Sandbox 模式](docs/sandbox-mode_CN.md)。

#### 启用评测

在 Docker、RJob 或 Sandbox 启动命令跑通后，如果需要执行评测任务，再追加 evaluator 相关参数：

```bash
python launcher.py \
  --agent-config env/openrt/openrt_config.yaml \
  --agent-start-config env/openrt/openrt_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_LLM_MODEL \
  --enable-evaluation \
  --db-path sqlite://env_trajs.db \
  --pool-size 1
```

评测只使用按约定动态发现的 `<agent-root>/<env_name>/rule_evaluator.py`。RJob 和 Sandbox 模式使用相同的 `--enable-evaluation` 参数。见[评测](docs/evaluation_CN.md)。

## 可选：风洞数据平台（LanceDB）

Safactory 可以通过 `wt-data-platform-sdk` 将轨迹和环境数据持久化到基于 LanceDB 的风洞数据平台。SQLite 仍是默认的本地存储策略；云存储相关依赖单独维护在 `requirements-cloud.txt` 中。

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

然后将 gateway 的 `storage_type` 设置为 `cloud`，并使用 `--storage-type cloud` 启动 Safactory。`production` profile 会选择生产 landing/serving 表，`test` profile 会选择对应的测试表。完整配置和表说明请参阅 [AI45Lab/wt-data-platform-sdk](https://github.com/AI45Lab/wt-data-platform-sdk)。

## 运行数据

本地运行默认将任务行和轨迹写入 `env_trajs.db`。建议启动时显式传入 `--job-id my-openrt-smoke`，便于后续查询、复现和训练过滤。

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

## RL 训练

Safactory 可以通过 `rl/buffer_server.py` 对接 Slime。当前 RL 脚本位于 `rl/examples/<task>/`，并读取各自目录下的 `env.sh`：

```bash
cd rl/examples/math500
./run_buffer_server.sh
```

Buffer Server 会启动 `launcher.py`，读取已完成的可训练行，按 `group_id` 聚合样本，并通过 `/get_rollout_data` 输出 batch。见 [RL 训练](docs/rl-training_CN.md)。


## 文档

| 指南                                     | 内容                                                                        |
|----------------------------------------|---------------------------------------------------------------------------|
| [Gateway](docs/gateway_CN.md)          | Gateway 端点、路由、Admission Control、telemetry、请求日志和存储一致性。                     |
| [配置](docs/configuration_CN.md)         | 当前 `launcher.py`、gateway、agent config、agent start config 和 RJob 字段。       |
| [支持的环境](docs/environments_CN.md)       | 当前仓库内置 adapter 及运行时依赖。                                                    |
| [评测](docs/evaluation_CN.md)            | Rule evaluator 配置和 reward commit 行为。 |
| [数据管理器](docs/data-manager_CN.md)       | SQLite/cloud 存储行为、表、事件类型和查询示例。                                            |
| [自定义环境](docs/custom-environment_CN.md) | 如何新增自定义环境。                                                                |
| [RL 训练](docs/rl-training_CN.md)        | Buffer Server 与 Slime 集成细节。                                               |

## <a id="architecture"></a>🏗️ 架构

![Safactory architecture](fig/overview.png)

整体上，`launcher.py` 会加载环境 YAML 文件，启动或连接环境服务，将观察发送到 OpenAI 兼容模型端点，通过数据管理器记录每次交互，并可选择将完成的 rollout 转发给 RL 训练。

## 数据集

Safactory 可以生成可复用的轨迹数据集。公开 OS 轨迹发布在 Hugging Face：

- [AI45Research/SATraj-OS](https://huggingface.co/datasets/AI45Research/SATraj-OS)，由 Safactory 生成、用于智能体训练和分析的 OS 轨迹数据集。

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

如果 Safactory 或 Safactory 生成的数据集对你的工作有帮助，请引用本仓库以及你使用的具体数据集或报告。

```bibtex
@misc{chen2026safactoryscalableagenticinfrastructure,
      title={Safactory: A Scalable Agentic Infrastructure for Training Trustworthy Autonomous Intelligence},
      author={Shanghai AI Lab},
      year={2026},
      eprint={2605.06230},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.06230},
}
```
