<div align="center">

# Safactory

<p align="center">
    中文 &nbsp ｜ &nbsp <a href="README.md">English</a>
</p>

**测训一体的下一代智能体基础设施，支持 Agent 快速接入、社区 Benchmark 快速接入、并发 rollout 运行、轨迹采集，以及在 OS、Android、Minecraft、具身智能、QA、数据处理、科学发现等多类任务上的强化学习训练。首次验证智能体可信 Scaling Law，实现安全能力提升且无对齐税。**

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

Safactory 是面向需要统一完成评测、数据生成和 RL 训练的团队的智能体沙箱。它帮助团队快速接入新的 Agent 和社区 Benchmark，通过可扩展的 rollout 池并发运行，统一路由 OpenAI 兼容模型流量，持久化轨迹数据，并将完成的数据桥接到 Slime / GRPO 训练。

| 需求 | Safactory 提供 |
|------|----------------|
| 评测 Agent 与 Benchmark | 在真实交互任务和社区 Benchmark 中运行 LLM 或 VLM Agent 并收集奖励。 |
| 构建轨迹数据 | 将消息、动作、观察、奖励和环境状态持久化到 SQLite。 |
| RL 训练 | 通过内置 Buffer Server 将 rollout 轨迹流式送入 Slime。 |
| 接入新 Agent 与 Bench | 快速接入智能体运行时和 Benchmark 套件，并通过并发 rollout worker 扩展运行。 |

核心能力：

- 多领域 Agent 与 Benchmark adapter：OS、Android、Minecraft、RoboTrustBench、Embodied ALFRED、QA、DABStep、DiscoveryWorld、DeepEyes、Geo3K-VL 和 Math500。
- 通过 runtime 池和异步 worker 支持高并发 rollout。
- 支持 vLLM、SGLang、托管 API 和本地代理等 OpenAI 兼容模型服务。
- 支持本地单机模式和基于 RayJob 的远程集群模式。
- 可选的经验抽取和 prompt 时经验注入。

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

复制示例配置，并将路由占位替换为自己的 OpenAI 兼容模型端点：

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
  YOUR_ROUTE_KEY:
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

### 3. 运行一个 Agent 配置

下面示例用 Docker 模式运行仓库内置 OpenClaw adapter：

```bash
python launcher.py \
  --agent-config env/openclaw/openclaw_config.yaml \
  --agent-start-config env/openclaw/openclaw_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --db-path sqlite://env_trajs.db \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 20
```

关键点：

- `--llm-model` 是 gateway `llm_routes` 中的 route key，不是任意上游模型名。
- `--agent-config` 定义任务和数据集。
- `--agent-start-config` 定义智能体运行时如何启动。
- `--gateway-base-url` 指向 gateway 的 session root。
- 使用 `sqlite` 时，`--db-path` 必须与 `gateway.storage_config.db_url` 一致。

### 4. 启用评测

rollout 结束后执行 evaluator：

```bash
python launcher.py \
  --agent-config env/openclaw/openclaw_config.yaml \
  --agent-start-config env/openclaw/openclaw_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --evaluation-model YOUR_ROUTE_KEY \
  --evaluation-config evaluator/configs/codex_cli_agent_eval.yaml \
  --enable-evaluation \
  --db-path sqlite://env_trajs.db \
  --pool-size 1
```

评测 spec 来自 `env/<agent>/eval_tasks/<dataset>/` 下的 markdown 文件，或 rule evaluator 文件，或者直接获取Bench运行后的结果。见[评测](docs/evaluation_CN.md)。

### 5. 使用 RJob 模式

RJob 模式仍使用同一个 launcher，但运行时资源由 RJob 提交：

```bash
python launcher.py \
  --mode rjob \
  --rjob-config config.yaml \
  --agent-config env/openrt/openrt_config.rjob.yaml \
  --agent-start-config env/openrt/openrt_start.rjob.yaml \
  --gateway-base-url http://YOUR_GATEWAY_HOST:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --storage-type cloud \
  --pool-size 8
```

全局 RJob 鉴权放在 `config.yaml` 或 `--rjob-config`。每个 agent 的镜像、资源、挂载、嵌入文件和运行命令放在 `--agent-start-config`。

## 数据与日志

默认本地路径：

| 产物 | 默认位置 |
|------|----------|
| SQLite 轨迹 DB | `env_trajs.db` |
| Launcher 日志 | `logs/<timestamp>/main.log` |
| Gateway 日志 | `logs/gateway.log` |
| Gateway 请求日志 | `logs/gateway_requests.jsonl` |
| Adapter 输出 | 通常在 `results/` 或 adapter 自己挂载的输出目录 |

表结构和查询示例见[数据管理器](docs/data-manager_CN.md)。

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
| [评测](docs/evaluation_CN.md)            | LLM judge、agent-eval、rule evaluator、markdown eval task 和 reward commit 行为。 |
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

1. 在 `env/<name>/` 下添加或更新自定义环境。
2. 同时提供 `<name>_config.yaml` 和 `<name>_start.yaml`，需要包含必须的字段。
3. 不要提交密钥和私有端点。
4. 使用 `launcher.py` 运行本地 smoke test。
5. 在 pull request 中说明安装步骤、预期输出和存储要求。

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
