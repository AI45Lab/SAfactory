<div align="center">

# SAfactory

<p align="center">
    中文 &nbsp;|&nbsp; <a href="README.md">English</a>
</p>

**SAfactory 是面向智能体评测、轨迹采集和强化学习训练的可扩展基础设施。它把 agent 和 benchmark 环境作为外部 runtime 调度，通过具备 session 感知能力的 OpenAI 兼容 Gateway 统一路由模型调用，记录轨迹，并把完成的 rollout 数据送入 Slime 等训练系统。**

<p align="center">
  <a href="#why-safactory">为什么使用 SAfactory</a> •
  <a href="#demo">演示</a> •
  <a href="#agent-skill">Agent Skill</a> •
  <a href="#quick-start">快速开始</a> •
  <a href="#documentation">详细文档</a> •
  <a href="#citation">引用</a>
</p>

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Execution](https://img.shields.io/badge/mode-docker%20%7C%20rjob%20%7C%20sandbox-orange)
![LLM](https://img.shields.io/badge/LLM-OpenAI--compatible-purple)

</div>

---

## <a id="why-safactory"></a>✨ 为什么使用 SAfactory

SAfactory 提供一条统一链路，用于 agent 接入、benchmark 接入、评测、rollout 数据生成和 RL 训练。

核心运行契约是：

- dataset 的一行会成为一个被调度的 episode；
- 每个 episode 都有独立的 `session_id` 和 Gateway session；
- runtime 通过 Gateway 调用目标模型；
- runtime 返回一个 JSON result；
- `rule_evaluator.py` 会把 runtime 输出和轨迹数据转换成 SAfactory 格式；
- 已完成且可训练的轨迹可由 RL Buffer Server 消费或落盘到数据库中固定为数据资产。

## <a id="demo"></a>🎬 演示

<div align="center">

https://github.com/user-attachments/assets/4c551b27-ce4d-4fc8-8df6-d6dc8100cc88

*点击播放查看完整演示*

</div>

## <a id="agent-skill"></a>🧩 Agent Skill 快速上手

仓库内置了一个轻量 Agent skill，用于帮助 Agent 按标准 workflow 使用 SAfactory：

```text
skills/safactory-workflows/SKILL.md
```

它覆盖三类高频请求：

- 接入新的 benchmark 或自定义环境到 SAfactory；
- 用 Docker 模式运行指定环境的测评；
- 启动指定环境的 GRPO / RL 训练。

使用 Agent 时，可以直接这样提问：

```text
请使用 skills/safactory-workflows，帮我把这个 benchmark 接入 SAfactory。
```

```text
请使用 safactory-workflows skill，用 Docker 模式跑 geo3k 测评。
```

```text
请使用 safactory-workflows skill，启动 my_env 环境的 GRPO 训练。
```

该 skill 不替代文档，而是引导 Agent 按需读取 `docs/guides/`、`docs/reference/` 和根 README，并优先参考标准环境 `env/geo3k/`。如果你的 Agent 支持本地 skill 搜索，可以把 `skills/safactory-workflows/` 加入其 skill 搜索路径；否则在请求中显式写出该路径即可。

## <a id="quick-start"></a>🚀 快速开始

### 1. SAfactory 安装与 Gateway 配置

运行 Geo3K 前，请先按[标准环境：Geo3K](docs/reference/environments_CN.md#standard-environment-geo3k)准备 runtime 镜像和数据集。

安装 SAfactory：

```bash
git clone https://github.com/AI45Lab/SAfactory.git
cd SAfactory
pip install -r requirements.txt
```

本地可以运行Docker，并且当前用户可以执行 `docker build`、`docker run` 和 `docker exec`

创建本地 Gateway 配置：

```bash
cp gateway/config.example.yaml gateway/config.local.yaml
```

编辑 `gateway/config.local.yaml`，显式设置存储路径和模型 route：

```yaml
listen_host: 0.0.0.0
listen_port: 8000
base_session_path: /v1/sessions
max_steps: -1

storage_type: sqlite
storage_config:
  db_url: sqlite://env_trajs.db

llm_routes:
  geo3k_model:
    base_url: http://YOUR_LLM_HOST/v1
    api_key: YOUR_API_KEY
    supports_stream: true
    max_concurrency: 64
```

启动 Gateway：

```bash
python -m gateway --config gateway/config.local.yaml
```

另开一个终端检查 ready 状态：

```bash
curl http://127.0.0.1:8000/readyz
```

### 2. 用 Geo3K 完成最小评测

通过 Docker 模式运行一个最小 Geo3K 评测：

```bash
python launcher.py \
  --mode docker \
  --agent-config env/geo3k/geo3k_config.yaml \
  --agent-start-config env/geo3k/geo3k_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model geo3k_model \
  --enable-evaluation \
  --db-path sqlite://env_trajs.db \
  --job-id geo3k-docker-smoke \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 10
```

`--llm-model` 必须匹配 `llm_routes` 中的 key。加上 `--enable-evaluation` 后，SAfactory 会调用 `env/geo3k/rule_evaluator.py` 并写入最终 reward。

### 3. 用 Geo3K 完成最小训练

Geo3K 训练使用 `rl/` 下的 RL bridge，以及示例配置 `rl/examples/geo3k_vl/env.sh`。

启动前先编辑 `rl/examples/geo3k_vl/env.sh`，或在每个启动终端里导出同一组变量。设置必要本地路径，并把首次运行规模调小：

```bash
export AIEVOBOX_ROOT=$(pwd)
export AIEVOBOX_MODE=docker
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.yaml
export AIEVOBOX_GATEWAY_HOST=127.0.0.1
export AIEVOBOX_GATEWAY_PORT=8000
export RL_MODEL=geo3k_model
export AIEVOBOX_POOL_SIZE=2
export RL_GROUP_SIZE=2
export RL_EPOCH=1
export HF_CKPT_DIR=/path/to/hf-checkpoint
export SLIME_HOME=/path/to/slime
export MEGATRON_HOME=/path/to/Megatron-LM
```

然后在仓库根目录启动两个进程：

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_slime_generator.sh
```

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_buffer_server.sh
```

RL 训练侧的 reward 来自数据库中的已完成轨迹：`rule_evaluator.py` 会在评测完成后写入 reward，Buffer Server 再从与 Launcher / Gateway 一致的数据库中抓取带有 reward 的轨迹数据，并提供给训练进程消费。

Buffer Server 可以自动启动一个 Gateway，并把 `RL_MODEL` 路由到 Slime 托管的 LLM proxy。如果同一端口上已有手动启动的 Gateway，请先停止它；只有当外部 Gateway 已经具备正确 route 和存储配置时，才设置 `AIEVOBOX_GATEWAY_AUTOSTART=0`。

## <a id="documentation"></a>📚 文档索引

### Guides

| 指南 | 内容 |
|------|------|
| [环境接入](docs/guides/custom-environment_CN.md) | 如何接入新的外部 runtime adapter。 |
| [测评](docs/guides/evaluation_CN.md) | Rule evaluator 发现、接口、reward 写入行为和 Geo3K 评测。 |
| [RL 训练](docs/guides/rl-training_CN.md) | Buffer Server、Slime generator、Geo3K 训练路径和关键变量。 |
| [数据查询](docs/guides/data-manager_CN.md) | SQLite 存储行为、表结构、行类型和查询示例。 |
| [存储切换](docs/guides/S3+LanceDB-storage_CN.md) | 如何从本地 SQLite 切换到 S3 + LanceDB 轨迹存储。 |

### Internal

| 内部文档 | 内容 |
|----------|------|
| [RJob 模式](docs/internal/rjob-mode_CN.md) | 远程 RJob runtime 配置、鉴权、挂载、Gateway 可达性和 Geo3K 示例。 |
| [Sandbox 模式](docs/internal/sandbox-mode_CN.md) | Brainbox Sandbox Environment 配置、volume、生命周期和启动流程。 |

### Reference

| 参考 | 内容 |
|------|------|
| [CLI 与配置参数](docs/reference/configuration_CN.md) | Launcher 参数、Gateway 配置、agent config、start config、RJob 和 Sandbox 设置。 |
| [支持的环境](docs/reference/environments_CN.md) | 仓库内置 adapter、标准 Geo3K 路径和 runtime 矩阵。 |
| [Gateway 参考](docs/reference/gateway_CN.md) | OpenAI 兼容路由、session 端点、telemetry、请求日志和存储一致性。 |
| [报告](https://arxiv.org/pdf/2605.06230) | SAfactory report。 |

## <a id="citation"></a>📖 引用

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
