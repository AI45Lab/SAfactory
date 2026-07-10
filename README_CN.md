<div align="center">

# Safactory

<p align="center">
    中文 &nbsp ｜ &nbsp <a href="README.md">English</a>
</p>

**测训一体的下一代智能体基础设施，支持在 OS、Android、Minecraft、具身智能、QA、数据处理、科学发现等多类环境中评测智能体、采集轨迹，并进行强化学习训练。首次验证智能体可信Scaling Law，实现安全能力提升且无对齐税。**

[快速开始](#quick-start) |
[演示](#demo) |
[环境](docs/environments_CN.md) |
[RL 训练](rl/README_CN.md) |
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

Safactory 是面向需要统一完成评测、数据生成和 RL 训练的团队的智能体沙箱。它提供统一的环境接口、并发 rollout 管理、OpenAI 兼容模型访问、轨迹持久化，以及面向 [Slime](https://github.com/THUDM/slime) 框架的 Buffer Server 桥接。

| 需求 | Safactory 提供 |
|------|----------------|
| 评测智能体 | 在真实交互环境中运行 LLM 或 VLM 智能体并收集奖励。 |
| 构建轨迹数据 | 将消息、动作、观察、奖励和环境状态持久化到 SQLite。 |
| RL 训练 | 通过内置 Buffer Server 将 rollout 轨迹流式送入 Slime。 |
| 添加新环境 | 通过标准接口接入新的环境。 |

核心能力：

- 多领域环境：OS、Android、Minecraft、RoboTrustBench、Embodied ALFRED、QA、DABStep、DiscoveryWorld、DeepEyes、Geo3K-VL 和 Math500。
- 通过环境池和异步 worker 支持高并发 rollout。
- 支持 vLLM、SGLang、托管 API 和本地代理等 OpenAI 兼容模型服务。
- 支持本地单机模式和基于 RayJob 的远程集群模式。
- 可选的经验抽取和 prompt 时经验注入。

## <a id="demo"></a>🎬 演示

<div align="center">

https://github.com/user-attachments/assets/4c551b27-ce4d-4fc8-8df6-d6dc8100cc88

*点击播放查看完整演示*

</div>

## <a id="quick-start"></a>🚀 快速开始

### 安装

```bash
git clone https://github.com/AI45Lab/Safactory.git
cd Safactory
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

根目录 `requirements.txt` 只安装 Safactory 基础运行依赖，包括 launcher、manager、data manager、OpenAI 兼容客户端、Ray/RayJob 集成和 Buffer Server 依赖。它不会安装 Slime / Megatron / SGLang 训练栈，也不会安装各环境的重型依赖。

进行 RL 训练前，请单独准备 Slime 环境：

- Conda 环境：参考 Slime 的 [`build_conda.sh`](https://github.com/THUDM/slime/blob/main/build_conda.sh)。
- Docker 环境：参考 Slime 的 [Docker quick start](https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md)。

部分环境有额外运行时依赖。只需要为实际运行的环境安装 `env/<name>/requirements.txt`；运行 Docker、模拟器、虚拟机或仿真器任务前，请先查看[支持的环境](docs/environments_CN.md)。

### 评测模型

```bash
python launcher.py \
  --env-config env/osgym/os_config.yaml \   # 选择评测环境（OS / Android / Minecraft 等）
  --llm-base-url http://YOUR_LLM_HOST/v1 \  # 模型服务地址
  --llm-api-key YOUR_API_KEY \              # API Key
  --llm-model YOUR_MODEL \                  # 模型名称
  --pool-size 500                           # 并发智能体实例数
```

该命令会启动 runner，加载选定的环境配置，调度任务，调用模型端点，并将 step 级记录写入 SQLite。

### 采集轨迹数据

每次 rollout 都会自动记录。默认 CLI 数据库路径为 `sqlite://env_trajs.db`；可以用 `--db-path` 覆盖：

```bash
python launcher.py \
  --env-config env/osgym/os_config.yaml \
  --db-path sqlite://runs/os_eval.db \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL
```

表结构和查询示例见[数据管理器](docs/data-manager_CN.md)。

### 使用 RL 训练

Safactory 通过 Buffer Server 与 [Slime](https://github.com/THUDM/slime) 集成：

```bash
# 终端 1：Slime 训练进程
source rl/examples/osgym/env.sh
bash rl/run_slime_generator_common.sh

# 终端 2：Safactory Buffer Server 和 rollout runner
source rl/examples/osgym/env.sh
bash rl/run_buffer_server_common.sh
```

完整说明见 [RL 训练](rl/README_CN.md)。

## <a id="datasets"></a>📦 数据集

![tax](fig/tax.png)

Safactory 可以生成可复用的轨迹数据集。公开 OS 轨迹发布在 Hugging Face：

- [AI45Research/SATraj-OS](https://huggingface.co/datasets/AI45Research/SATraj-OS)，一个由 Safactory 生成、用于智能体训练和分析的 OS 轨迹数据集。

使用 Safactory 生成的 OS 轨迹训练模型，可以在任务能力和安全 benchmark 上同时取得提升。以 **Qwen3-VL-8B** 为基座，在 SATraj-OS 上进行监督微调后，OSWorld 成功率从 **14.40%** 提升到 **30.19%**；继续进行 RL 训练后达到 **36.29%**，相比基座模型绝对提升 **+21.89 pp**。

同一批训练数据也提升了安全性，而不是用安全下降换取能力提升。在 OS-Harm 上，SATraj-OS SFT 将安全分数从 **68.67** 提升到 **96.67**；RL 训练后的模型安全分数为 **91.33**，仍显著高于基座模型。这说明 Safactory 数据可以支持智能体能力和安全性的共同提升。

<table>
  <tr>
    <td width="50%"><img src="fig/osworld.PNG" alt="OSWorld success rate comparison with SA-OS training improvements"></td>
    <td width="50%"><img src="fig/osharm.PNG" alt="OS-Harm safety comparison across agent LLMs"></td>
  </tr>
</table>

## <a id="documentation"></a>📚 文档

| 指南 | 内容 |
|------|------|
| [配置](docs/configuration_CN.md) | CLI 参数、manager YAML 和环境 YAML 格式。 |
| [支持的环境](docs/environments_CN.md) | 环境注册名、前置依赖和安装链接。 |
| [数据管理器](docs/data-manager_CN.md) | SQLite 表结构、存储行为和查询示例。 |
| [RL 训练](rl/README_CN.md) | Slime 集成、Buffer Server 设置和 RL 变量。 |
| [自定义环境](docs/custom-environment_CN.md) | 最小 `BaseEnv` 实现和注册流程。 |
| [经验抽取与注入](docs/experience-extraction-injection_CN.md) | 将历史轨迹作为 prompt 时经验复用。 |

## <a id="architecture"></a>🏗️ 架构

![Safactory architecture](fig/overview.png)

整体上，`launcher.py` 会加载环境 YAML 文件，启动或连接环境服务，将观察发送到 OpenAI 兼容模型端点，通过数据管理器记录每次交互，并可选择将完成的 rollout 转发给 RL 训练。

## <a id="contributing"></a>🤝 贡献

欢迎贡献新环境、bug 修复、文档改进和可复现实例。

1. Fork 仓库。
2. 在 `env/<name>/` 下添加或更新环境。
3. 包含 YAML 配置和简短 README，说明环境特定依赖。
4. 使用 `launcher.py` 运行本地 smoke test。
5. 打开 pull request，并说明安装步骤和预期行为。

## <a id="citation"></a>📝 引用

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
