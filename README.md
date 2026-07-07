<div align="center">

# Safactory

<p align="center">
    <a href="README_CN.md">中文</a> &nbsp ｜ &nbsp English
</p>

**A next-generation agent infrastructure that integrates evaluation and training, supporting rapid agent onboarding, fast integration of community benchmarks, concurrent rollout execution, trajectory collection, and reinforcement learning training across domains such as OS, Android, Minecraft, embodied AI, QA, data processing, and scientific discovery. It is the first to validate a trustworthy scaling law for agents, achieving improved safety capabilities without an alignment tax.**

[Quick Start](#quick-start) |
[Demo](#demo) |
[Environments](docs/environments.md) |
[RL Training](docs/rl-training.md) |
[Custom Environments](docs/custom-environment.md) |
[Configuration](docs/configuration.md) |
[Data](docs/data-manager.md) |
[Report](https://arxiv.org/pdf/2605.06230)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Execution](https://img.shields.io/badge/mode-local%20%7C%20remote-orange)
![LLM](https://img.shields.io/badge/LLM-OpenAI--compatible-purple)

</div>

---

## <a id="why-safactory"></a>✨ Why Safactory

![tax](fig/tax.png)

Safactory is an agent sandbox for teams that need one pipeline for evaluation, data generation, and RL training. It helps teams plug in new agents and community benchmarks quickly, run them concurrently through scalable rollout pools, route OpenAI-compatible model traffic, persist trajectories, and bridge completed data into Slime / GRPO training.

| Need | Safactory provides |
|------|--------------------|
| Evaluate agents and benchmarks | Run LLM or VLM agents against realistic interactive tasks and community benchmarks, then collect rewards. |
| Build trajectory data | Persist messages, actions, observations, rewards, and environment state to SQLite. |
| Train with RL | Stream rollout trajectories into Slime through the built-in Buffer Server. |
| Add new agents and benches | Onboard agent runtimes and benchmark suites quickly, then scale them with concurrent rollout workers. |

Core features:

- Multi-domain agent and benchmark adapters: OS, Android, Minecraft, RoboTrustBench, Embodied ALFRED, QA, DABStep, DiscoveryWorld, DeepEyes, Geo3K-VL, and Math500.
- High-concurrency rollouts through runtime pools and async workers.
- OpenAI-compatible model integration for vLLM, SGLang, hosted APIs, and local proxies.
- Local single-machine mode and remote RayJob-backed cluster mode.
- Optional experience extraction and prompt-time experience injection.

## <a id="demo"></a>🎬 Demo

<div align="center">

https://github.com/user-attachments/assets/4c551b27-ce4d-4fc8-8df6-d6dc8100cc88

*点击播放查看完整演示*

</div>

## <a id="quick-start"></a>🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/AI45Lab/Safactory.git
cd Safactory
pip install -r requirements.txt
```

If you want to use LanceDB/cloud storage features, install the optional cloud dependencies as well:

```bash
pip install -r requirements-cloud.txt
```

Docker mode requires Docker and an agent image that matches the selected adapter. RJob mode additionally requires a valid RJob client configuration.

### 2. Configure the Gateway

Copy the example and replace route placeholders with your own OpenAI-compatible model endpoint:

```bash
cp gateway/config.example.yaml gateway/config.local.yaml
```

In `gateway/config.local.yaml`, make sure the gateway and launcher share the same SQLite DB:

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

Start the gateway:

```bash
python -m gateway --config gateway/config.local.yaml
```

Check readiness in another terminal:

```bash
curl http://127.0.0.1:8000/readyz
```

### 3. Run One Agent Config

This example runs the checked-in OpenClaw adapter in Docker mode:

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

Important details:

- `--llm-model` is a gateway `llm_routes` key, not an arbitrary upstream model name.
- `--agent-config` defines tasks and datasets.
- `--agent-start-config` defines how the agent runtime is started.
- `--gateway-base-url` should point at the gateway session root.
- `--db-path` must match `gateway.storage_config.db_url` when `storage_type` is `sqlite`.

### 4. Run With Evaluation

Enable evaluator flow after rollout:

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

Evaluation specs can come from `env_params.eval`, `env_params.evaluation.specs`, markdown files under `env/<agent>/eval_tasks/<dataset>/`, or a rule evaluator file. See [Evaluation](docs/evaluation.md).

### 5. Run RJob Mode

RJob mode uses the same launcher but replaces Docker allocation with RJob submission:

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

Global RJob auth belongs in `config.yaml` or `--rjob-config`. Per-agent image, resources, mounts, embedded files, and run command belong in `--agent-start-config`.

## Data And Logs

Default local paths:

| Artifact | Default |
|----------|---------|
| SQLite trajectory DB | `env_trajs.db` |
| Launcher logs | `logs/<timestamp>/main.log` |
| Gateway log | `logs/gateway.log` |
| Gateway request log | `logs/gateway_requests.jsonl` |
| Adapter outputs | Usually `results/` or adapter-specific mounted output directories |

Use [Data Manager](docs/data-manager.md) for table details and query examples.

## RL Training

Safactory can feed Slime through `rl/buffer_server.py`. The current RL scripts live under `rl/examples/<task>/` and source task-specific `env.sh` files:

```bash
cd rl/examples/math500
./run_buffer_server.sh
```

The Buffer Server starts `launcher.py`, reads completed trainable rows, groups samples by `group_id`, and exposes batches through `/get_rollout_data`. See [RL Training](docs/rl-training.md).

## Documentation

| Guide | What it covers |
|-------|----------------|
| [Gateway](docs/gateway.md) | Gateway endpoints, routing, telemetry, request logs, and storage matching. |
| [Configuration](docs/configuration.md) | Current `launcher.py`, gateway, agent config, agent start config, and RJob fields. |
| [Supported Environments](docs/environments.md) | Checked-in v2 adapters and their runtime requirements. |
| [Evaluation](docs/evaluation.md) | LLM judge, agent-eval, rule evaluator, markdown eval tasks, and reward commit behavior. |
| [Data Manager](docs/data-manager.md) | SQLite/cloud storage behavior, tables, event types, and useful queries. |
| [Custom Runtime](docs/custom-environment.md) | How to add a v2 external agent runtime and the two required YAML files. |
| [RL Training](docs/rl-training.md) | Buffer Server and Slime integration details. |

## <a id="architecture"></a>🏗️ Architecture

![Safactory architecture](fig/overview.png)

At a high level, `launcher.py` loads environment YAML files, starts or connects to environment services, sends observations to an OpenAI-compatible model endpoint, records every interaction through the data manager, and optionally forwards completed rollouts to RL training.

## Datasets

Safactory can generate reusable trajectory datasets. The public OS trajectory release is available on Hugging Face:

- [AI45Research/SATraj-OS](https://huggingface.co/datasets/AI45Research/SATraj-OS), a Safactory-generated OS trajectory dataset for agent training and analysis.

## Contributing

Contributions are welcome for new custom environments, bug fixes, and reproducible examples.

1. Add or update a custom environment under `env/<name>/`.
2. Provide both `<name>_config.yaml` and `<name>_start.yaml`, including all required fields.
3. Keep secrets and private endpoints out of committed configs.
4. Run a local smoke test with `launcher.py`.
5. Include setup notes, expected outputs, and storage requirements in the pull request.

## Citation

If Safactory or Safactory-generated datasets are useful in your work, cite the repository and the specific dataset or report you used.

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
