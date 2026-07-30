<div align="center">

# SAfactory

<p align="center">
    <a href="README_CN.md">中文</a> &nbsp;|&nbsp; English
</p>

**SAfactory is a scalable infrastructure for agent evaluation, trajectory collection, and reinforcement learning training. It schedules agents and benchmark environments as external runtimes, routes model calls through a session-aware OpenAI-compatible Gateway, records trajectories, and feeds completed rollouts into training systems such as Slime.**

<p align="center">
  <a href="#why-safactory">Why SAfactory</a> •
  <a href="#demo">Demo</a> •
  <a href="#agent-skill">Agent Skill</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#citation">Citation</a>
</p>

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Execution](https://img.shields.io/badge/mode-docker%20%7C%20rjob%20%7C%20sandbox-orange)
![LLM](https://img.shields.io/badge/LLM-OpenAI--compatible-purple)

</div>

---

## <a id="why-safactory"></a>✨ Why SAfactory

SAfactory provides one unified workflow for agent onboarding, benchmark onboarding, evaluation, rollout data generation, and RL training.

The core runtime contract is:

- one dataset row becomes one scheduled episode;
- each episode gets its own `session_id` and Gateway session;
- the runtime calls the target model through the Gateway;
- the runtime returns one JSON result;
- `rule_evaluator.py` converts runtime output and trajectory data into the SAfactory format;
- completed trainable trajectories can be consumed by the RL Buffer Server or persisted as reusable data assets.

## <a id="demo"></a>🎬 Demo

<div align="center">

https://github.com/user-attachments/assets/4c551b27-ce4d-4fc8-8df6-d6dc8100cc88

*Click to watch the full demo*

</div>

## <a id="agent-skill"></a>🧩 Agent Skill Quick Start

This repository includes a lightweight Agent skill that helps agents use SAfactory through the standard workflows:

```text
skills/safactory-workflows/SKILL.md
```

It covers three common requests:

- onboard a new benchmark or custom environment into SAfactory;
- run Docker-mode evaluation for a selected environment;
- start GRPO / RL training for a selected environment.

When working with an Agent, use prompts such as:

```text
Use skills/safactory-workflows to help me onboard this benchmark into SAfactory.
```

```text
Use the safactory-workflows skill to run geo3k evaluation in Docker mode.
```

```text
Use the safactory-workflows skill to start GRPO training for my_env.
```

The skill does not replace the docs. It guides the Agent to read `docs/guides/`, `docs/reference/`, and the root README as needed, while using the standard `env/geo3k/` environment as the reference implementation. If your Agent supports local skill discovery, add `skills/safactory-workflows/` to its skill search path; otherwise mention this path explicitly in the request.

## <a id="quick-start"></a>🚀 Quick Start

### 1. SAfactory Installation And Gateway Configuration

Before running Geo3K, prepare the runtime image and dataset as described in [Standard Environment: Geo3K](docs/reference/environments.md#standard-environment-geo3k).

Install SAfactory:

```bash
git clone https://github.com/AI45Lab/SAfactory.git
cd SAfactory
pip install -r requirements.txt
```

Docker must be available locally, and the current user must be able to run `docker build`, `docker run`, and `docker exec`.

Create a local Gateway config:

```bash
cp gateway/config.example.yaml gateway/config.local.yaml
```

Edit `gateway/config.local.yaml` and explicitly set the storage path and model route:

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

Start the Gateway:

```bash
python -m gateway --config gateway/config.local.yaml
```

Check readiness from another terminal:

```bash
curl http://127.0.0.1:8000/readyz
```

### 2. Minimal Evaluation With Geo3K

Run a minimal Geo3K evaluation in Docker mode:

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

`--llm-model` must match a key under `llm_routes`. With `--enable-evaluation`, SAfactory calls `env/geo3k/rule_evaluator.py` and writes the final reward.

### 3. Minimal Training With Geo3K

Geo3K training uses the RL bridge under `rl/` and the example config at `rl/examples/geo3k_vl/env.sh`.

Before starting, edit `rl/examples/geo3k_vl/env.sh`, or export the same variables in each terminal. Set the required local paths and scale the first run down:

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

Then start two processes from the repository root:

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_slime_generator.sh
```

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_buffer_server.sh
```

Rewards for RL training come from completed trajectories in the database: `rule_evaluator.py` writes the reward after evaluation, and Buffer Server fetches trajectory rows with rewards from the same database used by Launcher / Gateway before serving them to the training process.

Buffer Server can automatically start a Gateway and route `RL_MODEL` to the Slime-hosted LLM proxy. If a manually started Gateway is already using the same port, stop it first. Set `AIEVOBOX_GATEWAY_AUTOSTART=0` only when the external Gateway already has the correct route and storage configuration.

## <a id="documentation"></a>📚 Documentation

### Guides

| Guide | What it covers |
|-------|----------------|
| [Custom Environments](docs/guides/custom-environment.md) | How to onboard a new external runtime adapter. |
| [Evaluation](docs/guides/evaluation.md) | Rule evaluator discovery, interfaces, reward writing, and Geo3K evaluation. |
| [RL Training](docs/guides/rl-training.md) | Buffer Server, Slime generator, Geo3K training path, and key variables. |
| [Data Manager](docs/guides/data-manager.md) | SQLite storage behavior, table schema, row types, and query examples. |
| [S3 + LanceDB Storage](docs/guides/S3+LanceDB-storage.md) | How to switch trajectory storage from local SQLite to S3 + LanceDB. |

### Internal

| Internal Doc | What it covers |
|--------------|----------------|
| [RJob Mode](docs/internal/rjob-mode.md) | Remote RJob runtime config, authentication, mounts, Gateway reachability, and Geo3K examples. |
| [Sandbox Mode](docs/internal/sandbox-mode.md) | Brainbox Sandbox Environment config, volumes, lifecycle, and launch flow. |

### Reference

| Reference | What it covers |
|-----------|----------------|
| [CLI And Configuration](docs/reference/configuration.md) | Launcher flags, Gateway config, agent config, start config, RJob, and Sandbox settings. |
| [Supported Environments](docs/reference/environments.md) | Checked-in adapters, the standard Geo3K path, and the runtime matrix. |
| [Gateway Reference](docs/reference/gateway.md) | OpenAI-compatible routes, session endpoints, telemetry, request logs, and storage consistency. |
| [Report](https://arxiv.org/pdf/2605.06230) | SAfactory report. |

## <a id="citation"></a>📖 Citation

If SAfactory or SAfactory-generated datasets are useful in your work, please cite this repository and the specific dataset or report you used.

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
