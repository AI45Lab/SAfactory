<div align="center">

# SAfactory

<p align="center">
    <a href="README_CN.md">中文</a> &nbsp ｜ &nbsp English
</p>

**A next-generation agent infrastructure that integrates evaluation and training, supporting rapid agent onboarding, fast integration of community benchmarks, concurrent rollout execution, trajectory collection, and reinforcement learning training across domains such as OS, Android, Minecraft, embodied AI, QA, data processing, and scientific discovery. It is the first to validate a trustworthy scaling law for agents, achieving improved safety capabilities without an alignment tax.**

**The built-in Gateway is a session-aware OpenAI-compatible API layer that routes model requests to configured upstream LLM services, applies concurrency and step controls, and records request telemetry into the same trajectory storage used by rollouts.**

[Quick Start](#quick-start) |
[Demo](#demo) |
[Environments](docs/environments.md) |
[RL Training](docs/rl-training.md) |
[RJob Mode](docs/rjob-mode.md) |
[Sandbox Mode](docs/sandbox-mode.md) |
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

## <a id="why-SAfactory"></a>✨ Why SAfactory

SAfactory is an agent sandbox for teams that need one pipeline for evaluation, data generation, and RL training. It helps teams plug in new agents and community benchmarks quickly, run them concurrently through scalable rollout pools, route OpenAI-compatible model traffic through the Gateway, persist trajectories, and bridge completed data into Slime / GRPO training.

| Need | SAfactory provides |
|------|--------------------|
| Evaluate agents and benchmarks | Run LLM or VLM agents against realistic interactive tasks and community benchmarks, then collect rewards. |
| Build trajectory data | Persist messages, actions, observations, rewards, and environment state to SQLite. |
| Train with RL | Stream rollout trajectories into Slime through the built-in Buffer Server. |
| Add new agents and benches | Onboard agent runtimes and benchmark suites quickly, then scale them with concurrent rollout workers. |

Core features:

- Multi-domain agent and benchmark adapters: OS, Android, Minecraft, RoboTrustBench, Embodied ALFRED, QA, DABStep, DiscoveryWorld, DeepEyes, Geo3K-VL, and Math500.
- High-concurrency rollouts through runtime pools and async workers.
- OpenAI-compatible model integration for vLLM, SGLang, hosted APIs, and local proxies.
- Local Docker mode, remote RJob mode, and Brainbox Sandbox mode.

## <a id="demo"></a>🎬 Demo

<div align="center">

https://github.com/user-attachments/assets/4c551b27-ce4d-4fc8-8df6-d6dc8100cc88

*点击播放查看完整演示*

</div>

## <a id="quick-start"></a>🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/AI45Lab/SAfactory.git
cd SAfactory
pip install -r requirements.txt
```

Docker mode requires Docker and an agent image that matches the selected adapter.

### 2. Configure the Gateway

Copy the example and replace route placeholders with your own OpenAI-compatible model endpoint:

```bash
cp gateway/config.example.yaml gateway/config.local.yaml
```

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

### 3. Run Evaluation In Docker

Run a small Docker evaluation:

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

Important details:

- `--llm-model` is a gateway `llm_routes` key, not an arbitrary upstream model name.
- `--agent-config` defines tasks and datasets.
- `--agent-start-config` defines how the agent runtime starts.
- `--gateway-base-url` should point at the gateway session root.
- `--db-path` must match `gateway.storage_config.db_url` when `storage_type` is `sqlite`.
- `--enable-evaluation` discovers `rule_evaluator.py` by convention and commits scores.

### 4. Run RL Training In Docker

RL training reuses the same Docker runtime, but the Gateway is usually auto-started by Buffer Server and routed to the Slime generator's built-in LLM proxy. Edit `rl/examples/geo3k_vl/env.sh` first:

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

If the standalone Gateway from the evaluation smoke test is still running on the same port, stop it before starting Buffer Server. Keep `AIEVOBOX_GATEWAY_AUTOSTART=0` only when your external Gateway already routes `RL_MODEL` to the Slime LLM proxy and uses the same `AIEVOBOX_DB_URL`.

Start two terminals from the repository root:

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_slime_generator.sh
```

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_buffer_server.sh
```

The Slime generator hosts `rl/llm_proxy.py`; Buffer Server auto-generates `logs/gateway.rl.generated.yaml`, starts Gateway, launches Docker rollout collection, and serves completed groups through `/get_rollout_data`. For the full set of RL variables, see [RL Training](docs/rl-training.md).

Remote runtimes use the same config concepts but different allocation backends. See [RJob mode](docs/rjob-mode.md) and [Sandbox mode](docs/sandbox-mode.md).

## Optional: S3 + LanceDB Storage

Safactory can optionally persist trajectory and environment data to an S3-backed LanceDB data platform through `wt-data-platform-sdk`. SQLite remains the default local strategy; cloud dependencies are kept separately in `requirements-cloud.txt`.

The optional LanceDB/cloud dependency stack requires Python 3.10-3.12. Python 3.12 is the currently verified environment.

Install the optional dependencies:

```bash
pip install -r requirements-cloud.txt
```

Create a local `.env` file (do not commit credentials) with the data platform connection settings:

```bash
# production or test
WT_SDK_PROFILE=test
WT_SDK_DB_URI=s3://YOUR_DATA_DATABASE
WT_SDK_ENV_CONFIG_DB_URI=s3://YOUR_ENV_CONFIG_DATABASE
WT_SDK_S3_ENDPOINT=https://YOUR_S3_ENDPOINT
WT_SDK_S3_ALLOW_HTTP=true
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
AWS_EC2_METADATA_DISABLED=true
```

Load it into the process environment before starting Safactory:

```bash
set -a
source .env
set +a
```

Then set the gateway `storage_type` to `cloud` and launch Safactory with `--storage-type cloud`. The `production` profile selects the production landing/serving tables, while `test` selects the test tables. For complete configuration, table documentation, and instructions for querying and retrieving data, see [AI45Lab/wt-data-platform-sdk](https://github.com/AI45Lab/wt-data-platform-sdk).

## Run Data

Local runs write task rows and trajectories to `env_trajs.db` by default. Pass an explicit `--job-id geo3k-docker-smoke` when launching to make later querying, reproduction, and training filters clearer.

- `job_id`: one `launcher.py` run.
- `session_id`: one environment/task instance, matching `job_environments.env_id` and `session_steps.session_id`.

Find recent runs:

```bash
sqlite3 env_trajs.db "
  SELECT id, job_id, env_id AS session_id, env_name, group_id, finished, created_at
  FROM job_environments
  ORDER BY id DESC
  LIMIT 20;"
```

Inspect one session's steps, rewards, and completion state:

```bash
sqlite3 env_trajs.db "
  SELECT step_id, llm_model, step_reward, reward,
         is_terminal, is_session_completed, is_trainable, created_at
  FROM session_steps
  WHERE session_id = '<session-id>'
  ORDER BY step_id, id;"
```

Default local artifacts:

| Artifact | Default |
|----------|---------|
| SQLite trajectory DB | `env_trajs.db` |
| Launcher logs | `logs/<timestamp>/main.log` |
| Gateway log | `logs/gateway.log` |
| Gateway request log | `logs/gateway_requests.jsonl` |
| Adapter outputs | `results/` or adapter-mounted directories |

See [Data Manager](docs/data-manager.md) for the full schema, row types, and more queries.

## Documentation

| Guide | What it covers |
|-------|----------------|
| [Gateway](docs/gateway.md) | Gateway endpoints, routing, telemetry, request logs, and storage matching. |
| [Configuration](docs/configuration.md) | Current `launcher.py`, gateway, agent config, agent start config, and RJob fields. |
| [RJob Mode](docs/rjob-mode.md) | Remote RJob runtime setup, credentials, mounts, Gateway reachability, and Geo3K examples. |
| [Sandbox Mode](docs/sandbox-mode.md) | Brainbox Sandbox Environment setup, volumes, lifecycle, and launch flow. |
| [Supported Environments](docs/environments.md) | Checked-in v2 adapters and their runtime requirements. |
| [Evaluation](docs/evaluation.md) | Rule evaluator configuration and reward commit behavior. |
| [Data Manager](docs/data-manager.md) | SQLite/cloud storage behavior, tables, event types, and useful queries. |
| [Custom Runtime](docs/custom-environment.md) | How to add a v2 external agent runtime and the two required YAML files. |
| [RL Training](docs/rl-training.md) | Buffer Server and Slime integration details. |

## <a id="architecture"></a>🏗️ Architecture

![SAfactory architecture](fig/overview.png)

At a high level, `launcher.py` loads environment YAML files, starts or connects to environment services, sends observations to an OpenAI-compatible model endpoint, records every interaction through the data manager, and optionally forwards completed rollouts to RL training.

## Datasets

![tax](fig/tax.png)

SAfactory can generate reusable trajectory datasets. The public OS trajectory release is available on Hugging Face:

- [AI45Research/SATraj-OS](https://huggingface.co/datasets/AI45Research/SATraj-OS), a SAfactory-generated OS trajectory dataset for agent training and analysis.

SATraj-OS can be used for SFT. The SCOPE models trained with this pipeline improve the balance between capability and safety on OSWorld and OS-BLIND:

![SCOPE capability-safety joint scaling](fig/scope_capability_safety_aaai_trend.png)


## Contributing

Contributions are welcome for new custom environments, bug fixes, and reproducible examples.

Each environment lives in its own subdirectory under `env/`. To add one:

1. Create a new `env/<name>/` directory.
2. Provide `dataset/`; JSONL datasets should contain one independently scheduled task per line.
3. Provide both `<name>_config.yaml` and `<name>_start.yaml`, including the required Docker image.
4. Add the runtime entry script required by the environment, usually named `runner` such as `runner.py` or `runner.mjs`.
5. Implement `rule_evaluator.py` when evaluation needs one.
6. Run a local smoke test with `launcher.py`.

See [Custom Environments](docs/custom-environment.md) for the full guide.

## Citation

If SAfactory or SAfactory-generated datasets are useful in your work, cite the repository and the specific dataset or report you used.

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
