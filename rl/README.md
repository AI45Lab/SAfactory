# Safactory RL Usage Guide

This document explains how to run RL training in Safactory with local mode. The local workflow includes one Safactory repository, local or reachable environment services, a SQLite trajectory database, Buffer Server, and one Slime training process.

For complex environments, local mode can still connect to HTTP services or Docker containers. RayJob is a separate remote deployment mode and is not required for local RL training.

## Components

| Component | Role |
| --- | --- |
| `rl/examples/<env>/env.sh` | Experiment configuration. Source this file before starting any RL process. |
| `rl/run_buffer_server.sh` | Starts Safactory Buffer Server and the rollout runner. |
| `rl/run_slime_generator.sh` | Starts Slime training and the Safactory rollout function. |
| `rl/buffer_server.py` | Starts rollout collection, reads completed trajectories from SQLite, aggregates sample groups, and serves them to the generator. |
| `rl/slime_generator.py` | Slime rollout function. It starts the LLM proxy, fetches trajectory groups from Buffer Server, builds masks/rewards, and returns Slime samples. |
| Slime / Megatron / SGLang | Training and inference stack. Install it separately according to the Slime environment or Docker instructions. |

## Workflow

### Configure `env.sh`

Each example keeps its runtime configuration in `rl/examples/<env>/env.sh`, and each environment's experiment configuration should be maintained in its own `env.sh`. Check at least the following variables before running:

```bash
export AIEVOBOX_ROOT=/path/to/Safactory
export AIEVOBOX_MODE=local
export AIEVOBOX_ENV_CONFIG=/path/to/env_config.yaml
export AIEVOBOX_DB_URL=sqlite:////path/to/rollout.db
export AIEVOBOX_ENV_TRANSPORT=inproc

export BUFFER_SERVER_HOST=127.0.0.1
export BUFFER_SERVER_PORT=18889
export LLM_PROXY_HOST=127.0.0.1
export LLM_PROXY_PORT=18890

export SLIME_HOME=/path/to/slime
export MEGATRON_HOME=/path/to/Megatron-LM
export HF_CKPT_DIR=/path/to/hf/checkpoint
export LOAD_DIR=
export SAVE_DIR=/path/to/save/checkpoints
```

Configuration suggestions:

- Use `AIEVOBOX_MODE=local`.
- Use `AIEVOBOX_ENV_CONFIG` to specify one environment YAML, or use `AIEVOBOX_ENV_ROOT` to specify a set of YAML files.
- Use `AIEVOBOX_DB_URL=sqlite:////absolute/path/to/file.db` to store trajectories.
- Use `AIEVOBOX_ENV_TRANSPORT=inproc` when the Python environment can run directly inside the Safactory process.
- Use `AIEVOBOX_ENV_TRANSPORT=http` when the environment is served by an independent process or container.

Configure rollout and training parallelism:

```bash
export AIEVOBOX_POOL_SIZE=2
export AIEVOBOX_LLM_MAX_CONCURRENCY=2
export AIEVOBOX_MAX_STEPS=10
export AIEVOBOX_MESSAGE_CUT=1

export RL_GROUP_SIZE=2
export RL_ROLLOUT_GROUP_BATCH_SIZE=1
export RL_GLOBAL_BATCH_SIZE=2
export RL_EPOCH=10
export NUM_ROLLOUT=8

export NUM_GPUS=2
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE=1
export ROLLOUT_NUM_GPUS=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=1
export TP_SIZE=1
```

Configuration suggestions:

- `AIEVOBOX_POOL_SIZE` is the number of concurrent environment instances.
- `AIEVOBOX_LLM_MAX_CONCURRENCY` limits concurrent requests to the generator's built-in LLM proxy.
- `RL_GROUP_SIZE` corresponds to Slime's `n_samples_per_prompt`.
- `RL_ROLLOUT_GROUP_BATCH_SIZE` controls how many completed groups each rollout batch requests.
- `RL_GLOBAL_BATCH_SIZE` is the Slime training global batch size.
- `NUM_GPUS`, `ACTOR_NUM_GPUS_PER_NODE`, `ROLLOUT_NUM_GPUS`, and `ROLLOUT_NUM_GPUS_PER_ENGINE` must match the number of GPUs visible to Ray and SGLang.

### Start Training

Open two terminals from the Safactory repository root.

Terminal 1 starts Slime training and the Safactory generator:

```bash
source rl/examples/geo3k_vl/env.sh
bash rl/run_slime_generator.sh
```

Terminal 2 starts Buffer Server and rollout collection:

```bash
source rl/examples/geo3k_vl/env.sh
bash rl/run_buffer_server.sh
```

The generator starts the LLM proxy first. Buffer Server starts the rollout runner, writes trajectories to SQLite, aggregates completed sessions, and serves these groups to the generator.

## Key Variables

| Variable | Description |
| --- | --- |
| `AIEVOBOX_ROOT` | Safactory repository path. |
| `AIEVOBOX_MODE` | Use `local` for local RL. |
| `AIEVOBOX_ENV_CONFIG` | Single environment YAML file. |
| `AIEVOBOX_ENV_ROOT` | Directory containing multiple environment YAML files. |
| `AIEVOBOX_ENV_TRANSPORT` | `inproc` or `http`. |
| `AIEVOBOX_DB_URL` | Trajectory database URL. SQLite is recommended for local mode. |
| `AIEVOBOX_POOL_SIZE` | Number of concurrent environment instances. |
| `AIEVOBOX_MAX_STEPS` | Maximum number of environment steps per episode. |
| `AIEVOBOX_MESSAGE_CUT` | Number of recent turns kept in the prompt. `0` keeps all turns. |
| `RL_GROUP_SIZE` | Number of samples per prompt. |
| `RL_ROLLOUT_GROUP_BATCH_SIZE` | Number of completed groups requested by each rollout batch. |
| `RL_GLOBAL_BATCH_SIZE` | Slime global batch size. |
| `RL_OFF_BY_N` | Maximum allowed policy-version lag. |
| `DAPO_filter` | Whether to discard groups whose rewards are all identical. |
| `BUFFER_SERVER_HOST` / `BUFFER_SERVER_PORT` | Address used by the generator to connect to Buffer Server. |
| `LLM_PROXY_HOST` / `LLM_PROXY_PORT` | Address used by rollout workers to connect to the generator's built-in LLM proxy. |
| `AIEVOBOX_BUFFER_INCOMPLETE_GROUP_TTL_SECONDS` | Timeout for incomplete pending groups in Buffer Server. |
