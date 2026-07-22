# Safactory RL Usage Guide

This document explains the standard RL training entrypoints in Safactory. Runtime logic lives in the shared scripts under `rl/`; each example only needs an `env.sh` configuration file.

## Components

| Component | Role |
| --- | --- |
| `rl/examples/<env>/env.sh` | Environment and experiment configuration. Switch environments by sourcing or passing a different file. |
| `rl/run_buffer_server.sh` | Starts Buffer Server and the Safactory rollout runner. |
| `rl/run_slime_generator.sh` | Starts Ray, Slime training, SGLang rollout engines, and the Safactory rollout function. |
| `rl/buffer_server.py` | Starts rollout collection, reads completed trajectories from storage, groups samples, and serves them to Slime. |
| `rl/slime_generator.py` | Slime rollout function. It starts the LLM proxy, fetches trajectory groups, builds masks/rewards, and returns training samples. |

Slime, Megatron-LM, and SGLang must be installed separately in the runtime used by `PYTHON_BIN` and `RAY_BIN`.

## Configure `env.sh`

Each example should keep its RL configuration in `rl/examples/<env>/env.sh`. For example:

```bash
export AIEVOBOX_ROOT=/path/to/SAfactory
export AIEVOBOX_MODE=docker
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.yaml
export AIEVOBOX_DB_URL=sqlite:///${AIEVOBOX_ROOT}/rl/examples/geo3k_vl/geo3k_vl.db

export BUFFER_SERVER_HOST=127.0.0.1
export BUFFER_SERVER_PORT=18889
export LLM_PROXY_HOST=127.0.0.1
export LLM_PROXY_PORT=18890

export SLIME_HOME=/path/to/slime
export MEGATRON_HOME=/path/to/Megatron-LM
export HF_CKPT_DIR=/path/to/hf/checkpoint
export LOAD_DIR=${HF_CKPT_DIR}
export SAVE_DIR=/path/to/save/checkpoints
```

Configure rollout and training parallelism in the same file:

```bash
export AIEVOBOX_POOL_SIZE=16
export AIEVOBOX_MAX_STEPS=10

export RL_GROUP_SIZE=8
export RL_GLOBAL_BATCH_SIZE=512
export RL_ROLLOUT_GROUP_BATCH_SIZE=64
export NUM_ROLLOUT=300

export NUM_GPUS=4
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE=1
export ROLLOUT_NUM_GPUS=3
export ROLLOUT_NUM_GPUS_PER_ENGINE=1
export TP_SIZE=1
```

Use `AIEVOBOX_MODE=docker`, `rjob`, or `sandbox` according to the runtime backend supported by the launcher.

## Start Training

Open two terminals from the Safactory repository root.

Terminal 1 starts Slime training and the rollout generator:

```bash
source rl/examples/geo3k_vl/env.sh
bash rl/run_slime_generator.sh
```

Terminal 2 starts Buffer Server and rollout collection:

```bash
source rl/examples/geo3k_vl/env.sh
bash rl/run_buffer_server.sh
```

You can also avoid sourcing into the current shell by passing the config file:

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_slime_generator.sh
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_buffer_server.sh
```

or:

```bash
bash rl/run_slime_generator.sh --env rl/examples/geo3k_vl/env.sh
bash rl/run_buffer_server.sh --env rl/examples/geo3k_vl/env.sh
```

To switch environments, pass a different `env.sh`; the entry scripts do not need to change.

## Key Variables

| Variable | Description |
| --- | --- |
| `AIEVOBOX_ROOT` | Safactory repository path. |
| `AIEVOBOX_MODE` | Launcher backend: `docker`, `rjob`, or `sandbox`. |
| `AIEVOBOX_AGENT_CONFIG` | Single agent YAML file. |
| `AIEVOBOX_AGENT_START_CONFIG` | Runtime startup YAML for Docker/RJob/Sandbox. |
| `AIEVOBOX_DB_URL` | Rollout trajectory storage URL. SQLite is recommended for local runs. |
| `AIEVOBOX_POOL_SIZE` | Number of concurrent rollout environment instances. |
| `AIEVOBOX_MAX_STEPS` | Maximum environment steps per episode. |
| `RL_GROUP_SIZE` | Samples per prompt, mapped to Slime `n_samples_per_prompt`. |
| `RL_ROLLOUT_GROUP_BATCH_SIZE` | Number of completed groups fetched per rollout batch. |
| `RL_GLOBAL_BATCH_SIZE` | Slime training global batch size. |
| `RL_OFF_BY_N` | Maximum allowed policy-version lag. |
| `DAPO_filter` | Whether to drop groups whose rewards are all identical. |
| `BUFFER_SERVER_HOST` / `BUFFER_SERVER_PORT` | Address used by the generator to connect to Buffer Server. |
| `LLM_PROXY_HOST` / `LLM_PROXY_PORT` | Address used by rollout workers to reach the generator's built-in LLM proxy. |
| `SLIME_HOME` | Slime repository path. |
| `MEGATRON_HOME` | Megatron-LM repository path. |
| `HF_CKPT_DIR` | HuggingFace checkpoint used to initialize training and rollout engines. |
| `LOAD_DIR` | Checkpoint path passed to Slime `--load`. Usually the HF checkpoint for first run. |
| `SAVE_DIR` | Directory for Megatron checkpoints. |
| `NUM_GPUS` | Number of GPUs registered to the local Ray head. |
| `ACTOR_NUM_GPUS_PER_NODE` | GPUs allocated to the training actor. |
| `ROLLOUT_NUM_GPUS` | Total GPUs allocated to rollout engines. |
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | GPUs per SGLang rollout engine. |
| `SGLANG_MEM_FRACTION_STATIC` | Static memory fraction reserved by SGLang. Lower it if online weight update OOMs. |

## Example Layout

Example directories should keep environment-specific files only. For `geo3k_vl`, the RL entrypoints are:

```bash
rl/run_buffer_server.sh
rl/run_slime_generator.sh
```

and the example-specific configuration is:

```bash
rl/examples/geo3k_vl/env.sh
```
