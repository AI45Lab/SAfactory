# RL Training

This document is the single entry point for Safactory RL training. The shared
entry scripts live under `rl/`; each environment keeps only its own `env.sh`
configuration. To switch environments or runtime backends, change the
environment variables in `env.sh` rather than copying the launch scripts.

## Components

| Component | Role |
| --- | --- |
| `rl/examples/<env>/env.sh` | Environment, runtime, service, and Slime training configuration. |
| `rl/run_buffer_server.sh` | Starts Buffer Server, auto-starts Gateway when enabled, and launches Safactory rollout collection. |
| `rl/run_slime_generator.sh` | Starts Ray, Slime training, SGLang rollout engines, and the Safactory rollout function. |
| `rl/buffer_server.py` | Starts `launcher.py`, reads trainable rows from storage, groups samples by `group_id`, and serves them to Slime through `/get_rollout_data`. |
| `rl/slime_generator.py` | Slime rollout function. It starts the LLM proxy, fetches trajectory groups, builds masks and rewards, and returns training samples. |
| `rl/llm_proxy.py` | OpenAI-compatible endpoint hosted by the Slime generator and used by Gateway during online rollout. |

Slime, Megatron-LM, SGLang, Ray, and the model runtime dependencies must be
available in the environment selected by `PYTHON_BIN` and `RAY_BIN`.

## Architecture

```text
Safactory runtime  <--- started by launcher.py / rl/buffer_server.py
  |
  | session-scoped rollout model requests
  v
Gateway  <--- auto-started by Buffer Server unless disabled
  |
  | route key from RL_MODEL
  v
LLM proxy  <--- hosted by Slime generator
  |
  | generation requests
  v
SGLang rollout engine

Gateway / evaluator
  |
  | rewarded trainable rows in session_steps
  v
Buffer Server /get_rollout_data
  |
  | grouped rollout samples
  v
Slime trainer
```

For v2 adapter runs, the launcher still needs a valid Gateway-compatible model route and runtime start config. Verify the generated launcher command in `logs/buffer_server.log` before scaling.

## Minimal Geo3K Training Path

First make Geo3K evaluation work outside RL with `launcher.py`, Gateway, and `--enable-evaluation`. That validates the Docker image, dataset, route key, storage, and `rule_evaluator.py`.

Then edit or override `rl/examples/geo3k_vl/env.sh`:

```bash
export AIEVOBOX_ROOT=$(pwd)
export AIEVOBOX_MODE=docker
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.yaml
export AIEVOBOX_GATEWAY_HOST=127.0.0.1
export AIEVOBOX_GATEWAY_PORT=8000
export AIEVOBOX_GATEWAY_BASE_URL=http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions
export RL_MODEL=geo3k_model

export AIEVOBOX_ENABLE_EVALUATION=1
export AIEVOBOX_POOL_SIZE=2
export AIEVOBOX_MAX_STEPS=10
export RL_GROUP_SIZE=2
export RL_EPOCH=1

export HF_CKPT_DIR=/path/to/hf-checkpoint
export LOAD_DIR=${HF_CKPT_DIR}
export SAVE_DIR=/path/to/save/checkpoints
export SLIME_HOME=/path/to/slime
export MEGATRON_HOME=/path/to/Megatron-LM
```

Use the checked-in sample dataset for smoke tests if the default Geo3K config points to a full local parquet dataset.

Start the two services from the repository root:

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_slime_generator.sh
```

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_buffer_server.sh
```

Buffer Server autostarts Gateway by default, generates `logs/gateway.rl.generated.yaml`, routes `RL_MODEL` to the Slime-hosted LLM proxy, starts Docker rollout collection, and serves completed groups through `/get_rollout_data`. Stop any manually started Gateway on the same port before using autostart.

- `AIEVOBOX_DB_URL` as its storage DB, matching `launcher.py --db-path`.
- `RL_MODEL` as the Gateway route key.
- `http://${LLM_PROXY_HOST}:${LLM_PROXY_PORT}/v1` as the route upstream.
- `AIEVOBOX_GATEWAY_PORT` as the listen port.

Set `AIEVOBOX_GATEWAY_AUTOSTART=0` only when you have started an external
Gateway yourself with the same storage backend and route key.

## Start Geo3K RL Training

This means the checked-in RL examples are configuration templates around the shared v2 launcher. Geo3K is the maintained standard template.

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

You can also pass the config file without sourcing it into the current shell:

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_slime_generator.sh
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_buffer_server.sh
```

or:

```bash
bash rl/run_slime_generator.sh --env rl/examples/geo3k_vl/env.sh
bash rl/run_buffer_server.sh --env rl/examples/geo3k_vl/env.sh
```

Training begins when the trainer or a client calls Buffer Server
`POST /start_rollout`. The Slime generator then fetches completed grouped
samples through `POST /get_rollout_data`.

## Common Geo3K `env.sh` Settings

`rl/examples/geo3k_vl/env.sh` is the only file users usually edit for Geo3K RL.
Keep the training, service, and runtime choices there.

```bash
export AIEVOBOX_ROOT=/path/to/SAfactory
export STORAGE_TYPE=sqlite
export AIEVOBOX_DB_URL=sqlite:///${AIEVOBOX_ROOT}/rl/examples/geo3k_vl/geo3k_vl.db

export BUFFER_SERVER_HOST=127.0.0.1
export BUFFER_SERVER_PORT=18889
export LLM_PROXY_HOST=127.0.0.1
export LLM_PROXY_PORT=18890

# Use 127.0.0.1 for local Docker, or a cluster-visible address for RJob.
export AIEVOBOX_GATEWAY_HOST=127.0.0.1
export AIEVOBOX_GATEWAY_PORT=8000
export AIEVOBOX_GATEWAY_BASE_URL=http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions

export RL_MODEL=model
export RL_GROUP_SIZE=8
export RL_GLOBAL_BATCH_SIZE=512
export RL_ROLLOUT_GROUP_BATCH_SIZE=64
export RL_EPOCH=1000
export RL_OFF_BY_N=0
export DAPO_filter=true

export AIEVOBOX_POOL_SIZE=16
export AIEVOBOX_MAX_STEPS=10
export AIEVOBOX_ENABLE_EVALUATION=1

export SLIME_HOME=/path/to/slime
export MEGATRON_HOME=/path/to/Megatron-LM
export HF_CKPT_DIR=/path/to/hf/checkpoint
export LOAD_DIR=${HF_CKPT_DIR}
export SAVE_DIR=/path/to/save/checkpoints
```

Geo3K needs evaluation enabled because `env/geo3k/rule_evaluator.py` converts
the runner metric `score` to Safactory's reward scale. If evaluation is
disabled, rollouts can still run, but rewards may not be committed as trainable
RL samples.

## Geo3K Docker Mode

Use Docker mode when the launcher machine can run local Docker containers.

In `rl/examples/geo3k_vl/env.sh`:

```bash
export AIEVOBOX_MODE=docker
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.yaml

export AIEVOBOX_GATEWAY_HOST=127.0.0.1
export AIEVOBOX_GATEWAY_PORT=8000
export AIEVOBOX_GATEWAY_BASE_URL=http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions
```

Docker-specific files:

| File | Purpose |
| --- | --- |
| `env/geo3k/geo3k_config.yaml` | Uses the local Docker image, Geo3K parquet path, selected dataset columns, and `env_params` such as `max_turns` and `max_images`. |
| `env/geo3k/geo3k_start.yaml` | Defines the local Docker runner mount, result mount, environment variables, and `host.docker.internal` networking. |

Docker requirements and notes:

- The launcher environment must have the `docker` CLI and permission to start containers.
- `env_image` in `geo3k_config.yaml` must exist locally or be pullable according to your Docker policy.
- `container.runner_entrypoint.source: ./` mounts the whole `env/geo3k` directory into the container, so `runner.py` and `math_utils.py` do not need to be baked into the image.
- The image must still contain Python and runtime dependencies used by Geo3K, such as `requests`, `sympy`, and `pylatexenc`.
- `geo3k_start.yaml` mounts `./results` for artifacts. Relative Docker mount sources are resolved from the launcher working directory.
- For local Gateway access, the Docker adapter injects a container-friendly session URL and `geo3k_start.yaml` adds `host.docker.internal`.

## Geo3K RJob Mode

Use RJob mode when rollout environments should run on the remote RJob cluster.
The Slime trainer, SGLang rollout engines, Buffer Server, Gateway, and launcher
still run from the RL launcher environment unless you deploy them separately.

In `rl/examples/geo3k_vl/env.sh`:

```bash
export AIEVOBOX_MODE=rjob
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.rjob.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.rjob.yaml

# Must be reachable from RJob containers; do not use 127.0.0.1 or localhost.
export AIEVOBOX_GATEWAY_HOST=<launcher-or-gateway-ip-visible-to-rjob>
export AIEVOBOX_GATEWAY_PORT=8000
export AIEVOBOX_GATEWAY_BASE_URL=http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions
```

RJob-specific files:

| File | Purpose |
| --- | --- |
| `env/geo3k/geo3k_config.rjob.yaml` | Uses a registry image that the RJob cluster can pull and a dataset path visible to the launcher when it materializes parquet rows. |
| `env/geo3k/geo3k_start.rjob.yaml` | Defines RJob resource requests, embedded runner files, cleanup behavior, result artifact mount, and proxy-related environment variables. |
| `config.yaml` | Default global RJob connection and auth config used by `launcher.py`. Current RL Buffer Server invokes `launcher.py` without `--rjob-config`, so this default file is the active global RJob config for RL runs. |

The launcher Python environment must be able to import RJob SDK classes:

```bash
python -c "from brainpp.rjob import RJobClient; print(RJobClient)"
```

Global RJob settings belong in `config.yaml`:

```yaml
rjob:
  cluster_entry: "https://your-rjob-platform.example"
  namespace: "your-namespace"
  access_key: "replace-me"
  secret_key: "replace-me"
  charged_group: "your-quota-or-project"
  gateway_base_url: "http://<gateway-host-visible-to-rjob>:8000/v1/sessions"
  submit_concurrency: 1
  cleanup_on_finish: true
  no_packaging: true
```

Important RJob configuration details:

- `access_key` and `secret_key` authenticate `RJobClient` when creating, polling, reading logs for, and deleting RJobs. They also determine which namespaces, charged groups, images, and mounts you are allowed to use. Keep them out of committed files when possible.
- `charged_group` selects the quota or billing group consumed by the RJob.
- `gateway_base_url` in the global or per-agent RJob config overrides the launcher request URL. If it is set, make sure it matches the reachable Gateway address. If you prefer to drive this from `env.sh`, omit stale `rjob.gateway_base_url` values.
- `mount_config` maps cluster-accessible storage into the RJob container. The left side must be storage the RJob cluster can mount, not a local Docker bind path. Geo3K uses this for result artifacts, for example `gpfs://gpfs1/evobox-share/chenxinquan/SAfactory/results:/app/results`.
- `container.runner_entrypoint.source: ./runner.py` is embedded or staged into the RJob runtime automatically at `target`. The Docker-style local bind mount is not used in RJob.
- Extra local files imported by the runner must be listed in `rjob.embedded_files`. Geo3K lists `math_utils.py`.
- The RJob image must contain the environment dependencies. Embedded runner files provide the adapter code, not the Python packages.

For RJob networking, the most common failure is using a local-only Gateway URL.
`AIEVOBOX_GATEWAY_BASE_URL=http://127.0.0.1:8000/v1/sessions` works only for
processes on the launcher machine. RJob containers need an address routed from
the cluster to the Gateway host.

## Key Variables

| Variable | Description |
| --- | --- |
| `AIEVOBOX_ROOT` | Safactory repository path. |
| `AIEVOBOX_MODE` | Runtime backend: `docker`, `rjob`, or `sandbox`. |
| `AIEVOBOX_AGENT_CONFIG` | Single Geo3K agent config YAML. |
| `AIEVOBOX_AGENT_START_CONFIG` | Docker or RJob startup YAML for Geo3K. |
| `STORAGE_TYPE` | `sqlite` or `cloud`; Geo3K RL examples commonly use `sqlite`. |
| `AIEVOBOX_DB_URL` | Rollout trajectory DB URL shared by Gateway, launcher, Buffer Server, and evaluator. |
| `AIEVOBOX_GATEWAY_HOST` / `AIEVOBOX_GATEWAY_PORT` | Host and port used by Buffer Server readiness checks and by `AIEVOBOX_GATEWAY_BASE_URL`. |
| `AIEVOBOX_GATEWAY_BASE_URL` | Gateway session root passed to launcher and runtime requests. Required. |
| `AIEVOBOX_GATEWAY_AUTOSTART` | Defaults to enabled. Set to `0` for a manually managed external Gateway. |
| `AIEVOBOX_ENABLE_EVALUATION` | Set to `1`, `true`, `yes`, or `on` to run the evaluator and commit trainable rewards. |
| `AIEVOBOX_POOL_SIZE` | Concurrent rollout environment instances. In RJob mode this controls the target number of concurrent RJobs. |
| `AIEVOBOX_MAX_STEPS` | Maximum environment steps per episode. |
| `RL_MODEL` | Gateway route key. It must match the route generated for the RL LLM proxy. |
| `RL_GROUP_SIZE` | Samples per prompt, mapped to Slime `n_samples_per_prompt` and launcher `--rl-group-size`. |
| `RL_ROLLOUT_GROUP_BATCH_SIZE` | Number of completed groups fetched per rollout batch. Defaults to `RL_GLOBAL_BATCH_SIZE / RL_GROUP_SIZE` when empty. |
| `RL_GLOBAL_BATCH_SIZE` | Slime training global batch size. |
| `RL_EPOCH` | Repeats the scheduled RL dataset rows for rollout epochs. |
| `RL_OFF_BY_N` | Maximum allowed policy-version lag consumed by training-side filtering. |
| `DAPO_filter` | Whether to drop groups whose rewards are all identical. |
| `BUFFER_SERVER_HOST` / `BUFFER_SERVER_PORT` | Address used by the generator to connect to Buffer Server. |
| `LLM_PROXY_HOST` / `LLM_PROXY_PORT` | Address used by Gateway to connect to the Slime generator's built-in LLM proxy. |
| `SLIME_HOME` | Slime repository path. |
| `MEGATRON_HOME` | Megatron-LM repository path. |
| `HF_CKPT_DIR` | HuggingFace checkpoint used to initialize training and rollout engines. |
| `LOAD_DIR` | Checkpoint path passed to Slime `--load`. Usually the HF checkpoint for the first run. |
| `SAVE_DIR` | Megatron checkpoint save directory. |
| `NUM_GPUS` | Number of GPUs registered to the local Ray head. |
| `ACTOR_NUM_GPUS_PER_NODE` | GPUs allocated to the training actor. |
| `ROLLOUT_NUM_GPUS` | Total GPUs allocated to SGLang rollout engines. |
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | GPUs per SGLang rollout engine. |
| `SGLANG_MEM_FRACTION_STATIC` | Static memory fraction reserved by SGLang. Lower it if online weight update OOMs. |

## Buffer Server API

| Endpoint | Purpose |
| --- | --- |
| `POST /start_rollout` | Starts one launcher subprocess unless one is already running. |
| `POST /get_rollout_data` | Returns grouped rollout samples since the last served DB row. |
| `GET /health` | Reports Buffer Server health, launcher process state, and DataManager initialization state. |

`/get_rollout_data` returns items shaped for Slime:

```json
{
  "uid": "...",
  "instance_id": "<group_id>",
  "messages": [],
  "reward": 0.0,
  "extra_info": {
    "session_id": "...",
    "env_id": "...",
    "group_id": "...",
    "weight_version": 0,
    "truncated": false
  }
}
```

## Operational Checks

Before scaling a run:

1. Run one Geo3K case outside RL with `launcher.py` and the same Docker or RJob configs.
2. Confirm Gateway `/readyz` is reachable from the launcher.
3. For RJob, confirm the Gateway URL is reachable from inside an RJob container.
4. Confirm `session_steps` contains trainable rows with the expected `job_id` and `group_id`.
5. Confirm `RL_GROUP_SIZE` completed rows are produced per group; otherwise the Buffer Server will hold incomplete groups.
6. Watch `logs/<run>/main.log`, `logs/<run>/slime.log`, `logs/gateway.log`, `logs/gateway_requests.jsonl`, and `logs/buffer_server.log`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `FileNotFoundError: docker` | Docker mode requires the Docker CLI in the launcher environment. Use Docker mode only where Docker is installed, or switch to RJob. |
| `RJob mode requires brainpp.rjob / RJobClient` | Install or activate the RJob SDK in the launcher Python environment. |
| RJob create returns `403 Forbidden` | Check RJob credentials, namespace, charged group, image permission, and mount permission. |
| RJob succeeds but no result JSON is parsed | Ensure the runner prints one `SimulationStartResult` JSON to stdout, and for artifact fallback make sure `SAFACTORY_RESULT_PATH` maps to a writable mounted path such as `/app/results`. |
| RJob runner cannot connect to Gateway | Do not use `127.0.0.1` or `localhost` for RJob. Use a Gateway host visible from the RJob cluster. |
| Gateway upstream returns `400` for Geo3K images | `RL_MODEL` must route to a multimodal-capable rollout model when `max_images > 0`. |
| Evaluator logs `total_rows=0 trainable_rows=0` | The runtime failed before model calls were recorded, or Gateway and launcher are not using the same DB. Check worker errors before the evaluator logs. |
| Groups never become ready | `RL_GROUP_SIZE` is larger than the number of completed samples for the same `group_id`. |
| Online weight update OOMs | Lower `SGLANG_MEM_FRACTION_STATIC`, reduce rollout engine concurrency, or reduce model parallel pressure. |
