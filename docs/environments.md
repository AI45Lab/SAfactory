# Supported Environments

Safactory v2 treats each environment as an external agent runtime. A runtime is described by two files:

- Agent config: task rows, dataset, `env_params`, and image.
- Agent start config: Docker or RJob startup details for the runtime.

The current checkout includes these v2 adapters.

## Overview

| Adapter | `env_name` / `agent_name` | Config | Start config | Runtime notes |
|---------|----------------------------|--------|--------------|---------------|
| OpenClaw | `openclaw` | `env/openclaw/openclaw_config.yaml` | `env/openclaw/openclaw_start.yaml` | Generic OpenClaw CLI runtime. Good for smoke tests and tool-use tasks. |
| OpenRT | `openrt` | `env/openrt/openrt_config.yaml` | `env/openrt/openrt_start.yaml` | Runs OpenRT `eval.py` against gateway session URLs. |
| OpenRT RJob | `openrt` | `env/openrt/openrt_config.rjob.yaml` | `env/openrt/openrt_start.rjob.yaml` | Remote RJob variant with image, resources, embedded runner, and GPFS mounts. |
| ExploitGym RJob | `exploitgym` | `env/exploitgym/exploitgym_config.rjob.yaml` | `env/exploitgym/exploitgym_start.rjob.yaml` | Privileged nested-Docker benchmark runtime for one user, V8, or kernel task per episode. |
| WildClawBench | `wildclawbench` | `env/wildclawbench/wildclawbench_config.yaml` | `env/wildclawbench/wildclawbench_start.yaml` | Requires a WildClawBench checkout and matching image. |
| DTAP | `dtap` | `env/dtap/dtap_config.yaml` | `env/dtap/dtap_start.yaml` | Runs DecodingTrust-Agent workloads and mounts Docker socket. |
| ClawEnvKit | `clawenvkit` | `env/clawenvkit/clawenvkit_config.yaml` | `env/clawenvkit/clawenvkit_start.yaml` | Runs ClawEnvKit / Auto-ClawEval tasks. |

The checked-in YAML files contain local paths and internal image names in places. Treat them as working examples and adjust paths, images, mounts, and model route keys for your machine or cluster.

## Run One Adapter

Start gateway first, then run:

```bash
python launcher.py \
  --agent-config env/openclaw/openclaw_config.yaml \
  --agent-start-config env/openclaw/openclaw_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --db-path sqlite://env_trajs.db \
  --pool-size 1 \
  --max-workers 1
```

Use `--agent-root env` to load all agent configs under `env/`. During discovery, YAML files that are not agent configs, such as `*_start.yaml`, are skipped with warnings.

## OpenClaw

Files:

- `env/openclaw/openclaw_config.yaml`
- `env/openclaw/openclaw_start.yaml`
- `env/openclaw/runner.mjs`

The runner receives a `SimulationStartRequest`, writes an OpenClaw config pointing at the gateway session URL, and runs:

```bash
openclaw agent --local --json --session-id <session_id> --message <task> --model safactory/<route>
```

Common settings:

| Field | Meaning |
|-------|---------|
| `env_params.task_family` | Included in the task prompt. |
| `env_params.dataset` | Dataset row merged by the YAML loader. |
| `container.workdir` | Workspace mounted into the OpenClaw image. |
| `container.extra_args` | Should include `--add-host=host.docker.internal:host-gateway` for local gateway access from Docker. |

OpenClaw is the easiest adapter for a smoke test because its runner is self-contained.

## OpenRT

Files:

- `env/openrt/openrt_config.yaml`
- `env/openrt/openrt_start.yaml`
- `env/openrt/runner.py`
- `env/openrt/rule_evaluator.py`

The runner executes OpenRT `eval.py` inside the container and uses the gateway session URL as an OpenAI-compatible base URL. It expects each dataset row to define one attack, for example:

```json
{"task_id": "case-001", "attack": "PAIR"}
```

Important `env_params`:

| Field | Meaning |
|-------|---------|
| `default_openrt_dataset` | Dataset name passed to OpenRT, defaulting to `harmbench`. |
| `default_attacker_model` | Attacker model name used by OpenRT. |
| `default_judge_model` | Judge model name used by OpenRT. |
| `default_target_models` | Target model list. These model names should be routable through gateway. |
| `results_root` | Optional output root. Defaults to `/app/results`. |
| `max_workers`, `evaluator_workers` | Passed to OpenRT CLI when set. |

OpenRT also has an RJob version:

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

## ExploitGym RJob

Files:

- `env/exploitgym/exploitgym_config.rjob.yaml`
- `env/exploitgym/exploitgym_start.rjob.yaml`
- `env/exploitgym/runner.py`
- `env/exploitgym/rule_evaluator.py`

ExploitGym is RJob-only in this integration. Each dataset row must contain one
`task_id` starting with `user:`, `v8:`, or `kernel:`. The runner starts the
nested-Docker ExploitGym image for only that task and routes its model requests
through the current Safactory Gateway session.

The default dataset contains one ARVO task for a safe first run. The adjacent
`datasets/exploitgym_tasks_full.jsonl` contains all 869 canonical v1 tasks
(502 user, 181 V8, and 186 kernel tasks); select it explicitly for full runs.

The checked-in RJob resources are 8 CPU, 16 GiB RAM, no GPU, privileged mode,
one `brainpp.cn/fuse` resource, and 100 GiB local storage. GPFS2 provides the
read-only target-image cache; GPFS1 preserves results. Kernel tasks
additionally require read/write `/dev/kvm`.

The checked-in config pins the shared image by immutable digest. Replace the
result-mount placeholder before running and use a Gateway hostname reachable
from RJob workers. Actual provider keys remain in the Gateway; they are not
mounted into the ExploitGym RJob. See `env/exploitgym/README.md` for setup,
run commands, and result interpretation.

## WildClawBench

Files:

- `env/wildclawbench/wildclawbench_config.yaml`
- `env/wildclawbench/wildclawbench_start.yaml`
- `env/wildclawbench/runner.py`

This adapter expects a WildClawBench checkout and a runtime image that can run its tasks. Update these fields before running:

| Field | Update |
|-------|--------|
| `env_params.wildclawbench_root` | Absolute path to the WildClawBench checkout. |
| `container.workdir` | Same root path inside the container. |
| `container.mounts` | Mount the WildClawBench checkout and runner file. |
| `env.DEFAULT_MODEL`, `env_params.model_ref` | Model reference used by WildClawBench/OpenClaw. |
| `env.JUDGE_MODEL`, `env_params.judge_model` | Judge route key. |

## DTAP

Files:

- `env/dtap/dtap_config.yaml`
- `env/dtap/dtap_start.yaml`
- `env/dtap/runner.py`

DTAP runs DecodingTrust-Agent tasks. The start config mounts:

- A `DecodingTrust-Agent` checkout.
- Safactory `results`.
- The DTAP runner script.
- `/var/run/docker.sock`, because DTAP may start nested Docker workloads.

Review these fields before running:

| Field | Meaning |
|-------|---------|
| `env_params.dtap_root` | Runtime path to DecodingTrust-Agent. |
| `env_params.dataset_root` | DTAP dataset path. |
| `env_params.route_model` / `model_ref` | Route/model reference used by the task runtime. |
| `container.network` | Defaults to `host` in the example. |
| `container.mounts` | Must point to real local paths. |

## ClawEnvKit

Files:

- `env/clawenvkit/clawenvkit_config.yaml`
- `env/clawenvkit/clawenvkit_start.yaml`
- `env/clawenvkit/runner.py`

ClawEnvKit runs Auto-ClawEval-style tasks with an OpenClaw harness. Update dataset and result mounts before running.

Important fields:

| Field | Meaning |
|-------|---------|
| `env_params.dataset_root` | Dataset path inside the runtime. |
| `env_params.clawenvkit_root` | ClawEnvKit install path. |
| `env_params.harness_entrypoint` | Harness entrypoint script. |
| `env_params.route_model` / `model_ref` | Gateway route/model reference. |
| `container.extra_args` | The example resets entrypoint, runs as user `0`, and adds host-gateway. |

## Dataset Loading

Agent config `dataset` supports JSON, JSONL, YAML list, and parquet.

- JSONL lines must be valid JSON objects.
- Relative paths resolve from the agent config directory.
- `dataset_load_mode: eager` materializes rows.
- `dataset_load_mode: parquet_row_ref` stores lightweight parquet row references.

Each dataset row becomes `env_params.dataset` in the runtime request.

## Runtime Result Contract

All adapters should output one JSON result on stdout:

```json
{
  "session_id": "same-session-id",
  "status": "succeeded",
  "total_reward": 0.0,
  "step_count": 1,
  "terminated": true,
  "truncated": false,
  "error_text": null,
  "metrics": {}
}
```

If the runtime exits non-zero in JSON mode, Docker/RJob runners treat it as failure even if stdout contains partial output.
