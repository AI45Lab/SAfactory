# Custom Environments

This page explains how to connect a new agent or benchmark to Safactory as a custom environment. In Safactory v2, a custom environment is an external runtime adapter: a Python script, Node.js script, shell command, or wrapper around an existing benchmark harness.

Each runtime adapter receives one `SimulationStartRequest`, runs one task or benchmark case, sends model requests through the Safactory gateway, and returns one `SimulationStartResult` JSON object. The terms `agent config` and `agent start config` are historical Safactory names. They are used for both agents and benchmarks.

The most important scheduling rule is:

> One dataset row is one scheduled episode.

For every dataset row, the launcher creates a separate `job_environments` row, `session_id`, and gateway session. It then starts the same image and runner for that single row. This keeps model calls, gateway telemetry, runtime output, and evaluation rewards tied to one session. When integrating a benchmark, do not make the runner loop over the full benchmark dataset inside one episode. Put each benchmark case in its own dataset row and let Safactory schedule the rows independently.

You usually need five pieces:

| Piece | Where | Role | Example |
|-------|-------|------|---------|
| Runtime image | `env_image` in the agent config. RJob deployments can override it from the start config. | Contains the agent or benchmark dependencies, the harness, and the language runtimes needed by the runner. | `myagent-image:latest`, `mybench-image:latest` |
| Runner entrypoint | Usually `env/<name>/runner.py` or `env/<name>/runner.mjs`, invoked by `container.runner_entrypoint.command`. | Adapts Safactory to the native agent or benchmark. It reads the request, extracts `env_params.dataset`, calls the target model through the gateway, runs one task or case, and returns the result JSON. | `python /tmp/safactory-mybench-runner.py` |
| Task config | `env/<name>/<name>_config.yaml`, passed with `--agent-config`. | Defines task rows: `env_name`, `env_image`, `dataset`, `env_num`, `env_params`, and optional evaluation settings. | `env/mybench/mybench_config.yaml` |
| Start config | `env/<name>/<name>_start.yaml`, passed with `--agent-start-config`. | Defines how the matching runtime starts: runner entrypoint, working directory, environment variables, Docker or RJob settings, and mounts. `agent_name` must match `env_name`. | `env/mybench/mybench_start.yaml` |
| Rule evaluator | Optional, commonly `env/<name>/rule_evaluator.py`. | Converts raw runtime metrics and the gateway trajectory into a Safactory score on the 0 to 10 scale. Simple smoke tests can omit it. Benchmarks usually should provide it. | `env/mybench/rule_evaluator.py` |

Agents and benchmarks mostly differ in the runner and evaluator:

- An agent runtime usually turns `env_params.dataset` into a prompt, tool task, or interaction flow. Evaluation uses a custom rule evaluator.
- A benchmark runtime usually wraps an existing benchmark harness. The runner handles only the current dataset row, writes native score, pass/fail status, reason, and output paths into `metrics`, and lets `rule_evaluator.py` normalize those details into a Safactory reward.

## 1. Write A Runner

Create `env/myagent/runner.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests


def read_request() -> dict[str, Any]:
    raw = sys.stdin.read().strip() or os.environ.get("SAFACTORY_START_REQUEST_JSON", "")
    if not raw:
        raise RuntimeError("missing SimulationStartRequest JSON")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError("SimulationStartRequest must be a JSON object")
    return data


def main() -> int:
    request = read_request()
    session_id = str(request["session_id"])
    base_url = os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER")
    if not base_url:
        base_url = f"{request['gateway_base_url'].rstrip('/')}/{session_id}"

    task = (request.get("env_params") or {}).get("dataset") or {}
    prompt = task.get("prompt") or task.get("question") or "Say hello from Safactory."

    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": request["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": request.get("temperature", 0.3),
        },
        timeout=300,
    )
    response.raise_for_status()
    body = response.json()
    answer = body["choices"][0]["message"].get("content", "")

    print(json.dumps({
        "session_id": session_id,
        "status": "succeeded",
        "total_reward": 0.0,
        "step_count": 1,
        "terminated": True,
        "truncated": False,
        "error_text": None,
        "metrics": {"answer": answer},
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({
            "session_id": os.environ.get("SAFACTORY_SESSION_ID", ""),
            "status": "failed",
            "total_reward": 0.0,
            "step_count": 0,
            "terminated": True,
            "truncated": False,
            "error_text": str(exc),
            "metrics": {},
        }, ensure_ascii=False), flush=True)
        raise SystemExit(0)
```

The runner should print a failed result and exit `0` when the task fails in a controlled way. In JSON result mode, a non-zero process exit means the runtime command itself failed. Docker and RJob treat that as an infrastructure/runtime failure, even if some partial output was printed.

For predictable parsing, keep stdout reserved for the result JSON. Send diagnostic logs to stderr. For long remote runs, the runner may also write the same result object to the path in `SAFACTORY_RESULT_PATH`; Safactory parses stdout first and falls back to that artifact path when stdout does not contain parseable JSON.

## 2. Read The Request

Safactory passes `SimulationStartRequest` both on stdin and in `SAFACTORY_START_REQUEST_JSON`.

Important fields:

| Field | Meaning |
|-------|---------|
| `job_id` | Launcher run ID. |
| `session_id` | Per-episode session UUID. Use it when building gateway URLs and result paths. |
| `agent_name`, `agent_id` | Runtime name and environment row ID. |
| `group_id` | RL grouping ID, if RL grouping is enabled. |
| `gateway_base_url` | Gateway session root, for example `http://127.0.0.1:8000/v1/sessions`. |
| `model` | Gateway route key from `--llm-model`. |
| `temperature` | Sampling temperature from the launcher. |
| `max_steps` | Step budget passed by the launcher. |
| `storage_type`, `storage_config` | Storage backend details. |
| `env_params` | Expanded YAML parameters. The current dataset row is available at `env_params.dataset`. |
| `metadata` | Runtime metadata such as container ID, image, row ID, and worker ID. |

`request_env()` also injects useful environment variables:

| Variable | Meaning |
|----------|---------|
| `SAFACTORY_START_REQUEST_JSON` | Full `SimulationStartRequest` JSON. |
| `SAFACTORY_JOB_ID` | Launcher run ID. |
| `SAFACTORY_SESSION_ID` | Current session ID. |
| `SAFACTORY_AGENT_NAME`, `SAFACTORY_AGENT_ID` | Runtime name and environment row ID. |
| `SAFACTORY_TASK_ID`, `SAFACTORY_TASK_PATH`, `SAFACTORY_CATEGORY` | Convenience values copied from the dataset row when present. |
| `SAFACTORY_RESULT_PATH` | Preferred artifact path for a result JSON file. |
| `SAFACTORY_GATEWAY_BASE_URL` | Gateway session root. |
| `SAFACTORY_GATEWAY_SESSION_URL` | Host-side session URL. |
| `SAFACTORY_GATEWAY_SESSION_URL_CONTAINER` | Container-friendly session URL. Local `localhost` addresses are rewritten to `host.docker.internal`. |
| `SAFACTORY_ROUTE_MODEL` | Route model inferred from dataset, `env_params`, or request. |
| `SAFACTORY_MODEL_REF` | Provider-style model reference, for example `safactory/<route>`. |
| `OPENROUTER_BASE_URL` | Alias for the container-friendly gateway session URL. |

## 3. Return The Result

Print one `SimulationStartResult` JSON object on stdout. Other logs should go to stderr.

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

Fields:

| Field | Required | Meaning |
|-------|----------|---------|
| `session_id` | yes | Must match the request session ID. |
| `status` | yes | Use `succeeded` when the runner completed normally, even if the task score is low. Use `failed` for runtime errors. |
| `total_reward` | yes | Runtime-reported reward before any optional evaluator override. |
| `step_count` | yes | Number of steps reported by the runtime. |
| `terminated` | yes | Whether the episode reached a normal stopping point. |
| `truncated` | yes | Whether the episode stopped because of a timeout or step limit. |
| `error_text` | no | Failure detail for runtime errors. |
| `metrics` | no | Adapter-specific JSON object. This is the best place to keep benchmark outputs and file paths. |

For benchmarks, `metrics` is the main interface between the runner and `rule_evaluator.py`. Store enough information for evaluation to score the case without rerunning it:

```json
{
  "metrics": {
    "bench_case_id": "case-001",
    "bench_score": 0.73,
    "bench_passed": true,
    "bench_reason": "all required checks passed",
    "bench_output_path": "/workspace/Safactory/results/mybench/case-001.json"
  }
}
```

## 4. Add The Task Config

The task config is the source of scheduled rows. `env_name` binds the rows to a start config, `env_image` selects the default runtime image, `dataset` controls how many rows are expanded, and `env_params` is passed to the runner.

Create `env/myagent/myagent_config.yaml`:

```yaml
environments:
  - env_name: myagent
    env_image: myagent-image:latest
    env_num: 1
    dataset: ./datasets/tasks.jsonl
    dataset_load_mode: eager
    env_params:
      task_family: myagent
      output_root: /workspace/Safactory/results/myagent
```

Create `env/myagent/datasets/tasks.jsonl`:

```jsonl
{"task_id": "hello-001", "prompt": "Write one short greeting."}
```

The dataset row appears in the request as `env_params.dataset`.

Benchmark integration uses the same config shape. The key difference is that each dataset row should represent one benchmark case, not a full benchmark batch:

```yaml
environments:
  - env_name: mybench
    env_image: mybench-image:latest
    env_num: 1
    dataset: ./datasets/cases.jsonl
    dataset_load_mode: eager
    env_params:
      task_family: mybench
      bench_root: /workspace/MyBench
      output_root: /workspace/Safactory/results/mybench
      evaluation:
        rule_evaluator: env/mybench/rule_evaluator.py
        rule_evaluator_timeout_s: 60
```

```jsonl
{"task_id": "case-001", "case_id": "case-001", "input": "example input", "expected": "example answer"}
{"task_id": "case-002", "case_id": "case-002", "input": "another input", "expected": "another answer"}
```

Those two rows become two independent episodes, each with its own `session_id`, gateway trajectory, result, and reward. Do not have `runner.py` read `cases.jsonl` and loop over it again. If multiple cases run inside one episode, their model calls land in the same trajectory and the resulting training or evaluation data becomes ambiguous.

## 5. Add The Start Config

The start config describes how to execute the runner after Safactory allocates the image. `container.runner_entrypoint.command` runs once per dataset row. It must read the request JSON and return the result JSON.

When `container.runner_entrypoint.source` points to a local file, the path is resolved relative to the start config file. Docker adds it as a mount at `target`; RJob embeds or stages the file through the RJob runtime config. The `command` should execute the file at the target path.

Create `env/myagent/myagent_start.yaml`:

```yaml
agent_name: myagent

container:
  workdir: /workspace
  runner_entrypoint:
    source: ./runner.py
    target: /tmp/safactory-myagent-runner.py
    command: "python /tmp/safactory-myagent-runner.py"
  mounts:
    - source: ./results
      target: /workspace/Safactory/results
      mode: rw
  env:
    PYTHONDONTWRITEBYTECODE: "1"
    NO_PROXY: host.docker.internal,localhost,127.0.0.1,::1
    no_proxy: host.docker.internal,localhost,127.0.0.1,::1
  extra_args:
    - --add-host=host.docker.internal:host-gateway
  idle_command: "tail -f /dev/null"
```

A benchmark start config uses the same shape, with the benchmark image and runner:

```yaml
agent_name: mybench

container:
  workdir: /workspace/MyBench
  runner_entrypoint:
    source: ./runner.py
    target: /tmp/safactory-mybench-runner.py
    command: "python /tmp/safactory-mybench-runner.py"
  mounts:
    - source: ./results
      target: /workspace/Safactory/results
      mode: rw
  env:
    PYTHONDONTWRITEBYTECODE: "1"
    NO_PROXY: host.docker.internal,localhost,127.0.0.1,::1
    no_proxy: host.docker.internal,localhost,127.0.0.1,::1
  extra_args:
    - --add-host=host.docker.internal:host-gateway
  idle_command: "tail -f /dev/null"
```

`agent_name: mybench` must match `env_name: mybench` in `mybench_config.yaml`; otherwise the launcher cannot find the startup definition for the scheduled rows.

## 6. Run A Smoke Test

Start the gateway first, then run one worker and one task at a time:

```bash
python launcher.py \
  --agent-config env/myagent/myagent_config.yaml \
  --agent-start-config env/myagent/myagent_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --db-path sqlite://env_trajs.db \
  --pool-size 1 \
  --max-workers 1
```

Check:

- `logs/<run>/main.log` for launcher and scheduler events.
- `logs/<run>/gateway.log` for gateway events.
- `logs/<run>/gateway_requests.jsonl` for request and response records.
- Adapter-specific output directories under `results/`.

## Optional Evaluation

You can add rule evaluation in one of two ways:

- Add inline `env_params.evaluation.specs` in the task config.
- Add `env/myagent/rule_evaluator.py`.

The runner should preserve raw benchmark output in `metrics` or output files, and the rule evaluator should normalize benchmark-specific score scales, pass conditions, and error cases into Safactory's 0 to 10 score. It runs only during evaluation and should not rerun the benchmark case.

See [Evaluation](evaluation.md).

## BaseEnv Note

`core.env.BaseEnv` still exists for older library-style, in-process environment implementations. The current v2 launcher path schedules external runtimes through agent configs and start configs. Prefer the runtime adapter approach above unless you are extending an older in-process integration.
