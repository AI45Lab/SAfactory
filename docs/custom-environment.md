# Custom Runtime

This page explains how to connect a new agent or benchmark as a Safactory custom environment. Safactory v2 treats both as external runtimes: a runtime can be Python, Node.js, shell, or a wrapper around the benchmark's native harness. It receives a `SimulationStartRequest`, runs exactly one task/case, calls the gateway for model requests, and prints one result JSON object.

The key scheduling model is: one dataset row is one task-level scheduling unit. The launcher creates a separate `job_environments` row, `session_id`, and gateway session for each task, then runs the same image + runner for that single task. This keeps model calls, gateway telemetry, runtime results, and evaluation rewards bound to one session, preventing trajectories from different cases from contaminating each other. When integrating a benchmark, do not make the runner loop over the entire benchmark dataset inside one episode; make each dataset row represent one benchmark case and let Safactory schedule cases task by task.

You usually add or verify five core pieces:

| Piece | Where | Role | Example |
|-------|-------|------|---------|
| `image` | `env_image` in the agent config; RJob can override it in start config | Runtime image. It contains agent/benchmark dependencies, the benchmark harness, and Python/Node runtimes. | `myagent-image:latest`, `mybench-image:latest` |
| `runner_entrypoint` | Usually `env/<name>/runner.py` or `env/<name>/runner.mjs`, invoked by `start_config.container.runner_entrypoint.command` | Adapter between Safactory and the native agent/benchmark. It reads `SimulationStartRequest`, extracts `env_params.dataset`, calls the gateway target model, runs one task/case, and prints result JSON to stdout. | `python /tmp/safactory-mybench-runner.py` |
| `rule_evaluator` | Optional, commonly `env/<name>/rule_evaluator.py` | Converts raw benchmark results from `metrics` plus gateway trajectory into a Safactory reward on the 0 to 10 scale. Simple agent smoke tests can omit it; benchmarks usually should include it. | `env/mybench/rule_evaluator.py` |
| `config` | `env/<name>/<name>_config.yaml` | Defines task rows: `env_name`, `env_image`, `dataset`, `env_num`, `env_params`, and optional evaluation settings. | `env/mybench/mybench_config.yaml` |
| `start_config` | `env/<name>/<name>_start.yaml` | Defines how the matching runtime starts: runner entrypoint, workdir/env, Docker/RJob settings, and extra mounts. `agent_name` must match `env_name` in the config. | `env/mybench/mybench_start.yaml` |

Agents and benchmarks mostly differ in the runner and evaluator:

- An agent runtime usually turns `env_params.dataset` into a prompt, tool task, or interaction flow. Evaluation can use the final response, markdown eval tasks, or a custom rule.
- A benchmark runtime usually wraps an existing benchmark harness. The runner handles only the current dataset row, writes native benchmark score/pass/output metadata into `metrics`, and a `rule_evaluator.py` converts those results into reward.

## 1. Write A Runtime Script

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

The runtime should exit `0` after printing a failure result. If it exits non-zero in JSON mode, the Docker/RJob runner treats the whole episode as a runtime failure.

## 2. Understand The Request

The runtime receives `SimulationStartRequest` on stdin and through `SAFACTORY_START_REQUEST_JSON`.

Important fields:

| Field | Meaning |
|-------|---------|
| `job_id` | Launcher run ID. |
| `session_id` | Session/environment UUID. Use this in gateway URLs. |
| `agent_name`, `agent_id` | Adapter name and environment row ID. |
| `group_id` | RL grouping ID. |
| `gateway_base_url` | Gateway session root, for example `http://127.0.0.1:8000/v1/sessions`. |
| `model` | Gateway route key from `--llm-model`. |
| `temperature` | Sampling temperature. |
| `max_steps` | Step budget passed by launcher. |
| `storage_type`, `storage_config` | Storage details. |
| `env_params` | Expanded YAML parameters, including `dataset`. |
| `metadata` | Runtime metadata such as container ID, image, row ID, and worker ID. |

`request_env()` also injects useful environment variables:

| Variable | Meaning |
|----------|---------|
| `SAFACTORY_SESSION_ID` | Current session ID. |
| `SAFACTORY_GATEWAY_BASE_URL` | Gateway session root. |
| `SAFACTORY_GATEWAY_SESSION_URL` | Host-side session URL. |
| `SAFACTORY_GATEWAY_SESSION_URL_CONTAINER` | Container-friendly session URL. Localhost is rewritten to `host.docker.internal`. |
| `SAFACTORY_ROUTE_MODEL` | Route model inferred from dataset/env params/request. |
| `SAFACTORY_MODEL_REF` | Provider-style model ref, for example `safactory/<route>`. |
| `OPENROUTER_BASE_URL` | Alias for the container-friendly gateway session URL. |

## 3. Return The Result

Print exactly one result JSON object on stdout. Additional logs should go to stderr.

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
| `status` | yes | Usually `succeeded` or `failed`. |
| `total_reward` | yes | Rollout reward before optional evaluator override. |
| `step_count` | yes | Runtime-reported step count. |
| `terminated` | yes | Whether the episode ended naturally. |
| `truncated` | yes | Whether the episode hit a timeout or step limit. |
| `error_text` | no | Failure detail. |
| `metrics` | no | JSON object with adapter-specific details. |

For benchmarks, `metrics` is the most important interface between the runner and `rule_evaluator.py`. Prefer storing at least the case id, native score, pass/fail flag, reason, and detailed output path:

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

## 4. Add Agent Config

The agent config is the source of task scheduling. `env_name` binds to the start config, `env_image` selects the runtime image, `dataset` determines how many tasks are expanded, and `env_params` is passed to the runner.

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

Dataset rows are available as `request.env_params.dataset`.

Benchmark integration uses the same config structure, but each dataset row should represent one benchmark case rather than a full benchmark batch:

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

Those two rows are scheduled as two independent episodes, each with its own `session_id`, gateway trajectory, and reward. Do not make `runner.py` read and loop over `cases.jsonl` again; otherwise multiple cases' model calls land in the same trajectory and training/evaluation becomes hard to interpret.

## 5. Add Agent Start Config

The start config describes how to run the runner entrypoint after the image is allocated. `runner_entrypoint.command` is executed once per task/case and must read request JSON and print result JSON. When `runner_entrypoint.source` points to a local file, Docker mounts it and RJob embeds it automatically; the source path is relative to the start config file.

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

`agent_name: mybench` must match `env_name: mybench` in `mybench_config.yaml`; otherwise the launcher cannot find the corresponding startup definition.

## 6. Run A Smoke Test

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

Check logs:

- `logs/<run>/main.log` for launcher and scheduler events.
- `logs/gateway.log` for gateway events.
- `logs/gateway_requests.jsonl` for request/response logs.
- Adapter-specific output directories under `results/`.

## Optional Evaluation

You can add evaluation in one of three ways:

- Add inline `env_params.evaluation.specs` in the agent config.
- Add markdown tasks under `env/myagent/eval_tasks/<dataset>/<task_name>.md`.
- Add `env/myagent/rule_evaluator.py`.

For new benchmarks, prefer `rule_evaluator.py`: the runner preserves raw benchmark output, and the rule evaluator normalizes different benchmark score scales, pass conditions, and error cases into Safactory reward. It runs only during evaluation and should not re-run the benchmark case.

See [Evaluation](evaluation.md).

## BaseEnv Note

`core.env.BaseEnv` still exists for library-style environment implementations, but the current v2 launcher path schedules external runtimes through agent configs and start configs. Prefer the runtime approach above unless you are extending older in-process integrations.
