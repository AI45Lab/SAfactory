# Custom Runtime

In Safactory v2, the normal way to add an environment is to add an external agent runtime. The runtime can be a Python, Node.js, shell, or benchmark-specific program that receives a `SimulationStartRequest`, performs one episode, calls the gateway for model requests, and prints a result JSON object.

You usually add:

1. A runtime script.
2. An agent config YAML.
3. An agent start config YAML.
4. Optional eval tasks or rule evaluator.

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

## 4. Add Agent Config

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

## 5. Add Agent Start Config

Create `env/myagent/myagent_start.yaml`:

```yaml
agent_name: myagent

container:
  workdir: /workspace
  mounts:
    - source: ./env/myagent/runner.py
      target: /tmp/safactory-myagent-runner.py
      mode: ro
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
  run_command: "python /tmp/safactory-myagent-runner.py"
```

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

See [Evaluation](evaluation.md).

## BaseEnv Note

`core.env.BaseEnv` still exists for library-style environment implementations, but the current v2 launcher path schedules external runtimes through agent configs and start configs. Prefer the runtime approach above unless you are extending older in-process integrations.
