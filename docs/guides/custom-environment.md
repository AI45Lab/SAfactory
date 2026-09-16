# Custom Environments

This page explains how to connect a new agent or benchmark to Safactory as a custom environment. In Safactory v2, a custom environment is an external runtime adapter: a Python script, Node.js script, shell command, or wrapper around an existing benchmark harness.

Each runtime adapter receives one `SimulationStartRequest`, runs one task or benchmark case, sends model requests through the Safactory gateway, and returns one `SimulationStartResult` JSON object. The terms `agent config` and `agent start config` are historical Safactory names. They are used for both agents and benchmarks.

The most important scheduling rule is:

> One dataset row is one scheduled episode.

For every dataset row, the launcher creates a separate `job_environments` row, `session_id`, and gateway session. It then starts the same image and runner for that single row. This keeps model calls, gateway telemetry, runtime output, and evaluation rewards tied to one session. When integrating a benchmark, do not make the runner loop over the full benchmark dataset inside one episode. Put each benchmark case in its own dataset row and let Safactory schedule the rows independently.

You usually need a runtime image, runner, task config, and start config. A rule evaluator is optional and is needed only when evaluation is requested. The integration boundary is the SAfactory adapter's input/output handling: the benchmark harness or image should already know how to execute one case and, when evaluation is requested, produce its native score, and onboarding should not rewrite that logic.

Start with local adapter contract tests; they do not require Docker, a running Gateway, model credentials, or an internal RJob cluster. Run a live smoke test when the deployment prerequisites are available. The Geo3K live baseline can help diagnose shared infrastructure, but it is not an onboarding prerequisite. Use `env/geo3k` as a reference for environment-specific behavior.

| Piece | Where | Role | Example |
|-------|-------|------|---------|
| Runtime image | `env_image` in the task config. The RJob config variant normally names a cluster-pullable image. | Contains the agent or benchmark dependencies, the harness, and the language runtimes needed by the runner. | `myagent-image:latest`, `mybench-image:latest` |
| Runner entrypoint | Usually `env/<name>/runner.py` or `env/<name>/runner.mjs`, invoked by `container.runner_entrypoint.command`. | Adapts Safactory to the native agent or benchmark. It reads the request, extracts `env_params.dataset`, calls the target model through the gateway, runs one task or case, and returns the result JSON. | `python /tmp/safactory-mybench-runner.py` |
| Task config | `env/<name>/<name>_config.yaml`, passed with `--agent-config`. RJob mode also provides `<name>_config.rjob.yaml`. | Defines task rows: `env_name`, `env_image`, `dataset`, `env_num`, and `env_params`. Each dataset row is one case/episode. | `env/mybench/mybench_config.yaml`, `env/mybench/mybench_config.rjob.yaml` |
| Start config | `env/<name>/<name>_start.yaml`, passed with `--agent-start-config`. RJob mode also provides `<name>_start.rjob.yaml`. | Defines how the matching runtime starts: runner entrypoint, working directory, environment variables, Docker or RJob settings, and mounts. `agent_name` must match `env_name`. | `env/mybench/mybench_start.yaml`, `env/mybench/mybench_start.rjob.yaml` |
| Rule evaluator | Optional, commonly `env/<name>/rule_evaluator.py`. | Converts raw runtime metrics and the gateway trajectory into a Safactory score on the 0 to 10 scale. Integration-only runs can omit it even when the native benchmark produces scores. Enable evaluation explicitly with `--enable-evaluation`. | `env/mybench/rule_evaluator.py` |

Agents and benchmarks mostly differ in the runner and evaluator:

- An agent runtime usually turns `env_params.dataset` into a prompt, tool task, or interaction flow. Evaluation uses a custom rule evaluator.
- A benchmark runtime usually wraps an existing benchmark harness. The runner handles only the current dataset row, writes available native outputs and paths into `metrics`, and optionally preserves score/pass details for `rule_evaluator.py` when evaluation is requested.

## 1. Start From The Fixed Templates

Create the guide-derived files with:

```bash
python skills/safactory-workflows/scripts/scaffold_environment.py myagent --mode docker
# Use --mode rjob to mark RJob as the first deployment target.
# Both Docker and RJob config pairs are generated; add --enable-evaluation only
# when evaluation is requested.
```

The [templates](../../skills/safactory-workflows/assets/environment/) keep protocol handling in `runner.py` and environment logic in `adapter.py:run_case`. Fill that hook and the YAML values; keep the protocol shell unchanged. For benchmarks, replace the generated greeting with the existing native single-case command and output mapping. The scaffolder refuses to overwrite existing directories. Its sample request/dataset verifies the scaffold only; replace those examples with representative cases before claiming integration success.

The optional `rule_evaluator.py` template isolates scoring in `score_metrics`. It deliberately fails until the native score mapping is supplied. Scoring data is not required for integration-only work. See the [integration workflow](../../skills/safactory-workflows/references/environment-integration.md) for the complete file map and validation commands.

The following compact example illustrates the underlying runner protocol; use the templates for new integration files:

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
| `total_reward` | yes | Runtime-reported reward before any optional evaluator override. Use `0.0` for ungraded integration-only runs; this does not imply an evaluation result. |
| `step_count` | yes | Number of steps reported by the runtime. |
| `terminated` | yes | Whether the episode reached a normal stopping point. |
| `truncated` | yes | Whether the episode stopped because of a timeout or step limit. |
| `error_text` | no | Failure detail for runtime errors. |
| `metrics` | no | Adapter-specific JSON object. This is the best place to keep benchmark outputs and file paths. |

When evaluation is requested, `metrics` is the main interface between the runner and `rule_evaluator.py`. Store enough information to score the case without rerunning it; these scoring fields are optional for integration-only work:

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
```

```jsonl
{"task_id": "case-001", "case_id": "case-001", "input": "example input", "expected": "example answer"}
{"task_id": "case-002", "case_id": "case-002", "input": "another input", "expected": "another answer"}
```

Those two rows become two independent episodes, each with its own `session_id`, gateway trajectory, result, and reward. Do not have `runner.py` read `cases.jsonl` and loop over it again. If multiple cases run inside one episode, their model calls land in the same trajectory and the resulting training or evaluation data becomes ambiguous.

## 5. Add The Start Config

The start config describes how to execute the runner after Safactory allocates the image. `container.runner_entrypoint.command` runs once per dataset row. It must read the request JSON and return the result JSON.

Docker `container.mounts[].source` paths are resolved from the launcher's current working directory. The repository workflows run from the repository root, so generated templates use `./env/<name>/...` for environment-local adapter and dataset mounts. `runner_entrypoint.source` and RJob `embedded_files[].source` are instead resolved relative to their start-config file; keep those path bases distinct.

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

`agent_name: mybench` must match `env_name: mybench` in the selected task config
(`mybench_config.yaml` for Docker or `mybench_config.rjob.yaml` for RJob);
otherwise the launcher cannot find the startup definition for the scheduled rows.

For RJob mode, provide `mybench_config.rjob.yaml` and
`mybench_start.rjob.yaml`, then run with `--mode rjob`. Keep
`container.runner_entrypoint` but add the `rjob:` settings. Do not copy local
Docker bind mounts into RJob: use cluster-accessible `rjob.mount_config` or
`rjob.mount`, list local runner dependencies in `rjob.embedded_files`, use an
image and storage visible to the cluster, and use a Gateway URL other than
`127.0.0.1` or `localhost`. See [RJob Mode](../internal/rjob-mode.md).

### Environment parameter transport

`env_params` is the benchmark's runtime configuration. The launcher includes
the fully expanded object (including the current dataset row) in
`SimulationStartRequest` on stdin and in `SAFACTORY_START_REQUEST_JSON`; the
fixed runner passes it unchanged to `adapter.py`. Use this channel for paths,
flags, timeouts, native command arguments, and benchmark-specific settings.

Values under `container.env`/`rjob.env` are static process environment values
only (for example `NO_PROXY`). They are merged with launcher-provided
`SAFACTORY_*` variables at episode start. Do not duplicate the dataset or put
secrets in committed start YAML. If a native program only accepts environment
variables, have the adapter derive them from `request['env_params']` and pass
them to its subprocess.

### The PRMEval reference layout

`env/prmeval/` is the checked-in example of this separation:

```text
env/prmeval/
  runner.py                 # fixed protocol shell
  adapter.py                # one-row PRMEval invocation
  rule_evaluator.py         # optional MSE -> 0..10 mapping
  prmeval_config.yaml       # Docker image + rows + env_params
  prmeval_start.yaml        # Docker command + mounts
  prmeval_config.rjob.yaml  # RJob image + rows + env_params
  prmeval_start.rjob.yaml   # RJob resources + embedded files + mounts
  datasets/samples.jsonl
```

Use it as the standard when adding another benchmark: keep the runner shell
stable, put native logic in the adapter, and let the two RJob files add only
cluster-specific image/storage/resource settings. `env/prmeval/README.md`
describes the paths and the fast contract check.

## 6. Validate Locally, Then Run A Live Smoke Test

First fill `request.smoke.json` with one case and its environment parameters:

```bash
python skills/safactory-workflows/scripts/contract_smoke.py \
  --runner env/myagent/runner.py \
  --request env/myagent/request.smoke.json \
  --require-model-call
```

If native dependencies are not installed locally, add
`--adapter path/to/adapter_fixture.py` to copy an explicit fixture beside the
runner. This validates only the SAfactory protocol and Gateway routing; it
does not claim that the native benchmark works.

Repeat with `--input-mode env` to check environment-variable input. The helper owns a mock non-streaming chat endpoint and checks the result JSON/session identity, with no Gateway service or cluster. Native dependencies must be available locally or explicitly mocked in environment-specific tests. It does not verify image pulls, mounts, real Gateway persistence, or RJob scheduling. Include native failure and output-mapping fixtures; report precisely what was mocked.

When a real image, dataset, route, and runtime are available, this command starts Gateway, waits for readiness, runs Launcher, and stops its processes. Run from the repository root using a config containing only 1–2 cases:

```bash
python skills/safactory-workflows/scripts/live_smoke.py \
  --gateway-config gateway/config.local.yaml -- \
  --mode docker \
  --agent-config env/myagent/myagent_config.yaml \
  --agent-start-config env/myagent/myagent_start.yaml \
  --llm-model YOUR_ROUTE_KEY \
  --job-id myagent-docker-smoke \
  --pool-size 1 --max-workers 1 --max-steps 10
```

The helper uses Gateway's SQLite URI when `--db-path` is omitted. If a Gateway is already running, verify its readiness, route, and storage, then run `launcher.py` directly; the helper does not take over an occupied port. RJob uses the same local contract tests, followed by a live run with `.rjob.yaml` files, the global `--rjob-config`, matching storage, and a cluster-reachable Gateway URL when a cluster is available. See the [integration workflow](../../skills/safactory-workflows/references/environment-integration.md) for the RJob command settings.

For a live integration-only run, check the runtime result, native output, completed rows, Launcher logs, and Gateway request/trajectory records. Gateway helper output is in `logs/smoke/gateway.log`; other logs follow the configured paths. Omit `--enable-evaluation` and do not require a final normalized reward. Report local contract, live deployment, and evaluation results separately.

## Optional Evaluation

When evaluation is requested, copy the [evaluator template](../../skills/safactory-workflows/assets/environment/rule_evaluator.py), fill `score_metrics`, and start the launcher with
`--enable-evaluation`. The file is discovered from `agent_root` and `env_name`;
no evaluator registration is read from `env_params`.

The runner should preserve raw benchmark output in `metrics` or output files, and the rule evaluator should normalize benchmark-specific score scales, pass conditions, and error cases into Safactory's 0 to 10 score. It runs only during evaluation and should not rerun the benchmark case.

See [Evaluation](evaluation.md).

## BaseEnv Note

`core.env.BaseEnv` still exists for older library-style, in-process environment implementations. The current v2 launcher path schedules external runtimes through agent configs and start configs. Prefer the runtime adapter approach above unless you are extending an older in-process integration.
