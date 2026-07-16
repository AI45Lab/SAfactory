# Rule Evaluator

Safactory only supports environment-local Python rule evaluation. When evaluation is enabled, the evaluator is discovered by convention:

```text
<agent-root>/<env_name>/rule_evaluator.py
```

The default agent root is `env`. For an environment named `mybench`, the file must be `env/mybench/rule_evaluator.py`. If the file does not exist, evaluation is skipped for that environment. No YAML registration, `env_params` evaluator setting, or separate evaluation config is required.

## Runtime Flow

1. A worker runs one case and receives its `SimulationStartResult`.
2. The launcher closes the gateway session and waits for trajectory persistence.
3. It discovers `rule_evaluator.py` from the agent root and environment name.
4. The rule reads the current case, runtime metrics, and trajectory, then returns a score from 0 to 10.
5. Safactory writes the score back to trajectory `reward` / `step_reward` fields.
6. The runtime resource is released after evaluation.

Docker, RJob, and Sandbox use the same flow.

Example:

```bash
python launcher.py \
  --agent-config env/mybench/mybench_config.yaml \
  --agent-start-config env/mybench/mybench_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --enable-evaluation \
  --db-path sqlite://env_trajs.db \
  --pool-size 1
```

## Runtime Output

The runtime reads `SimulationStartRequest`, runs one case, and prints one `SimulationStartResult` JSON object. Put raw inputs needed by the rule in `metrics`:

```json
{
  "session_id": "same-session-id",
  "status": "succeeded",
  "total_reward": 0.0,
  "step_count": 8,
  "terminated": true,
  "truncated": false,
  "error_text": null,
  "metrics": {
    "bench_case_id": "case-001",
    "bench_score": 0.73,
    "bench_passed": true,
    "bench_reason": "all required checks passed"
  }
}
```

When evaluation is enabled, the score returned by `rule_evaluator.py` becomes the final reward used for training and reporting.

## rule_evaluator.py Interface

The file must export `evaluate`, `evaluate_rule`, or `RuleEvaluator`. The common form is:

```python
def evaluate(request, spec, trajectory):
    dataset = request.env_params.get("dataset", {})
    metrics = getattr(request.start_result, "metrics", {}) or {}

    raw_score = float(metrics.get("bench_score", 0.0) or 0.0)
    passed = bool(metrics.get("bench_passed", False))
    score = max(0.0, min(10.0, raw_score * 10.0)) if passed else 0.0

    return {
        "score": score,
        "reason": metrics.get("bench_reason") or "converted from bench metrics",
        "artifacts": {
            "task_id": dataset.get("task_id") or dataset.get("case_id"),
            "bench_case_id": metrics.get("bench_case_id"),
            "raw_bench_score": raw_score,
        },
    }
```

Available data:

| Object | Content |
|--------|---------|
| `request.env_params["dataset"]` | Current dataset case. It is rule input only and is not used for evaluator discovery or configuration. |
| `request.start_result.metrics` | Raw runtime result. |
| `trajectory.final_response` | Final model response. |
| `trajectory.steps` | Full interaction trajectory. |
| `spec` | Automatically generated rule evaluation spec. |

The return value may be a number or a dict containing `score`, `normalized_score_10`, or `reward`. Safactory clamps the final score to 0 through 10.

Synchronous rules run in a thread to avoid blocking async workers. Each evaluation has a default 60-second timeout, and total evaluator concurrency follows launcher worker concurrency.

## Multiple Environments

A job may contain multiple environments. Put one evaluator in each environment directory:

```text
env/
├── bench_a/rule_evaluator.py
└── bench_b/rule_evaluator.py
```

The lease environment name selects the file. Safactory does not read evaluator settings from task data or accept custom evaluator paths.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Evaluation does not run | Confirm `--enable-evaluation` is present and `<agent-root>/<env_name>/rule_evaluator.py` exists. |
| Score is always 0 | Confirm the runtime writes the raw result to `metrics` and the rule reads matching field names. |
| Evaluation failure gives 0 reward | Inspect the rule evaluator exception; failures and timeouts are committed as 0. |
| Empty trajectory | Confirm gateway and launcher use the same SQLite DB URI and the runtime calls the model through the current gateway session. |
