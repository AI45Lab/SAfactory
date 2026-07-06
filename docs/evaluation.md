# Evaluation After Bench Integration: Custom rule_evaluator

This page covers the simplest evaluation path: a benchmark is already integrated as a Safactory runtime, and evaluation only uses a custom `rule_evaluator.py` to convert the benchmark's raw result into a Safactory reward.

## Runtime Flow

A run with evaluation follows this flow:

1. The launcher reads the benchmark agent config and expands the dataset into cases.
2. A worker creates a gateway session for each case and invokes the benchmark runtime.
3. The benchmark runtime runs the native benchmark harness and calls the target model through the current gateway session.
4. The benchmark runtime prints one `SimulationStartResult` JSON object to stdout and stores raw benchmark results in `metrics`.
5. The launcher closes the gateway session and waits for the trajectory to flush.
6. Evaluation calls `rule_evaluator.py`.
7. `rule_evaluator.py` reads the dataset row, benchmark metrics, and trajectory, then returns a 0 to 10 score.
8. Safactory writes that score back to `session_steps.reward` / `session_steps.step_reward`.

When integrating a benchmark, design one dataset row as one benchmark case and let Safactory schedule at task level. Each case then gets its own `session_id` and gateway trajectory, so the rule evaluator never sees model calls from other cases mixed into the current trajectory.

Example command:

```bash
python launcher.py \
  --agent-config env/mybench/mybench_config.yaml \
  --agent-start-config env/mybench/mybench_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --evaluation-config evaluator/configs/mybench_rule_eval.yaml \
  --enable-evaluation \
  --db-path sqlite://env_trajs.db \
  --pool-size 1
```

`--llm-model` is the gateway route key for the target model. This simple path does not need `--evaluation-model` because `rule_evaluator` does not call a judge model.

## Benchmark Runtime Output

The benchmark runtime still uses the normal runtime contract: read `SimulationStartRequest`, run one case, and print one `SimulationStartResult` JSON object.

The important part is to keep raw benchmark results in `metrics`:

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
    "bench_reason": "all required checks passed",
    "bench_output_path": "/workspace/results/mybench/case-001.json"
  }
}
```

Recommended fields:

| Field | Purpose |
|-------|---------|
| `bench_case_id` | Matches the dataset case id for debugging. |
| `bench_score` | Native benchmark score, such as 0 to 1 or 0 to 100. |
| `bench_passed` | Optional binary pass/fail result. |
| `bench_reason` | Native benchmark judgment reason. |
| `bench_output_path` | Optional path to detailed benchmark output. |

`total_reward` can be `0.0` or the native benchmark reward. When `rule_evaluator` is enabled, the final reward used by Safactory is the score returned by `rule_evaluator.py`.

## Evaluation Config

If a job runs only one benchmark, put the rule evaluator directly in `--evaluation-config` through `default_specs`.

Create `evaluator/configs/mybench_rule_eval.yaml`:

```yaml
evaluation:
  max_concurrency: 4
  fail_policy: zero_reward

  default_specs:
    - eval_id: mybench_rule
      method: rule_evaluator
      rule_evaluator: env/mybench/rule_evaluator.py
      timeout_s: 60
```

Fields:

| Field | Meaning |
|-------|---------|
| `max_concurrency` | Maximum number of rule evaluators running at once. |
| `fail_policy` | Reward behavior when evaluation fails. `zero_reward` gives 0. |
| `default_specs` | Default evaluation spec when a case has no case-specific spec. |
| `method: rule_evaluator` | Use Python rule evaluation. |
| `rule_evaluator` | Python file path, resolved from the run directory. |
| `timeout_s` | Timeout for one case evaluation. |

This config only needs the fields above; keep it minimal.

## rule_evaluator.py Interface

Create `env/mybench/rule_evaluator.py`. The file must export `evaluate`, `evaluate_rule`, or `RuleEvaluator`.

The common form is `evaluate(request, spec, trajectory)`:

```python
def evaluate(request, spec, trajectory):
    dataset = request.env_params.get("dataset", {})
    start_result = request.start_result
    metrics = getattr(start_result, "metrics", {}) or {}

    raw_score = float(metrics.get("bench_score", 0.0) or 0.0)
    passed = bool(metrics.get("bench_passed", False))

    # Example: bench_score is 0 to 1, convert it to Safactory's 0 to 10 scale.
    score = max(0.0, min(10.0, raw_score * 10.0))

    if not passed:
        score = 0.0

    return {
        "score": score,
        "reason": metrics.get("bench_reason") or "converted from bench metrics",
        "artifacts": {
            "task_id": dataset.get("task_id") or dataset.get("case_id"),
            "bench_case_id": metrics.get("bench_case_id"),
            "bench_output_path": metrics.get("bench_output_path"),
            "raw_bench_score": raw_score,
            "bench_passed": passed,
        },
    }
```

`evaluate` can read:

| Object | Available data |
|--------|----------------|
| `request.env_params["dataset"]` | The current dataset row, i.e. the benchmark case input. |
| `request.start_result.metrics` | Raw benchmark result returned by the runtime. |
| `trajectory.final_response` | The model's final response. |
| `trajectory.steps` | Full interaction trajectory. |
| `spec` | Current eval spec, such as `eval_id`, `timeout_s`, and `rule_evaluator`. |

The evaluator may return a number:

```python
10.0
```

or a dict:

```python
{
    "score": 8.0,
    "reason": "passed 4 of 5 checks",
    "artifacts": {"passed_checks": 4, "total_checks": 5},
}
```

Use one of `score`, `normalized_score_10`, or `reward`. Safactory clamps the final score to 0 through 10.

## Multiple Benchmarks

If one job contains multiple benchmarks, do not put one benchmark's evaluator in global `default_specs`. Configure each benchmark in its own agent config:

```yaml
environments:
  - env_name: mybench
    env_image: mybench-image:latest
    env_num: 1
    dataset: ./datasets/cases.jsonl
    env_params:
      task_family: mybench
      evaluation:
        rule_evaluator: env/mybench/rule_evaluator.py
        rule_evaluator_timeout_s: 60
```

Then `--evaluation-config` can keep only global runtime settings:

```yaml
evaluation:
  max_concurrency: 4
  fail_policy: zero_reward
```

If `env/<bench>/rule_evaluator.py` exists, `evaluation.rule_evaluator` can be omitted; Safactory will try to discover that file automatically.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Evaluation does not run | Confirm the command includes `--enable-evaluation`. |
| Rule evaluator not found | Check the `rule_evaluator` path or ensure `env/<bench>/rule_evaluator.py` exists. |
| Score is always 0 | Check that the benchmark runtime writes raw results to `metrics`, and that `rule_evaluator.py` reads the same field names. |
| Evaluation failure gives 0 reward | `fail_policy: zero_reward` turns failed evaluation into 0 reward; inspect the rule evaluator exception in logs. |
| Empty trajectory | Check that gateway and launcher use the same `--db-path`, and that the benchmark runtime calls the model through the current gateway session. |
