# Bench 接入后的评测：自定义 rule_evaluator

这一节只介绍最简单的一种评测方式：Bench 已经作为 Safactory runtime 接入并能跑完一个 case，评测阶段只用自定义 `rule_evaluator.py` 把 bench 的原始结果转换成 Safactory reward。

## 运行链路

一次带评测的运行流程是：

1. Launcher 读取 bench 的 agent config，把 dataset 展开成多条 case。
2. Worker 为每条 case 创建 gateway session，并调用 bench 的 runtime 脚本。
3. Bench runtime 运行原生 benchmark harness，通过当前 gateway session 调用被测模型。
4. Bench runtime 在 stdout 输出 `SimulationStartResult` JSON，并把 bench 原始结果放到 `metrics`。
5. Launcher 关闭 gateway session，等待轨迹写入完成。
6. Evaluation 阶段调用 `rule_evaluator.py`。
7. `rule_evaluator.py` 读取 dataset、bench metrics、trajectory，返回 0 到 10 的分数。
8. Safactory 把该分数写回 `session_steps` 的 `reward` / `step_reward`。

接入 bench 时应把 dataset 的一行设计成一个 bench case，让 Safactory 以 task 级别调度。这样每个 case 都有独立 `session_id` 和 gateway trajectory，rule evaluator 读到的轨迹不会混入其他 case 的模型调用。

Docker、RJob 和 Sandbox rollout 都遵循相同的评测顺序，Sandbox 实例会保留到评测结束。Agent-eval 可使用 `target_access_mode: sandbox_proxy`；Sandbox target 不支持 `direct_docker`。

启动命令示例：

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

`--llm-model` 是 rollout 模型在 gateway 里的 route key。Rule evaluator 不需要单独的评测模型。

## Bench runtime 需要产出什么

Bench runtime 仍然按普通 runtime 接口运行：读取 `SimulationStartRequest`，执行一个 case，最后输出一条 `SimulationStartResult` JSON。

关键是把 rule evaluator 需要的原始结果放进 `metrics`：

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

建议至少保留这些字段：

| 字段 | 用途 |
|------|------|
| `bench_case_id` | 对齐 dataset 中的 case id，便于排查。 |
| `bench_score` | bench 原生分数，例如 0 到 1、0 到 100 或其他原始尺度。 |
| `bench_passed` | 如果 bench 有二值通过结果，放在这里。 |
| `bench_reason` | bench 原生判定原因。 |
| `bench_output_path` | 可选。bench 详细结果文件路径。 |

`total_reward` 可以先填 `0.0` 或 bench 原始 reward。启用 `rule_evaluator` 后，最终训练/统计使用的 reward 会由 `rule_evaluator.py` 返回值覆盖。

## Evaluation config 怎么写

如果一个 job 只接入一个 bench，可以把 rule evaluator 直接写在 `--evaluation-config` 的 `default_specs` 中。

创建 `evaluator/configs/mybench_rule_eval.yaml`：

```yaml
evaluation:
  max_concurrency: 4

  default_specs:
    - eval_id: mybench_rule
      method: rule_evaluator
      rule_evaluator: env/mybench/rule_evaluator.py
      timeout_s: 60
```

字段含义：

| 字段 | 含义 |
|------|------|
| `max_concurrency` | 最多同时运行多少个 rule evaluator。 |
| `default_specs` | 当 case 没有单独 eval spec 时，默认使用这里的评测规则。 |
| `method: rule_evaluator` | 指定使用 Python 规则评测。 |
| `rule_evaluator` | Python 文件路径。相对路径按运行目录解析。 |
| `timeout_s` | 单条 case 的评测超时时间。 |

这个配置只需要以上字段，保持为最小配置即可。

## rule_evaluator.py 接口

创建 `env/mybench/rule_evaluator.py`。文件需要导出 `evaluate`、`evaluate_rule` 或 `RuleEvaluator`。

最常用的是导出 `evaluate(request, spec, trajectory)`：

```python
def evaluate(request, spec, trajectory):
    dataset = request.env_params.get("dataset", {})
    start_result = request.start_result
    metrics = getattr(start_result, "metrics", {}) or {}

    raw_score = float(metrics.get("bench_score", 0.0) or 0.0)
    passed = bool(metrics.get("bench_passed", False))

    # 示例：bench_score 是 0 到 1，转换成 Safactory 的 0 到 10。
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

`evaluate` 可以读取：

| 对象 | 能拿到什么 |
|------|------------|
| `request.env_params["dataset"]` | 当前 dataset 行，也就是当前 bench case 的输入数据。 |
| `request.start_result.metrics` | bench runtime 返回的原始评测结果。 |
| `trajectory.final_response` | 模型最后一次回复。 |
| `trajectory.steps` | 完整交互轨迹。 |
| `spec` | 当前 eval spec，例如 `eval_id`、`timeout_s`、`rule_evaluator`。 |

返回值可以是：

```python
10.0
```

或：

```python
{
    "score": 8.0,
    "reason": "passed 4 of 5 checks",
    "artifacts": {"passed_checks": 4, "total_checks": 5},
}
```

`score`、`normalized_score_10`、`reward` 三个字段任选一个即可。Safactory 会把最终分数限制在 0 到 10。

## 多个 bench 怎么配置

如果同一个 job 里有多个 bench，不要把某一个 bench 的 rule evaluator 写进全局 `default_specs`。改成在每个 bench 的 agent config 里指定自己的 evaluator：

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

此时 `--evaluation-config` 可以只保留全局运行参数：

```yaml
evaluation:
  max_concurrency: 4
```

如果 `env/<bench>/rule_evaluator.py` 存在，也可以不写 `evaluation.rule_evaluator`，Safactory 会自动尝试发现该文件。

## 排错

| 现象 | 检查项 |
|------|--------|
| 没有执行 evaluation | 确认启动命令包含 `--enable-evaluation`。 |
| 没有找到 rule evaluator | 检查 `rule_evaluator` 路径，或确认 `env/<bench>/rule_evaluator.py` 存在。 |
| 分数一直是 0 | 检查 bench runtime 是否把原始结果写进 `metrics`，以及 `rule_evaluator.py` 是否正确读取字段。 |
| 评测失败后 reward 为 0 | 查看日志里的 rule evaluator exception；失败的 rule 评测会按 0 分回写。 |
| 轨迹为空 | 检查 gateway 和 launcher 是否使用同一个 `--db-path`，以及 bench runtime 是否确实通过当前 gateway session 调模型。 |
