# Rule evaluator 评测

Safactory 只保留环境内的 Python 规则评测。开启评测后，系统会按约定查找：

```text
<agent-root>/<env_name>/rule_evaluator.py
```

默认 `agent-root` 是 `env`。例如环境名为 `mybench` 时，评测文件必须是 `env/mybench/rule_evaluator.py`。文件不存在时，该环境跳过评测；不需要在 YAML、`env_params` 或单独的 evaluation config 中注册路径。

## 运行链路

1. Worker 执行一个 case 并取得 `SimulationStartResult`。
2. Launcher 关闭 gateway session，等待轨迹写入完成。
3. 根据 `agent_root` 和环境名发现 `rule_evaluator.py`。
4. Rule evaluator 读取当前 case、runtime metrics 和轨迹，返回 0 到 10 的分数。
5. Safactory 把分数写回轨迹的 `reward` / `step_reward`。
6. 评测结束后释放 runtime 资源。

Docker、RJob 和 Sandbox 使用相同流程。

启动示例：

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

## Runtime 输出

Runtime 按普通接口读取 `SimulationStartRequest`，执行一个 case，最后输出一条 `SimulationStartResult` JSON。Rule evaluator 需要的原始结果应放进 `metrics`：

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

启用评测后，`rule_evaluator.py` 返回的分数会覆盖最终训练和统计使用的 reward。

## rule_evaluator.py 接口

文件需导出 `evaluate`、`evaluate_rule` 或 `RuleEvaluator`。最常用形式如下：

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

可读取的数据：

| 对象 | 内容 |
|------|------|
| `request.env_params["dataset"]` | 当前 dataset case。这里只作为规则输入，不参与 evaluator 的发现或配置。 |
| `request.start_result.metrics` | Runtime 返回的原始结果。 |
| `trajectory.final_response` | 模型最后一次回复。 |
| `trajectory.steps` | 完整交互轨迹。 |
| `spec` | 自动生成的 rule eval spec。 |

返回值可以是数字，也可以是包含 `score`、`normalized_score_10` 或 `reward` 的字典。最终分数会限制在 0 到 10。

同步规则会在线程中运行，避免阻塞异步 worker；单次评测默认超时 60 秒。评测总并发跟随 launcher worker 并发。

## 多环境

一个 job 可以包含多个环境。每个环境只需在自己的目录中放置文件：

```text
env/
├── bench_a/rule_evaluator.py
└── bench_b/rule_evaluator.py
```

系统根据每条 lease 的环境名选择对应文件，不从任务数据读取 evaluator 设置，也不接受自定义 evaluator 路径。

## 排错

| 现象 | 检查项 |
|------|--------|
| 没有执行评测 | 确认包含 `--enable-evaluation`，并检查 `<agent-root>/<env_name>/rule_evaluator.py` 是否存在。 |
| 分数一直为 0 | 检查 runtime 是否把原始结果写入 `metrics`，以及规则读取的字段名是否一致。 |
| 评测失败后 reward 为 0 | 查看 rule evaluator exception；失败和超时都会按 0 分回写。 |
| 轨迹为空 | 确认 gateway 与 launcher 使用相同的 SQLite DB URI，且 runtime 通过当前 gateway session 调用模型。 |
