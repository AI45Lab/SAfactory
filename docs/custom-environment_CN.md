# 自定义 Runtime

本文说明如何把一个新的 agent 或 bench 接入 Safactory 的 custom environment。Safactory v2 把两者都视为 external runtime：runtime 可以是 Python、Node.js、shell，也可以是 benchmark 自己的 harness 包装脚本。它接收 `SimulationStartRequest`，只执行一个 task/case，通过 gateway 调用模型，最后输出一条 result JSON。

最重要的调度模型是：dataset 的一行就是一个 task 级调度单元。Launcher 会为每个 task 创建独立的 `job_environments` 记录、`session_id` 和 gateway session，然后用同一个 image + runner 去执行这一条 task。这样模型请求、gateway telemetry、runtime result、evaluation reward 都绑定到同一个 session，避免不同 case 的轨迹互相污染。接入 bench 时不要让 runner 在一个 episode 里循环整个 benchmark dataset；应该让 dataset 每行表示一个 bench case，由 Safactory 逐 task 调度。

通常需要新增或确认五个核心组件：

| 组件 | 位置 | 作用 | 示例 |
|------|------|------|------|
| `image` | `agent config` 的 `env_image`，RJob 可在 start config 里覆盖 | 运行时镜像。包含 agent/bench 依赖、benchmark harness、Python/Node 运行环境等。 | `myagent-image:latest`、`mybench-image:latest` |
| `runner_entrypoint` | 通常是 `env/<name>/runner.py` 或 `env/<name>/runner.mjs`，并由 `start_config.container.runner_entrypoint.command` 调用 | Safactory 和原生 agent/bench 之间的适配层。读取 `SimulationStartRequest`，取出 `env_params.dataset`，调用 gateway 中的被测模型，运行一个 task/case，向 stdout 输出 result JSON。 | `python /tmp/safactory-mybench-runner.py` |
| `rule_evaluator` | 可选，常见为 `env/<name>/rule_evaluator.py` | 把 runner 写入 `metrics` 的原始 bench 结果和 gateway 轨迹转换为 Safactory 的 0 到 10 分 reward。纯 agent smoke test 可以先不写；bench 通常建议写。 | `env/mybench/rule_evaluator.py` |
| `config` | `env/<name>/<name>_config.yaml` | 定义 task 行：`env_name`、`env_image`、`dataset`、`env_num`、`env_params`，以及可选 evaluation 配置。 | `env/mybench/mybench_config.yaml` |
| `start_config` | `env/<name>/<name>_start.yaml` | 定义同名 runtime 如何启动：runner entrypoint、workdir/env、Docker/RJob 参数和额外挂载。`agent_name` 必须匹配 `config` 里的 `env_name`。 | `env/mybench/mybench_start.yaml` |

Agent 和 bench 的差别主要在 runner 和 evaluator：

- Agent runtime 通常直接根据 `env_params.dataset` 生成 prompt、工具任务或交互流程，评测可以来自最终回复、markdown eval task 或自定义 rule。
- Bench runtime 通常包装一个已有 benchmark harness。runner 只取当前 dataset row 对应的单个 case，把原生 bench 的分数、通过状态、输出路径等写入 `metrics`，再由 `rule_evaluator.py` 统一换算 reward。

## 1. 编写 Runtime 脚本

创建 `env/myagent/runner.py`：

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

Runtime 打印 failure result 后也建议以 `0` 退出。在 JSON mode 下，如果 runtime 非零退出，Docker/RJob runner 会把整个 episode 视为 runtime failure。

## 2. 理解 Request

Runtime 会通过 stdin 和 `SAFACTORY_START_REQUEST_JSON` 收到 `SimulationStartRequest`。

重要字段：

| 字段 | 含义 |
|------|------|
| `job_id` | Launcher run ID。 |
| `session_id` | Session/environment UUID。构造 gateway URL 时使用。 |
| `agent_name`, `agent_id` | Adapter 名和环境行 ID。 |
| `group_id` | RL grouping ID。 |
| `gateway_base_url` | Gateway session root，例如 `http://127.0.0.1:8000/v1/sessions`。 |
| `model` | 来自 `--llm-model` 的 gateway route key。 |
| `temperature` | 采样温度。 |
| `max_steps` | Launcher 传入的步数预算。 |
| `storage_type`, `storage_config` | 存储信息。 |
| `env_params` | 展开的 YAML 参数，包含 `dataset`。 |
| `metadata` | 容器 ID、image、row ID、worker ID 等 runtime 元数据。 |

`request_env()` 还会注入一些常用环境变量：

| 变量 | 含义 |
|------|------|
| `SAFACTORY_SESSION_ID` | 当前 session ID。 |
| `SAFACTORY_GATEWAY_BASE_URL` | Gateway session root。 |
| `SAFACTORY_GATEWAY_SESSION_URL` | 宿主机视角的 session URL。 |
| `SAFACTORY_GATEWAY_SESSION_URL_CONTAINER` | 容器可访问的 session URL。Localhost 会被改写为 `host.docker.internal`。 |
| `SAFACTORY_ROUTE_MODEL` | 从 dataset/env params/request 推断出的 route model。 |
| `SAFACTORY_MODEL_REF` | Provider 风格 model ref，例如 `safactory/<route>`。 |
| `OPENROUTER_BASE_URL` | 容器可访问 gateway session URL 的别名。 |

## 3. 返回 Result

向 stdout 输出一条 result JSON。其他日志建议写 stderr。

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

字段：

| 字段 | 必需 | 含义 |
|------|------|------|
| `session_id` | 是 | 必须匹配 request session ID。 |
| `status` | 是 | 通常为 `succeeded` 或 `failed`。 |
| `total_reward` | 是 | 可选 evaluator override 之前的 rollout reward。 |
| `step_count` | 是 | Runtime 上报的 step 数。 |
| `terminated` | 是 | Episode 是否自然结束。 |
| `truncated` | 是 | 是否因超时或步数限制截断。 |
| `error_text` | 否 | 失败详情。 |
| `metrics` | 否 | Adapter 自定义 JSON 对象。 |

对于 bench，`metrics` 是 runner 和 `rule_evaluator.py` 之间最重要的接口。建议至少写入 case id、原生分数、通过状态、原因和详细输出路径：

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

## 4. 添加 Agent Config

Agent config 是 task 调度的来源。`env_name` 绑定 start config，`env_image` 指定 runtime image，`dataset` 决定要展开多少个 task，`env_params` 会原样传给 runner。

创建 `env/myagent/myagent_config.yaml`：

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

创建 `env/myagent/datasets/tasks.jsonl`：

```jsonl
{"task_id": "hello-001", "prompt": "Write one short greeting."}
```

Dataset row 会出现在 `request.env_params.dataset`。

Bench 接入时也使用同一个 config 结构，只是 dataset 每行应该代表一个 bench case，而不是一个完整 benchmark batch：

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

上面的两行会被调度成两个独立 episode，并分别产生独立 `session_id`、gateway 轨迹和 reward。不要在 `runner.py` 里再次读取并循环 `cases.jsonl`，否则多个 case 的模型调用会落在同一条轨迹里，训练和评测都会变得不可解释。

## 5. 添加 Agent Start Config

Start config 描述 image 被拉起后如何执行 runner entrypoint。`runner_entrypoint.command` 是每个 task/case 都会执行一次的命令，它必须读取 request JSON 并输出 result JSON。当 `runner_entrypoint.source` 指向本地文件时，Docker 会挂载它，RJob 会自动嵌入它；该 source 路径相对 start config 文件解析。

创建 `env/myagent/myagent_start.yaml`：

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

Bench 的 start config 写法完全相同，只是换成 bench image 和 bench runner：

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

`agent_name: mybench` 必须和 `mybench_config.yaml` 里的 `env_name: mybench` 一致；否则 launcher 找不到对应的启动方式。

## 6. 运行 Smoke Test

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

查看日志：

- `logs/<run>/main.log`：launcher 和 scheduler 事件。
- `logs/gateway.log`：gateway 事件。
- `logs/gateway_requests.jsonl`：请求/响应日志。
- `results/` 下 adapter 自己的输出目录。

## 可选评测

可以用三种方式添加评测：

- 在 agent config 中添加 inline `env_params.evaluation.specs`。
- 在 `env/myagent/eval_tasks/<dataset>/<task_name>.md` 下添加 markdown task。
- 添加 `env/myagent/rule_evaluator.py`。

新增 bench 时优先使用 `rule_evaluator.py`：runner 保留 bench 原始输出，rule evaluator 负责把不同 bench 的分数尺度、通过条件和异常情况统一成 Safactory reward。这个文件只在 evaluation 阶段执行，不应该重新跑 bench case。

见[评测](evaluation_CN.md)。

## BaseEnv 说明

`core.env.BaseEnv` 仍然存在，适合库内环境实现。但当前 v2 launcher 路径是通过 agent config 和 start config 调度 external runtime。除非你在扩展旧的 in-process 集成，否则优先使用本文的 runtime 方式。
