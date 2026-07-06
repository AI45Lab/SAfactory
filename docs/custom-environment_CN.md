# 自定义 Runtime

在 Safactory v2 中，新增环境的常规方式是新增一个 external agent runtime。这个 runtime 可以是 Python、Node.js、shell 或 benchmark 自己的程序。它接收 `SimulationStartRequest`，执行一个 episode，通过 gateway 调用模型，然后输出 result JSON。

通常需要新增：

1. 一个 runtime 脚本。
2. 一个 agent config YAML。
3. 一个 agent start config YAML。
4. 可选 eval tasks 或 rule evaluator。

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

## 4. 添加 Agent Config

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

## 5. 添加 Agent Start Config

创建 `env/myagent/myagent_start.yaml`：

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

见[评测](evaluation_CN.md)。

## BaseEnv 说明

`core.env.BaseEnv` 仍然存在，适合库内环境实现。但当前 v2 launcher 路径是通过 agent config 和 start config 调度 external runtime。除非你在扩展旧的 in-process 集成，否则优先使用本文的 runtime 方式。
