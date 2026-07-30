# 自定义环境

本文说明如何把新的 agent 或 benchmark 接入 Safactory，作为一个自定义环境运行。在 Safactory v2 中，自定义环境本质上是外部运行时适配器：它可以是 Python 脚本、Node.js 脚本、shell 命令，也可以是对已有 benchmark harness 的一层包装。

每个运行时适配器都会收到一个 `SimulationStartRequest`，执行一个任务或一个 benchmark case，通过 Safactory gateway 发起模型请求，最后返回一个 `SimulationStartResult` JSON 对象。`agent config` 和 `agent start config` 是 Safactory 沿用的历史名称，agent 和 benchmark 都使用这两类配置。

最重要的调度规则是：

> dataset 的一行就是一个被调度的 episode。

Launcher 会为每一行 dataset 创建独立的 `job_environments` 记录、`session_id` 和 gateway session，然后用同一个镜像和 runner 执行这一行。这样模型调用、gateway 记录、运行时输出和评测 reward 都会绑定到同一个 session。接入 benchmark 时，不要让 runner 在一个 episode 里循环整个 benchmark dataset。应该让每个 benchmark case 对应一行 dataset，由 Safactory 按行独立调度。

通常需要准备五个组件：

接入新环境前，先运行根目录 README 中的标准 Geo3K Docker smoke test。它可以先验证 Gateway、模型 route、存储、Docker 权限和 evaluator 链路是否正常。基线跑通后，再以 `env/geo3k` 作为完整 runtime 参考：它包含 dataset 加载、runner、Docker 启动配置和 rule evaluation。

| 组件 | 位置 | 作用 | 示例 |
|------|------|------|------|
| 运行时镜像 | agent config 中的 `env_image`。RJob 部署可以在 start config 中覆盖。 | 包含 agent 或 benchmark 依赖、harness，以及 runner 需要的语言运行时。 | `myagent-image:latest`、`mybench-image:latest` |
| Runner entrypoint | 通常是 `env/<name>/runner.py` 或 `env/<name>/runner.mjs`，由 `container.runner_entrypoint.command` 调用。 | 连接 Safactory 与原生 agent 或 benchmark。它读取 request，取出 `env_params.dataset`，通过 gateway 调用被测模型，执行一个任务或 case，并返回结果 JSON。 | `python /tmp/safactory-mybench-runner.py` |
| 任务配置 | `env/<name>/<name>_config.yaml`，通过 `--agent-config` 传入。 | 定义任务行：`env_name`、`env_image`、`dataset`、`env_num`、`env_params`，以及可选评测配置。 | `env/mybench/mybench_config.yaml` |
| 启动配置 | `env/<name>/<name>_start.yaml`，通过 `--agent-start-config` 传入。 | 定义同名运行时如何启动：runner entrypoint、工作目录、环境变量、Docker 或 RJob 参数以及挂载。`agent_name` 必须匹配 `env_name`。 | `env/mybench/mybench_start.yaml` |
| Rule evaluator | 可选，常见路径为 `env/<name>/rule_evaluator.py`。 | 把运行时写入的原始 `metrics` 和 gateway 轨迹转换为 Safactory 的 0 到 10 分。简单冒烟测试可以省略，benchmark 通常建议提供。 | `env/mybench/rule_evaluator.py` |

Agent 和 benchmark 的差别主要体现在 runner 和 evaluator：

- Agent 运行时通常把 `env_params.dataset` 转换为 prompt、工具任务或交互流程。评测使用自定义 rule evaluator。
- Benchmark 运行时通常包装已有 benchmark harness。runner 只处理当前 dataset 行对应的单个 case，把原生分数、通过状态、原因和输出路径写入 `metrics`，再由 `rule_evaluator.py` 统一换算成 Safactory reward。

## 1. 编写 Runner

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

如果任务以可控方式失败，runner 应该打印一条失败结果并以 `0` 退出。在 JSON result mode 下，非零退出表示运行时命令本身失败。Docker 和 RJob 会把它视为基础设施或运行时故障，即使 stdout 中已经打印了部分内容。

为了让解析稳定，stdout 最好只输出结果 JSON，诊断日志写到 stderr。对于较长的远程运行，runner 也可以把同一份结果对象写到 `SAFACTORY_RESULT_PATH` 指向的文件中。Safactory 会先解析 stdout，如果 stdout 中没有可解析的 JSON，再从该 artifact 路径读取结果。

## 2. 读取 Request

Safactory 会通过 stdin 和 `SAFACTORY_START_REQUEST_JSON` 同时传入 `SimulationStartRequest`。

重要字段：

| 字段 | 含义 |
|------|------|
| `job_id` | Launcher run ID。 |
| `session_id` | 当前 episode 的 session UUID。构造 gateway URL 和结果路径时使用。 |
| `agent_name`, `agent_id` | 运行时名称和环境行 ID。 |
| `group_id` | RL 分组 ID，仅在启用 RL 分组时有意义。 |
| `gateway_base_url` | Gateway session root，例如 `http://127.0.0.1:8000/v1/sessions`。 |
| `model` | 来自 `--llm-model` 的 gateway route key。 |
| `temperature` | Launcher 传入的采样温度。 |
| `max_steps` | Launcher 传入的步数预算。 |
| `storage_type`, `storage_config` | 存储后端信息。 |
| `env_params` | 展开后的 YAML 参数。当前 dataset 行在 `env_params.dataset` 中。 |
| `metadata` | 容器 ID、镜像、row ID、worker ID 等运行时元数据。 |

`request_env()` 还会注入常用环境变量：

| 变量 | 含义 |
|------|------|
| `SAFACTORY_START_REQUEST_JSON` | 完整的 `SimulationStartRequest` JSON。 |
| `SAFACTORY_JOB_ID` | Launcher run ID。 |
| `SAFACTORY_SESSION_ID` | 当前 session ID。 |
| `SAFACTORY_AGENT_NAME`, `SAFACTORY_AGENT_ID` | 运行时名称和环境行 ID。 |
| `SAFACTORY_TASK_ID`, `SAFACTORY_TASK_PATH`, `SAFACTORY_CATEGORY` | 如果 dataset 行中存在这些字段，会被复制成便捷环境变量。 |
| `SAFACTORY_RESULT_PATH` | 推荐写入结果 JSON 文件的 artifact 路径。 |
| `SAFACTORY_GATEWAY_BASE_URL` | Gateway session root。 |
| `SAFACTORY_GATEWAY_SESSION_URL` | 宿主机视角的 session URL。 |
| `SAFACTORY_GATEWAY_SESSION_URL_CONTAINER` | 容器可访问的 session URL。本地 `localhost` 地址会改写为 `host.docker.internal`。 |
| `SAFACTORY_ROUTE_MODEL` | 从 dataset、`env_params` 或 request 推断出的 route model。 |
| `SAFACTORY_MODEL_REF` | Provider 风格的模型引用，例如 `safactory/<route>`。 |
| `OPENROUTER_BASE_URL` | 容器可访问 gateway session URL 的别名。 |

## 3. 返回 Result

向 stdout 输出一个 `SimulationStartResult` JSON 对象。其他日志写到 stderr。

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
| `session_id` | 是 | 必须与 request 中的 session ID 一致。 |
| `status` | 是 | runner 正常完成时使用 `succeeded`，即使任务得分很低。运行时错误使用 `failed`。 |
| `total_reward` | 是 | 运行时上报的 reward，可能会被后续 evaluator 覆盖。 |
| `step_count` | 是 | 运行时上报的 step 数。 |
| `terminated` | 是 | episode 是否到达正常停止点。 |
| `truncated` | 是 | episode 是否因为超时或步数限制而停止。 |
| `error_text` | 否 | 运行时错误的详细信息。 |
| `metrics` | 否 | 适配器自定义 JSON 对象。benchmark 输出和文件路径建议放在这里。 |

对于 benchmark，`metrics` 是 runner 和 `rule_evaluator.py` 之间最主要的接口。请保存足够信息，让评测阶段不需要重新运行 case 就能打分：

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

## 4. 添加任务配置

任务配置是调度行的来源。`env_name` 把任务行绑定到启动配置，`env_image` 指定默认运行时镜像，`dataset` 决定展开多少行任务，`env_params` 会传给 runner。

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

这行 dataset 会出现在 request 的 `env_params.dataset` 中。

Benchmark 接入也使用同样的配置结构。关键区别是每行 dataset 应该代表一个 benchmark case，而不是一个完整 benchmark batch：

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

这两行会变成两个独立 episode，分别拥有自己的 `session_id`、gateway 轨迹、结果和 reward。不要让 `runner.py` 再次读取 `cases.jsonl` 并在内部循环。如果多个 case 在同一个 episode 中运行，它们的模型调用会落入同一条轨迹，训练和评测数据都会变得含混。

## 5. 添加启动配置

启动配置描述 Safactory 分配镜像后如何执行 runner。`container.runner_entrypoint.command` 会针对每一行 dataset 执行一次。它必须读取 request JSON，并返回 result JSON。

当 `container.runner_entrypoint.source` 指向本地文件时，该路径会相对 start config 文件解析。Docker 会把它挂载到 `target`，RJob 会通过 RJob runtime config 嵌入或分发该文件。`command` 应该执行 target 路径上的文件。

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

Benchmark 的 start config 形态相同，只是换成 benchmark 镜像和 benchmark runner：

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

`agent_name: mybench` 必须与 `mybench_config.yaml` 中的 `env_name: mybench` 一致，否则 launcher 找不到这些任务行对应的启动定义。

## 6. 运行冒烟测试

先启动 gateway，然后以单 worker、单并发运行最小测试。命令形态应与 Geo3K smoke test 一致，只替换环境路径和 route key：

```bash
python launcher.py \
  --mode docker \
  --agent-config env/myagent/myagent_config.yaml \
  --agent-start-config env/myagent/myagent_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --db-path sqlite://env_trajs.db \
  --job-id myagent-docker-smoke \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 10
```

重点检查：

- `logs/<run>/main.log`：launcher 和 scheduler 事件。
- `logs/<run>/gateway.log`：gateway 事件。
- `logs/<run>/gateway_requests.jsonl`：请求和响应记录。
- `results/` 下该适配器自己的输出目录。

## 可选评测

添加 `env/myagent/rule_evaluator.py`，并使用 `--enable-evaluation` 启动
launcher。系统根据 `agent_root` 和 `env_name` 自动发现该文件，不从
`env_params` 读取 evaluator 注册信息。

runner 应该把原始 benchmark 输出保存在 `metrics` 或输出文件中，rule evaluator 负责把 benchmark 自己的分数尺度、通过条件和错误情况统一成 Safactory 的 0 到 10 分。它只在 evaluation 阶段执行，不应该重新运行 benchmark case。

见[评测](evaluation_CN.md)。

## BaseEnv 说明

`core.env.BaseEnv` 仍然存在，用于旧的库内、进程内环境实现。当前 v2 launcher 路径通过 agent config 和 start config 调度外部运行时。除非你在扩展旧的 in-process 集成，否则优先使用本文介绍的运行时适配器方式。
