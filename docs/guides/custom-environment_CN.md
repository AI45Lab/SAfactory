# 自定义环境

本文说明如何把新的 agent 或 benchmark 接入 Safactory，作为一个自定义环境运行。在 Safactory v2 中，自定义环境本质上是外部运行时适配器：它可以是 Python 脚本、Node.js 脚本、shell 命令，也可以是对已有 benchmark harness 的一层包装。

每个运行时适配器都会收到一个 `SimulationStartRequest`，执行一个任务或一个 benchmark case，通过 Safactory gateway 发起模型请求，最后返回一个 `SimulationStartResult` JSON 对象。`agent config` 和 `agent start config` 是 Safactory 沿用的历史名称，agent 和 benchmark 都使用这两类配置。

最重要的调度规则是：

> dataset 的一行就是一个被调度的 episode。

Launcher 会为每一行 dataset 创建独立的 `job_environments` 记录、`session_id` 和 gateway session，然后用同一个镜像和 runner 执行这一行。这样模型调用、gateway 记录、运行时输出和评测 reward 都会绑定到同一个 session。接入 benchmark 时，不要让 runner 在一个 episode 里循环整个 benchmark dataset。应该让每个 benchmark case 对应一行 dataset，由 Safactory 按行独立调度。

通常需要准备运行时镜像、runner、任务配置和启动配置。只有本次明确需要评测时才添加可选的 rule evaluator；原生 benchmark 有分数并不意味着接入必须包含评测。接入的边界是 SAfactory adapter 的输入/输出处理：benchmark 单 case 的执行和评测逻辑应当已经存在于 benchmark harness 或 Docker 镜像中，接入时不重写这部分逻辑。

先进行本地 adapter 契约测试，无需 Docker、已启动的 Gateway、模型凭据或内部 RJob 集群。部署条件具备后再进行真实 smoke test。Geo3K 基线可用于排查公共基础设施问题，不是接入的前置门槛；`env/geo3k` 可作为环境特定逻辑的参考。

| 组件 | 位置 | 作用 | 示例 |
|------|------|------|------|
| 运行时镜像 | 任务配置中的 `env_image`。RJob 配置通常改为集群可拉取的镜像。 | 包含 agent 或 benchmark 依赖、harness，以及 runner 需要的语言运行时。 | `myagent-image:latest`、`mybench-image:latest` |
| Runner entrypoint | 通常是 `env/<name>/runner.py` 或 `env/<name>/runner.mjs`，由 `container.runner_entrypoint.command` 调用。 | 连接 Safactory 与原生 agent 或 benchmark。它读取 request，取出 `env_params.dataset`，通过 gateway 调用被测模型，执行一个任务或 case，并返回结果 JSON。 | `python /tmp/safactory-mybench-runner.py` |
| 任务配置 | `env/<name>/<name>_config.yaml`，通过 `--agent-config` 传入。RJob 模式另提供 `<name>_config.rjob.yaml`。 | 定义任务行：`env_name`、`env_image`、`dataset`、`env_num` 和 `env_params`。每行 dataset 对应一个 case/episode。 | `env/mybench/mybench_config.yaml`、`env/mybench/mybench_config.rjob.yaml` |
| 启动配置 | `env/<name>/<name>_start.yaml`，通过 `--agent-start-config` 传入。RJob 模式另提供 `<name>_start.rjob.yaml`。 | 定义同名运行时如何启动：runner entrypoint、工作目录、环境变量、Docker 或 RJob 参数以及挂载。`agent_name` 必须匹配 `env_name`。 | `env/mybench/mybench_start.yaml`、`env/mybench/mybench_start.rjob.yaml` |
| Rule evaluator | 可选，常见路径为 `env/<name>/rule_evaluator.py`。 | 把运行时写入的原始 `metrics` 和 gateway 轨迹转换为 Safactory 的 0 到 10 分。仅接入时可以省略，即使 benchmark 自带分数；通过 `--enable-evaluation` 显式开启评测。 | `env/mybench/rule_evaluator.py` |

Agent 和 benchmark 的差别主要体现在 runner 和 evaluator：

- Agent 运行时通常把 `env_params.dataset` 转换为 prompt、工具任务或交互流程。评测使用自定义 rule evaluator。
- Benchmark 运行时通常包装已有 benchmark harness。runner 只处理当前 dataset 行对应的单个 case，把可用的原生输出和路径写入 `metrics`；需要评测时再保留评分信息，供 `rule_evaluator.py` 换算 reward。

## 1. 从固定模板开始

```bash
python skills/safactory-workflows/scripts/scaffold_environment.py myagent
# Docker 与 RJob 两套 config/start 文件固定都会生成。
# 只有需要评测时才追加 --enable-evaluation。
```

[模板目录](../../skills/safactory-workflows/assets/environment/)将协议处理固定在 `runner.py`，环境逻辑放在 `adapter.py:run_case`。填写 hook 和 YAML 参数，保留协议外壳。接入 benchmark 时，把示例问候替换成已有的原生单 case 命令和输出映射。脚手架拒绝覆盖已有目录；示例 request/dataset 只能验证脚手架，必须换成有代表性的 case 才能证明接入有效。

可选的 `rule_evaluator.py` 只需填写 `score_metrics`；没有配置评分映射时会明确失败。仅接入不需要评分字段。完整文件职责和命令见[接入 workflow](../../skills/safactory-workflows/references/environment-integration.md)。

下面的紧凑示例用于解释 runner 协议；新接入文件应从模板创建：

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

以下两条不变量由 `skills/safactory-workflows/scripts/validate_environment.py` 和
`check_environment.py` 自动检查：`env_params.results_root` 必须与对应 start 文件中
results 挂载目标一致；数据集行内存放绝对路径的字段（例如 PRMEval 的 `frames`）必须
落在某个容器挂载目标之下，以保证在 Docker 与 RJob 两种模式下解析一致。

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
| `SAFACTORY_NATIVE_PARALLEL` | 运行时是否可以并行执行原生 case。 |
| `SAFACTORY_OUTPUT_SUBDIR` | 结果根目录下的按 episode 输出子目录提示。 |
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

需要评测时，`metrics` 是 runner 和 `rule_evaluator.py` 的主要接口。请保存足够信息，让评测不必重新运行 case；仅接入时不要求这些评分字段：

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
      # 必须与 myagent_start.yaml 中的 results 挂载目标一致。
      results_root: /workspace/Safactory/results
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
      results_root: /workspace/Safactory/results
```

```jsonl
{"task_id": "case-001", "case_id": "case-001", "input": "example input", "expected": "example answer"}
{"task_id": "case-002", "case_id": "case-002", "input": "another input", "expected": "another answer"}
```

这两行会变成两个独立 episode，分别拥有自己的 `session_id`、gateway 轨迹、结果和 reward。不要让 `runner.py` 再次读取 `cases.jsonl` 并在内部循环。如果多个 case 在同一个 episode 中运行，它们的模型调用会落入同一条轨迹，训练和评测数据都会变得含混。

## 5. 添加启动配置

启动配置描述 Safactory 分配镜像后如何执行 runner。`container.runner_entrypoint.command` 会针对每一行 dataset 执行一次。它必须读取 request JSON，并返回 result JSON。

Docker 的 `container.mounts[].source` 按启动 Launcher 时的当前工作目录解析；本仓库的命令从仓库根目录执行，因此模板会使用 `./env/<name>/...` 指向环境目录。`runner_entrypoint.source` 以及 RJob 的 `embedded_files[].source` 则按各自 start 配置文件所在目录解析，二者不要混用。

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

`agent_name: mybench` 必须与所选任务配置（Docker 的
`mybench_config.yaml` 或 RJob 的 `mybench_config.rjob.yaml`）中的
`env_name: mybench` 一致，否则 launcher 找不到这些任务行对应的启动定义。

如果选择 RJob 模式，需要分别保存 `mybench_config.rjob.yaml` 和
`mybench_start.rjob.yaml`，并使用 `--mode rjob`。RJob start config 仍然保留
`container.runner_entrypoint`，但增加 `rjob:` 配置；本地 Docker 的
`container.mounts` 不应直接复制到 RJob，应改为集群可访问的
`rjob.mount_config`/`rjob.mount`。runner 依赖的本地文件要列在
`rjob.embedded_files` 中，镜像和结果存储必须能被 RJob 集群访问，Gateway URL
也不能使用 `127.0.0.1` 或 `localhost`。详见[RJob 模式](../internal/rjob-mode_CN.md)。

### 环境参数如何透传

`env_params` 是 benchmark 的运行时配置。Launcher 会把展开后的完整对象
（包括当前 dataset 行）放入 `SimulationStartRequest` 的 stdin 和
`SAFACTORY_START_REQUEST_JSON`，固定的 runner 再原样传给 `adapter.py`。
路径、开关、超时、原生命令参数和 benchmark 特有配置都应放在这里。

`container.env`/`rjob.env` 只用于静态进程环境变量（例如 `NO_PROXY`），episode
启动时会和 Launcher 注入的 `SAFACTORY_*` 变量合并。不要在提交的 start YAML
中重复 dataset 或写入密钥。如果原生程序只能读取环境变量，应由 adapter 从
`request['env_params']` 派生后传给子进程。

### PRMEval 标准目录

`env/prmeval/` 是这套分层的仓库内示例：

```text
env/prmeval/
  runner.py                 # 固定协议外壳
  adapter.py                # 单行 PRMEval 调用
  rule_evaluator.py         # 可选的 MSE -> 0..10 映射
  prmeval_config.yaml       # Docker 镜像、任务行和 env_params
  prmeval_start.yaml        # Docker 命令和挂载
  prmeval_config.rjob.yaml  # RJob 镜像、任务行和 env_params
  prmeval_start.rjob.yaml   # RJob 资源、嵌入文件和挂载
  datasets/samples.jsonl
```

接入其他 benchmark 时保持 runner 外壳稳定，把原生逻辑放到 adapter；RJob
的两个文件只增加集群相关的镜像、存储和资源设置。`env/prmeval/README.md`
说明了各文件职责和快速契约检查命令。

## 6. 本地验证与真实 Smoke Test

先将 `request.smoke.json` 填成单个 case 及其环境参数：

```bash
python skills/safactory-workflows/scripts/contract_smoke.py \
  --runner env/myagent/runner.py \
  --request env/myagent/request.smoke.json \
  --require-model-call
```

如果本地没有原生依赖，可追加
`--adapter path/to/adapter_fixture.py`，把明确的 fixture 复制到 runner 旁边。
这只验证 SAfactory 协议和 Gateway 路由，不代表 benchmark 原生逻辑可用。

追加 `--input-mode env` 再验证环境变量输入。helper 自动启动本地非流式 chat mock，检查结果 JSON 和 session 一致性，不依赖 Gateway 服务或集群。原生依赖需要本地可用，或在环境测试中明确用 fixture 替代；这不能证明镜像、挂载、真实 Gateway 落库或 RJob 调度可用。应补充原生命令失败和输出映射测试，并报告 mock 的范围。

真实镜像、数据、route 和 runtime 可用后，下面的单条命令负责启动 Gateway、等待 ready、运行 Launcher、清理自身进程。请在仓库根目录执行，任务配置只包含 1–2 个 case：

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

省略 `--db-path` 时，helper 使用 Gateway 的 SQLite URI。若 Gateway 已运行，应检查其 ready、route 和存储后直接调用 `launcher.py`；helper 不接管已占用端口。RJob 同样先执行本地契约测试；集群可用后再使用 `.rjob.yaml`、全局 `--rjob-config`、一致的存储和集群可达 Gateway URL 做真实部署验证。RJob 参数见[接入 workflow](../../skills/safactory-workflows/references/environment-integration.md)。

仅接入的真实运行检查 runner JSON、原生输出、完成状态及 Gateway 请求/轨迹记录。helper 的 Gateway 日志为 `logs/smoke/gateway.log`，其他日志按配置路径查看。省略 `--enable-evaluation`，不要求最终归一化 reward；本地契约、真实部署和评测结果分别报告。未评分时 `total_reward: 0.0` 只满足返回协议，不代表评测结果。

## 可选评测

需要评测时，从[评测器模板](../../skills/safactory-workflows/assets/environment/rule_evaluator.py)创建 `env/myagent/rule_evaluator.py`，填写 `score_metrics`，再使用 `--enable-evaluation` 启动
launcher。系统根据 `agent_root` 和 `env_name` 自动发现该文件，不从
`env_params` 读取 evaluator 注册信息。

runner 应该把原始 benchmark 输出保存在 `metrics` 或输出文件中，rule evaluator 负责把 benchmark 自己的分数尺度、通过条件和错误情况统一成 Safactory 的 0 到 10 分。它只在 evaluation 阶段执行，不应该重新运行 benchmark case。

见[评测](evaluation_CN.md)。

## BaseEnv 说明

`core.env.BaseEnv` 仍然存在，用于旧的库内、进程内环境实现。当前 v2 launcher 路径通过 agent config 和 start config 调度外部运行时。除非你在扩展旧的 in-process 集成，否则优先使用本文介绍的运行时适配器方式。
