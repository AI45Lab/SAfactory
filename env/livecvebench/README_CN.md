# LiveCVEBench adapter

该 adapter 将 `LiveCVEBench-verified` 和 `PatchEval-verified` 的每个任务作为一个
Safactory episode，在 `livecvebench-runner:latest` 容器内执行一次 `tb run`。

## 前置条件

先按照 LiveCVEBench 仓库的 `Dockerfile.runner` 构建镜像：

```bash
cd /home/qiupanjia/code/LiveCVEBench-Preview-master
docker build -f Dockerfile.runner -t livecvebench-runner:latest .
```

Safactory 会自行启动一个 privileged runner 容器。不要同时启动手工创建的
`livecvebench-runner` 容器，因为两个嵌套 Docker daemon 不能同时使用同一个
`livecvebench-runner-docker` 数据卷。

## 运行

先启动 Safactory Gateway，然后运行：

```bash
cd /home/qiupanjia/code/SAfactory
python launcher.py \
  --mode docker \
  --agent-config env/livecvebench/livecvebench_config.yaml \
  --agent-start-config env/livecvebench/livecvebench_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --db-path sqlite://env_trajs.db \
  --pool-size 1 \
  --max-workers 1 \
  --agent-start-timeout-s 960 \
  --enable-evaluation
```

默认执行 410 个任务，其中包括 183 个 LiveCVEBench 任务和 227 个 PatchEval
任务。agent 为 `oracle`，每个任务的 Terminal-Bench 测试超时为 900 秒。原生结果写入：

```text
/home/qiupanjia/code/LiveCVEBench-Preview-master/container-runs/
  safactory/<job_id>/<session_id>/
```

每个 session 根目录还会写入 `safactory-result.json`，其中直接包含
`metrics.suite`（`livecvebench` 或 `patcheval`）、`metrics.task_id`、
`metrics.is_resolved`、分数以及 Terminal-Bench 原生结果路径。例如：

```json
{
  "status": "succeeded",
  "metrics": {
    "suite": "patcheval",
    "task_id": "cve-2021-4315",
    "score": 1.0,
    "is_resolved": true
  }
}
```

若只想运行一个任务，可复制数据集 JSONL，仅保留：

```json
{"task_id":"cve-2025-3248","suite":"livecvebench"}
```

然后在 `livecvebench_config.yaml` 中将 `dataset` 指向这个文件。

`suite` 用于选择任务目录：

- `livecvebench`：`/benchmark/tasks/LiveCVEBench-verified`
- `patcheval`：`/benchmark/tasks/PatchEval-verified`

目录映射集中配置在 `livecvebench_config.yaml` 的 `env_params.dataset_paths`，
数据集无需在每一行重复容器绝对路径。若当前 runner 镜像构建时间早于
`PatchEval-verified` 加入上游仓库的时间，需要重新构建该镜像。

## 分数

runner 读取 Terminal-Bench 的 `results.json`：

- `is_resolved=true`：原始分数 1，Safactory evaluator reward 10。
- `is_resolved=false`：原始分数 0，Safactory evaluator reward 0。

每个 session 使用独立输出目录，避免并发任务之间混淆结果。当前配置保持
`pool-size=1`，因为所有 runner 共享同一个嵌套 Docker 数据卷。
