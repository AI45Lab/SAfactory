# 配置

Safactory 有几类明确的配置入口：

1. `gateway` 配置：模型 route、telemetry、请求日志和存储。
2. `launcher.py` CLI 参数：调度、存储、模型 route 选择、评测和运行模式。
3. `--agent-config` 指定的 agent config YAML，或 `--agent-root` 下的全部配置。
4. `--agent-start-config` 指定的 agent start config YAML。
5. 可选的 `--rjob-config` 全局 RJob 配置。
6. 可选的 `--sandbox-config` 全局 Sandbox 配置。
7. 可选的 `--evaluation-config` evaluator runtime 配置。

`--mode` 支持 `docker`、`rjob` 和 `sandbox`。第三种运行时详见 [Sandbox 模式](sandbox-mode_CN.md)。

本地 SQLite 运行时，gateway 和 launcher 必须共享同一个 DB URI。

## 最小本地运行

启动 gateway：

```bash
python -m gateway --config gateway/config.local.yaml
```

运行一个 agent config：

```bash
python launcher.py \
  --agent-config env/openclaw/openclaw_config.yaml \
  --agent-start-config env/openclaw/openclaw_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --db-path sqlite://env_trajs.db \
  --pool-size 1 \
  --max-workers 1
```

## 必要 CLI 参数

| 类别 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| Job | `--job-id` | 为空时生成 | 写入环境行和轨迹行的标识符。 |
| Runtime | `--mode` | `docker` | 运行时分配器：`docker`、`rjob` 或 `sandbox`。 |
| Config | `--agent-config` | `None` | 单个 agent task YAML 路径。 |
| Config | `--agent-root` | `env` | 未设置 `--agent-config` 时扫描子目录 YAML。无法解析的 YAML 会 warning 后跳过。 |
| Config | `--agent-start-config` | `None` | 定义每个 agent runtime 如何以 Docker、RJob 或 Sandbox 启动。 |
| Config | `--rjob-config` | `config.yaml` | 全局 RJob 连接和鉴权配置。 |
| Config | `--sandbox-config` | `config.yaml` | 全局 OpenSandbox/Brainbox 连接与 Environment 配置。 |
| Storage | `--storage-type` | `sqlite` | `sqlite` 或 `cloud`。 |
| Storage | `--db-path` | SQLite 下为 `sqlite://env_trajs.db` | SQLite DB URI。Cloud storage 会忽略。 |
| Gateway | `--gateway-base-url` | `http://127.0.0.1:8080/v1/sessions` | Gateway session root。需要按实际 gateway 端口覆盖。 |
| LLM | `--llm-model` | `default` | Agent rollout 使用的 gateway route key。 |
| LLM | `--llm-temperature` | `0.3` | 传给 agent runtime 的采样温度。 |
| Episode | `--max-steps` | `1000` | 传给 runtime request 的最大步数。Gateway `max_steps` 还能额外限制请求数。 |
| Pool | `--pool-size` | `1` | 基础并发。Warm pool size 为 `ceil(pool_size * multiplier)`。 |
| Pool | `--multiplier` | `1.2` | Warm-pool 倍率。 |
| Pool | `--max-workers` | `0` | Worker 数上限。`0` 使用 warm-pool size。 |

## 完整 CLI 参考

| 类别 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| Storage | `--rebuild-table` / `--no-rebuild-table` | `false` | SQLite 下，加载配置前删除 DB 文件。 |
| Storage | `--disable-buffer` | buffer 启用 | 禁用缓冲写入。 |
| Storage | `--buffer-size` | `100` | 写入缓冲区容量。 |
| Storage | `--flush-interval` | `5.0` | 写入缓冲刷新间隔，单位秒。 |
| Docker | `--docker-bin` | `docker` | Docker 可执行文件。 |
| Docker | `--docker-pull-policy` | `never` | `never` 或 `always`。 |
| Docker | `--docker-startup-concurrency` | `8` | Docker 启动操作最大并发。 |
| Docker | `--cleanup-docker-container` / `--no-cleanup-docker-container` | `true` | 完成后删除 rollout 容器。 |
| Docker | `--cleanup-stale-docker-containers` / `--no-cleanup-stale-docker-containers` | `true` | 启动时清理同 job 的旧 Safactory 容器。 |
| Timeout | `--agent-start-timeout-s` | `600.0` | Agent runtime 内层超时。 |
| Timeout | `--agent-start-timeout-grace-s` | `120.0` | 外层额外超时预算。 |
| Timeout | `--container-refill-timeout-s` | `300.0` | 释放并补充一个运行时资源的最大时间。 |
| Timeout | `--row-wait-timeout-s` | `60.0` | refill 时等待新增 DB 行的最大时间。 |
| Timeout | `--row-fetch-timeout-s` | `30.0` | 单次 scheduler DB fetch 最大时间。 |
| Timeout | `--gateway-close-timeout-s` | `15.0` | Gateway close 调用 HTTP 超时。 |
| Timeout | `--gateway-close-retries` | `1` | Gateway close 重试次数。 |
| Timeout | `--gateway-close-retry-backoff-s` | `1.0` | Gateway close 重试间隔。 |
| Timeout | `--shutdown-timeout-s` | `120.0` | Launcher shutdown 最大时间。 |
| Docker timeout | `--docker-command-timeout-s` | `300.0` | 默认 Docker 生命周期命令超时。 |
| Docker timeout | `--docker-start-timeout-s` | `300.0` | Docker run/copy 启动超时。 |
| Docker timeout | `--docker-remove-timeout-s` | `120.0` | Docker remove 超时。 |
| Docker timeout | `--docker-stop-timeout-s` | `10.0` | Docker stop grace period。 |
| Docker timeout | `--docker-inspect-timeout-s` | `10.0` | Docker inspect 超时。 |
| Docker timeout | `--docker-remove-retries` | `3` | 删除容器重试次数。 |
| Docker timeout | `--docker-remove-retry-delay-s` | `2.0` | 删除重试间隔。 |
| Docker timeout | `--docker-lifecycle-timeout-s` | `60.0` | 可选 per-container cleanup 和 healthcheck 超时。 |
| Evaluation | `--enable-evaluation` | `false` | Rollout 后运行 evaluator flow。 |
| Evaluation | `--evaluation-config` | 空 | 可选的 rule evaluator runtime YAML。 |
| RL | `--rl-group-size` | `0` | 覆盖每个 YAML 环境组的 `env_num`。 |
| RL | `--rl-epoch` | `1` | 为多个 rollout epoch 复制环境配置。 |
| Circuit breaker | `--circuit-breaker` / `--no-circuit-breaker` | `true` | 最近失败/超时超过阈值时停止调度。 |
| Circuit breaker | `--circuit-breaker-window` | `50` | 滑动窗口大小。 |
| Circuit breaker | `--circuit-breaker-min-samples` | `20` | 打开前所需最小样本数。 |
| Circuit breaker | `--circuit-breaker-failure-rate` | `0.8` | 失败率阈值。 |
| Circuit breaker | `--circuit-breaker-timeout-rate` | `0.5` | 超时率阈值。 |
| Circuit breaker | `--circuit-breaker-consecutive-timeouts` | `5` | 连续超时阈值。 |
| Logging | `--log-dir` | `logs` | 运行日志根目录。 |
| Logging | `--run-name` | 空 | 可选运行目录前缀。 |
| Logging | `--console-log-level` | `INFO` | 控制台日志级别。 |
| Logging | `--file-log-level` | `DEBUG` | 文件日志级别。 |
| Logging | `--log-backup-count` | `20` | 保留的最近日志目录数。 |
| Logging | `--debug-log` | `false` | 启用部分 debug 存储日志。 |

## Gateway 配置

完整说明见 [Gateway](gateway_CN.md)。最常改的字段如下：

```yaml
listen_port: 8000
base_session_path: /v1/sessions
max_steps: -1
storage_type: sqlite
storage_config:
  db_url: sqlite://env_trajs.db
llm_routes:
  route-key:
    base_url: http://model-server/v1
    api_key: null
    supports_stream: true
    max_concurrency: 256
```

## Agent Config YAML

Agent config YAML 定义任务行。每一行会展开成一个或多个 `job_environments` 记录。

```yaml
environments:
  - env_name: openclaw
    env_image: ghcr.io/openclaw/openclaw:latest
    env_num: 1
    dataset: ./datasets/task_brief_text_writer.jsonl
    dataset_load_mode: eager

    env_params:
      task_family: openclaw_brief_text_writer
      workload:
        total_tasks: 1
        expected_parallelism: 1
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `env_name` | 是 | Agent/runtime 名称。必须匹配 start config 中的 `agent_name`。 |
| `env_image` | 否 | Docker/RJob 分配时使用的运行时镜像或 image ID。 |
| `env_num` | 否 | 每条 dataset row 的并行拷贝数。必须为正整数。 |
| `dataset` | 否 | JSON、JSONL、YAML 或 parquet 数据路径。相对路径从配置文件所在目录解析。 |
| `dataset_load_mode` | 否 | 默认 `eager`。`parquet_row_ref` 为 parquet 文件保存轻量 row reference。 |
| `env_params` | 否 | 通过 `SimulationStartRequest.env_params` 传给 runtime 的公开参数。 |

每个 dataset item 会被处理为：

- `env_params.dataset` 设置为 dataset 行。
- 注入 config path、dataset path、dataset name 和 load mode 等内部元数据。
- 基于 `env_name` 和 task index 生成确定性的 `group_id`。

## Agent Start Config YAML

Agent start config 定义某个 `env_name` 的运行时如何启动。

单 agent 写法：

```yaml
agent_name: openclaw

container:
  workdir: /workspace
  runner_entrypoint:
    source: ./runner.mjs
    target: /tmp/safactory-openclaw-runner.mjs
    command: "node /tmp/safactory-openclaw-runner.mjs"
  mounts:
    - source: ./env/openclaw/workspace
      target: /workspace
      mode: rw
  env:
    NO_COLOR: "1"
  extra_args:
    - --add-host=host.docker.internal:host-gateway
  idle_command: "tail -f /dev/null"
```

多 agent 写法：

```yaml
agents:
  openclaw:
    container:
      workdir: /workspace
      runner_entrypoint:
        source: ./env/openclaw/runner.mjs
        target: /tmp/runner.mjs
        command: "node /tmp/runner.mjs"
  openrt:
    container:
      workdir: /app
      runner_entrypoint:
        source: ./env/openrt/runner.py
        target: /tmp/runner.py
        command: "python /tmp/runner.py"
```

Docker container 字段：

| 字段 | 说明 |
|------|------|
| `workdir` | `docker exec` 的工作目录。 |
| `runner_entrypoint` | 单个 episode 的 runner entrypoint。`source` 相对 start config 文件解析；本地文件会在 Docker 中挂载、在 RJob 中嵌入；`command` 是每个 episode 执行的命令。 |
| `idle_command` | 让已分配容器保持存活的命令。 |
| `run_command` | 兼容旧配置的命令字段。新配置优先使用 `runner_entrypoint.command`。 |
| `result_mode` | 默认 `json`。`exit_code` 会把 0 exit code 视为成功。 |
| `network`, `platform` | 可选 Docker runtime 设置。 |
| `env` | 注入容器的环境变量。 |
| `mounts` / `volumes` | Docker bind mounts。相对 `source` 路径从当前工作目录解析。 |
| `extra_args` | 额外 `docker run` 参数。 |
| `install_runner_script` | 兼容性布尔字段，默认 `false`。 |

Runtime 会通过 stdin 和 `SAFACTORY_START_REQUEST_JSON` 收到 `SimulationStartRequest`。它必须输出兼容下面结构的 JSON：

```json
{
  "session_id": "env-uuid",
  "status": "succeeded",
  "total_reward": 0.0,
  "step_count": 1,
  "terminated": true,
  "truncated": false,
  "error_text": null,
  "metrics": {}
}
```

## RJob 配置

全局 RJob 设置放在 `config.yaml` 或 `--rjob-config` 指定的文件中：

```yaml
rjob:
  cluster_entry: "https://your-rjob-platform.example"
  namespace: "your-namespace"
  access_key: "replace-me"
  secret_key: "replace-me"
  charged_group: "your-quota"
  gateway_base_url: "http://gateway.example/v1/sessions"
  submit_concurrency: 1
  cleanup_on_finish: true
  no_packaging: true
```

每个 agent 的 RJob 设置放在 `--agent-start-config` 的 `rjob:` 下：

```yaml
rjob:
  name_prefix: openrt
  image_pull_policy: IfNotPresent
  no_packaging: true
  cleanup_on_finish: true
  keep_failed_jobs: true
  resources:
    cpu: 1
    gpu: 0
    memory_in_mb: 1024
  mount_config:
    - "gpfs+gpfs://gpfs1/path/data:/app/data"
```

当 `container.runner_entrypoint.source` 指向本地文件时，RJob 模式会自动嵌入该文件。支持的 per-agent RJob key 包括连接覆盖、image pull policy、`resources`、`requests`、`env`、`labels`、`annotations`、`affinity`、`mount_config`、`mount`、`before_script`、`depends_on`、用于额外文件的 `embedded_files`、`replicas`、`poll_interval_s`、`termination_grace_period_seconds`、`local_storage_in_mb` 和清理相关 flags。

## 常用环境变量

| 变量 | 用途 |
|------|------|
| `AIEVOBOX_GATEWAY_CONFIG` | 可选路径，`launcher.py` 用它在启动前校验模型 route key。 |
| `SAFACTORY_GATEWAY_LOG_PATH` | Gateway 服务日志路径。 |
| `AIEVOBOX_SQLITE_BULK_INSERT_BATCH_SIZE` | 覆盖 SQLite 后台环境行插入 batch size。 |
| `AIEVOBOX_SQLITE_BULK_INSERT_PAUSE_S` | SQLite 后台插入 batch 之间的暂停时间。 |
