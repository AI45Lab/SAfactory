# Gateway

Safactory 的模型流量都经过 `gateway`。Gateway 提供 OpenAI 兼容端点，将 model 名映射到配置好的上游 route，把 telemetry 写入与 `launcher.py` 相同的存储后端，并执行 `max_steps` 等 session 级限制。

## 启动 Gateway

先创建本地配置：

```bash
cp gateway/config.example.yaml gateway/config.local.yaml
```

替换为自己的 route key 和上游模型端点：

```yaml
listen_host: 0.0.0.0
listen_port: 8000
base_session_path: /v1/sessions
max_steps: -1

storage_type: sqlite
storage_config:
  db_url: sqlite://env_trajs.db

llm_routes:
  YOUR_ROUTE_KEY:
    base_url: http://YOUR_LLM_HOST/v1
    api_key: YOUR_API_KEY
    supports_stream: true
    max_concurrency: 64
```

运行：

```bash
python -m gateway --config gateway/config.local.yaml
```

Gateway 默认将服务日志写入 `logs/gateway.log`。可以用 `SAFACTORY_GATEWAY_LOG_PATH` 覆盖。

## 端点

默认 `base_session_path: /v1/sessions` 时，gateway 暴露以下端点：

| 端点 | 用途 |
|------|------|
| `GET /healthz` | 存活检查。 |
| `GET /readyz` | 就绪检查，返回 `storage_type`、`storage_config` 和 `max_steps`。 |
| `GET /metrics` | Prometheus 风格指标，包括 admission、telemetry 和 route inflight。 |
| `POST /v1/chat/completions` | 标准 OpenAI 兼容 chat completions 透传，不绑定 session。 |
| `POST /v1/responses` | 标准 OpenAI 兼容 responses 透传，不绑定 session。 |
| `POST /v1/sessions/{session_id}/chat/completions` | Session 级 chat completions，是 agent runtime 主要使用的端点。 |
| `POST /v1/sessions/{session_id}/responses` | Session 级 responses 端点。 |
| `GET /v1/sessions/{session_id}` | 查看 gateway session 状态。 |
| `POST /v1/sessions/{session_id}/close` | soft-close session，并写入 close telemetry。 |

Session 级请求会把轨迹行关联到 Safactory 的 `session_id`。

## Route Key

`llm_routes` 是 route key 到上游端点的映射：

```yaml
llm_routes:
  dsv4pro:
    base_url: http://10.0.0.10:8182/v1
    api_key: null
    supports_stream: true
    max_concurrency: 256
```

请求体里的 `model` 必须是其中一个 key。对于 `launcher.py` 来说，`--llm-model dsv4pro` 必须匹配 `gateway.config.llm_routes.dsv4pro`。

如果 `AIEVOBOX_GATEWAY_CONFIG` 指向 gateway 配置文件，`launcher.py` 会在启动前校验 `--llm-model`。

## 存储必须与 Launcher 一致

使用 SQLite 时，gateway 和 launcher 必须使用同一个 DB：

```yaml
# gateway/config.local.yaml
storage_type: sqlite
storage_config:
  db_url: sqlite://env_trajs.db
```

```bash
python launcher.py \
  --db-path sqlite://env_trajs.db \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  ...
```

`launcher.py` 启动时会调用 `GET /readyz`，如果 gateway storage 与 launcher storage 不一致会提前失败，避免 evaluator 读到空轨迹或不完整轨迹。

使用 cloud storage 时，两个进程都应使用 `storage_type: cloud`，并依赖 `wt-data-gateway` 预期的默认 cloud 配置。

## Telemetry 与请求日志

Gateway telemetry 会写入 `session_steps`，并设置 `is_trainable = false`。这些行的 `messages` 中保存请求/响应 payload，`env_state` 中保存事件元数据。

相关配置：

```yaml
telemetry:
  mode: strict
  loss_policy: fail_closed
  capture_payload: full
  payload_sample_rate: 1.0
  redact_sensitive_fields: true
  batch_size: 200
  flush_interval_ms: 100

request_log:
  enabled: true
  path: logs/gateway_requests.jsonl
  max_bytes: 104857600
  backup_count: 5
  body_limit_bytes: 0
```

`telemetry.mode: strict` 表示 telemetry 写入失败时请求也 fail closed。`best_effort` 表示 telemetry 失败时请求仍可继续。

`request_log.body_limit_bytes: 0` 会在 JSONL 请求日志中记录完整请求和响应体。如需减少敏感信息暴露，可以设置正数限制或关闭 request log。

## Session 生命周期

1. Agent runtime 从 `launcher.py` 收到 `SimulationStartRequest`。
2. Runtime 调用 `POST /v1/sessions/{session_id}/chat/completions` 或 `/responses`。
3. Gateway 解析 session，尽量绑定到环境行，路由请求并写入 telemetry。
4. Rollout 结束后，`launcher.py` 调用 `POST /v1/sessions/{session_id}/close` 关闭 gateway session。
5. Gateway 将最新轨迹行标记完成，并写入 close event。

如果 `max_steps` 为非负数，模型在该 session 中达到步数上限后，gateway 会返回 synthetic stop。

## 常用检查

```bash
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:8000/metrics
curl http://127.0.0.1:8000/v1/sessions/<session-id>
```

常见启动失败：

| 现象 | 常见原因 |
|------|----------|
| `gateway is not reachable` | Gateway 没启动，或 `--gateway-base-url` host/port 写错。 |
| `gateway SQLite DB does not match launcher --db-path` | `storage_config.db_url` 与 `--db-path` 不一致。 |
| `gateway model route(s) are not configured` | `--llm-model` 不在 `llm_routes` 中。 |
| 上游 401 或 404 | route 的 `api_key`、`base_url` 或上游模型命名不正确。 |
