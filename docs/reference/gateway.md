# Gateway

Safactory v2 routes all model traffic through `gateway`. The gateway exposes OpenAI-compatible endpoints, maps model names to configured upstream routes, records telemetry into the same storage backend used by `launcher.py`, and enforces per-session limits such as `max_steps`.

## Start The Gateway

Create a local config first:

```bash
cp gateway/config.example.yaml gateway/config.local.yaml
```

Use your own route keys and upstream model endpoints:

```yaml
listen_host: 0.0.0.0
listen_port: 8000
base_session_path: /v1/sessions
max_steps: -1

storage_type: sqlite
storage_config:
  db_url: sqlite://env_trajs.db

llm_routes:
  geo3k_model:
    base_url: http://YOUR_LLM_HOST/v1
    api_key: YOUR_API_KEY
    supports_stream: true
    max_concurrency: 64
```

Run it:

```bash
python -m gateway --config gateway/config.local.yaml
```

The gateway writes its service log to `logs/gateway.log` by default. Override that path with `SAFACTORY_GATEWAY_LOG_PATH`.

## Endpoints

With the default `base_session_path: /v1/sessions`, the gateway exposes:

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` | Liveness probe. |
| `GET /readyz` | Readiness probe. Includes `storage_type`, `storage_config`, and `max_steps`. |
| `GET /metrics` | Prometheus-style metrics for admission, telemetry, and route inflight counts. |
| `POST /v1/chat/completions` | Standard OpenAI-compatible chat completions passthrough without session binding. |
| `POST /v1/responses` | Standard OpenAI-compatible responses passthrough without session binding. |
| `POST /v1/sessions/{session_id}/chat/completions` | Session-scoped chat completions. This is the main endpoint used by agent runtimes. |
| `POST /v1/sessions/{session_id}/responses` | Session-scoped responses endpoint. |
| `GET /v1/sessions/{session_id}` | Gateway session status. |
| `POST /v1/sessions/{session_id}/close` | Soft-close a session and flush close telemetry. |

Session-scoped requests are what make trajectory rows attach to a Safactory `session_id`.

## Route Keys

`llm_routes` is a map from route key to upstream endpoint:

```yaml
llm_routes:
  dsv4pro:
    base_url: http://10.0.0.10:8182/v1
    api_key: null
    supports_stream: true
    max_concurrency: 256
```

The request body `model` must be one of these keys. For `launcher.py`, this means `--llm-model dsv4pro` must match `gateway.config.llm_routes.dsv4pro`.

If `AIEVOBOX_GATEWAY_CONFIG` points to the gateway config file, `launcher.py` validates `--llm-model` before starting.

## Storage Must Match Launcher

For SQLite, gateway and launcher must use the same DB:

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
  --llm-model geo3k_model \
  ...
```

At startup, `launcher.py` calls `GET /readyz` and fails early if gateway storage does not match the launcher storage settings. This prevents evaluators from reading an empty or partial trajectory DB.

For cloud storage, start both processes with `storage_type: cloud` and the cloud storage defaults expected by `wt-data-gateway`.

## Telemetry And Request Logs

Gateway telemetry writes rows into `session_steps` with `is_trainable = false`. These rows contain request and response payloads in `messages`, plus event metadata in `env_state`.

Relevant config:

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

`telemetry.mode: strict` makes request handling fail closed when telemetry cannot be written. `best_effort` allows requests to proceed when telemetry fails.

`request_log.body_limit_bytes: 0` records full request and response bodies in the JSONL request log. Increase privacy by setting a positive byte limit or disabling request logs.

## Session Lifecycle

1. An agent runtime receives a `SimulationStartRequest` from `launcher.py`.
2. The runtime calls `POST /v1/sessions/{session_id}/chat/completions` or `/responses`.
3. Gateway resolves the session, binds it to an environment row if available, routes the request, and enqueues telemetry.
4. `launcher.py` closes the gateway session after rollout with `POST /v1/sessions/{session_id}/close`.
5. Gateway marks the latest rows complete and writes a close event.

If `max_steps` is non-negative, the gateway emits a synthetic stop response once a model reaches the configured step budget for that session.

## Useful Checks

```bash
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:8000/metrics
curl http://127.0.0.1:8000/v1/sessions/<session-id>
```

Common startup failures:

| Symptom | Likely cause |
|---------|--------------|
| `gateway is not reachable` | Gateway process is not running or `--gateway-base-url` points to the wrong host or port. |
| `gateway SQLite DB does not match launcher --db-path` | `storage_config.db_url` and `--db-path` differ. |
| `gateway model route(s) are not configured` | `--llm-model` is not present in `llm_routes`. |
| Upstream 401 or 404 | Route `api_key`, `base_url`, or upstream model naming is wrong. |
