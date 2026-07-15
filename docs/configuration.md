# Configuration

Safactory v2 uses several explicit configuration surfaces:

1. `gateway` config for model routes, telemetry, request logs, and storage.
2. `launcher.py` CLI flags for scheduling, storage, model route selection, evaluation, and runtime mode.
3. Agent config YAML passed with `--agent-config`, or all configs under `--agent-root`.
4. Agent start config YAML passed with `--agent-start-config`.
5. Optional global RJob config passed with `--rjob-config`.
6. Optional global Sandbox config passed with `--sandbox-config`.
7. Optional evaluator runtime config passed with `--evaluation-config`.

`--mode` accepts `docker`, `rjob`, or `sandbox`. See [Sandbox mode](sandbox-mode.md) for the third runtime.

For a local SQLite run, gateway and launcher must share the same DB URI.

## Minimal Local Run

Start gateway:

```bash
python -m gateway --config gateway/config.local.yaml
```

Run one agent config:

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

## Essential CLI Flags

| Category | Flag | Default | Description |
|----------|------|---------|-------------|
| Job | `--job-id` | generated when empty | Identifier written to environment rows and trajectory rows. |
| Runtime | `--mode` | `docker` | Runtime allocator: `docker`, `rjob`, or `sandbox`. |
| Config | `--agent-config` | `None` | Path to one agent task YAML. |
| Config | `--agent-root` | `env` | Directory scanned for child YAML files when `--agent-config` is not set. Invalid YAMLs are skipped with warnings. |
| Config | `--agent-start-config` | `None` | YAML that defines how each agent runtime starts in Docker and optionally RJob. |
| Config | `--rjob-config` | `config.yaml` | Global RJob connection and auth config. |
| Config | `--sandbox-config` | `config.yaml` | Global OpenSandbox/Brainbox connection and Environment config. |
| Storage | `--storage-type` | `sqlite` | `sqlite` or `cloud`. |
| Storage | `--db-path` | `sqlite://env_trajs.db` for SQLite | SQLite DB URI. Ignored by cloud storage. |
| Gateway | `--gateway-base-url` | `http://127.0.0.1:8080/v1/sessions` | Gateway session root. Override this to match your gateway port. |
| LLM | `--llm-model` | `default` | Gateway route key used by agent rollouts. |
| LLM | `--llm-temperature` | `0.3` | Sampling temperature passed to agent runtimes. |
| Episode | `--max-steps` | `1000` | Maximum step budget passed to the runtime request. Gateway `max_steps` can enforce an additional request-level limit. |
| Pool | `--pool-size` | `1` | Base concurrency. Warm pool size is `ceil(pool_size * multiplier)`. |
| Pool | `--multiplier` | `1.2` | Warm-pool multiplier. |
| Pool | `--max-workers` | `0` | Worker count cap. `0` uses warm-pool size. |

## Full CLI Reference

| Category | Flag | Default | Description |
|----------|------|---------|-------------|
| Storage | `--rebuild-table` / `--no-rebuild-table` | `false` | For SQLite, delete the DB file before loading configs. |
| Storage | `--disable-buffer` | buffer enabled | Disable buffered writes. |
| Storage | `--buffer-size` | `100` | Write buffer capacity. |
| Storage | `--flush-interval` | `5.0` | Write buffer flush interval in seconds. |
| Docker | `--docker-bin` | `docker` | Docker executable. |
| Docker | `--docker-pull-policy` | `never` | `never` or `always`. |
| Docker | `--docker-startup-concurrency` | `8` | Max concurrent Docker startup operations. |
| Docker | `--cleanup-docker-container` / `--no-cleanup-docker-container` | `true` | Remove rollout containers after completion. |
| Docker | `--cleanup-stale-docker-containers` / `--no-cleanup-stale-docker-containers` | `true` | Remove stale Safactory containers for the same job at startup. |
| Timeout | `--agent-start-timeout-s` | `600.0` | Inner agent runtime timeout. |
| Timeout | `--agent-start-timeout-grace-s` | `120.0` | Extra outer timeout budget. |
| Timeout | `--container-refill-timeout-s` | `300.0` | Max time to release and replace one runtime resource. |
| Timeout | `--row-wait-timeout-s` | `60.0` | Max time to wait for new DB rows while refilling. |
| Timeout | `--row-fetch-timeout-s` | `30.0` | Max time for one scheduler DB fetch. |
| Timeout | `--gateway-close-timeout-s` | `15.0` | HTTP timeout for gateway close calls. |
| Timeout | `--gateway-close-retries` | `1` | Retry count for gateway close calls. |
| Timeout | `--gateway-close-retry-backoff-s` | `1.0` | Backoff between gateway close retries. |
| Timeout | `--shutdown-timeout-s` | `120.0` | Max launcher shutdown time. |
| Docker timeout | `--docker-command-timeout-s` | `300.0` | Default Docker lifecycle command timeout. |
| Docker timeout | `--docker-start-timeout-s` | `300.0` | Docker run/copy startup timeout. |
| Docker timeout | `--docker-remove-timeout-s` | `120.0` | Docker remove timeout. |
| Docker timeout | `--docker-stop-timeout-s` | `10.0` | Docker stop grace period. |
| Docker timeout | `--docker-inspect-timeout-s` | `10.0` | Docker inspect timeout. |
| Docker timeout | `--docker-remove-retries` | `3` | Container removal retries. |
| Docker timeout | `--docker-remove-retry-delay-s` | `2.0` | Delay between removal retries. |
| Docker timeout | `--docker-lifecycle-timeout-s` | `60.0` | Optional per-container cleanup and healthcheck timeout. |
| Evaluation | `--enable-evaluation` | `false` | Run evaluator flow after rollout. |
| Evaluation | `--evaluation-config` | empty | Evaluator runtime YAML. |
| Evaluation | `--evaluation-model` | empty | Gateway route key used by evaluator model calls. |
| Evaluation | `--strict-eval-tasks` | `false` | Fail when expected markdown eval task is missing. |
| RL | `--rl-group-size` | `0` | Override each YAML environment group's `env_num`. |
| RL | `--rl-epoch` | `1` | Duplicate environment configs for multiple rollout epochs. |
| Circuit breaker | `--circuit-breaker` / `--no-circuit-breaker` | `true` | Stop scheduling when recent failures/timeouts exceed thresholds. |
| Circuit breaker | `--circuit-breaker-window` | `50` | Sliding window size. |
| Circuit breaker | `--circuit-breaker-min-samples` | `20` | Minimum samples before opening. |
| Circuit breaker | `--circuit-breaker-failure-rate` | `0.8` | Failure-rate threshold. |
| Circuit breaker | `--circuit-breaker-timeout-rate` | `0.5` | Timeout-rate threshold. |
| Circuit breaker | `--circuit-breaker-consecutive-timeouts` | `5` | Consecutive timeout threshold. |
| Logging | `--log-dir` | `logs` | Run log root. |
| Logging | `--run-name` | empty | Optional run directory prefix. |
| Logging | `--console-log-level` | `INFO` | Console log level. |
| Logging | `--file-log-level` | `DEBUG` | File log level. |
| Logging | `--log-backup-count` | `20` | Recent log directories to keep. |
| Logging | `--debug-log` | `false` | Enable selected debug storage loggers. |

## Gateway Config

See [Gateway](gateway.md) for full details. The fields most often changed are:

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

Agent config YAML defines task rows. Each row expands into one or more `job_environments` entries.

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

| Field | Required | Description |
|-------|----------|-------------|
| `env_name` | yes | Agent/runtime name. Must match `agent_name` in the start config. |
| `env_image` | no | Runtime image or image ID used by Docker/RJob allocation. |
| `env_num` | no | Number of parallel copies for each dataset row. Must be a positive integer. |
| `dataset` | no | Path to JSON, JSONL, YAML, or parquet data. Relative paths resolve from the config file directory. |
| `dataset_load_mode` | no | `eager` by default. `parquet_row_ref` stores lightweight row references for parquet files. |
| `env_params` | no | Public parameters passed to the runtime in `SimulationStartRequest.env_params`. |

For each dataset item, Safactory sets:

- `env_params.dataset` to the dataset row.
- internal metadata with config path, dataset path, dataset name, and load mode.
- a deterministic `group_id` based on `env_name` and task index.

## Agent Start Config YAML

Agent start config defines how to start the runtime for an `env_name`.

Single-agent form:

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

Multi-agent form:

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

Docker container fields:

| Field | Description |
|-------|-------------|
| `workdir` | Working directory for `docker exec`. |
| `runner_entrypoint` | Runner entrypoint for one episode. `source` is resolved relative to the start config file, mounted into Docker or embedded into RJob when local, and `command` is executed for each episode. |
| `idle_command` | Command used to keep an allocated container alive. |
| `run_command` | Legacy command field. Prefer `runner_entrypoint.command`. |
| `result_mode` | `json` by default. `exit_code` treats a zero exit code as success. |
| `network`, `platform` | Optional Docker runtime settings. |
| `env` | Environment variables injected into the container. |
| `mounts` / `volumes` | Docker bind mounts. Relative `source` paths resolve from current working directory. |
| `extra_args` | Additional `docker run` args. |
| `install_runner_script` | Boolean compatibility flag. Defaults to `false`. |

The runtime receives a `SimulationStartRequest` as stdin and through `SAFACTORY_START_REQUEST_JSON`. It must print a JSON object compatible with:

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

## RJob Config

Global RJob settings live in `config.yaml` or the file passed with `--rjob-config`:

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

Per-agent RJob settings live in `--agent-start-config` under `rjob:`:

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

When `container.runner_entrypoint.source` points to a local file, RJob mode embeds that file automatically. Supported per-agent RJob keys include connection overrides, image pull policy, `resources`, `requests`, `env`, `labels`, `annotations`, `affinity`, `mount_config`, `mount`, `before_script`, `depends_on`, `embedded_files` for additional files, `replicas`, `poll_interval_s`, `termination_grace_period_seconds`, `local_storage_in_mb`, and cleanup flags.

## Useful Environment Variables

| Variable | Purpose |
|----------|---------|
| `AIEVOBOX_GATEWAY_CONFIG` | Optional path used by `launcher.py` to validate model route keys before startup. |
| `SAFACTORY_GATEWAY_LOG_PATH` | Gateway service log path. |
| `AIEVOBOX_SQLITE_BULK_INSERT_BATCH_SIZE` | Override SQLite background env-row insert batch size. |
| `AIEVOBOX_SQLITE_BULK_INSERT_PAUSE_S` | Pause between SQLite background insert batches. |

## Common Pitfalls

| Symptom | Fix |
|---------|-----|
| `unrecognized arguments: --env-config` | v2 uses `--agent-config`, not the old `--env-config`. |
| `unrecognized arguments: --llm-base-url` | Model endpoints live in gateway `llm_routes`; launcher uses `--llm-model` route keys. |
| Gateway and launcher start but evaluator sees no rows | Ensure gateway `storage_config.db_url` equals launcher `--db-path`. |
| Docker runtime cannot reach gateway on localhost | Add `--add-host=host.docker.internal:host-gateway` and use the helper-provided gateway session URL inside the runtime. |
