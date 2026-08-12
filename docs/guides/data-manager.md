# Data Manager

Safactory records task rows and session rows through `core.data_manager`. The default local backend is SQLite at `sqlite://env_trajs.db`; cloud mode delegates to `wt-data-gateway` defaults.

For SQLite, keep these values identical:

```yaml
# gateway config
storage_type: sqlite
storage_config:
  db_url: sqlite://env_trajs.db
```

```bash
# launcher.py
--storage-type sqlite --db-path sqlite://env_trajs.db
```

## Storage Backends

| Backend | Use | Notes |
|---------|-----|-------|
| `sqlite` | Local development, smoke tests, single-machine evaluation. | Self-contained DB file. Gateway and launcher must share the same URI. |
| `cloud` | Cluster/RJob runs. | `--db-path` is ignored. Uses `wt-data-gateway` default database URIs and tables. |

Buffered writes are enabled by default. Disable with `--disable-buffer` when debugging write ordering.

## Tables

### `job_environments`

One row per scheduled environment instance.

| Field | Meaning |
|-------|---------|
| `id` | Auto-increment primary key. |
| `job_id` | Launcher run identifier. |
| `env_id` | Session/environment UUID. This becomes `session_steps.session_id`. |
| `env_name` | Adapter name, for example `openclaw` or `openrt`. |
| `env_params` | Expanded public runtime parameters, including `dataset`. |
| `image` | Runtime image from `env_image`. |
| `group_id` | Deterministic group identifier used by RL grouping. |
| `finished` | Whether scheduling marked the row finished. |
| `is_deleted` | Soft-delete marker when YAML config changes. |
| `created_at` | Row creation timestamp. |

### `session_steps`

One row per gateway telemetry event, trainable trajectory step, close event, or evaluation summary.

| Field | Meaning |
|-------|---------|
| `id` | Auto-increment primary key. |
| `session_id` | Session UUID. Matches `job_environments.env_id`. |
| `step_id` | Step index inside the session. Gateway telemetry uses request sequence IDs. |
| `env_name` | Adapter name. Gateway may patch it after resolving the environment row. |
| `llm_model` | Gateway route key or model name associated with the row. |
| `group_id` | RL group identifier. |
| `job_id` | Launcher run identifier. |
| `messages` | JSON-serialized OpenAI-style messages. Gateway appends assistant output when possible. |
| `response` | Raw response/action for direct runtime rows. Gateway telemetry stores response in `messages` and leaves this empty. |
| `step_reward` | Per-row reward. Evaluation commit writes final score here. |
| `reward` | Cumulative reward. Evaluation commit also writes final score here. |
| `env_state` | JSON metadata. Contains `event_type` for gateway/evaluation events. |
| `is_terminal` | Whether this row terminates the session. |
| `is_truncated` | Whether termination was caused by truncation. |
| `is_session_completed` | Whether the session is sealed for readers/evaluators. |
| `is_trainable` | Whether RL Buffer Server should read this row. |
| `created_at` | Row creation timestamp. |

## Row Types

The row type is usually determined by `env_state.event_type` and `is_trainable`.

| Type | Marker | Trainable | Produced by |
|------|--------|-----------|-------------|
| Gateway inference | `event_type = gateway_inference` | Usually `false`; may be marked trainable by reward commit when it is a valid trajectory row. | `gateway.storage` telemetry. |
| Gateway close | `event_type = gateway_session_close` | `false` | Gateway close telemetry. |
| Evaluation summary | `event_type = evaluation_summary` | `false` | `RewardCommitter` when no trainable row exists. |
| Runtime/direct step | no special event type | `true` when recorded as trainable | Custom runtimes or data manager callers. |

The evaluator and RL Buffer Server intentionally ignore non-trainable event rows.

## Dataset Expansion

`core.data_manager.load_yaml` expands each agent config as follows:

1. Load `environments`.
2. Load the optional dataset file.
3. For each dataset row, copy `env_params`.
4. Set `env_params.dataset` to that row.
5. Inject internal metadata for config path, dataset path, dataset name, and load mode.
6. Create `env_num` rows for that dataset item.

Supported dataset formats are JSON arrays, JSONL, YAML lists, and parquet. Relative dataset paths resolve from the agent config directory.

## Query Examples

List recent environment rows:

```bash
sqlite3 env_trajs.db "
  SELECT id, job_id, env_id, env_name, group_id, finished, created_at
  FROM job_environments
  ORDER BY id DESC
  LIMIT 20;"
```

List recent session rows:

```bash
sqlite3 env_trajs.db "
  SELECT id, session_id, step_id, env_name, llm_model, step_reward, reward,
         is_terminal, is_session_completed, is_trainable, created_at
  FROM session_steps
  ORDER BY id DESC
  LIMIT 20;"
```

Inspect gateway event types:

```bash
sqlite3 env_trajs.db "
  SELECT id, session_id, step_id,
         json_extract(env_state, '$.event_type') AS event_type,
         json_extract(env_state, '$.status_code') AS status_code,
         json_extract(env_state, '$.total_latency_ms') AS total_latency_ms
  FROM session_steps
  WHERE env_state IS NOT NULL
  ORDER BY id DESC
  LIMIT 20;"
```

Inspect one session:

```bash
sqlite3 env_trajs.db "
  SELECT id, step_id, step_reward, reward, is_terminal,
         is_session_completed, is_trainable, created_at
  FROM session_steps
  WHERE session_id = '<session-id>'
  ORDER BY step_id, id;"
```

Find rows consumed by the RL Buffer Server:

```bash
sqlite3 env_trajs.db "
  SELECT id, session_id, step_id, env_name, group_id, step_reward, is_truncated
  FROM session_steps
  WHERE job_id = '<job-id>' AND is_trainable = 1
  ORDER BY id
  LIMIT 50;"
```

Summarize final scores:

```bash
sqlite3 env_trajs.db "
  SELECT env_name, COUNT(*) AS completed_rows, AVG(step_reward) AS avg_score
  FROM session_steps
  WHERE is_session_completed = 1 AND is_trainable = 1
  GROUP BY env_name;"
```

## Training Data Use

`rl/buffer_server.py` reads completed rows through `fetch_done_steps_with_context`:

| Buffer field | Source |
|--------------|--------|
| `messages` | `session_steps.messages` plus assistant `response`. |
| `reward` | `session_steps.step_reward`. |
| `instance_id` | `session_steps.group_id`. |
| `extra_info.session_id` | `session_steps.session_id`. |
| `extra_info.weight_version` | Parsed from `env_state.weight_version` when present. |
| `extra_info.truncated` | `session_steps.is_truncated`. |

Rows are grouped by `group_id`. Set `--rl-group-size` or `RL_GROUP_SIZE` so each prompt group has the expected number of samples.

## Maintenance

SQLite strategy creates runtime indexes:

- `idx_job_environments_job_deleted_id` on `(job_id, is_deleted, id)`.
- `idx_session_steps_job_trainable_id` on `(job_id, is_trainable, id)`.

An existing `job_id` requires either `--resume` or `--rebuild-table`. Resume skips completed environments; rebuild deletes only the current job's configs and trajectories. The options are mutually exclusive.
