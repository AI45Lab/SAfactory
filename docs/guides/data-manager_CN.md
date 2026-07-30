# 数据管理器

Safactory v2 通过 `core.data_manager` 记录任务行和 session 行。默认本地后端是 `sqlite://env_trajs.db`；cloud 模式交给 `wt-data-gateway` 默认配置。

使用 SQLite 时，这两处必须一致：

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

## 存储后端

| 后端 | 用途 | 说明 |
|------|------|------|
| `sqlite` | 本地开发、smoke test、单机评测。 | 自包含 DB 文件。Gateway 和 launcher 必须使用同一 URI。 |
| `cloud` | 集群/RJob 运行。 | `--db-path` 会被忽略，使用 `wt-data-gateway` 默认 DB URI 和表。 |

默认启用缓冲写入。调试写入顺序时可用 `--disable-buffer` 关闭。

## 表结构

### `job_environments`

每个被调度的环境实例一行。

| 字段 | 含义 |
|------|------|
| `id` | 自增主键。 |
| `job_id` | Launcher run 标识。 |
| `env_id` | Session/environment UUID。对应 `session_steps.session_id`。 |
| `env_name` | Adapter 名，例如 `openclaw` 或 `openrt`。 |
| `env_params` | 展开后的公开 runtime 参数，包含 `dataset`。 |
| `image` | 来自 `env_image` 的 runtime 镜像。 |
| `group_id` | RL grouping 使用的确定性 group 标识。 |
| `finished` | Scheduler 是否标记该行完成。 |
| `is_deleted` | YAML config 变化后的软删除标记。 |
| `created_at` | 创建时间。 |

### `session_steps`

每个 gateway telemetry event、可训练轨迹 step、close event 或 evaluation summary 一行。

| 字段 | 含义 |
|------|------|
| `id` | 自增主键。 |
| `session_id` | Session UUID。匹配 `job_environments.env_id`。 |
| `step_id` | Session 内 step 索引。Gateway telemetry 使用请求序列号。 |
| `env_name` | Adapter 名。Gateway 解析环境行后可能 patch 该字段。 |
| `llm_model` | 与该行关联的 gateway route key 或 model name。 |
| `group_id` | RL group 标识。 |
| `job_id` | Launcher run 标识。 |
| `messages` | JSON 序列化的 OpenAI 风格 messages。Gateway 会尽量追加 assistant output。 |
| `response` | 直接 runtime 行的原始响应/action。Gateway telemetry 通常将响应存在 `messages`，这里为空。 |
| `step_reward` | 单行奖励。Evaluation commit 会把最终分数写到这里。 |
| `reward` | 累计奖励。Evaluation commit 也会写入最终分数。 |
| `env_state` | JSON 元数据。Gateway/evaluation 事件中包含 `event_type`。 |
| `is_terminal` | 该行是否终止 session。 |
| `is_truncated` | 是否因截断终止。 |
| `is_session_completed` | Session 是否对 reader/evaluator sealed。 |
| `is_trainable` | RL Buffer Server 是否读取该行。 |
| `created_at` | 创建时间。 |

## 行类型

通常通过 `env_state.event_type` 和 `is_trainable` 判断行类型。

| 类型 | 标记 | 可训练 | 产生方 |
|------|------|--------|--------|
| Gateway inference | `event_type = gateway_inference` | 通常为 `false`；reward commit 可能将有效轨迹行标成 trainable。 | `gateway.storage` telemetry。 |
| Gateway close | `event_type = gateway_session_close` | `false` | Gateway close telemetry。 |
| Evaluation summary | `event_type = evaluation_summary` | `false` | 没有 trainable row 时由 `RewardCommitter` 写入。 |
| Runtime/direct step | 没有特殊 event type | 记录为 trainable 时为 `true` | 自定义 runtime 或 data manager 调用方。 |

Evaluator 和 RL Buffer Server 会有意忽略非 trainable event 行。

## Dataset 展开

`core.data_manager.load_yaml` 会按以下方式展开 agent config：

1. 读取 `environments`。
2. 读取可选 dataset 文件。
3. 对每条 dataset row 复制 `env_params`。
4. 将 `env_params.dataset` 设置为该 row。
5. 注入 config path、dataset path、dataset name 和 load mode 等内部元数据。
6. 为该 dataset item 创建 `env_num` 行。

支持的数据格式包括 JSON 数组、JSONL、YAML list 和 parquet。相对 dataset 路径从 agent config 所在目录解析。

## 查询示例

列出最近环境行：

```bash
sqlite3 env_trajs.db "
  SELECT id, job_id, env_id, env_name, group_id, finished, created_at
  FROM job_environments
  ORDER BY id DESC
  LIMIT 20;"
```

列出最近 session 行：

```bash
sqlite3 env_trajs.db "
  SELECT id, session_id, step_id, env_name, llm_model, step_reward, reward,
         is_terminal, is_session_completed, is_trainable, created_at
  FROM session_steps
  ORDER BY id DESC
  LIMIT 20;"
```

查看 gateway event type：

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

查看单个 session：

```bash
sqlite3 env_trajs.db "
  SELECT id, step_id, step_reward, reward, is_terminal,
         is_session_completed, is_trainable, created_at
  FROM session_steps
  WHERE session_id = '<session-id>'
  ORDER BY step_id, id;"
```

找出 RL Buffer Server 会消费的行：

```bash
sqlite3 env_trajs.db "
  SELECT id, session_id, step_id, env_name, group_id, step_reward, is_truncated
  FROM session_steps
  WHERE job_id = '<job-id>' AND is_trainable = 1
  ORDER BY id
  LIMIT 50;"
```

汇总最终分数：

```bash
sqlite3 env_trajs.db "
  SELECT env_name, COUNT(*) AS completed_rows, AVG(step_reward) AS avg_score
  FROM session_steps
  WHERE is_session_completed = 1 AND is_trainable = 1
  GROUP BY env_name;"
```

## 训练数据用途

`rl/buffer_server.py` 通过 `fetch_done_steps_with_context` 读取完成行：

| Buffer 字段 | 来源 |
|-------------|------|
| `messages` | `session_steps.messages` 加 assistant `response`。 |
| `reward` | `session_steps.step_reward`。 |
| `instance_id` | `session_steps.group_id`。 |
| `extra_info.session_id` | `session_steps.session_id`。 |
| `extra_info.weight_version` | 存在时从 `env_state.weight_version` 解析。 |
| `extra_info.truncated` | `session_steps.is_truncated`。 |

行会按 `group_id` 聚合。设置 `--rl-group-size` 或 `RL_GROUP_SIZE`，确保每个 prompt group 有预期数量的样本。

## 维护

SQLite strategy 会创建运行时索引：

- `idx_job_environments_job_deleted_id` on `(job_id, is_deleted, id)`。
- `idx_session_steps_job_trainable_id` on `(job_id, is_trainable, id)`。

`--rebuild-table` 只建议用于可丢弃的本地运行；它会在加载配置前删除 SQLite DB 文件。
