# Data Manager

Safactory records environment configs and step-level interactions during every run. The default CLI database URI is `sqlite://env_trajs.db`; use `--db-path` to write elsewhere.

```bash
python launcher.py \
  --env-config env/osgym/os_config.yaml \
  --db-path sqlite://runs/os_eval.db \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL
```

Buffered writes are enabled by default. Use `--disable-buffer` only when you need immediate synchronous writes for debugging.

## Tables

Two tables capture each run:

| Table | Purpose |
|-------|---------|
| `job_environments` | One row per environment instance. Stores job ID, environment name, config parameters, group ID, finish state, and timestamps. |
| `session_steps` | One row per agent step. Stores messages, model response, reward, environment state, terminal flags, and timestamps. |

Core relationship:

```text
job_environments.id  ->  session_steps.session_id
```

Important `session_steps` fields:

| Field | Description |
|-------|-------------|
| `step_id` | Step index within the session. |
| `messages` | Full or truncated OpenAI-style chat history sent to the model. |
| `response` | Raw model output. |
| `step_reward` | Per-step reward signal. |
| `reward` | Cumulative reward. |
| `env_state` | Serialized environment state or observation metadata. |
| `is_terminal` | Whether this row ended the episode. |
| `is_session_completed` | Whether the session completed successfully. |

## Query Examples

List recent environment sessions:

```bash
sqlite3 env_trajs.db "
  SELECT id, env_name, group_id, finished, created_at
  FROM job_environments
  ORDER BY created_at DESC
  LIMIT 20;"
```

Inspect a session:

```bash
sqlite3 env_trajs.db "
  SELECT step_id, step_reward, reward, is_terminal, created_at
  FROM session_steps
  WHERE session_id = '<session-id>'
  ORDER BY step_id;"
```

Summarize completed rewards:

```bash
sqlite3 env_trajs.db "
  SELECT env_name, COUNT(*) AS episodes, AVG(reward) AS avg_reward, MAX(reward) AS best_reward
  FROM session_steps
  WHERE is_session_completed = 1
  GROUP BY env_name;"
```

## Training Data Use

Recorded trajectories can be converted into SFT or RL data:

- `messages` stores the conversation context.
- `response` stores the model action or answer.
- `step_reward` and `reward` provide reward signals.
- `env_state` preserves environment-side context for debugging and filtering.

For online RL, use the Buffer Server described in [RL Training](../rl/README.md).
