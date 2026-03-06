# Data Manager

All agent interactions are automatically persisted to SQLite during every run. No additional configuration is required — recording starts when `launcher.py` starts and captures every step.

---

## Database Location

The default database file is `test_envs.db` in the project root. Override with `--db-path`:

```bash
python launcher.py --db-path /path/to/my_runs.db ...
```

---

## Schema

Two tables capture every episode end-to-end:

```
job_environments                    session_steps
────────────────────────────        ──────────────────────────────────────
id                                  id
job_id                              session_id  ──────────► job_environments.id
env_id                              step_id
env_name                            env_name
env_params                          llm_model
image                               group_id
group_id                            job_id
finished                            messages          (full conversation history)
is_deleted                          response          (model output)
created_at                          step_reward       (per-step reward signal)
                                    reward            (cumulative reward)
                                    env_state         (serialised environment state)
                                    is_terminal
                                    is_session_completed
                                    created_at
```

- **`job_environments`** — one row per environment instance per run. Records the configuration used (Docker image, env params, group ID) and whether the episode finished successfully.
- **`session_steps`** — one row per agent step. Stores the full message history, model response, per-step and cumulative reward, and the environment state at each step.

---

## Querying the Database

Use the standard `sqlite3` CLI:

```bash
# List recent runs
sqlite3 test_envs.db "
  SELECT id, env_name, group_id, finished, created_at
  FROM job_environments
  ORDER BY created_at DESC
  LIMIT 20;"

# Inspect all steps for a specific session
sqlite3 test_envs.db "
  SELECT step_id, step_reward, reward, is_terminal, created_at
  FROM session_steps
  WHERE session_id = '<your-session-id>'
  ORDER BY step_id;"

# Summarise reward by environment across all runs
sqlite3 test_envs.db "
  SELECT env_name, COUNT(*) AS episodes, AVG(reward) AS avg_reward, MAX(reward) AS best_reward
  FROM session_steps
  WHERE is_session_completed = 1
  GROUP BY env_name;"
```

---

## Using Data for Training

Every recorded trajectory is immediately usable as SFT or RL training data:

- The `messages` column stores the full conversation history in OpenAI chat format.
- `step_reward` provides the per-step reward signal needed for GRPO/PPO training.
- `response` stores the raw model output at each step.

For integration with the Slime RL training loop, see the [RL Training](rl-training.md) guide.
