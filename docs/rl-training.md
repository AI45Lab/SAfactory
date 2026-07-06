# Reinforcement Learning Training

Safactory includes an RL bridge for Slime-style training:

- `rl/buffer_server.py` starts `launcher.py`, reads completed trainable rows, groups samples by `group_id`, and exposes batches through `/get_rollout_data`.
- `rl/llm_proxy.py` is hosted by `slime_generator` and exposes an OpenAI-like endpoint for online rollout generation.
- `rl/examples/*` contains task-specific scripts and `env.sh` files.

The RL bridge still uses the historical `AIEVOBOX_*` environment variable prefix. Some example scripts also contain older variable names, so review the variables below before running with the v2 launcher.

## Architecture

```text
Slime generator
  |
  | hosts rl/llm_proxy.py
  v
LLM proxy
  ^
  | rollout model requests
  |
Safactory launcher.py  <--- started by rl/buffer_server.py
  |
  | trainable rows in session_steps
  v
Buffer Server /get_rollout_data
  |
  | grouped rollout samples
  v
Slime trainer
```

For v2 adapter runs, the launcher still needs a valid gateway-compatible model path and runtime start config. Verify the generated launcher command in `logs/buffer_server.log` before scaling.

## Current Integration Notes

As of the current code:

- `buffer_server.py` maps `AIEVOBOX_AGENT_CONFIG` to `launcher.py --agent-config`.
- If `AIEVOBOX_AGENT_CONFIG` is unset, it maps `AIEVOBOX_AGENT_ROOT` to `--agent-root`.
- It maps `RL_MODEL` to `launcher.py --llm-model`.
- It maps `AIEVOBOX_GATEWAY_BASE_URL` to `launcher.py --gateway-base-url`; if unset, it falls back to the in-process LLM proxy URL.
- It does not currently pass `--agent-start-config`. For v2 Docker/RJob adapters, add that to your local launch wrapper or update `buffer_server.py` before production RL runs.

This means the checked-in RL examples should be treated as templates rather than turnkey v2 commands.

## Example Scripts

Scripts live under `rl/examples/`:

| Directory | Scripts |
|-----------|---------|
| `rl/examples/search` | `env.sh`, `run_buffer_server.sh`, `run_slime_generator.sh` |
| `rl/examples/deepeyes` | `env.sh`, `run_buffer_server.sh`, `run_slime_generator.sh` |
| `rl/examples/geo3k_vl` | `env.sh`, `run_buffer_server.sh`, `run_slime_generator.sh` |
| `rl/examples/math500` | `env.sh`, `run_buffer_server.sh`, `run_slime_generator_opd_sglang.sh` |

Run from an example directory:

```bash
cd rl/examples/math500
./run_buffer_server.sh
```

The Slime generator process should be started separately according to the example script and the Slime documentation.

## Key Variables

Launcher and data:

| Variable | Used as | Description |
|----------|---------|-------------|
| `AIEVOBOX_ROOT` | launcher path and Python path | Safactory repository root. |
| `STORAGE_TYPE` | `--storage-type` | `sqlite` or `cloud`. |
| `AIEVOBOX_DB_URL` | `--db-path` | SQLite URI used by launcher and Buffer Server. |
| `AIEVOBOX_AGENT_CONFIG` | `--agent-config` | Single v2 agent config YAML. |
| `AIEVOBOX_AGENT_ROOT` | `--agent-root` | Agent config root when `AIEVOBOX_AGENT_CONFIG` is unset. |
| `AIEVOBOX_GATEWAY_BASE_URL` | `--gateway-base-url` | Gateway session root. If unset, falls back to LLM proxy URL. |
| `RL_MODEL` | `--llm-model` | Gateway route key or rollout model identifier. |
| `LLM_TEMPERATURE` | `--llm-temperature` | Sampling temperature. |
| `AIEVOBOX_MAX_STEPS` | `--max-steps` | Episode step limit. |
| `AIEVOBOX_POOL_SIZE` | `--pool-size` | Launcher pool size. |

RL grouping:

| Variable | Description |
|----------|-------------|
| `RL_GROUP_SIZE` | Samples per prompt. Sent as `--rl-group-size` and used by Buffer Server grouping. |
| `RL_EPOCH` | Rollout epoch expansion. Sent as `--rl-epoch`. |
| `RL_OFF_BY_N` | Maximum allowed weight-version lag. Used by training-side scripts. |
| `SLIME_GLOBAL_BATCH_SIZE` | Slime global batch size. |
| `SLIME_N_SAMPLES_PER_PROMPT` | Usually set to `RL_GROUP_SIZE`. |

Services:

| Variable | Description |
|----------|-------------|
| `BUFFER_SERVER_HOST` | Host that Slime uses to reach Buffer Server. |
| `BUFFER_SERVER_PORT` | Buffer Server port, commonly `18889`. |
| `LLM_PROXY_HOST` | Host that rollout processes use to reach the Slime-hosted LLM proxy. |
| `LLM_PROXY_PORT` | LLM proxy port, commonly `18890`. |

## Buffer Server API

| Endpoint | Purpose |
|----------|---------|
| `POST /start_rollout` | Starts one launcher subprocess unless one is already running. |
| `POST /get_rollout_data` | Returns grouped rollout samples since the last served DB row. |
| `GET /health` | Diagnostic endpoint. In the current code it still references a legacy `llm_proxy_process` variable, so do not rely on it as a production readiness probe until that is fixed. |

`/get_rollout_data` returns items shaped for Slime:

```json
{
  "uid": "...",
  "instance_id": "<group_id>",
  "messages": [],
  "reward": 0.0,
  "extra_info": {
    "session_id": "...",
    "env_id": "...",
    "group_id": "...",
    "weight_version": 0,
    "truncated": false
  }
}
```

## Suggested V2 Workflow

1. Make the adapter run outside RL first with `launcher.py` and gateway.
2. Confirm trainable rows appear in `session_steps` with the desired `job_id`.
3. Set `AIEVOBOX_AGENT_CONFIG`, `AIEVOBOX_DB_URL`, `AIEVOBOX_GATEWAY_BASE_URL`, `RL_MODEL`, `RL_GROUP_SIZE`, and `AIEVOBOX_POOL_SIZE`.
4. Ensure your local Buffer Server wrapper passes `--agent-start-config` or uses a patched `buffer_server.py`.
5. Start Slime generator and Buffer Server.
6. Call `/start_rollout` from the trainer or script.
7. Watch `logs/buffer_server.log`, `logs/main.log`, gateway logs, and Slime logs.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Buffer Server starts launcher with old `--env-config` assumptions | Use `AIEVOBOX_AGENT_CONFIG` or `AIEVOBOX_AGENT_ROOT`; update old example `env.sh` files. |
| Launcher fails because start config is missing | Current `buffer_server.py` does not pass `--agent-start-config`; add it in your wrapper or patch the command builder. |
| No samples returned | Check `session_steps` for `is_trainable = 1`, correct `job_id`, and matching `group_id` counts. |
| Groups never become ready | `RL_GROUP_SIZE` is larger than the number of completed samples per `group_id`. |
| Weight-version parsing fails | Ensure `env_state.weight_version` is present when your training policy depends on it. |
