# GRPO Training Workflow

Use this reference when the user asks to start GRPO, RL, or Slime-style training for an SAfactory environment.

Canonical docs:

- `README.md`
- `README_CN.md`
- `docs/guides/rl-training.md`
- `docs/guides/rl-training_CN.md`
- `docs/guides/evaluation.md`
- `docs/guides/evaluation_CN.md`
- `docs/reference/gateway.md`
- `docs/reference/gateway_CN.md`

The RL guide uses generic `my_env`. Geo3K is only the standard example in the root README.

## Prerequisite

Do not jump straight to training for a new or unknown environment. First confirm a minimal Docker evaluation works with:

- `env/<env>/<env>_config.yaml`
- `env/<env>/<env>_start.yaml`
- Gateway session root
- model route key
- evaluator, when reward is required
- shared storage for Launcher, Gateway, and Buffer Server

If the user asks to start training and no evaluation has passed, run or request a minimal Docker evaluation first.

## Create Or Check `env.sh`

For environment `<env>`, create or inspect:

```text
rl/examples/<env>/env.sh
```

The file should set the variables described in `docs/guides/rl-training*.md`, especially:

- `AIEVOBOX_ROOT`
- `AIEVOBOX_AGENT_CONFIG`
- `AIEVOBOX_AGENT_START_CONFIG`
- `AIEVOBOX_MODE=docker`
- `AIEVOBOX_ENABLE_EVALUATION=1`
- `AIEVOBOX_DB_URL`
- `AIEVOBOX_GATEWAY_BASE_URL`
- `RL_MODEL`
- rollout group size, batch size, and concurrency variables used by the local scripts

Use absolute paths for repo-root-derived config values when the guide recommends them. Keep first runs small.

## Start Commands

The root README is the authority for script execution. Start from the repository root.

Terminal 1:

```bash
RL_ENV_SH=rl/examples/<env>/env.sh bash rl/run_slime_generator.sh
```

Terminal 2:

```bash
RL_ENV_SH=rl/examples/<env>/env.sh bash rl/run_buffer_server.sh
```

The scripts also support `--env` form according to the RL guide, but prefer the root README shape unless the user asks otherwise.

## Gateway Autostart

Buffer Server can auto-start Gateway by default:

- It generates `logs/gateway.rl.generated.yaml`.
- It routes `RL_MODEL` to the Slime-hosted LLM proxy.
- It starts Docker rollout collection.
- It serves completed groups through `/get_rollout_data`.

Use `AIEVOBOX_GATEWAY_AUTOSTART=0` only when an external Gateway is already running and all of these are true:

- `AIEVOBOX_GATEWAY_BASE_URL` points to that Gateway session root.
- Gateway `llm_routes` contains the route key named by `RL_MODEL`.
- Gateway storage matches `AIEVOBOX_DB_URL`.

## Run Checks

After launch, inspect:

- `logs/buffer_server.log`
- `logs/main.log`
- Gateway logs
- Slime logs
- generated Gateway config when autostart is enabled

Important mappings from Buffer Server to Launcher:

- `AIEVOBOX_AGENT_CONFIG` -> `launcher.py --agent-config`
- `AIEVOBOX_AGENT_START_CONFIG` -> `launcher.py --agent-start-config`
- `AIEVOBOX_DB_URL` -> `launcher.py --db-path`
- `AIEVOBOX_GATEWAY_BASE_URL` -> `launcher.py --gateway-base-url`
- `RL_MODEL` -> `launcher.py --llm-model`

## Completion Criteria

Training startup is ready when:

- Docker evaluation has passed or the blocker is explicitly documented.
- `rl/examples/<env>/env.sh` exists and points at the requested environment.
- Model route, Gateway mode, and DB path are consistent.
- The generator and Buffer Server commands are either running or provided exactly for the user to run.
- Logs to watch and expected failure points are reported.
