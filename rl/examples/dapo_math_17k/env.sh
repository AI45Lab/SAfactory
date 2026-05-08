# -------------------------------------------
# AIEvobox (rollout) Settings
# -------------------------------------------
export RAY_BIN="${RAY_BIN:-/mnt/shared-storage-user/evobox-share/yinzhenyun/slime-env-0.2.3/bin/ray}"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/evobox-share/yinzhenyun/slime-env-0.2.3/bin/python3.12}"
export SLIME_ENV_BIN="${SLIME_ENV_BIN:-/mnt/shared-storage-user/evobox-share/yinzhenyun/slime-env-0.2.3/bin}"
# Keep this env bin first so plain `python3`/`ray` resolve to py3.12 runtime.
case ":${PATH}:" in
  *":${SLIME_ENV_BIN}:"*) ;;
  *) export PATH="${SLIME_ENV_BIN}:${PATH}" ;;
esac

# Some environments only provide `python3.12` without `python3`.
# Create a lightweight shim so subprocess calls to `python3` use PYTHON_BIN.
export AIEVOBOX_SHIM_BIN="${AIEVOBOX_SHIM_BIN:-/tmp/aievobox-bin-shim}"
mkdir -p "${AIEVOBOX_SHIM_BIN}"
ln -sf "${PYTHON_BIN}" "${AIEVOBOX_SHIM_BIN}/python3"
ln -sf "${RAY_BIN}" "${AIEVOBOX_SHIM_BIN}/ray"
case ":${PATH}:" in
  *":${AIEVOBOX_SHIM_BIN}:"*) ;;
  *) export PATH="${AIEVOBOX_SHIM_BIN}:${PATH}" ;;
esac
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-/mnt/shared-storage-user/leishanzhe/repo/AIEvoBox}"
export STORAGE_TYPE="${STORAGE_TYPE:-sqlite}"
export AIEVOBOX_RUN_ID="${AIEVOBOX_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export AIEVOBOX_DB_URL="${AIEVOBOX_DB_URL:-sqlite:///${AIEVOBOX_ROOT}/rl/examples/dapo_math_17k/rl_${AIEVOBOX_RUN_ID}.db}"
export AIEVOBOX_MAX_STEPS="${AIEVOBOX_MAX_STEPS:-10}"
export AIEVOBOX_MESSAGE_CUT="${AIEVOBOX_MESSAGE_CUT:-0}"
export AIEVOBOX_ENV_CONFIG="${AIEVOBOX_ENV_CONFIG:-${AIEVOBOX_ROOT}/env/dapo_math_17k/dapo_math_17k_env_configs.yaml}"
export AIEVOBOX_POOL_SIZE="${AIEVOBOX_POOL_SIZE:-256}"

# -------------------------------------------
# RL Settings
# -------------------------------------------
export RL_GROUP_SIZE="${RL_GROUP_SIZE:-8}"
export RL_EPOCH="${RL_EPOCH:-1000}"
export RL_OFF_BY_N="${RL_OFF_BY_N:-1}"
# DAPO-style uniform-reward group filtering is useful for this dataset.
export DAPO_filter="${DAPO_filter:-true}"

# -------------------------------------------
# Buffer Server Settings
# -------------------------------------------
export BUFFER_SERVER_HOST="${BUFFER_SERVER_HOST:-127.0.0.1}"
export BUFFER_SERVER_PORT="${BUFFER_SERVER_PORT:-18889}"

# -------------------------------------------
# LLM Proxy Settings (hosted by Slime Generator)
# -------------------------------------------
export LLM_PROXY_HOST="${LLM_PROXY_HOST:-127.0.0.1}"
export LLM_PROXY_PORT="${LLM_PROXY_PORT:-18890}"
export LLM_MAX_LENGTH="${LLM_MAX_LENGTH:-1536}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-1.0}"

# -------------------------------------------
# OPD Teacher / RM Settings
# -------------------------------------------
export TEACHER_URL="${TEACHER_URL:-http://100.99.167.210:30172/generate}"

# -------------------------------------------
# Slime Training Settings
# -------------------------------------------
export SLIME_ROLLBUF_RESTART_TRAINING="${SLIME_ROLLBUF_RESTART_TRAINING:-True}"
export SLIME_N_SAMPLES_PER_PROMPT="${SLIME_N_SAMPLES_PER_PROMPT:-$RL_GROUP_SIZE}"
export SLIME_GLOBAL_BATCH_SIZE="${SLIME_GLOBAL_BATCH_SIZE:-512}"
export SLIME_ROLLOUT_BATCH_SIZE="${SLIME_ROLLOUT_BATCH_SIZE:-$((SLIME_GLOBAL_BATCH_SIZE / SLIME_N_SAMPLES_PER_PROMPT))}"
