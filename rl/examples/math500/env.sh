# -------------------------------------------
# AIEvobox (rollout) Settings
# -------------------------------------------
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-/mnt/shared-storage-user/leishanzhe/repo/AIEvoBox}"
export STORAGE_TYPE="${STORAGE_TYPE:-sqlite}"
export AIEVOBOX_DB_URL="${AIEVOBOX_DB_URL:-sqlite:///${AIEVOBOX_ROOT}/rl/examples/math500/rl.db}"
export AIEVOBOX_MAX_STEPS="${AIEVOBOX_MAX_STEPS:-1}"
export AIEVOBOX_MESSAGE_CUT="${AIEVOBOX_MESSAGE_CUT:-0}"
export AIEVOBOX_ENV_CONFIG="${AIEVOBOX_ENV_CONFIG:-${AIEVOBOX_ROOT}/env/math500_text/math500_text_env_configs.yaml}"
export AIEVOBOX_POOL_SIZE="${AIEVOBOX_POOL_SIZE:-256}"

# -------------------------------------------
# RL Settings
# -------------------------------------------
export RL_GROUP_SIZE="${RL_GROUP_SIZE:-16}"
export RL_EPOCH="${RL_EPOCH:-1}"
export RL_OFF_BY_N="${RL_OFF_BY_N:-0}"

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
export LLM_MAX_LENGTH="${LLM_MAX_LENGTH:-4096}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-1.0}"

# -------------------------------------------
# OPD Teacher / RM Settings
# -------------------------------------------
export TEACHER_URL="${TEACHER_URL:-http://100.99.167.229:30000/generate}"

# -------------------------------------------
# Slime Training Settings
# -------------------------------------------
export SLIME_ROLLBUF_RESTART_TRAINING="${SLIME_ROLLBUF_RESTART_TRAINING:-True}"
export SLIME_N_SAMPLES_PER_PROMPT="${SLIME_N_SAMPLES_PER_PROMPT:-$RL_GROUP_SIZE}"
export SLIME_GLOBAL_BATCH_SIZE="${SLIME_GLOBAL_BATCH_SIZE:-1024}"
export SLIME_ROLLOUT_BATCH_SIZE="${SLIME_ROLLOUT_BATCH_SIZE:-$((SLIME_GLOBAL_BATCH_SIZE / SLIME_N_SAMPLES_PER_PROMPT))}"
