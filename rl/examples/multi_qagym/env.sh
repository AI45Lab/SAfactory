# -------------------------------------------
# Multi-QAGym shared rollout settings
# -------------------------------------------
_MULTI_QAGYM_ENV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
_MULTI_QAGYM_DEFAULT_AIEVOBOX_ROOT="$(cd -- "${_MULTI_QAGYM_ENV_SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

export AIEVOBOX_ROOT="${_MULTI_QAGYM_DEFAULT_AIEVOBOX_ROOT}"
export STORAGE_TYPE="sqlite"
export AIEVOBOX_RUN_ID="multi_qagym_4gpu"
export AIEVOBOX_DB_URL="sqlite:///${AIEVOBOX_ROOT}/rl/examples/multi_qagym/multi_qagym_${AIEVOBOX_RUN_ID}.db"
export AIEVOBOX_ENV_CONFIG="${AIEVOBOX_ROOT}/env/multi_qagym/multi_qagym_env.yaml"
export AIEVOBOX_MAX_STEPS="6"
export AIEVOBOX_MESSAGE_CUT="0"
export AIEVOBOX_POOL_SIZE="8"

# -------------------------------------------
# Runtime
# -------------------------------------------
export RAY_BIN="/mnt/shared-storage-user/evobox-share/yinzhenyun/slime-env-0.2.3/bin/ray"
export PYTHON_BIN="/mnt/shared-storage-user/evobox-share/yinzhenyun/slime-env-0.2.3/bin/python3.12"
export SLIME_ENV_BIN="/mnt/shared-storage-user/evobox-share/yinzhenyun/slime-env-0.2.3/bin"

case ":${PATH}:" in
  *":${SLIME_ENV_BIN}:"*) ;;
  *) export PATH="${SLIME_ENV_BIN}:${PATH}" ;;
esac

# OpenRT imports expect these variables at import time.
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:18892/v1"

# -------------------------------------------
# Shared Buffer Server
# -------------------------------------------
export BUFFER_SERVER_HOST="127.0.0.1"
export BUFFER_SERVER_PORT="18889"

# -------------------------------------------
# Policy endpoints
# -------------------------------------------
export ATTACKER_LLM_PROXY_HOST="127.0.0.1"
export ATTACKER_LLM_PROXY_PORT="18890"
export DEFENDER_LLM_PROXY_HOST="127.0.0.1"
export DEFENDER_LLM_PROXY_PORT="18891"

export ATTACKER_POLICY_ID="attacker_policy"
export DEFENDER_POLICY_ID="defender_policy"

export AIEVOBOX_POLICY_CONFIG="{\"attacker\":{\"policy_id\":\"${ATTACKER_POLICY_ID}\",\"base_url\":\"http://${ATTACKER_LLM_PROXY_HOST}:${ATTACKER_LLM_PROXY_PORT}/v1\",\"model\":\"attacker\",\"session_suffix\":true},\"defender\":{\"policy_id\":\"${DEFENDER_POLICY_ID}\",\"base_url\":\"http://${DEFENDER_LLM_PROXY_HOST}:${DEFENDER_LLM_PROXY_PORT}/v1\",\"model\":\"defender\",\"session_suffix\":true}}"

# -------------------------------------------
# RL / Slime settings
# -------------------------------------------
export RL_GROUP_SIZE="2"
export RL_EPOCH="1000"
export RL_OFF_BY_N="0"
export DAPO_filter="false"

export LLM_MAX_LENGTH="4096"
export LLM_TEMPERATURE="1.0"

# 4-GPU quick validation layout:
#   attacker: 1 actor GPU + 1 rollout GPU
#   defender: 1 actor GPU + 1 rollout GPU
export NUM_GPUS="4"
export ACTOR_NUM_NODES="1"
export ACTOR_NUM_GPUS_PER_NODE="1"
export ROLLOUT_NUM_GPUS="1"
export ROLLOUT_NUM_GPUS_PER_ENGINE="1"
export TENSOR_MODEL_PARALLEL_SIZE="1"
export PIPELINE_MODEL_PARALLEL_SIZE="1"
export CONTEXT_PARALLEL_SIZE="1"
export MAX_TOKENS_PER_GPU="3000"

export SLIME_N_SAMPLES_PER_PROMPT="${RL_GROUP_SIZE}"
export SLIME_GLOBAL_BATCH_SIZE="8"
export SLIME_ROLLOUT_BATCH_SIZE="4"
