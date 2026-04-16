#!/usr/bin/env bash

# Increase file descriptor limit for high concurrency.
ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit -n 65536 (current: $(ulimit -n))"

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RL_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# Load example-specific environment defaults.
source "${SCRIPT_DIR}/env.sh"

# Force Ray and Python to come from the same runtime environment.
RAY_BIN="${RAY_BIN:-$(command -v ray)}"
if [ -z "${PYTHON_BIN:-}" ]; then
  RAY_BIN_DIR="$(cd -- "$(dirname -- "${RAY_BIN}")" &>/dev/null && pwd)"
  if [ -x "${RAY_BIN_DIR}/python" ]; then
    PYTHON_BIN="${RAY_BIN_DIR}/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

# Default to math500 env config when caller does not specify one.
AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-/mnt/shared-storage-user/leishanzhe/repo/AIEvoBox}"
export AIEVOBOX_ROOT
export AIEVOBOX_ENV_CONFIG="${AIEVOBOX_ENV_CONFIG:-${AIEVOBOX_ROOT}/env/math500_text/math500_text_env_configs.yaml}"

# Construct URLs from host and port.
ROLLOUT_BUFFER_URL="http://${BUFFER_SERVER_HOST}:${BUFFER_SERVER_PORT}"
LLM_PROXY_URL="http://${LLM_PROXY_HOST}:${LLM_PROXY_PORT}"

# Teacher endpoint used by OPD.
# Force set to correct SGLang server address
TEACHER_URL="http://100.99.167.245:30172/generate"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
OPD_TEACHER_MAX_CONCURRENCY="${OPD_TEACHER_MAX_CONCURRENCY:-16}"
OPD_TEACHER_TIMEOUT_SECONDS="${OPD_TEACHER_TIMEOUT_SECONDS:-60}"

echo "[opd] TEACHER_URL=${TEACHER_URL}"
echo "[opd] OPD_KL_COEF=${OPD_KL_COEF}"
echo "[opd] OPD_TEACHER_MAX_CONCURRENCY=${OPD_TEACHER_MAX_CONCURRENCY}"

export PYTHONBUFFERED=16
NUM_GPUS=${NUM_GPUS:-8}

SLIME_HOME=${SLIME_HOME:-/root/slime}
MODEL_PRESET_SCRIPT="${MODEL_PRESET_SCRIPT:-qwen2.5-7B.sh}"
HF_CKPT_DIR="${HF_CKPT_DIR:-/mnt/shared-storage-user/evobox-share/hf-hub/models--Qwen2.5-Math-7B-Instruct/snapshots/c1d08860689aca85d3fa9402334cac49ecde6037}"
REF_LOAD_DIR="${REF_LOAD_DIR:-${HF_CKPT_DIR}}"
LOAD_DIR="${LOAD_DIR:-${HF_CKPT_DIR}}"
SAVE_DIR="${SAVE_DIR:-/mnt/shared-storage-user/evobox-share-gpfs2/leishanzhe/model/checkpoints/opd/Qwen2.5-Math-7B-Instruct/$(date +%Y%m%d_%H%M)}"

if [ ! -d "${HF_CKPT_DIR}" ]; then
  echo "[math500] ERROR: HF_CKPT_DIR does not exist: ${HF_CKPT_DIR}"
  exit 1
fi
if [ ! -d "${REF_LOAD_DIR}" ]; then
  echo "[math500] WARN: REF_LOAD_DIR does not exist, fallback to HF_CKPT_DIR: ${REF_LOAD_DIR}"
  REF_LOAD_DIR="${HF_CKPT_DIR}"
fi
if [ ! -d "${LOAD_DIR}" ]; then
  echo "[math500] WARN: LOAD_DIR does not exist, fallback to HF_CKPT_DIR: ${LOAD_DIR}"
  LOAD_DIR="${HF_CKPT_DIR}"
fi
mkdir -p "${SAVE_DIR}"

source "${SLIME_HOME}/scripts/models/${MODEL_PRESET_SCRIPT}"
# Keep rotary-base aligned with Qwen2.5-Math checkpoints (rope_theta=10000).
MODEL_ARGS+=(--rotary-base 10000)
CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT_DIR}"
   --ref-load "${REF_LOAD_DIR}"
   --load "${LOAD_DIR}"
   --save "${SAVE_DIR}"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --rollout-function-path rl.slime_generator.generate_rollout
   --rollout-buffer-url ${ROLLOUT_BUFFER_URL}
   --disable-rollout-global-dataset
   --num-rollout 300
   --rollout-batch-size ${SLIME_ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${SLIME_N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len 64
   --rollout-temperature ${LLM_TEMPERATURE}
   --global-batch-size ${SLIME_GLOBAL_BATCH_SIZE}
   --loss-mask-type qwen
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 5000
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef ${OPD_KL_COEF}
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.2
)

TEACHER_ARGS=(
   --rm-url ${TEACHER_URL}
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-evobox}"
WANDB_TEAM="${WANDB_TEAM:-ben}"
WANDB_GROUP="${WANDB_GROUP:-slime-opd-math500}"
WANDB_DIR="${WANDB_DIR:-/root/wandb_logs}"

if [ "${ENABLE_WANDB}" = "1" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-team "${WANDB_TEAM}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-dir "${WANDB_DIR}"
  )
else
  WANDB_ARGS=()
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-log-level error
   --sglang-log-level-http error
)

MISC_ARGS=(
   --megatron-to-hf-mode bridge
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --calculate-per-token-loss
)

# Start Ray.
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RAY_PORT=${RAY_PORT:-6379}
RAY_RESTART=${RAY_RESTART:-1}

if [ "${RAY_RESTART}" = "1" ]; then
  "${RAY_BIN}" stop --force || true
  pkill -9 ray || true
  pkill -9 raylet || true
  pkill -9 gcs_server || true
  sleep 1
fi

"${PYTHON_BIN}" -V
"${RAY_BIN}" --version

"${RAY_BIN}" start --head --node-ip-address ${MASTER_ADDR} --port ${RAY_PORT} --num-gpus ${NUM_GPUS} --disable-usage-stats

export SGLANG_LOGGING_CONFIG_PATH=${SGLANG_LOGGING_CONFIG_PATH:-"/root/AIEvoBox/rl/sglang_logging.json"}

RUNTIME_ENV_JSON="{\
  \"env_vars\": {\
    \"PYTHONPATH\": \"${AIEVOBOX_ROOT}:${AIEVOBOX_ROOT}/rl:/root/Megatron-LM:${SLIME_HOME}\",\
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",\
    \"LLM_PROXY_URL\": \"${LLM_PROXY_URL}\",\
    \"ROLLOUT_BUFFER_URL\": \"${ROLLOUT_BUFFER_URL}\",\
    \"SLIME_OFF_BY_N\": \"${SLIME_OFF_BY_N:-0}\",\
    \"TEACHER_URL\": \"${TEACHER_URL}\",\
    \"RM_URL\": \"${TEACHER_URL}\",\
    \"OPD_TEACHER_MAX_CONCURRENCY\": \"${OPD_TEACHER_MAX_CONCURRENCY}\",\
    \"OPD_TEACHER_TIMEOUT_SECONDS\": \"${OPD_TEACHER_TIMEOUT_SECONDS}\",\
    \"WANDB_MODE\": \"${WANDB_MODE}\",\
    \"WANDB_DIR\": \"${WANDB_DIR}\"\
  }\
}"

"${RAY_BIN}" job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${PYTHON_BIN}" "${SLIME_HOME}/train_async.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 6 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${TEACHER_ARGS[@]}
