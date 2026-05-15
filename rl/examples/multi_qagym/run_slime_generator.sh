#!/usr/bin/env bash

ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit -n 65536 (current: $(ulimit -n))"

if [ "${SKIP_RUNTIME_CLEANUP:-0}" != "1" ]; then
  pkill -9 sglang || true
  sleep 2
  pkill -9 ray || true
  sleep 2
fi

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

NVIDIA_DRIVER_LIBS="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
case ":${LD_LIBRARY_PATH:-}:" in
  *":/usr/local/nvidia/lib:"*":/usr/local/nvidia/lib64:"*) ;;
  *) export LD_LIBRARY_PATH="${NVIDIA_DRIVER_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

if [ "${CUDA_HOME:-}" = "${SLIME_ENV_BIN%/bin}" ] || [ "${CUDA_HOME:-}" = "${PYTHON_BIN%/bin/python3.12}" ]; then
  unset CUDA_HOME
fi

ROLLOUT_BUFFER_URL="http://${BUFFER_SERVER_HOST}:${BUFFER_SERVER_PORT}"
LLM_PROXY_URL="http://${LLM_PROXY_HOST}:${LLM_PROXY_PORT}"

export PYTHONUNBUFFERED=1

SLIME_HOME=${SLIME_HOME:-/root/slime}
HF_CKPT_DIR=${HF_CKPT_DIR:-/mnt/shared-storage-user/evobox-share/hf-hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
SAVE_DIR=${SAVE_DIR:-/mnt/shared-storage-user/evobox-share-gpfs2/leishanzhe/model/checkpoints/rl/multi_qagym/${AIEVOBOX_POLICY_ID}/${AIEVOBOX_RUN_ID}}
MODEL_SCRIPT=${MODEL_SCRIPT:-${SLIME_HOME}/scripts/models/qwen3-1.7B.sh}
source "${MODEL_SCRIPT}"

CKPT_ARGS=(
  --hf-checkpoint ${HF_CKPT_DIR}
  --load ${HF_CKPT_DIR}
  --save ${SAVE_DIR}
  --save-interval ${SAVE_INTERVAL:-20}
)

ROLLOUT_ARGS=(
  --rollout-function-path rl.slime_generator.generate_rollout
  --rollout-buffer-url ${ROLLOUT_BUFFER_URL}
  --disable-rollout-global-dataset
  --num-rollout ${NUM_ROLLOUT:-300}
  --rollout-batch-size ${SLIME_ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${SLIME_N_SAMPLES_PER_PROMPT}
  --rollout-max-response-len ${LLM_MAX_LENGTH}
  --rollout-temperature ${LLM_TEMPERATURE}
  --global-batch-size ${SLIME_GLOBAL_BATCH_SIZE}
  --loss-mask-type qwen
)

MEGATRON_ARGS=(
  --train-backend megatron
  --megatron-to-hf-mode bridge
  --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE}
  --pipeline-model-parallel-size ${PIPELINE_MODEL_PARALLEL_SIZE}
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

TRAIN_ARGS=(
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
  --calculate-per-token-loss
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.2
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr ${LR:-1e-6}
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-evobox}"
WANDB_TEAM="${WANDB_TEAM:-sys555-ai}"
WANDB_GROUP="${WANDB_GROUP:-multi-qagym-${AIEVOBOX_POLICY_ID}-${AIEVOBOX_RUN_ID}}"
WANDB_DIR="${WANDB_DIR:-${SCRIPT_DIR}/db/${WANDB_GROUP}}"

if [ "${ENABLE_WANDB}" = "1" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-team "${WANDB_TEAM}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-dir "${WANDB_DIR}"
    --disable-wandb-random-suffix
  )
else
  WANDB_ARGS=()
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.7}
  --sglang-attention-backend ${SGLANG_ATTENTION_BACKEND:-fa3}
  --sglang-log-level error
  --sglang-log-level-http error
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RAY_PORT=${RAY_PORT:-6379}
RAY_RESTART=${RAY_RESTART:-1}

if [ "${RAY_RESTART}" = "1" ]; then
  pkill -9 sglang || true
  "${RAY_BIN}" stop --force || true
  pkill -9 ray || true
  pkill -9 raylet || true
  pkill -9 gcs_server || true
  pkill -9 dashboard || true
  pkill -9 dashboard_agent || true
  pkill -9 runtime_env_agent || true
  sleep 1
fi

"${PYTHON_BIN}" -V
"${RAY_BIN}" --version

if [ "${RAY_START:-1}" = "1" ]; then
  "${RAY_BIN}" start --head --node-ip-address ${MASTER_ADDR} --port ${RAY_PORT} --num-gpus ${NUM_GPUS} --disable-usage-stats
fi

export SGLANG_LOGGING_CONFIG_PATH=${SGLANG_LOGGING_CONFIG_PATH:-"${AIEVOBOX_ROOT}/rl/sglang_logging.json"}

RUNTIME_ENV_JSON="{\
  \"env_vars\": {\
    \"PYTHONPATH\": \"${SLIME_HOME}:${AIEVOBOX_ROOT}/rl:${AIEVOBOX_ROOT}:/root/Megatron-LM\",\
    \"PATH\": \"${PATH}\",\
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",\
    \"CUDA_HOME\": \"${CUDA_HOME:-}\",\
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",\
    \"PYTHONUNBUFFERED\": \"1\",\
    \"LLM_PROXY_HOST\": \"${LLM_PROXY_HOST}\",\
    \"LLM_PROXY_PORT\": \"${LLM_PROXY_PORT}\",\
    \"LLM_PROXY_URL\": \"${LLM_PROXY_URL}\",\
    \"ROLLOUT_BUFFER_URL\": \"${ROLLOUT_BUFFER_URL}\",\
    \"AIEVOBOX_POLICY_ID\": \"${AIEVOBOX_POLICY_ID}\",\
    \"AIEVOBOX_ROLLOUT_OWNER\": \"${AIEVOBOX_ROLLOUT_OWNER}\",\
    \"SLIME_OFF_BY_N\": \"${SLIME_OFF_BY_N:-0}\",\
    \"WANDB_MODE\": \"${WANDB_MODE}\",\
    \"WANDB_DIR\": \"${WANDB_DIR}\"\
  }\
}"

TRAIN_CMD=(
  -- "${PYTHON_BIN}" -u "${SLIME_HOME}/train.py"
  --actor-num-nodes ${ACTOR_NUM_NODES}
  --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE}
  --rollout-num-gpus ${ROLLOUT_NUM_GPUS}
  ${MODEL_ARGS[@]}
  ${MEGATRON_ARGS[@]}
  ${CKPT_ARGS[@]}
  ${ROLLOUT_ARGS[@]}
  ${OPTIMIZER_ARGS[@]}
  ${GRPO_ARGS[@]}
  ${WANDB_ARGS[@]}
  ${TRAIN_ARGS[@]}
  ${SGLANG_ARGS[@]}
)

RAY_JOB_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-multi-qagym-${AIEVOBOX_POLICY_ID}-$(date +%Y%m%d%H%M%S)}"

echo "Submitting Ray job: ${RAY_JOB_SUBMISSION_ID}"
"${RAY_BIN}" job submit --address="http://127.0.0.1:8265" \
  --submission-id="${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  "${TRAIN_CMD[@]}"

echo "Following Ray job logs: ${RAY_JOB_SUBMISSION_ID}"
"${RAY_BIN}" job logs --address="http://127.0.0.1:8265" -f "${RAY_JOB_SUBMISSION_ID}"
