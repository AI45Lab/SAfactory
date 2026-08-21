#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

ulimit -n 65536 2>/dev/null || \
  echo "Warning: could not set ulimit -n 65536 (current: $(ulimit -n))"

# Each invocation starts a fresh local Ray/SGLang training runtime. Keep the
# separately launched Buffer Server alive.
pkill -9 sglang || true
ray stop --force || true
pkill -9 ray || true
sleep 2

mkdir -p "${SAVE_DIR}" "${WANDB_DIR}" "${LOG_ROOT}"
if [[ -z "${AIEVOBOX_RUN_DIR:-}" ]]; then
  export AIEVOBOX_RUN_DIR="$(<"${LOG_ROOT}/.current_run")"
fi
mkdir -p "${AIEVOBOX_RUN_DIR}"

export ROLLOUT_BUFFER_URL="http://${BUFFER_SERVER_HOST}:${BUFFER_SERVER_PORT}"
export LLM_PROXY_URL="http://${LLM_PROXY_HOST}:${LLM_PROXY_PORT}"
export PYTHONPATH="${SLIME_HOME}:${AIEVOBOX_ROOT}/rl:${AIEVOBOX_ROOT}:${MEGATRON_HOME}${PYTHONPATH:+:${PYTHONPATH}}"

source "${MODEL_SCRIPT}"

CHECKPOINT_ARGS=(
  --hf-checkpoint "${HF_CKPT_DIR}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
  --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"
  --rollout-buffer-url "${ROLLOUT_BUFFER_URL}"
  --disable-rollout-global-dataset
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${RL_ROLLOUT_GROUP_BATCH_SIZE}"
  --n-samples-per-prompt "${RL_GROUP_SIZE}"
  --rollout-max-response-len "${RL_ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${LLM_TEMPERATURE}"
  --rollout-top-p "${LLM_TOP_P}"
  --global-batch-size "${RL_GLOBAL_BATCH_SIZE}"
  --custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH}"
)

MEGATRON_ARGS=(
  --train-backend "${TRAIN_BACKEND}"
  --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
  --qkv-format bshd
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size "${EP_SIZE}"
  --expert-tensor-parallel-size "${ETP_SIZE}"
  --sequence-parallel
  --recompute-granularity "${RECOMPUTE_GRANULARITY}"
  --recompute-method "${RECOMPUTE_METHOD}"
  --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend "${ATTENTION_BACKEND}"
  --freeze-params-name-list
  'vision_model\.'
)

TRAIN_ARGS=(
  --micro-batch-size 1
  --log-probs-chunk-size 1024
)

GRPO_ARGS=(
  --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
  --disable-grpo-std-normalization
  --entropy-coef "${ENTROPY_COEF}"
  --eps-clip "${EPS_CLIP}"
  --eps-clip-high "${EPS_CLIP_HIGH}"
)

OPTIMIZER_ARGS=(
  --optimizer "${OPTIMIZER}"
  --lr "${LR}"
  --lr-decay-style "${LR_DECAY_STYLE}"
  --weight-decay "${WEIGHT_DECAY}"
  --adam-beta1 "${ADAM_BETA1}"
  --adam-beta2 "${ADAM_BETA2}"
)

WANDB_ARGS=(
  --use-wandb
  --wandb-mode "${WANDB_MODE}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-dir "${WANDB_DIR}"
  --wandb-always-use-train-step
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
  --sglang-cuda-graph-bs ${SGLANG_CUDA_GRAPH_BS}
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --sglang-schedule-conservativeness "${SGLANG_SCHEDULE_CONSERVATIVENESS}"
  --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
  --sglang-mamba-scheduler-strategy extra_buffer
  --sglang-enable-mixed-chunk
  --sglang-log-level "${SGLANG_LOG_LEVEL}"
  --sglang-log-level-http "${SGLANG_LOG_LEVEL_HTTP}"
)

RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

names = (
    "AIEVOBOX_ROOT",
    "AIEVOBOX_RUN_DIR",
    "AIEVOBOX_LLM_PROXY_WORKERS",
    "AIEVOBOX_TRAININFO_WORKERS",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "DAPO_filter",
    "LLM_MAX_LENGTH",
    "LLM_PROXY_ENABLE_CONSOLE_LOG",
    "LLM_PROXY_PORT",
    "LLM_PROXY_URL",
    "LLM_TEMPERATURE",
    "LLM_TOP_P",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "PYTORCH_CUDA_ALLOC_CONF",
    "RL_OFF_BY_N",
    "ROLLOUT_BUFFER_URL",
    "SGLANG_LOGGING_CONFIG_PATH",
    "SLIME_ROLLBUF_RESTART_TRAINING",
    "WANDB_DIR",
    "WANDB_MODE",
)
print(json.dumps({"env_vars": {name: os.environ[name] for name in names}}))
PY
)"

echo "Starting OSGym Slime training"
echo "  Run dir: ${AIEVOBOX_RUN_DIR}"
echo "  HF checkpoint: ${HF_CKPT_DIR}"
echo "  Rollout buffer: ${ROLLOUT_BUFFER_URL}"

ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats

ray job submit --address="${RAY_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_HOME}/train.py" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${MEGATRON_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  2>&1 | tee "${AIEVOBOX_RUN_DIR}/slime.log"
