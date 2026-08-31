#!/usr/bin/env bash

set -euo pipefail

ulimit -n 65536 2>/dev/null || echo "Warning: could not set ulimit -n 65536 (current: $(ulimit -n))"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

ENV_SH="${RL_ENV_SH:-}"
if [[ "${1:-}" == "--env" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Usage: $0 [--env /path/to/env.sh]" >&2
    exit 1
  fi
  ENV_SH="$2"
  shift 2
elif [[ $# -gt 0 && "${1}" != -* ]]; then
  ENV_SH="$1"
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Unexpected argument(s): $*" >&2
  echo "Usage: $0 [--env /path/to/env.sh]" >&2
  exit 1
fi

if [[ -n "${ENV_SH}" ]]; then
  if [[ ! -f "${ENV_SH}" ]]; then
    echo "Missing env file: ${ENV_SH}" >&2
    exit 1
  fi
  source "${ENV_SH}"
elif [[ -z "${AIEVOBOX_ROOT:-}" ]]; then
  echo "No RL env loaded. Source an env.sh first, or run with RL_ENV_SH=/path/to/env.sh." >&2
  echo "Example: RL_ENV_SH=${SCRIPT_DIR}/examples/geo3k_vl/env.sh ${SCRIPT_DIR}/run_slime_generator.sh" >&2
  exit 1
fi

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if is_true "${CLEANUP_BEFORE_RUN}"; then
  pkill -9 sglang || true
  "${RAY_BIN}" stop --force || true
  pkill -9 ray || true
  pkill -9 raylet || true
  pkill -9 gcs_server || true
  if is_true "${KILL_PYTHON_BEFORE_RUN}"; then
    pkill -9 python || true
  fi
  sleep 2
fi

require_dir "${AIEVOBOX_ROOT}" "AIEVOBOX_ROOT"
require_dir "${SLIME_HOME}" "SLIME_HOME"
require_dir "${MEGATRON_HOME}" "MEGATRON_HOME"
require_file "${MODEL_SCRIPT}" "Slime model script"
require_file "${TRAIN_ENTRYPOINT}" "Slime train entrypoint"
require_dir "${HF_CKPT_DIR}" "HF_CKPT_DIR"
if [[ -n "${AIEVOBOX_AGENT_CONFIG:-}" ]]; then
  require_file "${AIEVOBOX_AGENT_CONFIG}" "AIEVOBOX_AGENT_CONFIG"
fi
if [[ -n "${AIEVOBOX_AGENT_START_CONFIG:-}" ]]; then
  require_file "${AIEVOBOX_AGENT_START_CONFIG}" "AIEVOBOX_AGENT_START_CONFIG"
fi
mkdir -p "${SAVE_DIR}" "${WANDB_DIR}" "${LOG_ROOT}"

if [[ -z "${RL_ROLLOUT_GROUP_BATCH_SIZE:-}" && -n "${RL_GLOBAL_BATCH_SIZE:-}" && -n "${RL_GROUP_SIZE:-}" ]]; then
  export RL_ROLLOUT_GROUP_BATCH_SIZE="$((RL_GLOBAL_BATCH_SIZE / RL_GROUP_SIZE))"
fi
if [[ -z "${RL_GLOBAL_BATCH_SIZE:-}" && -n "${RL_ROLLOUT_GROUP_BATCH_SIZE:-}" && -n "${RL_GROUP_SIZE:-}" ]]; then
  export RL_GLOBAL_BATCH_SIZE="$((RL_ROLLOUT_GROUP_BATCH_SIZE * RL_GROUP_SIZE))"
fi

if (( RL_GROUP_SIZE <= 0 || RL_ROLLOUT_GROUP_BATCH_SIZE <= 0 || RL_GLOBAL_BATCH_SIZE <= 0 )); then
  echo "RL_GROUP_SIZE, RL_ROLLOUT_GROUP_BATCH_SIZE, and RL_GLOBAL_BATCH_SIZE must be positive" >&2
  exit 1
fi

if [[ -z "${AIEVOBOX_RUN_DIR:-}" ]]; then
  export AIEVOBOX_RUN_DIR="${LOG_ROOT}/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "${AIEVOBOX_RUN_DIR}"
printf '%s\n' "${AIEVOBOX_RUN_DIR}" > "${LOG_ROOT}/.current_run"

ROLLOUT_BUFFER_URL="http://${BUFFER_SERVER_HOST}:${BUFFER_SERVER_PORT}"
LLM_PROXY_URL="http://${LLM_PROXY_HOST}:${LLM_PROXY_PORT}"
export ROLLOUT_BUFFER_URL LLM_PROXY_URL

export WANDB_MODE
export PYTHONUNBUFFERED
export PYTORCH_CUDA_ALLOC_CONF
export MODEL_ARGS_ROTARY_BASE

source "${MODEL_SCRIPT}"
if ! declare -p MODEL_ARGS >/dev/null 2>&1; then
  MODEL_ARGS=()
fi
if [[ -n "${MODEL_ARGS_EXTRA:-}" ]]; then
  read -r -a MODEL_ARGS_EXTRA_ARRAY <<< "${MODEL_ARGS_EXTRA}"
  MODEL_ARGS+=("${MODEL_ARGS_EXTRA_ARRAY[@]}")
fi

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT_DIR}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)
if [[ -n "${LOAD_DIR:-}" ]]; then
  require_dir "${LOAD_DIR}" "LOAD_DIR"
  CKPT_ARGS+=(--load "${LOAD_DIR}")
fi
if [[ -n "${REF_LOAD_DIR:-}" ]]; then
  require_dir "${REF_LOAD_DIR}" "REF_LOAD_DIR"
  CKPT_ARGS+=(--ref-load "${REF_LOAD_DIR}")
fi

if [[ -n "${SGLANG_LOGGING_CONFIG_PATH:-}" ]]; then
  require_file "${SGLANG_LOGGING_CONFIG_PATH}" "SGLang logging config"
  export SGLANG_LOGGING_CONFIG_PATH
else
  unset SGLANG_LOGGING_CONFIG_PATH
fi

ROLLOUT_ARGS=(
  --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"
  --rollout-buffer-url "${ROLLOUT_BUFFER_URL}"
  --disable-rollout-global-dataset
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${RL_ROLLOUT_GROUP_BATCH_SIZE}"
  --n-samples-per-prompt "${RL_GROUP_SIZE}"
  --rollout-max-response-len "${LLM_MAX_LENGTH}"
  --rollout-temperature "${LLM_TEMPERATURE}"
  --global-batch-size "${RL_GLOBAL_BATCH_SIZE}"
  --loss-mask-type "${LOSS_MASK_TYPE}"
)
if [[ -n "${CUSTOM_REWARD_POST_PROCESS_PATH:-}" ]]; then
  ROLLOUT_ARGS+=(--custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH}")
fi

MEGATRON_ARGS=(
  --train-backend "${TRAIN_BACKEND}"
  --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size "${EP_SIZE}"
  --expert-tensor-parallel-size "${ETP_SIZE}"
  --recompute-granularity "${RECOMPUTE_GRANULARITY}"
  --recompute-method "${RECOMPUTE_METHOD}"
  --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend "${ATTENTION_BACKEND}"
)

TRAIN_ARGS=(
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --qkv-format "${QKV_FORMAT:-thd}"
)
if is_true "${USE_DYNAMIC_BATCH_SIZE}"; then
  TRAIN_ARGS+=(--use-dynamic-batch-size)
else
  TRAIN_ARGS+=(--micro-batch-size "${MICRO_BATCH_SIZE:-1}")
fi
if is_true "${USE_DYNAMIC_GLOBAL_BATCH_SIZE}"; then
  TRAIN_ARGS+=(--use-dynamic-global-batch-size)
fi
if is_true "${CALCULATE_PER_TOKEN_LOSS}"; then
  TRAIN_ARGS+=(--calculate-per-token-loss)
fi

GRPO_ARGS=(
  --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
  --entropy-coef "${ENTROPY_COEF}"
  --eps-clip "${EPS_CLIP}"
  --eps-clip-high "${EPS_CLIP_HIGH}"
)
if is_true "${USE_OPD:-false}"; then
  GRPO_ARGS+=(--use-opd --opd-type "${OPD_TYPE:-sglang}" --opd-kl-coef "${OPD_KL_COEF:-1.0}")
fi

TEACHER_ARGS=()
if [[ -n "${TEACHER_URL:-}" ]]; then
  TEACHER_ARGS=(--rm-url "${TEACHER_URL}")
fi

OPTIMIZER_ARGS=(
  --optimizer "${OPTIMIZER}"
  --lr "${LR}"
  --lr-decay-style "${LR_DECAY_STYLE}"
  --weight-decay "${WEIGHT_DECAY}"
  --adam-beta1 "${ADAM_BETA1}"
  --adam-beta2 "${ADAM_BETA2}"
)

WANDB_ARGS=()
if is_true "${USE_WANDB}"; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-dir "${WANDB_DIR}"
  )
  if [[ -n "${WANDB_TEAM}" ]]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
  fi
  if is_true "${WANDB_ALWAYS_USE_TRAIN_STEP}"; then
    WANDB_ARGS+=(--wandb-always-use-train-step)
  fi
fi

read -r -a CUDA_GRAPH_BS_ARRAY <<< "${SGLANG_CUDA_GRAPH_BS:-}"
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
  --sglang-log-level "${SGLANG_LOG_LEVEL}"
  --sglang-log-level-http "${SGLANG_LOG_LEVEL_HTTP}"
)
if [[ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]]; then
  SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
fi
if [[ -n "${SGLANG_SCHEDULE_CONSERVATIVENESS:-}" ]]; then
  SGLANG_ARGS+=(--sglang-schedule-conservativeness "${SGLANG_SCHEDULE_CONSERVATIVENESS}")
fi
if [[ -n "${SGLANG_CHUNKED_PREFILL_SIZE:-}" ]]; then
  SGLANG_ARGS+=(--sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}")
fi
if ((${#CUDA_GRAPH_BS_ARRAY[@]} > 0)); then
  SGLANG_ARGS+=(--sglang-cuda-graph-bs "${CUDA_GRAPH_BS_ARRAY[@]}")
fi
if is_true "${SGLANG_ENABLE_MIXED_CHUNK:-false}"; then
  SGLANG_ARGS+=(--sglang-enable-mixed-chunk)
fi
# Prefix (RadixAttention) caching: reuse KV cache for shared prompt prefixes
# (system prompt, task template, conversation history across multi-turn agent
# steps). Big win for PatchEval where every episode shares the same system
# prompt and the same task description across GRPO samples. Without this the
# SGLang log shows #cached-token: 0 on every prefill. Default on; disable via
# SGLANG_ENABLE_PREFIX_CACHING=false.
if is_true "${SGLANG_ENABLE_PREFIX_CACHING:-true}"; then
  SGLANG_ARGS+=(--sglang-enable-prefix-caching)
fi

# Router policy: how the SGLang router distributes requests across engines.
#   cache_aware (sglang default) — greedy per-request prefix match; under high
#                     concurrency it scatters one session's turns across
#                     engines, so ~78% of prefills recompute the full prompt
#                     (cached-token=0). Wastes the multi-engine capacity.
#   manual (chosen here) — sticky-session routing via the X-SMG-Routing-Key
#                     header that llm_proxy now sends. Each session_id is pinned
#                     to one worker and stays there (only remaps if that worker
#                     dies). Stronger stickiness than consistent_hashing, ideal
#                     for fixed-engine RL rollouts. Supported since SGLang Model
#                     Gateway v0.3.1 (PR #15907, 2025-12-27); the installed
#                     sglang_router 0.3.2 has it. `consistent_hashing` is a
#                     newer CLI choice (PR #17972, 2026-02-15) NOT in 0.3.2, so do
#                     NOT set SGLANG_ROUTER_POLICY=consistent_hashing on this
#                     build (argparse will reject it and crash startup).
#                     Override via SGLANG_ROUTER_POLICY if needed.
SGLANG_ARGS+=(--router-policy "${SGLANG_ROUTER_POLICY:-manual}")

# Colocate mode: training (Megatron) and inference (SGLang) share the SAME GPUs.
# Required for big models on few GPUs (e.g. 27B on a single 8-card node): the
# dedicated-pool split (actor + rollout = NUM_GPUS) would need ~16 cards for 27B,
# but colocate time-shares 8 cards via CPU offload between rollout/train phases.
# When on, --rollout-num-gpus is ignored (auto = actor GPUs) and --offload is
# forced by the trainer. Set SLIME_COLOCATE=1 to enable.
COLOCATE_ARGS=()
if is_true "${SLIME_COLOCATE:-false}"; then
  COLOCATE_ARGS=(--colocate)
  echo "  Colocate: ON (train+rollout share ${ACTOR_NUM_GPUS_PER_NODE} GPUs)"
fi

RAY_RUNTIME_PYTHONPATH="${SLIME_HOME}:${AIEVOBOX_ROOT}/rl:${AIEVOBOX_ROOT}:${MEGATRON_HOME}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  RAY_RUNTIME_PYTHONPATH="${RAY_RUNTIME_PYTHONPATH}:${PYTHONPATH}"
fi

RUNTIME_ENV_JSON="{\
  \"env_vars\": {\
    \"AIEVOBOX_ROOT\": \"${AIEVOBOX_ROOT}\",\
    \"AIEVOBOX_RUN_DIR\": \"${AIEVOBOX_RUN_DIR}\",\
    \"PYTHONPATH\": \"${RAY_RUNTIME_PYTHONPATH}\",\
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"${CUDA_DEVICE_MAX_CONNECTIONS}\",\
    \"LLM_PROXY_PORT\": \"${LLM_PROXY_PORT}\",\
    \"LLM_MAX_LENGTH\": \"${LLM_MAX_LENGTH}\",\
    \"LLM_TEMPERATURE\": \"${LLM_TEMPERATURE}\",\
    \"LLM_TOP_P\": \"${LLM_TOP_P}\",\
    \"AIEVOBOX_LLM_PROXY_WORKERS\": \"${AIEVOBOX_LLM_PROXY_WORKERS}\",\
    \"AIEVOBOX_TRAININFO_WORKERS\": \"${AIEVOBOX_TRAININFO_WORKERS}\",\
    \"LLM_PROXY_URL\": \"${LLM_PROXY_URL}\",\
    \"ROLLOUT_BUFFER_URL\": \"${ROLLOUT_BUFFER_URL}\",\
    \"SLIME_ROLLBUF_RESTART_TRAINING\": \"${SLIME_ROLLBUF_RESTART_TRAINING}\",\
    \"RL_OFF_BY_N\": \"${RL_OFF_BY_N}\",\
    \"DAPO_filter\": \"${DAPO_filter}\",\
    \"LLM_PROXY_ENABLE_CONSOLE_LOG\": \"${LLM_PROXY_ENABLE_CONSOLE_LOG}\",\
    \"AIEVOBOX_DEBUG_CACHE_PROCESSOR_COMPARE\": \"${AIEVOBOX_DEBUG_CACHE_PROCESSOR_COMPARE:-0}\",\
    \"TEACHER_URL\": \"${TEACHER_URL:-}\",\
    \"RM_URL\": \"${TEACHER_URL:-}\",\
    \"OPD_TEACHER_MAX_CONCURRENCY\": \"${OPD_TEACHER_MAX_CONCURRENCY:-}\",\
    \"OPD_TEACHER_TIMEOUT_SECONDS\": \"${OPD_TEACHER_TIMEOUT_SECONDS:-}\",\
    \"WANDB_MODE\": \"${WANDB_MODE}\",\
    \"WANDB_DIR\": \"${WANDB_DIR}\"\
  }\
}"

echo "Starting ${AIEVOBOX_EXAMPLE_NAME:-rl} Slime generator"
echo "  Python: ${PYTHON_BIN}"
echo "  Ray: ${RAY_BIN}"
echo "  Train entrypoint: ${TRAIN_ENTRYPOINT}"
echo "  Mode: ${AIEVOBOX_MODE}"
echo "  Run dir: ${AIEVOBOX_RUN_DIR}"
echo "  Slime: ${SLIME_HOME}"
echo "  Megatron: ${MEGATRON_HOME}"
echo "  HF checkpoint: ${HF_CKPT_DIR}"
echo "  Load dir: ${LOAD_DIR:-<none>}"
echo "  Actor GPUs/node: ${ACTOR_NUM_GPUS_PER_NODE}"
echo "  Rollout GPUs: ${ROLLOUT_NUM_GPUS}"
echo "  Rollout GPUs/engine: ${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "  Rollout buffer: ${ROLLOUT_BUFFER_URL}"
echo "  LLM proxy: ${LLM_PROXY_URL}"

cd "${AIEVOBOX_ROOT}"

RAY_START_ARGS=(start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats)
if [[ -n "${RAY_PORT:-}" ]]; then
  RAY_START_ARGS+=(--port "${RAY_PORT}")
fi
# Multi-node: pre-build the Ray cluster manually (head + `ray start --address`
# on workers), then run with SKIP_RAY_START=1 so this script reuses the existing
# cluster instead of `ray start --head` (which would restart Ray and drop the
# workers). Also set CLEANUP_BEFORE_RUN=false so the pre-started cluster survives.
if is_true "${SKIP_RAY_START:-false}"; then
  echo "SKIP_RAY_START=1: reusing existing Ray cluster at ${RAY_ADDRESS} (multi-node)"
else
  "${RAY_BIN}" "${RAY_START_ARGS[@]}"
fi

"${RAY_BIN}" job submit --address="${RAY_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" "${TRAIN_ENTRYPOINT}" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${MEGATRON_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${COLOCATE_ARGS[@]}" \
  "${TEACHER_ARGS[@]}" \
  2>&1 | tee "${AIEVOBOX_RUN_DIR}/slime.log"
