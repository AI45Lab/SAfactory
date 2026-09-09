#!/usr/bin/env bash
# =============================================================================
# PatchEval RL training — Qwen3.5-9B (RJOB mode, single 8-GPU node)
# =============================================================================
# Usage:
#   bash rl/examples/patcheval/train_qwen3_5_9b.sh            # foreground
#   RUN_MODE=nohup bash rl/examples/patcheval/train_qwen3_5_9b.sh  # background
#
# Architecture (non-colocate, 8 GPUs):
#   Training : 4 GPUs (TP=2 / PP=1 / CP=1, DP=2)
#   Rollout  : 4 GPUs (4 SGLang engines x 1 GPU each)
#   Buffer   : buffer_server on :18889 (fronts gateway on :8000)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

# --- config ---
export PATCH_EVAL_GENERATED_DIR="${PATCH_EVAL_GENERATED_DIR:-${REPO_ROOT}/rl/examples/patcheval/generated_openhands_exp1_js77}"
export RL_ENV_SH="${RL_ENV_SH:-${REPO_ROOT}/rl/examples/patcheval/env.rjob.qwen3_5_9b.sh}"
export CLEANUP_BEFORE_RUN="${CLEANUP_BEFORE_RUN:-false}"
RUN_MODE="${RUN_MODE:-foreground}"

# --- preflight checks ---
echo "=== Pre-flight ==="
echo "Repo: ${REPO_ROOT}"
echo "Env : ${RL_ENV_SH}"

[[ -f "${RL_ENV_SH}" ]] || { echo "ERROR: env file not found: ${RL_ENV_SH}"; exit 1; }
[[ -d "${PATCH_EVAL_GENERATED_DIR}" ]] || { echo "ERROR: generated dir not found: ${PATCH_EVAL_GENERATED_DIR}"; exit 1; }

# Load env to check key paths
source "${RL_ENV_SH}" 2>/dev/null || true
echo "HF ckpt : ${HF_CKPT_DIR}"
echo "Load dir: ${LOAD_DIR}"
echo "Save dir: ${SAVE_DIR}"
echo "GPUs    : train=${ACTOR_NUM_GPUS_PER_NODE} rollout=${ROLLOUT_NUM_GPUS} (TP=${TP_SIZE} PP=${PP_SIZE} CP=${CP_SIZE})"
echo "Pool    : ${AIEVOBOX_POOL_SIZE}  colocate=${SLIME_COLOCATE}"

[[ -d "${HF_CKPT_DIR}" ]] || { echo "ERROR: HF checkpoint not found: ${HF_CKPT_DIR}"; exit 1; }
[[ -d "${LOAD_DIR}" ]] || { echo "ERROR: Megatron checkpoint not found: ${LOAD_DIR}"; echo "Convert it first with slime/tools/convert_hf_to_torch_dist.py"; exit 1; }
[[ -f "${MODEL_SCRIPT}" ]] || { echo "ERROR: model script not found: ${MODEL_SCRIPT}"; exit 1; }

mkdir -p "${SAVE_DIR}" "${WANDB_DIR:-${REPO_ROOT}/rl/examples/patcheval/wandb_logs}" "${LOG_ROOT}"

# --- clean stale processes (optional) ---
if [[ "${CLEANUP_BEFORE_RUN}" == "true" ]]; then
  echo "Cleaning up stale processes..."
  pkill -9 sglang 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
  sleep 2
fi

# --- start buffer server (background) ---
echo ""
echo "=== Starting buffer server (background) ==="
BUFFER_LOG="${LOG_ROOT}/buffer_server_$(date +%Y%m%d-%H%M%S).log"
PATCH_EVAL_GENERATED_DIR="${PATCH_EVAL_GENERATED_DIR}" \
RL_ENV_SH="${RL_ENV_SH}" \
CLEANUP_BEFORE_RUN=false \
nohup bash rl/run_buffer_server.sh >"${BUFFER_LOG}" 2>&1 &
BUFFER_PID=$!
echo "Buffer server PID: ${BUFFER_PID}"
echo "Buffer log      : ${BUFFER_LOG}"

# Wait for buffer server to be ready on :18889
echo "Waiting for buffer server on :18889..."
for i in $(seq 1 60); do
  if curl -fsS --max-time 2 http://127.0.0.1:18889/health >/dev/null 2>&1 \
     || curl -fsS --max-time 2 http://127.0.0.1:18889/ >/dev/null 2>&1; then
    echo "Buffer server ready."
    break
  fi
  if ! kill -0 "${BUFFER_PID}" 2>/dev/null; then
    echo "ERROR: buffer server exited early. Log:" >&2
    tail -20 "${BUFFER_LOG}" >&2
    exit 1
  fi
  sleep 1
done

# --- start training (foreground or background) ---
echo ""
echo "=== Starting training (slime generator) ==="
TRAIN_LOG="${LOG_ROOT}/train_$(date +%Y%m%d-%H%M%S).log"

start_train() {
  PATCH_EVAL_GENERATED_DIR="${PATCH_EVAL_GENERATED_DIR}" \
  RL_ENV_SH="${RL_ENV_SH}" \
  CLEANUP_BEFORE_RUN=false \
  bash rl/run_slime_generator.sh
}

case "${RUN_MODE}" in
  foreground)
    echo "Training in foreground. Log: ${TRAIN_LOG}"
    start_train 2>&1 | tee "${TRAIN_LOG}"
    ;;
  nohup)
    start_train >"${TRAIN_LOG}" 2>&1 &
    TRAIN_PID=$!
    echo "Training PID: ${TRAIN_PID}"
    echo "Train log  : ${TRAIN_LOG}"
    echo "Monitor   : tail -f ${TRAIN_LOG}"
    ;;
  *)
    echo "ERROR: RUN_MODE must be foreground|nohup (got: ${RUN_MODE})" >&2
    exit 1
    ;;
esac

# --- cleanup on exit (foreground mode) ---
if [[ "${RUN_MODE}" == "foreground" ]]; then
  echo ""
  echo "=== Training exited. Stopping buffer server. ==="
  kill "${BUFFER_PID}" 2>/dev/null || true
  wait "${BUFFER_PID}" 2>/dev/null || true
fi
