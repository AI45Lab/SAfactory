#!/usr/bin/env bash
#
# =============================================================================
# [RJOB MODE] Harbor RL environment
# =============================================================================
# Harbor (vulhub-exploit) RL training settings for rl/run_buffer_server.sh and
# rl/run_slime_generator.sh. Mirrors rl/examples/patcheval/env.rjob.sh but
# targets the Harbor environment (env/harbor/), whose runner.py drives the
# `harbor` CLI to spin up vulnerable services inside a nested-Docker RJob pod
# and scores the agent's exploit via Harbor's verifier.
#
# Key differences vs PatchEval:
#   1. No "generated dir" step. PatchEval needs generate_full_config.py to
#      materialize one env per CVE (77 envs x 300 copies). Harbor ships a
#      single env_name "harbor" with a JSONL dataset of 474 vulhub tasks
#      (env/harbor/datasets/harbor_vulhub_all.jsonl); the buffer server fans
#      the dataset out to per-task rollout groups. So AIEVOBOX_AGENT_CONFIG /
#      AIEVOBOX_AGENT_START_CONFIG point directly at the committed yamls in
#      env/harbor/.
#   2. Agent is `claude-code` (set in env_params of harbor_vulhub_all_config).
#      runner.py rewrites ANTHROPIC_BASE_URL -> the RL gateway URL, so the
#      agent's LLM calls hit the model under training. No per-task image
#      archive is needed; the single harbor runtime image
#      (safactory-harbor-runtime-v0.21.0) is pulled by the RJob pod.
#   3. Reward: Harbor verifier emits a 0/1 reward; rule_evaluator.py
#      (env/harbor/rule_evaluator.py) normalizes it onto SAfactory's 0-10
#      scale. Same binary-reward dynamic as PatchEval -> base model rarely
#      solves a vulhub task, so most groups are all-0 (DAPO_filter stays off
#      by default to avoid stalling the buffer).
#   4. Tasks are LONG: timeout_s=9000 (2.5h) per episode in the vulhub config.
#      AIEVOBOX_GATEWAY_MAX_STEPS and AIEVOBOX_AGENT_START_TIMEOUT_S are raised
#      accordingly vs PatchEval.
#
# Usage:
#   export HARBOR_VARIANT=vulhub_all   # vulhub_all (default) | cvebench | smoke
#   RL_ENV_SH=$this rl/run_buffer_server.sh
# =============================================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

# Self-contained: same infrastructure defaults that patcheval/env.rjob.sh
# inlines from geo3k_vl/env.sh. Only the vars actually consumed by
# run_slime_generator.sh / buffer_server.py / llm_proxy.py / slime_generator.py
# are set here; no geo3k-specific values are carried over.

: "${HARBOR_VARIANT:?Set HARBOR_VARIANT to one of: vulhub_all | cvebench | smoke}"

# --- Harbor agent config selection -----------------------------------------
# Harbor commits its rjob configs directly in env/harbor/ (no generation step).
#   <variant>_config.rjob.yaml  -> AIEVOBOX_AGENT_CONFIG       (env_types/datasets)
#   <variant>_start.rjob.yaml   -> AIEVOBOX_AGENT_START_CONFIG (container/rjob spec)
# rule_evaluator.py lives next to them, so AIEVOBOX_AGENT_ROOT = env/harbor.
HARBOR_ENV_DIR="${REPO_ROOT}/env/harbor"
case "${HARBOR_VARIANT}" in
  vulhub_all)
    HARBOR_CONFIG="${HARBOR_ENV_DIR}/harbor_vulhub_all_config.rjob.yaml"
    HARBOR_START_CONFIG="${HARBOR_ENV_DIR}/harbor_vulhub_start.rjob.yaml"
    HARBOR_DATASET_N=474
    ;;
  cvebench)
    # cvebench smoke (oracle, 1 task). Useful for end-to-end bring-up only.
    HARBOR_CONFIG="${HARBOR_ENV_DIR}/harbor_cvebench_config.rjob.yaml"
    HARBOR_START_CONFIG="${HARBOR_ENV_DIR}/harbor_cvebench_start.rjob.yaml"
    HARBOR_DATASET_N=1
    ;;
  smoke)
    # harbor oracle smoke (1 task). Useful for verifying the rjob pipeline.
    HARBOR_CONFIG="${HARBOR_ENV_DIR}/harbor_config.rjob.yaml"
    HARBOR_START_CONFIG="${HARBOR_ENV_DIR}/harbor_start.rjob.yaml"
    HARBOR_DATASET_N=1
    ;;
  *)
    echo "Unknown HARBOR_VARIANT='${HARBOR_VARIANT}'. Use vulhub_all | cvebench | smoke." >&2
    exit 1
    ;;
esac

if [[ ! -f "${HARBOR_CONFIG}" || ! -f "${HARBOR_START_CONFIG}" ]]; then
  echo "Missing Harbor rjob config for variant '${HARBOR_VARIANT}':" >&2
  echo "  config: ${HARBOR_CONFIG}" >&2
  echo "  start : ${HARBOR_START_CONFIG}" >&2
  exit 1
fi

# --- AIEVOBOX / env wiring -------------------------------------------------
export AIEVOBOX_EXAMPLE_NAME="harbor_qwen3_8_27b"
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-${REPO_ROOT}}"
export AIEVOBOX_MODE="rjob"
export AIEVOBOX_RJOB_CONFIG="${AIEVOBOX_RJOB_CONFIG:-${REPO_ROOT}/config.yaml}"
export STORAGE_TYPE="${STORAGE_TYPE:-sqlite}"
export AIEVOBOX_DB_URL="${HARBOR_DB_URL:-sqlite:///${AIEVOBOX_ROOT}/rl/examples/harbor/harbor_qwen3_8_27b.db}"
# Harbor runtime image is pulled by the RJob pod from the registry; no local
# archive dir is needed (unlike PatchEval's per-CVE tarballs). Keep empty so
# the launcher skips archive mounting.
export AIEVOBOX_DOCKER_IMAGE_ARCHIVE_DIR="${HARBOR_IMAGE_ARCHIVE_DIR:-}"
export AIEVOBOX_DOCKER_PULL_POLICY="${AIEVOBOX_DOCKER_PULL_POLICY:-always}"
export AIEVOBOX_AGENT_CONFIG="${HARBOR_CONFIG}"
export AIEVOBOX_AGENT_START_CONFIG="${HARBOR_START_CONFIG}"
export AIEVOBOX_AGENT_ROOT="${HARBOR_ENV_DIR}"
# Per-task rollout rounds (NOT LLM steps per episode). Each harbor task is
# rolled out this many times per rollout step.
export AIEVOBOX_MAX_STEPS="${HARBOR_MAX_STEPS:-1}"
export AIEVOBOX_ENABLE_EVALUATION="${AIEVOBOX_ENABLE_EVALUATION:-1}"
# RJob scales across the cluster; 16 concurrent episodes keeps SGLang decode
# batches full (same rationale as PatchEval). Override via HARBOR_POOL_SIZE.
export AIEVOBOX_POOL_SIZE="${HARBOR_POOL_SIZE:-16}"
# Harbor episodes are long (timeout_s up to 9000s). Give the RJob pod ample
# startup headroom (nested dockerd + image pull + harbor init).
export AIEVOBOX_AGENT_START_TIMEOUT_S="${HARBOR_AGENT_START_TIMEOUT_S:-3600}"
# Hard cap on LLM steps per episode, enforced by the RL gateway. Vulhub
# exploit tasks need more explore/exploit steps than PatchEval CVE patches;
# 60 is a first guess, tune per throughput (each step is a gateway round-trip).
export AIEVOBOX_GATEWAY_MAX_STEPS="${HARBOR_GATEWAY_MAX_STEPS:-60}"

# --- RL / GRPO batch sizing ------------------------------------------------
# Same shape as PatchEval: 8 trajectories per task, 8 tasks per rollout batch,
# 64 trajectories per training step. With 474 vulhub tasks one epoch covers
# 474/8 ~ 59 rollout batches. Override via HARBOR_* if needed.
export RL_GROUP_SIZE="${HARBOR_GROUP_SIZE:-8}"
export RL_GLOBAL_BATCH_SIZE="${HARBOR_GLOBAL_BATCH_SIZE:-64}"
export RL_ROLLOUT_GROUP_BATCH_SIZE="${HARBOR_ROLLOUT_GROUP_BATCH_SIZE:-8}"
export SLIME_ROLLOUT_BATCH_SIZE="${HARBOR_SLIME_ROLLOUT_BATCH_SIZE:-${RL_ROLLOUT_GROUP_BATCH_SIZE}}"
export SLIME_GLOBAL_BATCH_SIZE="${HARBOR_SLIME_GLOBAL_BATCH_SIZE:-${RL_GLOBAL_BATCH_SIZE}}"
export RL_EPOCH="${HARBOR_EPOCH:-100}"
export RL_MODEL="${RL_MODEL:-model}"
export RL_API_KEY="${RL_API_KEY:-openai_api_key}"

# --- Networking: buffer / proxy / gateway ---------------------------------
export BUFFER_SERVER_HOST="${BUFFER_SERVER_HOST:-127.0.0.1}"
export BUFFER_SERVER_PORT="${BUFFER_SERVER_PORT:-18889}"
export LLM_PROXY_HOST="${LLM_PROXY_HOST:-127.0.0.1}"
export LLM_PROXY_PORT="${LLM_PROXY_PORT:-18890}"
# Vulhub trajectories are long (multi-turn exploit + verification). Raise the
# rollout max response length vs PatchEval so the gateway doesn't truncate
# agent reasoning mid-exploit. 131072 = SGLang default max batch token budget.
export LLM_MAX_LENGTH="${LLM_MAX_LENGTH:-131072}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-1.0}"
# Gateway runs on THIS training pod (started by the buffer server via
# gateway_autostart). Default to this pod's IP so RJob pods can reach it.
export AIEVOBOX_GATEWAY_HOST="${HARBOR_GATEWAY_HOST:-$(hostname -i | awk '{print $1}')}"
export AIEVOBOX_GATEWAY_PORT="${HARBOR_GATEWAY_PORT:-8000}"
export AIEVOBOX_GATEWAY_BASE_URL="http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions"

# --- Slime / Megatron / model ----------------------------------------------
export SLIME_HOME="${SLIME_HOME:-/root/slime}"
export MEGATRON_HOME="${MEGATRON_HOME:-/root/Megatron-LM}"
# Model: Qwen3.8-27B (GQA, num_query_groups=4 -> TP must divide 4, so TP=4).
export HF_CKPT_DIR="${QWEN3_8_27B_CKPT_DIR:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
export LOAD_DIR="${QWEN3_8_27B_LOAD_DIR:-${HF_CKPT_DIR}}"
export SAVE_DIR="${HARBOR_SAVE_DIR:-${AIEVOBOX_ROOT}/rl/examples/harbor/checkpoints/Qwen3.8-27B_megatron}"
export WANDB_DIR="${HARBOR_WANDB_DIR:-${AIEVOBOX_ROOT}/rl/examples/harbor/wandb_logs}"
export LOG_ROOT="${HARBOR_LOG_ROOT:-${AIEVOBOX_ROOT}/logs/harbor_qwen3_8_27b}"
export MODEL_SCRIPT="${QWEN3_8_27B_MODEL_SCRIPT:-/root/slime/scripts/models/qwen3.5-27B.sh}"
export MODEL_ARGS_ROTARY_BASE=10000000

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${HARBOR_NUM_GPUS:-8}"

# Multi-node NCCL: disable IB (ibv_modify_qp fails across some node pairs) and
# force TCP socket transport. Override via NCCL_* env vars if needed.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE="${HARBOR_ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${HARBOR_ROLLOUT_NUM_GPUS:-8}"
export ROLLOUT_NUM_GPUS_PER_ENGINE=1
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-${SLIME_HOME}/train.py}"
export ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-rl.slime_generator.generate_rollout}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
export LOSS_MASK_TYPE="qwen3_5"
export TRAIN_BACKEND="${TRAIN_BACKEND:-megatron}"
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-bridge}"
export TP_SIZE="${HARBOR_TP_SIZE:-4}" PP_SIZE="${HARBOR_PP_SIZE:-1}" CP_SIZE=1 EP_SIZE=1 ETP_SIZE=1
export RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
export RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-64}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
# Trajectory truncation for training: long agent trajectories (60-step vulhub
# exploit) can exceed 50k tokens and OOM the training GPU. Truncates each
# trajectory to the last N tokens for training only; the full trajectory is
# still used for reward/advantage computation during rollout. 0 = disable.
export TRAJ_TRUNCATION_MAX_SEQ_LEN="${TRAJ_TRUNCATION_MAX_SEQ_LEN:-8192}"
# GDN packed-seq monkey-patch (see rl/patches/gdn_packed_seq.py).
export PYTHONPATH="${REPO_ROOT}/rl/patches${PYTHONPATH:+:${PYTHONPATH}}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-true}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-true}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
# DAPO group filter: drop groups where all samples share the same reward.
# Harbor reward is binary (0/1) and the base model rarely solves a vulhub
# task, so most groups are all-0 -> filtering would stall the buffer. Off.
export DAPO_filter="${HARBOR_DAPO_FILTER:-false}"
export LR="${LR:-1e-6}"
export OPTIMIZER="${OPTIMIZER:-adam}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}"
export ADAM_BETA2="${ADAM_BETA2:-0.98}"
# Colocate: training (Megatron TP=8) and inference (sglang) share all 8 GPUs.
export SLIME_COLOCATE="${SLIME_COLOCATE:-false}"
# CPU offload optimizer: moves fp32 master weights + Adam states (~41GB at
# TP=4) to CPU. Critical for 27B on 140GB GPUs.
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
export USE_WANDB="${USE_WANDB:-true}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime}"
export WANDB_GROUP="${WANDB_GROUP:-harbor_qwen3_8_27b}"
# KV cache pool fraction. 0.7 gives enough KV capacity to hold POOL_SIZE=16
# concurrent multi-turn episodes without eviction. Safe on H200 141GB.
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"
export SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-info}"
export SGLANG_LOG_LEVEL_HTTP="${SGLANG_LOG_LEVEL_HTTP:-error}"
export CLEANUP_BEFORE_RUN="${CLEANUP_BEFORE_RUN:-true}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED=1

# =============================================================================
# Infrastructure defaults — inlined from rl/examples/geo3k_vl/env.sh.
# Only vars actually consumed by run_slime_generator.sh / buffer_server.py /
# llm_proxy.py / slime_generator.py. Placed at the end so ${VAR:-default} can
# reference harbor values set earlier (POOL_SIZE, RL_GROUP_SIZE, ports).
# =============================================================================

# --- Ray / Python launcher ---
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export RAY_BIN="${RAY_BIN:-ray}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
export RAY_PORT="${RAY_PORT:-}"
export KILL_PYTHON_BEFORE_RUN="${KILL_PYTHON_BEFORE_RUN:-false}"

# --- Slime train / checkpoint args ---
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export MODEL_ARGS_EXTRA="${MODEL_ARGS_EXTRA:-}"
export REF_LOAD_DIR="${REF_LOAD_DIR:-}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"
export SGLANG_LOGGING_CONFIG_PATH="${SGLANG_LOGGING_CONFIG_PATH:-}"

# --- Optimizer / GRPO extras ---
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.00}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.2}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-false}"

# --- W&B extras ---
export WANDB_TEAM="${WANDB_TEAM:-}"
export WANDB_ALWAYS_USE_TRAIN_STEP="${WANDB_ALWAYS_USE_TRAIN_STEP:-false}"

# --- SGLang extras ---
# Tuned for high-concurrency rollout (POOL_SIZE=16), same as PatchEval.
export SGLANG_CUDA_GRAPH_BS="${SGLANG_CUDA_GRAPH_BS:-1 2 4 8 16 32}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-64}"
export SGLANG_SCHEDULE_CONSERVATIVENESS="${SGLANG_SCHEDULE_CONSERVATIVENESS:-}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
export SGLANG_ENABLE_MIXED_CHUNK="${SGLANG_ENABLE_MIXED_CHUNK:-false}"

# --- LLM proxy / buffer server workers & perf ---
export LLM_TOP_P="${LLM_TOP_P:-1.0}"
export LLM_PROXY_ENABLE_CONSOLE_LOG="${LLM_PROXY_ENABLE_CONSOLE_LOG:-0}"
export AIEVOBOX_LLM_MAX_CONCURRENCY="${AIEVOBOX_LLM_MAX_CONCURRENCY:-${AIEVOBOX_POOL_SIZE}}"
export AIEVOBOX_LLM_PROXY_WORKERS="${AIEVOBOX_LLM_PROXY_WORKERS:-${AIEVOBOX_POOL_SIZE}}"
export AIEVOBOX_LLM_STARTUP_JITTER_S="${AIEVOBOX_LLM_STARTUP_JITTER_S:-0}"
export AIEVOBOX_TRAININFO_WORKERS="${AIEVOBOX_TRAININFO_WORKERS:-${AIEVOBOX_POOL_SIZE}}"
export AIEVOBOX_SQLITE_BULK_INSERT_BATCH_SIZE="${AIEVOBOX_SQLITE_BULK_INSERT_BATCH_SIZE:-256}"
export AIEVOBOX_SQLITE_BULK_INSERT_PAUSE_S="${AIEVOBOX_SQLITE_BULK_INSERT_PAUSE_S:-0.01}"
export AIEVOBOX_BUFFER_INCOMPLETE_GROUP_TTL_SECONDS="${AIEVOBOX_BUFFER_INCOMPLETE_GROUP_TTL_SECONDS:-1800}"
export ROLLBUF_HOST="${ROLLBUF_HOST:-${BUFFER_SERVER_HOST}}"
export ROLLBUF_PORT="${ROLLBUF_PORT:-${BUFFER_SERVER_PORT}}"

# --- Slime rollout-buffer / GRPO filter ---
export SLIME_ROLLBUF_RESTART_TRAINING="${SLIME_ROLLBUF_RESTART_TRAINING:-True}"
export SLIME_N_SAMPLES_PER_PROMPT="${SLIME_N_SAMPLES_PER_PROMPT:-${RL_GROUP_SIZE}}"
export RL_OFF_BY_N="${RL_OFF_BY_N:-0}"

# --- AIEVOBOX env extras ---
export AIEVOBOX_MESSAGE_CUT="${AIEVOBOX_MESSAGE_CUT:-0}"
export AIEVOBOC_MULTIPLIER="${AIEVOBOC_MULTIPLIER:-1.2}"

# --- Runtime ---
# NOTE: expandable_segments:True is incompatible with torch_memory_saver
# (used in colocate mode). Disable it when colocate is on.
if [[ "${SLIME_COLOCATE:-false}" == "true" || "${SLIME_COLOCATE:-false}" == "1" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF=""
  export PYTORCH_ALLOC_CONF=""
else
  # PyTorch >= 2.5 renamed PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF.
  # Set both so old and new versions pick up expandable_segments.
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
fi
