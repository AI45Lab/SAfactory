#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
export AIEVOBOX_EXAMPLE_NAME="${AIEVOBOX_EXAMPLE_NAME:-osgym}"

# -------------------------------------------
# Safactory / rollout environment
# -------------------------------------------
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-${REPO_ROOT}}"
export AIEVOBOX_MODE="${AIEVOBOX_MODE:-remote}"
export STORAGE_TYPE="${STORAGE_TYPE:-sqlite}"
export AIEVOBOX_DB_URL="${AIEVOBOX_DB_URL:-sqlite://${AIEVOBOX_ROOT}/rl/examples/osgym/osgym.db}"
export AIEVOBOX_ENV_CONFIG="${AIEVOBOX_ENV_CONFIG:-${AIEVOBOX_ROOT}/env/osgym/os_config.yaml}"

export AIEVOBOX_MAX_STEPS="${AIEVOBOX_MAX_STEPS:-30}"
export AIEVOBOX_MESSAGE_CUT="${AIEVOBOX_MESSAGE_CUT:-100}"
export AIEVOBOX_POOL_SIZE="${AIEVOBOX_POOL_SIZE:-32}"
export AIEVOBOX_MULTIPLIER="${AIEVOBOX_MULTIPLIER:-1.1}"
export AIEVOBOX_ENV_TRANSPORT="${AIEVOBOX_ENV_TRANSPORT:-http}"

export AIEVOBOX_LLM_MAX_CONCURRENCY="${AIEVOBOX_LLM_MAX_CONCURRENCY:-32}"
export AIEVOBOX_LLM_PROXY_WORKERS="${AIEVOBOX_LLM_PROXY_WORKERS:-32}"
export AIEVOBOX_LLM_STARTUP_JITTER_S="${AIEVOBOX_LLM_STARTUP_JITTER_S:-0}"
export AIEVOBOX_TRAININFO_WORKERS="${AIEVOBOX_TRAININFO_WORKERS:-32}"

# -------------------------------------------
# RL rollout policy
# -------------------------------------------
export RL_GROUP_SIZE="${RL_GROUP_SIZE:-8}"
export RL_ROLLOUT_GROUP_BATCH_SIZE="${RL_ROLLOUT_GROUP_BATCH_SIZE:-2}"
export RL_GLOBAL_BATCH_SIZE="${RL_GLOBAL_BATCH_SIZE:-16}"
export RL_EPOCH="${RL_EPOCH:-10}"
export RL_OFF_BY_N="${RL_OFF_BY_N:-2}"

# Sparse-reward cold starts often produce all-zero groups. Set false to avoid
# starving training while the policy is still weak.
export DAPO_filter="${DAPO_filter:-true}"

# -------------------------------------------
# Services
# -------------------------------------------
export BUFFER_SERVER_HOST="${BUFFER_SERVER_HOST:-127.0.0.1}"
export BUFFER_SERVER_PORT="${BUFFER_SERVER_PORT:-18889}"

export LLM_PROXY_HOST="${LLM_PROXY_HOST:-127.0.0.1}"
export LLM_PROXY_PORT="${LLM_PROXY_PORT:-18890}"
export LLM_MAX_LENGTH="${LLM_MAX_LENGTH:-16384}"
export RL_ROLLOUT_MAX_RESPONSE_LEN="${RL_ROLLOUT_MAX_RESPONSE_LEN:-256}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-1.0}"
export LLM_TOP_P="${LLM_TOP_P:-1.0}"
export LLM_PROXY_ENABLE_CONSOLE_LOG="${LLM_PROXY_ENABLE_CONSOLE_LOG:-0}"

export SLIME_ROLLBUF_RESTART_TRAINING="${SLIME_ROLLBUF_RESTART_TRAINING:-True}"

# -------------------------------------------
# Paths
# -------------------------------------------
export LOG_ROOT="${LOG_ROOT:-${AIEVOBOX_ROOT}/logs}"
export SLIME_HOME="${SLIME_HOME:-/root/slime}"
export MEGATRON_HOME="${MEGATRON_HOME:-/root/Megatron-LM}"
export HF_CKPT_DIR="${HF_CKPT_DIR:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
export SAVE_DIR="${SAVE_DIR:-/mnt/shared-storage-user/evobox-share-gpfs2/kangzeyu/slime-checkpoint/Qwen3.5-9B_megatron}"
export WANDB_DIR="${WANDB_DIR:-/mnt/shared-storage-user/evobox-share-gpfs2/kangzeyu/wandb_logs}"
export SGLANG_LOGGING_CONFIG_PATH="${SGLANG_LOGGING_CONFIG_PATH:-}"

# Slime model script. It must define MODEL_ARGS.
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_HOME}/scripts/models/qwen3.5-9B.sh}"

# -------------------------------------------
# Ray / Slime placement
# -------------------------------------------
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
export NUM_GPUS="${NUM_GPUS:-4}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-2}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"

# Set CLEANUP_BEFORE_RUN=false when attaching to an existing Ray/SGLang setup.
export CLEANUP_BEFORE_RUN="${CLEANUP_BEFORE_RUN:-true}"
export KILL_PYTHON_BEFORE_RUN="${KILL_PYTHON_BEFORE_RUN:-false}"

# -------------------------------------------
# Slime checkpoint / rollout args
# -------------------------------------------
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-rl.examples.osgym.slime_generator.generate_rollout}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-rl.examples.osgym.trajectory_rewards.post_process_rewards}"

# -------------------------------------------
# Megatron backend
# -------------------------------------------
export TRAIN_BACKEND="${TRAIN_BACKEND:-megatron}"
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-bridge}"
export TP_SIZE="${TP_SIZE:-2}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-1}"
export ETP_SIZE="${ETP_SIZE:-1}"
export RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
export RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"

# -------------------------------------------
# Training / optimizer
# -------------------------------------------
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.00}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.2}"

export OPTIMIZER="${OPTIMIZER:-adam}"
export LR="${LR:-2e-7}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}"
export ADAM_BETA2="${ADAM_BETA2:-0.98}"

# -------------------------------------------
# W&B
# -------------------------------------------
export USE_WANDB="${USE_WANDB:-true}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime}"
export WANDB_TEAM="${WANDB_TEAM:-}"
export WANDB_GROUP="${WANDB_GROUP:-slime}"
export WANDB_ALWAYS_USE_TRAIN_STEP="${WANDB_ALWAYS_USE_TRAIN_STEP:-true}"

# -------------------------------------------
# SGLang
# -------------------------------------------
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"
export SGLANG_CUDA_GRAPH_BS="${SGLANG_CUDA_GRAPH_BS:-1 2 4 8 16 24 32}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-256}"
export SGLANG_SCHEDULE_CONSERVATIVENESS="${SGLANG_SCHEDULE_CONSERVATIVENESS:-0.8}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
export SGLANG_ENABLE_MIXED_CHUNK="${SGLANG_ENABLE_MIXED_CHUNK:-true}"
export SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-info}"
export SGLANG_LOG_LEVEL_HTTP="${SGLANG_LOG_LEVEL_HTTP:-error}"

# -------------------------------------------
# Runtime environment
# -------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
