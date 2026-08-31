#!/usr/bin/env bash
#
# =============================================================================
# [DOCKER MODE] PatchEval RL environment
# =============================================================================
# This is the DOCKER mode variant: AIEVOBOX_MODE=docker, agent containers run
# on a single remote Docker daemon (DOCKER_HOST). For the RJob variant (agent
# containers submitted as cluster jobs), see env.rjob.sh in this directory.
# =============================================================================
#
# PatchEval RL settings for rl/run_buffer_server.sh and
# rl/run_slime_generator.sh. Generate PATCH_EVAL_GENERATED_DIR first with
# env/patcheval/generate_full_config.py.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

# Keep this environment aligned with the common Slime launcher contract
# (PYTHON_BIN, RAY_BIN, optimizer, and SGLang defaults), then override all
# Geo3K-specific values below.
source "${REPO_ROOT}/rl/examples/geo3k_vl/env.sh"

: "${PATCH_EVAL_GENERATED_DIR:?Set PATCH_EVAL_GENERATED_DIR to a generated PatchEval config directory}"

export AIEVOBOX_EXAMPLE_NAME="patcheval_qwen3_5_9b"
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-${REPO_ROOT}}"
export AIEVOBOX_MODE="docker"
export PATCH_EVAL_BASELINE="openhands"
export PATCH_EVAL_AGENT_EXPERIMENT="exp1"
export STORAGE_TYPE="${STORAGE_TYPE:-sqlite}"
export AIEVOBOX_DB_URL="${PATCHEVAL_DB_URL:-sqlite:///${AIEVOBOX_ROOT}/rl/examples/patcheval/patcheval_qwen3_5_9b.db}"
export DOCKER_HOST="${DOCKER_HOST:-tcp://100.99.17.62:2376}"
export AIEVOBOX_DOCKER_IMAGE_ARCHIVE_DIR="${PATCH_EVAL_IMAGE_ARCHIVE_DIR:-/mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-images}"
export AIEVOBOX_DOCKER_PULL_POLICY="${AIEVOBOX_DOCKER_PULL_POLICY:-never}"
export AIEVOBOX_AGENT_CONFIG="${PATCH_EVAL_GENERATED_DIR}/patcheval_config.yaml"
export AIEVOBOX_AGENT_START_CONFIG="${PATCH_EVAL_GENERATED_DIR}/patcheval_start.yaml"
# Rule evaluators live in <generated_dir>/<env_name>/rule_evaluator.py; the
# launcher discovers them via --agent_root, so it must point at the generated dir.
export AIEVOBOX_AGENT_ROOT="${PATCH_EVAL_GENERATED_DIR}"
export AIEVOBOX_MAX_STEPS="${PATCHEVAL_MAX_STEPS:-1}"
export AIEVOBOX_ENABLE_EVALUATION="${AIEVOBOX_ENABLE_EVALUATION:-1}"
export AIEVOBOX_POOL_SIZE="${PATCHEVAL_POOL_SIZE:-1}"
export AIEVOBOX_AGENT_START_TIMEOUT_S="${PATCHEVAL_AGENT_START_TIMEOUT_S:-1800}"

export RL_GROUP_SIZE="${RL_GROUP_SIZE:-8}"
export RL_GLOBAL_BATCH_SIZE="${RL_GLOBAL_BATCH_SIZE:-64}"
export RL_ROLLOUT_GROUP_BATCH_SIZE="${RL_ROLLOUT_GROUP_BATCH_SIZE:-8}"
export SLIME_ROLLOUT_BATCH_SIZE="${SLIME_ROLLOUT_BATCH_SIZE:-${RL_ROLLOUT_GROUP_BATCH_SIZE}}"
export SLIME_GLOBAL_BATCH_SIZE="${SLIME_GLOBAL_BATCH_SIZE:-${RL_GLOBAL_BATCH_SIZE}}"
export RL_EPOCH="${RL_EPOCH:-1000}"
export RL_MODEL="${RL_MODEL:-model}"
export RL_API_KEY="${RL_API_KEY:-openai_api_key}"

export BUFFER_SERVER_HOST="${BUFFER_SERVER_HOST:-127.0.0.1}"
export BUFFER_SERVER_PORT="${BUFFER_SERVER_PORT:-18889}"
export LLM_PROXY_HOST="${LLM_PROXY_HOST:-127.0.0.1}"
export LLM_PROXY_PORT="${LLM_PROXY_PORT:-18890}"
export LLM_MAX_LENGTH="${LLM_MAX_LENGTH:-8192}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-1.0}"
export AIEVOBOX_GATEWAY_HOST="${PATCHEVAL_GATEWAY_HOST:-$(hostname -I | awk '{print $1}')}"
export AIEVOBOX_GATEWAY_PORT="${PATCHEVAL_GATEWAY_PORT:-8000}"
export AIEVOBOX_GATEWAY_BASE_URL="http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions"

export SLIME_HOME="${SLIME_HOME:-/root/slime}"
export MEGATRON_HOME="${MEGATRON_HOME:-/root/Megatron-LM}"
export HF_CKPT_DIR="${QWEN3_5_9B_CKPT_DIR:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
export LOAD_DIR="${QWEN3_5_9B_LOAD_DIR:-${HF_CKPT_DIR}}"
export SAVE_DIR="${SAVE_DIR:-${AIEVOBOX_ROOT}/rl/examples/patcheval/checkpoints/Qwen3.5-9B_megatron}"
export WANDB_DIR="${WANDB_DIR:-${AIEVOBOX_ROOT}/rl/examples/patcheval/wandb_logs}"
export LOG_ROOT="${LOG_ROOT:-${AIEVOBOX_ROOT}/logs/patcheval_qwen3_5_9b}"
export MODEL_SCRIPT="${QWEN3_5_9B_MODEL_SCRIPT:-${AIEVOBOX_ROOT}/rl/examples/geo3k_vl/qwen3_5_9b.sh}"
export MODEL_ARGS_ROTARY_BASE=10000000

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS=4
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE=1
export ROLLOUT_NUM_GPUS=3
export ROLLOUT_NUM_GPUS_PER_ENGINE=1
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-${SLIME_HOME}/train.py}"
export ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-rl.slime_generator.generate_rollout}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
export LOSS_MASK_TYPE="qwen3_5"
export TRAIN_BACKEND="${TRAIN_BACKEND:-megatron}"
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-bridge}"
export TP_SIZE=1 PP_SIZE=1 CP_SIZE=1 EP_SIZE=1 ETP_SIZE=1
export RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
export RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-5000}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-true}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-true}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
export LR="${LR:-1e-6}"
export OPTIMIZER="${OPTIMIZER:-adam}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}"
export ADAM_BETA2="${ADAM_BETA2:-0.98}"
export USE_WANDB="${USE_WANDB:-true}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime}"
export WANDB_GROUP="${WANDB_GROUP:-patcheval_qwen3_5_9b}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.45}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"
export SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-info}"
export SGLANG_LOG_LEVEL_HTTP="${SGLANG_LOG_LEVEL_HTTP:-error}"
export CLEANUP_BEFORE_RUN="${CLEANUP_BEFORE_RUN:-true}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED=1
