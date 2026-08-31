#!/usr/bin/env bash
#
# =============================================================================
# [RJOB MODE] PatchEval RL environment
# =============================================================================
# This is the RJOB mode variant: AIEVOBOX_MODE=rjob, each rollout episode is
# submitted as a cluster job (RJob) via h.pjlab.org.cn, so many episodes can
# run in parallel across the cluster (not limited to a single Docker host).
#
# Prerequisites (vs docker env.sh):
#   1. AIEVOBOX_RJOB_CONFIG must point to a cluster config with valid
#      access_key/secret_key (see ${REPO_ROOT}/config.yaml). The cybergym
#      example reuses that file; fill in the credentials before running.
#   2. Agent configs are the rjob variants:
#        AIEVOBOX_AGENT_CONFIG       -> patcheval_config.rjob.yaml
#        AIEVOBOX_AGENT_START_CONFIG -> patcheval_start.rjob.yaml
#      Both live next to the docker-generated configs in PATCH_EVAL_GENERATED_DIR
#      so they reuse the same datasets/ and per-env rule_evaluator.py.
#   3. The RL gateway (running on this training pod) must be reachable from the
#      RJob pods. PATCHEVAL_GATEWAY_HOST defaults to this pod's IP; confirm it
#      is routable from the cluster namespace (100.x pod IPs usually are).
#   4. RJob pods run DinD (privileged) to load CVE images from the gpfs-mounted
#      archive dir, so privileged=true is set in patcheval_start.rjob.yaml.
#
# Usage:
#   export PATCH_EVAL_GENERATED_DIR=<dir from generate_full_config.py>
#   RL_ENV_SH=$this rl/run_buffer_server.sh
# =============================================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

# Self-contained: previously this sourced rl/examples/geo3k_vl/env.sh for
# infrastructure defaults, but that leaked geo3k-specific values (DAPO_filter=true,
# geo3k db path, qwen3-vl-2b model, RL_GLOBAL_BATCH_SIZE=512, ...) into patcheval.
# Only the infrastructure vars actually consumed by run_slime_generator.sh /
# buffer_server.py / llm_proxy.py / slime_generator.py are inlined at the bottom.

: "${PATCH_EVAL_GENERATED_DIR:?Set PATCH_EVAL_GENERATED_DIR to a generated PatchEval config directory}"

export AIEVOBOX_EXAMPLE_NAME="patcheval_qwen3_8_27b"
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-${REPO_ROOT}}"
export AIEVOBOX_MODE="rjob"
export AIEVOBOX_RJOB_CONFIG="${AIEVOBOX_RJOB_CONFIG:-${REPO_ROOT}/config.yaml}"
export PATCH_EVAL_BASELINE="openhands"
export PATCH_EVAL_AGENT_EXPERIMENT="exp1"
export STORAGE_TYPE="${STORAGE_TYPE:-sqlite}"
export AIEVOBOX_DB_URL="${PATCHEVAL_DB_URL:-sqlite:///${AIEVOBOX_ROOT}/rl/examples/patcheval/patcheval_qwen3_8_27b.db}"
export DOCKER_HOST="${DOCKER_HOST:-tcp://100.99.17.62:2376}"
export AIEVOBOX_DOCKER_IMAGE_ARCHIVE_DIR="${PATCH_EVAL_IMAGE_ARCHIVE_DIR:-/mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-images}"
export AIEVOBOX_DOCKER_PULL_POLICY="${AIEVOBOX_DOCKER_PULL_POLICY:-never}"
# RJob variants of the agent configs (live alongside the docker-generated ones
# so datasets/ and rule_evaluator.py are reused).
export AIEVOBOX_AGENT_CONFIG="${PATCH_EVAL_GENERATED_DIR}/patcheval_config.rjob.yaml"
export AIEVOBOX_AGENT_START_CONFIG="${PATCH_EVAL_GENERATED_DIR}/patcheval_start.rjob.yaml"
export AIEVOBOX_AGENT_ROOT="${PATCH_EVAL_GENERATED_DIR}"
# Per-task rollout rounds (NOT LLM steps per episode). Each CVE task is rolled
# out this many times per rollout step.
export AIEVOBOX_MAX_STEPS="${PATCHEVAL_MAX_STEPS:-1}"
export AIEVOBOX_ENABLE_EVALUATION="${AIEVOBOX_ENABLE_EVALUATION:-1}"
# RJob can scale across the cluster; default higher than docker's 1.
# Raised from 4 to 16 to fix the rollout throughput bottleneck (SGLang was
# only seeing #running-req: 1, ~56 tok/s, because only a few agent episodes
# were in flight). 16 concurrent episodes gives the LLM proxy enough
# in-flight requests to keep SGLang's decode batches full. The derived
# concurrency vars (AIEVOBOX_LLM_MAX_CONCURRENCY, AIEVOBOX_LLM_PROXY_WORKERS,
# AIEVOBOX_TRAININFO_WORKERS) auto-track this via ${AIEVOBOX_POOL_SIZE}.
# Override via PATCHEVAL_POOL_SIZE if cluster capacity is tight.
# NOTE: 实验扫 POOL_SIZE=8/16/24/32 找效率甜点。当前测试值=16。
export AIEVOBOX_POOL_SIZE="${PATCHEVAL_POOL_SIZE:-16}"
export AIEVOBOX_AGENT_START_TIMEOUT_S="${PATCHEVAL_AGENT_START_TIMEOUT_S:-2400}"
# Hard cap on LLM steps per episode, enforced by the RL gateway
# (see rl/gateway_autostart.py). -1 = unlimited. Set >=0 to stop runaway
# agent rollouts (e.g. OpenHands looping 200+ steps without finishing).
# 12 was too few for CVE-fix tasks (binary reward → 0 solve → 0 RL signal).
# 40 gives the model a real shot at explore+edit+test while keeping
# throughput workable (~1.5hr/train step, ~6 days/100 epoch). max_tokens stays
# at 6144 (gateway default) — not lowered, per user choice.
export AIEVOBOX_GATEWAY_MAX_STEPS="${PATCHEVAL_GATEWAY_MAX_STEPS:-40}"

# NOTE: geo3k_vl/env.sh (sourced above) already sets these to its own defaults
# (e.g. RL_GLOBAL_BATCH_SIZE=512, RL_ROLLOUT_GROUP_BATCH_SIZE=64). Using
# ${VAR:-default} here would keep geo3k's values, so we override
# unconditionally. Override via PATCHEVAL_* if needed.
export RL_GROUP_SIZE="${PATCHEVAL_GROUP_SIZE:-8}"
export RL_GLOBAL_BATCH_SIZE="${PATCHEVAL_GLOBAL_BATCH_SIZE:-64}"
export RL_ROLLOUT_GROUP_BATCH_SIZE="${PATCHEVAL_ROLLOUT_GROUP_BATCH_SIZE:-8}"
export SLIME_ROLLOUT_BATCH_SIZE="${PATCHEVAL_SLIME_ROLLOUT_BATCH_SIZE:-${RL_ROLLOUT_GROUP_BATCH_SIZE}}"
export SLIME_GLOBAL_BATCH_SIZE="${PATCHEVAL_SLIME_GLOBAL_BATCH_SIZE:-${RL_GLOBAL_BATCH_SIZE}}"
# RL_EPOCH=100 training rounds. Override via PATCHEVAL_EPOCH / NUM_ROLLOUT
# env vars if needed.
export RL_EPOCH="${PATCHEVAL_EPOCH:-100}"
export RL_MODEL="${RL_MODEL:-model}"
export RL_API_KEY="${RL_API_KEY:-openai_api_key}"

export BUFFER_SERVER_HOST="${BUFFER_SERVER_HOST:-127.0.0.1}"
export BUFFER_SERVER_PORT="${BUFFER_SERVER_PORT:-18889}"
export LLM_PROXY_HOST="${LLM_PROXY_HOST:-127.0.0.1}"
export LLM_PROXY_PORT="${LLM_PROXY_PORT:-18890}"
export LLM_MAX_LENGTH="${LLM_MAX_LENGTH:-131072}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-1.0}"
# Gateway runs on THIS training pod (started by the buffer server via
# gateway_autostart). Default to this pod's IP so it always points at the live
# gateway, not a stale hardcoded IP. Override via PATCHEVAL_GATEWAY_HOST.
export AIEVOBOX_GATEWAY_HOST="${PATCHEVAL_GATEWAY_HOST:-$(hostname -I | awk '{print $1}')}"
export AIEVOBOX_GATEWAY_PORT="${PATCHEVAL_GATEWAY_PORT:-8000}"
export AIEVOBOX_GATEWAY_BASE_URL="http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions"

export SLIME_HOME="${SLIME_HOME:-/root/slime}"
export MEGATRON_HOME="${MEGATRON_HOME:-/root/Megatron-LM}"
# Model: Qwen3.8-27B (same architecture as Qwen3.5-27B, uses qwen3.5-27B.sh spec).
# Override via QWEN3_8_27B_CKPT_DIR / PATCHEVAL_* if needed.
export HF_CKPT_DIR="${QWEN3_8_27B_CKPT_DIR:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
export LOAD_DIR="${QWEN3_8_27B_LOAD_DIR:-${HF_CKPT_DIR}}"
# Override geo3k defaults unconditionally (geo3k sets these to its own paths).
export SAVE_DIR="${PATCHEVAL_SAVE_DIR:-${AIEVOBOX_ROOT}/rl/examples/patcheval/checkpoints/Qwen3.8-27B_megatron}"
export WANDB_DIR="${PATCHEVAL_WANDB_DIR:-${AIEVOBOX_ROOT}/rl/examples/patcheval/wandb_logs}"
export LOG_ROOT="${PATCHEVAL_LOG_ROOT:-${AIEVOBOX_ROOT}/logs/patcheval_qwen3_8_27b}"
export MODEL_SCRIPT="${QWEN3_8_27B_MODEL_SCRIPT:-/root/slime/scripts/models/qwen3.5-27B.sh}"
export MODEL_ARGS_ROTARY_BASE=10000000

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${PATCHEVAL_NUM_GPUS:-8}"
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE="${PATCHEVAL_ACTOR_NUM_GPUS_PER_NODE:-4}"
# Inference GPUs for sglang. Make overridable so the capacity experiment can
# sweep env/pool vs inference-GPU ratios. Must be <= NUM_GPUS.
export ROLLOUT_NUM_GPUS="${PATCHEVAL_ROLLOUT_NUM_GPUS:-4}"
export ROLLOUT_NUM_GPUS_PER_ENGINE=1
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-${SLIME_HOME}/train.py}"
export ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-rl.slime_generator.generate_rollout}"
# Debug-friendly: 300 rollout iterations is too many for a debug run.
export NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
export LOSS_MASK_TYPE="qwen3_5"
export TRAIN_BACKEND="${TRAIN_BACKEND:-megatron}"
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-bridge}"
export TP_SIZE="${PATCHEVAL_TP_SIZE:-4}" PP_SIZE="${PATCHEVAL_PP_SIZE:-1}" CP_SIZE=1 EP_SIZE=1 ETP_SIZE=1
export RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
export RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-5000}"
# bshd (padding) instead of thd (packing): Megatron GDN does not support packed
# sequences (NotImplementedError). bshd pads sequences in a micro-batch to equal
# length instead of packing them into one stream, so packed_seq_params is None
# and GDN's forward never hits the raise. Requires fixed micro-batch-size (no
# dynamic batch size). Costs some compute on padding tokens.
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-false}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export QKV_FORMAT="${QKV_FORMAT:-bshd}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-true}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
# DAPO group filter: drop groups where all samples share the same reward
# (zero advantage, no learning signal). For patcheval, base model rarely
# solves CVEs, so most groups are all-0 and get filtered → buffer never fills
# → pipeline stalls. Default off; flip with PATCHEVAL_DAPO_FILTER=true.
export DAPO_filter="${PATCHEVAL_DAPO_FILTER:-false}"
export LR="${LR:-1e-6}"
export OPTIMIZER="${OPTIMIZER:-adam}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}"
export ADAM_BETA2="${ADAM_BETA2:-0.98}"
export USE_WANDB="${USE_WANDB:-true}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime}"
export WANDB_GROUP="${WANDB_GROUP:-patcheval_qwen3_5_9b}"
# KV cache pool as a fraction of GPU memory (after weights). This is the
# primary throughput lever for multi-turn agent rollouts: each concurrently
# decoding request must keep its prompt+history KV resident, so KV capacity
# directly caps parallel decode (#running-req). At 0.45 the KV hits ~97% with
# 4 sessions/engine, forcing eviction (cached=0 thrashing) and queueing
# (waiting for decode to free KV, median ~26s/request). 0.6 gives +33% KV
# capacity, enough to hold 4 sessions/engine without eviction so all 4 decode
# in parallel — no queue, ~100% prefix reuse. Safe on H200 141GB: 27B weights
# ~54GB + KV 0.6*~87GB-free ≈ 52GB ≈ 106GB < 141GB. Override via env var.
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
# llm_proxy.py / slime_generator.py. geo3k-specific paths (geo3k db, geo3k
# config, qwen3-vl-2b model) are intentionally NOT carried over; patcheval
# overrides above already set those. Placed at the end so ${VAR:-default} can
# reference patcheval values set earlier (POOL_SIZE, RL_GROUP_SIZE, ports).
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
# Tuned for high-concurrency rollout (POOL_SIZE=16). These three together let
# SGLang actually batch dozens of decode requests instead of running 1 at a
# time (the #running-req:1 symptom seen before):
#   - MAX_RUNNING_REQUESTS: hard cap on the running decode batch. 64 gives
#     headroom over POOL_SIZE=16 (multi-turn episodes overlap).
#   - CUDA_GRAPH_BS: capture graphs for the batch sizes we expect to hit, so
#     decode doesn't fall back to the slow non-graph path on shape change
#     (fixes the first-step drop to 6.5 tok/s seen in the logs).
#   - CHUNKED_PREFILL_SIZE: split long prefills into 8192-token chunks so a
#     single long prompt can't starve the running decode batch.
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
