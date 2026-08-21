SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
export AIEVOBOX_EXAMPLE_NAME="${AIEVOBOX_EXAMPLE_NAME:-math500}"

# -------------------------------------------
# Safactory / rollout environment
# -------------------------------------------
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-${REPO_ROOT}}"
export AIEVOBOX_MODE="${AIEVOBOX_MODE:-local}"
export STORAGE_TYPE="${STORAGE_TYPE:-sqlite}"
export AIEVOBOX_DB_URL="${AIEVOBOX_DB_URL:-sqlite://${AIEVOBOX_ROOT}/rl/examples/math500/rl.db}"
export AIEVOBOX_ENV_CONFIG="${AIEVOBOX_ENV_CONFIG:-${AIEVOBOX_ROOT}/env/math500_text/math500_text_env_configs.yaml}"

export AIEVOBOX_MAX_STEPS="${AIEVOBOX_MAX_STEPS:-1}"
export AIEVOBOX_MESSAGE_CUT="${AIEVOBOX_MESSAGE_CUT:-0}"
export AIEVOBOX_POOL_SIZE="${AIEVOBOX_POOL_SIZE:-256}"
export AIEVOBOX_MULTIPLIER="${AIEVOBOX_MULTIPLIER:-1.2}"
export AIEVOBOX_ENV_TRANSPORT="${AIEVOBOX_ENV_TRANSPORT:-inproc}"

export AIEVOBOX_LLM_MAX_CONCURRENCY="${AIEVOBOX_LLM_MAX_CONCURRENCY:-${AIEVOBOX_POOL_SIZE}}"
export AIEVOBOX_LLM_PROXY_WORKERS="${AIEVOBOX_LLM_PROXY_WORKERS:-${AIEVOBOX_POOL_SIZE}}"
export AIEVOBOX_LLM_STARTUP_JITTER_S="${AIEVOBOX_LLM_STARTUP_JITTER_S:-0}"
export AIEVOBOX_TRAININFO_WORKERS="${AIEVOBOX_TRAININFO_WORKERS:-${AIEVOBOX_POOL_SIZE}}"

export AIEVOBOX_SQLITE_BULK_INSERT_BATCH_SIZE="${AIEVOBOX_SQLITE_BULK_INSERT_BATCH_SIZE:-256}"
export AIEVOBOX_SQLITE_BULK_INSERT_PAUSE_S="${AIEVOBOX_SQLITE_BULK_INSERT_PAUSE_S:-0.01}"
export AIEVOBOX_BUFFER_INCOMPLETE_GROUP_TTL_SECONDS="${AIEVOBOX_BUFFER_INCOMPLETE_GROUP_TTL_SECONDS:-1800}"

# -------------------------------------------
# RL rollout policy
# -------------------------------------------
export RL_GROUP_SIZE="${RL_GROUP_SIZE:-8}"
export RL_ROLLOUT_GROUP_BATCH_SIZE="${RL_ROLLOUT_GROUP_BATCH_SIZE:-64}"
export RL_GLOBAL_BATCH_SIZE="${RL_GLOBAL_BATCH_SIZE:-512}"
export RL_EPOCH="${RL_EPOCH:-10}"
export RL_OFF_BY_N="${RL_OFF_BY_N:-1}"
export DAPO_filter="${DAPO_filter:-false}"

# -------------------------------------------
# Services
# -------------------------------------------
export BUFFER_SERVER_HOST="${BUFFER_SERVER_HOST:-127.0.0.1}"
export BUFFER_SERVER_PORT="${BUFFER_SERVER_PORT:-18889}"

export LLM_PROXY_HOST="${LLM_PROXY_HOST:-127.0.0.1}"
export LLM_PROXY_PORT="${LLM_PROXY_PORT:-18890}"
export LLM_MAX_LENGTH="${LLM_MAX_LENGTH:-64}"
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
export HF_CKPT_DIR="${HF_CKPT_DIR:-/mnt/shared-storage-user/evobox-share/hf-hub/models--Qwen2.5-Math-7B-Instruct/snapshots/c1d08860689aca85d3fa9402334cac49ecde6037}"
export LOAD_DIR="${LOAD_DIR:-${HF_CKPT_DIR}}"
export REF_LOAD_DIR="${REF_LOAD_DIR:-${HF_CKPT_DIR}}"
export SAVE_DIR="${SAVE_DIR:-/mnt/shared-storage-user/evobox-share-gpfs2/leishanzhe/model/checkpoints/opd/Qwen2.5-Math-7B-Instruct/$(date +%Y%m%d_%H%M)}"
export WANDB_DIR="${WANDB_DIR:-/root/wandb_logs}"
export SGLANG_LOGGING_CONFIG_PATH="${SGLANG_LOGGING_CONFIG_PATH:-}"

export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_HOME}/scripts/models/qwen2.5-7B.sh}"
export MODEL_ARGS_ROTARY_BASE="${MODEL_ARGS_ROTARY_BASE:-}"
export MODEL_ARGS_EXTRA="${MODEL_ARGS_EXTRA:---rotary-base 10000}"

# -------------------------------------------
# Ray / Slime placement
# -------------------------------------------
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export RAY_BIN="${RAY_BIN:-ray}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
export RAY_PORT="${RAY_PORT:-6379}"
export NUM_GPUS="${NUM_GPUS:-8}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-6}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"

export CLEANUP_BEFORE_RUN="${CLEANUP_BEFORE_RUN:-true}"
export KILL_PYTHON_BEFORE_RUN="${KILL_PYTHON_BEFORE_RUN:-false}"

# -------------------------------------------
# Slime checkpoint / rollout args
# -------------------------------------------
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-${SLIME_HOME}/train_async.py}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-rl.slime_generator.generate_rollout}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
export LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-qwen}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"

# -------------------------------------------
# Megatron backend
# -------------------------------------------
export TRAIN_BACKEND="${TRAIN_BACKEND:-megatron}"
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-bridge}"
export TP_SIZE="${TP_SIZE:-1}"
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
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-true}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-false}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-5000}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-true}"

export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.00}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.2}"

export OPTIMIZER="${OPTIMIZER:-adam}"
export LR="${LR:-1e-6}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}"
export ADAM_BETA2="${ADAM_BETA2:-0.98}"

# -------------------------------------------
# OPD teacher / reward model
# -------------------------------------------
export USE_OPD="${USE_OPD:-true}"
export OPD_TYPE="${OPD_TYPE:-sglang}"
export OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
export TEACHER_URL="${TEACHER_URL:-http://100.99.167.245:30172/generate}"
export OPD_TEACHER_MAX_CONCURRENCY="${OPD_TEACHER_MAX_CONCURRENCY:-16}"
export OPD_TEACHER_TIMEOUT_SECONDS="${OPD_TEACHER_TIMEOUT_SECONDS:-60}"

# -------------------------------------------
# W&B
# -------------------------------------------
export USE_WANDB="${USE_WANDB:-true}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-evobox}"
export WANDB_TEAM="${WANDB_TEAM:-ben}"
export WANDB_GROUP="${WANDB_GROUP:-slime-opd-math500}"
export WANDB_ALWAYS_USE_TRAIN_STEP="${WANDB_ALWAYS_USE_TRAIN_STEP:-false}"

# -------------------------------------------
# SGLang
# -------------------------------------------
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"
export SGLANG_CUDA_GRAPH_BS="${SGLANG_CUDA_GRAPH_BS:-}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}"
export SGLANG_SCHEDULE_CONSERVATIVENESS="${SGLANG_SCHEDULE_CONSERVATIVENESS:-}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-}"
export SGLANG_ENABLE_MIXED_CHUNK="${SGLANG_ENABLE_MIXED_CHUNK:-false}"
export SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-error}"
export SGLANG_LOG_LEVEL_HTTP="${SGLANG_LOG_LEVEL_HTTP:-error}"

# -------------------------------------------
# Runtime environment
# -------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
