#!/usr/bin/env bash

# Increase file descriptor limit for high concurrency
ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit -n 65536 (current: $(ulimit -n))"

# Kill existing processes
pkill -9 sglang || true
sleep 2
ray stop --force || true
pkill -9 ray || true
# Don't kill all python processes to preserve buffer server
pkill -9 python || true
sleep 2

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Load environment variables
if [ -f "${SCRIPT_DIR}/.env" ]; then
    source "${SCRIPT_DIR}/.env"
fi

# Construct URLs from host and port
ROLLOUT_BUFFER_URL="http://${BUFFER_SERVER_HOST}:${BUFFER_SERVER_PORT}"
LLM_PROXY_URL="http://${LLM_PROXY_HOST}:${LLM_PROXY_PORT}"

export PYTHONBUFFERED=16
NUM_GPUS=${NUM_GPUS:-7}

SLIME_HOME=${SLIME_HOME:-/root/slime}
source "${SLIME_HOME}/scripts/models/qwen2.5-7B.sh"
CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen2.5-7B-Instruct
   --ref-load /root/steai-yinzhenyun/Qwen2.5-7B-Instruct_torch_dist
   --load /root/evobox-yinzhenyun/slime/checkpoints/Qwen2.5-7B-Instruct_slime
   --save /root/evobox-yinzhenyun/slime/checkpoints/Qwen2.5-7B-Instruct_slime
   --save-interval 20
)

# 实际上这里很多值都没有使用
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
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.2
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project slime
    --wandb-team aievobox
    --wandb-group slime
    --wandb-dir /root/wandb_logs
    # --wandb-always-use-train-step
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   # --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-log-level error
   --sglang-log-level-http error
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --calculate-per-token-loss
)

# Start Ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 7 --disable-usage-stats

export SGLANG_LOGGING_CONFIG_PATH=${SGLANG_LOGGING_CONFIG_PATH:-"/root/AIEvoBox/rl/sglang_logging.json"}

RUNTIME_ENV_JSON="{\
  \"env_vars\": {\
    \"PYTHONPATH\": \"/root:${SCRIPT_DIR}:/root/AIEvoBox:/root/Megatron-LM/:${SLIME_HOME}\",\
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",\
    \"LLM_PROXY_URL\": \"${LLM_PROXY_URL}\",\
    \"ROLLOUT_BUFFER_URL\": \"${ROLLOUT_BUFFER_URL}\",\
    \"SLIME_OFF_BY_N\": \"${SLIME_OFF_BY_N:-0}\"\
  }\
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${SLIME_HOME}/train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --rollout-num-gpus 6 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]}
