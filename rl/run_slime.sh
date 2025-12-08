#!/usr/bin/env bash

# Kill existing processes
pkill -9 sglang || true
sleep 2
ray stop --force || true
pkill -9 ray || true
# Don't kill all python processes to preserve buffer server
# pkill -9 python || true
sleep 2

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Load environment variables
if [ -f "${SCRIPT_DIR}/.env" ]; then
    source "${SCRIPT_DIR}/.env"
fi

# Rollout buffer configuration
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-/root/AIEvoBox}"
export AIEVOBOX_DB_URL="${AIEVOBOX_DB_URL:-sqlite:////root/AIEvoBox/rollout.db}"
export ROLLOUT_BUFFER_URL="${ROLLOUT_BUFFER_URL:-http://127.0.0.1:8889}"
export LLM_PROXY_URL="${LLM_PROXY_URL:-http://127.0.0.1:8890}"

export PYTHONBUFFERED=16

# Load model configuration
source "/root/slime/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3-8B
   --ref-load /root/steai-yinzhenyun/Qwen3-8B_torch_dist
   --load /root/steai-yinzhenyun/Qwen3-8B_slime
   --save /root/steai-yinzhenyun/Qwen3-8B_slime
   --save-interval 20
)

ROLLOUT_ARGS=(
   --rollout-function-path rl.rollout_buffer_slime.generate_rollout
   --rollout-buffer-url ${ROLLOUT_BUFFER_URL}
   --prompt-data ${SCRIPT_DIR}/dummy.jsonl
   --input-key prompt
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 256
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-5}
   --rollout-max-response-len 64
   --rollout-temperature 0.7
   --global-batch-size 256
   --loss-mask-type qwen
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
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
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
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
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats

export SGLANG_LOGGING_CONFIG_PATH=${SGLANG_LOGGING_CONFIG_PATH:-"/root/AIEvoBox/rl/sglang_logging.json"}

RUNTIME_ENV_JSON="{\
  \"env_vars\": {\
    \"PYTHONPATH\": \"/root:${SCRIPT_DIR}:/root/AIEvoBox:/root/Megatron-LM/\",\
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",\
    \"LLM_PROXY_URL\": \"${LLM_PROXY_URL}\",\
    \"ROLLOUT_BUFFER_URL\": \"${ROLLOUT_BUFFER_URL}\"\
  }\
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /root/slime/train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
