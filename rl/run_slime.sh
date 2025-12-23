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

export PYTHONBUFFERED=16
NUM_GPUS=${NUM_GPUS:-8}

CKPT_ARGS=(
   --hf-checkpoint /root/steai-yinzhenyun/Qwen3-VL-4B-Instruct
   --ref-load /root/steai-yinzhenyun/Qwen3-VL-4B-Instruct
   --load /root/steai-yinzhenyun/Qwen3-VL-4B-Instruct_fsdp_slime
   --save /root/steai-yinzhenyun/Qwen3-VL-4B-Instruct_fsdp_slime
   --save-interval 10
)

ROLLOUT_ARGS=(
   --rollout-function-path rl.rollout_buffer_slime.generate_rollout
   --rollout-buffer-url ${ROLLOUT_BUFFER_URL}
   --prompt-data ${SCRIPT_DIR}/dummy_vl.jsonl
   --input-key prompt
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size ${SLIME_ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${SLIME_N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len 64
   --rollout-temperature 1.0
   --global-batch-size ${SLIME_GLOBAL_BATCH_SIZE}
   --loss-mask-type qwen
)

PERF_ARGS=(
   --balance-data
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

FSDP_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size $((512 * 1024 * 1024))
   --gradient-checkpointing
   --sglang-attention-backend fa3
   # Use PyTorch SDPA instead of FlashAttention3 to avoid missing FA3 dependency errors.
   --attn-implementation flash_attention_3
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
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
    --wandb-always-use-train-step
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.9
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-log-level error
   --sglang-log-level-http error
)

MISC_ARGS=()

# Start Ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats

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
   -- python3 /root/zeocax/pip-e/slime-vl/train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${FSDP_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   --multimodal-keys '{"image": "images"}' \
   --disable-rewards-normalization
