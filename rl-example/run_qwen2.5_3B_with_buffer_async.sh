#!/usr/bin/env bash

pkill -9 sglang || true
sleep 2
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 2

set -ex

# Rollout buffer DB server (same DB as server script by default)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DB_DIR="${SCRIPT_DIR}/db"
mkdir -p "$DB_DIR"
export AIEVOBOX_DB_URL=${AIEVOBOX_DB_URL:-"sqlite:////${DB_DIR}/trading_rollout.db"}
export ROLLOUT_BUFFER_URL=${ROLLOUT_BUFFER_URL:-"http://127.0.0.1:8889"}

export PYTHONBUFFERED=16

source "/root/slime/scripts/models/qwen2.5-3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/zeocax/Qwen2.5-3B/
   --ref-load /root/zeocax/Qwen2.5-3B_torch_dist/
   --load /root/zeocax/Qwen2.5-3B_slime/
   --save /root/zeocax/Qwen2.5-3B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --rollout-function-path slime_plugins.rollout_buffer.rollout_buffer_example.generate_rollout
   --rollout-buffer-url ${ROLLOUT_BUFFER_URL}
   --prompt-data ${SCRIPT_DIR}/data/dummy.jsonl
   --input-key prompt
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 16
   --n-samples-per-prompt 1
   --rollout-max-response-len 64
   --rollout-temperature 0.7
   --global-batch-size 16
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
   # --use-wandb
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --calculate-per-token-loss
)

# Plan: 2 GPUs for training (actor), 2 GPUs for rollout (engines)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats

RUNTIME_ENV_JSON="{\
  \"env_vars\": {\
    \"PYTHONPATH\": \"/root:${SCRIPT_DIR}:/root/AIEvoBox:/root/Megatron-LM/\",\
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",\
    \"AIEVOBOX_DB_URL\": \"${AIEVOBOX_DB_URL}\"\
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
