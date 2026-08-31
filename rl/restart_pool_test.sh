#!/usr/bin/env bash
# 重启 patcheval RL run，用于扫 POOL_SIZE 找效率甜点。
# 用法: bash rl/restart_pool_test.sh <POOL_SIZE>
#   例如: bash rl/restart_pool_test.sh 8
#         bash rl/restart_pool_test.sh 24
set -euo pipefail

POOL="${1:-}"
if [[ -z "$POOL" ]]; then
  echo "用法: $0 <POOL_SIZE>   例如: $0 8" >&2
  exit 1
fi

cd /mnt/shared-storage-user/leishanzhe/repo/SAfactory
ENV_SH="rl/examples/patcheval/env.rjob.sh"

# 1) 改 POOL_SIZE 默认值（改 :-后的数字）
sed -i -E "s|(PATCHEVAL_POOL_SIZE:-)[0-9]+|\1${POOL}|" "$ENV_SH"
echo "[restart] AIEVOBOX_POOL_SIZE -> $(grep AIEVOBOX_POOL_SIZE "$ENV_SH" | head -1 | grep -oE ':-[0-9]+')"

# 2) 杀旧进程
echo "[restart] 杀旧进程..."
pkill -9 -f buffer_server        || true
pkill -9 -f run_slime_generator  || true
pkill -9 -f sglang               || true
pkill -9 -f "slime/train.py"     || true
sleep 5

# 3) 启动 buffer_server
export PATCHEVAL_GATEWAY_HOST="$(hostname -I | awk '{print $1}')"
echo "[restart] PATCHEVAL_GATEWAY_HOST=$PATCHEVAL_GATEWAY_HOST"
nohup bash rl/run_buffer_server.sh --env "$ENV_SH" > "/tmp/buffer_pool${POOL}.log" 2>&1 &
echo "[restart] buffer_server 启动 (pid $!) -> /tmp/buffer_pool${POOL}.log"
sleep 10

# 4) 启动 slime generator
nohup bash rl/run_slime_generator.sh --env "$ENV_SH" > "/tmp/slime_pool${POOL}.log" 2>&1 &
echo "[restart] slime generator 启动 (pid $!) -> /tmp/slime_pool${POOL}.log"
echo "[restart] 完成。POOL_SIZE=${POOL}。等 30 分钟后跑: bash rl/collect_pool_metrics.sh ${POOL}"
