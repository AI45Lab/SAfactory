#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Cleanup function
cleanup() {
    echo "Stopping services..."
    if [ -n "${BUFFER_SERVER_PID:-}" ]; then
        kill $BUFFER_SERVER_PID 2>/dev/null || true
    fi
    pkill -9 sglang 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Load environment variables
if [ -f "${SCRIPT_DIR}/.env" ]; then
    source "${SCRIPT_DIR}/.env"
fi

export PYTHONPATH="${PYTHONPATH:-}:/root/AIEvoBox"
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-/root/AIEvoBox}"
export AIEVOBOX_DB_URL="${AIEVOBOX_DB_URL:-sqlite:////root/AIEvoBox/rollout.db}"
export ROLLOUT_BUFFER_URL="${ROLLOUT_BUFFER_URL:-http://127.0.0.1:8889}"
export LLM_PROXY_URL="${LLM_PROXY_URL:-http://127.0.0.1:8890}"

echo "=========================================="
echo "Starting AIEvoBox RL Training"
echo "=========================================="

# Step 1: Start Buffer Server in background
echo "[1/2] Starting Buffer Server..."
python3 "${SCRIPT_DIR}/buffer_server.py" &
BUFFER_SERVER_PID=$!
echo "Buffer Server started with PID: $BUFFER_SERVER_PID"

# Wait for Buffer Server to be ready
echo "Waiting for Buffer Server to be ready..."
for i in {1..30}; do
    if curl -s "http://127.0.0.1:8889/health" > /dev/null 2>&1; then
        echo "Buffer Server is ready"
        break
    fi
    sleep 1
done

# Step 2: Start Slime training
echo "[2/2] Starting Slime training..."
"${SCRIPT_DIR}/run_slime.sh"

# Wait for background processes
wait
