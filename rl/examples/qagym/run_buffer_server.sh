#!/usr/bin/env bash

# Increase file descriptor limit for high concurrency
ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit -n 65536 (current: $(ulimit -n))"

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PYTHONPATH:-}:${AIEVOBOX_ROOT}"
export ROLLBUF_HOST="${ROLLBUF_HOST:-0.0.0.0}"
export ROLLBUF_PORT="${ROLLBUF_PORT:-${BUFFER_SERVER_PORT}}"

echo "Starting QAGym Buffer Server..."
echo "  Host: ${ROLLBUF_HOST}"
echo "  Port: ${BUFFER_SERVER_PORT}"
echo "  DB URL: ${AIEVOBOX_DB_URL}"
echo "  Env config: ${AIEVOBOX_ENV_CONFIG}"

python3 "${AIEVOBOX_ROOT}/rl/buffer_server.py"
