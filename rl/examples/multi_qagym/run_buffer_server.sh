#!/usr/bin/env bash

ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit -n 65536 (current: $(ulimit -n))"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${AIEVOBOX_ROOT}:${PYTHONPATH:-}"
export ROLLBUF_HOST="0.0.0.0"
export ROLLBUF_PORT="${BUFFER_SERVER_PORT}"

echo "Starting Multi-QAGym Buffer Server..."
echo "  Port: ${BUFFER_SERVER_PORT}"
echo "  DB URL: ${AIEVOBOX_DB_URL}"
echo "  Env config: ${AIEVOBOX_ENV_CONFIG}"
echo "  Policy config: ${AIEVOBOX_POLICY_CONFIG}"

"${PYTHON_BIN}" "${AIEVOBOX_ROOT}/rl/buffer_server.py"
