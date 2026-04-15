#!/usr/bin/env bash

# Increase file descriptor limit for high concurrency
ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit -n 65536 (current: $(ulimit -n))"

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PYTHONPATH:-}:${AIEVOBOX_ROOT}"

echo "Starting Buffer Server..."
echo "  Host: ${BUFFER_SERVER_HOST}"
echo "  Port: ${BUFFER_SERVER_PORT}"
echo "  DB URL: ${AIEVOBOX_DB_URL}"
echo "  ENV Config: ${AIEVOBOX_ENV_CONFIG}"

if [ ! -f "${AIEVOBOX_ENV_CONFIG}" ]; then
  echo "[math500] ERROR: AIEVOBOX_ENV_CONFIG not found: ${AIEVOBOX_ENV_CONFIG}"
  exit 1
fi

python3 "${AIEVOBOX_ROOT}/rl/buffer_server.py"
