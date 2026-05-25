#!/usr/bin/env bash

# Increase file descriptor limit for high concurrency
ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit -n 65536 (current: $(ulimit -n))"

set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PYTHONPATH:-}:/mnt/shared-storage-user/chenxinquan/Safactory"
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-/mnt/shared-storage-user/chenxinquan/Safactory}"
export AIEVOBOX_DB_URL="${AIEVOBOX_DB_URL:-sqlite://${SCRIPT_DIR}/rl.db}"
export ROLLBUF_HOST="${ROLLBUF_HOST:-0.0.0.0}"
export ROLLBUF_PORT="${ROLLBUF_PORT:-18889}"

LOG_ROOT="${AIEVOBOX_ROOT}/logs"
mkdir -p "${LOG_ROOT}"
if [[ -z "${AIEVOBOX_RUN_DIR:-}" && -f "${LOG_ROOT}/.current_run" ]]; then
  export AIEVOBOX_RUN_DIR="$(cat "${LOG_ROOT}/.current_run")"
fi
if [[ -z "${AIEVOBOX_RUN_DIR:-}" ]]; then
  export AIEVOBOX_RUN_DIR="${LOG_ROOT}/$(date +%Y%m%d-%H%M%S)"
  printf '%s\n' "${AIEVOBOX_RUN_DIR}" > "${LOG_ROOT}/.current_run"
fi
mkdir -p "${AIEVOBOX_RUN_DIR}"

echo "Starting Buffer Server..."
echo "  Host: ${ROLLBUF_HOST}"
echo "  Port: ${ROLLBUF_PORT}"
echo "  DB URL: ${AIEVOBOX_DB_URL}"
echo "  Log dir: ${AIEVOBOX_RUN_DIR}"

python3 "${AIEVOBOX_ROOT}/rl/buffer_server.py"
