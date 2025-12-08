#!/usr/bin/env bash
set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -f "${SCRIPT_DIR}/.env" ]; then
    source "${SCRIPT_DIR}/.env"
fi

export PYTHONPATH="${PYTHONPATH:-}:/root/AIEvoBox"
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-/root/AIEvoBox}"
export AIEVOBOX_DB_URL="${AIEVOBOX_DB_URL:-sqlite:////root/AIEvoBox/rollout.db}"
export ROLLBUF_HOST="${ROLLBUF_HOST:-0.0.0.0}"
export ROLLBUF_PORT="${ROLLBUF_PORT:-8889}"

echo "Starting Buffer Server..."
echo "  Host: ${ROLLBUF_HOST}"
echo "  Port: ${ROLLBUF_PORT}"
echo "  DB URL: ${AIEVOBOX_DB_URL}"

python3 "${SCRIPT_DIR}/buffer_server.py"
