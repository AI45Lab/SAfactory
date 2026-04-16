#!/usr/bin/env bash

# Increase file descriptor limit for high concurrency
ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit -n 65536 (current: $(ulimit -n))"

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

# Force Ray and Python to come from the same runtime environment.
RAY_BIN="${RAY_BIN:-$(command -v ray)}"
if [ -z "${PYTHON_BIN:-}" ]; then
  RAY_BIN_DIR="$(cd -- "$(dirname -- "${RAY_BIN}")" &>/dev/null && pwd)"
  if [ -x "${RAY_BIN_DIR}/python" ]; then
    PYTHON_BIN="${RAY_BIN_DIR}/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

export PYTHONPATH="${PYTHONPATH:-}:${AIEVOBOX_ROOT}"

echo "Starting Buffer Server..."
echo "  Host: ${BUFFER_SERVER_HOST}"
echo "  Port: ${BUFFER_SERVER_PORT}"
echo "  DB URL: ${AIEVOBOX_DB_URL}"
echo "  ENV Config: ${AIEVOBOX_ENV_CONFIG}"
echo "  Ray bin: ${RAY_BIN}"
echo "  Python bin: ${PYTHON_BIN}"

if [ ! -f "${AIEVOBOX_ENV_CONFIG}" ]; then
  echo "[math500] ERROR: AIEVOBOX_ENV_CONFIG not found: ${AIEVOBOX_ENV_CONFIG}"
  exit 1
fi

"${PYTHON_BIN}" -V
"${RAY_BIN}" --version

"${PYTHON_BIN}" "${AIEVOBOX_ROOT}/rl/buffer_server.py"
