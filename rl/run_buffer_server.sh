#!/usr/bin/env bash

set -euo pipefail

ulimit -n 65536 2>/dev/null || echo "Warning: could not set ulimit -n 65536 (current: $(ulimit -n))"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXAMPLE_NAME="${EXAMPLE_NAME:-${AIEVOBOX_EXAMPLE_NAME:-rl}}"

export PYTHONPATH="${AIEVOBOX_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOG_ROOT}"
if [[ -z "${AIEVOBOX_RUN_DIR:-}" && -f "${LOG_ROOT}/.current_run" ]]; then
  export AIEVOBOX_RUN_DIR="$(cat "${LOG_ROOT}/.current_run")"
fi
if [[ -z "${AIEVOBOX_RUN_DIR:-}" ]]; then
  export AIEVOBOX_RUN_DIR="${LOG_ROOT}/$(date +%Y%m%d-%H%M%S)"
  printf '%s\n' "${AIEVOBOX_RUN_DIR}" > "${LOG_ROOT}/.current_run"
fi
mkdir -p "${AIEVOBOX_RUN_DIR}"

echo "Starting ${EXAMPLE_NAME} rollout buffer server"
echo "  Python: ${PYTHON_BIN}"
echo "  Listen: 0.0.0.0:${BUFFER_SERVER_PORT}"
echo "  Client URL: http://${BUFFER_SERVER_HOST}:${BUFFER_SERVER_PORT}"
echo "  DB URL: ${AIEVOBOX_DB_URL}"
echo "  Run dir: ${AIEVOBOX_RUN_DIR}"

exec "${PYTHON_BIN}" "${AIEVOBOX_ROOT}/rl/buffer_server.py"
