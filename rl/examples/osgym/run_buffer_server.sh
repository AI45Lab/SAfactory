#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

ulimit -n 65536 2>/dev/null || \
  echo "Warning: could not set ulimit -n 65536 (current: $(ulimit -n))"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${AIEVOBOX_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${AIEVOBOX_DB_URL}" == sqlite://* ]]; then
  DB_FILE="${AIEVOBOX_DB_URL#sqlite://}"
  DB_FILE="${DB_FILE%%\?*}"
  mkdir -p -- "$(dirname -- "${DB_FILE}")"
fi

mkdir -p "${LOG_ROOT}"
if [[ -z "${AIEVOBOX_RUN_DIR:-}" ]]; then
  export AIEVOBOX_RUN_DIR="${LOG_ROOT}/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "${AIEVOBOX_RUN_DIR}"
printf '%s\n' "${AIEVOBOX_RUN_DIR}" > "${LOG_ROOT}/.current_run"

echo "Starting OSGym rollout buffer server"
echo "  Python: ${PYTHON_BIN}"
echo "  Listen: 0.0.0.0:${BUFFER_SERVER_PORT}"
echo "  Client URL: http://${BUFFER_SERVER_HOST}:${BUFFER_SERVER_PORT}"
echo "  DB URL: ${AIEVOBOX_DB_URL}"
echo "  Run dir: ${AIEVOBOX_RUN_DIR}"

exec "${PYTHON_BIN}" "${AIEVOBOX_ROOT}/rl/buffer_server.py"
