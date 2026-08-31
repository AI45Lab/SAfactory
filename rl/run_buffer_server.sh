#!/usr/bin/env bash

set -euo pipefail

ulimit -n 65536 2>/dev/null || echo "Warning: could not set ulimit -n 65536 (current: $(ulimit -n))"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

ENV_SH="${RL_ENV_SH:-}"
if [[ "${1:-}" == "--env" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Usage: $0 [--env /path/to/env.sh]" >&2
    exit 1
  fi
  ENV_SH="$2"
  shift 2
elif [[ $# -gt 0 && "${1}" != -* ]]; then
  ENV_SH="$1"
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Unexpected argument(s): $*" >&2
  echo "Usage: $0 [--env /path/to/env.sh]" >&2
  exit 1
fi

if [[ -n "${ENV_SH}" ]]; then
  if [[ ! -f "${ENV_SH}" ]]; then
    echo "Missing env file: ${ENV_SH}" >&2
    exit 1
  fi
  source "${ENV_SH}"
elif [[ -z "${AIEVOBOX_ROOT:-}" ]]; then
  echo "No RL env loaded. Source an env.sh first, or run with RL_ENV_SH=/path/to/env.sh." >&2
  echo "Example: RL_ENV_SH=${SCRIPT_DIR}/examples/geo3k_vl/env.sh ${SCRIPT_DIR}/run_buffer_server.sh" >&2
  exit 1
fi

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if is_true "${AIEVOBOX_RESET_SQLITE_DB:-false}"; then
  case "${AIEVOBOX_DB_URL:-}" in
    sqlite:///*)
      db_path="${AIEVOBOX_DB_URL#sqlite:///}"
      rm -f -- "${db_path}" "${db_path}-wal" "${db_path}-shm"
      echo "Removed SQLite DB for fresh rollout: ${db_path}"
      ;;
    *)
      echo "AIEVOBOX_RESET_SQLITE_DB requires a sqlite:/// DB URL" >&2
      exit 1
      ;;
  esac
fi

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

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

require_dir "${AIEVOBOX_ROOT}" "AIEVOBOX_ROOT"
require_file "${AIEVOBOX_ROOT}/rl/buffer_server.py" "buffer server entrypoint"
if [[ -n "${AIEVOBOX_AGENT_CONFIG:-}" ]]; then
  require_file "${AIEVOBOX_AGENT_CONFIG}" "AIEVOBOX_AGENT_CONFIG"
fi
if [[ -n "${AIEVOBOX_AGENT_START_CONFIG:-}" ]]; then
  require_file "${AIEVOBOX_AGENT_START_CONFIG}" "AIEVOBOX_AGENT_START_CONFIG"
fi

echo "Starting ${AIEVOBOX_EXAMPLE_NAME:-rl} rollout buffer server"
echo "  Python: ${PYTHON_BIN}"
echo "  Mode: ${AIEVOBOX_MODE}"
echo "  Listen: 0.0.0.0:${BUFFER_SERVER_PORT}"
echo "  Client URL: http://${BUFFER_SERVER_HOST}:${BUFFER_SERVER_PORT}"
echo "  DB URL: ${AIEVOBOX_DB_URL}"
echo "  Agent config: ${AIEVOBOX_AGENT_CONFIG:-<agent-root:${AIEVOBOX_AGENT_ROOT:-env}>}"
echo "  Agent start config: ${AIEVOBOX_AGENT_START_CONFIG:-<auto>}"
echo "  Gateway: ${AIEVOBOX_GATEWAY_BASE_URL}"
echo "  Run dir: ${AIEVOBOX_RUN_DIR}"

cd "${AIEVOBOX_ROOT}"

"${PYTHON_BIN}" -V
exec "${PYTHON_BIN}" "${AIEVOBOX_ROOT}/rl/buffer_server.py"
