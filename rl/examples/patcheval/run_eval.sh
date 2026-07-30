#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

: "${PATCH_EVAL_API_KEY:?Set PATCH_EVAL_API_KEY before running}"
: "${DOCKER_HOST:?Set DOCKER_HOST before running}"

PYTHON_BIN=${PYTHON_BIN:-python3}
PATCH_EVAL_API_BASE=${PATCH_EVAL_API_BASE:-http://35.220.164.252:3888/v1}
PATCH_EVAL_MODEL=${PATCH_EVAL_MODEL:-}
PATCH_EVAL_BASELINE=${PATCH_EVAL_BASELINE:-llm}
PATCH_EVAL_SETTING=${PATCH_EVAL_SETTING:-s1.1}
PATCH_EVAL_AGENT_EXPERIMENT=${PATCH_EVAL_AGENT_EXPERIMENT:-exp1}
PATCH_EVAL_AGENT_TOOL_LIMIT=${PATCH_EVAL_AGENT_TOOL_LIMIT:-100}
PATCH_EVAL_CLAUDE_INSTALL_TIMEOUT_S=${PATCH_EVAL_CLAUDE_INSTALL_TIMEOUT_S:-900}
PATCH_EVAL_TASK_LIMIT=${PATCH_EVAL_TASK_LIMIT:-0}
PATCH_EVAL_POOL_SIZE=${PATCH_EVAL_POOL_SIZE:-5}
PATCH_EVAL_DOCKER_STARTUP_CONCURRENCY=${PATCH_EVAL_DOCKER_STARTUP_CONCURRENCY:-3}
PATCH_EVAL_IMAGE_ARCHIVE_DIR=${PATCH_EVAL_IMAGE_ARCHIVE_DIR:-/mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-images}
PATCH_EVAL_OFFICIAL_RUNTIME_DIR=${PATCH_EVAL_OFFICIAL_RUNTIME_DIR:-/mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-runtime}
PATCH_EVAL_SHARED_TMP=${PATCH_EVAL_SHARED_TMP:-/mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-tmp}
PATCH_EVAL_HTTP_PROXY=${PATCH_EVAL_HTTP_PROXY:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}
PATCH_EVAL_EVALUATION_TIMEOUT_S=${PATCH_EVAL_EVALUATION_TIMEOUT_S:-3600}
PATCH_EVAL_GATEWAY_PORT=${PATCH_EVAL_GATEWAY_PORT:-18000}
PATCH_EVAL_CLAUDE_ADAPTER_PORT=${PATCH_EVAL_CLAUDE_ADAPTER_PORT:-18001}
PATCH_EVAL_SHUTDOWN_TIMEOUT_S=${PATCH_EVAL_SHUTDOWN_TIMEOUT_S:-600}
GATEWAY_HOST=${GATEWAY_HOST:-$(hostname -I | awk '{print $1}')}
PATCH_EVAL_NO_PROXY=${PATCH_EVAL_NO_PROXY:-host.docker.internal,localhost,127.0.0.1,::1,${GATEWAY_HOST},10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn}

case "${PATCH_EVAL_BASELINE}" in
  llm)
    PATCH_EVAL_MODEL=${PATCH_EVAL_MODEL:-bailian/deepseek-v4-flash}
    case "${PATCH_EVAL_SETTING}" in
      s1.1|s1.2|s1.3) PATCH_EVAL_AGENT_TIMEOUT_S=${PATCH_EVAL_AGENT_TIMEOUT_S:-900} ;;
      s1.4) PATCH_EVAL_AGENT_TIMEOUT_S=${PATCH_EVAL_AGENT_TIMEOUT_S:-4500} ;;
      *) echo "Unsupported PATCH_EVAL_SETTING: ${PATCH_EVAL_SETTING}" >&2; exit 2 ;;
    esac
    run_label="${PATCH_EVAL_SETTING}"
    ;;
  claudecode)
    : "${PATCH_EVAL_MODEL:?Set PATCH_EVAL_MODEL to the exact Claude route ID}"
    [[ "${PATCH_EVAL_AGENT_EXPERIMENT}" == "exp1" ]] || {
      echo "Only Claude Code exp1 is currently supported" >&2
      exit 2
    }
    PATCH_EVAL_AGENT_TIMEOUT_S=${PATCH_EVAL_AGENT_TIMEOUT_S:-2700}
    run_label="claudecode-${PATCH_EVAL_AGENT_EXPERIMENT}"
    ;;
  *) echo "Unsupported PATCH_EVAL_BASELINE: ${PATCH_EVAL_BASELINE}" >&2; exit 2 ;;
esac

mkdir -p "${PATCH_EVAL_SHARED_TMP}" "${SCRIPT_DIR}/logs"
export TMPDIR="${PATCH_EVAL_SHARED_TMP}"

OFFICIAL_SOURCE="${ROOT}/env/patcheval/PatchEval/patcheval"
mkdir -p "${PATCH_EVAL_OFFICIAL_RUNTIME_DIR}/exp_llm/helper"
mkdir -p "${PATCH_EVAL_OFFICIAL_RUNTIME_DIR}/exp_agent/claudecode/templates"
cp \
  "${OFFICIAL_SOURCE}/exp_llm/helper/llm_suite.py" \
  "${OFFICIAL_SOURCE}/exp_llm/helper/func_replacer.py" \
  "${OFFICIAL_SOURCE}/exp_llm/helper/__init__.py" \
  "${PATCH_EVAL_OFFICIAL_RUNTIME_DIR}/exp_llm/helper/"
touch "${PATCH_EVAL_OFFICIAL_RUNTIME_DIR}/exp_llm/__init__.py"
cp \
  "${OFFICIAL_SOURCE}/exp_agent/claudecode/templates/default.md" \
  "${PATCH_EVAL_OFFICIAL_RUNTIME_DIR}/exp_agent/claudecode/templates/default.md"

model_slug="${PATCH_EVAL_MODEL//\//_}"
model_slug="${model_slug//:/_}"
run_id="${PATCH_EVAL_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
PATCH_EVAL_DB=${PATCH_EVAL_DB:-${SCRIPT_DIR}/patcheval_${model_slug}_${run_label}_${run_id}.db}
GENERATED_DIR=${PATCH_EVAL_GENERATED_DIR:-${PATCH_EVAL_SHARED_TMP}/safactory-patcheval-${run_label}-${run_id}}
GATEWAY_CONFIG="${PATCH_EVAL_SHARED_TMP}/safactory-patcheval-gateway-${run_id}.yaml"
GATEWAY_LOG="${SCRIPT_DIR}/logs/gateway-${run_id}.log"
CLAUDE_ADAPTER_LOG="${SCRIPT_DIR}/logs/claude-adapter-${run_id}.log"

mkdir -p "${GENERATED_DIR}" "$(dirname -- "${PATCH_EVAL_DB}")"

PATCH_EVAL_DB="${PATCH_EVAL_DB}" \
PATCH_EVAL_API_BASE="${PATCH_EVAL_API_BASE}" \
PATCH_EVAL_API_KEY="${PATCH_EVAL_API_KEY}" \
PATCH_EVAL_MODEL="${PATCH_EVAL_MODEL}" \
PATCH_EVAL_GATEWAY_PORT="${PATCH_EVAL_GATEWAY_PORT}" \
GATEWAY_CONFIG="${GATEWAY_CONFIG}" \
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import yaml

db = Path(os.environ["PATCH_EVAL_DB"]).expanduser().resolve()
config = {
    "listen_host": "0.0.0.0",
    "listen_port": int(os.environ["PATCH_EVAL_GATEWAY_PORT"]),
    "base_session_path": "/v1/sessions",
    "max_steps": -1,
    "storage_type": "sqlite",
    "storage_config": {"db_url": f"sqlite:///{db}"},
    "llm_routes": {
        os.environ["PATCH_EVAL_MODEL"]: {
            "base_url": os.environ["PATCH_EVAL_API_BASE"].rstrip("/") + "/",
            "api_key": os.environ["PATCH_EVAL_API_KEY"],
            "supports_stream": True,
            "max_concurrency": 64,
        }
    },
}
path = Path(os.environ["GATEWAY_CONFIG"])
path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
path.chmod(0o600)
PY

cleanup() {
  if [[ -n "${CLAUDE_ADAPTER_PID:-}" ]] && kill -0 "${CLAUDE_ADAPTER_PID}" 2>/dev/null; then
    kill "${CLAUDE_ADAPTER_PID}" 2>/dev/null || true
    wait "${CLAUDE_ADAPTER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${GATEWAY_PID:-}" ]] && kill -0 "${GATEWAY_PID}" 2>/dev/null; then
    kill "${GATEWAY_PID}" 2>/dev/null || true
    wait "${GATEWAY_PID}" 2>/dev/null || true
  fi
  rm -f -- "${GATEWAY_CONFIG}"
}
trap cleanup EXIT INT TERM

cd "${ROOT}"
"${PYTHON_BIN}" -m gateway --config "${GATEWAY_CONFIG}" >"${GATEWAY_LOG}" 2>&1 &
GATEWAY_PID=$!

for _ in $(seq 1 60); do
  if ! kill -0 "${GATEWAY_PID}" 2>/dev/null; then
    echo "Gateway exited early; inspect ${GATEWAY_LOG}" >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:${PATCH_EVAL_GATEWAY_PORT}/readyz" >/dev/null; then
    break
  fi
  sleep 1
done
if ! curl -fsS --max-time 2 "http://127.0.0.1:${PATCH_EVAL_GATEWAY_PORT}/readyz" >/dev/null; then
  echo "Gateway did not become ready; inspect ${GATEWAY_LOG}" >&2
  exit 1
fi

if [[ "${PATCH_EVAL_BASELINE}" == "claudecode" ]]; then
  CLAUDE_ADAPTER_GATEWAY_SESSION_BASE_URL="http://127.0.0.1:${PATCH_EVAL_GATEWAY_PORT}/v1/sessions" \
  CLAUDE_ADAPTER_ROUTE_MODEL="${PATCH_EVAL_MODEL}" \
  CLAUDE_ADAPTER_PORT="${PATCH_EVAL_CLAUDE_ADAPTER_PORT}" \
  CLAUDE_ADAPTER_REQUEST_TIMEOUT_S="${PATCH_EVAL_AGENT_TIMEOUT_S}" \
    "${PYTHON_BIN}" -m env.patcheval.claude_adapter >"${CLAUDE_ADAPTER_LOG}" 2>&1 &
  CLAUDE_ADAPTER_PID=$!
  for _ in $(seq 1 60); do
    if ! kill -0 "${CLAUDE_ADAPTER_PID}" 2>/dev/null; then
      echo "Claude adapter exited early; inspect ${CLAUDE_ADAPTER_LOG}" >&2
      exit 1
    fi
    if curl -fsS --max-time 2 "http://127.0.0.1:${PATCH_EVAL_CLAUDE_ADAPTER_PORT}/readyz" >/dev/null; then
      break
    fi
    sleep 1
  done
  if ! curl -fsS --max-time 2 "http://127.0.0.1:${PATCH_EVAL_CLAUDE_ADAPTER_PORT}/readyz" >/dev/null; then
    echo "Claude adapter did not become ready; inspect ${CLAUDE_ADAPTER_LOG}" >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" env/patcheval/generate_full_config.py \
  --output-dir "${GENERATED_DIR}" \
  --archive-dir "${PATCH_EVAL_IMAGE_ARCHIVE_DIR}" \
  --official-runtime-dir "${PATCH_EVAL_OFFICIAL_RUNTIME_DIR}" \
  --baseline "${PATCH_EVAL_BASELINE}" \
  --setting "${PATCH_EVAL_SETTING}" \
  --agent-experiment "${PATCH_EVAL_AGENT_EXPERIMENT}" \
  --agent-tool-limit "${PATCH_EVAL_AGENT_TOOL_LIMIT}" \
  --claude-install-timeout-s "${PATCH_EVAL_CLAUDE_INSTALL_TIMEOUT_S}" \
  --claude-adapter-base-url "http://${GATEWAY_HOST}:${PATCH_EVAL_CLAUDE_ADAPTER_PORT}/v1/sessions" \
  --limit "${PATCH_EVAL_TASK_LIMIT}" \
  --evaluation-timeout-s "${PATCH_EVAL_EVALUATION_TIMEOUT_S}" \
  --shared-tmp "${PATCH_EVAL_SHARED_TMP}" \
  --http-proxy "${PATCH_EVAL_HTTP_PROXY}" \
  --no-proxy "${PATCH_EVAL_NO_PROXY}"

echo "PatchEval baseline: ${PATCH_EVAL_BASELINE}"
echo "PatchEval setting: ${run_label}"
echo "Model: ${PATCH_EVAL_MODEL}"
echo "Results DB: ${PATCH_EVAL_DB}"
echo "Generated config: ${GENERATED_DIR}"

"${PYTHON_BIN}" launcher.py \
  --mode docker \
  --docker-pull-policy never \
  --docker-image-archive-dir "${PATCH_EVAL_IMAGE_ARCHIVE_DIR}" \
  --cleanup-docker-image \
  --docker-startup-concurrency "${PATCH_EVAL_DOCKER_STARTUP_CONCURRENCY}" \
  --docker-start-timeout-s 1800 \
  --docker-inspect-timeout-s 60 \
  --agent-start-timeout-s "${PATCH_EVAL_AGENT_TIMEOUT_S}" \
  --container-refill-timeout-s 1800 \
  --shutdown-timeout-s "${PATCH_EVAL_SHUTDOWN_TIMEOUT_S}" \
  --agent-root "${GENERATED_DIR}" \
  --agent-config "${GENERATED_DIR}/patcheval_config.yaml" \
  --agent-start-config "${GENERATED_DIR}/patcheval_start.yaml" \
  --gateway-base-url "http://${GATEWAY_HOST}:${PATCH_EVAL_GATEWAY_PORT}/v1/sessions" \
  --llm-model "${PATCH_EVAL_MODEL}" \
  --llm-temperature 0 \
  --db-path "sqlite:///${PATCH_EVAL_DB}" \
  --storage-type sqlite \
  --pool-size "${PATCH_EVAL_POOL_SIZE}" \
  --max-workers "${PATCH_EVAL_POOL_SIZE}" \
  --max-steps 1 \
  --enable-evaluation \
  --no-circuit-breaker
