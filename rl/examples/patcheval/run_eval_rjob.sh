#!/usr/bin/env bash
# =============================================================================
# PatchEval — RJob mode evaluation runner (LLM baseline, s1.1~s1.4)
# =============================================================================
# Architecture:
#   SGLang (Qwen3.8-27B)  ->  100.104.113.76:30000   (GPU inference host)
#   Gateway (this host)   ->  100.99.17.62:8000       (fronts SGLang, reachable
#                                                     from RJob pods in cluster)
#   launcher.py           ->  submits each CVE episode as a cluster RJob pod
#                             that pulls CVE images from the private registry
#                             and calls back the gateway for LLM completions.
#
# Prereqs (must already be done once):
#   1. SGLang running on 100.104.113.76:30000 (see start_sglang_qwen38_27b.sh).
#   2. CVE images pushed to registry.h.pjlab.org.cn (push_patcheval_images.sh).
#   3. config.yaml at repo root has valid rjob access_key/secret_key.
#
# Usage:
#   PATCH_EVAL_API_KEY=sk-qwen38-27b-local ./run_eval_rjob.sh s1.1        # one setting
#   PATCH_EVAL_API_KEY=sk-qwen38-27b-local ./run_eval_rjob.sh s1.1 s1.2 s1.3 s1.4
#   PATCH_EVAL_API_KEY=sk-qwen38-27b-local PATCH_EVAL_TASK_LIMIT=2 ./run_eval_rjob.sh s1.1   # smoke test
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

# --- Required ---
: "${PATCH_EVAL_API_KEY:?Set PATCH_EVAL_API_KEY (the SGLang api-key, e.g. sk-qwen38-27b-local)}"

# --- Tunables (env-overridable) ---
PYTHON_BIN=${PYTHON_BIN:-/mnt/shared-storage-user/evobox-share/leishanzhe/env/slime-env-0.3.1/bin/python}

# The conda env bundles libicui18n which needs CXXABI_1.3.15 from a newer
# libstdc++ than the system one. Prepend the env's own lib/ so its bundled
# libstdc++.so.6 is loaded (otherwise `import sqlite3` fails and the gateway
# can't init its sqlite storage).
ENV_LIB_DIR=$(dirname "$(dirname "${PYTHON_BIN}")")/lib
if [[ -d "${ENV_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${ENV_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

# SGLang endpoint (inference host)
SGLANG_HOST=${SGLANG_HOST:-100.104.113.76}
SGLANG_PORT=${SGLANG_PORT:-30000}
SGLANG_MODEL=${SGLANG_MODEL:-qwen3.8-27b}

# Gateway (this host, must be reachable from RJob pods)
GATEWAY_HOST=${GATEWAY_HOST:-$(hostname -I | awk '{print $1}')}
GATEWAY_PORT=${GATEWAY_PORT:-18000}   # 8000 is taken by nginx on this host

# Baseline / settings
PATCH_EVAL_BASELINE=${PATCH_EVAL_BASELINE:-llm}
SETTINGS=()
if [[ $# -gt 0 ]]; then SETTINGS=("$@"); else SETTINGS=(s1.1 s1.2 s1.3 s1.4); fi

# Eval knobs
PATCH_EVAL_TASK_LIMIT=${PATCH_EVAL_TASK_LIMIT:-0}            # 0 = all 230 CVEs
PATCH_EVAL_POOL_SIZE=${PATCH_EVAL_POOL_SIZE:-16}             # parallel RJob pods
PATCH_EVAL_AGENT_TIMEOUT_S=${PATCH_EVAL_AGENT_TIMEOUT_S:-900}
PATCH_EVAL_EVALUATION_TIMEOUT_S=${PATCH_EVAL_EVALUATION_TIMEOUT_S:-3600}
PATCH_EVAL_SHUTDOWN_TIMEOUT_S=${PATCH_EVAL_SHUTDOWN_TIMEOUT_S:-600}
PATCH_EVAL_STORAGE_TYPE=${PATCH_EVAL_STORAGE_TYPE:-sqlite}

# Paths
PATCH_EVAL_IMAGE_ARCHIVE_DIR=${PATCH_EVAL_IMAGE_ARCHIVE_DIR:-/mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-images}
PATCH_EVAL_OFFICIAL_RUNTIME_DIR=${PATCH_EVAL_OFFICIAL_RUNTIME_DIR:-/mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-runtime}
PATCH_EVAL_SHARED_TMP=${PATCH_EVAL_SHARED_TMP:-/mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-tmp}
PATCH_EVAL_HTTP_PROXY=${PATCH_EVAL_HTTP_PROXY:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}
PATCH_EVAL_NO_PROXY=${PATCH_EVAL_NO_PROXY:-localhost,127.0.0.1,::1,${GATEWAY_HOST},10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn}

# RJob registry (must match push_patcheval_images.sh)
RJOB_REGISTRY=${RJOB_REGISTRY:-registry.h.pjlab.org.cn}
RJOB_REGISTRY_NS=${RJOB_REGISTRY_NS:-ailab-evobox-evobox_proxy}
RJOB_REPO=${RJOB_REPO:-patcheval}
RJOB_CONFIG=${RJOB_CONFIG:-${ROOT}/config.yaml}

run_id=${PATCH_EVAL_RUN_ID:-$(date +%Y%m%d-%H%M%S)}

# Results root for safactory_result.json. MUST be on shared storage (RJob pods
# in the cluster write here via gpfs mount), but isolated from RL runs by an
# eval-specific subdir + run_id so eval and RL results never mix.
RJOB_RESULTS_ROOT=${RJOB_RESULTS_ROOT:-${ROOT}/results/patcheval_eval/${run_id}}

mkdir -p "${PATCH_EVAL_SHARED_TMP}" "${SCRIPT_DIR}/logs"
export TMPDIR="${PATCH_EVAL_SHARED_TMP}"

# SQLite DBs live under the SAfactory repo (so they're persistent & collectible
# from shared storage), in an eval-specific subdir + run_id so they never clash
# with RL's patcheval_qwen3_8_27b.db. The repo is on GPFS; if sqlite hits a
# transient "disk I/O error" under concurrent load, override
# PATCH_EVAL_LOCAL_DB_DIR to a local /tmp path for that run.
LOCAL_DB_DIR="${PATCH_EVAL_LOCAL_DB_DIR:-${SCRIPT_DIR}/eval_runs/${run_id}}"
mkdir -p "${LOCAL_DB_DIR}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
echo "=== Pre-flight ==="
echo "LLM      : http://${SGLANG_HOST}:${SGLANG_PORT}  (model=${SGLANG_MODEL})"
echo "Gateway  : http://${GATEWAY_HOST}:${GATEWAY_PORT}  (this host, reachable from RJob pods)"
echo "RJob cfg : ${RJOB_CONFIG}"
echo "Registry : ${RJOB_REGISTRY}/${RJOB_REGISTRY_NS}/${RJOB_REPO}"
echo "Settings : ${SETTINGS[*]}  (limit=${PATCH_EVAL_TASK_LIMIT}, pool=${PATCH_EVAL_POOL_SIZE})"

# LLM endpoint reachability — try /health (SGLang), then /v1/models (OpenAI proxies
# like the claude-opus-5 gateway that lack /health). Either passing is enough.
LLM_BASE="http://${SGLANG_HOST}:${SGLANG_PORT}"
llm_ok=0
if curl -fsS --max-time 8 "${LLM_BASE}/health" >/dev/null 2>&1; then
  llm_ok=1
elif curl -fsS --max-time 10 "${LLM_BASE}/v1/models" -H "Authorization: Bearer ${PATCH_EVAL_API_KEY}" >/dev/null 2>&1; then
  llm_ok=1
fi
if [[ "${llm_ok}" -ne 1 ]]; then
  echo "ERROR: LLM endpoint not reachable at ${LLM_BASE} (tried /health and /v1/models)." >&2
  echo "       For SGLang: start it first (start_sglang_qwen38_27b.sh)." >&2
  echo "       For opus-5: check the proxy at ${LLM_BASE} and PATCH_EVAL_API_KEY." >&2
  exit 1
fi
[[ -f "${RJOB_CONFIG}" ]] || { echo "ERROR: RJob config not found: ${RJOB_CONFIG}" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Stage official runtime helpers (same as docker run_eval.sh)
# ---------------------------------------------------------------------------
OFFICIAL_SOURCE="${ROOT}/env/patcheval/PatchEval/patcheval"
mkdir -p "${PATCH_EVAL_OFFICIAL_RUNTIME_DIR}/exp_llm/helper"
mkdir -p "${PATCH_EVAL_OFFICIAL_RUNTIME_DIR}/exp_agent/claudecode/templates"
cp \
  "${OFFICIAL_SOURCE}/exp_llm/helper/llm_suite.py" \
  "${OFFICIAL_SOURCE}/exp_llm/helper/func_replacer.py" \
  "${OFFICIAL_SOURCE}/exp_llm/helper/__init__.py" \
  "${PATCH_EVAL_OFFICIAL_RUNTIME_DIR}/exp_llm/helper/" 2>/dev/null || true
touch "${PATCH_EVAL_OFFICIAL_RUNTIME_DIR}/exp_llm/__init__.py"
cp \
  "${OFFICIAL_SOURCE}/exp_agent/claudecode/templates/default.md" \
  "${PATCH_EVAL_OFFICIAL_RUNTIME_DIR}/exp_agent/claudecode/templates/default.md" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Start the gateway (one shared instance across all settings)
# ---------------------------------------------------------------------------
GATEWAY_CONFIG="${PATCH_EVAL_SHARED_TMP}/safactory-patcheval-gateway-rjob-${run_id}.yaml"
GATEWAY_LOG="${SCRIPT_DIR}/logs/gateway-rjob-${run_id}.log"

PATCH_EVAL_DB_BASE="${PATCH_EVAL_DB_BASE:-${LOCAL_DB_DIR}/patcheval_${SGLANG_MODEL}_rjob_${run_id}.db}"

model_slug="${SGLANG_MODEL//\//_}"
model_slug="${model_slug//:/_}"

PATCH_EVAL_DB="${PATCH_EVAL_DB_BASE}" \
PATCH_EVAL_API_BASE="http://${SGLANG_HOST}:${SGLANG_PORT}/v1" \
PATCH_EVAL_API_KEY="${PATCH_EVAL_API_KEY}" \
PATCH_EVAL_MODEL="${SGLANG_MODEL}" \
PATCH_EVAL_GATEWAY_PORT="${GATEWAY_PORT}" \
PATCH_EVAL_STORAGE_TYPE="${PATCH_EVAL_STORAGE_TYPE}" \
GATEWAY_CONFIG="${GATEWAY_CONFIG}" \
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import yaml

db = Path(os.environ["PATCH_EVAL_DB"]).expanduser().resolve()
storage_type = os.environ["PATCH_EVAL_STORAGE_TYPE"]
config = {
    "listen_host": "0.0.0.0",
    "listen_port": int(os.environ["PATCH_EVAL_GATEWAY_PORT"]),
    "base_session_path": "/v1/sessions",
    "max_steps": -1,
    "storage_type": storage_type,
    "storage_config": (
        {"db_url": f"sqlite:///{db}"} if storage_type == "sqlite" else {}
    ),
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
  if [[ -n "${GATEWAY_PID:-}" ]] && kill -0 "${GATEWAY_PID}" 2>/dev/null; then
    kill "${GATEWAY_PID}" 2>/dev/null || true
    wait "${GATEWAY_PID}" 2>/dev/null || true
  fi
  rm -f -- "${GATEWAY_CONFIG}"
}
trap cleanup EXIT INT TERM

cd "${ROOT}"
"${PYTHON_BIN}" -u -m gateway --config "${GATEWAY_CONFIG}" >"${GATEWAY_LOG}" 2>&1 &
GATEWAY_PID=$!

for _ in $(seq 1 60); do
  if ! kill -0 "${GATEWAY_PID}" 2>/dev/null; then
    echo "Gateway exited early; inspect ${GATEWAY_LOG}" >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:${GATEWAY_PORT}/readyz" >/dev/null; then
    break
  fi
  sleep 1
done
if ! curl -fsS --max-time 2 "http://127.0.0.1:${GATEWAY_PORT}/readyz" >/dev/null; then
  echo "Gateway did not become ready; inspect ${GATEWAY_LOG}" >&2
  exit 1
fi
echo "Gateway ready: http://${GATEWAY_HOST}:${GATEWAY_PORT}/v1/sessions  (log: ${GATEWAY_LOG})"

# ---------------------------------------------------------------------------
# Per-setting loop
# ---------------------------------------------------------------------------
GATEWAY_BASE_URL="http://${GATEWAY_HOST}:${GATEWAY_PORT}/v1/sessions"

for setting in "${SETTINGS[@]}"; do
  echo ""
  echo "############################################################"
  echo "# Setting ${setting}  (baseline=${PATCH_EVAL_BASELINE})"
  echo "############################################################"

  GENERATED_DIR="${PATCH_EVAL_SHARED_TMP}/safactory-patcheval-rjob-${setting}-${run_id}"
  mkdir -p "${GENERATED_DIR}"

  # Per-setting DB so results don't collide across settings.
  # The launcher --db-path MUST be the same file the gateway writes to (the
  # evaluator reads the gateway's trajectory DB). One shared DB per run,
  # episodes are distinguished by session_id/job_id (matches run_eval.sh).
  setting_db="${PATCH_EVAL_DB_BASE}"

  "${PYTHON_BIN}" env/patcheval/generate_full_config.py \
    --output-dir "${GENERATED_DIR}" \
    --archive-dir "${PATCH_EVAL_IMAGE_ARCHIVE_DIR}" \
    --official-runtime-dir "${PATCH_EVAL_OFFICIAL_RUNTIME_DIR}" \
    --baseline "${PATCH_EVAL_BASELINE}" \
    --setting "${setting}" \
    --claude-gateway-base-url "${GATEWAY_BASE_URL}" \
    --claude-model "${SGLANG_MODEL}" \
    --limit "${PATCH_EVAL_TASK_LIMIT}" \
    --evaluation-timeout-s "${PATCH_EVAL_EVALUATION_TIMEOUT_S}" \
    --shared-tmp "${PATCH_EVAL_SHARED_TMP}" \
    --http-proxy "${PATCH_EVAL_HTTP_PROXY}" \
    --no-proxy "${PATCH_EVAL_NO_PROXY}" \
    --mode rjob \
    --rjob-registry "${RJOB_REGISTRY}" \
    --rjob-registry-ns "${RJOB_REGISTRY_NS}" \
    --rjob-repo "${RJOB_REPO}" \
    --rjob-results-root "${RJOB_RESULTS_ROOT}"

  echo "Generated rjob config: ${GENERATED_DIR}"
  echo "  config: ${GENERATED_DIR}/patcheval_config.rjob.yaml"
  echo "  start : ${GENERATED_DIR}/patcheval_start.rjob.yaml"

  launcher_storage_args=(--storage-type "${PATCH_EVAL_STORAGE_TYPE}")
  if [[ "${PATCH_EVAL_STORAGE_TYPE}" == "sqlite" ]]; then
    launcher_storage_args+=(--db-path "sqlite:///${setting_db}")
  fi

  "${PYTHON_BIN}" launcher.py \
    --mode rjob \
    --rjob-config "${RJOB_CONFIG}" \
    --agent-root "${GENERATED_DIR}" \
    --agent-config "${GENERATED_DIR}/patcheval_config.rjob.yaml" \
    --agent-start-config "${GENERATED_DIR}/patcheval_start.rjob.yaml" \
    --gateway-base-url "${GATEWAY_BASE_URL}" \
    --llm-model "${SGLANG_MODEL}" \
    --llm-temperature 0 \
    --agent-start-timeout-s "${PATCH_EVAL_AGENT_TIMEOUT_S}" \
    --shutdown-timeout-s "${PATCH_EVAL_SHUTDOWN_TIMEOUT_S}" \
    "${launcher_storage_args[@]}" \
    --pool-size "${PATCH_EVAL_POOL_SIZE}" \
    --max-workers "${PATCH_EVAL_POOL_SIZE}" \
    --max-steps 1 \
    --enable-evaluation \
    --no-circuit-breaker \
    ${PATCHEVAL_RESUME:+--resume}

  echo "Setting ${setting} done. Results DB: ${setting_db}"
done

echo ""
echo "=== All settings complete ==="
echo "Gateway log: ${GATEWAY_LOG}"
