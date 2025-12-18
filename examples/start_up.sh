#!/usr/bin/env bash
set -euo pipefail

################################
# Step 0: YAML → DB 灌数
################################

YAML_AGGREGATOR_PY="../core/data_manager/yaml_aggregator.py"
DB_PATH=".test_data/env.db"
ENV_ROOT="../env"

################################
# Step 1: Env Service
################################
ENV_APP="../env_manager/app.py"
ENV_HOST="localhost"
ENV_PORT="36008"
HEALTHZ_URL="http://${ENV_HOST}:${ENV_PORT}/healthz"

WAIT_TIMEOUT_SEC=300
WAIT_INTERVAL_SEC=1

################################
# Step 2: Eval Server
################################
EVAL_APP="base_eval_server.py"

ENV_SERVICE_URL="http://localhost:36008"
MAX_WORKERS=100
MAX_STEPS=1000
LLM_API_KEY="EMPTY"
LLM_BASE_URL="http://100.99.130.249:30000/v1"
LLM_MODEL="Qwen3-30B-Instruct"
LLM_TEMPERATURE="0.99"

################################
# Pre-check
################################
command -v curl >/dev/null 2>&1 || {
  echo "[ERROR] curl not found"
  exit 1
}

[[ -f "${YAML_AGGREGATOR_PY}" ]] || {
  echo "[ERROR] yaml_aggregator.py not found: ${YAML_AGGREGATOR_PY}"
  exit 1
}

[[ -f "${ENV_APP}" ]] || {
  echo "[ERROR] app.py not found: ${ENV_APP}"
  exit 1
}

################################
# Step 0: 灌数据库
################################
echo "[0/3] Populate DB from YAML"
python - <<EOF
import sys
from pathlib import Path

yaml_file = Path("${YAML_AGGREGATOR_PY}").resolve()

repo_root = yaml_file.parents[2]   # .../AIEvoBox
sys.path.insert(0, str(repo_root))

from core.data_manager.yaml_aggregator import populate_env_table

db_path = Path("${DB_PATH}").resolve()
env_root = Path("${ENV_ROOT}").resolve()

print(f"[DB] db_path={db_path}")
print(f"[DB] env_root={env_root}")

populate_env_table(db_path=db_path, env_root=env_root)
print("[DB] populate_env_table done.")
EOF


################################
# Step 1: 启动 env service
################################
echo "[1/3] Start env service"
(
  cd ..
  python v2/app.py
) &
ENV_PID=$!
echo "ENV_PID=${ENV_PID}"

cleanup() {
  if kill -0 "${ENV_PID}" >/dev/null 2>&1; then
    echo "[CLEANUP] Stop env service"
    kill "${ENV_PID}" || true
    sleep 1
    kill -9 "${ENV_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

################################
# Step 2: 等待 healthz
################################
echo "[2/3] Waiting for service ready: ${HEALTHZ_URL}"
start_ts=$(date +%s)

while true; do
  if ! kill -0 "${ENV_PID}" >/dev/null 2>&1; then
    echo "[ERROR] env service exited unexpectedly"
    exit 1
  fi

  if curl -fsS "${HEALTHZ_URL}" >/dev/null 2>&1; then
    echo "Service ready ✅"
    break
  fi

  now_ts=$(date +%s)
  if (( now_ts - start_ts >= WAIT_TIMEOUT_SEC )); then
    echo "[ERROR] healthz timeout"
    exit 1
  fi

  sleep "${WAIT_INTERVAL_SEC}"
done

################################
# Step 3: 启动评测
################################
echo "[3/3] Start eval server"
python "${EVAL_APP}" \
  --env-service-url "${ENV_SERVICE_URL}" \
  --max-workers "${MAX_WORKERS}" \
  --max-steps "${MAX_STEPS}" \
  --llm-api-key "${LLM_API_KEY}" \
  --llm-base-url "${LLM_BASE_URL}" \
  --llm-model "${LLM_MODEL}" \
  --llm-temperature "${LLM_TEMPERATURE}"\
  --db-path "${db_path}"