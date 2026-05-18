#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

LOG_DIR="${SCRIPT_DIR}/logs/${AIEVOBOX_RUN_ID}"
mkdir -p "${LOG_DIR}"

wait_for_port() {
  local host="$1"
  local port="$2"
  local name="$3"
  local retries="${4:-60}"

  for _ in $(seq 1 "${retries}"); do
    if timeout 1 bash -c ":</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
      echo "${name} is ready at ${host}:${port}"
      return 0
    fi
    sleep 2
  done

  echo "Timed out waiting for ${name} at ${host}:${port}" >&2
  return 1
}

is_port_open() {
  local host="$1"
  local port="$2"
  timeout 1 bash -c ":</dev/tcp/${host}/${port}" >/dev/null 2>&1
}

cleanup_children() {
  local exit_code=$?
  if [ "${exit_code}" -ne 0 ]; then
    echo "run_train_4gpu.sh exited with ${exit_code}. Check logs in ${LOG_DIR}" >&2
  fi
}
trap cleanup_children EXIT

echo "Starting Multi-QAGym 4-GPU training"
echo "  Run ID: ${AIEVOBOX_RUN_ID}"
echo "  Logs: ${LOG_DIR}"
echo "  DB: ${AIEVOBOX_DB_URL}"
echo "  Buffer server: http://${BUFFER_SERVER_HOST}:${BUFFER_SERVER_PORT}"
echo "  Judge proxy: ${JUDGE_PROXY_URL}"

echo "Cleaning old Ray/SGLang processes..."
pkill -9 sglang || true
"${RAY_BIN}" stop --force || true
pkill -9 ray || true
pkill -9 raylet || true
pkill -9 gcs_server || true
pkill -9 dashboard || true
pkill -9 dashboard_agent || true
pkill -9 runtime_env_agent || true
sleep 2

echo "Starting shared Ray..."
"${RAY_BIN}" start \
  --head \
  --node-ip-address 127.0.0.1 \
  --port 6379 \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats
wait_for_port 127.0.0.1 8265 "Ray dashboard"

if is_port_open "${JUDGE_PROXY_HOST}" "${JUDGE_PROXY_PORT}"; then
  echo "Judge proxy is already running at ${JUDGE_PROXY_URL}"
else
  echo "Judge proxy is not reachable at ${JUDGE_PROXY_URL}" >&2
  echo "Start the external judge proxy first, then rerun this script." >&2
  exit 1
fi

wait_for_port "${BUFFER_SERVER_HOST}" "${BUFFER_SERVER_PORT}" "buffer server"

echo "Starting defender trainer..."
(
  cd "${SCRIPT_DIR}"
  export SKIP_RUNTIME_CLEANUP=1
  export RAY_RESTART=0
  export RAY_START=0
  exec ./run_defender_generator.sh
) >"${LOG_DIR}/defender_generator.log" 2>&1 &
DEFENDER_PID=$!

DEFENDER_WARMUP_SECONDS=60
echo "Waiting ${DEFENDER_WARMUP_SECONDS}s for defender policy endpoint to come up..."
sleep "${DEFENDER_WARMUP_SECONDS}"

echo "Starting attacker trainer and shared rollout..."
(
  cd "${SCRIPT_DIR}"
  export SKIP_RUNTIME_CLEANUP=1
  export RAY_RESTART=0
  export RAY_START=0
  exec ./run_attacker_generator.sh
) >"${LOG_DIR}/attacker_generator.log" 2>&1 &
ATTACKER_PID=$!

echo "All processes started."
echo "  Defender submit PID: ${DEFENDER_PID}"
echo "  Attacker submit PID: ${ATTACKER_PID}"
echo
echo "Logs:"
echo "  ${LOG_DIR}/defender_generator.log"
echo "  ${LOG_DIR}/attacker_generator.log"
echo
echo "Press Ctrl-C to stop this launcher process. Ray jobs may need to be stopped separately."

wait "${DEFENDER_PID}" "${ATTACKER_PID}"
