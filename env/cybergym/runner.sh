#!/usr/bin/env bash
set -euo pipefail

CYBERGYM_RUNNER_ROOT="${CYBERGYM_RUNNER_ROOT:-/opt/safactory/cybergym}"
CYBERGYM_RUNNER_TMP="$(mktemp -d /tmp/safactory-cybergym.XXXXXX)"
export CYBERGYM_RUNNER_TMP

# shellcheck source=runtime/common.sh
source "${CYBERGYM_RUNNER_ROOT}/common.sh"
# shellcheck source=runtime/docker_prepare.sh
source "${CYBERGYM_RUNNER_ROOT}/docker_prepare.sh"
# shellcheck source=runtime/rjob_prepare.sh
source "${CYBERGYM_RUNNER_ROOT}/rjob_prepare.sh"

REQUEST_JSON="${CYBERGYM_RUNNER_TMP}/request.json"
EPISODE_JSON="${CYBERGYM_RUNNER_TMP}/episode.json"
EPISODE_ENV="${CYBERGYM_RUNNER_TMP}/episode.env"
DOCKER_JSON="${CYBERGYM_RUNNER_TMP}/docker.json"
DOCKER_ENV="${CYBERGYM_RUNNER_TMP}/docker.env"
NATIVE_JSON="${CYBERGYM_RUNNER_TMP}/native.json"
NATIVE_ENV="${CYBERGYM_RUNNER_TMP}/native.env"
NATIVE_OUTPUT="${CYBERGYM_RUNNER_TMP}/agent.log"
VERIFY_OUTPUT="${CYBERGYM_RUNNER_TMP}/verify.log"
SERVER_PID=""
RESULT_EMITTED=0

cleanup() {
  terminate_process "${SERVER_PID:-}"
  cleanup_rjob_docker
  rm -rf "$CYBERGYM_RUNNER_TMP"
}

emit_unexpected_failure() {
  local returncode=$?
  local command="${BASH_COMMAND:-unknown command}"
  trap - ERR
  if [[ "$RESULT_EMITTED" != "1" ]]; then
    python3.12 "${CYBERGYM_RUNNER_ROOT}/result_writer.py" failure \
      --reason "CyberGym runner command failed with code ${returncode}: ${command}" \
      --request "$REQUEST_JSON" \
      --episode "$EPISODE_JSON" || true
    RESULT_EMITTED=1
  fi
  exit 0
}

trap cleanup EXIT
trap emit_unexpected_failure ERR

request_payload="$(cat)"
if [[ -z "${request_payload//[[:space:]]/}" ]]; then
  request_payload="${SAFACTORY_START_REQUEST_JSON:-}"
fi
if [[ -z "${request_payload//[[:space:]]/}" ]]; then
  cybergym_log "SimulationStartRequest JSON was not provided on stdin"
  false
fi
printf '%s\n' "$request_payload" >"$REQUEST_JSON"

python3.12 "${CYBERGYM_RUNNER_ROOT}/episode_prepare.py" \
  --request "$REQUEST_JSON" \
  --output "$EPISODE_JSON" \
  --env-out "$EPISODE_ENV"
# shellcheck disable=SC1090
source "$EPISODE_ENV"

export PYTHONPATH="${EPISODE_CYBERGYM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"
export LLM_API_KEY="$EPISODE_GATEWAY_API_KEY"
export OPENAI_API_KEY="$EPISODE_GATEWAY_API_KEY"
export ANTHROPIC_API_KEY="$EPISODE_GATEWAY_API_KEY"
export OPENAI_BASE_URL="$EPISODE_GATEWAY_URL"
export OPENAI_API_BASE="$EPISODE_GATEWAY_URL"
export OPENAI_API_BASE_URL="$EPISODE_GATEWAY_URL"
export LLM_BASE_URL="$EPISODE_GATEWAY_URL"

case "${CYBERGYM_DOCKER_MODE:-host}" in
  host)
    ;;
  dind)
    prepare_rjob_docker
    ;;
  *)
    cybergym_log "Unsupported CYBERGYM_DOCKER_MODE: ${CYBERGYM_DOCKER_MODE}"
    false
    ;;
esac

image_timeout_s="$(phase_timeout "$EPISODE_IMAGE_LOAD_TIMEOUT_S" 240 60)"
prepare_docker_assets "$DOCKER_JSON" "$DOCKER_ENV" "$image_timeout_s"
# shellcheck disable=SC1090
source "$DOCKER_ENV"

# The following three commands intentionally mirror CyberGym's README flow:
# start the server, run the agent, then verify the final submission.
server_command=(
  "$EPISODE_PYTHON_BIN" -m cybergym.server
  --host 0.0.0.0
  --port "$EPISODE_SERVER_PORT"
  --log_dir "$EPISODE_SERVER_DIR"
  --db_path "$EPISODE_DB_PATH"
)
if [[ -n "$EPISODE_MASK_MAP_PATH" ]]; then
  server_command+=(--mask_map_path "$EPISODE_MASK_MAP_PATH")
fi
CYBERGYM_API_KEY="$EPISODE_CYBERGYM_API_KEY" \
  "${server_command[@]}" >/dev/null 2>&1 &
SERVER_PID=$!
server_wait_s="$(phase_timeout 60 120 10)"
wait_for_cybergym_server "$EPISODE_SERVER_URL" "$SERVER_PID" "$server_wait_s"

agent_timeout_s="$(phase_timeout \
  "$EPISODE_AGENT_TIMEOUT_S" \
  "$((EPISODE_VERIFY_TIMEOUT_S + 180))" \
  60)"
process_timeout_s="$(phase_timeout \
  "$((agent_timeout_s + 120))" \
  "$((EPISODE_VERIFY_TIMEOUT_S + 60))" \
  60)"

agent_command=(
  "$EPISODE_PYTHON_BIN"
  "${CYBERGYM_RUNNER_ROOT}/agent_dispatch.py"
  --episode "$EPISODE_JSON"
  --runner-tmp "$CYBERGYM_RUNNER_TMP"
  --runtime-host "$EPISODE_RUNTIME_HOST"
  --agent-server-url "$EPISODE_AGENT_SERVER_URL"
  --timeout "$agent_timeout_s"
)

printf 'command:' >"$NATIVE_OUTPUT"
printf ' %q' "${agent_command[@]}" >>"$NATIVE_OUTPUT"
printf '\n' >>"$NATIVE_OUTPUT"
agent_started_s=$SECONDS
if timeout --signal=TERM --kill-after=30 "${process_timeout_s}s" \
  "${agent_command[@]}" >>"$NATIVE_OUTPUT" 2>&1; then
  native_returncode=0
else
  native_returncode=$?
fi
agent_elapsed_s=$((SECONDS - agent_started_s))
if (( native_returncode == 124 || \
      (native_returncode != 0 && agent_elapsed_s >= agent_timeout_s) )); then
  printf '\nagent timed out: elapsed=%ss timeout=%ss\n' \
    "$agent_elapsed_s" "$agent_timeout_s" >>"$NATIVE_OUTPUT"
fi
printf '\nreturncode: %s\n' "$native_returncode" >>"$NATIVE_OUTPUT"

python3.12 "${CYBERGYM_RUNNER_ROOT}/result_writer.py" discover \
  --log-dir "$EPISODE_LOGS_DIR" \
  --task-id "$EPISODE_TASK_ID" \
  --agent-type "$EPISODE_AGENT_TYPE" \
  --output "$NATIVE_JSON" \
  --env-out "$NATIVE_ENV"
# shellcheck disable=SC1090
source "$NATIVE_ENV"

verification_returncode=""
verification_error=""
if [[ -n "$EPISODE_AGENT_ID" ]]; then
  verify_timeout_s="$(phase_timeout "$EPISODE_VERIFY_TIMEOUT_S" 30 30)"
  verify_command=(
    "$EPISODE_PYTHON_BIN"
    "${EPISODE_CYBERGYM_ROOT}/scripts/verify_agent_result.py"
    --server "$EPISODE_SERVER_URL"
    --pocdb_path "$EPISODE_DB_PATH"
    --agent_id "$EPISODE_AGENT_ID"
  )
  printf 'command:' >"$VERIFY_OUTPUT"
  printf ' %q' "${verify_command[@]}" >>"$VERIFY_OUTPUT"
  printf '\n' >>"$VERIFY_OUTPUT"
  set +e
  CYBERGYM_API_KEY="$EPISODE_CYBERGYM_API_KEY" \
    timeout --signal=TERM --kill-after=30 "${verify_timeout_s}s" \
      "${verify_command[@]}" >>"$VERIFY_OUTPUT" 2>&1
  verify_rc=$?
  set -e
  printf '\nreturncode: %s\n' "$verify_rc" >>"$VERIFY_OUTPUT"
  if (( verify_rc == 124 )); then
    verification_error="verify_agent_result.py timed out after ${verify_timeout_s}s"
  else
    verification_returncode="$verify_rc"
    if (( verify_rc != 0 )); then
      verification_error="verify_agent_result.py exited with code ${verify_rc}"
    fi
  fi
else
  : >"$VERIFY_OUTPUT"
fi

python3.12 "${CYBERGYM_RUNNER_ROOT}/result_writer.py" final \
  --episode "$EPISODE_JSON" \
  --docker "$DOCKER_JSON" \
  --native "$NATIVE_JSON" \
  --native-returncode "$native_returncode" \
  --native-output "$NATIVE_OUTPUT" \
  --verification-returncode "$verification_returncode" \
  --verification-error "$verification_error" \
  --verification-output "$VERIFY_OUTPUT"
RESULT_EMITTED=1
