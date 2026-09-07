#!/usr/bin/env bash
set -euo pipefail

CYBERGYM_RUNNER_ROOT="${CYBERGYM_RUNNER_ROOT:-/opt/safactory/cybergym}"
CYBERGYM_RUNNER_TMP="$(mktemp -d /tmp/safactory-cybergym.XXXXXX)"
export CYBERGYM_RUNNER_TMP

# RJob embeds the editable result writer at this path. Prefer it so result
# classification changes can be tested without rebuilding the controller;
# Docker/local mode falls back to the baked-in helper.
if [[ -f /workspace/cybergym/result_writer.py ]]; then
  CYBERGYM_RESULT_WRITER=/workspace/cybergym/result_writer.py
else
  CYBERGYM_RESULT_WRITER="${CYBERGYM_RUNNER_ROOT}/result_writer.py"
fi
if [[ -f /workspace/cybergym/agent_dispatch.py ]]; then
  CYBERGYM_AGENT_DISPATCH=/workspace/cybergym/agent_dispatch.py
else
  CYBERGYM_AGENT_DISPATCH="${CYBERGYM_RUNNER_ROOT}/agent_dispatch.py"
fi
export CYBERGYM_RESULT_WRITER
export CYBERGYM_AGENT_DISPATCH

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
RUNNER_PHASE="bootstrap"
DEBUG_DIR=""

cleanup() {
  terminate_process "${SERVER_PID:-}"
  cleanup_rjob_docker
  rm -rf "$CYBERGYM_RUNNER_TMP"
}

emit_unexpected_failure() {
  local returncode=$?
  local command="${BASH_COMMAND:-unknown command}"
  trap - ERR
  if [[ -n "${DEBUG_DIR:-}" ]]; then
    {
      printf 'phase: %s\n' "${RUNNER_PHASE:-unknown}"
      printf 'returncode: %s\n' "$returncode"
      printf 'command: %s\n' "$command"
    } >"${DEBUG_DIR}/runner-failure.log" || true
  fi
  if [[ "$RESULT_EMITTED" != "1" ]]; then
    python3.12 "$CYBERGYM_RESULT_WRITER" failure \
      --reason "CyberGym runner failed in phase ${RUNNER_PHASE:-unknown} with code ${returncode}: ${command}" \
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

RUNNER_PHASE="episode_prepare"
python3.12 "${CYBERGYM_RUNNER_ROOT}/episode_prepare.py" \
  --request "$REQUEST_JSON" \
  --output "$EPISODE_JSON" \
  --env-out "$EPISODE_ENV"
# shellcheck disable=SC1090
source "$EPISODE_ENV"

DEBUG_DIR="${EPISODE_RESULTS_DIR}/debug"
mkdir -p "$DEBUG_DIR"
cp "$REQUEST_JSON" "${DEBUG_DIR}/request.json"
python3.12 - "$EPISODE_JSON" "${DEBUG_DIR}/episode.json" <<'PY'
import json
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
payload = json.loads(source.read_text(encoding="utf-8"))
for key in ("gateway_api_key", "cybergym_api_key"):
    if key in payload:
        payload[key] = "<redacted>"
destination.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
DOCKER_JSON="${DEBUG_DIR}/docker.json"
DOCKER_ENV="${DEBUG_DIR}/docker.env"
NATIVE_JSON="${DEBUG_DIR}/native.json"
NATIVE_ENV="${DEBUG_DIR}/native.env"
NATIVE_OUTPUT="${DEBUG_DIR}/agent.log"
VERIFY_OUTPUT="${DEBUG_DIR}/verify.log"
export CYBERGYM_DEBUG_DIR="$DEBUG_DIR"
export CYBERGYM_DOCKER_LOG="${DEBUG_DIR}/dockerd.log"

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
    RUNNER_PHASE="dind_start"
    prepare_rjob_docker
    ;;
  *)
    cybergym_log "Unsupported CYBERGYM_DOCKER_MODE: ${CYBERGYM_DOCKER_MODE}"
    false
    ;;
esac

image_timeout_s="$(phase_timeout "$EPISODE_IMAGE_LOAD_TIMEOUT_S" 240 60)"
RUNNER_PHASE="docker_assets"
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
if [[ -n "${EPISODE_MASK_MAP_PATH:-}" ]]; then
  server_command+=(--mask_map_path "$EPISODE_MASK_MAP_PATH")
fi
RUNNER_PHASE="server_start"
CYBERGYM_API_KEY="$EPISODE_CYBERGYM_API_KEY" \
  "${server_command[@]}" >"${DEBUG_DIR}/server.log" 2>&1 &
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
  "$CYBERGYM_AGENT_DISPATCH"
  --episode "$EPISODE_JSON"
  --runner-tmp "$CYBERGYM_RUNNER_TMP"
  --runtime-host "$EPISODE_RUNTIME_HOST"
  --agent-server-url "$EPISODE_AGENT_SERVER_URL"
  --timeout "$agent_timeout_s"
)

printf 'command:' >"$NATIVE_OUTPUT"
printf ' %q' "${agent_command[@]}" >>"$NATIVE_OUTPUT"
printf '\n' >>"$NATIVE_OUTPUT"
set +e
RUNNER_PHASE="agent_run"
agent_started_epoch="$(date +%s)"
agent_started_s=$SECONDS
timeout --signal=TERM --kill-after=30 "${process_timeout_s}s" \
  "${agent_command[@]}" >>"$NATIVE_OUTPUT" 2>&1
native_returncode=$?
set -e
agent_elapsed_s=$((SECONDS - agent_started_s))
if (( native_returncode == 124 || \
      (native_returncode != 0 && agent_elapsed_s >= agent_timeout_s) )); then
  printf '\nagent timed out: elapsed=%ss timeout=%ss\n' \
    "$agent_elapsed_s" "$agent_timeout_s" >>"$NATIVE_OUTPUT"
fi
printf '\nreturncode: %s\n' "$native_returncode" >>"$NATIVE_OUTPUT"

# Preserve enough information to diagnose native-agent exits (especially when
# the outer timeout returns 1/143 instead of 124).  These files are copied to
# the result debug directory and survive controller cleanup.
{
  printf '\n--- agent execution diagnostics ---\n'
  printf 'runner_phase=%s\n' "$RUNNER_PHASE"
  printf 'agent_started_epoch=%s\n' "${agent_started_epoch:-unknown}"
  printf 'agent_elapsed_s=%s\n' "$agent_elapsed_s"
  printf 'agent_timeout_s=%s\n' "$agent_timeout_s"
  printf 'process_timeout_s=%s\n' "$process_timeout_s"
  printf 'native_returncode=%s\n' "$native_returncode"
  case "$native_returncode" in
    124) printf 'exit_class=outer_timeout\n' ;;
    137) printf 'exit_class=SIGKILL\n' ;;
    143) printf 'exit_class=SIGTERM\n' ;;
    0) printf 'exit_class=success\n' ;;
    *) printf 'exit_class=agent_or_dispatch_failure\n' ;;
  esac
  printf 'native_output_bytes=%s\n' "$(wc -c <"$NATIVE_OUTPUT")"
  printf 'trajectory_files:\n'
  find "$EPISODE_LOGS_DIR" -maxdepth 3 -type f \( -name 'trajectory.jsonl' -o -name 'console.log' -o -name 'docker-runtime.json' \) -printf '  %p %s bytes\n' 2>/dev/null || true
  printf '%s\n' '--- native output tail ---'
  tail -80 "$NATIVE_OUTPUT" || true
  for f in $(find "$EPISODE_LOGS_DIR" -maxdepth 3 -type f -name 'console.log' 2>/dev/null); do
    printf '%s\n' "--- console tail: $f ---"
    tail -80 "$f" || true
  done
  for f in $(find "$EPISODE_LOGS_DIR" -maxdepth 3 -type f -name 'trajectory.jsonl' 2>/dev/null); do
    printf '%s\n' "--- trajectory tail: $f ---"
    tail -20 "$f" || true
  done
} >"${DEBUG_DIR}/agent-exit-diagnostics.log" 2>&1 || true

python3.12 "$CYBERGYM_RESULT_WRITER" discover \
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
  RUNNER_PHASE="verification"
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

python3.12 "$CYBERGYM_RESULT_WRITER" final \
  --episode "$EPISODE_JSON" \
  --docker "$DOCKER_JSON" \
  --native "$NATIVE_JSON" \
  --native-returncode "$native_returncode" \
  --native-output "$NATIVE_OUTPUT" \
  --verification-returncode "$verification_returncode" \
  --verification-error "$verification_error" \
  --verification-output "$VERIFY_OUTPUT"
RUNNER_PHASE="complete"
RESULT_EMITTED=1
