#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

RESULT_PREFIX='SAFACTORY_RESULT_JSON '
DEFAULT_RESULT_ROOT='/app/results'
CANONICAL_SOURCE_ROOT='/mnt/shared-storage-user/evoagi-share/cyberrange'
NATIVE_RESULT_NAME='runtime-test-result.json'
MILESTONES_NAME='milestones.json'
RUNTIME_LOG_NAME='runtime-task.log'
ARTIFACT_SYNC_FAST_INTERVAL_SECONDS=60
ARTIFACT_SYNC_FAST_DURATION_SECONDS=600
ARTIFACT_SYNC_SLOW_INTERVAL_SECONDS=600

result_emitted=0
model_key=''
job_log=''
artifact_sync_pid=''

json_escape() {
  local value="${1:-}"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "$value"
}

safe_component() {
  local value="${1:-unknown}"
  value="$(printf '%s' "$value" | sed 's/[^A-Za-z0-9._-]/_/g')"
  printf '%.160s' "${value:-unknown}"
}

# SAfactory exports task_id as a simple value. Recover the benchmark scenario
# from the strictly validated CyberRange task naming convention.
if [[ -z "${CYBERRANGE_SCENARIO_ID:-}" ]]; then
  scenario_candidate="${CYBERRANGE_CASE_ID:-${SAFACTORY_TASK_ID:-}}"
  if [[ "$scenario_candidate" =~ ^(postexploitbench-range[3-6]-v1)(-smoke)?$ ]]; then
    export CYBERRANGE_SCENARIO_ID="${BASH_REMATCH[1]}"
    printf '[safactory-cyberrange] derived scenario=%s from task metadata\n' \
      "$CYBERRANGE_SCENARIO_ID" >&2
  fi
fi

job_id="${SAFACTORY_JOB_ID:-unknown}"
session_id="${SAFACTORY_SESSION_ID:-unknown}"
task_id="${SAFACTORY_TASK_ID:-unknown}"
safe_job_id="$(safe_component "$job_id")"
safe_session_id="$(safe_component "$session_id")"
result_path="${SAFACTORY_RESULT_PATH:-$DEFAULT_RESULT_ROOT/$safe_job_id/$safe_session_id/safactory_result.json}"
result_dir="$(dirname -- "$result_path")"
started_at="$(date +%s)"

write_payload() {
  local payload="$1"
  local temporary
  if [[ "$result_path" == /* ]]; then
    mkdir -p -m 0700 -- "$result_dir"
    temporary="$result_dir/.safactory_result.json.$$.tmp"
    printf '%s\n' "$payload" >"$temporary"
    mv -f -- "$temporary" "$result_path"
    chmod 0444 "$result_path"
  fi
  printf '%s%s\n' "$RESULT_PREFIX" "$payload"
  result_emitted=1
}

publish_artifact() {
  local source="$1"
  local name="$2"
  local target="$result_dir/$name"
  local temporary="$result_dir/.$name.${BASHPID:-$$}.tmp"
  cp -f -- "$source" "$temporary"
  mv -f -- "$temporary" "$target"
  chmod 0444 "$target"
}

publish_current_artifacts() {
  local native_path milestones_path
  if [[ -n "$job_log" && -f "$job_log" && ! -L "$job_log" ]]; then
    publish_artifact "$job_log" "$RUNTIME_LOG_NAME"
  fi

  native_path="$(find "$report_dir" -type f -name "$NATIVE_RESULT_NAME" -print | sort | tail -n 1)"
  if [[ -n "$native_path" && -f "$native_path" && ! -L "$native_path" ]]; then
    publish_artifact "$native_path" "$NATIVE_RESULT_NAME"
  fi

  milestones_path="$(find "$report_dir" -type f -name "$MILESTONES_NAME" -print | sort | tail -n 1)"
  if [[ -n "$milestones_path" && -f "$milestones_path" && ! -L "$milestones_path" ]]; then
    publish_artifact "$milestones_path" "$MILESTONES_NAME"
  fi
}

artifact_sync_loop() {
  local elapsed=0
  # This subshell must not run the parent's error/exit handlers.
  trap - ERR EXIT
  set +Ee

  while (( elapsed < ARTIFACT_SYNC_FAST_DURATION_SECONDS )); do
    sleep "$ARTIFACT_SYNC_FAST_INTERVAL_SECONDS" || return 0
    publish_current_artifacts
    elapsed=$((elapsed + ARTIFACT_SYNC_FAST_INTERVAL_SECONDS))
  done

  while sleep "$ARTIFACT_SYNC_SLOW_INTERVAL_SECONDS"; do
    publish_current_artifacts
  done
}

start_artifact_sync() {
  artifact_sync_loop &
  artifact_sync_pid=$!
}

stop_artifact_sync() {
  local pid="${artifact_sync_pid:-}"
  [[ -n "$pid" ]] || return 0
  artifact_sync_pid=''
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

emit_failure() {
  local message="$1"
  local truncated="${2:-false}"
  local now duration log_artifact payload
  set +e
  stop_artifact_sync
  now="$(date +%s)"
  duration="$(( (now - started_at) * 1000 ))"
  log_artifact=''
  if [[ -n "$job_log" && -f "$job_log" && ! -L "$job_log" ]]; then
    mkdir -p -m 0700 -- "$result_dir"
    cp -f -- "$job_log" "$result_dir/$RUNTIME_LOG_NAME"
    chmod 0444 "$result_dir/$RUNTIME_LOG_NAME"
    log_artifact=',"runtime_log_artifact":"runtime-task.log"'
  fi
  payload="$(printf '{"session_id":"%s","status":"failed","total_reward":0.0,"step_count":0,"terminated":true,"truncated":%s,"error_text":"%s","metrics":{"bench":"cyberrange","task_id":"%s","duration_ms":%s%s}}' \
    "$(json_escape "$session_id")" \
    "$truncated" \
    "$(json_escape "$message")" \
    "$(json_escape "$task_id")" \
    "$duration" \
    "$log_artifact")"
  write_payload "$payload"
  set -e
}

fail() {
  emit_failure "$1" "${2:-false}"
  exit 0
}

cleanup() {
  set +e
  stop_artifact_sync
  if [[ -n "$model_key" ]]; then
    rm -f -- "$model_key"
  fi
}

on_error() {
  local status="$1"
  local line="$2"
  trap - ERR
  if [[ "$result_emitted" != 1 ]]; then
    emit_failure "runner failed at line $line with status $status"
  fi
  exit 0
}

trap cleanup EXIT
trap 'on_error $? $LINENO' ERR

required_env=(
  SAFACTORY_TASK_ID
  SAFACTORY_ROUTE_MODEL
  SAFACTORY_GATEWAY_SESSION_URL_CONTAINER
  CYBERRANGE_SOURCE_ROOT
  CYBERRANGE_WHEELHOUSE_ROOT
  CYBERRANGE_RELEASE_ROOT
  CYBERRANGE_REPORT_ROOT
  CYBERRANGE_OUTPUT_ROOT
  CYBERRANGE_SCENARIO_ID
  CYBERRANGE_WALL_TIME_SECONDS
)
missing_env=()
for name in "${required_env[@]}"; do
  [[ -n "${!name:-}" ]] || missing_env+=("$name")
done
(( ${#missing_env[@]} == 0 )) || fail "missing required environment(s): ${missing_env[*]}"

task_id="$SAFACTORY_TASK_ID"
case_id="${CYBERRANGE_CASE_ID:-$CYBERRANGE_SCENARIO_ID}"
source_root="$CYBERRANGE_SOURCE_ROOT"
wheelhouse_root="$CYBERRANGE_WHEELHOUSE_ROOT"
release_root="$CYBERRANGE_RELEASE_ROOT"
report_root="$CYBERRANGE_REPORT_ROOT"
output_root="$CYBERRANGE_OUTPUT_ROOT"
scenario_id="$CYBERRANGE_SCENARIO_ID"
wall_time="$CYBERRANGE_WALL_TIME_SECONDS"

[[ "$result_path" == /* ]] || fail 'SAFACTORY_RESULT_PATH must be absolute'
[[ "${EUID:-$(id -u)}" == 0 ]] || fail 'CyberRange runtime requires root'
[[ "$source_root" == "$CANONICAL_SOURCE_ROOT" ]] || fail 'source_root does not match the canonical worker checkout'
[[ "$report_root" == "$source_root/results/chengzikai" ]] || fail 'report_root must be the configured CyberRange results directory'
[[ "$output_root" == "$report_root" ]] || fail 'output_root must match report_root'
[[ "$(dirname -- "$wheelhouse_root")" == "$source_root/var/brainpp-wheelhouses" ]] || fail 'wheelhouse_root must be directly below var/brainpp-wheelhouses'
[[ "$(dirname -- "$release_root")" == "$source_root/releases" ]] || fail 'release_root must be directly below releases'
[[ -d "$source_root" && ! -L "$source_root" ]] || fail 'CyberRange source_root is unavailable'
[[ -d "$wheelhouse_root" && ! -L "$wheelhouse_root" ]] || fail 'CyberRange wheelhouse_root is unavailable'
[[ -d "$release_root" && ! -L "$release_root" ]] || fail 'CyberRange release_root is unavailable'
[[ "$scenario_id" =~ ^postexploitbench-range[3-6]-v1$ ]] || fail 'invalid CyberRange scenario_id'
[[ "$wall_time" =~ ^[0-9]+$ && "$wall_time" -ge 300 && "$wall_time" -le 36000 ]] || fail 'invalid CyberRange wall_time_seconds'
[[ -r /dev/kvm && -w /dev/kvm ]] || fail '/dev/kvm is unavailable'
[[ -r /dev/net/tun && -w /dev/net/tun ]] || fail '/dev/net/tun is unavailable'

for command in bash chmod cp date dirname find mkdir mv rm sed sleep sort tail; do
  command -v "$command" >/dev/null 2>&1 || fail "missing runtime command $command"
done

bootstrap="$source_root/scripts/brainpp_source_bootstrap_acceptance.sh"
[[ -f "$bootstrap" && ! -L "$bootstrap" ]] || fail 'CyberRange runtime-task bootstrap is unavailable'

attempt_id="$(date +%Y%m%d-%H%M%S)-$$-${RANDOM:-0}"
run_root="$output_root/safactory_runtime_task_$(safe_component "$task_id")_${safe_session_id}_$attempt_id"
report_dir="$run_root/report"
job_log="$run_root/$RUNTIME_LOG_NAME"
model_key="$run_root/secrets/model-api.key"

[[ ! -e "$run_root" && ! -L "$run_root" ]] || fail 'CyberRange run directory already exists'
mkdir -p -m 0700 -- "$report_dir" "$(dirname -- "$model_key")" "$result_dir"
# The native runtime requires a key file, but the provider credential remains
# inside Safactory Gateway. This value is disposable and scoped to one episode.
printf 'safactory-gateway-session\n' >"$model_key"
chmod 0600 "$model_key"

export AGENT_RANGE_BRAINPP_SOURCE_BOOTSTRAP_ACCEPTANCE=1
export AGENT_RANGE_BRAINPP_SOURCE_BOOTSTRAP_MODE=runtime-task
export AGENT_RANGE_BRAINPP_PROJECT_ROOT="$source_root"
export AGENT_RANGE_BRAINPP_SOURCE_ROOT="$source_root"
export AGENT_RANGE_BRAINPP_ACCEPTANCE_WHEELHOUSE_ROOT="$wheelhouse_root"
export AGENT_RANGE_BRAINPP_RELEASE_ROOT="$release_root"
export AGENT_RANGE_BRAINPP_ACCEPTANCE_REPORT_ROOT="$report_root"
export AGENT_RANGE_BRAINPP_ACCEPTANCE_REPORT_DIR="$report_dir"
export AGENT_RANGE_BRAINPP_JOB_LOG="$job_log"
export AGENT_RANGE_BRAINPP_RUNTIME_SCENARIO_ID="$scenario_id"
export AGENT_RANGE_BRAINPP_RUNTIME_MODEL_NAME="$SAFACTORY_ROUTE_MODEL"
export AGENT_RANGE_BRAINPP_RUNTIME_MODEL_BASE_URL="$SAFACTORY_GATEWAY_SESSION_URL_CONTAINER"
export AGENT_RANGE_BRAINPP_RUNTIME_MODEL_KEY_PATH="$model_key"
export AGENT_RANGE_BRAINPP_RUNTIME_WALL_TIME_SECONDS="$wall_time"

printf '[safactory-cyberrange] starting runtime-task scenario=%s model=%s run_root=%s\n' \
  "$scenario_id" "$SAFACTORY_ROUTE_MODEL" "$run_root" >&2

start_artifact_sync
set +e
/bin/bash "$bootstrap" >>"$job_log" 2>&1
runtime_status=$?
set -e
stop_artifact_sync
if (( runtime_status != 0 )); then
  fail "CyberRange runtime-task exited with status $runtime_status" "$([[ "$runtime_status" == 124 ]] && printf true || printf false)"
fi

native_path="$(find "$report_dir" -type f -name "$NATIVE_RESULT_NAME" -print | sort | tail -n 1)"
[[ -n "$native_path" && -f "$native_path" && ! -L "$native_path" ]] || fail "CyberRange native result $NATIVE_RESULT_NAME is missing"
milestones_path="$(find "$report_dir" -type f -name "$MILESTONES_NAME" -print | sort | tail -n 1)"

publish_artifact "$native_path" "$NATIVE_RESULT_NAME"
artifact_metrics=',"runtime_test_result_json_artifact":"runtime-test-result.json"'
if [[ -n "$milestones_path" && -f "$milestones_path" && ! -L "$milestones_path" ]]; then
  publish_artifact "$milestones_path" "$MILESTONES_NAME"
  artifact_metrics+=',"milestones_json_artifact":"milestones.json"'
fi
if [[ -f "$job_log" && ! -L "$job_log" ]]; then
  publish_artifact "$job_log" "$RUNTIME_LOG_NAME"
  artifact_metrics+=',"runtime_task_log_artifact":"runtime-task.log"'
fi

finished_at="$(date +%s)"
duration_ms="$(( (finished_at - started_at) * 1000 ))"
published_native_path="$result_dir/$NATIVE_RESULT_NAME"
payload="$(printf '{"session_id":"%s","status":"succeeded","total_reward":0.0,"step_count":1,"terminated":true,"truncated":false,"error_text":null,"metrics":{"bench":"cyberrange","job_id":"%s","task_id":"%s","case_id":"%s","scenario_ref":"%s","native_result_path":"%s"%s,"duration_ms":%s}}' \
  "$(json_escape "$session_id")" \
  "$(json_escape "$job_id")" \
  "$(json_escape "$task_id")" \
  "$(json_escape "$case_id")" \
  "$(json_escape "$scenario_id")" \
  "$(json_escape "$published_native_path")" \
  "$artifact_metrics" \
  "$duration_ms")"
write_payload "$payload"
exit 0
