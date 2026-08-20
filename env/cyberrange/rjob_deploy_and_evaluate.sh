#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

fatal() {
  printf '[safactory-cyberrange-rjob] fatal: %s\n' "$*" >&2
  exit 2
}

SOURCE_ROOT="${AGENT_RANGE_BRAINPP_SOURCE_ROOT:?missing AGENT_RANGE_BRAINPP_SOURCE_ROOT}"
REPORT_DIR="${AGENT_RANGE_BRAINPP_ACCEPTANCE_REPORT_DIR:?missing AGENT_RANGE_BRAINPP_ACCEPTANCE_REPORT_DIR}"
RESULT_PATH="$REPORT_DIR/runtime-test-result.json"
REPORT_ROOT="$SOURCE_ROOT/results/brainpp"

[[ "$(id -u)" -eq 0 ]] || fatal "DEPLOYMENT.md host deployment requires root"
[[ "$SOURCE_ROOT" == /mnt/shared-storage-user/wangyixu/cyberrange ]] \
  || fatal "source root must use cyberrange's canonical GPFS path"
[[ -d "$SOURCE_ROOT" && ! -L "$SOURCE_ROOT" ]] || fatal "source root is unavailable or is a symlink"
[[ "$REPORT_DIR" == "$REPORT_ROOT"/* && "$REPORT_DIR" != "$REPORT_ROOT" ]] \
  || fatal "report directory must be a unique child below $REPORT_ROOT"
[[ ! -e "$REPORT_DIR" && ! -L "$REPORT_DIR" ]] || fatal "report directory already exists"
[[ -c /dev/kvm && -r /dev/kvm && -w /dev/kvm ]] || fatal "/dev/kvm is unavailable"
[[ -c /dev/net/tun && -r /dev/net/tun && -w /dev/net/tun ]] || fatal "/dev/net/tun is unavailable"

for command in bash python3 apt-get; do
  command -v "$command" >/dev/null 2>&1 || fatal "required deployment command is missing: $command"
done

printf '[safactory-cyberrange-rjob] deploying production stack and running one case report=%s\n' \
  "$REPORT_DIR" >&2

# The native runtime-task detects an HTTP model URL and passes the explicit
# `agent-range test submit --allow-insecure-http` CLI opt-in. No temporary
# TestSpec or source-tree override is needed.
/bin/bash "$SOURCE_ROOT/scripts/brainpp_source_bootstrap_acceptance.sh"

[[ -f "$RESULT_PATH" && ! -L "$RESULT_PATH" ]] \
  || fatal "deployment/evaluation did not produce runtime-test-result.json"
python3 - "$RESULT_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(value, dict):
    raise SystemExit("runtime-test-result.json must contain one JSON object")
required = ("run_outcome", "e2e_success", "platform_health", "evidence_status")
missing = [key for key in required if key not in value]
if missing:
    raise SystemExit(f"runtime-test-result.json is missing fields: {missing}")
PY

chmod 0444 "$RESULT_PATH"
printf '[safactory-cyberrange-rjob] sealed native result=%s\n' "$RESULT_PATH" >&2
