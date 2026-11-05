#!/usr/bin/env bash
# Run from any working directory. User-provided gateway defaults are configured below.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${EXPGYM_OFFLINE_ROOT:-}" ]]; then
  candidate="${SUDO_USER:+$(getent passwd "$SUDO_USER" | cut -d: -f6)}"
  candidate="${candidate:-$HOME}/exploitgym-offline"
  if [[ -f "$candidate/activate.sh" ]]; then export EXPGYM_OFFLINE_ROOT="$candidate"; fi
fi
source "$ROOT/env_cn.sh"
export AGENT_BINARY="$(cd -- "$ROOT/.." && pwd)/claude-code-2.1.228-linux-amd64/claude"
if [[ -n "${EXPGYM_OFFLINE_ROOT:-}" ]]; then
  "$UV_PYTHON" "$EXPGYM_OFFLINE_ROOT/relocate.py"
fi
export PJ_USER_BATCH_DIR="${MINING_BATCH_DIR:-$ROOT/pj-mining-400}"
BATCH_NAME="$(basename -- "$PJ_USER_BATCH_DIR")"
export MODEL="${MODEL:-glm-5.2}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://10.12.111.133:49183}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-4113c1af3b3ff6006da324ce1919ce68cf658eea556d5d179834a1e98c6defec}"
export WORKERS="${WORKERS:-20}"
export TIMEOUT="${TIMEOUT:-2700}"
export BUDGET="${BUDGET:-20}"
if [[ "$BATCH_NAME" == pj-mining-400 ]]; then DEFAULT_OUT="$ROOT/out/mining_400"; else DEFAULT_OUT="$ROOT/out/$BATCH_NAME"; fi
export OUT_DIR="${OUT_DIR:-$DEFAULT_OUT}"
MAX_TURNS="${MAX_TURNS:-1}"
RESUME="${RESUME:-1}"
SETUP="${SETUP:-0}"
CHECK=0
DRY_RUN=0
TASK_NAME=""
case "${1:-}" in
  --dry-run) DRY_RUN=1; shift ;;
  --check) CHECK=1; shift ;;
  -h|--help)
    cat <<'EOF'
Usage: bash run_mining_400.sh [TASK_NAME|--check|--dry-run]

--dry-run generates this partition's execution plans without Docker/model calls.
Agent: ../claude-code-2.1.228-linux-amd64/claude (delivery copy).

Configured defaults: model, upstream gateway and user-provided API key.
Optional overrides:
  MINING_BATCH_DIR       Partition path; defaults to ROOT/pj-mining-400
  ANTHROPIC_API_KEY       Override the configured upstream model key
  ANTHROPIC_BASE_URL      Upstream Anthropic-compatible gateway
  MODEL=glm-5.2           Model
  WORKERS=20             Concurrent tasks
  MAX_TURNS=1            Agent turns; set 300 for a full evaluation
  TIMEOUT=2700           Per-task timeout in seconds
  BUDGET=20              Per-task proxy budget in USD
  OUT_DIR=.../out/mining_400
  RESUME=1               Skip tasks already having result.json (including failures)
  SETUP=1                Run setup_cn.sh with domestic mirrors
  PROXY_URL              Proxy address reachable by agent containers
  CYBERGYM_ADMIN_KEY     Existing proxy admin key, if reusing a running proxy

--check validates the selected partition task list and required bundle files only; no Docker,
model calls, service startup, dependency installs, or image loads.

Stages: check -> optional first-time setup -> Docker checks -> Squid image ->
shared image load -> firewall/controller/proxy -> per-task image loads,
Harbor agent execution and native verification.
Results: OUT_DIR/<task>/result.json, logs/load_image.log, harbor-jobs/;
         OUT_DIR/pj_user_batch_summary.json
Logs:    OUT_DIR/launcher-<timestamp>.log
Set a new OUT_DIR or RESUME=0 to rerun all tasks.
EOF
    exit 0 ;;
esac
if [[ $# == 1 ]]; then TASK_NAME="$1"; shift; fi
[[ $# == 0 ]] || { echo "Unknown arguments: $*" >&2; exit 2; }
for value in "$WORKERS" "$MAX_TURNS" "$TIMEOUT"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'WORKERS/MAX_TURNS/TIMEOUT must be positive integers' >&2; exit 2; }
done
[[ "$RESUME" =~ ^[01]$ && "$SETUP" =~ ^[01]$ ]] || { echo 'RESUME/SETUP must be 0 or 1' >&2; exit 2; }
python3 - "$ROOT" "$PJ_USER_BATCH_DIR" "$TASK_NAME" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);batch=Path(sys.argv[2]);requested=sys.argv[3]
rows=[x.strip() for x in (batch/'selected_tasks.txt').read_text().splitlines() if x.strip()]
names=set(rows)
assert rows and len(rows)==len(names),'Expected a non-empty unique task list'
assert len(names)==json.loads((batch/'tasks.json').read_text())['task_count'],'Task count differs from index'
assert all(n.startswith('arvo-') and n.endswith('-lv0') for n in names),'Expected ARVO lv0 tasks'
actual={p.name for p in batch.iterdir() if (p/'harbor-runtime-profile/task.toml').is_file()}
assert actual==names, f'Task set differs: missing={names-actual}; extra={actual-names}'
assert set((batch/'selected_tasks.txt').read_text().splitlines())==names,'pj.txt does not match manifest'
for name in ['scripts/run_pj_exploit.sh','examples/run_agent.py','scripts/setup/pre_run.py']:
 assert (root/name).is_file(),f'Missing {name}'
for name in ['bin/load_image.py']:
 assert (batch/name).is_file(),f'Missing {name}'
assert (batch/'shared-images').is_dir(),'Missing shared images'
assert (root.parent/'claude-code-2.1.228-linux-amd64/claude').is_file(),'Missing delivery Claude binary'
for name in names:
 assert any((batch/name/'images').glob('*.tar*')),f'Missing image archive: {name}'
print(f'Validated: {len(names)} mining lv0 profiles in {batch.name}.',flush=True)
if requested:
 assert requested in names,f'Unknown task: {requested}'
 print(f'Selected task: {requested}',flush=True)
PY
(( CHECK == 0 )) || exit 0
[[ -f "$AGENT_BINARY" ]] || { echo "Missing agent binary: $AGENT_BINARY" >&2; exit 2; }
if (( DRY_RUN )); then
  cd "$ROOT"
  exec uv run examples/run_agent.py --pj-user-batch "$PJ_USER_BATCH_DIR" \
    --pj-user-workers "$WORKERS" --model "$MODEL" --timeout "$TIMEOUT" \
    --out-dir "${DRY_RUN_OUT_DIR:-$ROOT/out/${BATCH_NAME}_dry_run}" \
    --proxy-admin-key dry-run-only --pj-max-turns "$MAX_TURNS" \
    --pj-user-agent-binary "$AGENT_BINARY" --dry-run
fi
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before starting}"
if [[ "$SETUP" == 1 ]]; then
  bash "$ROOT/setup_cn.sh"
fi
command -v uv >/dev/null || { echo 'uv is required; see docs/setup.md' >&2; exit 2; }
# Check Harbor early, in this project's uv environment, before Docker or image loading.
(cd "$ROOT" && uv run --extra proxy python -c 'import shutil; print("harbor:", shutil.which("harbor")); assert shutil.which("harbor"), "Harbor 未安装或不在 PATH 中"')
command -v docker >/dev/null || { echo 'docker is required' >&2; exit 2; }
mkdir -p "$OUT_DIR"
export OUT_DIR="$(cd -- "$OUT_DIR" && pwd)"
command -v flock >/dev/null || { echo 'flock is required' >&2; exit 2; }
exec 9>"$OUT_DIR/.launcher.lock"
flock -n 9 || { echo "Another launcher is using $OUT_DIR" >&2; exit 2; }
LOG="$OUT_DIR/launcher-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; echo "Launcher ended with exit $rc; results: $OUT_DIR; log: $LOG"' EXIT
cd "$ROOT"
docker info >/dev/null
docker compose version
if ! docker image inspect ubuntu/squid:latest >/dev/null 2>&1; then
  if [[ -n "${EXPGYM_OFFLINE_ROOT:-}" && -f "$EXPGYM_OFFLINE_ROOT/firewall-image.tar" ]]; then
    docker load -i "$EXPGYM_OFFLINE_ROOT/firewall-image.tar"
  else
    echo 'Missing ubuntu/squid:latest; unpack the offline environment first.' >&2
    exit 2
  fi
fi
args=(--pj-user-max-turns "$MAX_TURNS" --pj-user-agent-binary "$AGENT_BINARY")
if [[ -z "$TASK_NAME" ]]; then args=(--batch "${args[@]}"); fi
if [[ "$RESUME" == 1 ]]; then args+=(--pj-user-resume); fi
echo "Starting $BATCH_NAME: model=$MODEL workers=$WORKERS max_turns=$MAX_TURNS resume=$RESUME"
if [[ -n "$TASK_NAME" ]]; then
  python3 - "$PJ_USER_BATCH_DIR" "$TASK_NAME" <<'LOAD_SHARED'
import hashlib,json,subprocess,sys
from pathlib import Path
batch=Path(sys.argv[1]); name=sys.argv[2]
entries=json.loads((batch/name/'image-lock.json').read_text())['images']
seen=set()
for entry in entries:
    if entry.get('scope') not in ('runtime','controller'): continue
    ref=entry['ref']; sha=entry['archive_sha256']
    if subprocess.run(['docker','image','inspect',ref],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0: continue
    if sha in seen: continue
    seen.add(sha)
    candidates=[p for p in (batch/'shared-images').glob('*.tar.gz') if p.stat().st_size==entry['archive_size']]
    archive=None
    for p in candidates:
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
        if h.hexdigest()==sha: archive=p; break
    if archive is None: raise SystemExit(f'No matching shared image for {ref}: {sha}')
    print(f'Loading shared image archive: {archive}',flush=True)
    subprocess.run(['docker','load','-i',str(archive)],check=True)
LOAD_SHARED
fi
# This entry point loads shared images, runs pre_run.py, and then the batch.
# pj_user.py loads per-task images and records failures in each result.json.
if [[ -n "$TASK_NAME" ]]; then
  bash "$ROOT/scripts/run_pj_exploit.sh" "$TASK_NAME" "${args[@]}"
else
  bash "$ROOT/scripts/run_pj_exploit.sh" "${args[@]}"
fi
