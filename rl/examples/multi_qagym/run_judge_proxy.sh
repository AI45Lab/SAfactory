#!/usr/bin/env bash

ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit -n 65536 (current: $(ulimit -n))"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${AIEVOBOX_ROOT}:${PYTHONPATH:-}"
export JUDGE_PROXY_HOST="${JUDGE_PROXY_HOST:-0.0.0.0}"
export JUDGE_PROXY_PORT="${JUDGE_PROXY_PORT:-18892}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://35.220.164.252:3888/v1}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-sk-pxhM1nUxEc0nCDMyMCkhRklp8cJJIT6VFuegNQ1ygNv8WAj9}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gpt-4o-mini}"
export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.1}"
export JUDGE_SUCCESS_THRESHOLD="${JUDGE_SUCCESS_THRESHOLD:-5}"
export JUDGE_TARGET_MODEL_HOLDER="${JUDGE_TARGET_MODEL_HOLDER:-OpenAI}"
export JUDGE_VERBOSE="${JUDGE_VERBOSE:-0}"

echo "Starting Multi-QAGym Judge Proxy..."
echo "  Listen: ${JUDGE_PROXY_HOST}:${JUDGE_PROXY_PORT}"
echo "  Judge base URL: ${JUDGE_BASE_URL}"
echo "  Judge model: ${JUDGE_MODEL_NAME}"
echo "  Max concurrency: ${JUDGE_PROXY_MAX_CONCURRENCY}"
echo "  Timeout: ${JUDGE_TIMEOUT_S}s"
echo "  Dump judge IO: ${JUDGE_PROXY_DUMP_INPUTS:-0}"
echo "  Dump dir: ${JUDGE_PROXY_DUMP_DIR:-}"

"${PYTHON_BIN}" "${AIEVOBOX_ROOT}/rl/judge_proxy.py"
