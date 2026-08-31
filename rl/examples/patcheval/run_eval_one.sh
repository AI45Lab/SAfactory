#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

: "${PATCH_EVAL_API_KEY:?Set PATCH_EVAL_API_KEY before running}"

export DOCKER_HOST="${DOCKER_HOST:-tcp://100.99.17.62:2376}"
export PATCH_EVAL_BASELINE="${PATCH_EVAL_BASELINE:-llm}"
export PATCH_EVAL_SETTING="${PATCH_EVAL_SETTING:-s1.1}"
export PATCH_EVAL_TASK_LIMIT=1
export PATCH_EVAL_POOL_SIZE=1
export PATCH_EVAL_DOCKER_STARTUP_CONCURRENCY=1

exec "${SCRIPT_DIR}/run_eval.sh"
