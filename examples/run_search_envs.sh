#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PROJECT_ROOT}/examples/base_eval.py" \
  --env-config-yaml "${PROJECT_ROOT}/env/search/search_env_configs.yaml" \
  --max-workers 4 \
  --max-steps 10 \
  --visual-save-path "${PROJECT_ROOT}/visualize/search_eval" \
  --llm-api-key "EMPTY" \
  --llm-base-url "http://localhost:8001/v1" \
  --llm-model "Qwen3-30B-Instruct" \
  --llm-temperature 0.3 \
  "$@"
