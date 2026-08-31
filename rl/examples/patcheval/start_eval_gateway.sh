#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

: "${PATCH_EVAL_API_KEY:?Set PATCH_EVAL_API_KEY before starting the Eval Gateway}"

PATCH_EVAL_API_BASE="${PATCH_EVAL_API_BASE:-http://35.220.164.252:3888/v1}"
PATCH_EVAL_MODEL="${PATCH_EVAL_MODEL:-bailian/deepseek-v4-flash}"
EVAL_GATEWAY_HOST="${EVAL_GATEWAY_HOST:-0.0.0.0}"
EVAL_GATEWAY_PORT="${EVAL_GATEWAY_PORT:-18000}"
EVAL_GATEWAY_DB="${EVAL_GATEWAY_DB:-${SCRIPT_DIR}/patcheval_eval_gateway.db}"
EVAL_GATEWAY_CONFIG="${EVAL_GATEWAY_CONFIG:-${SCRIPT_DIR}/patcheval_eval_gateway.yaml}"

PATCH_EVAL_API_BASE="${PATCH_EVAL_API_BASE}" \
PATCH_EVAL_API_KEY="${PATCH_EVAL_API_KEY}" \
PATCH_EVAL_MODEL="${PATCH_EVAL_MODEL}" \
EVAL_GATEWAY_HOST="${EVAL_GATEWAY_HOST}" \
EVAL_GATEWAY_PORT="${EVAL_GATEWAY_PORT}" \
EVAL_GATEWAY_DB="${EVAL_GATEWAY_DB}" \
EVAL_GATEWAY_CONFIG="${EVAL_GATEWAY_CONFIG}" \
python3 - <<'PY'
import os
from pathlib import Path

import yaml

config = {
    "listen_host": os.environ["EVAL_GATEWAY_HOST"],
    "listen_port": int(os.environ["EVAL_GATEWAY_PORT"]),
    "base_session_path": "/v1/sessions",
    "max_steps": -1,
    "storage_type": "sqlite",
    "storage_config": {"db_url": f"sqlite:///{Path(os.environ['EVAL_GATEWAY_DB']).resolve()}"},
    "llm_routes": {
        os.environ["PATCH_EVAL_MODEL"]: {
            "base_url": os.environ["PATCH_EVAL_API_BASE"].rstrip("/") + "/",
            "api_key": os.environ["PATCH_EVAL_API_KEY"],
            "supports_stream": True,
            "max_concurrency": 64,
        }
    },
}
path = Path(os.environ["EVAL_GATEWAY_CONFIG"])
path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
path.chmod(0o600)
PY

echo "Starting Eval Gateway on ${EVAL_GATEWAY_HOST}:${EVAL_GATEWAY_PORT}"
echo "Route model: ${PATCH_EVAL_MODEL}"
exec python3 -m gateway --config "${EVAL_GATEWAY_CONFIG}"
