#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

export AIEVOBOX_POLICY_ID="${DEFENDER_POLICY_ID}"
export AIEVOBOX_ROLLOUT_OWNER="${AIEVOBOX_ROLLOUT_OWNER:-false}"
export LLM_PROXY_HOST="${DEFENDER_LLM_PROXY_HOST}"
export LLM_PROXY_PORT="${DEFENDER_LLM_PROXY_PORT}"

export WANDB_GROUP="${WANDB_GROUP:-multi-qagym-defender-${AIEVOBOX_RUN_ID}}"
export SAVE_DIR="${SAVE_DIR:-/mnt/shared-storage-user/evobox-share-gpfs2/leishanzhe/model/checkpoints/rl/multi_qagym/defender/${AIEVOBOX_RUN_ID}}"

# Non-owner trainer should not kill or restart the shared Ray runtime.
export SKIP_RUNTIME_CLEANUP="${SKIP_RUNTIME_CLEANUP:-1}"
export RAY_RESTART="${RAY_RESTART:-0}"
export RAY_START="${RAY_START:-0}"

exec bash "${SCRIPT_DIR}/run_slime_generator.sh"
