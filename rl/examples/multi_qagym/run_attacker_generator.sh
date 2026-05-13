#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

export AIEVOBOX_POLICY_ID="${ATTACKER_POLICY_ID}"
export AIEVOBOX_ROLLOUT_OWNER="${AIEVOBOX_ROLLOUT_OWNER:-true}"
export LLM_PROXY_HOST="${ATTACKER_LLM_PROXY_HOST}"
export LLM_PROXY_PORT="${ATTACKER_LLM_PROXY_PORT}"

export WANDB_GROUP="${WANDB_GROUP:-multi-qagym-attacker-${AIEVOBOX_RUN_ID}}"
export SAVE_DIR="${SAVE_DIR:-/mnt/shared-storage-user/evobox-share-gpfs2/leishanzhe/model/checkpoints/rl/multi_qagym/attacker/${AIEVOBOX_RUN_ID}}"

# The first trainer owns shared rollout and may restart Ray.
export SKIP_RUNTIME_CLEANUP="${SKIP_RUNTIME_CLEANUP:-0}"
export RAY_START="${RAY_START:-1}"

exec bash "${SCRIPT_DIR}/run_slime_generator.sh"
