#!/usr/bin/env bash
set -euo pipefail

#!/usr/bin/env bash
 
# Ensure PYTHONPATH includes AIEvoBox
export PYTHONPATH="${PYTHONPATH:-}:/root/AIEvoBox"

# Configure DB for this run in repo path. Override by exporting AIEVOBOX_DB_URL before running.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DB_DIR="${SCRIPT_DIR}/db"
mkdir -p "$DB_DIR"
export AIEVOBOX_DB_URL="${AIEVOBOX_DB_URL:-sqlite:////${DB_DIR}/trading_rollout.db}"

echo "Using DB: $AIEVOBOX_DB_URL"

# Default Trading env parameters (override by exporting before running)
export TRADING_DATA_DIR=${TRADING_DATA_DIR:-"/mnt/shared-storage-user/evobox-share/chenxinquan/data/trading/trading"}
export TRADING_PRICE_FILE=${TRADING_PRICE_FILE:-"AMZN.csv"}
export TRADING_TWEET_FILE=${TRADING_TWEET_FILE:-"amzn_stockmo.csv"}
export TRADING_WINDOW=${TRADING_WINDOW:-7}
export TRADING_MAX_STEPS=${TRADING_MAX_STEPS:-2}

echo "TRADING_DATA_DIR=$TRADING_DATA_DIR"
echo "TRADING_PRICE_FILE=$TRADING_PRICE_FILE"
echo "TRADING_TWEET_FILE=$TRADING_TWEET_FILE"
echo "TRADING_WINDOW=$TRADING_WINDOW"
echo "TRADING_MAX_STEPS=$TRADING_MAX_STEPS"

# Skip rendering to speed up and avoid font warnings (1 to skip)
export AIEVOBOX_NO_RENDER=${AIEVOBOX_NO_RENDER:-1}
echo "AIEVOBOX_NO_RENDER=$AIEVOBOX_NO_RENDER"
python3 trading-aievobox/rollout_buffer_db_server.py
