#!/usr/bin/env bash
# Quick test script for MCGym environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIEVOBOX_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "================================================"
echo "Running MCGym Environment Tests"
echo "================================================"
echo ""
echo "AIEvoBox Root: ${AIEVOBOX_ROOT}"
echo "Test Script: ${SCRIPT_DIR}/test_mc_env.py"
echo ""

cd "${AIEVOBOX_ROOT}"

python env/mc/test/test_mc_env.py "$@"

