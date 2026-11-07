#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRAPH_ROOT="$(cd -- "$EXPERIMENT/../.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
: "${TRAIN_MODEL:?Set TRAIN_MODEL to your local original Qwen2.5-7B-Instruct directory}"
DATA="$GRAPH_ROOT/data/3_5_1_3_6"
"$PYTHON_BIN" "$GRAPH_ROOT/tools/restore_data.py"
export TRAIN_EPISODES="$DATA/train_episodes.json"
export TEST_EPISODES="$DATA/test_episodes.json"
exec bash "$EXPERIMENT/scripts/run_direct_batch4_1epoch.sh" "${1:-prepare}"
