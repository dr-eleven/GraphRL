#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
: "${TRAIN_MODEL:?Set TRAIN_MODEL to your local original Qwen2.5-7B-Instruct directory}"
# TRAIN_EPISODES and TEST_EPISODES may override the independent pipeline defaults.
exec bash "$EXPERIMENT/scripts/run_direct_batch4_1epoch.sh" "${1:-prepare}"
