#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
: "${TRAIN_MODEL:?Set TRAIN_MODEL to your local original Qwen2.5-7B-Instruct directory}"
# TRAIN_EPISODES selects training; TEST_ONECLASS_EPISODES and
# TEST_NOTONECLASS_EPISODES select the two mandatory evaluation inputs.
exec bash "$EXPERIMENT/scripts/run_direct_batch4_1epoch.sh" "${1:-prepare}"
