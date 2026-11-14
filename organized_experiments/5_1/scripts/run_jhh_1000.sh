#!/usr/bin/env bash
# Resource settings referenced to 3_6_1; 5_1 remains one binary-reward epoch.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRAPH_ROOT="$(cd -- "$ROOT/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in "$HOME/.venvs/graphrl-3_5_1-3_6/bin/python" /opt/data/private/.venvs/graphrl-3_3-trl-latest/bin/python; do
    if [[ -x "$candidate" ]]; then export PYTHON_BIN="$candidate"; break; fi
  done
fi
if [[ -z "${TRAIN_MODEL:-}" ]]; then
  for candidate in "$HOME/models/Qwen2.5-7B-Instruct" /opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct; do
    if [[ -f "$candidate/config.json" ]]; then export TRAIN_MODEL="$candidate"; break; fi
  done
fi
: "${PYTHON_BIN:?Set PYTHON_BIN to the compatible Python 3.11 environment}"
: "${TRAIN_MODEL:?Set TRAIN_MODEL to the original Qwen2.5-7B-Instruct weights}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}" GRAD_ACCUM="${GRAD_ACCUM:-1}" NUM_GENERATIONS=6
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$ROOT/configs/zero2_gpu.json}"
export TRAIN_EPISODES="$GRAPH_ROOT/1000_oneclass_sorted/train_grpo_episodes_redesign.json"
export TEST_ONECLASS_EPISODES="$GRAPH_ROOT/600_test_data_oneclass/test_grpo_episodes.json"
export TEST_NOTONECLASS_EPISODES="$GRAPH_ROOT/600_test_data_notoneclass/test_grpo_episodes.json"
export TRAIN_FRACTION=1 FINAL_EPOCH=1 EVAL_CONDITIONS=restored SAVE_STEPS=10
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}" GENERATION_MICRO_BATCH_SIZE="${GENERATION_MICRO_BATCH_SIZE:-1}"
export RUN_DIR="${RUN_DIR:-$ROOT/runs/5_1_1000_oneclass_$(date -u +%Y%m%d_%H%M%S)}"
exec bash "$ROOT/scripts/run_binary_1epoch.sh" "${1:-all}"
