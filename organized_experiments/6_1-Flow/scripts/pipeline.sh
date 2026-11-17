#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
export FLOW_RUN_DIR="${RUN_DIR:-$ROOT/runs/6_1_Flow_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$FLOW_RUN_DIR"
"$PY" "$ROOT/code/resolve_settings.py" --source "${SETTINGS_FILE:-$ROOT/configs/settings.json}" --output "$FLOW_RUN_DIR/runtime_settings.json"
export FLOW_SETTINGS="$FLOW_RUN_DIR/runtime_settings.json"
export FLOW_BASE_MODEL="$("$PY" -c 'import json,os; print(json.load(open(os.environ["FLOW_SETTINGS"]))["model"])')"
export PYTHONPATH="$ROOT/code:$ROOT/upstream${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}" VLLM_USE_V1=1 WANDB_MODE=disabled
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
STAGE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$STAGE" in prepare|train|merge|test|all) ;; *) echo 'Usage: pipeline.sh prepare|train|merge|test|all' >&2; exit 2;; esac
mkdir -p "$FLOW_RUN_DIR"
STATUS="$FLOW_RUN_DIR/pipeline_status.json"
fixed_pid='' planner_pid=''
cleanup() {
  for task_pid in "$fixed_pid" "$planner_pid"; do
    if [[ -n "$task_pid" ]]; then kill -- "-$task_pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
on_error() {
  local rc=$?
  "$PY" "$ROOT/code/set_status.py" "$STATUS" failed "$STAGE" "$rc"
  exit "$rc"
}
trap on_error ERR
wait_server() {
  "$PY" "$ROOT/code/wait_server.py" "$1" "$2"
}
start_fixed() {
  CUDA_VISIBLE_DEVICES="${FIXED_GPUS:-6,7}" setsid "$PY" "$ROOT/code/serve.py" --model "$FLOW_BASE_MODEL" --served-model-name graphrl-fixed --host 127.0.0.1 --port 8621 --dtype bfloat16 --tensor-parallel-size "${FIXED_TP:-2}" --max-model-len 32768 --gpu-memory-utilization 0.85 --max-num-seqs "${FIXED_MAX_SEQS:-12}" --enforce-eager > "$FLOW_RUN_DIR/fixed_server.log" 2>&1 &
  fixed_pid=$!
  wait_server http://127.0.0.1:8621/v1/models "$fixed_pid"
}
if [[ "$STAGE" == all || "$STAGE" == prepare ]]; then
  "$PY" "$ROOT/code/set_status.py" "$STATUS" running prepare
  args=(); if [[ -n "${TRAIN_LIMIT:-}" ]]; then args+=(--limit "$TRAIN_LIMIT"); fi
  "$PY" "$ROOT/code/prepare_data.py" --run-dir "$FLOW_RUN_DIR" "${args[@]}" | tee "$FLOW_RUN_DIR/prepare.log"
  cp "$FLOW_SETTINGS" "$FLOW_RUN_DIR/settings.json"
  mkdir -p "$FLOW_RUN_DIR/code_snapshot"
  cp -a "$ROOT/code" "$ROOT/configs" "$ROOT/scripts" "$ROOT/upstream" "$ROOT/source_manifest.json" "$ROOT/plan.md" "$ROOT/README.md" "$ROOT/requirements-native.txt" "$FLOW_RUN_DIR/code_snapshot/"
fi
if [[ "$STAGE" == all || "$STAGE" == train ]]; then
  "$PY" "$ROOT/code/set_status.py" "$STATUS" running train
  [[ -f "$FLOW_RUN_DIR/data/preflight.json" ]]
  [[ ! -e "$FLOW_RUN_DIR/checkpoints" ]] || { echo 'Existing checkpoints: use a fresh RUN_DIR.' >&2; exit 2; }
  "$PY" "$ROOT/code/check_launch.py" --check-gpus | tee "$FLOW_RUN_DIR/launch_check_$STAGE.json"
  start_fixed
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3,4,5}" "$PY" "$ROOT/code/train.py" "$@" 2>&1 | tee "$FLOW_RUN_DIR/train.log"
  kill -- "-$fixed_pid" 2>/dev/null || true; wait "$fixed_pid" 2>/dev/null || true; fixed_pid=''
fi
if [[ "$STAGE" == all || "$STAGE" == merge ]]; then
  "$PY" "$ROOT/code/set_status.py" "$STATUS" running merge
  ADAPTER="$($PY "$ROOT/code/latest_adapter.py" "$FLOW_RUN_DIR/checkpoints")"
  CUDA_VISIBLE_DEVICES='' "$PY" "$ROOT/code/merge_adapter.py" --adapter "$ADAPTER" --output "$FLOW_RUN_DIR/merged_model" 2>&1 | tee "$FLOW_RUN_DIR/merge.log"
fi
if [[ "$STAGE" == all || "$STAGE" == test ]]; then
  "$PY" "$ROOT/code/set_status.py" "$STATUS" running test
  "$PY" "$ROOT/code/check_launch.py" --check-gpus | tee "$FLOW_RUN_DIR/launch_check_$STAGE.json"
  start_fixed
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" setsid "$PY" "$ROOT/code/serve.py" --model "$FLOW_RUN_DIR/merged_model" --served-model-name graphrl-planner --host 127.0.0.1 --port 8622 --dtype bfloat16 --tensor-parallel-size "${EVAL_TP:-1}" --max-model-len 32768 --gpu-memory-utilization 0.85 --max-num-seqs 2 --enforce-eager > "$FLOW_RUN_DIR/planner_server.log" 2>&1 &
  planner_pid=$!
  wait_server http://127.0.0.1:8622/v1/models "$planner_pid"
  eval_args=(); eval_folder=test
  if [[ -n "${EVAL_LIMIT:-}" ]]; then eval_args+=(--limit "$EVAL_LIMIT"); eval_folder=smoke_test; fi
  for split in test_data_600_oneclass test_data_600_notoneclass; do
    "$PY" "$ROOT/code/evaluate.py" --data "$FLOW_RUN_DIR/data/$split.json" --output-dir "$FLOW_RUN_DIR/$eval_folder/$split/restored" "${eval_args[@]}" 2>&1 | tee "$FLOW_RUN_DIR/test_$split.log"
  done
fi
"$PY" "$ROOT/code/set_status.py" "$STATUS" complete "$STAGE"
