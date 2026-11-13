#!/usr/bin/env bash
# Independent 3_5_1_2 pipeline: prepare -> 7B LoRA GRPO -> merge -> both test sets.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRAPH_ROOT="$(cd -- "$ROOT/../.." && pwd)"
PY="${PYTHON_BIN:-$GRAPH_ROOT/.venv-3_5_1_2/bin/python}"
RUN="${RUN_DIR:-$ROOT/runs/3_5_1_2_$(date +%Y%m%d_%H%M%S)}"
MODEL="${TRAIN_MODEL:?Set TRAIN_MODEL to the local original Qwen2.5-7B-Instruct directory}"
SOURCE="${TRAIN_EPISODES:-$GRAPH_ROOT/2000_oneclass_sorted/train_grpo_episodes_redesign.json}"
TEST_ONECLASS="${TEST_ONECLASS_EPISODES:-$GRAPH_ROOT/data/test_data_600_oneclass/test_grpo_episodes.json}"
TEST_NOTONECLASS="${TEST_NOTONECLASS_EPISODES:-$GRAPH_ROOT/data/test_data_600_notoneclass/test_grpo_episodes.json}"
if [[ -n "${TEST_EPISODES:-}" ]]; then
    echo "Use TEST_ONECLASS_EPISODES and TEST_NOTONECLASS_EPISODES; both test sets are required." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUs <<< "$CUDA_VISIBLE_DEVICES"
NGPU="${#GPUs[@]}"
# 8x H200 starting profile; tune batch/accumulation after measuring throughput.
G="${NUM_GENERATIONS:-6}" BATCH="${TRAIN_BATCH_SIZE:-48}" ACCUM="${GRAD_ACCUM:-1}"
for value in "$G" "$BATCH" "$ACCUM"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'G, batch and accumulation must be positive integers.' >&2; exit 2; }
done
[[ "$G" == 6 ]] || { echo 'This experiment keeps six trajectories per source (G=6).' >&2; exit 2; }
MICROBATCH=$((NGPU * BATCH))
(( MICROBATCH % G == 0 )) || { echo 'Global microbatch must contain complete G-trajectory groups.' >&2; exit 2; }
printf 'Batch profile: GPUs=%s, per-device=%s, accumulation=%s, G=%s, normal update=%s trajectories/%s sources\n' \
    "$NGPU" "$BATCH" "$ACCUM" "$G" "$((MICROBATCH * ACCUM))" "$((MICROBATCH * ACCUM / G))"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PREPARE_TASK_FLAGS=()
if [[ "${REQUIRE_TEST_TASK_COVERAGE:-0}" == 1 ]]; then PREPARE_TASK_FLAGS=(--require-test-task-coverage); fi
STAGE="${1:-all}"
case "$STAGE" in all|prepare|train|merge|test) ;; *) echo 'Usage: run_direct_batch4_1epoch.sh [all|prepare|train|merge|test]' >&2; exit 2;; esac
if [[ "$STAGE" == all || "$STAGE" == prepare ]]; then
    [[ ! -e "$RUN" ]] || { echo "Refusing existing run: $RUN" >&2; exit 1; }
fi
mkdir -p "$RUN"
phase="$STAGE"
trap 'rc=$?; if ((rc != 0)); then printf "failed: %s (exit %s)\n" "$phase" "$rc" > "$RUN/status.txt"; fi' EXIT
if [[ "$STAGE" == all || "$STAGE" == prepare ]]; then
    mkdir "$RUN/code_snapshot"
    cp -r "$ROOT/code" "$ROOT/scripts" "$ROOT/configs" "$ROOT/source_manifest.json" "$ROOT/requirements-native.txt" "$RUN/code_snapshot/"
    phase=prepare
    "$PY" "$ROOT/code/prepare_data.py" --source "$SOURCE" --test-source "$TEST_ONECLASS" --test-notoneclass-source "$TEST_NOTONECLASS" --model "$MODEL" \
        --output "$RUN/data" --fraction "${TRAIN_FRACTION:-1}" --partition "${TRAIN_PARTITION:-prefix}" \
        --num-generations "$G" --num-gpus "$NGPU" --train-batch-size "$BATCH" --gradient-accumulation-steps "$ACCUM" \
        --expected-test-count "${EXPECTED_TEST_COUNT:-600}" "${PREPARE_TASK_FLAGS[@]}" | tee "$RUN/prepare.log"
fi
if [[ "$STAGE" == all || "$STAGE" == train ]]; then
    phase=train
    "$PY" -m torch.distributed.run --standalone --nproc_per_node "$NGPU" "$ROOT/code/train_grpo.py" \
        --model_id "$MODEL" --data_path "$RUN/data/train_episodes.json" --output_dir "$RUN/grpo" \
        --deepspeed "${DEEPSPEED_CONFIG:-$ROOT/configs/zero3_7b.json}" \
        --per_device_train_batch_size "$BATCH" --per_device_eval_batch_size "$BATCH" --gradient_accumulation_steps "$ACCUM" \
        --num_generations "$G" --steps_per_generation 1 --num_iterations 1 --max_tool_calling_iterations 0 --num_train_epochs "${FINAL_EPOCH:-1}" \
        --lora_rank 32 --lora_alpha 32 --lora_dropout 0.05 --beta 0.01 --loss_type grpo \
        --learning_rate "${LEARNING_RATE:-2e-6}" --warmup_steps "${WARMUP_STEPS:-0.05}" --lr_scheduler_type cosine --weight_decay 0.01 --max_grad_norm 1.0 --optim adamw_torch \
        --max_prompt_length 8192 --max_completion_length "${PREDICTION_TOKENS:-512}" \
        --prediction_max_new_tokens "${PREDICTION_TOKENS:-512}" --logprob_micro_batch_size "${LOGPROB_MICRO_BATCH_SIZE:-4}" \
        --temperature 1.0 --top_p 0.95 --top_k 50 \
        --disable_flash_attn2 False --disable_dropout True --shuffle_dataset False --use_vllm False --bf16 True --fp16 False --tf32 True \
        --gradient_checkpointing True --gradient_checkpointing_kwargs '{"use_reentrant":true}' --use_liger_kernel True --logging_steps 1 \
        --save_strategy steps --save_steps "${SAVE_STEPS:-5}" --save_total_limit 3 --report_to none \
        --seed 42 --remove_unused_columns False --dataloader_num_workers 0 \
        2>&1 | tee "$RUN/train.log"
fi
if [[ "$STAGE" == all || "$STAGE" == merge ]]; then
    phase=merge
    "$PY" "$ROOT/code/merge_adapter.py" --adapter "$RUN/grpo/final_adapter" --output "$RUN/merged_model" \
        2>&1 | tee "$RUN/merge.log"
fi
if [[ "$STAGE" == all || "$STAGE" == test ]]; then
    phase=test
    # Check both inputs before starting any evaluation.
    for variant in oneclass notoneclass; do
        [[ -f "$RUN/data/test_${variant}_episodes.json" ]] || {
            echo "Missing prepared $variant test set; prepare a new run with both inputs." >&2; exit 1;
        }
    done
    for variant in oneclass notoneclass; do
        CUDA_VISIBLE_DEVICES="${EVAL_GPU:-${GPUs[0]}}" "$PY" "$ROOT/code/evaluate_grpo_600.py" \
            --model "$RUN/merged_model" --data "$RUN/data/test_${variant}_episodes.json" --condition noisy_policy \
            --rollout-config "$RUN/grpo/experiment_config.json" --batch-size "${EVAL_BATCH_SIZE:-8}" \
            --expected-count "${EXPECTED_TEST_COUNT:-600}" \
            --output-dir "$RUN/test/$variant/noisy_policy" 2>&1 | tee "$RUN/test_${variant}_noisy_policy.log"
    done
fi
printf 'complete: %s\n' "$STAGE" > "$RUN/status.txt"
printf 'Run: %s\n' "$RUN"
