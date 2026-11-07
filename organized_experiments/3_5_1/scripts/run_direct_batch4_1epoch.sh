#!/usr/bin/env bash
# Independent 3_5_1 pipeline: prepare -> 7B LoRA GRPO -> merge -> same-history tests.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-/opt/data/private/.venvs/graphrl-3_3-trl-latest/bin/python}"
RUN="${RUN_DIR:-$ROOT/runs/3_5_1_$(date +%Y%m%d_%H%M%S)}"
MODEL="${TRAIN_MODEL:-/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct}"
SOURCE="${TRAIN_EPISODES:-/opt/data/private/GraphRL/GraphRL_train_data_clean20_badgraph80_degree_sorted/train_grpo_episodes_redesign.json}"
TEST_SOURCE="${TEST_EPISODES:-/opt/data/private/GraphRL/outputs/redesign_binary_rewrite_positive_sequence_20261014_225139/data/redesign/test_episodes.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
IFS=',' read -r -a GPUs <<< "$CUDA_VISIBLE_DEVICES"
NGPU="${#GPUs[@]}"
BATCH="${TRAIN_BATCH_SIZE:-2}" ACCUM="${GRAD_ACCUM:-2}" G="${NUM_GENERATIONS:-6}"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
    "$PY" "$ROOT/code/prepare_data.py" --source "$SOURCE" --test-source "$TEST_SOURCE" --model "$MODEL" \
        --output "$RUN/data" --fraction "${TRAIN_FRACTION:-1}" --partition "${TRAIN_PARTITION:-prefix}" \
        --num-generations "$G" --num-gpus "$NGPU" --train-batch-size "$BATCH" | tee "$RUN/prepare.log"
fi
if [[ "$STAGE" == all || "$STAGE" == train ]]; then
    phase=train
    "$PY" -m torch.distributed.run --standalone --nproc_per_node "$NGPU" "$ROOT/code/train_grpo.py" \
        --model_id "$MODEL" --data_path "$RUN/data/train_episodes.json" --output_dir "$RUN/grpo" \
        --deepspeed "${DEEPSPEED_CONFIG:-$ROOT/configs/zero3_7b.json}" \
        --per_device_train_batch_size "$BATCH" --per_device_eval_batch_size "$BATCH" --gradient_accumulation_steps "$ACCUM" \
        --num_generations "$G" --steps_per_generation 1 --num_iterations 1 --max_tool_calling_iterations 0 --num_train_epochs "${FINAL_EPOCH:-1}" \
        --lora_rank 32 --lora_alpha 32 --lora_dropout 0.05 --beta 0.01 --loss_type grpo \
        --learning_rate 2e-6 --warmup_steps 55 --lr_scheduler_type cosine --weight_decay 0.01 --max_grad_norm 1.0 --optim adamw_torch \
        --max_prompt_length 8192 --max_completion_length "${PREDICTION_TOKENS:-512}" \
        --prediction_max_new_tokens "${PREDICTION_TOKENS:-512}" --logprob_micro_batch_size 1 \
        --temperature 1.0 --top_p 0.95 --top_k 50 \
        --disable_flash_attn2 False --disable_dropout True --shuffle_dataset False --use_vllm False --bf16 True --fp16 False --tf32 True \
        --gradient_checkpointing True --gradient_checkpointing_kwargs '{"use_reentrant":true}' --use_liger_kernel True --logging_steps 1 \
        --save_strategy steps --save_steps 100 --save_total_limit 3 --report_to none \
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
    for condition in noisy_policy clean_policy noisy_baseline clean_baseline; do
        model="$RUN/merged_model"
        if [[ "$condition" == *_baseline ]]; then model="$MODEL"; fi
        CUDA_VISIBLE_DEVICES="${EVAL_GPU:-${GPUs[0]}}" "$PY" "$ROOT/code/evaluate_grpo_600.py" \
            --model "$model" --data "$RUN/data/test_episodes.json" --condition "$condition" \
            --rollout-config "$RUN/grpo/experiment_config.json" --batch-size "${EVAL_BATCH_SIZE:-1}" \
            --output-dir "$RUN/test/$condition" 2>&1 | tee "$RUN/test_$condition.log"
    done
fi
printf 'complete: %s\n' "$STAGE" > "$RUN/status.txt"
printf 'Run: %s\n' "$RUN"
