#!/usr/bin/env bash
# Explicit stages. Invoking this launcher starts work; code implementation does not run it automatically.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$(basename "$ROOT")"
GRAPH_ROOT="$(cd -- "$ROOT/../.." && pwd)"
PY="${PYTHON_BIN:-/opt/data/private/.venvs/graphrl-3_3-trl-latest/bin/python}"
RUN="${RUN_DIR:-$ROOT/runs/${PROJECT}_$(date +%Y%m%d_%H%M%S)}"
MODEL="${TRAIN_MODEL:-/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct}"
SOURCE="${TRAIN_EPISODES:-$GRAPH_ROOT/1000_oneclass_sorted/train_grpo_episodes_redesign.json}"
TEST_NOTONECLASS="${TEST_NOTONECLASS_EPISODES:-$GRAPH_ROOT/600_test_data_notoneclass/test_grpo_episodes.json}"
TEST_ONECLASS="${TEST_ONECLASS_EPISODES:-$GRAPH_ROOT/600_test_data_oneclass/test_grpo_episodes.json}"
export TRAIN_MODEL="$MODEL"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NGPU="${#GPUS[@]}"
BATCH="${TRAIN_BATCH_SIZE:-12}" ACCUM="${GRAD_ACCUM:-1}" G="${NUM_GENERATIONS:-6}"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
STAGE="${1:-prepare}"
case "$STAGE" in all|prepare|train|merge|test) ;; *) echo 'Usage: run_binary_1epoch.sh [prepare|train|merge|test|all]' >&2; exit 2;; esac
if [[ "$STAGE" == all || "$STAGE" == prepare ]]; then
    [[ ! -e "$RUN" ]] || { echo "Refusing existing run: $RUN" >&2; exit 1; }
    mkdir -p "$RUN/code_snapshot"
    cp -r "$ROOT/code" "$ROOT/scripts" "$ROOT/configs" "$ROOT/source_manifest.json" "$ROOT/requirements-native.txt" "$RUN/code_snapshot/"
    "$PY" "$ROOT/code/prepare_data.py" --source "$SOURCE" --model "$MODEL" \
        --test-source "600_test_data_notoneclass=$TEST_NOTONECLASS" \
        --test-source "600_test_data_oneclass=$TEST_ONECLASS" \
        --output "$RUN/data" --fraction "${TRAIN_FRACTION:-1}" --partition "${TRAIN_PARTITION:-prefix}" \
        --num-generations "$G" --num-gpus "$NGPU" --train-batch-size "$BATCH" | tee "$RUN/prepare.log"
fi
[[ -f "$RUN/data/preflight.json" ]] || { echo 'Prepare this RUN_DIR before train/merge/test.' >&2; exit 2; }
CODE="$RUN/code_snapshot/code"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$STAGE" == all || "$STAGE" == train ]]; then
    "$PY" -m torch.distributed.run --standalone --nproc_per_node "$NGPU" "$CODE/train_grpo.py" \
        --model_id "$MODEL" --data_path "$RUN/data/train_episodes.json" --output_dir "$RUN/grpo" \
        --deepspeed "${DEEPSPEED_CONFIG:-$RUN/code_snapshot/configs/zero2_gpu.json}" \
        --per_device_train_batch_size "$BATCH" --per_device_eval_batch_size "$BATCH" --gradient_accumulation_steps "$ACCUM" \
        --num_generations "$G" --steps_per_generation 1 --num_iterations 1 --max_tool_calling_iterations 10 --max_model_turns 10 --num_train_epochs "${FINAL_EPOCH:-1}" \
        --lora_rank 32 --lora_alpha 32 --lora_dropout 0.05 --beta 0.01 --loss_type grpo --scale_rewards group \
        --learning_rate 2e-6 --warmup_steps 55 --lr_scheduler_type cosine --weight_decay 0.01 --max_grad_norm 1.0 --optim adamw_torch \
        --max_initial_prompt_tokens 8192 --max_completion_length 16384 --max_completion_tokens 16384 \
        --max_new_tokens_per_turn 2048 --max_total_model_tokens 12288 \
        --generation_micro_batch_size "${GENERATION_MICRO_BATCH_SIZE:-1}" --logprob_micro_batch_size 1 \
        --temperature 1.0 --top_p 0.95 --top_k 50 --mask_truncated_completions False \
        --disable_flash_attn2 False --disable_dropout True --shuffle_dataset False --use_vllm False --bf16 True --fp16 False --tf32 True \
        --gradient_checkpointing True --gradient_checkpointing_kwargs '{"use_reentrant":true}' --use_liger_kernel True --logging_steps 1 \
        --save_strategy steps --save_steps "${SAVE_STEPS:-10}" --save_total_limit 3 --report_to none \
        --seed "${SEED:-42}" --remove_unused_columns False --dataloader_num_workers 0 \
        2>&1 | tee "$RUN/train.log"
fi
if [[ "$STAGE" == all || "$STAGE" == merge ]]; then
    "$PY" "$CODE/merge_adapter.py" --adapter "$RUN/grpo/final_adapter" --output "$RUN/merged_model" 2>&1 | tee "$RUN/merge.log"
fi
if [[ "$STAGE" == all || "$STAGE" == test ]]; then
    for test_set in 600_test_data_notoneclass 600_test_data_oneclass; do
      for condition in ${EVAL_CONDITIONS:-restored}; do
        model="$RUN/merged_model"
        if [[ "$condition" == *_baseline ]]; then model="$MODEL"; fi
        CUDA_VISIBLE_DEVICES="${EVAL_GPU:-${GPUS[0]}}" "$PY" "$CODE/evaluate_grpo_600.py" \
            --model "$model" --data "$RUN/data/${test_set}_episodes.json" --condition "$condition" \
            --rollout-config "$RUN/grpo/experiment_config.json" --batch-size "${EVAL_BATCH_SIZE:-1}" \
            --output-dir "$RUN/test/$test_set/$condition" 2>&1 | tee "$RUN/test_${test_set}_${condition}.log"
      done
    done
fi
printf 'Run: %s\n' "$RUN"
