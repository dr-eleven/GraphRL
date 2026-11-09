#!/usr/bin/env bash
# 延续 degree-sorted 的 DeepSpeed/TRL 标准参数；MRRA 行为由本地配置决定。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
strategy="${1:?Usage: run_mrra.sh llm|executor train|all|probe|prepare|evaluate|merge}"
stage="${2:-all}"
shift "$(( $# >= 2 ? 2 : 1 ))"
case "$strategy" in llm|executor) ;; *) exit 2 ;; esac
PY="${PYTHON_BIN:-/opt/data/private/.conda/envs/GraphLLM/bin/python}"
DS="${DEEPSPEED_BIN:-/opt/data/private/.conda/envs/GraphLLM/bin/deepspeed}"
CONFIG="${MRRA_CONFIG:-$ROOT/configs/mrra_${strategy}.json}"
RUN="${RUN_DIR:-$ROOT/outputs/${strategy}_$(date +%Y%m%d_%H%M%S_%N)}"
[[ "$RUN" == /* ]] || RUN="$ROOT/$RUN"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
MODEL="${TRAIN_MODEL:-/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-3B-Instruct}"
BATCH="${TRAIN_BATCH_SIZE:-8}"
ACCUM="${GRAD_ACCUM:-4}"
# 配置采用 vLLM 的 -1 表示关闭 top-k；HF/TRL 用 0 表示同一含义。
SAMPLING_PARAMS="$("$PY" - "$CONFIG" <<'PY'
import sys
from config import load_config
generation = load_config(sys.argv[1])['generation']
top_k = generation.get('top_k', 50)
print(generation['temperature'], generation['top_p'], 0 if top_k == -1 else top_k)
PY
)"
read -r SAMPLE_TEMPERATURE SAMPLE_TOP_P SAMPLE_TOP_K <<< "$SAMPLING_PARAMS"
train_run() {
    local mode="$1"
    shift
    local extra=()
    if [[ "$mode" == probe ]]; then
        extra=(--validation_only True --learning_rate 0 --max_steps 1 --limit_data_sources 4 --memory_stress_tokens "${STRESS_TOKENS:-8704}")
    else
        extra=(--learning_rate 2e-6)
    fi
    if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then extra+=(--continuation_checkpoint "$RESUME_CHECKPOINT"); fi
    "$DS" --include="localhost:${GPU_IDS:-0,1,2}" --master_port="${MASTER_PORT:-29610}" "$ROOT/code/train_grpo.py" \
        --deepspeed "$ROOT/configs/zero2.json" \
        --mrra_config "$CONFIG" --model_id "$MODEL" --output_dir "$RUN/grpo" \
        --lora_enable True --lora_rank 32 --lora_alpha 32 --lora_dropout 0.05 \
        --num_generations 6 --max_completion_length 8704 --max_prompt_length 8192 \
        --shuffle_dataset False --disable_dropout True --beta 0.01 \
        --temperature "$SAMPLE_TEMPERATURE" --top_p "$SAMPLE_TOP_P" --top_k "$SAMPLE_TOP_K" \
        --repetition_penalty 1.0 --use_vllm False --num_train_epochs 1 \
        --per_device_train_batch_size "$BATCH" --per_device_eval_batch_size "$BATCH" \
        --gradient_accumulation_steps "$ACCUM" --steps_per_generation 1 --num_iterations 1 \
        --logprob_micro_batch_size 1 --loss_type grpo --liger_grpo_loss_type grpo --use_liger_loss True \
        --warmup_steps 55 --lr_scheduler_type cosine --weight_decay 0.01 --max_grad_norm 1.0 \
        --bf16 True --fp16 False --tf32 True --gradient_checkpointing True --disable_flash_attn2 False \
        --logging_steps 1 --save_strategy steps --save_steps 100 --save_total_limit 3 \
        --seed 42 --report_to none --remove_unused_columns False --dataloader_num_workers 0 \
        "${extra[@]}" "$@"
}
case "$stage" in
    train|probe) train_run "$stage" "$@" ;;
    all)
        train_run train "$@"
        printf 'merging\n' > "$RUN/pipeline_status.txt"
        "$PY" "$ROOT/code/merge_adapter.py" --adapter "$RUN/grpo/final_adapter" --output "$RUN/merged"
        for condition in noisy clean; do
            printf 'evaluating_policy_%s\n' "$condition" > "$RUN/pipeline_status.txt"
            "$PY" "$ROOT/code/evaluate_grpo_600.py" --config "$CONFIG" --model "$RUN/merged" \
                --condition "$condition" --output "$RUN/eval_${condition}"
            printf 'evaluating_baseline_%s\n' "$condition" > "$RUN/pipeline_status.txt"
            "$PY" "$ROOT/code/evaluate_grpo_600.py" --config "$CONFIG" --strategy direct \
                --condition "$condition" --output "$RUN/baseline_${condition}"
        done
        "$PY" "$ROOT/code/compare_results.py" --run "baseline=$RUN/baseline_noisy/predictions.jsonl" \
            --run "restored=$RUN/eval_noisy/predictions.jsonl" --output "$RUN/comparison.json"
        printf 'complete\n' > "$RUN/pipeline_status.txt"
        ;;
    prepare) "$PY" "$ROOT/code/prepare_data.py" --config "$CONFIG" --output "$ROOT/data/prepared" "$@" ;;
    evaluate) "$PY" "$ROOT/code/evaluate_grpo_600.py" --config "$CONFIG" --output "$RUN" "$@" ;;
    merge) "$PY" "$ROOT/code/merge_adapter.py" --output "$RUN" "$@" ;;
    *) exit 2 ;;
esac
