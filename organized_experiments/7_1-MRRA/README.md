# 7_1-MRRA B：执行器反馈图恢复与 GRPO 实验

本项目对文本属性图（TAG）执行结构恢复、文本恢复及 NC/LC 预测。B 策略按 **Modify → Python 执行 → Rewrite → Python 执行 → Reason/Answer** 完成一条轨迹，采用直接 GRPO 和 LoRA，不进行项目内 SFT。本文说明如何在其他环境复现实验。

训练继承链：`ThinkWithGraphTrainer → GraphR1GRPOTrainer → trl.GRPOTrainer → transformers.Trainer`。训练循环、优化器、梯度累积和 checkpoint 由框架管理。

## 1. 实验约定：固定量与环境配置

**采样量：每条源数据生成 6 条完整候选轨迹，`num_generations=6`。** 一条轨迹包含 Modify、Rewrite、Reason/Answer 三段，不是每段各自作为一个训练样本。6 是跨全部训练进程的同源候选总数，不是每卡各生成 6 条。

**更新量：每次常规梯度更新使用 2 条源数据，即 12 条完整候选轨迹。** 显卡数、微批大小及累积次数根据环境选择，不能通过增大实际更新量来换取速度。`per_device_train_batch_size` 的单位是候选轨迹，不是源数据。

- 采用当前 `steps_per_generation=1`、`num_iterations=1` 时，常规窗口满足：`进程数 × 每卡训练batch × 梯度累积次数 = 12`。
- 生成分块、组内优势计算和优化器更新是不同概念。算法上可以分批生成候选，但同源 6 条必须齐全后统一计算优势，完整 12 条贡献一次更新；不能把两个 3 候选组分别归一化后当成一个 6 候选组。
- 当前实现尚无跨生成微批的候选缓存，`TailSafeRankBatchSampler` 要求每个全局微批包含完整候选组，并要求 `6 % 进程数 == 0`。这属于现有代码限制，不是 GRPO 的理论限制；不能只把单卡 batch 改成 3 就获得跨微批组装功能。
- 原训练集为 5,495 条，原实现不丢弃、不补齐尾部：2,747 次完整更新后，最后 1 条数据／6 条候选单独更新，总计 2,748 次更新。**这是参考实现的尾批例外，不能声称最后一步也有 12 条。** 若要求所有更新严格等长，必须另行确定尾部策略，并记录数据覆盖变化；本文不静默删样或重复采样。

### 不应随显卡环境改变的实验设置

| 项目 | 固定设置 |
|---|---|
| 同源候选／常规更新量 | 6 条／2 条源数据共 12 条候选；尾批例外见上文 |
| 模型 | 同一实验组使用相同规模、相同 revision 的原始 Qwen2.5-Instruct；3B、7B 是不同对照组 |
| 初始化 | 从原始基座新建 LoRA，不使用旧 A/B adapter 或探针权重 |
| LoRA | `r=32, alpha=32, dropout=0.05, bias=none, task_type=CAUSAL_LM` |
| LoRA 注入位置 | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`，冻结全部基座参数 |
| 数据 | 原 5,495 条训练和 600 条测试，保留源顺序、划分及候选标签，不换成其他新数据集 |
| 训练时长 | 1 epoch；不通过增加 epoch 或减少数据量补偿环境差异 |
| 采样 | `temperature=1, top_p=1, top_k` 关闭，`repetition_penalty=1` |
| GRPO | `loss_type=grpo, liger_grpo_loss_type=grpo, beta=0.01, epsilon=0.2, scale_rewards=group` |
| 重用 | `steps_per_generation=1, num_iterations=1`，每个微批读取新数据 |
| 优化器 | `adamw_torch, lr=2e-6, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01` |
| 调度与裁剪 | `warmup_steps=55, cosine, max_grad_norm=1.0` |
| 随机性 | `seed=42, shuffle_dataset=False` |
| 生成预算 | Modify 4096、Rewrite 4096、Answer 512；总生成上限 8704 |
| 输入预算 | 初始 prompt 8192、反馈 prompt 24576、总上下文 32768；超限报错，不静默截断图 |
| 协议与奖励 | `executor_three_stage_tags_v1`、完整 B 策略、原 prompt、停止词、执行与回滚、`B+T+F` |
| 训练目标 | 三段生成 token 联合计算 loss；Python 反馈只参与 attention，不参与 loss |
| 数值与后端 | 参考设置 BF16、TF32、FlashAttention2、Liger；保持 `disable_dropout=True`，不自行改变 dropout 行为 |
| 测试 | 每源一条 greedy 输出，原 600 条全量；同时测试 noisy/clean 和同规模未训练基座 |

`epsilon=0.2` 和组内 reward 标准化继承原 TRL 行为，下面命令显式写出以便核验。LoRA 配置中的 dropout 与 `disable_dropout=True` 应同时保留，不能只看一个字段改变训练行为。

### 可按运行环境配置的项目

模型、数据、Python 和输出路径；使用哪些 GPU、端口；在保持更新量不变且满足现有采样器约束时调整微批／累积；DeepSpeed ZeRO-2/3、CPU offload、通信桶及 checkpointing；日志和保存频率、评测分片等。调整后保留配置与运行记录。

这些调整可以保持同一实验定义，但不保证逐 token 结果完全相同：GPU 架构、归约顺序、随机数消耗和 kernel 可能产生差异。不要为省显存静默缩短上下文、改为量化训练、切换 FP16、修改 reward/loss、使用 vLLM 或改变候选组；这类变化应作为单独实验条件。

## 2. 环境设置

从包含本项目的分支获取代码：

```bash
git clone --branch upload/mrra-b-20261109 https://github.com/dr-eleven/GraphRL.git
cd GraphRL
export REPO_ROOT="$PWD"
export PROJECT_ROOT="$REPO_ROOT/organized_experiments/7_1-MRRA"
```

本项目使用独立环境，不要直接套用仓库根目录针对 3_5_1/3_6 的 Python 3.11 / 新 TRL 环境。参考环境为 Linux x86_64、Python 3.10、PyTorch 2.6.0、CUDA 12.4 构建。精确依赖见 [requirements.txt](requirements.txt)：

```text
torch 2.6.0             transformers 5.17.0       trl 0.25.0
peft 0.19.1            deepspeed 0.16.4           liger_kernel 0.8.0
flash-attn 2.7.4.post1  accelerate 1.14.0          numpy 2.1.3
safetensors 0.8.0      tokenizers 0.23.2
```

一个参考安装过程：

```bash
python3.10 -m venv "$REPO_ROOT/.venv-mrra-b"
source "$REPO_ROOT/.venv-mrra-b/bin/activate"
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
DS_BUILD_OPS=0 python -m pip install --no-build-isolation -r "$PROJECT_ROOT/requirements.txt"

export PYTHON_BIN="$(command -v python)"
export DEEPSPEED_BIN="$(command -v deepspeed)"
export PYTHONPATH="$PROJECT_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
```

FlashAttention 需要匹配 Python、PyTorch、CUDA 和 ABI；没有匹配 wheel 时，上述安装可能需要本地 CUDA 编译工具链、内存和磁盘。CPU offload 若需要额外 DeepSpeed 算子，应单独编译并验证。驱动、CUDA 构建与编译工具链按设备匹配，但不要顺便升级 TRL/Transformers、替换本地 Trainer 或改变 loss 实现。

启动前检查 GPU/BF16 支持，记录实际环境：

```bash
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
assert torch.cuda.is_available()
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        print(i, torch.cuda.get_device_name(i), 'BF16:', torch.cuda.is_bf16_supported())
PY
python -m pip freeze > "$REPO_ROOT/mrra-b-environment.txt"
```

不支持 BF16 的设备无法直接声称按本参考数值设置复现；需要调整精度时应明确记录为变体。

## 3. 数据集与校验

训练为 5,495 条 degree-sorted NC/LC 数据，测试为固定 600 条。clean/noisy 参考、真实标签和攻击元数据用于环境与评分，模型只接收白名单公开观察，不能把原始数据整行直接拼入 prompt。

本仓库已包含相同训练数据和相同测试任务内容的压缩文件：

```bash
cd "$REPO_ROOT"
"$PYTHON_BIN" tools/restore_data.py
export DATA_ROOT="$REPO_ROOT/data/3_5_1_3_6"
```

恢复脚本校验 gzip、解压字节哈希和条数：

| 文件 | 条数 | 解压后 SHA256 |
|---|---:|---|
| `train_episodes.json` | 5495 | `d46d781788e09a7086fc16f86c1deff94d0eb89aaf3a08c19988cf8a8746b778` |
| `test_episodes.json` | 600 | `3c024845c128fc14a02406815a818320f48c3fe6bc5a44f5d9d3299f452f72b6` |

原 MRRA 本地测试文件哈希为 `13a217378fbc5e8ac19d719d826f1b7b69400caf7a68dcff040b17ff47170bff`。它与上述发布测试文件不是同一份字节，但已逐条核对：600 条的顺序、sample_id、目标、候选、标签、noisy/clean 图及保护字段一致；规范化后只有 `reward_version`、`text_action_version` 元数据不同。本项目奖励／动作规则由本地代码定义，不读取这两个版本字段。不要用其他 600 条数据或后续 one-class 数据替换。

NC 只允许改写目标节点；LC 预测有序节点对的关系，结构编辑按无向边处理。LC 查询边在 noisy/clean 图中均被屏蔽并受保护，不能把缺少查询边当作噪声补回。

## 4. 基座模型与 tokenizer

同一对照组固定规模、权重、tokenizer、chat template 和 revision，禁止从 A 的 adapter 接着训练。原模型下载元数据记录的 revision：

| 组别 | Hugging Face 模型 | revision |
|---|---|---|
| 3B | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| 7B | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |

以下以 7B 为例，模型目录使用绝对路径；3B 实验修改这三个变量为对应组别：

```bash
export MODEL_SIZE=7B
export MODEL_REPO=Qwen/Qwen2.5-7B-Instruct
export MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
export MODEL_DIR="$HOME/models/Qwen2.5-7B-Instruct"

"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ['MODEL_REPO'],
                  revision=os.environ['MODEL_REVISION'],
                  local_dir=os.environ['MODEL_DIR'])
PY
```

保留完整 tokenizer、配置、generation config 和权重分片。下载结束后训练／merge／测试都使用本地文件；`TRAIN_MODEL` 和实验配置中的 `model.path` 必须一致，否则训练入口会拒绝。仅改启动环境变量而不改 JSON 还会导致基线评测误用旧模型路径。

## 5. 生成本机配置并准备数据

不要直接使用服务器历史绝对路径。下面创建本机继承配置，保持原实验定义，设置路径及选定的微批／累积。示例是三卡 7B 的 `batch=2, accumulation=2`；可按设备自行配置，但常规更新量保持 12。

```bash
export GPU_IDS=0,1,2
export TRAIN_BATCH_SIZE=2
export GRAD_ACCUM=2
export MASTER_PORT=29610
export WORK_ROOT="$PROJECT_ROOT/outputs/local_setup"
export RUN_DIR="$PROJECT_ROOT/outputs/B_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)"
export LOCAL_CONFIG="$WORK_ROOT/executor_${MODEL_SIZE}.json"
mkdir -p "$WORK_ROOT"

"$PYTHON_BIN" - <<'PY'
import os,json
from pathlib import Path
project=Path(os.environ['PROJECT_ROOT'])
gpus=os.environ['GPU_IDS'].split(',');w=len(gpus)
b=int(os.environ['TRAIN_BATCH_SIZE']);a=int(os.environ['GRAD_ACCUM'])
assert len(set(gpus))==w and b>0 and a>0
assert w*b*a==12, '每次常规更新必须是12条候选'
assert 6%w==0 and (w*b)%6==0, '当前采样器不支持这个跨rank/微批布局'
base=project/'configs'/('mrra_executor_7b.json' if os.environ['MODEL_SIZE']=='7B' else 'mrra_executor.json')
cfg={'extends':str(base), 'model':{'path':str(Path(os.environ['MODEL_DIR']).resolve())},
     'data':{'train':str(Path(os.environ['DATA_ROOT'])/'train_episodes.json'),
             'test':str(Path(os.environ['DATA_ROOT'])/'test_episodes.json'), 'partition':'all'},
     'training':{'num_generations':6,'per_device_train_batch_size':b,
                 'per_device_eval_batch_size':b,'gradient_accumulation_steps':a}}
Path(os.environ['LOCAL_CONFIG']).write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
print('world_size=',w,'candidate_batch=',b,'accumulation=',a,'effective_candidates=',w*b*a)
PY

"$PYTHON_BIN" "$PROJECT_ROOT/code/prepare_data.py" \
  --config "$LOCAL_CONFIG" --output "$WORK_ROOT/prepared"
```

`prepared/manifest.json` 应记录 5,495/600 条数据、原顺序和无训练／测试源重叠。实际训练仍读取本机配置指定的源文件，由同一规范化函数现场构造 prompt，不能把已经生成的旧 oracle prompt 当作新输入。

可检查 tokenizer 对初始 prompt 的长度，超限必须排查，不能截断图或删数据：

```bash
"$PYTHON_BIN" "$PROJECT_ROOT/code/check_data.py" \
  --config "$LOCAL_CONFIG" --output "$WORK_ROOT/prompt_lengths.json"
```

## 6. Prompt、动作与奖励

实际加载文件：

- [NC system](code/prompts/executor_nc_system.txt)、[LC system](code/prompts/executor_lc_system.txt)：任务、恢复判断、约束、三轮协议。
- [executor_protocol.py](code/executor_protocol.py)：当前轮 user 提示及 Python 反馈拼接。
- [完整 NC/LC 示例](reports/MRRA_B_prompt_revision_20261104.md)：其中 NO_OP 是人工协议演示，不是训练结果。

三轮分别输出：

```text
<reason>根据文本和结构判断是否需要结构恢复</reason>
<modify_action>[结构动作JSON，或空数组]</modify_action>

<reason>根据最新实际图判断是否需要文本恢复</reason>
<rewrite_action>[文本动作JSON，或空数组]</rewrite_action>

<reason>根据最终实际图推理</reason>
<answer>label_space中的一个候选标签</answer>
```

每轮只输出自己的两个块，不一次写完三轮。停止词分别是 `</modify_action>`、`</rewrite_action>`、`</answer>`，保留停止标签。Python 在动作后追加状态及 `<restored_graph>实际图</restored_graph>`，附加“使用最新修改图进行后续推理。”；完整原对话与模型输出均保留。

结构操作为 ADD_EDGE/DEL_EDGE/DELETE_NODE，文本操作为 REWRITE，允许 `[]`。非法阶段整组回滚，之前成功阶段保留；不重试，继续后续阶段。不要改为“跳过非法动作但执行剩余动作”，不要移植 3_4 的格式失败立即结束并全奖励归零规则。

奖励 `R=B+T+F`，各项 0/1：

- B：所有动作阶段合法，且实际干净图不编辑／实际受损图有非空编辑。
- T：独立提取的最终答案正确；前面阶段失败仍可能得到 T。
- F：三轮格式全符合要求只得 1 分，不是每轮各 1 分；检查标签、顺序、非空 reason、严格动作数组及标签外输出，不检查动作语义或答案正确性。

B 的非空编辑不是恢复正确性证明；clean 参考与标签只用于评分。三段模型 token 共同参与 GRPO loss，两个 Python 反馈区间的 loss mask 为 0、attention 为 1。不能对三段分别优化、使用两个不同的组内优势，或把 Python 返回图当成监督生成 token。

## 7. 训练与显存验证

以下使用显式命令，避免旧包装脚本默认 `batch=8, accumulation=4` 改变本次要求的更新量。JSON 的 `training` 字段也不会自动替代 CLI 参数，实际值以 `grpo/experiment_config.json` 中 `training_arguments` 为准。

```bash
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 不额外设置与 --include 冲突的 CUDA_VISIBLE_DEVICES。
unset CUDA_VISIBLE_DEVICES

TRAIN_ENTRY="$PROJECT_ROOT/code/train_grpo.py"
DS_CONFIG="$PROJECT_ROOT/configs/zero2.json"
CHECKPOINT_ARGS=()
if [[ "$MODEL_SIZE" == 7B ]]; then
  TRAIN_ENTRY="$PROJECT_ROOT/code/train_grpo_7b.py"
  DS_CONFIG="$PROJECT_ROOT/configs/zero3_7b.json"
  CHECKPOINT_ARGS=(--gradient_checkpointing_kwargs '{"use_reentrant":true}')
fi

TRAIN_ARGS=(
  --deepspeed "$DS_CONFIG"
  --mrra_config "$LOCAL_CONFIG" --model_id "$MODEL_DIR"
  --lora_enable True --lora_rank 32 --lora_alpha 32 --lora_dropout 0.05
  --num_generations 6 --num_train_epochs 1
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --steps_per_generation 1 --num_iterations 1
  --max_completion_length 8704 --max_prompt_length 8192
  --temperature 1.0 --top_p 1.0 --top_k 0 --repetition_penalty 1.0
  --beta 0.01 --epsilon 0.2 --scale_rewards group
  --loss_type grpo --liger_grpo_loss_type grpo --use_liger_loss True
  --optim adamw_torch --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8
  --warmup_steps 55 --lr_scheduler_type cosine --weight_decay 0.01 --max_grad_norm 1.0
  --shuffle_dataset False --disable_dropout True --logprob_micro_batch_size 1
  --bf16 True --fp16 False --tf32 True --gradient_checkpointing True
  --disable_flash_attn2 False --use_vllm False
  --logging_steps 1 --save_strategy steps --save_steps 100 --save_total_limit 3
  --seed 42 --report_to none --remove_unused_columns False --dataloader_num_workers 0
  "${CHECKPOINT_ARGS[@]}"
)

# 独立探针：不产生可复用的训练adapter，不占用正式输出目录。
PROBE_DIR="$PROJECT_ROOT/outputs/probe_$(date +%Y%m%d_%H%M%S)"
"$DEEPSPEED_BIN" --include="localhost:$GPU_IDS" --master_port="$MASTER_PORT" \
  "$TRAIN_ENTRY" "${TRAIN_ARGS[@]}" --output_dir "$PROBE_DIR/grpo" \
  --validation_only True --learning_rate 0 --max_steps 1 \
  --limit_data_sources 4 --memory_stress_tokens 8704

# 正式训练：重新初始化原始基座和LoRA。
"$DEEPSPEED_BIN" --include="localhost:$GPU_IDS" --master_port="$MASTER_PORT" \
  "$TRAIN_ENTRY" "${TRAIN_ARGS[@]}" --output_dir "$RUN_DIR/grpo" --learning_rate 2e-6
```

本机选择 ZeRO/offload 时，调整 `DS_CONFIG` 及其副本；保留 batch 配置的 `auto`，使其与 CLI 一致。7B/ZeRO-3 使用 `train_grpo_7b.py` 并保留显式 reentrant checkpointing，它会在生成后恢复设置。不要绕过该入口后仍声称与同组实现一致。

探针覆盖长生成的前向、参考概率、loss、反向与优化器路径，但只选前几条数据，不覆盖训练后期最大图；须结合 prompt 长度检查与实际峰值显存监控，不能保证所有 GPU 均能运行。OOM 时先检查更新量与路径，再调整合规微批／累积、ZeRO/offload；不要减少 G、缩短文本预算或复用探针权重。

所有输出目录必须新建；代码拒绝覆盖已有输出。重复命令前重新设置 RUN_DIR/PROBE_DIR。保存目录需容纳 checkpoint、adapter、后续 merged 模型以及很大的逐候选 JSONL 日志。

## 8. Merge

训练完成应有 `grpo/status.txt=complete`、`final_adapter/`、训练状态与 epoch 完整 checkpoint。先核对实际训练参数和全量覆盖，再 merge：

```bash
"$PYTHON_BIN" "$PROJECT_ROOT/code/merge_adapter.py" \
  --adapter "$RUN_DIR/grpo/final_adapter" --output "$RUN_DIR/merged"
```

merge 默认在 CPU 上以 BF16 加载基座，机器需要足够的 CPU 内存及磁盘；仅 GPU 显存充足不够。adapter_config 中的 `base_model_name_or_path` 指向训练时本地基座，跨机器 merge 必须保证它可用且权重与训练时一致，不得合入另一个 revision。merged 目录已存在时会拒绝覆盖。

## 9. 测试与原始基座对照

固定 600 条源数据，分别做 noisy/clean。clean 是同一测试源的 clean_graph，不是换另一批数据。B 策略和同规模原始基座直接预测各测两组，共四组，每组每源一条 greedy 输出，不做 6 候选投票或挑选。训练采样正确率不能冒充测试准确率。

```bash
export EVAL_GPU=0
for condition in noisy clean; do
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_BIN" "$PROJECT_ROOT/code/evaluate_grpo_600.py" \
    --config "$LOCAL_CONFIG" --strategy executor --model "$RUN_DIR/merged" \
    --condition "$condition" --output "$RUN_DIR/eval_$condition"

  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_BIN" "$PROJECT_ROOT/code/evaluate_grpo_600.py" \
    --config "$LOCAL_CONFIG" --strategy direct --model "$MODEL_DIR" \
    --condition "$condition" --output "$RUN_DIR/baseline_$condition"
done

for condition in noisy clean; do
  "$PYTHON_BIN" "$PROJECT_ROOT/code/compare_results.py" \
    --run "baseline=$RUN_DIR/baseline_$condition/predictions.jsonl" \
    --run "MRRA_B=$RUN_DIR/eval_$condition/predictions.jsonl" \
    --output "$RUN_DIR/comparison_$condition.json"
done
```

默认语义相似度诊断关闭，本实验运行不要求下载 bge-m3。测试单进程单卡，可按固定源ID做无重叠分片来加速，但必须合并为相同的600个唯一ID；不得更换解码、漏掉失败轨迹或将失败记为“未测试”。每组输出目录必须不存在，禁止使用 `--limit` 产生部分结果后当全量结果。

## 10. 结果交付与其他容易遗漏的事项

至少保存并交付：

1. **版本与实际参数**：Git commit、pip freeze、GPU/驱动/CUDA、模型 revision 与权重/tokenizer 哈希、数据哈希、prompt 哈希、DeepSpeed 文件、`experiment_config.json`、代码快照。
2. **覆盖与更新量**：5,495个训练源，每源6候选共32,970条；顺序和组边界正确；常规更新12条，尾部例外明确。不能把每卡数量误当全局数量，也不能用小 batch 各自做3候选组标准化。
3. **训练过程**：step、loss、KL、reward、B/T/F、格式通过率、动作合法率、clean NO_OP、受损编辑率、截断和峰值显存。reward提高不等于恢复正确。
4. **测试**：四组各600条唯一源ID、准确率、正负翻转、分任务/数据集指标；基座直接预测和MRRA B分别报告，不挑最好的一次随机结果。
5. **恢复与断点**：正式恢复使用含optimizer/scheduler/RNG的完整checkpoint，只有final_adapter不能等价续训；当前代码不允许续训时更换batch/累积/G，也不允许A或旧两段B的checkpoint接入新B。
6. **随机性与精度**：固定seed仍不能保证异构硬件逐token一致。显存策略是环境配置，量化、精度／attention实现、框架版本或算法路径变化需要披露，不能混同为相同条件结果。

本目录 `scripts/check.sh` 可运行33项CPU回归；其中真实Qwen tokenizer检查保留了原服务器7B目录硬编码。在其他环境应将该测试目录指向本机同revision tokenizer或建立对应路径，并记录这项仅测试路径的适配；不能将“路径不存在”的失败说成模型/协议实现失败。项目训练本身通过本机配置使用MODEL_DIR，不依赖该测试硬编码路径。

项目只发布源码、配置、脚本和说明；共享模块中保留A/直接预测分支作为依赖，不改变B实验定义。原始来源及发布时逐文件校验见 [source_provenance.json](source_provenance.json) 与 [publication_manifest.json](publication_manifest.json)。
