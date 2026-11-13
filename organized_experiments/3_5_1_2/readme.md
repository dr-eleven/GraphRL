# 3_5_1_2：跨服务器执行说明

本文供接手服务器上的Codex直接执行，组织方式参考`organized_experiments/7_1-MRRA/github_readme.md`，参数以本目录实际代码为准。仓库：https://github.com/dr-eleven/GraphRL 。从仓库根目录执行以下命令；不依赖原服务器的`/opt/data/private`路径。

## 任务与实验边界

在原始 **Qwen/Qwen2.5-7B-Instruct** 上分别使用两套配对2000条数据，从新建LoRA开始进行一轮GRPO训练，然后合并adapter并评测。支持NC（目标节点分类）、LC（查询边关系分类）、LP（原图中指定链接是否存在）三种任务。训练是直接预测：一次生成`{"analysis":"...","prediction":"完整候选标签"}`，没有图修复工具、外部teacher、额外SFT或先修图后预测步骤。两组训练独立初始化，不能将第一组adapter作为第二组起点。

本包只提供执行代码、数据与验证记录；发布时没有启动这两组正式7B训练，也没有打包模型权重。接手Codex收到执行授权后应完成环境恢复、数据验证、训练、合并、测试和报告；先检查GPU和磁盘，不要因为能运行prepare就声称训练完成。不要读取或依赖原服务器的其他实验运行目录。

## 数据与两组实验

| 实验 | 训练文件（仓库根目录相对路径） | 测试文件 |
| --- | --- | --- |
| oneclass | `2000_oneclass_sorted/train_grpo_episodes_redesign.json` | `data/test_data_600_oneclass/test_grpo_episodes.json` |
| notoneclass | `2000_notoneclass_sorted/train_grpo_episodes_redesign.json` | `data/test_data_600_notoneclass/test_grpo_episodes.json` |

oneclass攻击候选来自按目标文本语义选择的同一个最远类别；notoneclass为此前不限制候选来自单一类别的版本。两套训练数据使用相同2000个源ID、任务、标签和逐ID的clean/budget分配。攻击图分别来自各自版本。每条源数据仅使用一个选定版本，不能把clean.parquet或其余budget追加成更多训练条目。

| 训练任务 | 条数 | clean | budget1 | budget2 | budget3 | budget4 | budget5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NC | 1000 | 200 | 160 | 160 | 160 | 160 | 160 |
| LC | 200 | 40 | 32 | 32 | 32 | 32 | 32 |
| LP | 800 | 160 | 128 | 128 | 128 | 128 | 128 |
| 合计 | 2000 | 400 | 320 | 320 | 320 | 320 | 320 |

共有19种“数据集＋任务”组合：NC的arxiv、children、citeseer、computer、history、instagram、photo、pubmed、reddit、sports各100条；LP的arxiv、children、computer、history、photo、pubmed、reddit、sports各100条；LC的wn18rr为200条。LP正例407、负例393。各目录README、manifest和pair_alignment.csv提供完整构成及配对。

数据已经按degree升序排列，训练不打乱。NC取目标节点的无向去重邻居数；LC/LP先屏蔽查询边，再取两端degree均值；忽略自环及关系类型，同degree按source_index升序。两套攻击图degree可能不同，因此最终行号不保证相同，按源ID/budget配对，不要按行号合并，也不要为了行号一致重新排序。

每套测试恰好600条：NC450（cora/products/wikics各150），LC150（fb15k237）；budget1–5各120条，每档NC90+LC30，混合文件中的每条源样本只出现一次。两套测试的源ID、顺序、budget分配及clean图对应，攻击图不同。clean评测通过同一episode的clean_graph获得。`badgraph/budget1–5`各600条是完整攻击轨迹对应版本，不是正式测试的3000条输入。

**测试集没有LP。** 训练支持LP、测试只覆盖NC/LC；不能在报告中声称验证了LP准确率。`preflight.json`会显示`missing_test_tasks=["LP"]`。默认`REQUIRE_TEST_TASK_COVERAGE=0`，若设1则预检主动拒绝这套测试集。需要LP测试时另行提供独立数据，并记录新实验定义，不要从训练集抽样冒充LP测试。

notoneclass目录中的混合文件是从已核验的`GraphRL_test_data`逐字节复制而来，SHA256见该目录README_mixed_eval.md。与3_5正式运行保存的600条逐条核对，ID/顺序、标签、budget、clean_graph、noisy_graph等完全一致，仅旧prompt和reward/text_action/prompt版本标记不同。这里补齐的是可独立迁移的文件组织，没有换测试样本。prepare会统一重建当前prompt，不会复用旧prompt。

## 恢复仓库数据

```bash
git clone https://github.com/dr-eleven/GraphRL.git
cd GraphRL
# 系统需有zstd（例如Ubuntu：sudo apt-get install zstd）
python3 tools/restore_3_5_1_2_data.py
python3 tools/restore_3_5_1_2_data.py --verify-only
```

四个数据目录通过各自`archive/`中的分卷随Git仓库发布，无需原服务器或外部数据下载。恢复脚本先验证分卷、完整压缩包及每个文件的SHA256，再恢复为原目录结构；已有文件内容不一致时报错，不能覆盖用户文件。压缩只是Git传输形式，不改变数据内容。恢复后保留archive和manifest不影响训练，训练只读取表中指定episode文件。

## 基座模型与环境

模型固定为`Qwen/Qwen2.5-7B-Instruct`，revision：`a09a35458c702b33eeacc393d103063234e8bc28`。使用原始Instruct权重和对应tokenizer；不能替换为Qwen2.5-3B、其他SFT模型或已有LoRA。下载可在联网机器完成再拷贝完整目录，训练脚本以离线方式加载本地模型。

推荐Linux x86_64、Python3.11、NVIDIA GPU与支持CUDA12.4的驱动。核心环境：torch2.6.0/cu124、transformers5.22.1、trl1.17.1、peft0.21.2、accelerate1.15.0、datasets5.1.0、deepspeed0.19.9、liger-kernel0.8.0、flash-attn2.7.4.post1。完整版本见`requirement.txt`，不要照搬MRRA的Python/TRL版本。

```bash
bash organized_experiments/3_5_1_2/scripts/setup_native_trl.sh
export PYTHON_BIN="$PWD/.venv-3_5_1_2/bin/python"
export TRAIN_MODEL="$PWD/models/Qwen2.5-7B-Instruct"
"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct',
    revision='a09a35458c702b33eeacc393d103063234e8bc28',
    local_dir=os.environ['TRAIN_MODEL'])
PY
nvidia-smi
"$PYTHON_BIN" -m unittest discover -s organized_experiments/3_5_1_2/tests -v
```

安装脚本支持`BOOTSTRAP_PYTHON`（默认python3.11）、`EXPERIMENT_ENV`（默认仓库/.venv-3_5_1_2）。FlashAttention锁定的公开wheel适用于CPython3.11、torch2.6、CUDA12、CXX11_ABI=False；其他平台不要静默跳过或降级内核。安装失败应记录环境并解决对应依赖，不能改实验参数掩盖错误。

## 固定训练与生成参数

| 项目 | 值 |
| --- | --- |
| 基座及训练方式 | 原始Qwen2.5-7B-Instruct；新建LoRA；GRPO |
| LoRA | r=32，alpha=32，dropout=0.05，bias=none |
| LoRA目标层 | q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj |
| 训练精度 | BF16；FP16=False；TF32=True；无量化 |
| 分布式 | 默认3GPU、DeepSpeed ZeRO-3；配置不含CPU offload |
| 每卡candidate batch / 梯度累积 | 2 / 2 |
| 全局每源样本候选数G | 6；num_iterations=1；steps_per_generation=1 |
| 每次优化更新有效候选数 | 3×2×2=12，即2个源样本的6候选组 |
| 数据及更新数 | 2000源样本，1epoch，12000个候选生成；默认1000次优化更新 |
| 优化器 | adamw_torch；betas=(0.9,0.999)，eps=1e-8；lr=2e-6；weight_decay=0.01；max_grad_norm=1.0 |
| 学习率 | warmup_steps=55；cosine |
| GRPO | loss_type=grpo；beta=0.01；epsilon=0.2；scale_rewards=group |
| prompt / completion上限 | 8192 / 512 tokens；prediction_max_new_tokens=512 |
| 训练采样 | temperature=1.0；top_p=0.95；top_k=50；repetition_penalty=1.0 |
| 省显存配置 | FlashAttention2、Liger、梯度检查点use_reentrant=True、logprob_micro_batch_size=1 |
| 其他 | disable_dropout=True；use_vllm=False；max_tool_calling_iterations=0 |
| 顺序及种子 | seed=42；shuffle_dataset=False；全2000条，TRAIN_FRACTION=1 |
| 日志及保存 | logging_steps=1；save_steps=100；save_total_limit=3；report_to=none |
| 测试 | 贪心生成；每条1次；同样512生成上限；不做投票或按测试表现挑checkpoint |

G=6是所有GPU合计每个源样本的6个候选，不是每GPU各6个。默认全局microbatch有3×2=6个候选，对应一个完整源样本组；累积2次更新。2000为偶数，不存在尾组丢弃。若迁移服务器GPU数不同，应同时满足`6 % GPU数 == 0`、`(GPU数×每卡batch) % 6 == 0`，并保持`GPU数×每卡batch×累积步数=12`。例如1GPU/batch6/累积2，2GPU/batch3/累积2，6GPU/batch1/累积2；须按可用显存预检。不能仅减小batch造成group不完整或静默丢数据。显存不足时先确认其他进程、可用设备、评测batch；如需改变实验核心参数，应单独说明，不能混入同一结果。

## 当前prompt、查询边与奖励

权威prompt由`code/episode.py::build_prediction_prompt`生成。NC/LC/LP共用结构与输出协议，分别明确预测语义。约束要求只回答指定目标；LC/LP屏蔽查询边，LP中未观察到查询边不等于负例；综合文本与结构证据，不依赖单个关键词、路径、共同类别、共同邻居或重复文本；相似不证明存在关系/链接，不相似不否定；缺失/冲突表示不确定，不能编造事实或候选标签。没有额外的`Query direction`字段。完整候选标签来自数据，包括LP正负两个候选；不能擅自改成简写yes/no。

`task_utils.prepare_episode`对副本进行处理，不修改原文件；clean与noisy视图都屏蔽LC/LP查询边，默认按无向查询屏蔽两方向；显式有向元数据的处理保留在代码中，不新增prompt方向要求。prompt只公开任务、目标、候选标签和当前视图图信息，不泄露ground_truth、clean_graph、攻击位置或budget。数据中已有占位文本作为缺失证据处理，不填造节点内容。

输出必须是一个JSON对象，顺序为analysis、prediction，两个值均为非空字符串，没有多余字段或外部文字；重复键、不完整JSON、截断且未正常结束等无效。奖励`R=S+T`：输出结构合法S=1，否则0；合法且候选正确T=1，合法但选错候选T=0.1，合法但不在候选中T=0；无效输出总奖励0。因此正确2、候选内错误1.1、候选外1、无效0。候选比较使用现有规范化规则，不把句子标签的单个yes/no视为完整匹配。以`reward.py`实现为准。

## 执行两组训练与测试

在同一个shell中保持以下变量。默认用物理GPU0,1,2；如设备不同只修改显式设备与批量配置，并满足上述等式。不要覆盖已有RUN目录。

```bash
export REPO="$PWD"
export EXP="$REPO/organized_experiments/3_5_1_2"
export PYTHON_BIN="$REPO/.venv-3_5_1_2/bin/python"
export TRAIN_MODEL="$REPO/models/Qwen2.5-7B-Instruct"
export CUDA_VISIBLE_DEVICES=0,1,2
export TRAIN_BATCH_SIZE=2 GRAD_ACCUM=2 NUM_GENERATIONS=6
export TRAIN_FRACTION=1 FINAL_EPOCH=1 PREDICTION_TOKENS=512
export EXPECTED_TEST_COUNT=600 REQUIRE_TEST_TASK_COVERAGE=0

# A：oneclass。prepare不会加载完整模型权重或启动训练。
export TRAIN_EPISODES="$REPO/2000_oneclass_sorted/train_grpo_episodes_redesign.json"
export TEST_EPISODES="$REPO/data/test_data_600_oneclass/test_grpo_episodes.json"
export RUN_DIR="$EXP/runs/oneclass_2000"
bash "$EXP/scripts/run_portable.sh" prepare
cat "$RUN_DIR/data/preflight.json"
bash "$EXP/scripts/run_portable.sh" train
bash "$EXP/scripts/run_portable.sh" merge
bash "$EXP/scripts/run_portable.sh" test

# B：notoneclass。从同一原始基座重新创建adapter，不沿用A的adapter。
export TRAIN_EPISODES="$REPO/2000_notoneclass_sorted/train_grpo_episodes_redesign.json"
export TEST_EPISODES="$REPO/data/test_data_600_notoneclass/test_grpo_episodes.json"
export RUN_DIR="$EXP/runs/notoneclass_2000"
bash "$EXP/scripts/run_portable.sh" prepare
cat "$RUN_DIR/data/preflight.json"
bash "$EXP/scripts/run_portable.sh" train
bash "$EXP/scripts/run_portable.sh" merge
bash "$EXP/scripts/run_portable.sh" test
```

先确认每组preflight为train_count=2000、test_count=600、train NC1000/LC200/LP800、test NC450/LC150、source_order_preserved=true、missing_test_tasks=[LP]，最大prompt小于8192。prepare不截断超长prompt，发现异常会报错。prepare完成后要在同一RUN_DIR运行train/merge/test，不能再次all/prepare覆盖；`all`只用于新的RUN目录从头完整执行。不要并行启动两组占满同一批GPU。

训练阶段输出adapter到`RUN_DIR/grpo/final_adapter`；合并阶段从adapter配置定位原始基座，在CPU加载BF16权重合并到`RUN_DIR/merged_model`，需预留主机内存和磁盘。此CPU合并与训练时CPU offload是两回事。评测默认单GPU，`EVAL_GPU`指定物理GPU，`EVAL_BATCH_SIZE`默认1。修改EVAL_BATCH_SIZE不改变样本和生成上限。

## 测试矩阵与比较

每组test阶段均评测下表四种条件，使用相同当前prompt和生成预算，不沿用3_5旧工具prompt。

| condition | 模型 | 输入图 |
| --- | --- | --- |
| noisy_policy | 本组训练合并模型 | 本组攻击混合测试图 |
| clean_policy | 本组训练合并模型 | 同600条clean图 |
| noisy_baseline | 原始基座 | 本组攻击混合测试图 |
| clean_baseline | 原始基座 | 同600条clean图 |

匹配版本的A/B结果同时改变了训练攻击分布与测试攻击分布，报告必须写明，不能把其差值全部归因于训练方法。若要比较训练方法在固定测试攻击下的差异，补充2×2交叉评测：分别将两个训练模型都测在oneclass与notoneclass混合测试上。下面两项补足已执行矩阵中的另外两格（在上述变量已设置的shell运行）：

```bash
for TRAIN_VARIANT in oneclass notoneclass; do
  if [ "$TRAIN_VARIANT" = oneclass ]; then TEST_VARIANT=notoneclass; else TEST_VARIANT=oneclass; fi
  RUN="$EXP/runs/${TRAIN_VARIANT}_2000"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$EXP/code/evaluate_grpo_600.py" \
    --model "$RUN/merged_model" \
    --data "$REPO/data/test_data_600_${TEST_VARIANT}/test_grpo_episodes.json" \
    --condition noisy_policy --rollout-config "$RUN/grpo/experiment_config.json" \
    --expected-count 600 --batch-size 1 --output-dir "$RUN/test/cross_${TEST_VARIANT}"
done
```

clean图相同，可比较两模型clean_policy；两个clean_baseline在相同模型/prompt/预算下应相符。所有正式评测用完整600条，不能使用`--limit`抽样结果替代。现有evaluate脚本会重建prompt并屏蔽查询边。

## 输出、核验与故障处理

每个RUN应保留：`code_snapshot/`、`data/preflight.json`及处理后的train/test episodes、`prepare.log`、`train.log`、`grpo/experiment_config.json`、checkpoint/trainer_state、final_adapter、merged_model、每种条件的predictions.jsonl与summary.json、status.txt。记录Git commit、数据manifest哈希、模型revision、GPU型号/数量、实际依赖freeze及启动命令。

训练结束检查实际global_step（默认1000）、epoch=1、final_adapter和合并权重存在、候选采样按6组成完整组、完整测试600条。读取评测summary报告总体/NC/LC/数据集分组准确率、无效输出与截断数量；LP应标为无测试覆盖，不能填0伪装测得结果。预测无效保留为错误，不能删掉再计算准确率。不要通过加token、改prompt、反复挑checkpoint来选择性优化测试；需要改动另建运行并注明。跨版本比较必须使用同一测试版本与预算。

训练中断先检查日志与checkpoint，不要自动从头覆盖目录或宣称已完成；当前launcher没有自动resume开关，如需恢复应使用train_grpo.py已有的--continuation_checkpoint显式指定完整checkpoint（内部传入Trainer的resume_from_checkpoint），复用原启动参数并验证optimizer/scheduler/global_step恢复，保留改动和命令，不能把加载adapter但优化器重置当成无缝恢复。GPU显存不足/环境不兼容/模型下载失败等应给出实际原因与已完成阶段。

验证目录保留单元测试和此次迁移preflight摘要，不包含原服务器大型运行目录、模型、缓存、密钥。`design_notes.md`为实现背景，执行与最终prompt以本readme和当前代码为准。`source_manifest.json`保留来源及当前代码哈希。
