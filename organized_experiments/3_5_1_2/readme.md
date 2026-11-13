# 3_5_1_2：跨服务器执行说明

本文供接手服务器上的Codex直接执行，组织方式参考`organized_experiments/7_1-MRRA/github_readme.md`，参数以本目录实际代码为准。仓库：https://github.com/dr-eleven/GraphRL 。从仓库根目录执行以下命令；不依赖原服务器的`/opt/data/private`路径。

## 任务与实验边界

在原始 **Qwen/Qwen2.5-7B-Instruct** 上分别使用两套配对2000条数据，从新建LoRA开始进行一轮GRPO训练，然后合并adapter并评测。支持NC（目标节点分类）、LC（查询边关系分类）、LP（原图中指定链接是否存在）三种任务。训练是直接预测：一次生成`{"analysis":"...","prediction":"完整候选标签"}`，没有图修复工具、外部teacher、额外SFT或先修图后预测步骤。两组训练独立初始化，不能将第一组adapter作为第二组起点。

本包只提供执行代码、数据与验证记录；发布时没有启动这两组正式7B训练，也没有打包模型权重。接手Codex收到执行授权后应完成环境恢复、数据验证、训练、合并、测试和报告；先检查GPU和磁盘，不要因为能运行prepare就声称训练完成。不要读取或依赖原服务器的其他实验运行目录。

## 数据与两组实验

| 实验 | 训练文件（仓库根目录相对路径） | 测试文件 |
| --- | --- | --- |
| oneclass | `2000_oneclass_sorted/train_grpo_episodes_redesign.json` | 下述oneclass、notoneclass两套测试集都测 |
| notoneclass | `2000_notoneclass_sorted/train_grpo_episodes_redesign.json` | 下述oneclass、notoneclass两套测试集都测 |

两组都必须测试`data/test_data_600_oneclass/test_grpo_episodes.json`和`data/test_data_600_notoneclass/test_grpo_episodes.json`，不能只测与训练同名的一套。prepare阶段同时预检并保存两套测试输入。

oneclass攻击候选来自按目标文本语义选择的同一个最远类别；notoneclass为此前不限制候选来自单一类别的版本。两套训练数据使用相同2000个源ID、任务、标签和逐ID的clean/budget分配。攻击图分别来自各自版本。每条源数据仅使用一个选定版本，不能把clean.parquet或其余budget追加成更多训练条目。

| 训练任务 | 条数 | clean | budget1 | budget2 | budget3 | budget4 | budget5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NC | 1000 | 200 | 160 | 160 | 160 | 160 | 160 |
| LC | 200 | 40 | 32 | 32 | 32 | 32 | 32 |
| LP | 800 | 160 | 128 | 128 | 128 | 128 | 128 |
| 合计 | 2000 | 400 | 320 | 320 | 320 | 320 | 320 |

共有19种“数据集＋任务”组合：NC的arxiv、children、citeseer、computer、history、instagram、photo、pubmed、reddit、sports各100条；LP的arxiv、children、computer、history、photo、pubmed、reddit、sports各100条；LC的wn18rr为200条。LP正例407、负例393。各目录README、manifest和pair_alignment.csv提供完整构成及配对。

数据已经按degree升序排列，训练不打乱。NC取目标节点的无向去重邻居数；LC/LP先屏蔽查询边，再取两端degree均值；忽略自环及关系类型，同degree按source_index升序。两套攻击图degree可能不同，因此最终行号不保证相同，按源ID/budget配对，不要按行号合并，也不要为了行号一致重新排序。

每套测试恰好600条：NC450（cora/products/wikics各150），LC150（fb15k237）；budget1–5各120条，每档NC90+LC30，混合文件中的每条源样本只出现一次。两套测试的源ID、顺序、budget分配及clean图对应，攻击图不同。clean_graph保留为参考字段，正式测试使用noisy_graph。`badgraph/budget1–5`各600条是完整攻击轨迹对应版本，不是正式测试的3000条输入。

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

## 8×H200参考参数与可调整项

| 项目 | 值 |
| --- | --- |
| 基座及训练方式 | 原始Qwen2.5-7B-Instruct；新建LoRA；GRPO |
| LoRA | r=32，alpha=32，dropout=0.05，bias=none |
| LoRA目标层 | q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj |
| 训练精度 | BF16；FP16=False；TF32=True；无量化 |
| 分布式 | 8×H200，每卡约140GB显存；DeepSpeed ZeRO-3；无CPU offload |
| 每卡candidate batch / 梯度累积 | 参考48 / 1，可自行调整 |
| 全局每源样本候选数G | 固定6，保持每条数据6条轨迹；num_iterations=1；steps_per_generation=1 |
| 常规每次更新 | 8×48×1=384条轨迹，即64条源数据，每条6条轨迹 |
| 数据及更新数 | 2000源样本，1epoch，12000条候选轨迹；参考配置32次更新，末次16源数据/96轨迹 |
| 优化器 | adamw_torch；betas=(0.9,0.999)，eps=1e-8；lr=2e-6；weight_decay=0.01；max_grad_norm=1.0 |
| 学习率 | 参考lr=2e-6；warmup占总更新数5%，参考配置2步；cosine |
| GRPO | loss_type=grpo；beta=0.01；epsilon=0.2；scale_rewards=group |
| prompt / completion上限 | 8192 / 512 tokens；prediction_max_new_tokens=512 |
| 训练采样 | temperature=1.0；top_p=0.95；top_k=50；repetition_penalty=1.0 |
| 省显存配置 | FlashAttention2、Liger、梯度检查点use_reentrant=True、logprob_micro_batch_size=4，可调至8等 |
| 其他 | disable_dropout=True；use_vllm=False；max_tool_calling_iterations=0 |
| 顺序及种子 | seed=42；shuffle_dataset=False；全2000条，TRAIN_FRACTION=1 |
| 日志及保存 | logging_steps=1；save_steps=5（可调）；save_total_limit=3；report_to=none |
| 测试 | 参考单卡batch=8，可调整；贪心生成，每条1次，生成上限512；不投票或按测试表现挑checkpoint |

以上是**8张H200（每卡约140GB）的吞吐优先参考起点，可自行调整，并非已在该硬件实测的最优参数**。不再限制每次更新12条轨迹或2条源数据。参考配置使用全部8卡；batch表示每卡候选轨迹数，G表示每条源数据的全局轨迹数，不是每卡各生成G条。

**每条源数据固定生成6条轨迹，G=6保持不变。** 常规每次更新轨迹数=`GPU数×TRAIN_BATCH_SIZE×GRAD_ACCUM`；源数据数=`上述轨迹数/6`。参考batch48、累积1时，8卡一次更新384轨迹/64源数据。

当前采样器已经支持G=6与8卡组合，不再要求G能被GPU数整除。必须满足`(GPU数×每卡batch) % 6 == 0`，保证每个全局microbatch包含完整6轨迹组；还需`训练条数×6 % GPU数 == 0`，让尾批轨迹在各rank等量。2000条、8卡满足；8卡batch必须为3的倍数，例如24、48、72、96。源数据的6条轨迹可以跨rank分配，汇总奖励后仍按完整6轨迹组计算。尾批无法等量分配时预检报错，不丢弃、不重复补齐源数据。

| 8卡方案 | 每卡batch | 梯度累积 | 每条数据轨迹数 | 常规每次更新轨迹 / 源数据 | 2000条单轮预计更新数 |
| --- | ---: | ---: | ---: | --- | ---: |
| 参考起点 | 48 | 1 | 6 | 384 / 64 | 32 |
| 降低峰值显存，保持有效batch | 24 | 2 | 6 | 384 / 64 | 32 |
| 较小有效batch，增加更新次数 | 24 | 1 | 6 | 192 / 32 | 63 |
| 显存与吞吐允许时增大batch | 72 | 1 | 6 | 576 / 96 | 21 |
| 进一步试探吞吐 | 96 | 1 | 6 | 768 / 128 | 16 |

参考配置前31次各64条源数据，末次16条，合计2000条，轨迹总数始终12000。实际更新数为`ceil(ceil(训练条数/每个全局microbatch源数据数)/梯度累积)`，由preflight和Trainer状态核验，不再硬编码1000步。表中的轨迹数与源数据数是常规完整更新值，最后一次允许较小。

接手Codex可在不再询问固定batch的情况下，根据显存和实测吞吐自行调整TRAIN_BATCH_SIZE、GRAD_ACCUM、LOGPROB_MICRO_BATCH_SIZE、EVAL_BATCH_SIZE；两组正式训练使用同一套最终参数。先记录一次独立探测运行，使用最长prompt样本并覆盖生成512 token的峰值，测每卡峰值显存、源数据/秒、生成token/秒及step耗时。推荐从batch48开始，显存富余且吞吐继续提升时尝试72、96；显存不足时退回36或24，必要时增加累积。增大梯度累积主要改变有效batch，不会像增大单次batch那样充分利用额外显存，因此优先实测单次batch。不要为填满显存而接受吞吐下降或频繁OOM；保留约10%–15%显存余量应对长样本。探测输出与正式RUN分开，不能消耗正式训练的一部分后又从头重复而未记录。

LOGPROB_MICRO_BATCH_SIZE参考4，可试8以减少分块开销；每次只改一个参数，观察峰值与吞吐。G固定6；有效batch变化也会影响学习过程，不能只按显存判断效果，报告中记录最终参数和更新数。学习率保留2e-6作为起点，不按batch盲目线性放大；支持LEARNING_RATE覆盖。

warmup改为按总更新数约5%计算，避免32步训练仍使用旧55步预热。**锁定的Transformers版本使用`warmup_steps=0.05`表达比例**，即`ceil(总步数×0.05)`；脚本环境变量WARMUP_STEPS支持0.05或明确整数，不要使用该版本不存在的warmup_ratio参数。SAVE_STEPS参考5，可调整；最终始终保存final_adapter，不根据测试集选择checkpoint。换batch后重新检查实际warmup与保存间隔。

## 当前prompt、查询边与奖励

权威prompt由`code/episode.py::build_prediction_prompt`生成。NC/LC/LP共用结构与输出协议，分别明确预测语义。约束要求只回答指定目标；LC/LP屏蔽查询边，LP中未观察到查询边不等于负例；综合文本与结构证据，不依赖单个关键词、路径、共同类别、共同邻居或重复文本；相似不证明存在关系/链接，不相似不否定；缺失/冲突表示不确定，不能编造事实或候选标签。没有额外的`Query direction`字段。完整候选标签来自数据，包括LP正负两个候选；不能擅自改成简写yes/no。

`task_utils.prepare_episode`对副本进行处理，不修改原文件；clean与noisy视图都屏蔽LC/LP查询边，默认按无向查询屏蔽两方向；显式有向元数据的处理保留在代码中，不新增prompt方向要求。prompt只公开任务、目标、候选标签和当前视图图信息，不泄露ground_truth、clean_graph、攻击位置或budget。数据中已有占位文本作为缺失证据处理，不填造节点内容。

输出必须是一个JSON对象，顺序为analysis、prediction，两个值均为非空字符串，没有多余字段或外部文字；重复键、不完整JSON、截断且未正常结束等无效。奖励`R=S+T`：输出结构合法S=1，否则0；合法且候选正确T=1，合法但选错候选T=0.1，合法但不在候选中T=0；无效输出总奖励0。因此正确2、候选内错误1.1、候选外1、无效0。候选比较使用现有规范化规则，不把句子标签的单个yes/no视为完整匹配。以`reward.py`实现为准。

## 执行两组训练与测试

在同一个shell中保持以下参考变量，使用物理GPU0–7。可按上节步骤自行调整批量参数；两组使用一致的最终设置，不要覆盖已有RUN目录。

```bash
export REPO="$PWD"
export EXP="$REPO/organized_experiments/3_5_1_2"
export PYTHON_BIN="$REPO/.venv-3_5_1_2/bin/python"
export TRAIN_MODEL="$REPO/models/Qwen2.5-7B-Instruct"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRAIN_BATCH_SIZE=48 GRAD_ACCUM=1 NUM_GENERATIONS=6
export LOGPROB_MICRO_BATCH_SIZE=4 EVAL_BATCH_SIZE=8
export LEARNING_RATE=2e-6 WARMUP_STEPS=0.05 SAVE_STEPS=5
export TRAIN_FRACTION=1 FINAL_EPOCH=1 PREDICTION_TOKENS=512
export EXPECTED_TEST_COUNT=600 REQUIRE_TEST_TASK_COVERAGE=0
unset TEST_EPISODES  # 清除旧版单测试集配置
export TEST_ONECLASS_EPISODES="$REPO/data/test_data_600_oneclass/test_grpo_episodes.json"
export TEST_NOTONECLASS_EPISODES="$REPO/data/test_data_600_notoneclass/test_grpo_episodes.json"

# A：oneclass。prepare不会加载完整模型权重或启动训练。
export TRAIN_EPISODES="$REPO/2000_oneclass_sorted/train_grpo_episodes_redesign.json"
export RUN_DIR="$EXP/runs/oneclass_2000"
bash "$EXP/scripts/run_portable.sh" prepare
cat "$RUN_DIR/data/preflight.json"
bash "$EXP/scripts/run_portable.sh" train
bash "$EXP/scripts/run_portable.sh" merge
bash "$EXP/scripts/run_portable.sh" test

# B：notoneclass。从同一原始基座重新创建adapter，不沿用A的adapter。
export TRAIN_EPISODES="$REPO/2000_notoneclass_sorted/train_grpo_episodes_redesign.json"
export RUN_DIR="$EXP/runs/notoneclass_2000"
bash "$EXP/scripts/run_portable.sh" prepare
cat "$RUN_DIR/data/preflight.json"
bash "$EXP/scripts/run_portable.sh" train
bash "$EXP/scripts/run_portable.sh" merge
bash "$EXP/scripts/run_portable.sh" test
```

先确认每组preflight为train_count=2000、train NC1000/LC200/LP800、source_order_preserved=true；test_sets中的oneclass和notoneclass均为600条、NC450/LC150、missing_test_tasks=[LP]；参考配置下update_batch为sources_per_update=64、trajectories_per_update=384、optimizer_updates_per_epoch=32、final_update_sources=16；调整参数后按公式核验新的实际值。最大prompt小于8192。prepare不截断超长prompt，发现异常会报错。prepare完成后要在同一RUN_DIR运行train/merge/test，不能再次all/prepare覆盖；`all`只用于新的RUN目录从头完整执行。不要并行启动两组占满同一批GPU。

训练阶段输出adapter到`RUN_DIR/grpo/final_adapter`；合并阶段从adapter配置定位原始基座，在CPU加载BF16权重合并到`RUN_DIR/merged_model`，需预留主机内存和磁盘。此CPU合并与训练时CPU offload是两回事。评测默认单GPU，`EVAL_GPU`指定物理GPU，`EVAL_BATCH_SIZE`参考8。修改EVAL_BATCH_SIZE不改变样本和生成上限。

## 测试矩阵与比较

**只评测训练合并后的模型，condition固定为noisy_policy。每个训练run在oneclass、notoneclass攻击测试集上各测一次，每次600条；每个run共2次，两组训练合计4次。**

| 训练集 / 模型 | oneclass攻击测试集 | notoneclass攻击测试集 |
| --- | --- | --- |
| oneclass训练合并模型 | noisy_policy，600条 | noisy_policy，600条 |
| notoneclass训练合并模型 | noisy_policy，600条 | noisy_policy，600条 |

输出分别为`RUN_DIR/test/oneclass/noisy_policy/`和`RUN_DIR/test/notoneclass/noisy_policy/`，日志为`test_oneclass_noisy_policy.log`和`test_notoneclass_noisy_policy.log`。两套测试输入在prepare时保存为`RUN_DIR/data/test_oneclass_episodes.json`和`test_notoneclass_episodes.json`；test阶段缺少任意一套均报错。旧`test_episodes.json`保留为oneclass兼容副本，正式test阶段只读两个带名称的文件。

训练使用哪套数据都不会改变这个2×2测试矩阵。比较训练方法时，在同一测试集上比较两个训练模型；跨测试集比较时，固定训练模型。使用相同当前prompt和生成预算，每条只预测一次，始终输入对应测试版本的noisy_graph。所有正式评测用完整600条，不能使用`--limit`抽样结果替代；两套均无LP测试覆盖。

数据位置通过`TEST_ONECLASS_EPISODES`与`TEST_NOTONECLASS_EPISODES`分别指定；旧单一`TEST_EPISODES`变量会报错，请先`unset TEST_EPISODES`。如果准备目录来自旧版单测试流程，应新建RUN重新prepare，不能遗漏第二套测试。

## 输出、核验与故障处理

每个RUN应保留：`code_snapshot/`、`data/preflight.json`及处理后的train/test episodes、`prepare.log`、`train.log`、`grpo/experiment_config.json`、checkpoint/trainer_state、final_adapter、merged_model、两个测试集各自的predictions.jsonl与summary.json、status.txt。记录Git commit、数据manifest哈希、模型revision、GPU型号/数量、实际依赖freeze及启动命令。

训练结束检查实际global_step（参考配置32，随batch/累积调整）、epoch=1、final_adapter和合并权重存在、候选采样按配置的G组成完整组、两套攻击测试各完整600条。读取评测summary报告总体/NC/LC/数据集分组准确率、无效输出与截断数量；LP应标为无测试覆盖，不能填0伪装测得结果。预测无效保留为错误，不能删掉再计算准确率。不要通过加token、改prompt、反复挑checkpoint来选择性优化测试；需要改动另建运行并注明。跨版本比较必须使用同一测试版本与预算。

训练中断先检查日志与checkpoint，不要自动从头覆盖目录或宣称已完成；当前launcher没有自动resume开关，如需恢复应使用train_grpo.py已有的--continuation_checkpoint显式指定完整checkpoint（内部传入Trainer的resume_from_checkpoint），复用原启动参数并验证optimizer/scheduler/global_step恢复，保留改动和命令，不能把加载adapter但优化器重置当成无缝恢复。GPU显存不足/环境不兼容/模型下载失败等应给出实际原因与已完成阶段。

验证目录保留单元测试和此次迁移preflight摘要，不包含原服务器大型运行目录、模型、缓存、密钥。`design_notes.md`为实现背景，执行与最终prompt以本readme和当前代码为准。`source_manifest.json`保留来源及当前代码哈希。
