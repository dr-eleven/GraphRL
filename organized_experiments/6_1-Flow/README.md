# 6_1-Flow：8×H200 跨服务器执行说明

本文供接手服务器上的 Codex 或实验执行者直接运行，内容组织参考仓库中的 [3_5_1_2 执行说明](../3_5_1_2/readme.md)。参数以本目录实际代码、`configs/settings.json` 和 `configs/graph_flow.yaml` 为准。仓库：https://github.com/dr-eleven/GraphRL 。以下命令从仓库根目录执行，不依赖原服务器的 `/opt/data/private` 路径。

## 任务与实验边界

使用 AgentFlow 的 Planner → Executor → Verifier → Memory → Generator 流程解决 GraphRL 的 NC（目标节点分类）、LC（查询边关系分类）、LP（查询链接存在性预测）任务。保留原 AgentFlow Solver、各模块提示词和 Flow-GRPO 的按 Planner 决策训练方式，将任务、工具、模型及奖励适配为 GraphRL。底层训练为 **VERL 0.5.0 + Ray + FSDP + vLLM**，不是 TRL。

所有模块的基座统一为原始 **Qwen/Qwen2.5-7B-Instruct**。只有 Planner 使用 LoRA 更新；问题分析、Executor 命令生成、Verifier 和最终 Generator 共用冻结的原始 7B 模型。训练结束后先 merge Planner adapter，再评测合并后的 Planner 与冻结模块组成的完整流程。Generator 不会加载 Planner adapter。

本包发布的是 8×H200 的执行配置与代码，**尚未在 8×H200 上完成实际训练或吞吐验证**。原服务器的 3B 运行、checkpoint、模型权重、虚拟环境和缓存不随包发布，也不作为这次 7B 训练的起点。`prepare` 成功只表示数据准备成功，不代表训练、merge 或测试完成。

## 数据与测试范围

| 用途 | 仓库根目录相对路径 | 条数与任务 |
| --- | --- | --- |
| 训练 | `1000_oneclass_sorted/train_grpo_episodes_redesign.json` | 1000：NC 500、LC 100、LP 400 |
| oneclass 测试 | `600_test_data_oneclass/test_grpo_episodes.json` | 600：NC 450、LC 150 |
| notoneclass 测试 | `600_test_data_notoneclass/test_grpo_episodes.json` | 600：NC 450、LC 150 |

训练数据为 200 条 clean 和 800 条攻击图，budget1–5 各 160 条。默认只训练 oneclass 的这一组 1000 条，不启动第二组训练。两个测试集均必须评测；测试没有 LP，应在结果报告中写“无 LP 测试覆盖”，不能把它当作测得的 0% 或声称验证了 LP 泛化。

这些指定 episode 文件已在仓库中直接提供，clone 后即可读取，不需要恢复 `3_5_1_2` 的四套压缩包。不要把其他 budget 文件、clean 参考图或原目录中的 parquet 追加为额外训练样本。准备脚本重建公开任务问题；输入使用 noisy_graph，ground_truth 和 clean_graph 只用于奖励与离线统计，不送入模型。LC/LP 的查询边屏蔽沿用 GraphRL 数据加载逻辑。

默认训练 `shuffle=true`、seed=42：源文件虽按 degree 排序，本实验会打乱训练顺序。这一点与 `3_5_1_2` 的不打乱设置不同。

## AgentFlow 流程、工具与奖励

一次完整轨迹先由冻结模型进行 Query Analysis；随后每轮由 Planner 选择一个工具和子目标，Executor 生成具体命令，GraphRL 校验并执行，Verifier 判断 STOP 或 CONTINUE。最多 **10 轮工具规划**，正常路径最多 `1 + 10×3 + 1 = 32` 次 LLM 调用，其中 Planner 最多 10 次，其余由冻结模型完成。Verifier 可提前停止，500 秒流程时间预算也可能减少实际轮数；时间预算在工具轮次之间检查，不是强制终止正在执行的 HTTP 请求。单次 HTTP 请求超时为 600 秒。

| 工具 | 动作 op | 约束概要 |
| --- | --- | --- |
| `del_node` | `DELETE_NODE` | 删除有依据的非保护噪声节点 |
| `del_edge` | `DEL_EDGE` | 删除当前存在、可编辑的边 |
| `add_edge` | `ADD_EDGE` | 添加当前不存在、可编辑且有证据支持的边 |
| `rewrite` | `REWRITE` | 修复允许编辑的目标/查询端点文本 |

每次只选择一个工具，但一次命令可以提交同种操作的 **1–64 个 action**。`actions=[]` 被提示词和校验器共同禁止。节点存在性、保护节点/边、自环、重复与冲突动作、边存在性、字段、操作类型等均检查。Executor 只接受 `execution = tool.execute(analysis="...", actions=[...])` 形式的单条字面量调用，用 AST 解析，不执行任意生成代码。

失败调用返回错误并保留原图；错误进入 Memory 后允许下一轮重新规划，失败也占用一次工具轮次。普通 STOP、轮数用尽或轮间超时后会调用最终 Generator；模块容量异常则按代码提前结束，不保证进入 Generator。没有强制要求四种工具全部使用成功才允许回答。

Planner 收到原始公开问题及 initial_graph、问题分析、动作/错误历史和最新 current_graph。完整历史图保留在审计 Memory 中，旧轮图不全部重新注入 Planner。最终回答从 Generator 输出提取候选标签；正确候选奖励 1，其余情况为 0，图恢复精确率不参与训练奖励。同一轨迹的终局奖励传播到它的 Planner 转移，按 AgentFlow 的 Flow-GRPO 逻辑形成训练数据；冻结模块输出不计算策略梯度。

## 基座模型与环境

推荐单节点 Linux x86_64、Python 3.10（也可使用兼容的 3.11）、8 张 H200，以及支持 CUDA 12.4 的 NVIDIA 驱动。H200 的显存余量应通过实际最长轨迹探测确认。核心版本为 torch 2.6.0/cu124、vLLM 0.8.5.post1、VERL 0.5.0、Transformers 4.52.4、PEFT 0.19.1、Ray 2.46.0、FlashAttention 2.7.4.post1。不要直接使用 `3_5_1_2` 的 TRL/Transformers 环境。

`requirements-native.txt` 锁定本实现的直接运行依赖，`verification/environment_source.json` 记录源环境版本。`setup.sh` 创建独立 venv，安装 CUDA 版 PyTorch、运行依赖和 FlashAttention，最后执行 pip check 并保存安装环境。FlashAttention 源码安装需要匹配的 CUDA toolkit/nvcc、C++ 编译器与构建资源；也可事先在该环境安装对应 wheel。安装失败需记录具体依赖问题，不要将不同版本默认为已验证。

```bash
git clone https://github.com/dr-eleven/GraphRL.git
cd GraphRL
export REPO="$PWD"
export EXP="$REPO/organized_experiments/6_1-Flow"
BASE_PYTHON=python3.10 bash "$EXP/scripts/setup.sh"
export PYTHON_BIN="$EXP/.venv/bin/python"
export TRAIN_MODEL="$REPO/models/Qwen2.5-7B-Instruct"
"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct',
    revision='a09a35458c702b33eeacc393d103063234e8bc28',
    local_dir=os.environ['TRAIN_MODEL'])
PY
nvidia-smi
"$PYTHON_BIN" "$EXP/code/preflight.py"
"$PYTHON_BIN" -m unittest discover -s "$EXP/tests" -v
"$PYTHON_BIN" "$EXP/code/check_launch.py"
```

若使用自定义模型目录，preflight 之前可先运行下节 prepare，以 `FLOW_SETTINGS="$RUN_DIR/runtime_settings.json"` 调用 preflight。模型应保留完整 config、tokenizer 与原始权重；训练以离线方式读取本地文件，不自动下载。`BASE_PYTHON` 可指定现有 Python，`EXPERIMENT_ENV` 可改变 venv 位置；使用后应将 PYTHON_BIN 指向对应解释器。

## 8×H200 启动参数

| 项目 | 发布默认值 |
| --- | --- |
| 所有角色模型 | Qwen2.5-7B-Instruct，原始 Instruct 权重，无量化 |
| GPU 0–5 | 6 个 FSDP Planner 训练 rank；采样阶段每卡一个 TP=1 的 vLLM Planner 服务 |
| GPU 6–7 | 冻结 7B 服务，TP=2；Query Analysis / Executor / Verifier / Generator 共享 |
| 精度 | BF16 模型、LoRA 参数、梯度通信；Adam 一二阶状态由 BF16 参数初始化；merge 输出 BF16 |
| LoRA | rank=32，alpha=32；q/k/v/o_proj、gate/up/down_proj |
| 每次更新原始问题数 | 2 条，全局值 |
| 每个问题采样数 | 6 次完整轨迹尝试，全局值；每个源批次合计 12 次 |
| 轨迹并发 | 全局最多 12；每个 Planner 服务最多 2 条 |
| 每卡训练微批次 | 2 个 Planner 转移；不是 2 条完整轨迹 |
| 更新方式 | 汇总本批保留轨迹的 Planner 转移，动态累积后执行一次 optimizer step；ppo_epochs=1 |
| 学习率 / KL / clip | 1e-6 / 0.001 / low 0.2、high 0.3 |
| 生成上限 | **每次 LLM 调用最多生成 8192 token**，训练与测试、Planner 与冻结模块均生效 |
| Planner 单转移训练阈值 | **prompt 实际 token + response 实际 token ≤8192**；任一轮超限则排除整条轨迹 |
| 独立 prompt 容量 | 18432；超出时报容量异常；不是默认训练准入阈值 |
| 服务上下文容量 | 32768，允许 18432 prompt + 8192 生成；冻结服务也为 32768 |
| 训练采样 | Planner temperature=0.7；冻结模块 temperature=0；top_p=0.99、frequency_penalty=1.2 |
| 测试采样 | Planner 与冻结模块 temperature=0；每样本一条轨迹，无投票 |
| 轮次 / 时间 | 最多 10 轮工具规划，流程时间预算 500 秒 |
| CPU offload | H200 发布配置关闭 actor 参数、优化器及 reference 参数 offload |
| GPU 服务显存比例 | 0.85；Planner max_num_seqs=4；冻结服务 max_num_seqs=12 |
| NCCL | 默认启用 P2P 和 IB；可通过环境变量适配具体互联 |
| 训练轮次 / 保存 | 1 epoch；每次有效更新保存，最多保留 2 个 checkpoint |

**8192 的两个限制是不同含义。** 单次输出预算为 8192，不表示每次一定生成这么多；训练准入的 8192 同时计算 prompt 和 response，因此输入 6000 token 时，能进入训练的该次输出最多为 2192 token。服务窗口 32768 是为完整采样请求预留的上下文容量，不表示 32768 token 的序列会参与反向传播。不要将 max_num_batched_tokens（单次调度 token 预算）当作总上下文窗口。

FSDP 在 6 卡间分配 Planner 参数和训练转移；冻结模型的两张卡不参与梯度更新。一个源批次的实际转移数随轨迹长度改变，代码将它补齐为 `6×每卡微批次` 的倍数；补齐项的 response mask 为 0，不产生训练梯度。默认每卡微批次 2 时对齐到 12。这里没有“每 12 个转移更新一次”的固定限制，也不是把 12 条完整轨迹同时送进一次前向。

超过长度限制的轨迹不会截断后强行训练，也不会重采样补足 6 条。如果某个原始问题保留下来的轨迹不足 2 条，本源批次跳过，不执行 optimizer/scheduler 更新。1000 源问题一轮有 500 个源批次、6000 次轨迹尝试；**有效 optimizer 更新最多 500 次**，有跳批时会更少。保存最新成功更新，merge 不按测试效果选择 checkpoint。

H200 参数是可执行起点，不是实测的最优吞吐配置。微批次可通过 train 的 Hydra 参数覆盖；调整后记录实际值与显存峰值。保持每批 2 条源数据、每题 6 次采样的实验定义；不要照搬 `3_5_1_2` 的 batch48 或 64 源问题/更新。要调整并发或 token 阈值，应复制 settings 到新文件，设 SETTINGS_FILE 并创建独立 RUN_DIR。

## 训练、merge、测试

在同一个 shell 中设置以下变量，父进程应能看到物理 GPU 0–7。launcher 在启动子进程时分别设置训练和冻结服务可见卡；无需用 torchrun 包裹，Ray/VERL 负责创建 FSDP worker。

```bash
export REPO="$PWD"
export EXP="$REPO/organized_experiments/6_1-Flow"
export PYTHON_BIN="$EXP/.venv/bin/python"
export TRAIN_MODEL="$REPO/models/Qwen2.5-7B-Instruct"
export RUN_DIR="$EXP/runs/7b_h200_8gpu_$(date +%Y%m%d_%H%M%S)"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRAIN_GPUS=0,1,2,3,4,5
export FIXED_GPUS=6,7 FIXED_TP=2 FIXED_MAX_SEQS=12
export EVAL_GPU=0 EVAL_TP=1
unset TRAIN_LIMIT EVAL_LIMIT

bash "$EXP/scripts/pipeline.sh" prepare
cat "$RUN_DIR/data/preflight.json"
FLOW_SETTINGS="$RUN_DIR/runtime_settings.json" "$PYTHON_BIN" "$EXP/code/preflight.py"
bash "$EXP/scripts/pipeline.sh" train
bash "$EXP/scripts/pipeline.sh" merge
bash "$EXP/scripts/pipeline.sh" test
```

也可在全新 RUN_DIR 中执行 `bash "$EXP/scripts/pipeline.sh" all`，自动依次完成四阶段。分阶段执行时必须保持相同 RUN_DIR；不要在已 prepare 的目录再次执行 all/prepare。train 遇到已有 checkpoints 会拒绝覆盖。若仅探测可设 TRAIN_LIMIT 为正偶数、EVAL_LIMIT 为测试抽样数，输出到独立 smoke RUN；抽样评测写入 smoke_test，不能作为正式 600 条结果。

prepare 应显示 train=1000、两套 test 各 600，并记录源文件 SHA256。运行设置解析为 `RUN_DIR/runtime_settings.json`，模型和数据的相对路径统一相对于仓库根目录；TRAIN_MODEL、TRAIN_EPISODES、TEST_ONECLASS_EPISODES、TEST_NOTONECLASS_EPISODES 可覆盖路径。同一 RUN 的设置变化会被拒绝，避免 train 与 test 使用不同配置。

例如降低微批次：`bash "$EXP/scripts/pipeline.sh" train actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`。这不改变全局 12 次轨迹尝试或最多 12 条并发，仅改变训练阶段每卡同时处理的 Planner 转移数量。配置检查默认要求 6 张训练卡、2 张冻结卡，且两组卡不能重叠。

冻结服务仅绑定 `127.0.0.1:8621`，用于本机模块推理；合并后的评测 Planner 绑定 `127.0.0.1:8622`。不需要向公网开放这两个端口。merge 从最近成功 checkpoint 的 `actor/lora_adapter` 读取 adapter，使用其记录的基座路径在 CPU 上 BF16 合并，输出 merged_model；需要预留主机内存和磁盘，CPU merge 与训练 CPU offload 的设置无关。

## 测试矩阵与结果报告

| 训练模型 | oneclass 攻击测试 | notoneclass 攻击测试 |
| --- | --- | --- |
| oneclass 1000 条训练后的合并 Planner + 冻结原始 7B 模块 | noisy_graph → 工具修复 → 最终预测，600 条 | noisy_graph → 工具修复 → 最终预测，600 条 |

每个测试样本只运行一条完整 AgentFlow 轨迹。测试开始时同样使用原始 noisy_graph；不给 clean_graph、攻击节点列表或正确答案。两个测试输出分别保存在：

- `RUN_DIR/test/test_data_600_oneclass/restored/predictions.jsonl` 和 `summary.json`。
- `RUN_DIR/test/test_data_600_notoneclass/restored/predictions.jsonl` 和 `summary.json`。

summary 包含总体准确率、有效终答数、容量失败数、工具尝试/错误数、图恢复精确数与 per_task 的 NC/LC 结果。无效答案和容量失败保留在总体 600 条分母中。按工具的平均调用数、成功数、是否进入最终回答以及平均/最大 token，可从 predictions.jsonl 和 train_rollouts.jsonl 中另行汇总；不要把“工具执行合法”“图恢复正确”“最终预测正确”混作同一指标。

## 输出核验与故障处理

每个 RUN 应保留 runtime_settings.json、settings.json、code_snapshot、data/preflight.json、处理后的 train/test 数据、prepare/train/merge/test 日志、pipeline_status.json、train_rollouts.jsonl、checkpoint、merged_model、两套测试 predictions/summary、actor_dtype_rank*.json、optimizer_dtype_rank*.json、request_diagnostics 等。记录 Git commit、GPU/驱动、模型 revision、依赖 freeze、数据 SHA256、实际启动命令、有效更新数、跳批数、丢弃轨迹数和最终参数。

检查每套正式测试 count=600；训练是否完成一轮应结合源批次消费、日志与 checkpoint 判断，不能只要求 global_step=500。发布时的验证记录见 verification/publication_checks.json；其中的 CPU tiny-Qwen 测试证明真实 LoRA 前后向及单步更新逻辑可执行，不代表 8 卡 H200 分布式训练已验证。

常见容量错误包括：固定 Query Analysis 在首次 Planner 前达到生成上限，轨迹记录 NO_PLANNER_TRACE_CAPACITY；或某轮 Planner 的实际 prompt+response 超过 8192，整条轨迹记录 PLANNER_TOTAL_TOKEN_LIMIT。已有原始输出表明，固定模型可能复制 JSON Schema 或生成重复空白，本次提高生成预算没有重写这些提示词，也不能据此声称消除了输出退化。finish_reason=length 仍记为容量失败，不伪造终答。达到过长阈值后按现有准入规则排除，不按奖励挑选重采样。

训练异常退出先读取 train.log、fixed_server.log 和 request_diagnostics；不得覆盖原运行目录或静默更换模型。当前 pipeline 的 train 入口默认从原始 7B 开始，未提供自动续训命令；需要恢复时应显式使用底层 VERL 的完整 checkpoint 恢复配置，并核验 optimizer、scheduler、RNG 与 dataloader 状态，不能将仅加载 adapter 说成完整恢复。

本目录 upstream 包含 AgentFlow 源码及原 LICENSE，source_manifest.json 保留来源 commit 与原始文件哈希；publication_manifest.json 记录本次发布文件哈希和实际修改的上游文件。plan.md 与实现实例文档保留历史设计背景，硬件、模型和 token 配置以本 README 与当前发布配置为准。
