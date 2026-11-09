# 7_1-MRRA B：执行器反馈的三阶段 GRPO

这是 `7_1-MRRA` 当前 B 策略的代码发布版本。训练基于 `ThinkWithGraphTrainer → GraphR1GRPOTrainer → trl.GRPOTrainer → transformers.Trainer`，沿用 3_degree-sorted 的底层实现和参数风格。

## 交互协议

1. 模型生成 `<reason>...</reason><modify_action>[...]</modify_action>`，在动作结束标签处停止；Python 执行结构操作。
2. 将实际图放入 `<restored_graph>...</restored_graph>`，附加“使用最新修改图进行后续推理。”；模型生成 `<reason>...</reason><rewrite_action>[...]</rewrite_action>`，停止并执行文本操作。
3. 再反馈实际图，模型生成 `<reason>...</reason><answer>候选标签</answer>`。

允许 `[]` 空动作。非法动作原子回滚当前阶段，保留此前成功修改，继续下一阶段，不重试。NC 只允许改写目标节点；LC 只允许改写查询端点，查询边受保护。

三段模型生成 token 共享一个完整轨迹的 GRPO 优势；Python 图反馈参与 attention，不参与 loss。

## 奖励

`R = B + T + F`，各项为 0/1，总分 0–3。B 判断合法动作是否符合实际干净输入不编辑、受损输入非空编辑；T 判断答案正确；F 判断三段输出格式。格式奖励不等价于动作合法或恢复正确。隐藏 clean 图和真实标签不进入模型输入或执行器。

## 文件

- `code/prompts/executor_nc_system.txt`、`executor_lc_system.txt`：实际加载的 system prompt。
- `code/executor_protocol.py`：每轮提示、反馈、解析、阶段执行与回滚。
- `code/train_grpo.py`、`train_grpo_7b.py`：TRL 训练入口。
- `configs/mrra_executor.json`、`mrra_executor_7b.json`：3B/7B 配置。
- `reports/MRRA_B_prompt_revision_20261104.md`：NC/LC 三轮完整提示示例，其中 NO_OP 输出是协议演示。
- `publication_manifest.json`：发布文件与本地原文件的 SHA256 对照。

部分本地公共模块保留 A/直接预测分支，作为共享实现及回归测试依赖。本次仅提供 B 启动配置，没有附带历史训练结果、模型权重或数据。

## 环境、模型和数据

Python 3.10，依赖版本见 `requirements.txt`。GPU 训练需要与 PyTorch 匹配的 CUDA、DeepSpeed、FlashAttention2 和 Liger。

本发布保持原实验代码与配置内容。默认模型和 Python 路径是原服务器路径；在其他环境设置 `PYTHON_BIN`、`DEEPSPEED_BIN`、`TRAIN_MODEL`。数据文件通过 `configs/base.json` 的 `data.train` / `data.test` 指定，相对路径以本项目目录为基准；请改为本机数据路径或使用 `MRRA_CONFIG` 指定继承配置。模型与数据需另行准备。

```bash
cd organized_experiments/7_1-MRRA
export PYTHON_BIN=/path/to/env/bin/python
export DEEPSPEED_BIN=/path/to/env/bin/deepspeed
export TRAIN_MODEL=/path/to/Qwen2.5-3B-Instruct

# CPU 回归检查；Qwen tokenizer 检查会使用本地 7B tokenizer 路径。
bash scripts/check.sh

# 3B B 策略：默认 3 卡、每卡 batch=8、累积=4、同源候选=6。
GPU_IDS=0,1,2 CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/run_mrra_executor.sh train

# 若显存不足，明确回退到 batch=4、累积=1。
TRAIN_BATCH_SIZE=4 GRAD_ACCUM=1 bash scripts/run_mrra_executor.sh train

# 训练后 merge，再评估 noisy/clean 以及直接预测基线。
bash scripts/run_mrra_executor.sh all

# 7B B 策略；先按设备资源调整 batch、累积和 DeepSpeed 配置。
TRAIN_MODEL=/path/to/Qwen2.5-7B-Instruct bash scripts/run_mrra_7b.sh executor train
```

训练采样 `temperature=1, top_p=1, top_k=-1`（配置约定；HF/TRL 实际传 0 关闭 top-k）。默认 Modify/Rewrite 各4096 token、预测512 token；最新三阶段协议应重新进行目标设备显存验证，不能用旧两阶段探针替代。

`all` 的主训练模型由 TRAIN_MODEL 控制；基线模型和其他推理默认值读取配置，因此迁移环境时也需同步配置中的 model.path。

## 来源

Graph-R1/degree-sorted 公共组件来源记录保留在 `source_provenance.json`。本目录源码、配置和脚本与服务器原实验逐文件校验一致；仅新增发布说明、忽略规则和清单。
