# 3_5_1：带噪图直接预测

基于 3_5 复制源码、配置、依赖和启动脚本，改为单轮分析和类别预测。共享原有基座、源数据和 Python 环境；代码和正式运行输出独立。复制来源及原文件哈希见 `source_manifest.json`。

## Prompt

保留原任务类型、预测目标、候选标签、图节点文本和边，以及图可能干净或包含结构/文本噪声、噪声位置及预算未知的说明。针对 NC 预测目标节点类别，针对 LC 预测查询边类别。

删除图编辑约束、工具说明/schema、tool_call/actions 格式、工具顺序、阶段标记及执行反馈。模型直接根据提供的图结构和节点文本进行分析，从候选标签中选择类别。只保留预测阶段输出格式：

```json
{"analysis":"非空的预测分析","prediction":"一个候选标签"}
```

字段必须按上述顺序出现且只能包含这两个字段；两者均为非空字符串，不允许额外文字或 Markdown 代码围栏。prompt 使用公开字段重新生成，不继承源数据中的旧修复 prompt，也不暴露 clean_graph、ground_truth 或噪声注入元数据。

## Reward

`R=S+T`，满分 2 分，无 B 或工具相关奖励。

| 输出情况 | S 结构分 | T 预测分 | 总分 |
| --- | ---: | ---: | ---: |
| 格式合法，标签正确 | 1 | 1 | 2 |
| 格式合法，候选内标签但预测错误 | 1 | 0.1 | 1.1 |
| 格式合法，标签在候选外 | 1 | 0 | 1 |
| 格式错误、缺失预测、额外消息或工具调用 | 0 | 0 | 0 |
| 生成触顶且未以 EOS 正常结束 | 0 | 0 | 0 |

预测标签沿用 3_5 的归一化比较规则（忽略大小写、规整空白和首尾标点）。ground_truth 仅由评分端读取。候选外非空字符串仍满足结构格式，但不得预测分。

## 训练和评估

训练和评估都直接生成一次，不注册任何工具或环境、不修改图。保留三卡 7B LoRA、ZeRO-3、每卡 batch=2、梯度累积=2、每源 6 候选、原采样顺序和 1 epoch 配置。单轮预测默认最多 512 token，支持 `PREDICTION_TOKENS`；总 completion 预算同步使用该值。

完整流程（prepare → train → merge → test）：

```bash
bash /opt/data/private/GraphRL/organized_experiments/3_5_1/scripts/run_direct_batch4_1epoch.sh all
```

输出默认写入 `3_5_1/runs/3_5_1_<时间>`，每 100 步保存 checkpoint，保留最近 3 个。评估条件为 `noisy_policy`、`clean_policy`、`noisy_baseline`、`clean_baseline`，分别比较训练后模型和原始基座在输入图与 clean 图上的直接预测。所有条件使用相同 prompt 和 S/T 评分规则。

## 验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /opt/data/private/.venvs/graphrl-3_3-trl-latest/bin/python -m unittest discover -s /opt/data/private/GraphRL/organized_experiments/3_5_1/tests -v
```

`verification/tests.log` 包含奖励边界、格式/截断、隐藏信息隔离、CPU 小模型原生 TRL 单轮 LoRA 更新和真实生成验证。LoRA 更新测试只替换生成叶子，实际执行 reward、advantage、loss 和 optimizer；另一次未替换生成叶子的验证检查单轮输出与预算。未在测试中加载完整 7B 模型。

`verification/prepared_run/data/preflight.json` 是全量数据 prompt 长度审计，`verification/prepared_run/data/example_prompt.txt` 提供完整 prompt 示例。正式训练尚未启动。
