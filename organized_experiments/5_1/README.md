# 5_1：二元终局奖励，最多 10 轮（自由调用）

基于 3_5，环境奖励仅为最终预测正确=1，否则=0。当前实现按用户最新要求采用 MatchTIR 风格的全轨迹轮数控制：最多 10 个模型生成轮次，没有单阶段/单工具次数上限，没有错误次数上限，没有独立工具总次数上限，不预留终答轮或终答 token，也不在最后额外补生成。

## 轨迹停止和评分

- 每次模型生成计一轮：成功调用、失败调用、最终答案或答案修正均计入10轮。
- 有效最终提交立即结束，预测正确=1、错误=0，错误合法答案不能重猜。
- 第10轮仍调用工具时，执行并记录这一轮工具结果后停止；没有有效终答则R=0，不生成第11轮。
- 第10轮输出有效终答时正常评分，不能因为已到10轮就覆盖正确终答的1分。
- 可解析的schema、工具名、阶段或动作错误返回全部可独立检测的错误，只消耗本轮，不因累计错误数提前停止。每批动作整批原子执行，任一错误都不提交。
- JSON/事件解析错误（重复键、边界错误、多事件、不完整输出等）仍按此前要求直接结束，R=0、不回错误obs、不重采。
- token/context硬容量可能早于10轮耗尽；无有效终答则R=0，不强制转终答。工具成功反馈放不下时不提交未被观察的图修改，直接以容量失败结束。
- 模型预算失败轨迹保留在GRPO中；真正的基础设施故障按完整G组在同一策略下最多额外重采2次，持续失败终止批次，不伪造R=0。

## Prompt消息结构

参考3_6重新设计：全部任务要求、图约束、输出协议、10轮预算及调度规则放在system；user只放公开任务/候选标签/保护信息/当前图。工具保留3_6的噪声修复含义；共同的证据要求、最新图、analysis顺序和空操作规则只在system说明，工具description仅保留用途，参数说明仅保留各自action字段。图结构/字段合法性仍由原校验器执行；证据充分性、保留事实、analysis与actions语义一致等仍为prompt要求，不代表存在额外语义检测器。完整实际渲染见Prompt.md。主训练始终是图交互后最终预测，同一模型同一轨迹；仅预测prompt只是评估基线。

## 工具调度和反馈

4_1：起始del_node，顺序为del_node→del_edge→add_edge→rewrite。每次成功（包括合法actions=[]）后，模型可重复当前工具或进入紧邻下一阶段；不能跳过阶段或回退。进入下一阶段后，该阶段调用失败也不能退回前阶段。失败后需在当前阶段得到新的成功才重新开放下一阶段选择。rewrite成功后可继续rewrite或终答。阶段绝不因调用/错误次数自动跳过。

成功反馈例：`next_tool: del_node or del_edge`。错误反馈列明全部可独立检测问题，不给下一阶段选择。

5_1：四工具任意顺序、可重复，可以零工具终答。成功反馈为`next_tool: any tool or final`。

两者均要求工具arguments严格按analysis、actions排序，analysis为非空字符串；最终JSON严格为analysis、prediction。每轮只能一个工具事件或一个终答。初始图使用 `<initial_graph>`，实际图变化才返回 `<restored_graph>`。无图变化不重复快照：已有更新则复用最近更新图，否则复用初始图。第10轮的工具反馈不再带下一轮assistant生成前缀。

## 默认资源预算

唯一代码定义在 `code/budgets.py`；快照在 `configs/binary_rollout.json`。原有阶段、错误次数和终答预留配置字段已经删除。

| 项目 | 默认值 |
|---|---:|
| 模型最大生成轮数 | 10 |
| 每轮最大新生成token（包含终答） | 2048 |
| 累计模型token | 12288 |
| completion总量，含反馈与消息边界 | 16384 |
| 初始prompt，含工具schema | 8192 |
| 每个调用最多actions | 64 |
| 工具执行限时 | 10秒 |
| 候选采集watchdog | 900秒 |

64个actions是单请求大小限制，不是阶段调用次数限制。工具超时和未知执行异常归入基础设施故障。上下文长度以实际模型配置为准。

## 数据和优化参数

本次训练：`GraphRL/1000_oneclass_sorted/train_grpo_episodes_redesign.json`，完整1000条，NC500、LC100、LP400，保持原排序，不进行1/4或其他抽样。

测试分别为以下两个文件，各600条，独立评估、独立保存结果：

- `GraphRL/600_test_data_notoneclass/test_grpo_episodes.json`
- `GraphRL/600_test_data_oneclass/test_grpo_episodes.json`

2026-11-14任务的启动参数参考3_6_1：基础模型Qwen2.5-7B-Instruct；LoRA r/alpha=32/32，配置dropout=0.05，disable_dropout=True；默认独占GPU1，ZeRO-2无CPU卸载，每卡候选batch=12、梯度累积1、G=6、BF16=True、FP16=False、TF32=True，学习率2e-6、warmup55、cosine、AdamW、KL系数0.01、梯度检查点与Liger开启，每10步保存。1000条完整训练1 epoch，共500次优化器更新。两套测试集各600条，分别评估训练模型的`restored`条件，即实际图工具交互后预测；不额外生成clean_policy或基座条件。

只参考资源和优化参数，保留5_1的原始10轮自由调度、0/1终局奖励及上述2048/12288/16384 rollout预算，不套用3_6_1的四工具强制顺序、两epoch分离、5120长度丢弃或S/B/H奖励。所有样本参与；预算失败按5_1原规则保留为失败轨迹。较长rollout会比3_6_1更耗显存，batch12是否可用需在目标GPU实际验证，不能将短序列显存快照当作最坏上限。配置的`GENERATION_MICRO_BATCH_SIZE=1`只分块生成候选，不改变训练batch。

模型路径通过`TRAIN_MODEL`传入，Python路径通过`PYTHON_BIN`传入；`train_grpo.py`按该模型路径校验，支持远端相同原始基座的不同绝对路径。训练、合并、评估均使用prepare时冻结的`RUN_DIR/code_snapshot`代码。

## 运行

```bash
cd /实际/GraphRL/organized_experiments/5_1
export PYTHON_BIN=/实际/兼容环境/bin/python
export TRAIN_MODEL=/实际/Qwen2.5-7B-Instruct
export RUN_DIR="$PWD/runs/5_1_1000_oneclass_seed42"
CUDA_VISIBLE_DEVICES=1 bash scripts/run_jhh_1000.sh all
```

通用脚本`run_binary_1epoch.sh`默认prepare；本次包装入口`run_jhh_1000.sh`默认all，顺序执行完整prepare→train→merge→两个600条测试。结果保存在`RUN_DIR/test/<测试集>/restored/`。可显式分段调用包装入口的prepare/train/merge/test，但必须沿用同一RUN_DIR。旧协议的已准备run不能冒充当前run；恢复检查点须本项目同一二元reward与完全一致的当前budget。

## 代码与验证

- `code/episode.py`：10轮状态机、阶段/自由调度、原子提交与终局奖励审计。
- `code/protocol.py`：严格事件解析；`code/validation.py`：schema/阶段/图约束的多错误收集。
- `code/generation_limits.py`：真实token累计、obs回填、独立attention/loss mask、完整G组基础设施重采。
- `code/train_grpo.py`：原生TRL GRPO优势和loss、LoRA更新与完整组sampler。
- [Prompt.md](Prompt.md)：实际样本完整渲染的系统消息、工具schema、用户prompt、图和生成前缀。运行`python tools/export_prompts.py`可重新导出。
- `verification/unit_and_integration.log`：确定性状态机、生成循环、真实CPU tiny-Qwen GRPO更新、mask、全0/全1组验证。
- `verification/data_preflight.json`：历史2000/600/600数据预检查；本次1000/600/600检查另存到本次run的`data/preflight.json`。`data_smoke.json`为32条预选实际图的控制器验证，终答是脚本化标签，不是模型准确率。

```bash
PY=/opt/data/private/.venvs/graphrl-3_3-trl-latest/bin/python
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest discover -s tests -v
"$PY" tests/data_smoke.py
```

历史verification文件的2000条与GPU未启动状态仅对应先前代码验证，不能代替本次部署与运行状态。CPU集成测试仅在生成叶子使用脚本化场景，真实运行策略/reference前向、GRPO loss、反向和LoRA优化器更新，并另行验证真实HF有界生成。正式训练以本次run日志、checkpoint及完成标记为准。
