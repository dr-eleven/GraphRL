# MRRA B 三阶段生成、执行与格式奖励：实现说明

本次按用户确认的三阶段流程修改 `7_1-MRRA`。新协议版本为 `executor_three_stage_tags_v1`。用户提供的 [MRRA_B_complete_prompts.md](MRRA_B_complete_prompts.md) 保持原样，NC、LC 的 SYSTEM 分别提取到 [executor_nc_system.txt](../code/prompts/executor_nc_system.txt)、[executor_lc_system.txt](../code/prompts/executor_lc_system.txt)，运行时增加精确 JSON 动作字段、分阶段回滚和标签唯一性要求。

## 1. 实际执行流程

```text
初始 SYSTEM + USER（任务/可见图，Current stage: Modify）
  ↓ generate
<reason>文本与结构共同支持的结构分析</reason>
<modify_action>结构动作 JSON 数组或 []</modify_action>
  ↓ 遇到 </modify_action> 停止
Python 执行 Modify，一次阶段事务
  ↓ 追加 USER
执行状态/错误信息
<restored_graph>结构阶段结束后的实际完整图</restored_graph>
使用最新修改图进行后续推理。
Current stage: Rewrite
  ↓ generate
<reason>基于最新图的文本分析</reason>
<rewrite_action>文本动作 JSON 数组或 []</rewrite_action>
  ↓ 遇到 </rewrite_action> 停止
Python 执行 Rewrite，一次阶段事务
  ↓ 追加 USER
执行状态/错误信息
<restored_graph>文本阶段结束后的实际完整图</restored_graph>
使用最新修改图进行后续推理。
Current stage: Reason and Prediction
  ↓ generate
<reason>根据最终实际图推理</reason>
<answer>一个候选标签</answer>
  ↓ 遇到 </answer> 停止，计算整条轨迹奖励
```

每次 generate 都读取完整历史；第二次包含原始任务、Modify 原始输出和实际结构图，第三次继续包含 Rewrite 原始输出和实际最终图。Python 图反馈不是模型生成的恢复图。固定三次调用，不增加纠错重试。

首轮恢复只有结构操作 ADD_EDGE/DEL_EDGE/DELETE_NODE，第二轮恢复只有 REWRITE；NC 只改目标文本，LC 只改查询端点文本，查询端点和查询对的保护规则保持原义。

## 2. 非法阶段与实际图

- Modify 非法：该阶段所有动作回滚，返回原始输入图和错误；Rewrite 仍正常生成、执行。
- Rewrite 非法：该阶段所有文本动作回滚，保留 Modify 成功结果，再进行预测。
- 阶段输出无法解析：视为该阶段执行失败、保持该阶段输入图；仍反馈实际图并继续。
- 不保留非法阶段中的“部分合法动作”。此前成功的其他阶段不回滚。

日志中的 `stage_executions` 分别保存两个阶段的解析结果、动作、执行状态、回滚图和错误。汇总 `execution.error` 表示任一阶段失败；`rollback_scope=stage`，不表示整个图一定退回原始图。`execution.actions` 记录可解析的尝试动作，`execution.executed_actions` 只记录真正成功的阶段动作。

恢复指标以实际执行动作和实际最终图统计。例如 Modify 成功删掉一条错误边、Rewrite 失败时，B=0，但结构指标仍应记录那次真实的成功删除，不能因为整条轨迹出现错误就抹掉其恢复效果。

## 3. 奖励：R = B + T + F

三个奖励各为 0/1，每条完整轨迹只计一次，满分 3。

- **B**：任一阶段执行失败则为 0；都合法时，实际干净图要求整体 NO_OP，实际污染图要求至少一个非空编辑。仍不要求编辑真正恢复正确。
- **T**：最终 answer 标签内的文本按既有标签规范化与真值比较，正确得 1。前面动作失败或整体格式不合规时仍可独立得分，但缺失或重复 answer 标签不能提取为有效答案。
- **F**：三个阶段各自符合规定格式，才得 1；任一失败得 0，不按阶段累加。

F 检查正确阶段标签、顺序、唯一完整闭合、非空 reason、动作内容为严格 JSON 数组、非空 answer，以及无块外额外文本（允许前后空白）。重复 JSON 字段和 NaN 等非标准 JSON 不通过。动作对象的 op/字段/节点范围/保护约束及执行前提由执行器检查，属于 B；F 不使用真实标签判断答案正确性，也不评判 reason 是否事实正确。候选标签选择仍由 prompt 约束，T 负责标签匹配。

因此，格式正确但试图删除 protected node 的候选可以得到 F=1、B=0；三个阶段格式都正确但答案错误可以得到 F=1、T=0。多余说明导致格式失败，但唯一 answer 标签中答案正确时，T 仍可为 1。

A 保持原 JSON 协议和 B+T；直接预测基线仍只用 T。新 B 的总 Reward 与旧 A 的总 Reward 上限不同，不应直接比较原始 R 均值作为优劣结论。

## 4. generate 停止和 GRPO loss

训练批量生成和独立评估共享 `StopAfterStage`：按当前阶段设置 `</modify_action>`、`</rewrite_action>`、`</answer>` 停止关键词。每个候选独立停止，并记录停止位置，避免同批其他候选继续生成时，把 HF 后补的 padding/EOS 当成该候选实际采样 token。停止标签本身被保留；没有人为强制输出这些标签。

```text
输入上下文 | Modify输出 | Python反馈1 | Rewrite输出 | Python反馈2 | Reason/Answer输出
loss目标   |     1      |      0      |      1      |      0      |         1
```

所有历史可参与 attention；三个生成段共享同一个轨迹 R、组内 advantage 和 GRPO loss，按三个生成段 token 总数归一化，再在轨迹间平均。标签、动作 JSON 和文本都是模型实际输出的一部分。参考概率和策略前向读取真实 token 拼接的完整上下文；不对环境反馈做策略目标，也不以解码重编码替换前段采样 token。

## 5. 配置与入口

- [mrra_executor.json](../configs/mrra_executor.json)：新协议版本；Modify 上限 4096、Rewrite 上限 4096、最终回答上限 512，总生成预算仍为 8704。初始 prompt 上限 8192，反馈 prompt 上限 24576，总上下文上限 32768，超限报错而不静默截断图。
- B 采样仍为 temperature=1、top_p=1、top_k=-1（配置表示关闭），HF/TRL 实际接收 top_k=0；两个恢复段和预测段均使用同一训练采样配置。独立评估仍是贪心生成。
- 3B 入口保持 [run_mrra_executor.sh](../scripts/run_mrra_executor.sh)。
- 7B 入口 [run_mrra_7b.sh](../scripts/run_mrra_7b.sh) 现在按策略选择配置，B 使用 [mrra_executor_7b.json](../configs/mrra_executor_7b.json) 并读取 B 的采样参数；A 仍选择自己的 7B 配置。
- 旧两段 B checkpoint 不允许作为新三段 B 训练状态直接续训，防止协议和奖励静默切换。

## 6. 验证与复核文件

33 项 CPU 测试通过，含真实小型 Qwen 模型的 HF generate、真实 TRL/Trainer 的 A/B LoRA 更新与梯度累积、三段生成非零 loss 梯度和反馈位置零 loss 梯度、真实 Qwen tokenizer 的 EOS/反馈精确拼接、两种阶段失败和格式奖励独立性。三卡组分配、尾批、保存恢复等既有回归也通过。

此外，对全部 5495 条训练输入复核：A 的完整 prompt 与修改前完全相同，NO_OP 对应的 B+T 奖励与修改前一致。3B/7B shell 入口经语法和仅打印命令的干运行核验。

- [测试日志](../validation/B_three_stage_20261102/tests.log)
- [验证摘要与源代码哈希](../validation/B_three_stage_20261102/summary.json)
- 修改前关键实现备份：`validation/B_three_stage_20261102/before/`
- [新版 NC/LC 三次调用的完整 prompt 示例](MRRA_B_three_stage_prompts.md)：使用人工空动作和真实执行器展示，无 LLM 实验结果。

本次没有启动新训练、GPU 显存探针或新模型评估。先前旧 B 的 GPU 探针不能当作新三阶段协议的显存验证；正式训练前应使用现有 probe 入口重新检查。正在运行或已经完成的 A 实验输出没有被改写。
