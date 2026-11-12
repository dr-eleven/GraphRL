# 3_6_1：先图操作、后预测的两个epoch

基于3_6实现，两阶段严格分离，每阶段遍历同一份抽样源数据一次。启动条件：3_6 checkpoint-2200 的 clean/noisy策略测试完成、3_6分析完成、新900数据集到位并通过校验。基座测试跳过。

## 数据

默认输入为 `GraphRL/900_train_data_clean20_badgraph80_degree_sorted/train_grpo_episodes_redesign.json`。目录下只有 `train_episodes.json` 时也支持；若有歧义则拒绝。可用 `--source /实际路径` 显式指定。绝不退回旧5495条数据。

已对接4090生成的900条版本：训练JSON的SHA256为`d95c14ae183e4a6b491622a9187339b6fbfba0d45f01f3f6e75c4058ceb7560b`。接口从`configs/data_source.json`校验900条总数和clean180/budget1–5各144的分层数量；抽样后固定225条，即clean45、各budget36。GitHub仓库为`dr-eleven/GraphRL`，数据在仓库根目录同名文件夹。数据迁移仍保留原始JSON字节、parquet及来源manifest。

以原始 `attack_budget=0..5` 对应clean、budget1–5分层，seed42分别随机保留floor(N/4)条，然后保持原degree顺序。个别预算>0但图实际clean的样本仍留在原预算层。保存源/子集哈希、各层数量、完整source_index清单。不能将数据集名称中的900当作已核验总数。

## 两个epoch

| 项目 | epoch1 | epoch2 |
| --- | --- | --- |
| 输入 | 抽样后的原始输入图 | epoch1最终策略实际生成的固定终图 |
| 模型行为 | del_node、del_edge、add_edge、rewrite各一次 | 一次analysis+prediction |
| 终止 | 第四次外部工具反馈后，不再生成第五轮 | 输出预测JSON后 |
| 奖励 | S四阶段各0.2+B四阶段各0.25+原始H | 预测格式S=1/0，T=1/0.1/0 |
| 注册工具 | 四个 | 无 |
| 检查 | 工具格式、动作合法性、四阶段结构 | 预测JSON、标签、EOS；无图执行检查 |
| clean监督 | 隐藏评分端 | 答案评分端；不进入prompt |

epoch1从原始Qwen2.5-7B-Instruct初始化，不从3_6 checkpoint续训。epoch1结束merge自身LoRA，为每个入选源用最终策略greedy生成一条终图，无clean替换、无用奖励选最优候选；格式失败时保留实际部分编辑后的图并记录原因。这样每个源都有一个stage2输入。epoch2在相同原始基座上加载epoch1的同一个LoRA adapter并继续训练，不新增第二套adapter，rank/alpha仍为32；每个源再遍历一次。两阶段权重连续，新建优化器和调度器，优化器状态不连续。

epoch2最终adapter已经包含连续训练的两阶段更新，必须与其配置记录的原始基座merge一次，不能再叠加到epoch1合并模型上重复计算更新。测试时先用冻结的epoch1策略处理clean/noisy测试图，再用epoch2合并模型预测；不让epoch2重新执行图工具。

## 参数与显存

默认沿用当前3_6：GPU1、ZeRO-2无CPU卸载、batch12/累积1/G6、BF16、LoRA r32/alpha32/dropout0.05、lr2e-6、warmup55、cosine、beta0.01。每阶段num_train_epochs=1，总计两遍。原始最大prompt8192，单轮512，epoch1总completion8704。

epoch1保留3_6的完整序列<5120整组排除规则及整批无梯度时不移动Adam/调度器的实现；记录attempted、kept、discarded及实际更新数。epoch2只有图和512输出，去掉所有tool反馈；正式运行需验证最长输入显存，不能以OOM后静默截断替代。warmup55相对四分之一数据可能较长，这里按原设置保留并在结果中记录。

## 目录与执行

`code/stage1/` 和 `code/stage2/` 分进程导入，避免同名episode/reward模块串用。运行时冻结代码和配置到唯一run目录。

```bash
bash scripts/run.sh --source /实际/900_train_data_clean20_badgraph80_degree_sorted --wait
```

`--wait`只在三个前置条件都成立后启动。默认只检查一次；条件不满足即退出。`pending_status.json`显示等待项，`latest_run.json`和run内status.json显示实际运行状态。

远端当前采用后台自动衔接：`3_6/tools/finalize_after_tests.py`等待两组600条策略测试完成后生成分析及`analysis_completed.json`；`scripts/run_pipeline.py --wait`每30秒检查测试完成、分析标记、新900数据的哈希与分层数量，然后等待GPU1释放后自动启动。独占协调锁避免重复队列启动。无需Codex再次手动发起训练；等待进程与测试进程独立运行。

阶段1训练→merge→冻结训练终图→阶段2训练→merge→600条noisy/clean两阶段测试。保存完整命令、日志、数据抽样清单、handoff图、原始rollout、两个adapter、两次merge模型和测试summary。

CPU契约检查：

```bash
python tests/test_stage_contracts.py stage1
python tests/test_stage_contracts.py stage2
python tests/test_stage_contracts.py sampling
```

覆盖四工具后停止、反馈mask、S4+B+H计分且与答案无关、第二阶段不含工具prompt、预测计分和分层随机抽样。未以这些CPU检查冒充真实GPU训练通过；数据到位后启动前还需要检查实际数据和训练运行。
