# 600条oneclass：20% clean / 80% BadGraph

保留3_5对应的600个源样本及顺序。两个版本的source_index、标签、clean图、clean选择和budget分配一一对应；只有480条BadGraph输入图分别取oneclass/notoneclass原攻击版本。原始数据不修改。

| 数据 | clean | budget1 | budget2 | budget3 | budget4 | budget5 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cora_node/NC | 30 | 20 | 26 | 28 | 26 | 20 | 150 |
| fb15k237/LC | 30 | 24 | 24 | 24 | 24 | 24 | 150 |
| products/NC | 30 | 28 | 20 | 21 | 24 | 27 | 150 |
| wikics/NC | 30 | 24 | 26 | 23 | 22 | 25 | 150 |
| 总计 | 120 | 96 | 96 | 96 | 96 | 96 | 600 |

`test_episodes.json`与`test_grpo_episodes.json`字节一致，均为可读取的episode列表。旧prompt与oracle已移除，评测代码应按自身协议从当前图重建prompt。clean样本的攻击记录清空、budget设0；`original_selected_budget`只记录原始分配用于溯源。

seed42：每个数据集/任务/原budget分组选择6条为clean；其余维持原budget。每个数据集150条中clean30条。`group_indices.json`是稳定source_index，不是数组位置。`manifest.json`记录配对选择、数量和SHA256。
