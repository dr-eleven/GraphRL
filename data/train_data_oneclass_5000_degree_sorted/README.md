# 5,000条同类候选攻击混合训练数据

来源：`../train_data_15785_oneclass`；随机种子42；NC 3,500条、LC 1,500条，无LP。每个原始sample_id只选一个版本。

| 任务 | clean | budget1 | budget2 | budget3 | budget4 | budget5 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NC | 700 | 560 | 560 | 560 | 560 | 560 | 3,500 |
| LC | 300 | 240 | 240 | 240 | 240 | 240 | 1,500 |
| 合计 | 1,000 | 800 | 800 | 800 | 800 | 800 | 5,000 |

按episode实际输入的noisy_graph计算目标度并升序排列：NC取目标节点度，LC取两个查询端点度数的平均值。邻居无向、去重，忽略自环和关系类型；LC查询边先移除。同度保留种子42生成的混合顺序。

- `train.parquet`、`train.jsonl`：选定clean或budget版本的5,000条记录。
- `clean.parquet`：相同5,000个sample_id对应的攻击前clean记录，按训练数据顺序排列；它是奖励参考，不是额外训练样本。
- `train_grpo_episodes.json`：旧格式episode，候选标签取结构化label_list。
- `groups/`：六个互斥组的Parquet，各自维持目标度排序。
- `attack_metadata.jsonl`：按训练顺序保存对应版本的攻击元信息。
- `target_degree_order.csv`：排序位置、原始ID、目标端点及其度数。
- `group_indices.json`：组内原始sample_id；source_index同样使用稳定sample_id，不是新文件行号。
- `manifest.json`：来源、抽样配额、ID、排序规则、完整性验证和文件哈希。

所有选中的攻击样本均来自完整完成5轮的源记录，但每条训练样本只使用分配到的budget版本。未重新调用模型或生成攻击。原始prompt和图字段在train.parquet中保留；episode额外移除LC查询边，目标度使用episode口径。
