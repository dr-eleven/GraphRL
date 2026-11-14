# 1000_oneclass_sorted

从`2000_oneclass_sorted`按“数据集＋任务类型＋clean/budget档位”逐格随机抽取50%，不放回，seed=42。两套使用相同source_index和原budget分配，节点、连接、文本、标签及目标不作改动。

“10% clean＋40% attack”以原2000条为分母：clean200条、attack800条，共1000条。新数据内部仍为20% clean、80% attack；budget1–5各160条（各占新数据16%）。

| 数据集/任务 | clean | b1 | b2 | b3 | b4 | b5 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arxiv/LP | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| arxiv/NC | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| children/LP | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| children/NC | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| citeseer/NC | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| computer/LP | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| computer/NC | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| history/LP | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| history/NC | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| instagram/NC | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| photo/LP | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| photo/NC | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| pubmed/LP | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| pubmed/NC | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| reddit/LP | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| reddit/NC | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| sports/LP | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| sports/NC | 10 | 8 | 8 | 8 | 8 | 8 | 50 |
| wn18rr/LC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| 合计 | 200 | 160 | 160 | 160 | 160 | 160 | 1000 |

NC500条、LC100条、LP400条。19种数据集＋任务组合中，18种各50条，wn18rr/LC为100条。每个普通组合clean10条、每个budget8条；LC为clean20条、每个budget16条。

分别继承各自父目录degree升序、同degree按source_index升序。NC按目标节点无向去重邻居度；LC/LP屏蔽查询边后取两端degree均值。两套图的degree不同，因此按source_index/sample_id和budget配对，排序后行号不保证一致。

训练入口为`train_grpo_episodes_redesign.json`，保留NC/LC/LP原任务。`train.parquet`和`train.jsonl`为同1000条原任务记录；`clean.parquet`为这些ID的1000条clean参考，不是额外训练数据。`group_indices.json`记录分组；`target_degree_order.csv`记录新位置和原2000条位置；`pair_alignment.csv`记录两套对应关系；`manifest.json`记录抽样、来源哈希及验证。复现代码：`tools/build_paired_1000_sorted.py`。
