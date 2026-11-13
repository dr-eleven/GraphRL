# 2000_notoneclass_sorted

随机种子42；沿用900条数据的分层抽样方法，在两类攻击完整结果共有的15364个sample_id中，19种数据集+任务组合各100条，WN18RR/LC额外100条。每个源ID只使用一个版本。

| 数据集/任务 | clean | b1 | b2 | b3 | b4 | b5 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arxiv/LP | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| arxiv/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| children/LP | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| children/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| citeseer/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| computer/LP | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| computer/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| history/LP | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| history/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| instagram/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| photo/LP | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| photo/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| pubmed/LP | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| pubmed/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| reddit/LP | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| reddit/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| sports/LP | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| sports/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| wn18rr/LC | 40 | 32 | 32 | 32 | 32 | 32 | 200 |
| 合计 | 400 | 320 | 320 | 320 | 320 | 320 | 2000 |

NC 1000条、LC 200条、LP 800条。两套数据的sample_id、任务、标签、目标节点及clean/budget分配完全相同，clean的400条图文本也相同；攻击版本来自各自来源目录。

以episode的noisy_graph计算无向、去重邻居度，忽略自环及关系类型：NC取目标节点度，LC/LP移除查询边后取两端度数平均。按degree升序，同degree按source_index升序。两套图的degree可能不同，排序后行号不必相同；按sample_id/budget配对，见pair_alignment.csv。

train.parquet、train.jsonl为选中的原始任务格式；train_grpo_episodes_redesign.json沿用900条版本的结构化episode核心字段，并保留LP任务类型及二元候选标签。该episode不含预生成prompt或oracle，由训练协议的数据准备代码生成。clean.parquet是同2000个ID的奖励参考，不是另外的训练样本。原始图/任务prompt在Parquet中保留，查询边屏蔽作用于episode。

manifest.json记录ID、分组、哈希、来源和验证；target_degree_order.csv记录各样本排序度；pair_alignment.csv给出两套数据的行号与degree对应关系。构造代码：tools/build_paired_2000_sorted.py。
