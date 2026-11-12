# 900条训练数据

随机种子42；原始训练集按数据集+任务类型分层，各组100条，组内20条clean、budget1–5各16条。先抽源样本，再指定budget并读取原始攻击版本；不同budget不会重复同一个源样本。

| 数据集/任务 | clean | b1 | b2 | b3 | b4 | b5 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arxiv/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| children/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| citeseer/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| computer/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| instagram/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| photo/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| pubmed/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| sports/NC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| wn18rr/LC | 20 | 16 | 16 | 16 | 16 | 16 | 100 |
| 合计 | 180 | 144 | 144 | 144 | 144 | 144 | 900 |

按noisy图NC目标节点度数、LC两个查询节点度数均值升序；同度按source_index升序。原始budget4有1条未实际改变图，保留并披露；分配比例按数据来源计算。

`train.parquet`为原始任务prompt格式；`train_grpo_episodes_redesign.json`为训练episode；`manifest.json`记录源文件SHA256、分层source_index、分配与排序。复现脚本位于 `organized_experiments/3_5_2/code/build_900_data.py`。
