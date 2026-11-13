# 600条one-class BadGraph测试数据

全部600条源样本均完整完成5轮攻击。clean/保留原始clean数据；badgraph/budget1至budget5各有600条完整对应的攻击版本与元信息。

test.parquet、test_grpo_episodes.json是沿用旧测试集ID、顺序及具体budget分配组成的600条混合测试数据：每个budget含90条NC和30条LC，共450条NC、150条LC。group_indices.json、groups/、episodes_by_group/保存对应分组；manifest.json记录来源与核验信息。
