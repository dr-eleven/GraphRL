# GraphRL oneclass checkpoint-32 双测试集评测

同一个 oneclass 训练模型在 oneclass 攻击集答对 **182/600（30.33%）**，在 notoneclass 攻击集答对 **179/600（29.83%）**，oneclass 减 notoneclass 为 **+0.50 个百分点**。

使用第 32 步 LoRA 与训练基座 Qwen2.5-7B-Instruct 合并后的模型。采用仓库的 noisy_policy 协议，BF16、SDPA、贪心解码、评测批量 8，每个输入生成一次，输出上限 512 token。两套测试同源配对，各测完整 600 条，共 1,200 次预测。每套覆盖 450 条 NC、150 条 LC。两套测试均无 LP 样本。

## 配对结果

| 两种攻击下的正确性 | 样本对数 |
|---|---:|
| 两边都正确 | 115 |
| 仅 oneclass 正确 | 67 |
| 仅 notoneclass 正确 | 64 |
| 两边都错 | 354 |

## 分组正确率

| 任务 | 样本对 | oneclass 正确率 | notoneclass 正确率 | 差值（百分点） |
|---|---:|---:|---:|---:|
| LC | 150 | 46/150 (30.67%) | 40/150 (26.67%) | +4.00 |
| NC | 450 | 136/450 (30.22%) | 139/450 (30.89%) | -0.67 |

| 数据集 / 任务 | 样本对 | oneclass 正确率 | notoneclass 正确率 | 差值（百分点） |
|---|---:|---:|---:|---:|
| cora_node/NC | 150 | 55/150 (36.67%) | 57/150 (38.00%) | -1.33 |
| fb15k237/LC | 150 | 46/150 (30.67%) | 40/150 (26.67%) | +4.00 |
| products/NC | 150 | 20/150 (13.33%) | 22/150 (14.67%) | -1.33 |
| wikics/NC | 150 | 61/150 (40.67%) | 60/150 (40.00%) | +0.67 |

| 攻击预算 | 样本对 | oneclass 正确率 | notoneclass 正确率 | 差值（百分点） |
|---|---:|---:|---:|---:|
| 1 | 120 | 46/120 (38.33%) | 40/120 (33.33%) | +5.00 |
| 2 | 120 | 37/120 (30.83%) | 40/120 (33.33%) | -2.50 |
| 3 | 120 | 39/120 (32.50%) | 40/120 (33.33%) | -0.83 |
| 4 | 120 | 30/120 (25.00%) | 32/120 (26.67%) | -1.67 |
| 5 | 120 | 30/120 (25.00%) | 27/120 (22.50%) | +2.50 |

## 差异分析

- 131/600 对样本（21.83%）在两种攻击下的正确性发生变化。其中 67 条仅在 oneclass 正确，64 条仅在 notoneclass 正确。
- 分数据集差异最大的是 fb15k237/LC，正负方向按 oneclass 减 notoneclass 计算，相差 +4.00 个百分点。按攻击预算分组，预算 1 的差异最大，为 +5.00 个百分点。
- 候选内选错在两套测试上的计数分别为 355 和 365，格式无效分别为 2 和 3。这些计数反映同一个模型面对两种攻击图时的表现。

## 输出质量

| 指标 | oneclass | notoneclass |
|---|---:|---:|
| 格式有效 | 598/600 (99.67%) | 597/600 (99.50%) |
| 格式无效（含截断） | 2/600 (0.33%) | 3/600 (0.50%) |
| 格式有效但标签超出候选 | 61/600 (10.17%) | 53/600 (8.83%) |
| 格式有效但选错候选 | 355/600 (59.17%) | 365/600 (60.83%) |
| 到达生成上限且未输出 EOS | 0/600 (0.00%) | 0/600 (0.00%) |
| 平均生成 token 数 | 105.05 | 101.88 |
| 平均奖励 S+T | 1.3592 | 1.3542 |

## 失败案例

每类最多展示两个样本，按测试顺序取例。格式无效和截断时保留实际输出摘录。

### 仅 oneclass 正确

- 样本 8 · wikics/NC · 预算 4 · 标准答案 **computer architecture**
  - oneclass（正确） computer architecture
  - notoneclass（候选内错误） computer security
- 样本 10 · fb15k237/LC · 预算 4 · 标准答案 **Award Nominees**
  - oneclass（正确） Award Nominees
  - notoneclass（候选内错误） Film Release Region and Date

### 仅 notoneclass 正确

- 样本 3 · products/NC · 预算 1 · 标准答案 **Health &amp; Personal Care**
  - oneclass（候选外标签） Pain Relief
  - notoneclass（正确） Health &amp; Personal Care
- 样本 5 · fb15k237/LC · 预算 2 · 标准答案 **Film Genres**
  - oneclass（候选外标签） None
  - notoneclass（正确） Film Genres

### 至少一边格式无效且两边答错

- 样本 27 · fb15k237/LC · 预算 5 · 标准答案 **Person&#x27;s Profession**
  - oneclass（格式无效） {&quot;analysis&quot;:&quot;Node26 is a British actress, and node10 is a person who performs in films, TV, theater, or radio. The connection between node10 and multiple nodes relat…
  - notoneclass（候选外标签） None
- 样本 70 · fb15k237/LC · 预算 2 · 标准答案 **Person&#x27;s Profession**
  - oneclass（候选外标签） Musician and actress
  - notoneclass（格式无效） {&quot;analysis&quot;:&quot;Mike Mills is associated with R.E.M., a band, and is connected to &#x27;musician&#x27; through &#x27;Gustavo Santaolalla&#x27; and &#x27;2009 HBO project filmed in a city&#x27;. This…

### 至少一边输出候选外标签且两边答错

- 样本 21 · products/NC · 预算 5 · 标准答案 **Tools &amp; Home Improvement**
  - oneclass（候选外标签） Music &amp; Audio
  - notoneclass（候选内错误） Video Games
- 样本 30 · cora_node/NC · 预算 5 · 标准答案 **Probabilistic Methods**
  - oneclass（候选内错误） Reinforcement Learning
  - notoneclass（候选外标签） Model-Based Cluster Analysis

### 两边均选错候选

- 样本 0 · fb15k237/LC · 预算 3 · 标准答案 **Transportation to a Travel Destination**
  - oneclass（候选内错误） Major Field of Study for Graduates
  - notoneclass（候选内错误） Award Ceremony for Category Winners
- 样本 2 · cora_node/NC · 预算 4 · 标准答案 **Case-Based**
  - oneclass（候选内错误） Genetic Algorithms
  - notoneclass（候选内错误） Probabilistic Methods

## 文件

- [结构化汇总](comparison.json)
- [600 条逐样本对照](paired_samples.jsonl)
- [oneclass 逐样本原始输出](oneclass/predictions.jsonl)
- [notoneclass 逐样本原始输出](notoneclass/predictions.jsonl)
