# 5,000条同类候选攻击混合数据：下载与恢复

本数据集从 `train_data_15785_oneclass` 中用种子42抽样，每个原始sample_id只选一个版本；共3,500条NC、1,500条LC，无LP。

| 任务 | clean | budget1 | budget2 | budget3 | budget4 | budget5 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NC | 700 | 560 | 560 | 560 | 560 | 560 | 3,500 |
| LC | 300 | 240 | 240 | 240 | 240 | 240 | 1,500 |
| 合计 | 1,000 | 800 | 800 | 800 | 800 | 800 | 5,000 |

全量20% clean、80%攻击；budget1–5各占攻击部分20%。攻击版本均来自源记录完整完成5轮的样本。按实际episode的目标度升序：NC取目标节点度，LC先屏蔽查询边再取两端平均度；按无向去重邻居计数，忽略自环及关系类型。同度保留种子42的混合顺序。目标度范围0–15。

## 恢复数据

全部原始文件通过两个tar.zst分卷随仓库提供，不使用Git LFS。压缩包56,419,402字节；解压约399 MiB。需要Python 3.9+和系统 `zstd` 命令，恢复已有数据不需要模型或API。

在仓库根目录运行：

```bash
python data/train_data_oneclass_5000_degree_sorted/restore_data.py
```

脚本先验证分卷及合并压缩包的SHA256，再恢复至本目录。仅校验下载完整性：

```bash
python data/train_data_oneclass_5000_degree_sorted/restore_data.py --verify-only
```

恢复后的主要文件：

| 文件 | 内容 |
| --- | --- |
| `train.parquet`、`train.jsonl` | 已排序的5,000条选定版本 |
| `clean.parquet` | 同一组5,000条的clean参考，不是额外训练样本 |
| `train_grpo_episodes.json` | 同序旧格式episode，包含clean/noisy图及任务字段 |
| `groups/`、`group_indices.json` | 六个互斥组及稳定原始sample_id |
| `attack_metadata.jsonl` | 对应版本的攻击元信息 |
| `target_degree_order.csv` | 排序位置、ID、目标端点及degree |
| `manifest.json`、`independent_validation.json` | 配额、来源、哈希及独立验证 |

本目录没有 `train_grpo_episodes_redesign.json`。旧训练启动器如果依赖该文件，需要按其数据准备流程转换；这里提供的是选定源版本与旧格式episode。原始prompt和图字段在Parquet中保留，degree按屏蔽LC查询边后的episode计算。

## 重现抽样和排序

先恢复仓库中的完整源池 `data/train_data_15785_oneclass`，安装 `pyarrow`，然后执行：

```bash
python tools/build_oneclass_5000_degree_sorted.py \
  --source data/train_data_15785_oneclass \
  --output /your/new_dataset_directory \
  --seed 42
```

输出目录必须尚不存在。构造脚本和所需的旧episode解析辅助代码均随仓库提供。source_index使用稳定sample_id；source_row_index另存源clean文件行位置。训练、clean参考、episode和metadata保持相同ID顺序。
