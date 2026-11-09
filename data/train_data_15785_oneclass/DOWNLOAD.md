# 下载和恢复完整5轮同类攻击数据

停止补跑后共有 **15,364条完整样本**：NC 7,710、LC 2,490、LP 5,164。目录名中的15785是计划样本池大小，未完整完成的421条没有混入数据。

仓库内的 `archive/` 保存完整数据的 tar.zst 分卷，不使用 Git LFS。原始文件约15.25 GiB；压缩分卷合计805,372,261字节。每个分卷不超过48 MiB。下载整个仓库后，无需攻击模型或API即可恢复数据。

在仓库根目录运行（Python 3.9+，系统需要 `zstd` 命令）：

```bash
python data/train_data_15785_oneclass/restore_data.py
```

脚本会验证每个分卷和完整压缩包的SHA256，再恢复到 `data/train_data_15785_oneclass/`。需要约16 GiB可用空间，另需约0.8 GiB临时空间。

只校验下载文件而不解压：

```bash
python data/train_data_15785_oneclass/restore_data.py --verify-only
```

也可恢复到其他目录：

```bash
python data/train_data_15785_oneclass/restore_data.py --output-dir /your/data
```

恢复后包含：

| 文件 | 内容 | 数量 |
| --- | --- | ---: |
| `train_rl.parquet`、`train_rl.jsonl` | 攻击前clean子图 | 各15,364条 |
| `budget1`至`budget5`下的Parquet | 累计1至5轮攻击后的数据 | 每份15,364条 |
| 对应的`*_metadata.jsonl` | 攻击过程及元信息 | 每份15,364条 |
| `sample_results/{sample_id}.json` | 单样本完整5轮结果 | 15,364份 |
| `manifest.json`、`provenance.jsonl`、`stop_summary.json` | ID、来源、数量、停止记录 | — |

clean与五份budget的sample_id集合相同且无重复。每条收入的样本有5个成功轮次、5次文本修改和5条新增边。来源包括此前500条实验、首批Luna完整95条、原Luna批次完整14,756条，以及hezubus GPT-6.1-sol补跑完整13条。攻击任务已停止。

节点候选规则和任务语义见 [README.md](README.md)。完整完成攻击不等于受害模型一定预测错误。
