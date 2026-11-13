本目录完整原文件以archive/中的zstd分卷保存，文件清单与SHA256见archive_manifest.json。

在仓库根目录执行：

```bash
python3 tools/restore_3_5_1_2_data.py
python3 tools/restore_3_5_1_2_data.py --verify-only
```

恢复后使用本目录中的数据文件。实验执行说明见[3_5_1_2 readme](../../organized_experiments/3_5_1_2/readme.md)。
