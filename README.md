# GraphRL：3_5_1 与 3_6 代码、数据和环境

本目录打包两组实验的独立源码、配置、启动脚本、测试，以及启动器指定的同一份训练/测试源数据。两组目前均未启动正式训练；已有 CPU 回归和数据准备验证，不将它们描述为训练完成的模型。

| 实验 | 方法 | 入口 |
|---|---|---|
| 3_5_1 | noisy 图直接输出 analysis+prediction；奖励 S+T | [实验说明](organized_experiments/3_5_1/README.md) |
| 3_6 | 四阶段工具执行；奖励 S+B+T+H | [实验说明](organized_experiments/3_6/README.md) |

## 文件

- `organized_experiments/<实验>/code/`：独立训练、评测、奖励、图工具和 adapter 合并代码。
- `organized_experiments/<实验>/configs/`：ZeRO-2/3 和参考训练配置。
- `organized_experiments/<实验>/scripts/`：原始启动器及跨机器入口 `run_portable.sh`。
- `organized_experiments/<实验>/tests/`：实际 CPU 回归入口；3_6 还保留 `reference_tests/` 帮助模块。
- `data/3_5_1_3_6/`：5495 条训练数据、600 条测试数据的 gzip 文件和哈希清单。
- `requirement.txt`：实际共用 Python 环境的完整包版本导出；两个实验内各有相同副本。
- `environment/`：Python/CUDA/依赖版本、原始 freeze 和本次路径兼容修改记录。
- `upload_manifest.json`：本次发布文件的 SHA256 和大小。

两个实验使用的基座均为**原始 Qwen2.5-7B-Instruct**，不是 DeepSeek 蒸馏模型、SFT 模型或其他实验的 adapter。3_6 另需冻结的 `BAAI/bge-m3`，用于 CPU 文本相似度评分。模型权重自行准备；数据恢复不需要模型权重。

原实验 README 和 `source_manifest.json` 中保留了源机器的历史绝对路径与复制来源。迁移运行请使用本页命令；`a7b_reference.json` 内历史路径仅作参考来源，执行用本实验实际 `configs/zero3_7b.json`。

## 1. 恢复环境

导出环境：Linux x86_64、Python 3.11.18、PyTorch 2.6.0+cu124、TRL 1.17.1、Transformers 5.22.1、PEFT 0.21.2、DeepSpeed 0.19.9、FlashAttention 2.7.4.post1。完整信息见 `environment/environment.json`。

在仓库根目录执行：

```bash
python3.11 -m venv .venv-graphrl
source .venv-graphrl/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
DS_BUILD_OPS=0 python -m pip install --no-build-isolation -r requirement.txt
```

`requirement.txt` 来源于实际安装环境，不是仅按 import 猜测的最小依赖。原 freeze 的 torch/triton/cuSPARSELt 本地 wheel URI 已转为版本约束；FlashAttention 换成安装脚本记录的公开、带 SHA256 的精确 wheel 地址，避免引用另一台机器的缓存路径。该 FlashAttention wheel 仅适用于 CPython 3.11、Linux x86_64、torch 2.6、CUDA 12、CXX11_ABI=False；不同平台应按其兼容环境重建。

本目录保留 `requirements-native.txt` 和 `setup_native_trl.sh` 作为原环境来源；后者仍有源机器工具路径，迁移机器优先按本页创建环境。`pip` 安装不负责安装系统 GPU 驱动。

## 2. 恢复并校验数据

```bash
python tools/restore_data.py
```

将生成：

```text
data/3_5_1_3_6/train_episodes.json  # 5495 条，保留原始 degree-sorted 顺序
data/3_5_1_3_6/test_episodes.json   # 600 条，保留原测试顺序
```

还原的是启动器原本读取的源 JSON 的精确字节，SHA256 分别为：

```text
train: d46d781788e09a7086fc16f86c1deff94d0eb89aaf3a08c19988cf8a8746b778
test:  3c024845c128fc14a02406815a818320f48c3fe6bc5a44f5d9d3299f452f72b6
```

源数据含 noisy/clean 图、标准标签和评分元数据。两个实验各自的 `prepare_data.py` 会重建本协议 prompt、移除旧 oracle completion，并保留评分需要的字段；不要把整个源 JSON 当作模型可见 prompt。解压文件已加入忽略规则，避免重复上传。

压缩后合计约 6.1 MiB，使用普通 Git blob 保存；本数据目录的 `.gitattributes` 明确取消仓库外层的 gzip LFS 过滤，因此 clone 后可直接恢复，不需要下载 LFS 数据对象。

## 3. 指定本机模型位置

```bash
export PYTHON_BIN="$(command -v python)"
export TRAIN_MODEL=/your/models/Qwen2.5-7B-Instruct
export BGE_M3_MODEL=/your/models/bge-m3
export CUDA_VISIBLE_DEVICES=0,1,2
```

模型应已完整下载，因为现有启动器设置离线加载。`TRAIN_MODEL` 同时供训练的基座检查与 CPU 测试使用；`BGE_M3_MODEL` 供 3_6 的冻结相似度编码器使用。路径兼容修改保留原默认值，未调整奖励、数据顺序、学习率或训练预算。

## 4. 仅准备数据与检查 prompt

`run_portable.sh` 会先解压校验本仓库携带的数据，再调用实验原始入口。

```bash
RUN_DIR="$PWD/organized_experiments/3_5_1/runs/remote_run" \
  bash organized_experiments/3_5_1/scripts/run_portable.sh prepare

RUN_DIR="$PWD/organized_experiments/3_6/runs/remote_run" \
  bash organized_experiments/3_6/scripts/run_portable.sh prepare
```

此过程不启动正式 GPU 训练。生成的各自 `data/train_episodes.json`、`data/test_episodes.json` 才含对应实验重建的 prompt；两个实验的 prompt 与 reward 协议不同，不能共用某个实验已经准备后的数据来替代另一个实验的准备过程。

## 5. 执行训练、merge、测试

若一次执行全部流程，请选择尚未存在的运行目录：

```bash
RUN_DIR="$PWD/organized_experiments/3_5_1/runs/full_run" \
  bash organized_experiments/3_5_1/scripts/run_portable.sh all

RUN_DIR="$PWD/organized_experiments/3_6/runs/full_run" \
  bash organized_experiments/3_6/scripts/run_portable.sh all
```

不要在同一组三卡上同时启动两组训练。如果已经执行过 `prepare`，用相同 `RUN_DIR` 依次执行 `train`、`merge`、`test`，不要再对同一目录执行 `all` 或 `prepare`。原脚本会拒绝覆盖运行目录。

两个实验默认三卡 7B LoRA、每卡 batch=2、梯度累积=2、每源 6 候选、1 epoch。两组的不同协议和 3_6 的 H 细则见实验 README。此处只导出实验定义，没有上传模型 checkpoint。

## 6. CPU 回归验证

安装环境、下载基座 tokenizer 并指定 `TRAIN_MODEL` 后：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m unittest discover -s organized_experiments/3_5_1/tests -v

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m unittest discover -s organized_experiments/3_6/tests -v
```

测试使用 CPU 小模型验证原生 TRL 更新，不加载完整 7B 模型。3_6 的语义奖励若遇到非完全一致的真实文本恢复，需要本地 BGE-m3；相关单元测试以可控相似度覆盖规则边界。
