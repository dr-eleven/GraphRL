# 1000条oneclass训练与两套600条测试的部署

本次发布到`dr-eleven/GraphRL`，完整包含三个同名数据目录和`organized_experiments/5_1`代码、配置、启动文件、依赖与验证记录。数据文件按原始字节上传，不包含模型权重、访问令牌或SSH凭据。

训练读取`1000_oneclass_sorted/train_grpo_episodes_redesign.json`全部1000条；测试分别读取`600_test_data_oneclass/test_grpo_episodes.json`与`600_test_data_notoneclass/test_grpo_episodes.json`，每套完整600条。两套测试的source_index、clean参考、标签与预算分配配对，但攻击输入不同。测试集中各有120条分配为clean的记录，仍作为测试集原始成员参加`restored`条件评估；不另跑clean_policy或基座。

目标服务器连接信息须先核实。GPU编号默认1，启动前检查空闲并取得共享锁，不停止其他任务。Python、Qwen2.5-7B-Instruct原始基座必须与已验证环境兼容。当前环境参考版本见`requirements-native.txt`；已有兼容环境时直接复用，不重装。

## 从GitHub下载

将`COMMIT`替换为本次发布的完整40位commit SHA，`DESTINATION`替换为实际仓库路径。下载工具逐文件校验SHA256；已存在且相同的文件直接验证，不同文件先备份，不操作已有训练目录。

```bash
COMMIT=本次完整提交SHA
DESTINATION="$HOME/GraphRL"
curl -fL "https://raw.githubusercontent.com/dr-eleven/GraphRL/$COMMIT/organized_experiments/5_1/tools/download_release.py" -o /tmp/download_graphrl_5_1.py
python3 /tmp/download_graphrl_5_1.py --commit "$COMMIT" --destination "$DESTINATION"
```

如果仓库为私有，先在目标机器通过用户授权的GitHub凭据下载；不能将认证信息写入代码或发布清单。

## 启动完整任务

```bash
cd "$HOME/GraphRL/organized_experiments/5_1"
export PYTHON_BIN="$HOME/.venvs/graphrl-3_5_1-3_6/bin/python"
export TRAIN_MODEL="$HOME/models/Qwen2.5-7B-Instruct"
nohup "$PYTHON_BIN" -u scripts/run_managed.py --gpu 1 --wait > setup_1000.log 2>&1 < /dev/null &
```

该进程独立执行prepare→单epoch训练→adapter与原始基座合并→oneclass/notoneclass各600条测试。`latest_run.json`记录run路径和阶段；`setup_logs/`保留各阶段启动日志；run中的`train.log`、`grpo/checkpoint-*`、`merged_model`、`test/<数据集>/restored/predictions.jsonl`与`summary.json`保留实际结果，最后汇总为`test_results.json`。只有训练完成且两套测试均覆盖600唯一源，才标记completed。

硬件与优化参数参考3_6_1：ZeRO-2无卸载、batch12/累积1、G6、BF16、LoRA32/32、lr2e-6、warmup55、beta0.01、梯度检查点与Liger。5_1仍保留原始10轮自由调度和原有生成预算；不转为3_6_1的两阶段训练，不丢弃长轨迹。1000条/每更新2个源，对应500次更新。需要先通过真实目标GPU验证batch12的长轨迹显存，CPU测试不代表GPU可运行。

## 已有验证的解释

`verification/jhh_1000_data_validation.json`校验本次1000/600/600数量、任务与预算分布、原始文件SHA256及两套测试配对。37项控制器/运行时/真实CPU tiny-Qwen GRPO集成测试通过，另有32个实际源数据控制器检查；这些都不是7B测试准确率。部署后应重新prepare，读取本次run的`data/preflight.json`，不使用历史2000条数据的预检查冒充本次输入。
