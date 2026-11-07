# 3_6：可执行阶段需求与原始噪声命中分 H

基于修订后的 3_5 复制代码、配置、启动脚本和依赖，独立输出目录。S、B、T 沿用修订后的 3_5，新增 H：`R=S+B+T+H`。

- S：五阶段结构合法各 0.2。
- B：四工具各 0.25；按阶段输入图与 clean 的差异判断当前工具是否有可执行的正确操作。端点缺失、自环、保护对象等不可执行差异只记录，不产生强制操作需求。
- 允许 add_edge 补回此前 del_edge 删除的边；同阶段重复边操作仍非法，整轮原子回退。
- T：有效预测正确 1、候选内错误 0.1、其他 0。

## H

| 操作 | 正确命中条件 | 每命中分数 |
| --- | --- | ---: |
| del_node | 实际删除原始输入中存在、clean 中不存在的节点 | 0.1 |
| del_edge | 显式删除原始输入中存在、clean 中不存在的边 | 0.1 |
| add_edge | 实际补回原始输入缺失、clean 存在的边 | 0.1 |
| rewrite | 每个允许改写节点满足下述恢复条件 | 0.2 |

H 不截断、不归一化，不新增负分。修复自己造成的错误不获得 H；删除节点自动消失的关联边不另计删边分。仅结构合法且有可见成功执行反馈的阶段获得 H，原子回退阶段 H=0。后续格式错误、触顶或预测失败保留此前完成阶段的 S/B/H。

rewrite：原始文本必须与 clean 不同，且当前成功改写后的文本与 clean 完全一致，或者同时满足 `similarity_after >= 0.85` 和 `similarity_after-similarity_before >= 0.05`。NC 按目标节点，LC 两个端点分别计分。完整恢复 clean 不受提升幅度门槛限制。相似度使用冻结本地 BGE-m3、CPU、mean pooling、归一化 cosine 裁剪到 [0,1]；完整文本编码，超过 8192 token 的模型容量则按容量分块并按内容 token 数加权聚合。与旧 6 号的前 256 token 截断口径不同。阈值是用户认可的配置，尚无正式训练效果结论。

日志包括 hit_reward、每工具 hit_*_count/reward、每个改写节点的 before/after/gain，以及 residual_count/actionable_count/unactionable_count。clean 图和相似度仅用于评分，不返回给模型。

## 使用

```bash
bash /opt/data/private/GraphRL/organized_experiments/3_6/scripts/run_direct_batch4_1epoch.sh all
```

沿用三卡 7B LoRA、ZeRO-3、每卡 batch=2、梯度累积=2、每源 6 候选、四工具各一轮、单轮上限 512 token。输出为 `3_6/runs/3_6_<时间>`。当前仅完成代码修改，正式训练未启动。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /opt/data/private/.venvs/graphrl-3_3-trl-latest/bin/python -m unittest discover -s /opt/data/private/GraphRL/organized_experiments/3_6/tests -v
```

`verification/tests.log` 为 H 边界和 CPU 原生 TRL LoRA 更新测试；`verification/semantic_smoke.log` 为真实冻结 BGE-m3 验证。`reference_tests` 保留来自 3_5 的测试帮助函数和基准用例，正式测试入口为 `tests`；其中共享 CPU 更新用例已更新 H 后的总分预期。
