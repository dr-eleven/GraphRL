# 3_5_1_2：支持NC、LC、LP的单轮直接预测

从3_5_1复制独立代码、配置、依赖和启动脚本。保留原始Qwen2.5-7B-Instruct初始化、LoRA、原生TRL GRPO、三卡默认配置、degree顺序、单轮生成和 `R=S+T`；新增LP任务语义、数据约束与二分类评测。原3_5_1文件保持不变，复制来源见 `source_manifest.json`。

## Prompt与输出

三类任务共用一套模板骨架和输出协议，加入任务专用说明：

| 任务 | 预测目标 | 任务约束 |
| --- | --- | --- |
| NC | 单个目标节点的类别 | 目标节点必须存在，使用提供的候选标签 |
| LC | 两个查询端点的关系类别 | 屏蔽查询边，按查询端点预测关系 |
| LP | 原始底层图中指定边是否存在 | 查询边对正负样本都屏蔽；不能仅因观察图缺少该边判为负例 |

只输出 `{"analysis":"非空分析","prediction":"一个完整候选标签"}`，两个字段按上述顺序排列。预测分析与标签均为非空字符串；额外字段、额外消息、工具调用、代码围栏、重复JSON键以及未以EOS结束的触顶输出均不给格式奖励。

模型输入仅使用任务类型、由结构化节点ID重建的目标、候选标签和当前图。存储的旧prompt、oracle、clean_graph、ground_truth、攻击位置及budget不会拼入模型prompt。Missing等占位文本保留为缺失证据，要求模型不编造节点事实。

## 新增的必要约束

1. **目标边泄漏防护**：LC/LP的clean/noisy视图在加载时屏蔽查询端点间的边，prompt函数对显式传入的图也执行相同处理。原输入数据文件和原始样本对象不修改；ground_truth不会根据屏蔽后的边重新计算。
2. **端点合法性**：查询端点、边端点必须存在；LP要求两个不同的端点。重复归一化节点ID、矛盾的目标字段和不支持的任务类型会报错。
3. **二元标签约束**：LP必须恰有两个相反极性的候选标签，正确答案必须在候选内。支持Yes/No开头的完整标签以及True/False、1/0等常见表示；自定义标签可通过 `lp_label_polarity` 显式指定，例如 `{"connected":true,"unconnected":false}`。
4. **完整标签评分**：沿用3_5_1的大小写、空白和首尾标点归一化后做完整标签比较。若候选是 `Yes, they have citation relationships`，只输出 `Yes` 不得预测分；不会用字符串包含yes/no来判断正误。
5. **方向语义**：当前数据默认无向查询，屏蔽两个方向及同端点间的所有关系。若样本显式设置 `query_directed=true`，仅屏蔽给定方向；最终prompt不新增方向字段。不会仅凭数据集名称推断有向性。
6. **评测覆盖与不平衡监测**：preflight报告训练/测试任务分布与LP正负数；评测新增按任务、数据集/任务分组及LP混淆矩阵、balanced accuracy、macro-F1。无效预测单列，不能算负类；缺少一个真实类别时不报告有误导性的双类平均指标。

## 奖励

保持3_5_1的分值表：

| 情况 | S | T | 总分 |
| --- | ---: | ---: | ---: |
| 合法JSON且预测正确 | 1 | 1 | 2 |
| 合法JSON、候选内但错误 | 1 | 0.1 | 1.1 |
| 合法JSON、非空候选外标签 | 1 | 0 | 1 |
| 格式错误或未正常结束 | 0 | 0 | 0 |

没有新增工具、图编辑动作或奖励权重；三种任务均按相同表评分。输入标签分布保持原样，不自动重采样。

## 当前数据与测试覆盖

默认训练源：`GraphRL/2000_oneclass_sorted/train_grpo_episodes_redesign.json`，NC 1000、LC 200、LP 800；LP正例407、负例393。默认测试源：`GraphRL/data/test_data_600_oneclass/test_grpo_episodes.json`，NC 450、LC 150，**没有LP**。

所以默认600条测试只能给出NC/LC评测，不能验证LP模型准确率。需要评测LP时，应提供含LP的独立测试episodes并设置 `TEST_EPISODES`、`EXPECTED_TEST_COUNT`；数据准备支持 `--require-test-task-coverage`，可要求测试覆盖训练涉及的全部任务，缺失时在加载模型前报错。启动脚本可设置 `REQUIRE_TEST_TASK_COVERAGE=1` 启用此强制检查。常规preflight会显式记录 `missing_test_tasks=["LP"]`。

## 运行

只准备数据（默认本地Python环境和基座目录与3_5_1相同）：

```bash
RUN_DIR=/your/runs/3_5_1_2_oneclass \
  bash organized_experiments/3_5_1_2/scripts/run_direct_batch4_1epoch.sh prepare
```

完整prepare → train → merge → test：

```bash
RUN_DIR=/your/runs/3_5_1_2_oneclass \
  bash organized_experiments/3_5_1_2/scripts/run_direct_batch4_1epoch.sh all
```

输出目录不可覆盖。支持原有 `PYTHON_BIN`、`TRAIN_MODEL`、`CUDA_VISIBLE_DEVICES`、`TRAIN_EPISODES`、`TEST_EPISODES`、`PREDICTION_TOKENS` 等环境变量。使用not-one-class配对数据：

```bash
TRAIN_EPISODES=/opt/data/private/GraphRL/2000_notoneclass_sorted/train_grpo_episodes_redesign.json \
TEST_EPISODES=/opt/data/private/GraphRL/data/test_data_600_notoneclass/test_grpo_episodes.json \
RUN_DIR=/your/runs/3_5_1_2_notoneclass \
  bash organized_experiments/3_5_1_2/scripts/run_direct_batch4_1epoch.sh prepare
```

四种评估条件仍为noisy_policy、clean_policy、noisy_baseline、clean_baseline，并使用相同任务prompt和评分规则。

## 验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /opt/data/private/.venvs/graphrl-3_3-trl-latest/bin/python \
  -m unittest discover -s organized_experiments/3_5_1_2/tests -v
```

覆盖NC原有奖励、格式、截断和隐藏信息隔离，LP正负完整标签、双向/有向查询边屏蔽、非法端点与候选、缺失文本、评测无效预测，以及CPU小模型的真实TRL GRPO LoRA更新和未替换生成叶子的预算检查。只用小模型验证训练链路；没有启动完整7B训练。

`verification/prepared_oneclass`、`verification/prepared_notoneclass`记录两套实际数据准备与token长度审计；`verification/tests.log`记录测试结果。原数据文件的degree顺序保持不变。


迁移执行请以 [readme.md](readme.md) 为准；最终prompt以当前code/episode.py为准，不含Query direction字段。
