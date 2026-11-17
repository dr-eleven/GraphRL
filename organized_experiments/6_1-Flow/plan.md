> 发布说明：本文保留原服务器的历史设计、实例和参数演变。2026-11-17 的 GitHub 发布启动配置为 Qwen2.5-7B、8×H200、每次生成上限 8192；当前执行参数请以 [README.md](README.md) 和 configs 为准。

# 6_1-Flow：使用 AgentFlow 完成 GraphRL 任务

## 目标与边界

基本保留本地 `/opt/data/private/AgentFlow` 的实际实现：Solver 调用固定模型分析问题，训练中的 Planner 生成下一步 context/sub_goal/tool_name，Executor 生成工具命令并执行，Memory记录结果，Verifier决定继续/停止，固定 Generator 生成最终答案。在线训练沿用 AgentFlow 的 rollout tracing、逐Planner调用转移、终局奖励传播、按源问题分组的 Flow-GRPO 和 VERL actor/reference 更新。原 AgentFlow 与4_1保持独立，本目录保存上游快照、适配代码、配置、验证与运行产物。

## 用户指定的训练设置

- 基础模型：本地原始 Qwen2.5-7B-Instruct；所有模型和LoRA参数均BF16。
- 训练方式：Planner LoRA，r/alpha=32/32；固定模块使用冻结原始7B，无外部付费API。
- 每源问题采样6条完整流程；每次训练批次2条源问题，即12条流程。
- 每2条源问题只做一次actor梯度更新，累积该批全部有效Planner转移，不能将“2条数据”误写成2条转移或把每个Planner步骤单独更新。
- 使用4_1同一1000条训练数据和两个独立600条测试集；1 epoch，预计500批。训练→合并LoRA→两测试集分别restored评估。

## 保留与适配映射

| 内容 | 实施方案 |
| --- | --- |
| Solver、Planner、Executor、Verifier、Memory | 保存AgentFlow源码，优先复用原类与方法，只扩展图任务必要接口 |
| Planner选择动作 | 可选工具恰为del_node、del_edge、add_edge、rewrite；自由顺序、可重复；不引入4_1顺序状态机 |
| Executor命令 | 保留生成tool.execute(...)的接口；图动作参数和图校验使用GraphRL；限制为可解析的字面量工具调用，拒绝任意Python副作用 |
| 图状态 | 每个候选流程独立；只向模型提供公开任务、标签、约束和当前图，隐藏clean/ground_truth不进入提示 |
| Memory | 保留各步工具、子目标、命令和反馈；反馈包含真实更新图或拒绝原因 |
| 固定模块 | query analysis、Executor、Verifier、Generator使用冻结Qwen2.5-7B，确定性推理；只Planner调用经过训练代理并被计入loss |
| 奖励 | 用GraphRL确定性候选标签匹配替换AgentFlow通用LLM judge；最终预测正确1，否则0；基础设施异常与错误答案分离 |
| 迭代次数与温度 | 以AgentFlow训练默认3次工具迭代、Planner温度0.7、评估0为起点；不继承4_1的总10次模型生成限制 |
| token | 单次输出2048，Planner输入容量18432；保持原框架分模块调用，显式检测容量；禁止把截断后的输出算成功 |
| GRPO | 沿用AgentFlow逐Planner调用转移与源问题分组的优势；6条流程的终局奖励传播至各自Planner转移，不替换成4_1的整轨迹TRL训练 |
| 参数与资源 | 其余优先保留AgentFlow：lr=1e-6，KL=0.001，clip低0.2/高0.3；按4090显存适配microbatch=1、卸载和本地服务并发 |
| 梯度批次兼容 | 保留Flow-GRPO优势后，将当前批全部转移累计为一次actor更新；分布式对齐只添加零loss权重padding，不能随机丢弃有效转移 |
| 评估 | 用合并后的Planner与冻结原始辅助模型运行同一Solver；两个测试集独立输出预测、轨迹、准确率、工具错误与图质量 |

## 实施顺序

1. 完成源码与依赖审计，记录上游commit和文件哈希，保存原实现副本。
2. 实现数据转换、候选隔离的图状态和四工具；测试保护约束、原子拒绝、多次操作和无隐藏信息泄漏。
3. 接入原Solver及模块：使用本地模型适配器、真实Planner tracing，固定辅助模块不入策略loss；保留原框架的停止/最终生成流程。
4. 接入原AgentFlow/VERL训练管道，设置LoRA与6采样，确保每批2源、全部转移一次梯度更新；验证分组、变长流程、padding权重和模型合并。
5. 提供prepare/train/merge/test及完整流水线脚本、依赖安装/预检、README和参数对照。
6. 先CPU单元/集成测试，再真实7B小批训练、LoRA更新与合并/评估冒烟；通过后启动完整1000条训练流水线。记录实际通过的测试，不将脚本化轨迹或CPU测试称为7B任务准确率。

## 验收与产物

- `plan.md`：本方案及实施状态。
- `upstream/`、`source_manifest.json`：上游源码来源和修改边界。
- `code/`、`configs/`、`scripts/`、`tests/`：独立可执行实现与测试。
- `verification/`：依赖预检、原框架调用验证、分组与梯度批次、真实模型冒烟证据。
- `runs/<run>/`：准备数据、训练日志、Planner LoRA、合并模型、两个测试集结果、阶段状态。

## 初始状态

2026-11-15：已读上游Solver、Planner、Executor、Verifier、Memory、rollout、AgentFlowTrainer、daemon及默认配置。待完成实现与验证。环境中原4_1训练环境不含VERL/vLLM；需建立独立兼容环境，不修改4_1环境。对依赖API不匹配会先核对本地源码并做必要兼容修改。


## 实施记录（持续更新）

- 已保存AgentFlow上游commit与文件SHA256；原Solver及各模块核心方法复用。
- 已实现GraphRL图工具、候选独立状态、公开输入白名单和确定性终局匹配；固定角色共用127.0.0.1:8621本地冻结7B服务。
- 已复用原AgentModeDaemon转移构造和AgentFlowTrainer/VERL更新，SDK传输替换为本地有界线程池；所有Planner实际token来自vLLM响应，无重新编码伪造。
- 为LoRA增加原Trainer的ref_in_actor兼容分支；异步推理采用不可变基础权重+LoRA delta同步，训练模型不merge。
- 12项测试通过；原生VERL+tiny-Qwen真实前向/反向/LoRA更新验证2源×6流程的13个变长转移仅一次optimizer.step；LoRA同步与PEFT merge一致。
- prepare已通过2源冒烟及两个完整600条测试集校验。真实7B冒烟正在执行，完整1000条训练与最终测试尚未完成；以runs状态与verification证据为准。

- GPU兼容：移除与vLLM休眠池不兼容的expandable_segments；休眠池使用普通CPU内存备份，小型真实CUDA休眠/唤醒校验已通过。VERL原卸载helper仅将non_blocking=True改为False，保留其FSDP状态和优化器处理逻辑，以兼容当前驱动的GPU→CPU传输。


用户最新要求：所有模型采用BF16。冻结基础模型、可训练Planner LoRA、梯度与梯度归约、优化器动量、固定角色和合并模型统一BF16；禁用PEFT默认的FP32 adapter提升，不保留FP32主权重。框架中的整数token/计数器及统计标量保持原生类型。实际dtype按rank记录并校验。


## 真实 7B 验证通过

2026-11-16：真实 7B 已完成 2 源 × 6 采样的一次 BF16 梯度更新、LoRA 导出、BF16 merge，以及两个测试集各 2 条的小批测试。全部 dtype 与真实权重更新均有审计文件；细节见 `verification/verification.md`。正式实验运行 1000 条训练、500 次更新，再 merge 和两个完整 600 条测试集。

## 2026-11-16 正式运行中断修复

首次正式运行在00:18:39退出，完成2/500次更新；第3批写日志时遇到Python集合不可JSON序列化。已复现：字面量解析允许集合，图执行虽拒绝该操作，审计记录仍保存了集合。现在执行前递归限制为JSON兼容字面量，非法命令保留原文并正常返回命令错误；不改变奖励或按结果重采样。补充全Solver流程的日志回归测试。

原先每50步保存，首次正式运行未产生检查点，故保留失败运行证据，从原始模型新建运行。保存策略改为第1步、之后每10步及最后一步，仍仅保留最近两份。增加仅针对当前运行目录与进程组的Ray残留清理；BF16、LoRA、6采样、2源/更新以及训练→merge→测试顺序不变。


## 2026-11-16：禁止空动作

按用户要求，四工具调用必须提供1—64条实际action。公共任务提示与每工具metadata同步声明非空，并用非空示例代替空列表示例；空列表返回EMPTY_ACTIONS，整次调用原子拒绝、图保持不变。规则不新增额外奖励项，也不重新采样。Verifier仍负责提前停止。旧训练运行保留为旧规则证据；新运行从旧运行global_step_10恢复模型、优化器、scheduler、RNG及dataloader，之后重新采样并应用新规则。轮数上限仍为3，用户尚未要求修改。


## 2026-11-16：工具轮数上限提高到10

按用户要求，tool_steps及run_workflow默认max_steps从3改为10，训练和merge后测试均使用10轮。每轮至多一次Planner、一次Executor、一次Verifier，另加初始问题分析与最终Generator，正常路径最多32次LLM生成。Verifier可提前STOP；500秒循环软限制和2048单次输出上限保持原配置。四工具继续要求非空actions，BF16、LoRA、每源6采样、每次更新2源保持原设置。

旧运行已完成到第36次更新，但最新完整可恢复检查点为global_step_30；新运行从该检查点恢复模型、优化器、scheduler、RNG与dataloader，第31步之后未保存的更新重跑。新运行独立目录保留来源记录，最终目标仍为累计500次更新，之后merge并测试两组各600条。旧运行及本文中3轮实例作为历史证据保留，不代表当前轮数配置。


## 2026-11-16：完整审计Memory与精简模型输入分离

增加`code/prompt_memory.py`：原Memory仍逐轮保存完整tool_name、sub_goal、command、result及各轮current_graph；GraphSession.audit与train_rollouts.jsonl继续保存完整证据。Planner、Verifier与最终Generator通过PromptMemory读取一份最新current_graph和逐轮操作摘要（工具、子目标、原始命令、ok/changed/graph_unchanged/errors），不再读取历史图快照。原Query中的初始图保留，Executor继续直接注入GraphSession最新真实图。问题分析只执行一次，不涉及历史图。

该视图是确定性字典过滤，不调用LLM摘要、不修改实际图、奖励或Flow-GRPO公式。所有历史错误仍可供纠错。摘要目前保留完整命令及子目标，因此不承诺固定token上限；模型自行在context中复述图也仍可能增加文本量。采样前即构造新prompt，训练继续使用服务端真实prompt/response token IDs，禁止采样后为训练另改输入。

对之前OOM批次最长样本做离线固定输出回放：原Planner最长输入13541 token，新视图最长5505 token，减少59.35%；固定角色最长输入从15198降为5569。回放最终图与原记录一致，保留10轮完整审计；这仅证明prompt构造与历史保留，不是新策略预测效果或真实GPU反向传播通过的证明。证据见`verification/prompt_memory_comparison.json`、`prompt_memory_replay.json`。

训练从原始3轮运行的第30步完整检查点重启，恢复模型、优化器、scheduler、RNG与dataloader，然后按10轮、禁止空动作、新视图重新采样。之前10轮失败运行第31步未保存，故需重跑。BF16、LoRA、每源6采样、每2源一次更新保留。


## 2026-11-16：每张Planner卡两条轨迹并行

采样线程池workers从2提高到4，新增planner_flows_per_server=2信号量，按原候选序号轮询两个Planner服务；每个服务至多同时推进两条flow。等待固定模块的HTTP调用只占对应flow线程，不占Planner GPU，vLLM可调度另一条flow。训练microbatch仍为每卡1转移，2源×6flow一次更新。为保留已完成但尚未保存的训练状态，在当前运行保存第40步完整检查点后自动停止并从该检查点以新配置恢复；具体状态见runs/pending_parallel4.json。


## 按需检查点接口

新启动的Trainer会在每次完整更新结束后检查RUN_DIR/SAVE_CHECKPOINT_REQUEST；存在时保存模型、优化器、RNG和dataloader，成功后重命名为SAVE_CHECKPOINT_COMPLETED_步数。失败不会删除请求。该接口只在新加载的Trainer中生效，不能使已运行的旧进程热更新。当前运行仍由原第40步保存机制及pending_parallel4切换程序接管。

## 2026-11-16：自动GPU吞吐调参

用户授权持续尝试提高并发和微批次。已向当前训练请求在完整更新边界保存检查点；控制器等待保存确认后停止旧运行。所有试验从这份相同检查点和dataloader位置开始，每档运行两次真实更新，禁止将调参试验模型作为正式训练进度累计。并发测试4/6/8/12，每服务flow槽数为并发/2；固定模型max_num_seqs同步调整。选较快并发后尝试每卡microbatch=2/4/8，发生OOM或吞吐无明显改善就停止继续增加该项。至少3%的step耗时收益才采用新配置，候选额外复验两步；正式训练最后从保存的原检查点恢复，仍训练到累计500步，再merge/test。

新增SETTINGS_FILE与FIXED_MAX_SEQS环境覆盖，避免候选设置互相覆盖或污染运行中的配置。每卡微批次变大时，先将总转移数对齐到world_size×microbatch，避免末尾不完整microbatch；policy和KL采用按转移token均值再平均，并对padding分母clamp，padding梯度按真实转移数补偿。这与原microbatch=1逐转移平均一致，不因多个转移长度不同而改变目标权重。17项测试通过，包括tiny-Qwen BF16 microbatch2的一次真实前向/反向/LoRA更新。

控制器为tuning/autotune.py；最新搜索入口tuning/latest_search.json，阶段、候选结果与GPU监测文件保存在独立tuning/search_*目录。GPU统计排除模型启动阶段，实际step/s与gen/s用于决策。有限候选搜索是当前授权的首轮调参；不能保证并发越大越快。

## 2026-11-16：按用户要求返回探索前组合

停止参数搜索，恢复全局workers=4、每Planner服务flow槽=2、训练microbatch=1；code/batching.py及AgentFlow trainer从探索前正式运行code_snapshot还原，移除探索阶段为microbatch>1增加的归一化/对齐路径。恢复点为探索前保存的global_step_44，调参试验第45步不计入正式进度。保持10轮、精简Memory、非空动作、BF16、LoRA、每源6采样、2源一次更新。此前同参数在长输入上OOM的记录保留，恢复不代表已解决长输入显存问题。

## 2026-11-16：Planner单次总token阈值与整轨迹丢弃

用户要求超阈值轨迹丢弃。新增planner_max_total_tokens=6144，按服务端真实prompt_token_ids+response_token_ids检查每次Planner调用。每flow正常采样结束后检查全部转移，任意一轮>6144即整flow不进入GRPO张量；日志保留完整记录、discarded、discard_reason、token_limit_audit，不将其伪造为0分失败样本，不补采。恰好6144允许。该阈值不是累计全flow输出token，也不是仅生成token；固定模块不进入训练反向传播，因此此阈值只用于训练Planner trace。

每源仍尝试6条flow，但实际参与更新的flow可能少于6；两个源各至少保留2条才更新，否则跳过该数据批次并记录，不推进优化器或scheduler。单epoch实际更新总数因跳过可能少于500，结束时保存真实最后更新检查点。测试集不启用按长度丢弃，避免改变600条评估分母。

6144是基于约6.1k单次总长度通过、约8.4k长度OOM设置的保守初始值，并非显卡绝对安全上限。仍保留BF16、全局4并发/每服务2flow、训练microbatch1、10轮与精简Memory。历史失败第46批有6/12条flow会按新规则丢弃。需要新运行验证显存，不能仅凭阈值配置断言不再OOM。


## 2026-11-16：tokenizer Unicode编码错误修复

上一运行在第48次更新后、下一采样批Executor请求阶段收到HTTP400 TextEncodeInput。已使用实际Qwen tokenizer复现：JSON孤立代理字符经Pydantic严格JSON拒绝后，原Planner宽松json.loads+Python模型回退仍可接受，进入下游prompt会触发完全相同报错。原失败payload未保存，故此为已复现且吻合的故障路径，不声称已证明原请求中的具体异常字符。

新增text_encoding.repair_surrogates，在请求发送前仅规范化代理字符：合法UTF-16代理对合成为原Unicode字符；孤立代理码点转为可见反斜杠u转义文本。正常Unicode、中文和emoji不改动。采样记录的是实际发送后的messages与服务端真实token IDs，训练不会另改prompt。unicode修复前后原文、位置与请求失败payload存入RUN_DIR/request_diagnostics，采用ensure_ascii保留非法字符原始码点；flow和测试JSONL也改用ASCII转义序列，避免审计文件自身UTF-8写入失败。基础设施错误仍不自动伪造0分或按奖励重采。

22项测试通过，包括正常文本不变、代理对、孤立字符、实际发送prompt与trace一致、诊断记录及完整训练基础检查。下一运行继续从第44步恢复，固定4并发/每卡2flow、micro1、BF16、10轮、非空动作和6144整轨迹过滤。运行参数save_freq临时设为1，每次实际更新后保存，仍只保留最近2份，以减少异常重跑损失。


## 2026-11-17：结束7B，改用3B，微批次6→4→3

用户要求停止7B并用Qwen2.5-3B-Instruct重新完成实验。7B正式运行停止，完整global_step_50保留。Planner与冻结问题分析/Executor/Verifier/Generator全部切换到本地3B原始权重，LoRA从头训练，不加载7B adapter。每卡训练microbatch先6，若训练OOM自动降4，再OOM降3；如已有完整3B检查点则从它恢复，否则从3B原始权重重跑。非显存故障不会盲目换微批次。控制器scripts/run_3b_fallback.py，状态runs/3b_microbatch_control.json。

训练转移先补零loss至world_size×microbatch的倍数；每个本地batch全部转移仍只做一次optimizer.step。policy/KL按逐转移token均值再平均，保持micro1原本的转移权重；padding分母clamp并做全局有效样本缩放。原VERL初始化对固定minibatch的整除检查与可变流程不匹配，因此只在worker初始化阶段使用micro1完成资源初始化，再恢复真实micro并对齐占位mini；每次真实更新重新设为本卡全部转移数。22项测试通过，包含tiny-Qwen BF16 micro6/4/3三档真实前后向、一次optimizer更新与BF16动量。真实3B稳定性以run日志为准。

保留全局4条流程并发/每Planner卡2条、每源6采样、每次2源、10轮工具上限、禁止空action、精简Memory、Unicode处理、6144训练token过滤、每步保存并保留最近2份、训练→merge→两组各600测试。模型路径由pipeline读取settings.json，避免冻结服务仍误用7B。


## 2026-11-17：首个Planner前容量失败的处理

3B首次启动于00:18:44在首批采样中退出，直接异常为No Planner transition，尚未完成梯度更新；不是OOM，故未降微批次。对于workflow已明确标记capacity_failure且trace为空的情况，现在整flow排除并记录NO_PLANNER_TRACE_CAPACITY，不给伪造训练样本、不补采。此类情况可由初始冻结问题分析达到2048输出上限触发；原失败flow缺乏完整审计，故不声称已识别当时具体输出。非容量原因的空trace仍中止，但新增详细diagnostic。23项测试通过，含12条全部在Planner前容量失败时有完整丢弃日志、无重采与批次不可更新的验证。保持micro6，重新启动3B实验，训练OOM时才降4/3。

## 2026-11-17：用户指定微批次降到2

3B micro6/4/3依次因OOM退出，最新完整3B检查点为micro3运行的global_step_2。按用户最新要求，将每卡训练微批次改为2，从该检查点续训；不再自行降到1。全局workers4、每Planner服务2条flow的并发设置仍保留，与训练微批次独立。模型与固定角色仍全部3B，BF16、LoRA、6采样、2源一次更新、10轮、非空action、6144过滤、精简Memory和每步保存均沿用。
