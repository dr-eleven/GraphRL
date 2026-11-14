"""Render complete current training prompts from a real source sample."""
from pathlib import Path
import inspect,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from transformers import AutoTokenizer
from trl.chat_template_utils import get_training_chat_template
from prompt_template import single_call_template
from episode import GraphEnvironment,initial_messages
from project_config import EXPERIMENT_ID,DEFAULT_MODE

def main():
    workspace=ROOT.parents[2]
    sample=json.loads((workspace/'GraphRL/2000_oneclass_sorted/train_grpo_episodes_redesign.json').read_text())[0]
    tok=AutoTokenizer.from_pretrained(str(workspace/'Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct'),local_files_only=True)
    template=single_call_template(get_training_chat_template(tok) or tok.chat_template)
    env=GraphEnvironment()
    tools=[getattr(env,n) for n,f in inspect.getmembers(type(env),inspect.isfunction) if not n.startswith('_') and n!='reset']
    intro=f"""# {EXPERIMENT_ID} 当前完整 Prompt：最多 10 轮，无终答预留

本文从当前代码实际渲染。system包含全部任务/协议/图约束和工具schema；user仅包含公开任务数据、候选标签、保护信息和图。四工具保留3_6的噪声识别/修复含义；共同的证据要求、最新图、analysis顺序和空操作规则已集中到system，工具只保留用途与各自action字段。主训练始终在同一轨迹中进行图操作并最终预测；“仅预测基线”为额外评估对照，不是另一训练阶段。包含完整系统消息、工具schema、公开任务、初始图和assistant前缀。样本为 2000_oneclass_sorted 第一条，source_index={sample['source_index']}，无图省略、无任务占位符；ground_truth、clean_graph 不进入 prompt。

最多10个模型生成轮次，成功调用、失败请求、终答修正都计轮；没有单阶段、单工具或错误次数上限，也没有工具总次数独立上限。没有终答预留和额外终答轮。第10轮若为工具调用则执行并回填结果后停止，不追加下一轮 assistant 前缀；若无有效最终提交，奖励0。第10轮有效终答仍正常评分。token/context等硬容量可能更早结束，无答案同样0；不会为终答强制腾出轮次或token。

4_1：按 del_node → del_edge → add_edge → rewrite 前进。成功（含合法 actions=[]）后可重复当前工具或进入紧邻下一阶段，不可跳阶段或回退；失败后只能修正当前阶段，新的成功才重新开放下一阶段。rewrite成功后可继续rewrite或终答。成功反馈提供 next_tool 选择集，失败反馈列出全部可独立检测错误，不开放下一阶段。

5_1：四工具可任意顺序、重复调用，允许零工具终答。两组均要求非空 analysis 在 actions 前，终答为 analysis 在 prediction 前。JSON/事件解析错误仍直接终止，无错误obs、不重采；可解析的schema/动作错误返回完整合并反馈，不按错误次数停止。失败请求整批回滚；初始图标记为 initial_graph；实际变更才返回 restored_graph。无图变化仅提示复用最近更新图，否则使用 initial_graph。有效终答结束后才查询正确性，错误合法答案不能重猜。

约束并未取消：system统一保留基于当前图结构/文本证据、analysis先于且对应actions、无必要编辑则actions=[]、不伪造执行结果；图约束包括节点/边保护、存在性、禁止自环/重复冲突/relation字段、删节点同步删关联边、限定可改写节点、完整非空替换文本、禁止仅改空白/大小写。改写须保留有依据事实，不编造事实或插入答案/标签/答案提示。字段、图合法性等由现有校验器执行；证据充分性、analysis与actions语义一致等是模型指令，并没有新增语义判定器。

User中的<initial_graph>是本轨迹初始输入，可能含噪声，不代表已去噪。只有成功且实际改变图的反馈才含<restored_graph>；它表示工具执行后的当前图，不保证已恢复到真实干净图。后续使用最新restored_graph，尚无更新时使用initial_graph；空操作/失败反馈会明确指向这两者之一，不重复图。

本项目默认模式：{DEFAULT_MODE}。下面代码块为完整实际首轮 prompt，分别覆盖主方案和仅预测基线。
"""
    parts=[intro]
    for bypass in [False,True]:
        prompt=tok.apply_chat_template(initial_messages(sample,bypass_actions=bypass),tools=tools,chat_template=template,tokenize=False,add_generation_prompt=True)
        parts.append('## '+('仅预测基线' if bypass else '图编辑主方案')+'\n\n```text\n'+prompt+'\n```\n')
    (ROOT/'Prompt.md').write_text('\n'.join(parts))
    print('Exported',ROOT/'Prompt.md')
if __name__=='__main__':main()
