"""
GraphSSR 的 Sample-Select-Reason (SSR) 流程 prompt 模板构建器。

本文件实现的功能：
- 定义 GraphSSR 中共享的标准 SSR prompt 结构。
- 强制模型按照"先采样子图 → 再选择一个子图 → 最后在选择的子图上进行推理"的流程操作。

在训练流程中的角色：
- SFT 数据生成阶段：将 GraphR1 样本转换为带有固定输出格式要求的教师模型 prompt。
- RL 数据生成与评估阶段：确保所有 prompt 与下游的解析器和奖励函数保持一致的对齐。
"""

# Sample-Select-Reasoning (SSR) 流程的 Prompt 模板类
class PromptTemplete():
    
    # 任务描述：总起，定义图分析专家的角色和任务流程概述
    def task_description_all_prompt(self):
        task_description = "Task Description:\nYou are a graph analysis expert. Given a complete subgraph structure, which includes the description of the central node(s), its neighboring nodes, and their connection relationships, please follow the following steps to operate the analysis:\n\n"
        return task_description
    
    # 完整子图描述：列出中心节点、邻居节点和连接关系
    def complete_subgraph_prompt(self, target_nodes, node_information, edge_information):
        complete_subgraph = f"Complete subgraph:\n- Central node ID(s): {target_nodes}\n- Node texts (central node(s) and neighboring nodes): [\n{node_information}\n]\n- Connection relationships: [\n  {edge_information}\n]\nIf the node texts and connection relationships are empty, it indicates that these pieces of information do not exist, don't try to fabricate.\n\n"
        return complete_subgraph
    
    # 步骤描述标题
    def step_description_title_prompt(self):
        step_description_title = "Step description:\n"
        return step_description_title

    # 采样阶段的输出格式模板
    def step_description_subgraph_sampling_prompt(self, step_index, task_name, group_size):
        title = f"{step_index}. **Subgraph Sampling**:\n"

        sample_num = f"  - In order to complete the {task_name} task of the central node(s), you need to first sample **{group_size}** subgraphs from the complete graph that are most suitable for the current task from this subgraph.\n"

        node_constrain = "  - These subgraphs must **keep all central node(s)** as mentioned in the complete subgraph, they can include no, partial or all one-hop or two-hop neighbors of the central node(s).\n"

        real_subgraph = "  - You must ensure that the sampled subgraphs actually exist and do not create fictitious nodes and edges.\n"

        central_node_constrain = "  - There must be a subgraph that consists only the central node(s).\n"

        if task_name == "Node Classification" or task_name == "Link Classification":
            recommend_subgraph = "  - There should also be a subgraph, where as many neighboring nodes as possible can semantically support the central node(s). If there is no such subgraph, do not fabricate one.\n"
        elif task_name == "Link Prediction":
            recommend_subgraph = "  - There should also be a subgraph that contains common neighboring nodes, and these nodes can connect the two central nodes. If there is no such subgraph, do not fabricate one.\n"
        else:
            recommend_subgraph = ""

        diversity_1 = "  - You must ensure that the information differences between these subgraphs are as significant as possible, so that these subgraphs can represent different semantic and structural meanings. The sizes of the subgraphs should also be as different as possible.\n"
        diversity_2 = "  - If the complete subgraph does not support to sample distinct and dissimilar subgraphs, you can sample similar subgraphs.\n"
        diversity = diversity_1 + diversity_2

        reason = "  - Explain the reason for sampling each subgraph.\n"

        output_constraint = (
            f"  - The final structured answer must contain exactly {group_size} complete "
            f"subgraph blocks numbered consecutively from 0 to {group_size - 1}.\n"
        )

        step_description_subgraph_sampling = title + sample_num + node_constrain + real_subgraph + central_node_constrain + recommend_subgraph + diversity + reason + output_constraint

        return step_description_subgraph_sampling
    
    # 选择阶段：从采样的子图中选出最适合当前任务的一个
    def step_description_subgraph_chosen_prompt(self, step_index):
        step_description_subgraph_chosen = f"{step_index}. **Subgraph Choosing**: From the sampled subgraphs, choose one subgraph that is best suited for the current task and use it for subsequent reasoning. Explain the reason for choosing this subgraph.\n"
        return step_description_subgraph_chosen
    
    # 推理阶段：在选中的子图上独立进行推理并给出答案
    def step_description_reasoning_prompt(self, step_index, task_name):

        reasoning_title = f"{step_index}. **{task_name} Reasoning**:\n"

        subgraph_repeat = "  - Repeat and verify the subgraph you have chosen, ensuring that the chosen subgraph you analyze and output is consistent with the sampled subgraphs.\n"

        analyze_independent = "  - Analyze the chosen subgraph independently. Only consider the nodes and edges of the current subgraph. The obtained answer is entirely based on the chosen subgraph and is independent of other sampled subgraphs.\n"

        whole_graph = "  - You must treat the current subgraph as a whole for reasoning, taking into account the information of all nodes in this subgraph, not just the information of the central node(s).\n"
        
        only_connected_nodes = "  - Except for all the nodes and edges contained within the subgraph, do not consider the information of any other nodes and edges, even if they are mentioned in the complete subgraph.\n"
            
        if task_name == "Link Prediction":
            bond_value = "  - Provide the predicted connection bond value of the two central nodes, set the threshold to 0.5. "
        else:
            bond_value = "  - "

        answer = bond_value + f"Provide the {task_name} answer of the central node(s) obtained through reasoning, the answer must be selected strictly from the following {task_name} options. Each option is enclosed within < >, but when you answer, do not include < >.\n"

        reason = "  - Explain the reason for select this answer.\n"

        step_description_reasoning = reasoning_title + subgraph_repeat + analyze_independent + whole_graph + only_connected_nodes + answer+ reason

        return step_description_reasoning

    # 选项列表
    def step_description_options_prompt(self, step_index, task_name, label_list):
        step_description_options = f"{step_index}. **{task_name} Options**: {label_list}.\n"
        return step_description_options
    

    # 答案格式标题
    def answer_format_title_prompt(self):
        answer_format_title = "Answer format:\nYour response should be in forms as follows, no other outputs are needed:\n"
        return answer_format_title
    
    # 采样阶段的输出格式模板
    def answer_format_subgraph_sampling_prompt(self, group_size=5):
        """Expand every required candidate block without duplicating long prose."""
        field_guide = (
            "For every numbered subgraph block below, replace the bracketed hints: "
            "Central_node_ID must contain all central node ID(s); "
            "Neighboring_node_ID may be blank; Connection_relationship may be blank "
            "and every edge must use <node_ID, node_ID>; Sampling_reason must be non-empty.\n"
        )
        blocks = [field_guide]
        for subgraph_index in range(group_size):
            blocks.append(
                f"<Subgraph_{subgraph_index}>\n"
                "  <Central_node_ID>[central node ID(s)]</Central_node_ID>\n"
                "  <Neighboring_node_ID>[neighboring node ID(s), or blank]</Neighboring_node_ID>\n"
                "  <Connection_relationship>[edges, or blank]</Connection_relationship>\n"
                "  <Sampling_reason>[non-empty sampling reason]</Sampling_reason>\n"
                f"</Subgraph_{subgraph_index}>\n"
            )
        return "".join(blocks)
    
    # 选择阶段：从采样的子图中选出最适合当前任务的一个 and Reasoning Phase
    def answer_format_chosen_reasoning_prompt(self, task_name):
        if task_name == "Link Prediction":
            bond_value = "  <Bond_value>the bond value</Bond_value>\n"
        else:
            bond_value = ""
        answer_format_chosen_reasoning = f"<Chosen_subgraph_reasoning>\n  <Chosen_subgraph>directly give the index number of the chosen subgraph</Chosen_subgraph>\n  <Chosen_subgraph_reason>the reason for choosing this subgraph</Chosen_subgraph_reason>\n  <Central_node_ID>the central node id(s), if there are multiple ones, please separate them with commas</Central_node_ID> \n  <Neighboring_node_ID>the neighboring node ids, separate them with commas, if it does not exist, leave it blank</Neighboring_node_ID>\n  <Connection_relationship>the connection relationships, each relation is presented in the form of <node_ID, node_ID>, separated by commas, if it does not exist, leave it blank</Connection_relationship>\n{bond_value}  <Answer>the answer</Answer>\n  <Brief_reasoning>the brief reasoning</Brief_reasoning>\n</Chosen_subgraph_reasoning>\n"
        return answer_format_chosen_reasoning

# 构造完整 SSR prompt：串联任务描述 → 完整子图 → 步骤描述 → 答案格式
def construct_prompt_chosen_reasoning(task, target_nodes, node_information, edge_information, label_list, group_size=5):
    if task == 'nc':
        task_name = "Node Classification"
    elif task == 'lp':
        task_name = "Link Prediction"
    elif task == 'lc':
        task_name = "Link Classification"
    else:
        task_name = "Unknown Task"

    pt = PromptTemplete()
    task_description = pt.task_description_all_prompt()
    complete_subgraph = pt.complete_subgraph_prompt(target_nodes, node_information, edge_information)
    step_description = pt.step_description_title_prompt() + pt.step_description_subgraph_sampling_prompt(1, task_name, group_size) + pt.step_description_subgraph_chosen_prompt(2) + pt.step_description_reasoning_prompt(3, task_name) + pt.step_description_options_prompt(4, task_name, label_list) + "\n"
    answer_format = pt.answer_format_title_prompt() + pt.answer_format_subgraph_sampling_prompt(group_size) + pt.answer_format_chosen_reasoning_prompt(task_name)

    prompt = task_description + complete_subgraph + step_description + answer_format

    return prompt
