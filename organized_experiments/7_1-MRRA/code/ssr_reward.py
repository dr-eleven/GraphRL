import numpy as np
import re

def format_edge(edge_matches, multiple_directions=True):
    edges = [[node.strip("<").strip(">").strip() for node in match.split(",")] for match in edge_matches]
    edges_new = []
    for edge in edges:
        edge1 = "<" + edge[0] + ", " + edge[1] + ">"
        edges_new.append(edge1)
        if multiple_directions:
            edge2 = "<" + edge[1] + ", " + edge[0] + ">"
            edges_new.append(edge2)

    return edges_new

def extract_prompt_info(prompt, multiple_directions=True):
    target_node_pattern = r'- Central node ID\(s\): (.*?)\n'
    target_nodes = re.findall(target_node_pattern, prompt, re.DOTALL)[-1].strip().split(",")
    target_nodes = [node.strip() for node in target_nodes]

    node_pattern = r'node\d+:.*?\n'
    node_matches = re.findall(node_pattern, prompt, re.DOTALL)
    neighboring_nodes = [match.strip().split(":")[0] for match in node_matches]

    edge_pattern = r'<node\d+, node\d+>'
    edge_matches = re.findall(edge_pattern, prompt, re.DOTALL)
    edges = format_edge(edge_matches, multiple_directions)

    return target_nodes, neighboring_nodes, edges

def central_node_id_regex(subgraph):
    central_node_pattern = r"<Central_node_ID>(.*?)</Central_node_ID>"
    central_node_match = re.findall(central_node_pattern, subgraph, re.DOTALL)
    if len(central_node_match) == 0 or central_node_match[-1] == "":
        central_node_ans = []
    else:
        central_node_ans = central_node_match[-1].strip().split(",")
        central_node_ans = [node.strip() for node in central_node_ans]
    return central_node_ans

def neighboring_node_id_regex(subgraph):
    neighboring_node_pattern = r"<Neighboring_node_ID>(.*?)</Neighboring_node_ID>"
    neighboring_node_match = re.findall(neighboring_node_pattern, subgraph, re.DOTALL)
    if len(neighboring_node_match) == 0 or neighboring_node_match[-1] == "":
        neighboring_node_ans = []
    else:
        neighboring_node_ans = neighboring_node_match[-1].strip().split(",")
        neighboring_node_ans = [node.strip() for node in neighboring_node_ans]
    return neighboring_node_ans

def connection_relationship_regex(subgraph):
    connection_relationship_pattern = r"<Connection_relationship>(.*?)</Connection_relationship>"
    connection_relationship_match = re.findall(connection_relationship_pattern, subgraph, re.DOTALL)
    if len(connection_relationship_match) == 0 or connection_relationship_match[-1] == "":
        connection_relationship_ans = []
    else:
        connection_relationship_ans = connection_relationship_match[-1]
        edge_pattern = r'<node\d+,\s*node\d+>'
        edge_matches = re.findall(edge_pattern, connection_relationship_ans, re.DOTALL)
        connection_relationship_ans = format_edge(edge_matches, multiple_directions=False)
    return connection_relationship_ans

def extract_sampled_subgraph_info(response):
    response = response.split("</think>", 1)[-1]

    subgraph_pattern = r"<Subgraph_(\d+)>(.*?)</Subgraph_\1>"
    subgraphs = re.findall(subgraph_pattern, response, re.DOTALL)
    subgraphs = sorted(subgraphs, key=lambda x: int(x[0]))
    
    subgraph_list = []
    for subgraph in subgraphs:
        subgraph = subgraph[1]

        central_node_ans = central_node_id_regex(subgraph)
        neighboring_node_ans = neighboring_node_id_regex(subgraph)
        connection_relationship_ans = connection_relationship_regex(subgraph)

        subgraph_list.append((subgraph, central_node_ans, neighboring_node_ans, connection_relationship_ans, ""))

    return subgraph_list

def extract_chosen_subgraph_reasoning_info(response):
    response = response.split("</think>", 1)[-1]

    chosen_subgraph_reasoning_pattern = r"<Chosen_subgraph_reasoning>(.*?)</Chosen_subgraph_reasoning>"
    chosen_subgraph_reasoning_match = re.findall(chosen_subgraph_reasoning_pattern, response, re.DOTALL)
    if len(chosen_subgraph_reasoning_match) == 0:
        chosen_subgraph_reasoning = ""
    else:
        chosen_subgraph_reasoning = chosen_subgraph_reasoning_match[-1].strip()

    chosen_subgraph_pattern = r"<Chosen_subgraph>(.*?)</Chosen_subgraph>"
    chosen_subgraph_match = re.findall(chosen_subgraph_pattern, chosen_subgraph_reasoning, re.DOTALL)
    if len(chosen_subgraph_match) == 0 or chosen_subgraph_match[-1] == "":
        chosen_subgraph = ""
    else:
        try:
            chosen_subgraph = int(chosen_subgraph_match[-1].strip())
        except:
            chosen_subgraph = ""

    chosen_central_node_ans = central_node_id_regex(chosen_subgraph_reasoning)
    chosen_neighboring_node_ans = neighboring_node_id_regex(chosen_subgraph_reasoning)
    chosen_connection_relationship_ans = connection_relationship_regex(chosen_subgraph_reasoning)

    answer_pattern = r"<Answer>(.*?)</Answer>"
    answer_match = re.findall(answer_pattern, chosen_subgraph_reasoning, re.DOTALL)
    if len(answer_match) == 0 or answer_match[-1] == "":
        answer = ""
    else:
        answer = answer_match[-1].strip()

    return chosen_subgraph, chosen_central_node_ans, chosen_neighboring_node_ans, chosen_connection_relationship_ans, answer

def check_real_subgraph(prompt_info, sampled_subgraph_list):
    prompt_target_nodes, prompt_neighboring_nodes, prompt_edges = prompt_info
    prompt_target_nodes = set(prompt_target_nodes)
    prompt_neighboring_nodes = set(prompt_neighboring_nodes)
    prompt_edges = set(prompt_edges)

    status = True
    false_reasons = []
    for i in range(len(sampled_subgraph_list)):
        _, central_node_ans, neighboring_node_ans, connection_relationship_ans, _ = sampled_subgraph_list[i]
        central_node_ans = set(central_node_ans)
        neighboring_node_ans = set(neighboring_node_ans)
        connection_relationship_ans = set(connection_relationship_ans)

        if not central_node_ans == prompt_target_nodes:
            status = False
            false_reasons.append(f"Central nodes do not match for subgraph {i}: {prompt_target_nodes} vs {central_node_ans}")
        
        if not neighboring_node_ans.issubset(prompt_neighboring_nodes):
            status = False
            false_reasons.append(f"Neighboring nodes for subgraph {i} are not subset of prompt neighboring nodes: {neighboring_node_ans}vs {prompt_neighboring_nodes}")
        
        if not connection_relationship_ans.issubset(prompt_edges):
            status = False
            false_reasons.append(f"Connection relationships for subgraph {i} are not subset of prompt edges: {connection_relationship_ans} vs {prompt_edges}")

    return status, false_reasons

def check_chosen_subgraph_consist(sampled_subgraph_list, chosen_subgraph_reasoning_info):
    chosen_subgraph, chosen_central_node_ans, chosen_neighboring_node_ans, chosen_connection_relationship_ans, _ = chosen_subgraph_reasoning_info
    if chosen_subgraph == "" or chosen_subgraph >= len(sampled_subgraph_list):
        return False, [f"Chosen subgraph index wrong: {chosen_subgraph}"]

    status = True
    false_reasons = []

    _, central_node_ans, neighboring_node_ans, connection_relationship_ans, _ = sampled_subgraph_list[chosen_subgraph]
    central_node_ans = set(central_node_ans)
    neighboring_node_ans = set(neighboring_node_ans)
    connection_relationship_ans = set(connection_relationship_ans)

    chosen_central_node_ans = set(chosen_central_node_ans)
    chosen_neighboring_node_ans = set(chosen_neighboring_node_ans)
    chosen_connection_relationship_ans = set(chosen_connection_relationship_ans)

    if not central_node_ans == chosen_central_node_ans:
        status = False
        false_reasons.append(f"Central nodes do not match for subgraph {chosen_subgraph}: {central_node_ans} vs {chosen_central_node_ans}")
    
    if not neighboring_node_ans == chosen_neighboring_node_ans:
        status = False
        false_reasons.append(f"Neighboring nodes do not match for subgraph {chosen_subgraph}: {neighboring_node_ans} vs {chosen_neighboring_node_ans}")
    
    if not connection_relationship_ans == chosen_connection_relationship_ans:
        status = False
        false_reasons.append(f"Connection relationships do not match for subgraph {chosen_subgraph}: {connection_relationship_ans} vs {chosen_connection_relationship_ans}")

    return status, false_reasons

def check_chosen_answer(truth_label, chosen_subgraph_reasoning_info):
    truth_label = truth_label.strip().lower()
    _, _, _, _, answer = chosen_subgraph_reasoning_info
    answer = answer.strip().lower()
    if answer == truth_label:
        return True
    else:
        return False

def rank_of_i(arr, i):
    return sum(1 for x in arr if x > arr[i])

def cal_size_score(sampled_subgraph_list, chosen_subgraph_reasoning_info):
    chosen_subgraph, chosen_central_node_ans, chosen_neighboring_node_ans, chosen_connection_relationship_ans, _ = chosen_subgraph_reasoning_info
    if chosen_subgraph == "" or chosen_subgraph >= len(sampled_subgraph_list):
        return 0.0
    
    size_list = []
    for i in range(len(sampled_subgraph_list)):
        _, central_node_ans, neighboring_node_ans, connection_relationship_ans, _ = sampled_subgraph_list[i]
        subgraph_size = len(central_node_ans) + len(neighboring_node_ans)
        size_list.append(subgraph_size)

    size_rank = rank_of_i(size_list, chosen_subgraph)
    size_score = size_rank / (len(size_list) - 1) if len(size_list) > 1 else 1.0

    return size_score
        

def compute_score(solution_str, ground_truth, data_source, extra_info):

    prompt_info = extract_prompt_info(extra_info["prompt_str"])
    sampled_subgraph_list = extract_sampled_subgraph_info(solution_str)
    chosen_subgraph_reasoning_info = extract_chosen_subgraph_reasoning_info(solution_str)
    real_subgraph_status, _ = check_real_subgraph(prompt_info, sampled_subgraph_list)
    chosen_subgraph_consist_status, _ = check_chosen_subgraph_consist(sampled_subgraph_list, chosen_subgraph_reasoning_info)
    answer_status = check_chosen_answer(ground_truth, chosen_subgraph_reasoning_info)

    if extra_info["in_second_stage"]:
        size_score = cal_size_score(sampled_subgraph_list, chosen_subgraph_reasoning_info)
    else:
        size_score = 0.0

    if answer_status:
        pred = 1.0
    else:
        pred = 0.0

    if real_subgraph_status:
        if chosen_subgraph_consist_status:
            if answer_status:
                if extra_info["in_second_stage"]:
                    second_stage_reward_lambda = extra_info["second_stage_reward_lambda"]
                    score = 1.0 + second_stage_reward_lambda * size_score
                else:
                    score = 1.0
                real_pred = 1.0
            else:
                score = 0.1
                real_pred = 0.0
        else:
            score = 0.05
            real_pred = 0.0
    else:
        score = 0.0
        real_pred = 0.0

    return {"score": score, "pred": pred, "real_pred": real_pred}
