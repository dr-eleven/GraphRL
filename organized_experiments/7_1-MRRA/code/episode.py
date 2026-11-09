"""MRRA 的可见图、严格输出协议与一次性执行器；沿用 degree-sorted 的图表示。"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


# 图节点键统一为字符串；边沿用原实验的无向端点身份，保留已有 relation。
# 原数据允许的反向重复边在数据准备时去重，模型声明图不允许重复边。
def normalize_graph(graph, *, allow_duplicate_edges=False):
    if not isinstance(graph, dict) or set(graph) != {'nodes', 'edges'}:
        raise ValueError('Graph requires exactly nodes and edges')
    nodes = {}
    if not isinstance(graph['nodes'], dict) or not isinstance(graph['edges'], list):
        raise ValueError('nodes must be an object and edges a list')
    for node, text in graph['nodes'].items():
        if not str(node).isdigit() or not isinstance(text, str):
            raise ValueError('Node IDs must be nonnegative integers and texts strings')
        key = str(int(node))
        if key in nodes:
            raise ValueError('Duplicate node ID')
        nodes[key] = text
    edges = {}
    for edge in graph['edges']:
        if not isinstance(edge, dict) or not {'u', 'v'} <= set(edge) or set(edge) - {'u', 'v', 'relation'}:
            raise ValueError('Edge requires u,v and optional relation')
        u, v = _edge_key(edge['u'], edge['v'])
        if str(u) not in nodes or str(v) not in nodes:
            raise ValueError('Missing edge endpoint')
        value = {'u': u, 'v': v}
        if 'relation' in edge:
            if not isinstance(edge['relation'], str):
                raise ValueError('relation must be a string')
            value['relation'] = edge['relation']
        if (u, v) in edges and (not allow_duplicate_edges or edges[u, v] != value):
            raise ValueError('Duplicate or conflicting edge')
        edges[u, v] = value
    return {'nodes': dict(sorted(nodes.items(), key=lambda x: int(x[0]))),
            'edges': [edges[key] for key in sorted(edges)]}


def _node_id(value):
    if type(value) is not int or value < 0:
        raise ValueError('Action node IDs must be nonnegative JSON integers')
    return value


def _edge_key(u, v):
    return tuple(sorted((_node_id(u), _node_id(v))))


def _edge_map(graph):
    return {_edge_key(e['u'], e['v']): e for e in graph['edges']}


def text_nodes(sample):
    return list(dict.fromkeys(sample['protected_query_nodes'])) if sample['task_type'] == 'LC' else [sample['target_node']]


# 白名单隔离策略输入：不得传递旧 prompt、oracle、clean、ground_truth 或攻击元数据。
# 执行器也只收到这份公开观察；所有保护约束同时出现在模型输入中。
def public_observation(sample):
    targets = list(sample['protected_query_nodes']) if sample['task_type'] == 'LC' else text_nodes(sample)
    protected = sorted(set(targets + sample.get('protected_nodes', [])))
    return {'task_type': sample['task_type'], 'target_nodes': targets,
            'label_space': list(sample['label_space']),
            'protected_nodes': protected,
            'protected_edges': deepcopy(sample.get('protected_edges', [])),
            'text_edit_nodes': text_nodes(sample), 'graph': deepcopy(sample['noisy_graph'])}


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('Duplicate JSON field: ' + key)
        obj[key] = value
    return obj


def read_json(text):
    def invalid_constant(value):
        raise ValueError('Non-finite JSON value: ' + value)
    return json.loads(text.strip(), object_pairs_hook=_unique_object, parse_constant=invalid_constant)


# 阶段顺序与外层字段严格检查；动作合法性独立由执行器检查，不将恢复一致性加入 B。
def parse_completion(text, stage):
    obj = read_json(text)
    keys = {'llm': ['modify', 'rewrite', 'restored_graph', 'reason', 'answer'],
            'edit': ['modify', 'rewrite'], 'answer': ['reason', 'answer']}[stage]
    if not isinstance(obj, dict) or list(obj) != keys:
        raise ValueError('Expected ordered fields: ' + ', '.join(keys))
    for phase in ('modify', 'rewrite'):
        if phase not in obj:
            continue
        part = obj[phase]
        if not isinstance(part, dict) or set(part) != {'reason', 'actions'}:
            raise ValueError(phase + ' requires reason and actions')
        if not isinstance(part['reason'], str) or not part['reason'].strip() or not isinstance(part['actions'], list):
            raise ValueError(phase + ' requires nonempty reasoning and an action list')
    if 'reason' in obj and (not isinstance(obj['reason'], str) or not obj['reason'].strip()):
        raise ValueError('Reason must be nonempty')
    if 'answer' in obj and not isinstance(obj['answer'], str):
        raise ValueError('Answer must be a string')
    return obj


# 用于生成停止条件：只检测完整 JSON，既不替模型修复字段也不提前检查动作。
def complete_json(text):
    try:
        return isinstance(json.loads(text), dict)
    except (ValueError, TypeError):
        return False


# NO_OP 统一表示为空动作列表；也接受单独的 {"op":"NO_OP"}，禁止与编辑混用。
def phase_actions(actions, allowed):
    if not isinstance(actions, list):
        raise ValueError('actions must be a list')
    if actions == [{'op': 'NO_OP'}]:
        return []
    for action in actions:
        if not isinstance(action, dict) or action.get('op') not in allowed:
            raise ValueError('Action placed in the wrong phase or unsupported')
    return actions


# 原实验的整组事务回滚语义保留。先结构后文本，只执行一次。
# 不读取 clean_graph/injected_nodes；删普通非目标节点是可执行但可能错误的决策。
def execute_actions(observation, modify, rewrite):
    original = deepcopy(observation['graph'])
    graph = deepcopy(original)
    statuses, actions = [], []
    try:
        structural = phase_actions(modify, {'ADD_EDGE', 'DEL_EDGE', 'DELETE_NODE'})
        textual = phase_actions(rewrite, {'REWRITE'})
        actions = structural + textual
        nodes, edges = graph['nodes'], _edge_map(graph)
        protected = set(observation['protected_nodes'])
        protected_edges = {_edge_key(*pair) for pair in observation['protected_edges']}
        touched, deleted, rewritten = set(), set(), set()
        for index, action in enumerate(actions):
            op = action['op']
            fields = {'ADD_EDGE': {'op', 'u', 'v'}, 'DEL_EDGE': {'op', 'u', 'v'},
                      'DELETE_NODE': {'op', 'node'}, 'REWRITE': {'op', 'node', 'text'}}[op]
            if set(action) != fields:
                raise ValueError('Unexpected/missing action fields')
            if op in {'ADD_EDGE', 'DEL_EDGE'}:
                u, v = _edge_key(action['u'], action['v'])
                if u == v or str(u) not in nodes or str(v) not in nodes:
                    raise ValueError('Missing endpoint or self-loop')
                if (u, v) in protected_edges or (u, v) in touched or u in deleted or v in deleted:
                    raise ValueError('Protected, duplicate or conflicting edge action')
                if op == 'ADD_EDGE':
                    if (u, v) in edges:
                        raise ValueError('ADD_EDGE requires an absent edge')
                    edges[u, v] = {'u': u, 'v': v}
                else:
                    if (u, v) not in edges:
                        raise ValueError('DEL_EDGE requires a present edge')
                    del edges[u, v]
                touched.add((u, v))
            elif op == 'DELETE_NODE':
                node = _node_id(action['node'])
                if str(node) not in nodes or node in protected or node in deleted:
                    raise ValueError('Missing, protected or repeated node deletion')
                if any(node in pair for pair in touched):
                    raise ValueError('Node deletion conflicts with edge edits')
                deleted.add(node)
            else:
                node = _node_id(action['node'])
                if node not in observation['text_edit_nodes'] or str(node) not in nodes or node in deleted:
                    raise ValueError('REWRITE outside target text scope')
                if node in rewritten:
                    raise ValueError('Repeated target rewrite')
                value = action['text']
                if not isinstance(value, str) or not value.strip():
                    raise ValueError('REWRITE requires nonempty text')
                if ' '.join(value.split()).casefold() == ' '.join(nodes[str(node)].split()).casefold():
                    raise ValueError('REWRITE copies the input; use NO_OP')
                nodes[str(node)] = value
                rewritten.add(node)
            statuses.append({'index': index, 'op': op, 'status': 'executed'})
        for node in deleted:
            del nodes[str(node)]
        graph['edges'] = [edge for (u, v), edge in sorted(edges.items()) if u not in deleted and v not in deleted]
        return {'graph': graph, 'error': None, 'actions': actions, 'operation_status': statuses, 'rolled_back': False}
    except (ValueError, TypeError, KeyError) as exc:
        for status in statuses:
            status['status'] = 'rolled_back'
        statuses.append({'index': len(statuses), 'status': 'rejected', 'error': str(exc)})
        return {'graph': original, 'error': str(exc), 'actions': actions,
                'operation_status': statuses, 'rolled_back': True}


def rejected_execution(observation, error):
    return {'graph': deepcopy(observation['graph']), 'error': str(error), 'actions': [],
            'operation_status': [], 'rolled_back': True}


RULES = '''You are a graph recovery and reasoning policy. The graph may be clean or contain unknown
edge additions/deletions, misleading/missing target text, injected nodes, or combinations.
Infer edits only from visible evidence. Never invent unsupported facts. Prefer minimal edits.
Modify: explain structural judgments, then choose ADD_EDGE(u,v), DEL_EDGE(u,v), DELETE_NODE(node).
Rewrite: explain target text judgments using the input and proposed structural edits, then choose REWRITE(node,text).
Use [] for NO_OP in either phase. Do not edit merely because noise is possible.
Use integer IDs and exact action fields: {"op":"ADD_EDGE","u":0,"v":1},
{"op":"DEL_EDGE","u":0,"v":1}, {"op":"DELETE_NODE","node":2},
{"op":"REWRITE","node":0,"text":"complete supported replacement text"}.
These are field examples, not actions for the current graph. Existing IDs may be sparse.
Do not add nodes, self-loops, duplicate/conflicting edits or relation labels. Preserve any pre-existing self-loop.
Only add absent edges
and delete present edges. Do not delete protected nodes or edit protected query edges.
DELETE_NODE removes its incident edges; do not separately edit those edges.
Only text_edit_nodes may be rewritten, at most once per node. Preserve reliable information.
Do not copy unchanged text or insert task answers into node text. Any illegal action rolls back the entire set.
Edges are undirected, matching this experiment's input representation. Preserve existing relation attributes.
Return one JSON object, without Markdown or text outside it. All reason fields must explain your decisions.
'''


def first_messages(sample, strategy, ablation='full'):
    if strategy == 'executor':
        from executor_protocol import first_executor_messages
        return first_executor_messages(sample, ablation)
    observation = public_observation(sample)
    instruction = RULES
    if ablation == 'modify_only':
        instruction += '\nRewrite is disabled in this ablation; its actions must be [].\n'
    elif ablation == 'rewrite_only':
        instruction += '\nModify is disabled in this ablation; its actions must be [].\n'
    if strategy == 'llm':
        instruction += '''Complete all phases in ONE response, in this exact key order:
{"modify":{"reason":"...","actions":[]},"rewrite":{"reason":"...","actions":[]},
"restored_graph":{"nodes":{"0":"full node text"},"edges":[]},
"reason":"reason from your restored graph","answer":"one exact candidate label"}.
The restored_graph is the complete graph YOU construct after both edit phases, including all retained nodes,
full texts, and edges. It must correspond exactly to your actions. No executor feedback will be provided.
The schema above is illustrative, not the current graph. Reason and answer using your restored graph.
'''
    else:
        raise ValueError('Unknown strategy')
    return [{'role': 'system', 'content': instruction},
            {'role': 'user', 'content': 'Visible task and input graph:\n' + json.dumps(observation, ensure_ascii=False)}]


def direct_messages(sample):
    return [{'role': 'system', 'content': 'Reason from the supplied graph without editing it. Return exactly '
             '{"reason":"your reasoning","answer":"one exact candidate label"}.'},
            {'role': 'user', 'content': json.dumps(public_observation(sample), ensure_ascii=False)}]
