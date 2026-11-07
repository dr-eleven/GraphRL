"""3_6四阶段状态机：失败不改图、不重试、继续下一工具；不调用模型。"""
from __future__ import annotations
import json
import re
import hashlib
from copy import deepcopy
from typing import Any


def serialize_graph(graph: dict[str, Any]) -> str:
    nodes = graph.get("nodes", {})
    node_lines = [f"node{node_id}: {text}" for node_id, text in sorted(nodes.items(), key=lambda x: int(x[0]))]
    edge_lines = []
    for edge in sorted(graph.get("edges", []), key=lambda e: (int(e["u"]), int(e["v"]), e.get("relation", ""))):
        line = f"<node{edge['u']}, node{edge['v']}>"
        if edge.get("relation"):
            line += f":{edge['relation']}"
        edge_lines.append(line)
    return "\n".join(node_lines + ["", "Connections:"] + (edge_lines or ["(none)"]))


def _node_id(value: Any) -> int:
    if type(value) is int:
        return value
    match = re.fullmatch(r"\s*node[_ ]?(\d+)\s*", str(value), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid node id: {value!r}")
    return int(match.group(1))


def _edge_key(u: Any, v: Any) -> tuple[int, int]:
    a, b = _node_id(u), _node_id(v)
    return (a, b) if a <= b else (b, a)


def _edge_map(graph: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    result = {}
    for edge in graph.get("edges", []):
        key = _edge_key(edge["u"], edge["v"])
        result[key] = {**edge, "u": key[0], "v": key[1]}
    return result


def execute_actions(sample: dict[str, Any], actions: list[dict[str, Any]], *, text_attacked_nodes=None) -> tuple[dict[str, Any], str | None]:
    """Validate on a private copy. Content-only text actions execute in list order."""
    original = deepcopy(sample["noisy_graph"])
    graph = deepcopy(original)
    nodes = graph["nodes"]
    node_ids = set(map(int, nodes))
    edges = _edge_map(graph)
    protected_nodes = {_node_id(x) for x in sample.get("protected_nodes", [])}
    if sample.get("target_node") is not None:
        protected_nodes.add(_node_id(sample["target_node"]))
    protected_edges = {_edge_key(*x) for x in sample.get("protected_edges", [])}
    # 合法性只使用公开约束；不读取 injected_nodes / clean_graph。
    protected_nodes.update(text_nodes(sample))
    editable = set(text_nodes(sample))
    touched_edges, deleted, edited = set(), set(), set()
    try:
        if not isinstance(actions, list):
            raise ValueError("actions must be a list")
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("Each action must be an object")
            op = str(action.get("op", "")).upper()
            if op in {"ADD_EDGE", "DEL_EDGE"}:
                if "relation" in action:
                    raise ValueError("Edge actions cannot specify a relation label")
                u, v = _node_id(action["u"]), _node_id(action["v"])
                key = _edge_key(u, v)
                if u == v or u not in node_ids or v not in node_ids:
                    raise ValueError("Edge endpoint missing or self-loop")
                if u in deleted or v in deleted:
                    raise ValueError("Edge action conflicts with node deletion")
                if key in protected_edges or key in touched_edges:
                    raise ValueError("Protected, repeated or conflicting edge action")
                if op == "ADD_EDGE":
                    if key in edges:
                        raise ValueError("ADD_EDGE requires an absent edge")
                    edges[key] = {"u":key[0], "v":key[1]}
                else:
                    if key not in edges:
                        raise ValueError("DEL_EDGE requires a present edge")
                    edges.pop(key)
                touched_edges.add(key)
            elif op == "DELETE_NODE":
                node = _node_id(action["node"])
                if node not in node_ids or node in protected_nodes or node in deleted:
                    raise ValueError("Cannot delete a missing, protected or repeated node")
                if node in edited or any(node in edge for edge in touched_edges):
                    raise ValueError("Node deletion conflicts with another action")
                deleted.add(node)
            elif op == "REWRITE":
                node = _node_id(action["node"])
                if node not in editable or node not in node_ids or node in deleted:
                    raise ValueError("Text edits are restricted to NC target or LC query endpoints")
                content = action["text"]
                if not isinstance(content, str) or not content:
                    raise ValueError("Span text must be a nonempty string")
                key = str(node) if str(node) in nodes else node
                current = str(nodes[key])
                if node in edited:
                    raise ValueError("At most one REWRITE per target node")
                normalize = lambda text: " ".join(str(text).split()).casefold()
                if normalize(content) in {normalize(current), normalize(_node_text_value(original, node))}:
                    raise ValueError("REWRITE must change text; copying original/current text is illegal")
                nodes[key] = content
                edited.add(node)
            else:
                raise ValueError(f"Unsupported action: {op!r}")
        for node in deleted:
            nodes.pop(str(node), None)
            nodes.pop(node, None)
        graph["edges"] = [edge for (u,v),edge in edges.items() if u not in deleted and v not in deleted]
        return graph, None
    except (KeyError, TypeError, ValueError) as exc:
        return original, str(exc)


def _node_text_value(graph: dict[str, Any], node: int) -> str:
    nodes = graph.get("nodes", {})
    return str(nodes.get(str(node), nodes.get(node, "")))


def text_nodes(sample: dict[str, Any]) -> list[int]:
    if sample.get("task_type", "").upper() == "LC":
        return list(dict.fromkeys(map(_node_id, sample.get("protected_query_nodes", sample.get("protected_nodes", [])))))
    target = sample.get("target_node")
    return [_node_id(target)] if target is not None else []


TOOL_ORDER = ('del_node', 'del_edge', 'add_edge', 'rewrite')
TOOL_OP = {'del_node': 'DELETE_NODE', 'del_edge': 'DEL_EDGE',
           'add_edge': 'ADD_EDGE', 'rewrite': 'REWRITE'}
ACTION_FIELDS = {'DELETE_NODE': {'op', 'node'}, 'DEL_EDGE': {'op', 'u', 'v'},
                 'ADD_EDGE': {'op', 'u', 'v'}, 'REWRITE': {'op', 'node', 'text'}}


def public_episode(sample):
    """白名单隔离监督：环境和 prompt 只拿任务、当前输入图及公开保护规则。"""
    fields = ('task_type', 'target', 'target_node', 'label_space', 'noisy_graph',
              'protected_nodes', 'protected_edges', 'protected_query_nodes')
    visible = deepcopy({key: sample[key] for key in fields if key in sample})
    visible['protected_nodes'] = sorted(set(map(_node_id, visible.get('protected_nodes', []))) | set(text_nodes(visible)))
    return visible


def first_json_object(text):
    """生成停止条件只看完整 JSON；非法 schema 也停下，交给环境返回错误。"""
    start, depth, quoted, escaped = None, 0, False, False
    for index, char in enumerate(text):
        if start is None:
            if char == '{':
                start, depth = index, 1
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return json.loads(text[start:index + 1])
    raise ValueError('No complete JSON object')


def parse_action_completion(text):
    """保留 3 号 actions 列表，新增前置 analysis 和本轮 tool；严格检查字段。"""
    payload = json.loads(text.strip())
    if not isinstance(payload, dict) or list(payload) != ['analysis', 'tool', 'actions']:
        raise ValueError('Use fields in order: analysis, tool, actions')
    if not isinstance(payload['analysis'], str) or not payload['analysis'].strip():
        raise ValueError('Analyze the current graph before calling the tool')
    if not isinstance(payload['tool'], str) or payload['tool'] not in TOOL_ORDER:
        raise ValueError('Unknown tool')
    actions = payload['actions']
    if not isinstance(actions, list):
        raise ValueError('actions must be a list')
    for action in actions:
        op = TOOL_OP[payload['tool']]
        if not isinstance(action, dict) or set(action) != ACTION_FIELDS[op] or action['op'] != op:
            raise ValueError('Use only the current tool operation and its exact fields')
        for key in ('node', 'u', 'v'):
            if key in action and type(action[key]) is not int:
                raise ValueError('Node IDs must be JSON integers')
        if op == 'REWRITE' and (not isinstance(action['text'], str) or not action['text'].strip()):
            raise ValueError('REWRITE requires nonempty full text')
    return payload


CURRENT_GRAPH_GUIDANCE = (
    'The current valid graph is the latest graph explicitly provided in this conversation; '
    'if no updated graph has been provided, it is the original input graph G0. '
    'Continue the task using ONLY this current valid graph.'
)


def build_restoration_prompt(sample):
    """公开输入与四阶段规则；analysis自然结合结构和语义寻找可能噪声。"""
    sample = public_episode(sample)
    rules = '''You are a Think-with-Graph policy for robust node classification on Text-Attributed Graphs (TAGs).

Given a target node and its subgraph, restore the graph when necessary and predict the target node's class based on the processed graph.

The graph may be clean, contain structural noise, textual noise, or a combination of both. The noise locations and budgets are unknown. Base all restoration decisions on the available evidence.

GRAPH EDITING CONSTRAINTS:
Use existing integer node IDs. Delete only nodes with evidence of injection, never protected/target nodes; deletion also removes incident edges.
No self-loops, missing endpoints, protected query edge edits, repeated/conflicting edge edits within one stage, or relation fields. DEL_EDGE requires a present edge; ADD_EDGE requires an absent edge. An absent edge deleted in the earlier del_edge stage may be added back when both endpoints still exist.
REWRITE only NC target or LC query endpoints, at most once per node, using the complete text. Preserve supported facts; do not invent facts or insert answers, labels, or answer hints. Copying current text or changing only whitespace/case is invalid.

TOOL PROTOCOL:
Call exactly one native tool per turn, once each in order: del_node -> del_edge -> add_edge -> rewrite. No retries or extra tools.
Follow the supplied tool schemas. Arguments must be analysis (nonempty reasoning first) and actions (all edits for this stage). Reason from graph structure and node-text semantics about possible noise; justify actions or actions=[] when no edit is warranted, including on clean graphs. Actions must match the analysis. Never simulate execution or graph feedback.
Action objects use exactly these fields (IDs below are syntax examples):
- del_node: {"op":"DELETE_NODE","node":4}
- del_edge: {"op":"DEL_EDGE","u":1,"v":2}
- add_edge: {"op":"ADD_EDGE","u":0,"v":2}
- rewrite: {"op":"REWRITE","node":0,"text":"Complete restored node text"}
Python returns <restored_graph> only after successful nonempty actions. Empty actions make no graph modifications. If actions are illegal, the entire stage is rejected without changes; Python reports why. Neither empty nor rejected actions repeat the graph; both consume the stage and advance. Ignore the rejected call's analysis and analysis results.
{current_graph_guidance}
Each generation allows at most 512 tokens. Any format error, including truncation at a generation budget without EOS, immediately terminates the trajectory. S, B and H earned in earlier completed stages are retained; the failed stage and unvisited stages earn 0, and T is 0.
After rewrite feedback, stop calling tools and return only {"analysis":"nonempty prediction reasoning","prediction":"one exact candidate label"}, with fields in this order. Base the prediction on the current valid graph, original task, and candidate labels; do not predict before the four stages end.
'''
    rules = rules.replace('{current_graph_guidance}', CURRENT_GRAPH_GUIDANCE)
    visible = dict(task_type=sample['task_type'], target=sample['target'],
                   candidate_labels=sample.get('label_space', []),
                   valid_node_ids=sorted(map(int, sample['noisy_graph']['nodes'])),
                   editable_text_nodes=text_nodes(sample), protected_nodes=sample['protected_nodes'],
                   protected_edges=sample.get('protected_edges', []), current_tool='del_node')
    return rules + '\nCURRENT INPUT\n' + json.dumps(visible, ensure_ascii=False) + '\nCurrent graph G0:\n' + serialize_graph(sample['noisy_graph'])


def graph_hash(graph):
    return hashlib.sha256(serialize_graph(graph).encode()).hexdigest()


class GraphEnvironment:
    """每个候选独立持有 G_t；单轮原子提交，错误不撤销之前成功的阶段。"""
    def __init__(self, sample, max_turns=4):
        if max_turns != len(TOOL_ORDER):
            raise ValueError('Exactly four tool-call opportunities are required')
        self.sample = public_episode(sample)
        self.graph = deepcopy(self.sample['noisy_graph'])
        self.max_turns, self.turns, self.stage, self.version = max_turns, 0, 0, 0
        self.actions, self.events, self.errors = [], [], []
        self.touched_edges = set()
        self.done, self.finish_reason = False, None

    @property
    def current_tool(self):
        return None if self.done else TOOL_ORDER[self.stage]

    def observation(self, **extra):
        return dict(graph_version=self.version, graph_hash=graph_hash(self.graph),
                    current_graph=serialize_graph(self.graph), current_tool=self.current_tool,
                    remaining_turns=self.max_turns-self.turns, editing_finished=self.done,
                    finish_reason=self.finish_reason, **extra)

    def step(self, text, rejection=None):
        """消耗当前机会；仅合法轮原子提交，非法轮不改图但也推进stage。"""
        if self.done:
            raise RuntimeError('Cannot call a finished environment')
        self.turns += 1
        before = graph_hash(self.graph)
        expected = self.current_tool
        call, error, format_valid = None, None, False
        try:
            call = parse_action_completion(text)
            format_valid = rejection is None
            if rejection is not None:
                raise ValueError(rejection)
            if call['tool'] != expected:
                raise ValueError(f'Expected tool: {expected}; this opportunity is consumed')
            actions = call['actions']
            touched = {_edge_key(a['u'], a['v']) for a in actions if a['op'] in {'DEL_EDGE', 'ADD_EDGE'}}
            graph, execution_error = (execute_actions({**self.sample, 'noisy_graph': self.graph}, actions)
                                      if actions else (deepcopy(self.graph), None))
            if execution_error is not None:
                raise ValueError(execution_error)
            # 只有完全合法的当前调用才提交图；隐藏监督不参与执行。
            self.graph = graph
            self.actions.extend(deepcopy(actions))
            self.touched_edges.update(touched)
            self.version += int(bool(actions))
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            self.errors.append(error)
        # 失败同样推进；不留重试机会。第四轮后自动关闭编辑并请求预测。
        self.stage += 1
        if self.turns == self.max_turns:
            self.done, self.finish_reason = True, 'four_stages_complete'
        self.events.append(dict(turn=self.turns, expected_tool=expected, raw=text, call=call,
                                format_valid=format_valid, error=error,
                                before_hash=before, after_hash=graph_hash(self.graph)))
        return self.observation(ok=error is None, error=error)


def load_episodes(path, limit=None):
    from pathlib import Path
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == '.jsonl' else json.loads(text)
    required = {'source_index', 'task_type', 'target', 'noisy_graph', 'clean_graph', 'ground_truth', 'label_space'}
    for row in rows:
        if required - row.keys():
            raise ValueError(f'Missing episode fields: {required-row.keys()}')
    if len({r['source_index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source_index')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    return rows[:limit] if limit is not None else rows


def clean_episode(sample):
    sample = deepcopy(sample)
    sample['noisy_graph'] = deepcopy(sample['clean_graph'])
    return sample
