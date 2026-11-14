"""Public graph constraints and atomic executor inherited from 3_5; scheduling lives in episode.py."""
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
