"""Shared graph serialization and data loading copied from 3_5; no graph editing."""
import json
from copy import deepcopy
from typing import Any
from task_utils import prepare_episode

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


def load_episodes(path, limit=None):
    from pathlib import Path
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == '.jsonl' else json.loads(text)
    required = {'source_index', 'task_type', 'target', 'noisy_graph', 'clean_graph', 'ground_truth', 'label_space'}
    for row in rows:
        if required - row.keys():
            raise ValueError(f'Missing episode fields: {required-row.keys()}')
        if type(row['source_index']) is not int or row['source_index']<0:
            raise ValueError('source_index must be a nonnegative integer')
    rows = [prepare_episode(row) for row in rows]
    if len({r['source_index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source_index')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    return rows[:limit] if limit is not None else rows


def clean_episode(sample):
    sample = prepare_episode(sample)
    sample['noisy_graph'] = deepcopy(sample['clean_graph'])
    return sample
