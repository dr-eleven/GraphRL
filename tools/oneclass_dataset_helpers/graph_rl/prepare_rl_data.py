"""Build GraphRL episodes from the saved clean/BadGraph group assignment."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

from episode import _edge_key, _gold_text_spans, build_restoration_prompt, parse_graph_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_metadata(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _label_space(prompt: str) -> list[str]:
    match = re.search(r"Select strictly from:\s*(.*?)\.\s*Respond only", prompt, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    return re.findall(r"'([^']+)'", match.group(1))


def _target(prompt: str, task: str) -> tuple[str, int | None, list[int]]:
    if task.lower() == "lc":
        match = re.search(r"denoted as node(\d+)\s+and\s+node(\d+)", prompt, flags=re.IGNORECASE)
        if not match:
            raise ValueError("Could not find LC query nodes in prompt")
        pair = [int(match.group(1)), int(match.group(2))]
        return f"query pair node{pair[0]}, node{pair[1]}", None, pair
    match = re.search(r"represented by node(\d+)", prompt, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"target(?: node)?\s*[:=]?\s*node(\d+)", prompt, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Could not find NC target node in prompt")
    node = int(match.group(1))
    return f"node{node}", node, [node]


def _load_badgraph_data(root: Path, budget: int) -> tuple[list[dict], list[dict]]:
    attack_dir = root / "badgraph_attack_train_data" / f"budget{budget}" / "badgraph"
    parquet_path = next(attack_dir.glob("*.parquet"))
    metadata_path = next(attack_dir.glob("*_metadata.jsonl"))
    return pq.read_table(parquet_path).to_pylist(), _load_metadata(metadata_path)


def _graph_edges(graph: dict) -> dict[tuple[int, int], dict]:
    return {_edge_key(edge["u"], edge["v"]): edge for edge in graph.get("edges", [])}


def _node_text(graph: dict, node: int | None) -> str:
    if node is None:
        return ""
    nodes = graph.get("nodes", {})
    return str(nodes.get(str(node), nodes.get(node, "")))


def make_episode(clean: dict, noisy_prompt: str, metadata: dict, attack_kind: str, source_index: int) -> dict:
    task = str(clean["task"]).lower()
    target, target_node, query_nodes = _target(clean["prompt"], task)
    clean_graph = parse_graph_prompt(clean["prompt"])
    noisy_graph = parse_graph_prompt(noisy_prompt)
    if task == "lc":
        # Match the RL_eng target definition and prevent the query relation from
        # leaking the answer through either the clean target or noisy observation.
        query_key = _edge_key(*query_nodes)
        clean_graph["edges"] = [edge for edge in clean_graph["edges"] if _edge_key(edge["u"], edge["v"]) != query_key]
        noisy_graph["edges"] = [edge for edge in noisy_graph["edges"] if _edge_key(edge["u"], edge["v"]) != query_key]

    injected_nodes = sorted(set(map(int, noisy_graph["nodes"])) - set(map(int, clean_graph["nodes"])))
    protected_edges = [query_nodes] if task == "lc" else []
    changes = metadata.get("text_changes") or []
    if target_node is None and changes:
        candidate = int(changes[0].get("node", query_nodes[0]))
        if candidate in noisy_graph["nodes"] or str(candidate) in noisy_graph["nodes"]:
            target_node = candidate

    clean_target_text = _node_text(clean_graph, target_node)
    noisy_target_text = _node_text(noisy_graph, target_node)
    gold_spans = _gold_text_spans(noisy_target_text, clean_target_text)
    removed_spans = [
        {"start": start, "end": end, "text": text}
        for op, start, end, text in sorted(gold_spans)
        if op == "REMOVE_SPAN"
    ]
    added_spans = [
        {"position": start, "text": text}
        for op, start, _end, text in sorted(gold_spans)
        if op == "ADD_SPAN"
    ]

    clean_edges = _graph_edges(clean_graph)
    noisy_edges = _graph_edges(noisy_graph)
    added_attack_edges = [noisy_edges[key] for key in sorted(noisy_edges.keys() - clean_edges.keys())]
    deleted_clean_edges = [clean_edges[key] for key in sorted(clean_edges.keys() - noisy_edges.keys())]

    sample = {
        "source_index": source_index,
        "dataset": clean.get("dataset", metadata.get("dataset", "unknown")),
        "task_type": "LC" if task == "lc" else "NC",
        "target": target,
        "target_node": target_node,
        "label_space": _label_space(clean["prompt"]),
        "ground_truth": clean.get("ground_truth", metadata.get("ground_truth", "")),
        "ground_truth_label": clean.get("ground_truth", metadata.get("ground_truth", "")),
        "clean_graph": clean_graph,
        "noisy_graph": noisy_graph,
        "clean_target_text": clean_target_text,
        "noisy_target_text": noisy_target_text,
        "added_attack_edges": added_attack_edges,
        "deleted_clean_edges": deleted_clean_edges,
        "inserted_text_spans": removed_spans,
        "deleted_clean_text_spans": added_spans,
        "injected_nodes": injected_nodes,
        "protected_query_edge": query_nodes if task == "lc" else None,
        "protected_query_nodes": query_nodes,
        # Compatibility aliases used by the executor and prompt builder.
        "protected_nodes": query_nodes,
        "protected_edges": protected_edges,
        "attack_kind": attack_kind,
        "attack_budget": int(metadata.get("budget", 0)),
        "attack_status": metadata.get("status", "clean"),
    }
    sample["prompt"] = build_restoration_prompt(sample)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "train_data_5k")
    parser.add_argument("--assignment", type=Path, default=PROJECT_ROOT / "GraphRL_train_data" / "group_indices.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "GraphRL_train_data" / "train_grpo_episodes.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = pq.read_table(args.data_dir / "train_sft_data_badgraph_input.parquet").to_pylist()
    assignment = json.loads(args.assignment.read_text(encoding="utf-8"))
    expected_groups = {"clean", *(f"badgraph_budget_{budget}" for budget in range(1, 6))}
    if set(assignment) != expected_groups:
        raise ValueError(f"Assignment must contain exactly these groups: {sorted(expected_groups)}")
    flattened = [int(index) for indices in assignment.values() for index in indices]
    if len(flattened) != len(source) or len(set(flattened)) != len(source) or set(flattened) != set(range(len(source))):
        raise ValueError("Assignment indices must partition every source row exactly once")

    episodes_by_index: dict[int, dict] = {}
    clean_indices = set(map(int, assignment["clean"]))
    for index in sorted(clean_indices):
        clean = source[index]
        meta = {
            "source_index": index,
            "dataset": clean.get("dataset"),
            "task": clean.get("task"),
            "ground_truth": clean.get("ground_truth"),
            "budget": 0,
            "status": "clean",
        }
        episodes_by_index[index] = make_episode(clean, clean["prompt"], meta, "clean", index)

    for budget in range(1, 6):
        attacked_rows, metadata_rows = _load_badgraph_data(args.data_dir, budget)
        if len(attacked_rows) != len(source) or len(metadata_rows) != len(source):
            raise ValueError(f"BadGraph budget {budget} does not cover all source rows")
        selected = sorted(map(int, assignment[f"badgraph_budget_{budget}"]))
        for index in selected:
            metadata = metadata_rows[index]
            if int(metadata.get("source_index", index)) != index:
                raise ValueError(f"BadGraph metadata misaligned at source row {index}",)
            episodes_by_index[index] = make_episode(
                source[index], attacked_rows[index]["prompt"], metadata, "badgraph", index
            )

    episodes = list(episodes_by_index.values())
    random.Random(args.seed).shuffle(episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(episodes, ensure_ascii=False), encoding="utf-8")
    summary = {
        "episodes": len(episodes),
        "groups": {
            name: sum(episode["attack_kind"] == "clean" if name == "clean" else episode["attack_budget"] == int(name.rsplit("_", 1)[1]) for episode in episodes)
            for name in sorted(expected_groups)
        },
        "task_counts": {task: sum(episode["task_type"] == task for episode in episodes) for task in ("NC", "LC")},
        "assignment": str(args.assignment),
        "output": str(args.output),
        "seed": args.seed,
    }
    (args.output.parent / "train_grpo_episodes_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
