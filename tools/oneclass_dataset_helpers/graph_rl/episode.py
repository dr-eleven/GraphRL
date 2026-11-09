"""Deterministic graph-edit executor and dense restoration rewards for GraphRL."""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from typing import Any


_NODE_RE = re.compile(r"^\s*node(\d+)\s*:\s*(.*?)\s*$", re.MULTILINE)
_EDGE_RE = re.compile(r"<\s*node(\d+)\s*,\s*node(\d+)\s*>(?:\s*:\s*([^,\n<]+))?")


def parse_graph_prompt(prompt: str) -> dict[str, Any]:
    """Parse Graph-R1's nodeN / <nodeN,nodeM> prompt representation."""
    nodes = {int(i): text.strip() for i, text in _NODE_RE.findall(prompt)}
    edges = []
    for u, v, relation in _EDGE_RE.findall(prompt):
        edge = {"u": int(u), "v": int(v)}
        relation = relation.strip()
        if relation:
            edge["relation"] = relation
        edges.append(edge)
    return {"nodes": nodes, "edges": edges}


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
    if isinstance(value, int):
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


def parse_action_completion(text: str) -> dict[str, Any]:
    """Read a JSON object from the model's first-phase completion."""
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth, in_string, escaped = 0, False, False
        for end in range(start, len(text)):
            current = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict) and isinstance(parsed.get("actions"), list):
                        return parsed
                    break
    raise ValueError("Completion does not contain a complete JSON action object")


def execute_actions(sample: dict[str, Any], actions: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
    """Validate a complete action list against the original noisy graph, then execute once."""
    graph = deepcopy(sample["noisy_graph"])
    nodes = graph.get("nodes", {})
    node_ids = {int(i) for i in nodes}
    edges = _edge_map(graph)
    protected_nodes = {_node_id(x) for x in sample.get("protected_nodes", [])}
    protected_edges = {_edge_key(*x) for x in sample.get("protected_edges", [])}
    injected_nodes = {_node_id(x) for x in sample.get("injected_nodes", [])}
    target_node = sample.get("target_node")
    text_edit_nodes = {_node_id(x) for x in sample.get("text_edit_nodes", ([target_node] if target_node is not None else []))}
    if target_node is not None:
        protected_nodes.add(_node_id(target_node))

    add_edges, del_edges, deletes, removes, inserts = [], [], [], [], []
    added_edge_keys, deleted_edge_keys, deleted_nodes = set(), set(), set()
    original_texts = {node: str(nodes.get(str(node), nodes.get(node, ""))) for node in text_edit_nodes}
    remove_ranges_by_node = {node: [] for node in text_edit_nodes}
    add_positions_by_node = {node: [] for node in text_edit_nodes}
    labels = [str(x) for x in sample.get("label_space", [])]
    if sample.get("ground_truth"):
        labels.append(str(sample["ground_truth"]))
    clean_text_for_leak_check = (
        str(sample.get("clean_graph", {}).get("nodes", {}).get(str(target_node), ""))
        if target_node is not None
        else ""
    )

    try:
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("Each action must be an object")
            op = str(action.get("op", "")).upper()
            if op in {"ADD_EDGE", "DEL_EDGE"}:
                if "relation" in action:
                    raise ValueError("Edge actions cannot specify a relation label")
                u, v = _node_id(action["u"]), _node_id(action["v"])
                if u == v or u not in node_ids or v not in node_ids:
                    raise ValueError(f"Illegal {op}: endpoint missing or self-loop")
                key = _edge_key(u, v)
                if key in protected_edges:
                    raise ValueError("Cannot modify a protected query edge")
                if key in added_edge_keys or key in deleted_edge_keys:
                    raise ValueError("Conflicting or repeated edge action")
                if op == "ADD_EDGE":
                    if key in edges:
                        raise ValueError("ADD_EDGE requires a currently absent edge")
                    added_edge_keys.add(key)
                    add_edges.append({"u": key[0], "v": key[1]})
                else:
                    if key not in edges:
                        raise ValueError("DEL_EDGE requires a currently present edge")
                    deleted_edge_keys.add(key)
                    del_edges.append(key)
            elif op == "DELETE_NODE":
                node = _node_id(action["node"])
                if node not in node_ids or node not in injected_nodes:
                    raise ValueError("DELETE_NODE is allowed only for an injected node")
                if node in protected_nodes or node in deleted_nodes:
                    raise ValueError("Cannot delete a protected or already deleted node")
                deleted_nodes.add(node)
                deletes.append(node)
            elif op == "REMOVE_SPAN":
                node = _node_id(action["node"])
                if node not in text_edit_nodes:
                    raise ValueError("Text edits are restricted to NC target or LC query endpoints")
                original_text = original_texts[node]
                remove_ranges = remove_ranges_by_node[node]
                start, end = int(action["start"]), int(action["end"])
                text_value = str(action["text"])
                if start < 0 or end <= start or end > len(original_text) or original_text[start:end] != text_value:
                    raise ValueError("REMOVE_SPAN range/text does not match noisy text")
                if any(start < old_end and old_start < end for old_start, old_end in remove_ranges):
                    raise ValueError("Overlapping REMOVE_SPAN operations")
                remove_ranges.append((start, end))
                removes.append((node, start, end))
            elif op == "ADD_SPAN":
                node = _node_id(action["node"])
                if node not in text_edit_nodes:
                    raise ValueError("Text edits are restricted to NC target or LC query endpoints")
                original_text = original_texts[node]
                remove_ranges = remove_ranges_by_node[node]
                add_positions = add_positions_by_node[node]
                clean_nodes = sample.get("clean_graph", {}).get("nodes", {})
                clean_text_for_leak_check = str(clean_nodes.get(str(node), clean_nodes.get(node, "")))
                position, text_value = int(action["position"]), str(action["text"])
                if position < 0 or position > len(original_text) or not text_value:
                    raise ValueError("ADD_SPAN position/text is invalid")
                if position in add_positions:
                    raise ValueError("Ambiguous multiple ADD_SPAN operations at the same position")
                if any(start < position < end for start, end in remove_ranges):
                    raise ValueError("ADD_SPAN position is inside a removed range")
                if any(
                    label
                    and label.casefold() in text_value.casefold()
                    and label.casefold() not in clean_text_for_leak_check.casefold()
                    for label in labels
                ):
                    raise ValueError("ADD_SPAN contains a task label")
                add_positions.append(position)
                inserts.append((node, position, text_value))
            else:
                raise ValueError(f"Unsupported action: {op!r}")
    except (KeyError, TypeError, ValueError) as exc:
        return graph, str(exc)

    for key in del_edges:
        edges.pop(key, None)
    for edge in add_edges:
        key = _edge_key(edge["u"], edge["v"])
        edges[key] = edge
    for node in deletes:
        nodes.pop(str(node), None)
        nodes.pop(node, None)
        edges = {key: edge for key, edge in edges.items() if node not in key}

    for target_node in text_edit_nodes:
        original_text = original_texts[target_node]
        node_removes = [(start, end) for node, start, end in removes if node == target_node]
        node_inserts = [(pos, value) for node, pos, value in inserts if node == target_node]
        if not node_removes and not node_inserts:
            continue
        chars = []
        removed_at = {start: end for start, end in node_removes}
        inserted_at: dict[int, list[str]] = {}
        for pos, value in node_inserts:
            inserted_at.setdefault(pos, []).append(value)
        i = 0
        while i <= len(original_text):
            chars.extend(inserted_at.get(i, []))
            if i == len(original_text):
                break
            if i in removed_at:
                i = removed_at[i]
            else:
                chars.append(original_text[i])
                i += 1
        restored = "".join(chars)
        if str(target_node) in nodes:
            nodes[str(target_node)] = restored
        else:
            nodes[target_node] = restored

    graph["edges"] = list(edges.values())
    return graph, None


def _f1(predicted: set[Any], truth: set[Any]) -> float:
    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0
    tp = len(predicted & truth)
    precision, recall = tp / len(predicted), tp / len(truth)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _token_f1(pred: str, truth: str) -> float:
    p, t = Counter(pred.casefold().split()), Counter(truth.casefold().split())
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    overlap = sum((p & t).values())
    precision, recall = overlap / sum(p.values()), overlap / sum(t.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _gold_text_spans(noisy: str, clean: str) -> set[tuple[str, int, int, str]]:
    import difflib
    spans = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=noisy, b=clean, autojunk=False).get_opcodes():
        if tag in {"delete", "replace"} and i1 != i2:
            spans.add(("REMOVE_SPAN", i1, i2, noisy[i1:i2]))
        if tag in {"insert", "replace"} and j1 != j2:
            spans.add(("ADD_SPAN", i1, i1, clean[j1:j2]))
    return spans


def score_episode_v1(sample: dict[str, Any], action_text: str, prediction_text: str) -> tuple[float, dict[str, float]]:
    try:
        parsed = parse_action_completion(action_text)
    except (ValueError, json.JSONDecodeError):
        return -1.0, {"illegal": 1.0, "restore": 0.0, "task": 0.0}
    actions = parsed["actions"]
    restored, error = execute_actions(sample, actions)
    if error:
        return -1.0, {"illegal": 1.0, "restore": 0.0, "task": 0.0}

    noisy, clean = sample["noisy_graph"], sample["clean_graph"]
    noisy_edges, restored_edges, clean_edges = _edge_map(noisy), _edge_map(restored), _edge_map(clean)
    del_truth = set(noisy_edges) - set(clean_edges)
    add_truth = set(clean_edges) - set(noisy_edges)
    del_pred = set(noisy_edges) - set(restored_edges)
    add_pred = set(restored_edges) - set(noisy_edges)
    structure = 0.5 * _f1(del_pred, del_truth) + 0.5 * _f1(add_pred, add_truth)

    target = sample.get("target_node")
    if target is None:
        text_reward = 1.0
        span_reward = 1.0
        final_text_reward = 1.0
    else:
        target = str(target)
        clean_text = str(clean.get("nodes", {}).get(target, clean.get("nodes", {}).get(int(target), "")))
        noisy_text = str(noisy.get("nodes", {}).get(target, noisy.get("nodes", {}).get(int(target), "")))
        restored_text = str(restored.get("nodes", {}).get(target, restored.get("nodes", {}).get(int(target), "")))
        gold_spans = _gold_text_spans(noisy_text, clean_text)
        predicted_spans = set()
        for a in actions:
            op = str(a.get("op", "")).upper()
            if op == "REMOVE_SPAN":
                predicted_spans.add((op, int(a["start"]), int(a["end"]), str(a["text"])))
            elif op == "ADD_SPAN":
                pos = int(a["position"])
                predicted_spans.add((op, pos, pos, str(a["text"])))
        span_reward = _f1(predicted_spans, gold_spans)
        exact = float(restored_text == clean_text)
        final_text_reward = 0.7 * _token_f1(restored_text, clean_text) + 0.3 * exact
        text_reward = 0.4 * span_reward + 0.6 * final_text_reward

    injected_truth = {_node_id(x) for x in sample.get("injected_nodes", [])}
    deleted_pred = {_node_id(a["node"]) for a in actions if str(a.get("op", "")).upper() == "DELETE_NODE"}
    node_reward = _f1(deleted_pred, injected_truth)

    restore_reward = 0.50 * text_reward + 0.35 * structure + 0.15 * node_reward
    task = float(_prediction_matches(prediction_text, str(sample["ground_truth"])))
    noisy_target_text = str(noisy.get("nodes", {}).get(str(target), "")) if target is not None else ""
    clean_target_text = str(clean.get("nodes", {}).get(str(target), "")) if target is not None else ""
    gold_text_spans = _gold_text_spans(noisy_target_text, clean_target_text)
    required_actions = {("DEL_EDGE", key) for key in del_truth}
    required_actions.update(("ADD_EDGE", key) for key in add_truth)
    required_actions.update(("DELETE_NODE", node) for node in injected_truth)
    required_actions.update(gold_text_spans)

    predicted_actions = set()
    for action in actions:
        op = str(action.get("op", "")).upper()
        if op in {"ADD_EDGE", "DEL_EDGE"}:
            predicted_actions.add((op, _edge_key(action["u"], action["v"])))
        elif op == "DELETE_NODE":
            predicted_actions.add((op, _node_id(action["node"])))
        elif op == "REMOVE_SPAN":
            predicted_actions.add((op, int(action["start"]), int(action["end"]), str(action["text"])))
        elif op == "ADD_SPAN":
            position = int(action["position"])
            predicted_actions.add((op, position, position, str(action["text"])))
    unnecessary_actions = len(predicted_actions - required_actions)
    minimal_penalty = 0.05 * min(1.0, unnecessary_actions / max(1, len(required_actions) + 1))
    total = restore_reward + restore_reward * task - minimal_penalty
    return total, {
        "illegal": 0.0,
        "text": text_reward,
        "span": span_reward,
        "text_final": final_text_reward,
        "structure": structure,
        "node": node_reward,
        "restore": restore_reward,
        "task": task,
        "minimal_penalty": minimal_penalty,
        "total": total,
    }


def _node_text_value(graph: dict[str, Any], node: int) -> str:
    nodes = graph.get("nodes", {})
    return str(nodes.get(str(node), nodes.get(node, "")))


def parse_candidate_labels(prompt: str) -> list[str]:
    match = re.search(r"(?:Select strictly from|Choose from the following):\s*(.*?)\.\s*Respond only", prompt, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("Missing candidate-label list in task prompt")
    raw = match.group(1)
    labels = [m.group(2) for m in re.finditer(r"(['\"])(.*?)\1(?=\s*(?:[,;]|$))", raw)]
    if not labels:
        labels = raw.split(";")
    return [label.strip().replace("Theroy", "Theory") for label in labels if label.strip()]


def text_nodes(sample: dict[str, Any]) -> list[int]:
    if sample.get("task_type", "").upper() == "LC":
        return list(dict.fromkeys(map(_node_id, sample.get("protected_query_nodes", sample.get("protected_nodes", [])))))
    target = sample.get("target_node")
    return [_node_id(target)] if target is not None else []


def _text_distance(a: str, b: str) -> float:
    # Sequence similarity preserves character order and punctuation.
    import difflib
    return 1.0 - difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def oracle_actions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Training-only reference restoration; never exposed in inference prompts."""
    noisy, clean = sample["noisy_graph"], sample["clean_graph"]
    ne, ce = _edge_map(noisy), _edge_map(clean)
    injected = set(map(_node_id, sample.get("injected_nodes", [])))
    actions = [{"op": "DELETE_NODE", "node": node} for node in sorted(injected)]
    for u, v in sorted(ne.keys() - ce.keys()):
        if u not in injected and v not in injected:
            actions.append({"op": "DEL_EDGE", "u": u, "v": v})
    for u, v in sorted(ce.keys() - ne.keys()):
        actions.append({"op": "ADD_EDGE", "u": u, "v": v})
    for node in text_nodes(sample):
        spans = _gold_text_spans(_node_text_value(noisy, node), _node_text_value(clean, node))
        for op, start, end, text in sorted(spans, key=lambda x: (x[1], x[0])):
            if op == "REMOVE_SPAN":
                actions.append({"op": op, "node": node, "start": start, "end": end, "text": text})
            else:
                actions.append({"op": op, "node": node, "position": start, "text": text})
    return actions


def score_episode(sample: dict[str, Any], action_text: str, prediction_text: str) -> tuple[float, dict[str, float]]:
    """V2: improvement over the noisy state, with no credit for absent corruption."""
    try:
        actions = parse_action_completion(action_text)["actions"]
        restored, error = execute_actions(sample, actions)
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        error = str(exc)
    if error:
        noisy, clean = sample['noisy_graph'], sample['clean_graph']
        corrupted = (set(_edge_map(noisy)) != set(_edge_map(clean))
                     or bool(sample.get('injected_nodes'))
                     or any(_node_text_value(noisy, n) != _node_text_value(clean, n) for n in text_nodes(sample)))
        return -1.0, {"illegal": 1.0, "restore": 0.0, "restore_progress": 0.0,
                      "task_gate": 0.0, "task": 0.0, "total": -1.0,
                      "empty_actions": 0.0, "corrupted": float(corrupted)}
    noisy, clean = sample["noisy_graph"], sample["clean_graph"]
    ne, ce, re = set(_edge_map(noisy)), set(_edge_map(clean)), set(_edge_map(restored))
    edge_initial, edge_final = len(ne ^ ce), len(re ^ ce)
    injected = set(map(_node_id, sample.get("injected_nodes", [])))
    remaining = injected & set(map(int, restored["nodes"]))
    node_initial, node_final = len(injected), len(remaining)
    clip = lambda x: max(-1.0, min(1.0, x))
    structure = clip(1.0 - edge_final / edge_initial) if edge_initial else 0.0
    node_score = clip(1.0 - node_final / node_initial) if node_initial else 0.0
    text_progress, text_preservation_losses = [], []
    final_text_scores = []
    for node in text_nodes(sample):
        c, n, r = [_node_text_value(g, node) for g in (clean, noisy, restored)]
        initial, final = _text_distance(n, c), _text_distance(r, c)
        final_text_scores.append(1.0 - final)
        if n != c:
            text_progress.append(clip((initial - final) / max(initial, 1e-8)))
        elif r != c:
            text_preservation_losses.append(final)
    text_score = sum(text_progress) / len(text_progress) if text_progress else 0.0
    values = {"text": text_score, "structure": structure, "node": node_score}
    active = {"text": bool(text_progress), "structure": bool(edge_initial), "node": bool(node_initial)}
    weights = {"text": 0.50, "structure": 0.35, "node": 0.15}
    task = float(_prediction_matches(prediction_text, str(sample["ground_truth"])))
    preservation_damage = max(text_preservation_losses, default=0.0)
    if not edge_initial and edge_final:
        preservation_damage = max(preservation_damage, min(1.0, edge_final / max(len(ce), 1)))
    if any(active.values()):
        weight_sum = sum(weights[k] for k in values if active[k])
        progress = clip(sum(weights[k] * values[k] for k in values if active[k]) / weight_sum - preservation_damage)
        gate = max(0.0, min(values[k] for k in values if active[k])) * (1.0 - preservation_damage)
        # Unchanged corrupt observation receives -0.25; full restoration receives 1.
        restoration = -0.25 + 1.25 * progress
        total = restoration + 0.5 * gate * task
    else:
        progress = 1.0 - preservation_damage
        gate = 1.0 - preservation_damage
        restoration = 0.2 - 1.2 * preservation_damage
        total = restoration + 0.1 * gate * task
    return total, {
        "illegal": 0.0, "text": text_score, "structure": structure, "node": node_score,
        "text_final": sum(final_text_scores) / max(len(final_text_scores), 1),
        "restore": restoration, "restore_progress": progress, "task_gate": gate,
        "task": task, "minimal_penalty": preservation_damage, "total": total,
        "empty_actions": float(not actions), "corrupted": float(any(active.values())),
    }


def _prediction_matches(prediction: str, gold: str) -> bool:
    pred = prediction.strip()
    match = re.search(r'"prediction"\s*:\s*"([^"]+)"', pred, flags=re.IGNORECASE)
    if match:
        pred = match.group(1)
    else:
        match = re.search(r"Answer\s*:\s*(.+)", pred, flags=re.IGNORECASE)
        if match:
            pred = match.group(1).splitlines()[0]
    clean = lambda x: re.sub(r"\s+", " ", x).strip(" \t\r\n\"'.,").casefold()
    return clean(pred) == clean(gold)


def build_restoration_prompt(sample: dict[str, Any]) -> str:
    return (
        "You are a Think-with-Graph restoration policy for robust reasoning over a noisy Text-Attributed Graph.\n\n"
        "Given one noisy task subgraph, analyze its graph structure, target-node text, and possible injected nodes, then output the complete set of restoration operations in a single response.\n\n"
        "The subgraph may contain any combination of the following attacks, or it may already be clean:\n\n"
        "1. adversarial edges were added;\n"
        "2. clean edges were deleted;\n"
        "3. adversarial text spans were inserted into the target-node text;\n"
        "4. clean text spans were removed from the target-node text;\n"
        "5. adversarial nodes and their connections were injected.\n\n"
        "You are not given the attack type, attacked positions, or attack budget.\n\n"
        "The only allowed restoration operations are:\n\n"
        "- ADD_EDGE(u, v)\n"
        "- DEL_EDGE(u, v)\n"
        "- REMOVE_SPAN(node, start, end, text)\n"
        "- ADD_SPAN(node, position, text)\n"
        "- DELETE_NODE(node)\n\n"
        "Requirements:\n\n"
        "1. Decide all restoration operations jointly in this single response.\n"
        "2. Do not perform iterative graph editing.\n"
        "3. Text operations must make localized edits to the provided noisy text. For NC edit only its target node; for LC edit either query endpoint. Do not rewrite the entire text.\n"
        "4. If there is insufficient evidence that an edge, text span, or node is corrupted, keep it unchanged.\n"
        "5. Use the minimum necessary edits to recover a coherent, task-relevant clean graph.\n"
        "6. Do not insert class labels, relation labels, task answers, or other explicit task-leakage hints into the text.\n"
        "7. If the graph is already clean, return an empty action list.\n"
        "8. For link classification, any protected query edge or protected query nodes must not be modified or deleted.\n"
        "9. All text-span positions must refer to the original noisy text, not to an intermediate edited version.\n"
        "10. The complete action list will be checked for legality before execution.\n\n"
        f"Task type:\n{sample['task_type']}\n\n"
        f"Task target:\n{sample['target']}\n\n"
        f"Nodes allowed for localized text edits:\n{json.dumps(sample.get('text_edit_nodes', []))}\n\n"
        f"Candidate labels:\n{json.dumps(sample.get('label_space', []), ensure_ascii=False)}\n\n"
        f"Protected query edge / nodes:\n{json.dumps({'edges': sample.get('protected_edges', []), 'nodes': sample.get('protected_nodes', [])}, ensure_ascii=False)}\n\n"
        f"Noisy task subgraph:\n{serialize_graph(sample['noisy_graph'])}\n\n"
        "Return only valid JSON. Each action must be a JSON object, never a quoted function-call string. "
        "Use this schema and replace the example values with your own:\n\n"
        '{"diagnosis":{"structure":"brief evidence summary","text":"brief evidence summary",'
        '"nodes":"brief evidence summary"},"actions":[\n'
        '{"op":"ADD_EDGE","u":"node_1","v":"node_3"},\n'
        '{"op":"DEL_EDGE","u":"node_1","v":"node_7"},\n'
        '{"op":"REMOVE_SPAN","node":"node_1","start":35,"end":52,"text":"adversarial inserted text"},\n'
        '{"op":"ADD_SPAN","node":"node_1","position":35,"text":"restored clean text"},\n'
        '{"op":"DELETE_NODE","node":"node_9"}\n]}\n\n'
        "Use an empty actions array when no edit is needed. Return no prose outside this JSON object."
    )


def build_prediction_prompt(sample: dict[str, Any], restored_graph: dict[str, Any]) -> str:
    return (
        "The graph restoration step has been completed.\n\n"
        "Below is the actual restored graph produced by executing your restoration actions.\n\n"
        "Based on the graph you restored, derive the final answer for the graph task.\n\n"
        "Do not modify the graph again. Only solve the task using the restored graph.\n\n"
        f"Task type:\n{sample['task_type']}\n\n"
        f"Task target:\n{sample['target']}\n\n"
        f"Candidate labels:\n{json.dumps(sample.get('label_space', []), ensure_ascii=False)}\n\n"
        f"Restored graph:\n{serialize_graph(restored_graph)}\n\n"
        "Select exactly one answer from the candidate labels and return it as the final prediction. "
        "Do not provide chain-of-thought, analysis, justification, markdown, or any text before or after the JSON. "
        "Your entire response must be one valid JSON object in this exact format:\n\n"
        '{\n  "prediction": "FINAL_LABEL"\n}'
    )
