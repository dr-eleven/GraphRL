"""R=S+B+T+H: five structure stages and four behavior stages scored independently."""
from copy import deepcopy
from label_utils import _prediction_value, _label_key


def actual_corrupt(sample):
    """与 3 号一致：比较输入与参考图的节点文本和无向边集合，不依赖 budget。"""
    from episode import _edge_map
    noisy, clean = sample['noisy_graph'], sample['clean_graph']
    return noisy['nodes'] != clean['nodes'] or set(_edge_map(noisy)) != set(_edge_map(clean))


def operation_targets(sample, graph, tool, *, actionable=True):
    """Hidden differences, optionally restricted to public executable constraints."""
    from graph_state import _edge_map, text_nodes, _node_text_value, _node_id, _edge_key
    clean = sample['clean_graph']
    nodes, gold_nodes = set(map(int, graph['nodes'])), set(map(int, clean['nodes']))
    protected = {_node_id(n) for n in sample.get('protected_nodes', [])} | set(text_nodes(sample))
    if sample.get('target_node') is not None:
        protected.add(_node_id(sample['target_node']))
    if tool == 'del_node':
        return nodes - gold_nodes - (protected if actionable else set())
    edges, gold = set(_edge_map(graph)), set(_edge_map(clean))
    if tool in ('del_edge', 'add_edge'):
        targets = edges-gold if tool == 'del_edge' else gold-edges
        if actionable:
            blocked = {_edge_key(*edge) for edge in sample.get('protected_edges', [])}
            targets = {e for e in targets if e not in blocked and e[0] != e[1] and set(e) <= nodes}
        return targets
    if tool == 'rewrite':
        return {n for n in text_nodes(sample) if n in nodes & gold_nodes
                and _node_text_value(graph,n) != _node_text_value(clean,n)}
    raise ValueError(tool)


def operation_needed(sample, graph, tool):
    return bool(operation_targets(sample, graph, tool))


def score_episode(sample, environment, prediction_text, *, stage_valid, prediction_valid=False):
    from graph_state import TOOL_ORDER, execute_actions
    completed = environment.finish_reason == 'four_stages_complete' and environment.turns == 4
    legal = not environment.errors and completed
    stage_s = {name: 0.2 if stage_valid.get(name, False) else 0.0
               for name in (*TOOL_ORDER, 'prediction')}
    structure = sum(stage_s.values())
    graph = deepcopy(sample['noisy_graph'])
    stage_b = {name: 0.0 for name in TOOL_ORDER}
    needs = {}
    target_counts = {}
    from hit_reward import stage_hits, empty_hits
    hits = empty_hits()
    for event in environment.events:
        name = event['expected_tool']
        needs[name] = operation_needed(sample, graph, name)
        raw = operation_targets(sample, graph, name, actionable=False)
        possible = operation_targets(sample, graph, name)
        target_counts.update({f'residual_count_{name}': len(raw),
            f'actionable_count_{name}': len(possible), f'unactionable_count_{name}': len(raw-possible)})
        actions = event['call']['actions'] if event.get('call') else []
        if stage_valid.get(name, False) and event['error'] is None:
            stage_b[name] = 0.25 if needs[name] == bool(actions) else 0.0
        # Replay only successful edits. Hidden supervision never controls execution.
        if event['error'] is None:
            before = graph
            graph, error = execute_actions({**environment.sample, 'noisy_graph': graph}, actions)
            if error is not None:
                raise RuntimeError(f'Reward replay differs from execution: {error}')
            if stage_valid.get(name, False):
                values = stage_hits(sample, name, actions, before, graph)
                for key, value in values.items():
                    hits[key] = hits.get(key, 0.0) + value
    behavior = sum(stage_b.values())
    predicted = _label_key(_prediction_value(prediction_text))
    correct = prediction_valid and bool(predicted) and predicted == _label_key(sample['ground_truth'])
    in_candidates = prediction_valid and bool(predicted) and predicted in {_label_key(v) for v in sample['label_space']}
    task = 1.0 if correct else (0.1 if in_candidates else 0.0)
    dirty = actual_corrupt(sample)
    has_actions = bool(environment.actions)  # 实际成功执行的编辑；空actions不算编辑。
    metrics = dict(structure_reward=structure, behavior_reward=behavior,
                   legality_reward=behavior, action_reward=behavior, task_reward=task, prediction_correct=float(correct), prediction_in_candidates=float(in_candidates),
                   total=structure+behavior+task, legal_execution=float(not environment.errors),
                   legal_completion=float(legal), actual_corrupt=float(dirty), has_actions=float(has_actions),
                   correct_clean_noop=float(legal and not dirty and not has_actions),
                   attacked_noop=float(dirty and not has_actions),
                   unnecessary_clean_edit=float(not dirty and has_actions),
                   illegal=float(bool(environment.errors)), finished=float(completed),
                   tool_turns=environment.turns, edit_count=len(environment.actions),
                   empty_actions=float(not environment.actions))
    metrics.update(hits)
    metrics['total'] += hits['hit_reward']
    metrics.update(target_counts)
    metrics.update({f'structure_{name}': value for name, value in stage_s.items()})
    metrics.update({f'behavior_{name}': value for name, value in stage_b.items()})
    metrics.update({f'operation_needed_{name}': float(needs.get(name, False)) for name in TOOL_ORDER})
    metrics.update({f'stage_reached_{name}': float(name in needs) for name in TOOL_ORDER})
    # 参考图在评分端判断输入是否受损；下面的恢复质量指标不额外加分。
    # clean_graph 不传给执行器、prompt 或模型。legality_reward 为原 3 号兼容别名，等于 B。
    if 'clean_graph' in sample:
        from episode import _edge_map
        clean = sample['clean_graph']
        current = environment.graph
        edges, gold = set(_edge_map(current)), set(_edge_map(clean))
        metrics.update(reference_edge_wrong_count=len(edges-gold),
                       reference_edge_missing_count=len(gold-edges),
                       reference_node_wrong_count=len(set(map(int,current['nodes']))-set(map(int,clean['nodes']))),
                       reference_node_missing_count=len(set(map(int,clean['nodes']))-set(map(int,current['nodes']))),
                       reference_text_exact=float(current['nodes'] == clean['nodes']))
    return structure + behavior + task + hits['hit_reward'], metrics
