"""Epoch 1 reward: four structure checks + four behavior checks + original-noise hits."""
from copy import deepcopy
from graph_state import TOOL_ORDER, execute_actions, _edge_map, text_nodes, _node_text_value, _node_id, _edge_key
from hit_reward import stage_hits, empty_hits


def operation_targets(sample,graph,tool,*,actionable=True):
    clean=sample['clean_graph'];nodes=set(map(int,graph['nodes']));gold_nodes=set(map(int,clean['nodes']))
    protected=set(map(_node_id,sample.get('protected_nodes',[])))|set(text_nodes(sample))
    if sample.get('target_node') is not None:protected.add(_node_id(sample['target_node']))
    if tool=='del_node':return nodes-gold_nodes-(protected if actionable else set())
    if tool in ('del_edge','add_edge'):
        edges,gold=set(_edge_map(graph)),set(_edge_map(clean))
        targets=edges-gold if tool=='del_edge' else gold-edges
        if actionable:
            blocked={_edge_key(*e) for e in sample.get('protected_edges',[])}
            targets={e for e in targets if e not in blocked and e[0]!=e[1] and set(e)<=nodes}
        return targets
    if tool=='rewrite':return {n for n in text_nodes(sample) if n in nodes&gold_nodes and _node_text_value(graph,n)!=_node_text_value(clean,n)}
    raise ValueError(tool)


def score_episode(sample,environment,*,stage_valid):
    graph=deepcopy(sample['noisy_graph']);structure=sum(.2*bool(stage_valid.get(t)) for t in TOOL_ORDER)
    behavior={t:0.0 for t in TOOL_ORDER};hits=empty_hits();metrics={}
    for e in environment.events:
        t=e['expected_tool'];targets=operation_targets(sample,graph,t);acts=(e.get('call') or {}).get('actions',[])
        metrics['operation_needed_'+t]=float(bool(targets));metrics['actionable_count_'+t]=len(targets)
        if e['error'] is None:
            if stage_valid.get(t):behavior[t]=.25*int(bool(targets)==bool(acts))
            before=graph;graph,error=execute_actions({**sample,'noisy_graph':graph},acts)
            if error:raise RuntimeError('Reward replay mismatch: '+error)
            if stage_valid.get(t):
                for k,v in stage_hits(sample,t,acts,before,graph).items():hits[k]=hits.get(k,0)+v
    total=structure+sum(behavior.values())+hits['hit_reward']
    dirty=sample['noisy_graph']['nodes']!=sample['clean_graph']['nodes'] or set(_edge_map(sample['noisy_graph']))!=set(_edge_map(sample['clean_graph']))
    metrics.update(structure_reward=structure,behavior_reward=sum(behavior.values()),total=total,
        actual_corrupt=float(dirty),has_actions=float(bool(environment.actions)),
        legal_execution=float(not environment.errors),legal_completion=float(not environment.errors and environment.turns==4))
    metrics.update(hits)
    metrics.update({'structure_'+t:.2*bool(stage_valid.get(t)) for t in TOOL_ORDER})
    metrics.update({'behavior_'+t:behavior[t] for t in TOOL_ORDER})
    return total,metrics
