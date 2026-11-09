"""只用于记录的图/文本恢复指标，绝不参与 B+T 或优势计算。"""
from collections import Counter
from episode import _edge_map, _edge_key, text_nodes, normalize_graph


def prf(pred, gold):
    tp = len(pred & gold)
    p, g = len(pred), len(gold)
    return {'correct': tp, 'wrong': p - tp, 'missed': g - tp,
            'precision': tp / p if p else float(not g), 'recall': tp / g if g else float(not p),
            'f1': 2 * tp / (p + g) if p + g else 1.0}


def structure_actions(sample):
    noisy, clean = sample['noisy_graph'], sample['clean_graph']
    injected = set(noisy['nodes']) - set(clean['nodes'])
    ne, ce = set(_edge_map(noisy)), set(_edge_map(clean))
    result = {('DELETE_NODE', int(n)) for n in injected}
    result |= {('ADD_EDGE', *edge) for edge in ce - ne}
    result |= {('DEL_EDGE', *edge) for edge in ne - ce if not any(str(n) in injected for n in edge)}
    return result


# 仅用于日志分组，不进入模型 Prompt 或执行器；细分 badgraph 中实际存在的噪声。
def noise_components(sample):
    noisy, clean = sample['noisy_graph'], sample['clean_graph']
    ne, ce = set(_edge_map(noisy)), set(_edge_map(clean))
    components = []
    if ne - ce:
        components.append('added_edges')
    if ce - ne:
        components.append('deleted_edges')
    if set(noisy['nodes']) - set(clean['nodes']):
        components.append('injected_nodes')
    if any(noisy['nodes'].get(str(n)) != clean['nodes'].get(str(n)) for n in text_nodes(sample)):
        components.append('target_text')
    return components


def compute_reference_metrics(sample, execution, declared=None, semantic=None):
    clean, noisy, actual = sample['clean_graph'], sample['noisy_graph'], execution['graph']
    values, details = {}, []
    attempted = execution['actions']
    pred = set()
    executed = execution.get('executed_actions', attempted if execution['error'] is None else [])
    if executed:
        for action in executed:
            if action['op'] in {'ADD_EDGE', 'DEL_EDGE'}:
                pred.add((action['op'], *_edge_key(action['u'], action['v'])))
            elif action['op'] == 'DELETE_NODE':
                pred.add(('DELETE_NODE', action['node']))
    values.update({'structure_action_' + k: v for k, v in prf(pred, structure_actions(sample)).items()})
    values['attempted_edit_count'] = len(attempted)
    values['executed_edit_count'] = len(executed)
    values['declared_graph_valid'] = None
    values['declared_execution_consistent'] = None
    graphs = {'executed': actual}
    if declared is not None:
        try:
            declared = normalize_graph(declared)
            graphs['declared'] = declared
            values['declared_graph_valid'] = 1.0
            values['declared_execution_consistent'] = float(declared == normalize_graph(actual))
        except (ValueError, TypeError, KeyError):
            values['declared_graph_valid'] = 0.0
            values['declared_execution_consistent'] = 0.0
    for name, graph in graphs.items():
        ne, ce, ge = set(_edge_map(noisy)), set(_edge_map(clean)), set(_edge_map(graph))
        values[name + '_edge_f1'] = prf(ge, ce)['f1']
        values[name + '_edge_error_reduction'] = len(ne ^ ce) - len(ge ^ ce)
        values[name + '_structure_exact'] = float(ge == ce and set(graph['nodes']) == set(clean['nodes']))
        values[name + '_target_text_exact'] = float(all(graph['nodes'].get(str(n)) == clean['nodes'][str(n)] for n in text_nodes(sample)))
        for node in text_nodes(sample):
            before, after, gold = noisy['nodes'][str(node)], graph['nodes'].get(str(node), ''), clean['nodes'][str(node)]
            # lexical F1 明确标注；可选 BGE 另列，不能把词面重叠冒充语义恢复。
            def lexical(a, b):
                ac, bc = Counter(a.casefold().split()), Counter(b.casefold().split())
                total = sum(ac.values()) + sum(bc.values())
                return 2 * sum((ac & bc).values()) / total if total else 1.0
            item = {'graph': name, 'node': node, 'lexical_before': lexical(before, gold),
                    'lexical_after': lexical(after, gold), 'exact_after': after == gold}
            if semantic is not None:
                item.update(semantic_before=semantic(before, gold), semantic_after=semantic(after, gold))
            details.append(item)
        rows = [d for d in details if d['graph'] == name]
        for metric in ('lexical', 'semantic'):
            if rows and metric + '_before' in rows[0]:
                for state in ('before', 'after'):
                    values[name + '_text_' + metric + '_' + state] = sum(d[metric + '_' + state] for d in rows) / len(rows)
                values[name + '_text_' + metric + '_gain'] = sum(d[metric + '_after'] - d[metric + '_before'] for d in rows) / len(rows)
    return values, details
