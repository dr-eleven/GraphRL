"""Original-corruption hit rewards. Only called for observed, successfully executed stages."""
import os
from functools import lru_cache

STRUCTURE_WEIGHT = 0.1
REWRITE_WEIGHT = 0.2
REWRITE_MIN_SIMILARITY = 0.85
REWRITE_MIN_GAIN = 0.05


def empty_hits():
    return dict(hit_reward=0.0, **{f'hit_{name}_{suffix}':0.0
        for name in ('del_node','del_edge','add_edge','rewrite') for suffix in ('count','reward')})


@lru_cache(maxsize=1)
def encoder():
    from semantic_encoder import FrozenTextEncoder
    return FrozenTextEncoder(os.environ.get('BGE_M3_MODEL', '/opt/data/private/Graph-R1-main/locals/bge-m3'), 'cpu')


def semantic(a,b):
    if a == b: return 1.0
    if not a.strip() or not b.strip(): return 0.0
    return encoder().similarity(a,b)


def rewrite_result(before, after, clean, similarity=None):
    sim = similarity or semantic
    sb, sa = sim(before,clean), sim(after,clean)
    exact = before != clean and after == clean
    qualifies = before != clean and (exact or (sa >= REWRITE_MIN_SIMILARITY and sa-sb >= REWRITE_MIN_GAIN-1e-12))
    return qualifies, sb, sa


def stage_hits(sample, tool, actions, before, after, similarity=None):
    from graph_state import _edge_map, _node_text_value, _edge_key, text_nodes
    result = empty_hits()
    noisy, clean = sample['noisy_graph'], sample['clean_graph']
    ne, ce = set(_edge_map(noisy)), set(_edge_map(clean))
    be, ae = set(_edge_map(before)), set(_edge_map(after))
    bn, an = set(map(int,before['nodes'])), set(map(int,after['nodes']))
    count = 0
    if tool == 'del_node':
        original = set(map(int,noisy['nodes']))-set(map(int,clean['nodes']))
        count = len({a['node'] for a in actions} & original & (bn-an))
    elif tool in ('del_edge','add_edge'):
        original = ne-ce if tool == 'del_edge' else ce-ne
        changed = be-ae if tool == 'del_edge' else ae-be
        count = len({_edge_key(a['u'],a['v']) for a in actions} & original & changed)
    elif tool == 'rewrite':
        for node in {a['node'] for a in actions} & set(text_nodes(sample)):
            if node not in bn & an or str(node) not in {str(n) for n in clean['nodes']}: continue
            original = _node_text_value(noisy,node)
            truth = _node_text_value(clean,node)
            if original == truth: continue
            qualified,sb,sa = rewrite_result(_node_text_value(before,node),_node_text_value(after,node),truth,similarity)
            count += int(qualified)
            result[f'rewrite_node_{node}_similarity_before'] = sb
            result[f'rewrite_node_{node}_similarity_after'] = sa
            result[f'rewrite_node_{node}_similarity_gain'] = sa-sb
    result[f'hit_{tool}_count'] = float(count)
    result[f'hit_{tool}_reward'] = count*(REWRITE_WEIGHT if tool == 'rewrite' else STRUCTURE_WEIGHT)
    result['hit_reward'] = result[f'hit_{tool}_reward']
    return result
