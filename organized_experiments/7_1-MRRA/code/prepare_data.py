"""保留 degree-sorted 顺序，规范化现有 NC/LC episode，重新构造无隐藏信息的 Prompt。"""
import argparse
import json
from collections import Counter
from copy import deepcopy

from config import load_config, resolve, write_json, digest
from episode import normalize_graph, public_observation, text_nodes, _edge_key


def normalize_episode(source):
    sample = deepcopy(source)
    # 旧 prompt/oracle 不进入新项目的数据，防止无意中复用旧协议或真值轨迹。
    for key in ('prompt', 'oracle_completion', 'clean_target_text', 'noisy_target_text'):
        sample.pop(key, None)
    sample['task_type'] = sample['task_type'].upper()
    if sample['task_type'] not in {'NC', 'LC'}:
        raise ValueError('Only NC/LC are supported')
    targets = list(sample['protected_query_nodes']) if sample['task_type'] == 'LC' else text_nodes(sample)
    if len(targets) != (2 if sample['task_type'] == 'LC' else 1):
        raise ValueError('Invalid task targets')
    sample['protected_nodes'] = sorted(set(targets + sample.get('protected_nodes', [])))
    if sample['task_type'] == 'LC':
        query = _edge_key(*targets)
        sample['protected_edges'] = [list(query)]
    for name in ('noisy_graph', 'clean_graph'):
        graph = normalize_graph(sample[name], allow_duplicate_edges=True)
        if not all(str(node) in graph['nodes'] for node in targets):
            raise ValueError('Target missing from graph')
        if sample['task_type'] == 'LC':
            graph['edges'] = [e for e in graph['edges'] if _edge_key(e['u'], e['v']) != query]
        sample[name] = graph
    if not sample['label_space'] or sample['ground_truth'] not in sample['label_space']:
        raise ValueError('Ground truth is outside candidate labels')
    sample['sample_id'] = str(sample.get('source_id') or f"{sample['dataset']}:{sample['source_index']}")
    public_observation(sample)
    return sample


def load_episodes(path, partition='all', fraction=0.5, limit=None):
    rows = json.loads(resolve(path).read_text())
    cut = int(len(rows) * fraction)
    if partition not in {'all', 'prefix', 'suffix'} or not 0 < fraction < 1:
        raise ValueError('Invalid partition/fraction')
    if partition == 'prefix':
        rows = rows[:cut]
    elif partition == 'suffix':
        rows = rows[cut:]
    if limit is not None:
        rows = rows[:limit]
    samples = [normalize_episode(row) for row in rows]
    ids = [row['sample_id'] for row in samples]
    if not samples or len(set(ids)) != len(ids):
        raise ValueError('Empty dataset or duplicate source IDs')
    return samples


def prepare(cfg, output):
    data = cfg['data']
    train = load_episodes(data['train'], data['partition'], data['fraction'], data.get('limit'))
    test = load_episodes(data['test'])
    if {x['sample_id'] for x in train} & {x['sample_id'] for x in test}:
        raise ValueError('Overlapping train/test source IDs')
    write_json(output / 'train_episodes.json', train)
    write_json(output / 'test_episodes.json', test)
    manifest = {'train_count': len(train), 'test_count': len(test), 'partition': data['partition'],
                'source_train_sha256': digest(resolve(data['train'])), 'source_test_sha256': digest(resolve(data['test'])),
                'ordered_source_ids': [x['sample_id'] for x in train],
                'train_groups': dict(Counter(f"{x['task_type']}/{x.get('attack_kind')}/{x.get('attack_budget')}" for x in train))}
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='data/prepared')
    args = parser.parse_args()
    manifest = prepare(load_config(args.config), resolve(args.output))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'ordered_source_ids'}, ensure_ascii=False, indent=2))
