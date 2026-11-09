"""逐样本对齐评估，统计恢复前后正负翻转和 A/B 差异。"""
import argparse
import json
from config import resolve, write_json


def compare(paths, output):
    runs = {}
    for spec in paths:
        name, path = spec.split('=', 1)
        rows = [json.loads(line) for line in resolve(path).read_text().splitlines() if line.strip()]
        runs[name] = {row['sample_id']: row for row in rows}
        if len(rows) != len(runs[name]):
            raise ValueError('Duplicate IDs in ' + name)
    names = list(runs)
    ids = set(runs[names[0]])
    if any(set(run) != ids for run in runs.values()):
        raise ValueError('Comparison requires exactly matching sample IDs')
    pairs = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a, b = runs[first], runs[second]
            positive = [key for key in sorted(ids) if not a[key]['metrics']['task_reward'] and b[key]['metrics']['task_reward']]
            negative = [key for key in sorted(ids) if a[key]['metrics']['task_reward'] and not b[key]['metrics']['task_reward']]
            pairs[f'{first}->{second}'] = {'positive_flips': positive, 'negative_flips': negative,
                                          'accuracy_delta': (len(positive) - len(negative)) / len(ids)}
    write_json(resolve(output), {'count': len(ids), 'pairs': pairs,
        'per_sample': {key: {name: {'prediction': runs[name][key]['prediction'], 'metrics': runs[name][key]['metrics']}
                             for name in names} for key in sorted(ids)}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, help='name=path/to/predictions.jsonl')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    compare(args.run, args.output)
