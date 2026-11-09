"""同一 NC/LC 测试集上的 MRRA、直接预测和原 GraphSSR 协议评估。"""
import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json

import torch

from config import load_config, resolve, write_json
from prepare_data import load_episodes
from model_backend import Policy
from rollout import run_trajectory
from episode import text_nodes
from label_utils import _label_key


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups['all'].append(row)
        for key in ('dataset', 'task_type', 'attack_kind', 'attack_budget', 'noise_profile'):
            groups[f'{key}={row.get(key)}'].append(row)
    result = {}
    for name, rows in groups.items():
        values = defaultdict(list)
        for row in rows:
            for key, value in row['metrics'].items():
                if isinstance(value, (float, int)):
                    values[key].append(value)
        result[name] = {'count': len(rows), 'means': {k: sum(v) / len(v) for k, v in values.items()},
                        'metric_counts': {k: len(v) for k, v in values.items()}}
    return result


# GraphSSR 的原始 prompt/reward 文件直接保留在本项目内；不修改其采样、选择协议。
# 同一输入图适配为 GraphSSR 格式，基线权重通过 --model 显式提供。
def ssr_trajectory(policy, sample):
    from ssr_prompt import construct_prompt_chosen_reasoning
    from ssr_reward import compute_score, extract_chosen_subgraph_reasoning_info
    graph = sample['noisy_graph']
    nodes = '\n'.join(f'node{key}: {value}' for key, value in graph['nodes'].items())
    edges = ', '.join(f"<node{e['u']}, node{e['v']}>" for e in graph['edges'])
    labels = ', '.join(f'<{label}>' for label in sample['label_space'])
    prompt = construct_prompt_chosen_reasoning(sample['task_type'].lower(),
        ', '.join(f'node{n}' for n in (sample['protected_query_nodes'] if sample['task_type'] == 'LC' else text_nodes(sample))), nodes, edges, labels)
    messages = [{'role': 'user', 'content': prompt}]
    segment = policy.generate(messages, 'ssr', sample=False)
    errors, prediction, score = [], '', {}
    try:
        chosen = extract_chosen_subgraph_reasoning_info(segment.text)
        prediction = chosen[-1]
        score = compute_score(segment.text, sample['ground_truth'], sample['dataset'],
                              {'prompt_str': prompt, 'in_second_stage': False})
    except (ValueError, TypeError, IndexError, KeyError, AttributeError) as exc:
        errors.append(str(exc))
    task = float(_label_key(prediction) == _label_key(sample['ground_truth']))
    return {'sample_id': sample['sample_id'], 'dataset': sample['dataset'], 'task_type': sample['task_type'],
            'attack_kind': sample.get('attack_kind'), 'attack_budget': sample.get('attack_budget'),
            'strategy': 'ssr', 'prediction': prediction, 'ground_truth': sample['ground_truth'],
            'original_graph': graph, 'messages': messages, 'segments': [asdict(segment)],
            'protocol_errors': errors, 'ssr_score': score,
            'metrics': {'task_reward': task, 'prompt_tokens': len(segment.prompt_ids),
                        'generated_tokens': len(segment.completion_ids), 'generation_seconds': segment.elapsed_seconds,
                        'truncated': float(segment.truncated), 'model_calls': 1}}


def evaluate(args):
    cfg = load_config(args.config)
    samples = load_episodes(args.episodes or cfg['data']['test'], limit=args.limit)
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    strategy = args.strategy or cfg['strategy']
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    policy = Policy.load(cfg, device, adapter=resolve(args.adapter) if args.adapter else None, model_path=args.model)
    semantic = None
    if args.semantic or cfg['evaluation']['use_semantic']:
        from semantic_encoder import FrozenTextEncoder
        encoder = FrozenTextEncoder(str(resolve(cfg['evaluation']['semantic_model'])), cfg['evaluation']['semantic_device'])
        semantic = encoder.similarity
    write_json(output / 'evaluation_config.json', {'config': cfg, 'arguments': vars(args),
               'policy_model': args.model or cfg['model']['path'], 'adapter': args.adapter,
               'strategy': strategy, 'condition': args.condition, 'count': len(samples)})
    records = []
    for sample in samples:
        if args.condition == 'clean':
            sample = deepcopy(sample)
            sample['noisy_graph'] = deepcopy(sample['clean_graph'])
        if strategy == 'ssr':
            record = ssr_trajectory(policy, sample)
        else:
            _, record = run_trajectory(policy, sample, strategy, ablation=cfg['ablation'], semantic=semantic)
        record['condition'] = args.condition
        records.append(record)
        with (output / 'predictions.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"{len(records)}/{len(samples)} {sample['sample_id']} T={record['metrics']['task_reward']}", flush=True)
    write_json(output / 'summary.json', summarize(records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes')
    parser.add_argument('--model', help='Base or merged model; SSR baseline requires the desired SSR checkpoint here')
    parser.add_argument('--adapter')
    parser.add_argument('--strategy', choices=['llm', 'executor', 'direct', 'ssr'])
    parser.add_argument('--condition', choices=['noisy', 'clean'], default='noisy')
    parser.add_argument('--device')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--semantic', action='store_true')
    evaluate(parser.parse_args())
