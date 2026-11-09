"""离线检查全量数据、源覆盖和真实 tokenizer 的初始 Prompt 长度，不加载模型权重。"""
import argparse
from collections import Counter

from transformers import AutoTokenizer

from config import load_config, resolve, write_json
from prepare_data import load_episodes
from episode import first_messages
from reference_metrics import noise_components


def check(cfg):
    tokenizer = AutoTokenizer.from_pretrained(str(resolve(cfg['model']['path'])), local_files_only=True)
    result = {}
    for split, path in [('train', cfg['data']['train']), ('test', cfg['data']['test'])]:
        samples = load_episodes(path)
        stats = {}
        for strategy in ('llm', 'executor'):
            counts = [len(tokenizer.apply_chat_template(first_messages(s, strategy), tokenize=True, add_generation_prompt=True, return_dict=False)) for s in samples]
            stats[strategy] = {'max_prompt_tokens': max(counts), 'mean_prompt_tokens': sum(counts) / len(counts),
                'over_limit': [s['sample_id'] for s, count in zip(samples, counts) if count > cfg['generation']['prompt_tokens']]}
        result[split] = {'count': len(samples), 'prompts': stats,
                         'noise_profiles': dict(Counter('+'.join(noise_components(s)) or 'clean' for s in samples))}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = check(load_config(args.config))
    write_json(resolve(args.output), result)
    print(result)
    if any(row['over_limit'] for split in result.values() for row in split['prompts'].values()):
        raise SystemExit('Prompt budget exceeded; inspect report and explicitly adjust shared A/B configuration')
