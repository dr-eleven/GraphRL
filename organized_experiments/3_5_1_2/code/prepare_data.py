"""Preserve degree order, rebuild prompts and preflight both test distributions."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from episode import load_episodes, initial_messages
from trl.chat_template_utils import get_training_chat_template
from tail_batching import TailSafeRankBatchSampler
from task_utils import dataset_statistics


def main():
    p = argparse.ArgumentParser()
    for name in ('source', 'test-source', 'model', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--test-notoneclass-source', required=True, help='Mandatory second test input')
    p.add_argument('--fraction', type=float, default=1)
    p.add_argument('--partition', choices=['prefix', 'suffix'], default='prefix')
    p.add_argument('--num-generations', type=int, default=6)
    p.add_argument('--num-gpus', type=int, default=3)
    p.add_argument('--train-batch-size', type=int, default=2)
    p.add_argument('--gradient-accumulation-steps', type=int, default=2)
    p.add_argument('--max-prompt-length', type=int, default=8192)
    p.add_argument('--expected-test-count', type=int, default=600)
    p.add_argument('--require-test-task-coverage', action='store_true')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f'Refusing existing prepared data directory: {out}')
    if not 0 < a.fraction <= 1:
        raise ValueError('fraction must be in (0,1]')
    if min(a.num_gpus, a.train_batch_size, a.gradient_accumulation_steps) < 1:
        raise ValueError('GPU count, batch and accumulation must be positive')
    if a.num_generations != 6 or a.num_gpus*a.train_batch_size*a.gradient_accumulation_steps != 12:
        raise ValueError('Require six trajectories per source and twelve trajectories/two sources per update')
    train = load_episodes(a.source)
    test_sources = {'oneclass': a.test_source, 'notoneclass': a.test_notoneclass_source}
    tests = {name: load_episodes(path) for name, path in test_sources.items()}
    original_count = len(train)
    start, end = ((0, int(len(train)*a.fraction)) if a.partition == 'prefix' else
                  (int(len(train)*(1-a.fraction)), len(train)))
    train = train[start:end]
    if not train or len(train) % 2:
        raise ValueError('Training selection must contain complete two-source updates')
    train_statistics = dataset_statistics(train)
    test_reports = {}
    for name, rows in tests.items():
        if len(rows) != a.expected_test_count:
            raise ValueError(f'{name}: unexpected test count {len(rows)}')
        statistics = dataset_statistics(rows)
        missing = sorted(set(train_statistics['by_task']) - set(statistics['by_task']))
        if a.require_test_task_coverage and missing:
            raise ValueError(f'{name}: test set does not cover training tasks: ' + ', '.join(missing))
        test_reports[name] = dict(count=len(rows), statistics=statistics, missing_test_tasks=missing,
                                 lp_test_coverage='available' if statistics['by_task'].get('LP') else
                                 'absent: LP accuracy cannot be evaluated with this test set')
    sampler = TailSafeRankBatchSampler(len(train), a.num_generations, a.num_gpus, 0, a.train_batch_size)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    template = get_training_chat_template(tokenizer) or tokenizer.chat_template
    lengths = {}
    splits = {'train': train, **{'test_' + name: rows for name, rows in tests.items()}}
    for split, rows in splits.items():
        lengths[split] = []
        for row in rows:
            row.pop('oracle_completion', None)
            row['prompt'] = initial_messages(row)[0]['content']
            row['prompt_version'] = '3_5_1_2_nc_lc_lp_prediction'
            row['reward_version'] = '3_5_1_2_structure_prediction'
            tokenized = tokenizer.apply_chat_template(conversation=[initial_messages(row)],
                        chat_template=template, add_generation_prompt=True, tokenize=True, return_dict=True)
            size = len(tokenized['input_ids'][0])
            if size > a.max_prompt_length:
                raise ValueError(f'{split}/{row["source_index"]}: {size} prompt tokens; no truncation allowed')
            lengths[split].append(size)
    for name, item in test_reports.items():
        item['max_prompt_tokens'] = max(lengths['test_' + name])
    primary = test_reports['oneclass']
    report = dict(experiment_id='3_5_1_2', original_train_count=original_count,
                  train_count=len(train), test_count=primary['count'],
                  supported_tasks=['NC', 'LC', 'LP'], train_statistics=train_statistics,
                  test_statistics=primary['statistics'], missing_test_tasks=primary['missing_test_tasks'],
                  lp_test_coverage=primary['lp_test_coverage'], test_sets=test_reports,
                  update_batch=dict(num_gpus=a.num_gpus, per_device_batch=a.train_batch_size,
                                    gradient_accumulation_steps=a.gradient_accumulation_steps,
                                    trajectories_per_source=6, trajectories_per_update=12,
                                    sources_per_update=2, optimizer_updates_per_epoch=len(train)//2),
                  fraction=a.fraction, partition=a.partition, selection=[start, end],
                  source_order_preserved=True, prediction_only=True,
                  first_source_indices=[r['source_index'] for r in train[:10]],
                  microbatches_per_rank=len(sampler),
                  max_prompt_tokens={k: max(v) for k, v in lengths.items()},
                  inputs={str(Path(f).resolve()): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                          for f in (a.source, *test_sources.values())})
    out.mkdir(parents=True)
    for name, rows in splits.items():
        (out/f'{name}_episodes.json').write_text(json.dumps(rows, ensure_ascii=False) + '\n')
    # Legacy filename is an alias of oneclass; the launcher always evaluates named files.
    (out/'test_episodes.json').write_bytes((out/'test_oneclass_episodes.json').read_bytes())
    (out/'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    (out/'example_prompt.txt').write_text(train[0]['prompt'])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
