"""Read full episodes; preserve degree order; replace old prompts and audit lengths."""
import argparse
import hashlib
import json
import inspect
import re
from pathlib import Path
from transformers import AutoTokenizer
from episode import load_episodes, initial_messages, GraphEnvironment, label_map
from budgets import RolloutBudget
from project_config import EXPERIMENT_ID, DEFAULT_MODE
from dataclasses import asdict
from label_utils import _label_key
from trl.chat_template_utils import get_training_chat_template
from prompt_template import single_call_template
from tail_batching import TailSafeRankBatchSampler


def main():
    p = argparse.ArgumentParser()
    for name in ('source','model','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--test-source',action='append',required=True,help='Repeat NAME=PATH for separate test sets; a single PATH is also accepted')
    p.add_argument('--fraction', type=float, default=1)
    p.add_argument('--partition', choices=['prefix','suffix'], default='prefix')
    p.add_argument('--num-generations', type=int, default=6)
    p.add_argument('--num-gpus', type=int, default=3)
    p.add_argument('--train-batch-size', type=int, default=2)
    p.add_argument('--budget-config',help='Optional JSON RolloutBudget overrides')
    p.add_argument('--max-prompt-length', type=int, default=8192)
    p.add_argument('--expected-test-count', type=int, default=600)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f'Refusing existing prepared data directory: {out}')
    if not 0 < a.fraction <= 1:
        raise ValueError('fraction must be in (0,1]')
    test_sources = {}
    for spec in a.test_source:
        name, sep, path = spec.partition('=')
        if not sep:
            name, path = 'test', spec
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name) or name == 'train' or name in test_sources:
            raise ValueError('Test split names must be unique safe identifiers other than train')
        test_sources[name] = path
    train = load_episodes(a.source)
    tests = {name:load_episodes(path) for name,path in test_sources.items()}
    original_count = len(train)
    start, end = ((0,int(len(train)*a.fraction)) if a.partition=='prefix' else
                  (int(len(train)*(1-a.fraction)),len(train)))
    train = train[start:end]
    if not train or any(len(rows)!=a.expected_test_count for rows in tests.values()):
        raise ValueError('Empty training selection or unexpected test coverage')
    sampler = TailSafeRankBatchSampler(len(train),a.num_generations,a.num_gpus,0,a.train_batch_size)
    tokenizer = AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    template = single_call_template(get_training_chat_template(tokenizer) or tokenizer.chat_template)
    overrides = json.loads(Path(a.budget_config).read_text()) if a.budget_config else {}
    budget = RolloutBudget(**{**overrides,'max_initial_prompt_tokens':a.max_prompt_length})
    env = GraphEnvironment(budget=budget)
    tools = [getattr(env,name) for name,fn in inspect.getmembers(type(env),predicate=inspect.isfunction)
             if not name.startswith('_') and name not in ('reset','get_reward')]
    lengths = {}
    for split,rows in [('train',train),*tests.items()]:
        lengths[split] = []
        for row in rows:
            if _label_key(row['ground_truth']) not in label_map(row):
                raise ValueError('ground_truth not in public label space')
            row.pop('oracle_completion',None)
            row['prompt'] = initial_messages(row,budget)
            row['prompt_version'] = EXPERIMENT_ID+'_'+DEFAULT_MODE+'_system_user_shared_tool_rules_v2'
            row['reward_version'] = 'binary_terminal_v1'
            tokenized = tokenizer.apply_chat_template(conversation=[initial_messages(row,budget)],tools=tools,
                        chat_template=template,add_generation_prompt=True,tokenize=True,return_dict=True)
            size = len(tokenized['input_ids'][0])
            if size>a.max_prompt_length:
                raise ValueError(f'{split}/{row["source_index"]}: {size} prompt tokens; no truncation allowed')
            lengths[split].append(size)
    report = dict(experiment_id=EXPERIMENT_ID,mode=DEFAULT_MODE,rollout_budget=asdict(budget),original_train_count=original_count,train_count=len(train),test_counts={name:len(rows) for name,rows in tests.items()},
                  fraction=a.fraction,partition=a.partition,selection=[start,end],source_order_preserved=True,source_indices=[r['source_index'] for r in train],
                  native_tool_names=[tool.__name__ for tool in tools],
                  first_source_indices=[r['source_index'] for r in train[:10]],microbatches_per_rank=len(sampler),
                  max_prompt_tokens={k:max(v) for k,v in lengths.items()},
                  test_sources={name:str(Path(path).resolve()) for name,path in test_sources.items()},
                  test_source_indices={name:[r['source_index'] for r in rows] for name,rows in tests.items()},
                  inputs={str(Path(f).resolve()):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [a.source,*test_sources.values()]})
    out.mkdir(parents=True)
    for name,rows in [('train',train),*tests.items()]:
        (out/f'{name}_episodes.json').write_text(json.dumps(rows,ensure_ascii=False)+'\n')
    (out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'example_prompt.txt').write_text(json.dumps(train[0]['prompt'],ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
