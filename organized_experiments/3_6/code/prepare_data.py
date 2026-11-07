"""Read full episodes; preserve degree order; replace old prompts and audit lengths."""
import argparse
import hashlib
import json
import inspect
from pathlib import Path
from transformers import AutoTokenizer
from episode import load_episodes, initial_messages, GraphEnvironment
from trl.chat_template_utils import get_training_chat_template
from tail_batching import TailSafeRankBatchSampler


def main():
    p = argparse.ArgumentParser()
    for name in ('source','test-source','model','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--fraction', type=float, default=1)
    p.add_argument('--partition', choices=['prefix','suffix'], default='prefix')
    p.add_argument('--num-generations', type=int, default=6)
    p.add_argument('--num-gpus', type=int, default=3)
    p.add_argument('--train-batch-size', type=int, default=2)
    p.add_argument('--max-tool-turns',type=int,default=4)
    p.add_argument('--max-prompt-length', type=int, default=8192)
    p.add_argument('--expected-test-count', type=int, default=600)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f'Refusing existing prepared data directory: {out}')
    if not 0 < a.fraction <= 1:
        raise ValueError('fraction must be in (0,1]')
    train, test = load_episodes(a.source), load_episodes(a.test_source)
    original_count = len(train)
    start, end = ((0,int(len(train)*a.fraction)) if a.partition=='prefix' else
                  (int(len(train)*(1-a.fraction)),len(train)))
    train = train[start:end]
    if not train or len(test)!=a.expected_test_count:
        raise ValueError('Empty training selection or unexpected test coverage')
    sampler = TailSafeRankBatchSampler(len(train),a.num_generations,a.num_gpus,0,a.train_batch_size)
    tokenizer = AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    template = get_training_chat_template(tokenizer) or tokenizer.chat_template
    env = GraphEnvironment(max_turns=a.max_tool_turns)
    tools = [getattr(env,name) for name,fn in inspect.getmembers(type(env),predicate=inspect.isfunction)
             if not name.startswith('_') and name not in ('reset','get_reward')]
    lengths = {}
    for split,rows in [('train',train),('test',test)]:
        lengths[split] = []
        for row in rows:
            row.pop('oracle_completion',None)
            row['prompt'] = initial_messages(row,a.max_tool_turns)[0]['content']
            row['prompt_version'] = '3_6_native_trl_four_stage_no_retry_reasoning'
            row['reward_version'] = '3_6_stage_structure_behavior_hits_v1'
            tokenized = tokenizer.apply_chat_template(conversation=[initial_messages(row,a.max_tool_turns)],tools=tools,
                        chat_template=template,add_generation_prompt=True,tokenize=True,return_dict=True)
            size = len(tokenized['input_ids'][0])
            if size>a.max_prompt_length:
                raise ValueError(f'{split}/{row["source_index"]}: {size} prompt tokens; no truncation allowed')
            lengths[split].append(size)
    report = dict(experiment_id='3_6',original_train_count=original_count,train_count=len(train),test_count=len(test),
                  fraction=a.fraction,partition=a.partition,selection=[start,end],source_order_preserved=True,max_tool_turns=a.max_tool_turns,
                  native_tool_names=[tool.__name__ for tool in tools],
                  first_source_indices=[r['source_index'] for r in train[:10]],microbatches_per_rank=len(sampler),
                  max_prompt_tokens={k:max(v) for k,v in lengths.items()},
                  inputs={str(Path(f).resolve()):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (a.source,a.test_source)})
    out.mkdir(parents=True)
    for name,rows in [('train',train),('test',test)]:
        (out/f'{name}_episodes.json').write_text(json.dumps(rows,ensure_ascii=False)+'\n')
    (out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'example_prompt.txt').write_text(train[0]['prompt'])
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
