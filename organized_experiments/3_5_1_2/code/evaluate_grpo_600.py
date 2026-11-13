"""Single-turn evaluation using the same prompt and reward as training."""
import argparse
from collections import defaultdict
from dataclasses import fields
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from episode import load_episodes, clean_episode
from evaluation_metrics import summarize
from train_grpo import (GraphRLDataset, GraphRLDataArguments, GraphRLTrainingArguments,
                        ThinkWithGraphTrainer, graph_reward)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--rollout-config',required=True,help='Training experiment_config.json')
    p.add_argument('--condition',choices=['noisy_policy','clean_policy','noisy_baseline','clean_baseline'],default='noisy_policy')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--expected-count',type=int,default=600)
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    rows=load_episodes(a.data)
    if len(rows)!=a.expected_count or a.batch_size<1:
        raise ValueError('Invalid test coverage/batch size')
    if a.limit is not None:
        if a.limit<1: raise ValueError('limit must be positive')
        rows=rows[:a.limit]
    if a.condition in {'clean_policy','clean_baseline'}:
        rows=[clean_episode(row) for row in rows]
    out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(a.rollout_config).read_text())
    allowed={f.name for f in fields(GraphRLDataArguments)}
    data_args=GraphRLDataArguments(**{k:v for k,v in cfg['data'].items() if k in allowed})
    cuda=torch.cuda.is_available()
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
        dtype=torch.bfloat16 if cuda else torch.float32,attn_implementation='sdpa')
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='left'
    args=GraphRLTrainingArguments(output_dir=str(out/'trainer'),per_device_train_batch_size=2,
        num_generations=2,num_generations_eval=1,gradient_accumulation_steps=1,
        beta=0.0,bf16=cuda,tf32=cuda,use_cpu=not cuda,report_to='none',
        deepspeed=None,use_liger_kernel=False,gradient_checkpointing=False,
        max_completion_length=cfg['training']['max_completion_length'],
        temperature=cfg['training']['temperature'],top_p=cfg['training']['top_p'],
        top_k=cfg['training']['top_k'],scale_rewards='none',
        max_tool_calling_iterations=0)
    dataset=GraphRLDataset(rows)
    trainer=ThinkWithGraphTrainer(model=model,args=args,processing_class=tokenizer,
        train_dataset=dataset.to_hf_dataset(),reward_funcs=[graph_reward],data_args=data_args)
    trainer.model.eval()
    trainer.generation_config.do_sample=False
    trainer.generation_config.temperature=1.0
    trainer.generation_config.top_p=1.0
    trainer.generation_config.top_k=0
    results=[]
    with (out/'predictions.jsonl').open('w') as stream:
        for start in range(0,len(dataset),a.batch_size):
            batch=[dataset[i] for i in range(start,min(start+a.batch_size,len(dataset)))]
            trainer._generate_and_score_completions(batch)
            for item,record in zip(batch,trainer.last_rollout_records,strict=True):
                result=dict(record,task_type=item['episode']['task_type'],condition=a.condition,
                    dataset_task=f"{item['episode'].get('dataset','unknown')}/{item['episode']['task_type']}",
                    ground_truth=item['episode']['ground_truth'])
                results.append(result)
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            stream.flush()
            print(f'{a.condition}: {len(results)}/{len(rows)}',flush=True)
    summary=summarize(results)
    groups=defaultdict(list)
    for row in results: groups[row['task_type']].append(row)
    summary.update(condition=a.condition,development_subset=a.limit is not None,
                   by_task={k:summarize(v) for k,v in groups.items()},
                   by_dataset_and_task={key:summarize([r for r in results if r['dataset_task']==key])
                       for key in sorted({r['dataset_task'] for r in results})})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
