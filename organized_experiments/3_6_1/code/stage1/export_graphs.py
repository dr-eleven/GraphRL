"""Freeze one greedy terminal graph per source using the final epoch-1 model."""
import argparse
from dataclasses import fields
from functools import partial
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from episode import GraphEnvironment,load_episodes
from train_grpo import GraphRLDataset,GraphRLDataArguments,GraphRLTrainingArguments,ThinkWithGraphTrainer,graph_reward


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--clean',action='store_true')
    a=p.parse_args();data=load_episodes(a.data)
    if a.clean:
        from graph_state import clean_episode
        data=[clean_episode(s) for s in data]
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    args=GraphRLTrainingArguments(output_dir=str(out/'trainer'),per_device_train_batch_size=2,num_generations=2,num_generations_eval=1,
        beta=0,bf16=True,tf32=True,report_to='none',deepspeed=None,use_liger_kernel=False,gradient_checkpointing=False,
        max_tool_calling_iterations=4,max_completion_length=8704,scale_rewards='none')
    ds=GraphRLDataset(data)
    trainer=ThinkWithGraphTrainer(model=model,args=args,processing_class=tok,train_dataset=ds.to_hf_dataset(),
        reward_funcs=[graph_reward],environment_factory=partial(GraphEnvironment,max_turns=4),data_args=GraphRLDataArguments())
    trainer.model.eval();trainer.generation_config.do_sample=False;trainer.generation_config.temperature=1.0;trainer.generation_config.top_p=1.0;trainer.generation_config.top_k=0
    graphs=[];by_id={s['source_index']:s for s in data}
    with (out/'rollouts.jsonl').open('w') as stream:
        for start in range(0,len(ds),a.batch_size):
            batch=[ds[i] for i in range(start,min(start+a.batch_size,len(ds)))]
            trainer._generate_and_score_completions(batch)
            for record in trainer.last_rollout_records:
                s=dict(by_id[record['source_index']]);s['noisy_graph']=record['final_graph']
                # Remove stale prebuilt prompts and oracle actions. Epoch 2 builds fresh input.
                s.pop('prompt',None);s.pop('oracle_completion',None)
                s['stage1_graph_provenance']=dict(model=a.model,source_index=s['source_index'],format_errors=record['format_errors'],
                    finish_reason=record['finish_reason'],selection='single greedy candidate; no reward/label-based selection')
                graphs.append(s);stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            stream.flush();print(f'EXPORT {len(graphs)}/{len(ds)}',flush=True)
    text=json.dumps(graphs,ensure_ascii=False,indent=2)+'\n'
    (out/'prediction_episodes.json').write_text(text)
    (out/'completed.json').write_text(json.dumps(dict(count=len(graphs),source_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
        graphs_sha256=hashlib.sha256(text.encode()).hexdigest(),model=a.model,condition='clean' if a.clean else 'noisy'),indent=2))


if __name__=='__main__':main()
