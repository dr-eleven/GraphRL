"""Evaluate answer-only epoch-2 model on frozen epoch-1 terminal graphs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from train_grpo import GraphRLDataset,GraphRLDataArguments,GraphRLTrainingArguments,ThinkWithGraphTrainer


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    ds=GraphRLDataset(a.data)
    args=GraphRLTrainingArguments(output_dir=str(out/'trainer'),per_device_train_batch_size=2,num_generations=2,num_generations_eval=1,
        beta=0,bf16=True,tf32=True,report_to='none',deepspeed=None,use_liger_kernel=False,gradient_checkpointing=False,
        max_tool_calling_iterations=0,max_completion_length=512,scale_rewards='none')
    trainer=ThinkWithGraphTrainer(model=model,args=args,processing_class=tok,train_dataset=ds.to_hf_dataset(),data_args=GraphRLDataArguments())
    trainer.model.eval();trainer.generation_config.do_sample=False;trainer.generation_config.temperature=1.0;trainer.generation_config.top_p=1.0;trainer.generation_config.top_k=0
    results=[]
    with (out/'predictions.jsonl').open('w') as stream:
        for start in range(0,len(ds),a.batch_size):
            batch=[ds[i] for i in range(start,min(start+a.batch_size,len(ds)))];trainer._generate_and_score_completions(batch)
            for item,r in zip(batch,trainer.last_rollout_records,strict=True):
                result=dict(r,task_type=item['episode']['task_type'],ground_truth=item['episode']['ground_truth']);results.append(result)
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            stream.flush();print(f'EVAL {len(results)}/{len(ds)}',flush=True)
    def summarize(rr):return dict(count=len(rr),accuracy=sum(r['metrics']['prediction_correct'] for r in rr)/len(rr),format_valid=sum(r['metrics']['format_valid'] for r in rr)/len(rr))
    groups=defaultdict(list)
    for r in results:groups[r['task_type']].append(r)
    summary=summarize(results);summary['by_task']={k:summarize(v) for k,v in groups.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
