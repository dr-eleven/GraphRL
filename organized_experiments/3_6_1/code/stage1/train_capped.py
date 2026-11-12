"""Single-GPU 3_6 with strict full-sequence exclusion and genuine ZeRO-2 resume."""
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import TrainerCallback

RUN = Path(os.environ['CAP_RUN'])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_grpo as base

torch.cuda.set_per_process_memory_fraction(float(os.environ.get('CAP_MEMORY_FRACTION','0.95')),0)

@dataclass
class CappedDataArguments(base.GraphRLDataArguments):
    max_total_sequence_length: int = 5120

base.GraphRLDataArguments = CappedDataArguments

_original_graph_reward=base.graph_reward
def graph_reward(completions,environments,**kwargs):
    values=_original_graph_reward(completions,environments,**kwargs)
    # Environment._score rebuilds its audit dictionary, so merge exclusion metadata afterwards.
    for env in environments:env._record.update(getattr(env,'_cap_discard_audit',{}))
    return values
base.graph_reward=graph_reward

class EntireBatchDiscarded(Exception):
    pass

class ExclusionAwareScheduler:
    def __init__(self,scheduler,trainer):
        self.scheduler=scheduler;self.trainer=trainer
    def step(self,*args,**kwargs):
        if not self.trainer._cap_skip_optimizer:
            return self.scheduler.step(*args,**kwargs)
    def state_dict(self):return self.scheduler.state_dict()
    def load_state_dict(self,state):return self.scheduler.load_state_dict(state)
    def __getattr__(self,name):return getattr(self.scheduler,name)

def excluded_groups(lengths, marked, group_size, limit):
    """Preserve complete GRPO groups and bound separately padded prompt/completion widths."""
    excluded = set()
    for start in range(0, len(lengths), group_size):
        group = range(start, min(start+group_size,len(lengths)))
        if any(i in marked or sum(lengths[i]) >= limit for i in group):
            excluded.update(group)
    while True:
        kept = [i for i in range(len(lengths)) if i not in excluded]
        if not kept:
            return excluded
        prompt_width = max(lengths[i][0] for i in kept)
        offenders = [i for i in kept if prompt_width+lengths[i][1] >= limit]
        if not offenders:
            return excluded
        for i in offenders:
            start = i//group_size*group_size
            excluded.update(range(start,min(start+group_size,len(lengths))))

class RuntimeAudit(TrainerCallback):
    def __init__(self,trainer): self.trainer=trainer
    def on_train_begin(self,args,state,control,**kwargs):
        t=self.trainer
        t._cap_skip_optimizer=False
        t.lr_scheduler=ExclusionAwareScheduler(t.lr_scheduler,t)
        if not t.is_deepspeed_enabled or t.model_wrapped.zero_optimization_stage()!=2:
            raise RuntimeError('Expected ZeRO-2')
        if os.environ.get('CAP_VERIFY_RESUME')=='1':
            ckpt=Path(t.data_args.continuation_checkpoint)
            expected=torch.load(ckpt/'global_step320/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt',map_location='cpu',weights_only=False)['optimizer_state_dict']
            current=t.model_wrapped.optimizer.state_dict()
            assert state.global_step==320 and args.per_device_train_batch_size==12 and args.gradient_accumulation_steps==1
            assert torch.equal(current['single_partition_of_fp32_groups'][0].cpu(),expected['single_partition_of_fp32_groups'][0])
            for name in ['step','exp_avg','exp_avg_sq']:
                assert torch.equal(current['base_optimizer_state']['state'][0][name].cpu(),expected['base_optimizer_state']['state'][0][name])
            assert current['base_optimizer_state']['param_groups'][0]['lr']==expected['base_optimizer_state']['param_groups'][0]['lr']
            assert t.lr_scheduler.state_dict()['last_epoch']==320
            report=dict(global_step=state.global_step,zero_stage=2,batch=12,accum=1,
                        bf16=args.bf16,cap=t.data_args.max_total_sequence_length,
                        fp32_master_and_adam_verified=True,scheduler_epoch=320,
                        lr=current['base_optimizer_state']['param_groups'][0]['lr'],next_source_offset=640)
            (RUN/'live_resume_verification.json').write_text(json.dumps(report,indent=2)+'\n')
            print('CAP_RESUME_VERIFIED '+json.dumps(report),flush=True)
        print('CAP_BACKEND '+json.dumps(dict(zero_stage=2,bf16=args.bf16,batch=args.per_device_train_batch_size,
              accum=args.gradient_accumulation_steps,cap=t.data_args.max_total_sequence_length)),flush=True)
    def on_step_begin(self,args,state,control,**kwargs):
        torch.cuda.synchronize();self.start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        self.engine_step=self.trainer.model_wrapped.global_steps
        self.trainer._cap_skip_optimizer=False
        self.scheduler_epoch=self.trainer.lr_scheduler.state_dict()['last_epoch']
    def on_step_end(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        row=dict(global_step=state.global_step,seconds=time.perf_counter()-self.start,
                 peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                 peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                 engine_global_step=self.trainer.model_wrapped.global_steps,
                 scheduler_epoch=self.trainer.lr_scheduler.state_dict()['last_epoch'],
                 optimizer_updated=self.trainer.model_wrapped.global_steps>self.engine_step,
                 **getattr(self.trainer,'_cap_batch_audit',{}))
        if not row['optimizer_updated']:assert row['scheduler_epoch']==self.scheduler_epoch
        with (Path(args.output_dir)/'performance_steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('CAP_STEP '+json.dumps(row),flush=True)
        (RUN/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
        if os.environ.get('CAP_VERIFY_RESUME')=='1' and state.global_step==321:
            control.should_save=True

class CappedTrainer(base.ThinkWithGraphTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.accelerator.num_processes!=1 or self.args.gradient_accumulation_steps!=1:
            raise ValueError('Capped exclusion entry requires one GPU and accumulation=1')
        self._length_excluded=set()
        self.add_callback(RuntimeAudit(self))
    def _mark_length_discard(self,index,length,reason):
        self._length_excluded.add(index)
        env=self.environments[index]
        event=dict(attempted_total_tokens=int(length),limit=self.data_args.max_total_sequence_length,reason=reason)
        env._cap_discard_audit.setdefault('length_discard_events',[]).append(event)
    def _generate(self,prompts):
        self._length_excluded=set()
        for env in self.environments:env._cap_discard_audit={}
        output=list(super()._generate(prompts))
        p,c,m,completions,logps,sampling,extra,images,tool_images=output
        lengths=[(len(a),len(b)) for a,b in zip(p,c,strict=True)]
        self._raw_cap_lengths=lengths
        self._excluded=excluded_groups(lengths,self._length_excluded,self.num_generations,self.data_args.max_total_sequence_length)
        for i in sorted(self._excluded):
            env=self.environments[i]
            env._cap_discard_audit['training_discarded']=True
            env._cap_discard_audit['training_discard_reason']='full_sequence_limit_or_same_GRPO_group'
            env._cap_discard_audit['discarded_original_prompt_tokens']=lengths[i][0]
            env._cap_discard_audit['discarded_original_completion_tokens']=lengths[i][1]
            # Only an internal placeholder used to keep native reward/group bookkeeping aligned.
            # The entire row is removed before policy loss; raw audit messages remain in the environment.
            p[i]=[self._tokenizer.eos_token_id];c[i]=[self._tokenizer.eos_token_id]
            if m is not None:m[i]=[0]
            if logps is not None:logps[i]=[0.0]
            if sampling is not None:sampling[i]=[[self._tokenizer.eos_token_id]]
        for i in range(len(p)):
            self.environments[i]._cap_discard_audit.setdefault('training_discarded',False)
        assert max(map(len,p))+max(map(len,c))<self.data_args.max_total_sequence_length
        return tuple(output)
    def _generate_single_turn(self,prompt_ids,images,multimodal_fields,num_generations,has_tool_images=False):
        # Base stage budget is additionally reduced by the total-sequence budget in the snapshot.
        return super()._generate_single_turn(prompt_ids,images,multimodal_fields,num_generations,has_tool_images)
    def _generate_and_score_completions(self,inputs):
        if os.environ.get('CAP_VERIFY_RESUME')=='1' and self.state.global_step==320:
            expected=[4376,4322]
            actual=[x['source_index'] for x in inputs[::self.num_generations]]
            assert actual==expected,dict(expected=expected,actual=actual)
            (RUN/'data_cursor_verification.json').write_text(json.dumps(dict(global_step=320,
                skipped_source_count=640,expected_source_indices=expected,actual_source_indices=actual,passed=True),indent=2)+'\n')
            print('CAP_DATA_CURSOR_VERIFIED '+json.dumps(actual),flush=True)
        output=super()._generate_and_score_completions(inputs)
        kept=[i for i in range(len(inputs)) if i not in self._excluded]
        self._cap_batch_audit=dict(attempted_candidates=len(inputs),kept_candidates=len(kept),
            discarded_candidates=len(self._excluded),source_indices=[x['source_index'] for x in inputs[::self.num_generations]],
            max_total_sequence_length=self.data_args.max_total_sequence_length)
        self._metrics['train']['length_discard/candidate_fraction'].append(len(self._excluded)/len(inputs))
        self._metrics['train']['length_discard/kept_candidates'].append(len(kept))
        if kept:
            retained_reward=sum(self.last_rollout_records[i]['reward'] for i in kept)/len(kept)
            self._cap_batch_audit['retained_reward_mean']=retained_reward
            self._metrics['train']['length_discard/retained_reward_mean'].append(retained_reward)
        if not kept:
            self._cap_batch_audit['padded_training_tokens_per_candidate']=0
            raise EntireBatchDiscarded()
        indices=torch.tensor(kept,device=output['prompt_ids'].device)
        result={key:(value.index_select(0,indices) if isinstance(value,torch.Tensor) and value.ndim>0 and value.shape[0]==len(inputs) else value)
                for key,value in output.items()}
        # Remove prompt padding introduced solely by discarded rows.
        width=int(result['prompt_mask'].sum(1).max())
        for key in ['prompt_ids','prompt_mask']:result[key]=result[key][:,-width:]
        length=int(result['completion_mask'].sum(1).max())
        for key in ['completion_ids','completion_mask','tool_mask','old_per_token_logps',
                    'ref_per_token_logps','sampling_per_token_logps','sampling_mask','importance_sampling_ratio']:
            if isinstance(result.get(key),torch.Tensor):result[key]=result[key][:,:length]
        total=width+length
        assert total<self.data_args.max_total_sequence_length
        result['num_items_in_batch']=(result['completion_mask']*result['tool_mask']).sum()
        self._cap_batch_audit['padded_training_tokens_per_candidate']=total
        for record in self.last_rollout_records:
            if not record['training_discarded']:
                assert len(record['prompt_ids'])+len(record['completion_ids'])<self.data_args.max_total_sequence_length
        return result
    def training_step(self,model,inputs,num_items_in_batch):
        start=time.perf_counter()
        try:return super().training_step(model,inputs,num_items_in_batch)
        except EntireBatchDiscarded:
            # Do not call accelerator.backward: under DeepSpeed that would also execute engine.step,
            # moving Adam momentum, applying weight decay, and advancing its scheduler.
            self._buffered_inputs=None
            self._cap_skip_optimizer=True
            self._step+=1
            self._metrics['train']['step_time'].append(time.perf_counter()-start)
            print('CAP_ALL_DISCARDED_NO_OPTIMIZER_UPDATE '+json.dumps(self._cap_batch_audit),flush=True)
            return torch.zeros((),device=self.accelerator.device)

base.ThinkWithGraphTrainer=CappedTrainer
if __name__=='__main__':base.train()
