"""Keep all Planner transitions from two source questions in one BF16 actor update."""
import json,os
from pathlib import Path
import torch
def align_transitions(batch,world_size):
    from verl.protocol import pad_dataproto_to_divisor
    real=len(batch);aligned,padding=pad_dataproto_to_divisor(batch,world_size)
    if padding:
        aligned.batch['response_mask'][real:]=0
        aligned.batch['advantages'][real:]=0
        if 'returns' in aligned.batch:aligned.batch['returns'][real:]=0
        for k in ['token_level_scores','token_level_rewards']:
            if k in aligned.batch:aligned.batch[k][real:]=0
    aligned.meta_info.update(flow_real_transitions=real,flow_aligned_transitions=len(aligned))
    return aligned,padding

def install_actor_batching():
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from omegaconf import open_dict
    cls=DataParallelPPOActor
    if getattr(cls,'_flow_installed',False):return
    # Match the original microbatch=1 objective: average each transition's
    # token mean, including KL. Padding contributes exactly zero, not 0/0.
    import verl.trainer.ppo.core_algos as core
    import verl.workers.actor.dp_actor as actor_backend
    native_agg=core.agg_loss
    def safe_agg(loss_mat,loss_mask,loss_agg_mode):
        if loss_agg_mode=='seq-mean-token-mean':
            return ((loss_mat*loss_mask).sum(-1)/loss_mask.sum(-1).clamp_min(1)).mean()
        return native_agg(loss_mat,loss_mask,loss_agg_mode)
    core.agg_loss=safe_agg;actor_backend.agg_loss=safe_agg
    update,step=cls.update_policy,cls._optimizer_step
    def full_update(self,data):
        if any(p.dtype!=torch.bfloat16 for p in self.actor_module.parameters() if p.is_floating_point()):
            raise RuntimeError("Actor and LoRA parameters must all be BF16")
        original=self.config.ppo_mini_batch_size
        original_agg=self.config.loss_agg_mode
        if len(data)%self.config.ppo_micro_batch_size_per_gpu:
            raise RuntimeError('Align transitions to world_size * microbatch before update')
        with open_dict(self.config):
            self.config.ppo_mini_batch_size=len(data)
            self.config.loss_agg_mode='seq-mean-token-mean'
        self._flow_pad_scale=data.meta_info['flow_aligned_transitions']/data.meta_info['flow_real_transitions']
        self._flow_real=data.meta_info['flow_real_transitions']
        try:return update(self,data)
        finally:
            with open_dict(self.config):
                self.config.ppo_mini_batch_size=original
                self.config.loss_agg_mode=original_agg
            self._flow_pad_scale=1.
    def normalized_step(self):
        scale=getattr(self,'_flow_pad_scale',1.)
        if scale!=1.:
            for parameter in self.actor_module.parameters():
                if parameter.grad is not None:parameter.grad.mul_(scale)
        gradients=[p.grad for p in self.actor_module.parameters() if p.grad is not None]
        if any(g.dtype!=torch.bfloat16 for g in gradients):raise RuntimeError('Actor gradients must all be BF16')
        result=step(self)
        moments=[value for state in self.actor_optimizer.state.values() for key,value in state.items() if key in ['exp_avg','exp_avg_sq','max_exp_avg_sq']]
        if any(t.dtype!=torch.bfloat16 for t in moments):raise RuntimeError('Optimizer moments must all be BF16')
        if os.environ.get('FLOW_RUN_DIR'):
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            report=dict(parameters_bf16=True,gradients_bf16=True,optimizer_moments_bf16=True,real_transitions=getattr(self,'_flow_real',None))
            (Path(os.environ['FLOW_RUN_DIR'])/f'optimizer_dtype_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
        return result
    cls.update_policy=full_update;cls._optimizer_step=normalized_step;cls._flow_installed=True
