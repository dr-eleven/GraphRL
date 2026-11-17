import bootstrap
from batching import install_actor_batching
from lora_sync import install_lora_sync
from transfer_compat import install_blocking_transfers
install_blocking_transfers();install_actor_batching();install_lora_sync()
# This driver/runtime rejects CUDA pinned host allocations during CuMem sleep.
# Pageable CPU backup retains identical model values with synchronous memcpy.
from vllm.device_allocator import cumem
cumem.is_pin_memory_available=lambda:False
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
class FlowActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    def __init__(self,config,role,**kwargs):
        from omegaconf import open_dict
        width=config.actor.ppo_micro_batch_size_per_gpu
        # Native initialization derives a fixed placeholder mini-batch from
        # source count*n/world. Actual Flow batches have variable turn counts.
        # Validate native setup at width 1, then align the placeholder; the
        # actor update wrapper replaces it with all local transitions per batch.
        with open_dict(config.actor):config.actor.ppo_micro_batch_size_per_gpu=1
        try:super().__init__(config,role,**kwargs)
        finally:
            with open_dict(config.actor):config.actor.ppo_micro_batch_size_per_gpu=width
        with open_dict(self.config.actor):
            self.config.actor.ppo_mini_batch_size=((self.config.actor.ppo_mini_batch_size+width-1)//width)*width


# PEFT normally promotes adapters to FP32. Disable that promotion explicitly.
import json,os,torch
from collections import Counter
from pathlib import Path
import verl.workers.fsdp_workers as native_workers
_original_get_peft_model=native_workers.get_peft_model

def bf16_peft_model(*args,**kwargs):
    kwargs['autocast_adapter_dtype']=False
    model=_original_get_peft_model(*args,**kwargs).to(dtype=torch.bfloat16)
    if any(p.dtype!=torch.bfloat16 for p in model.parameters() if p.is_floating_point()):
        raise RuntimeError('All model and LoRA parameters must be BF16')
    if os.environ.get('FLOW_RUN_DIR'):
        rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        report=dict(all_model_parameters_bf16=True,all_trainable_parameters_bf16=True,parameter_dtypes=dict(Counter(str(p.dtype) for p in model.parameters())),trainable_parameter_dtypes=dict(Counter(str(p.dtype) for p in model.parameters() if p.requires_grad)))
        (Path(os.environ['FLOW_RUN_DIR'])/f'actor_dtype_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    return model

native_workers.get_peft_model=bf16_peft_model
