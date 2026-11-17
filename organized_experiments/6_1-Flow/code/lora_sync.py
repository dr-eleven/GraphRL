"""Compatibility for VERL 0.5 async vLLM: stream merged LoRA projection weights.

The native async worker has no llm_engine.add_lora endpoint. Merge into inference
weights only; the training actor remains an unmerged PEFT LoRA model.
"""
import json, os, inspect, textwrap
from pathlib import Path
import torch
from safetensors import safe_open

class BaseWeightStore:
    def __init__(self,path):
        self.root=Path(path);self.cache={}
        index=self.root/'model.safetensors.index.json'
        self.index=json.loads(index.read_text())['weight_map'] if index.exists() else None
    def get(self,key):
        if key not in self.cache:
            file=self.index[key] if self.index else 'model.safetensors'
            with safe_open(str(self.root/file),framework='pt',device='cpu') as stream:self.cache[key]=stream.get_tensor(key)
        return self.cache[key]

def merged_weights(params,base,alpha,rank,device):
    for name,a in params.items():
        if '.lora_A.' not in name:continue
        bkey=name.replace('.lora_A.','.lora_B.')
        if bkey not in params:raise ValueError('Missing paired LoRA B: '+name)
        key=name.split('.lora_A.')[0]+'.weight'
        if key.startswith('base_model.model.'):key=key[len('base_model.model.'):]
        key=key.replace('_fsdp_wrapped_module.','').replace('.base_layer','')
        a=a.to(device=device,dtype=torch.bfloat16);b=params[bkey].to(device=device,dtype=torch.bfloat16)
        original=base.get(key).to(device=device,dtype=torch.bfloat16)
        yield key,original+(b@a)*(alpha/rank)

def install_lora_sync():
    import verl.workers.sharding_manager.fsdp_vllm as backend
    cls=backend.FSDPVLLMShardingManager
    if getattr(cls,'_flow_lora_installed',False):return
    # Native code keeps the actor's shards on GPU until after the inference
    # weights wake up. LoRA tensors are already independent CPU copies, so move
    # actor shards back first and avoid simultaneous actor/inference residency.
    enter=inspect.unwrap(cls.__enter__)
    if enter.__name__!='__enter__':
        functions=[c.cell_contents for c in (enter.__closure__ or ()) if inspect.isfunction(c.cell_contents)]
        enter=next((f for f in functions if f.__name__=='__enter__'),enter)
    enter_source=textwrap.dedent(inspect.getsource(enter))
    marker='        if self.rollout_config.free_cache_engine:\n'
    if marker not in enter_source:raise RuntimeError('Unexpected VERL sharding manager layout')
    enter_source=enter_source.replace(marker,'        if self.offload_param and peft_config is not None:\n            offload_fsdp_model_to_cpu(self.module)\n            get_torch_device().empty_cache()\n\n'+marker,1)
    exec(compile(enter_source,'<6_1-Flow offload actor before rollout wake>','exec'),backend.__dict__)
    cls.__enter__=backend.__dict__['__enter__']
    original=cls.update_params
    def update(self,updated_params,peft_config=None):
        if not peft_config:return original(self,updated_params,peft_config)
        if not hasattr(self,'_flow_base'):self._flow_base=BaseWeightStore(os.environ['FLOW_BASE_MODEL'])
        model=self.model_runner.model
        model.load_weights(merged_weights(updated_params,self._flow_base,peft_config.lora_alpha,peft_config.r,next(model.parameters()).device))
        self.base_sync_done=True
    cls.update_params=update;cls._flow_lora_installed=True
