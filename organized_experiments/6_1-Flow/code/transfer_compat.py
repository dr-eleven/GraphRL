"""Use synchronous copies in unchanged VERL offload helpers on this CUDA runtime."""
import inspect,sys

def install_blocking_transfers():
    import verl.utils.fsdp_utils as helpers
    if getattr(helpers,'_flow_blocking_transfers',False):return
    names=['offload_fsdp_model_to_cpu','offload_fsdp2_model_to_cpu','load_fsdp_model_to_gpu','load_fsdp2_model_to_gpu','offload_fsdp_optimizer','load_fsdp_optimizer']
    for name in names:
        source=inspect.getsource(getattr(helpers,name))
        if 'non_blocking=True' not in source:raise RuntimeError('Unexpected VERL transfer helper: '+name)
        # Compile the installed, version-checked native helper with only its
        # transfer flag changed. Keep FSDP invariants and state handling intact.
        source=source.replace('non_blocking=True','non_blocking=False')
        namespace=helpers.__dict__
        exec(compile(source,'<6_1-Flow blocking '+name+'>','exec'),namespace)
        replacement=namespace[name]
        setattr(helpers,name,replacement)
        for consumer in ['verl.workers.fsdp_workers','verl.workers.sharding_manager.fsdp_vllm']:
            module=sys.modules.get(consumer)
            if module is not None and hasattr(module,name):setattr(module,name,replacement)
    helpers._flow_blocking_transfers=True
