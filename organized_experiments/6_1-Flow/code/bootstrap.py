from pathlib import Path
import os, sys
ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'upstream'
def activate():
    for p in [str(ROOT/'code'),str(UPSTREAM)]:
        if p not in sys.path:sys.path.insert(0,p)
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('WANDB_MODE','disabled')
    for key,value in {'VLLM_NO_USAGE_STATS':'1','DO_NOT_TRACK':'1','RAY_USAGE_STATS_ENABLED':'0','HF_HUB_DISABLE_TELEMETRY':'1'}.items():os.environ.setdefault(key,value)
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('NCCL_P2P_DISABLE','0')
    os.environ.setdefault('NCCL_IB_DISABLE','0')
    os.environ.setdefault('VLLM_USE_V1','1')
    os.environ.setdefault('RAY_DEDUP_LOGS','0')
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    os.environ.pop('PYTORCH_ALLOC_CONF',None)
    os.environ['NO_PROXY']='*'  # All task inference and Ray endpoints are local.
activate()
