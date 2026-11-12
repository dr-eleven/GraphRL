"""Gated two-epoch experiment: restoration, graph handoff, prediction, merge, test."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
GRAPH_ROOT=ROOT.parents[1]
PREVIOUS=GRAPH_ROOT/'organized_experiments/3_6/runs/3_6_gpu1_z2_b12_a1_cap5120_resume320_20261109_122155'
PYTHON=os.environ.get('PYTHON_BIN','/home/temptest/.venvs/graphrl-3_5_1-3_6/bin/python')
DEFAULT_SOURCE=GRAPH_ROOT/'900_train_data_clean20_badgraph80_degree_sorted'


def prerequisite(source):
    pointer=PREVIOUS/'final_evaluation_pointer.json'
    if not pointer.exists():return False,'waiting_3_6_test'
    try:folder=Path(json.loads(pointer.read_text())['path'])
    except (ValueError,KeyError):return False,'waiting_readable_3_6_pointer'
    status=folder/'status.json'
    try:test_state=json.loads(status.read_text()).get('state') if status.exists() else None
    except ValueError:test_state=None
    if test_state!='completed':return False,'waiting_3_6_test'
    try:
        analysis=json.loads((folder/'analysis_completed.json').read_text())
        checkpoint=json.loads((folder/'manifest.json').read_text())['checkpoint_step']
        report=Path(analysis['report'])
        if analysis.get('state')!='completed' or analysis.get('checkpoint_step')!=checkpoint:return False,'waiting_matching_3_6_analysis'
        if hashlib.sha256(report.read_bytes()).hexdigest()!=analysis['report_sha256']:return False,'waiting_verified_3_6_report'
    except (FileNotFoundError,ValueError,KeyError):return False,'waiting_3_6_analysis'
    if not source.exists():return False,'waiting_new_900_dataset'
    candidates=[source] if source.is_file() else [p for p in [source/'train_grpo_episodes_redesign.json',source/'train_episodes.json'] if p.is_file()]
    if len(candidates)!=1:return False,'waiting_unambiguous_900_training_file'
    try:
        raw=candidates[0].read_bytes()
        expected=json.loads((ROOT/'configs/data_source.json').read_text())
        if hashlib.sha256(raw).hexdigest()!=expected['training_sha256']:return False,'waiting_verified_900_sha256'
        data=json.loads(raw)
        if not isinstance(data,list) or not data:return False,'waiting_valid_900_dataset'
        if len({s['source_index'] for s in data})!=len(data):return False,'waiting_unique_900_source_ids'
        counts={b:0 for b in range(6)}
        for s in data:
            if not all(k in s for k in ['clean_graph','noisy_graph','target','task_type','ground_truth','label_space','attack_budget']):return False,'waiting_complete_900_fields'
            if type(s['attack_budget']) is not int or s['attack_budget'] not in counts:return False,'waiting_valid_900_budget'
            counts[s['attack_budget']]+=1
        if any(n<4 for n in counts.values()):return False,'waiting_all_six_900_strata'
        if len(data)!=expected['training_count'] or counts!={int(k):v for k,v in expected['budget_counts'].items()}:return False,'waiting_expected_900_stratum_counts'
    except (ValueError,KeyError,TypeError):return False,'waiting_valid_900_dataset'
    return True,'ready'


def run(source):
    from prepare_quarter_data import prepare
    ready,reason=prerequisite(source)
    if not ready:raise RuntimeError(reason)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run=ROOT/'runs'/('two_epochs_quarter_'+stamp);run.mkdir(parents=True)
    def status(state,**extra):
        obj=dict(state=state,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=str(run),**extra)
        (run/'status.json').write_text(json.dumps(obj,indent=2)+'\n')
        (ROOT/'latest_run.json').write_text(json.dumps(obj,indent=2)+'\n')
    lock=(GRAPH_ROOT/'setup_logs/gpu1_experiment.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_used=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
    if gpu_used>1024:raise RuntimeError('GPU1 still occupied: '+str(gpu_used))
    manifest=prepare(source,run/'data',42)
    expected=json.loads((ROOT/'configs/data_source.json').read_text())
    assert manifest['selected_count']==expected['quarter_training_count']
    assert manifest['selected_sha256']==expected['quarter_training_sha256']
    shutil.copytree(ROOT/'configs',run/'configs')
    for stage in ['stage1','stage2']:
        shutil.copytree(ROOT/'code'/stage,run/stage/'code',ignore=shutil.ignore_patterns('__pycache__'))
    base=GRAPH_ROOT.parent/'models/Qwen2.5-7B-Instruct'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',TRAIN_MODEL=str(base),BGE_M3_MODEL=str(GRAPH_ROOT.parent/'models/bge-m3'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
    env.pop('CAP_VERIFY_RESUME',None)
    for key in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','ACCELERATE_USE_DEEPSPEED']:env.pop(key,None)
    def execute(name,command,stage=None):
        status(name);(run/(name+'_command.json')).write_text(json.dumps(command,indent=2)+'\n')
        process_env=dict(env)
        if stage:process_env['CAP_RUN']=str(run/stage)
        with (run/(name+'.log')).open('w') as log:
            subprocess.run(command,env=process_env,stdout=log,stderr=subprocess.STDOUT,check=True)
    def train_command(stage,model,data):
        entry='train_capped.py' if stage=='stage1' else 'train_grpo.py'
        cmd=[PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=1',str(run/stage/'code'/entry),
            '--model_id',str(model),'--data_path',str(data),'--output_dir',str(run/stage/'grpo'),
            '--deepspeed',str(run/'configs/zero2_gpu.json'),'--per_device_train_batch_size','12','--gradient_accumulation_steps','1',
            '--num_train_epochs','1','--num_generations','6','--lora_rank','32','--lora_alpha','32','--lora_dropout','0.05',
            '--learning_rate','2e-6','--warmup_steps','55','--beta','0.01','--bf16','True','--fp16','False','--tf32','True',
            '--gradient_checkpointing','True','--gradient_checkpointing_kwargs','{"use_reentrant":true}',
            '--disable_flash_attn2','False','--use_liger_kernel','True','--save_steps','10','--logging_steps','1','--report_to','none']
        if stage=='stage1':cmd+=['--max_total_sequence_length','5120']
        else:cmd+=['--adapter_init',str(run/'stage1/grpo/final_adapter')]
        return cmd
    quarter=run/'data/train_quarter.json'
    try:
        execute('epoch1_train',train_command('stage1',base,quarter),'stage1')
        (run/'stage1/grpo/final_adapter/stage1_completed.json').write_text(json.dumps(dict(epoch=1,run=str(run),train_manifest_sha256=hashlib.sha256((run/'data/manifest.json').read_bytes()).hexdigest()),indent=2))
        merge=GRAPH_ROOT/'organized_experiments/3_6/code/merge_adapter.py'
        execute('epoch1_merge',[PYTHON,str(merge),'--adapter',str(run/'stage1/grpo/final_adapter'),'--output',str(run/'stage1_merged')])
        (run/'stage1_merged/stage1_handoff.json').write_text(json.dumps(dict(epoch=1,train_manifest=manifest,adapter=str(run/'stage1/grpo/final_adapter')),indent=2))
        execute('freeze_epoch1_graphs',[PYTHON,str(run/'stage1/code/export_graphs.py'),'--model',str(run/'stage1_merged'),'--data',str(quarter),'--output',str(run/'handoff')])
        execute('epoch2_train',train_command('stage2',base,run/'handoff/prediction_episodes.json'),'stage2')
        execute('epoch2_merge',[PYTHON,str(merge),'--adapter',str(run/'stage2/grpo/final_adapter'),'--output',str(run/'final_merged')])
        test=GRAPH_ROOT/'data/3_5_1_3_6/test_episodes.json'
        for condition in ['noisy','clean']:
            cmd=[PYTHON,str(run/'stage1/code/export_graphs.py'),'--model',str(run/'stage1_merged'),'--data',str(test),'--output',str(run/'test_graphs'/condition)]
            if condition=='clean':cmd+=['--clean']
            execute('test_graphs_'+condition,cmd)
            execute('test_prediction_'+condition,[PYTHON,str(run/'stage2/code/evaluate.py'),'--model',str(run/'final_merged'),
                '--data',str(run/'test_graphs'/condition/'prediction_episodes.json'),'--output',str(run/'test_predictions'/condition)])
        status('completed',selected_train_sources=manifest['selected_count'],epochs=2)
    except Exception as exc:
        status('failed',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=DEFAULT_SOURCE);p.add_argument('--wait',action='store_true')
    a=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    coordinator=(ROOT/'.pipeline.lock').open('a')
    fcntl.flock(coordinator,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        ok,state=prerequisite(a.source)
        if ok:
            used=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
            if used>1024:ok,state=False,'waiting_gpu1_release'
        (ROOT/'pending_status.json').write_text(json.dumps(dict(state=state,source=str(a.source),time=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
        if ok:
            try:run(a.source);break
            except BlockingIOError:
                if not a.wait:raise
        if not a.wait:raise SystemExit(state)
        time.sleep(30)
