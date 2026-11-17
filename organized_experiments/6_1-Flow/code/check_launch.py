"""Validate the six training GPU plus two frozen-module GPU deployment."""
import argparse,json,os
from pathlib import Path
import yaml


def validate(settings,config,train_gpus='0,1,2,3,4,5',fixed_gpus='6,7',fixed_tp=2):
    train=train_gpus.split(',');fixed=fixed_gpus.split(',')
    if len(train)!=config['trainer']['n_gpus_per_node']:raise ValueError('TRAIN_GPUS must match trainer.n_gpus_per_node')
    if len(fixed)!=fixed_tp:raise ValueError('FIXED_GPUS must contain FIXED_TP devices')
    if len(set(train+fixed))!=8 or len(train)+len(fixed)!=8:raise ValueError('Use eight distinct GPUs, six for Planner and two for fixed roles')
    if settings['train_batch_size']!=2 or config['data']['train_batch_size']!=2:raise ValueError('One update must use two source questions')
    if settings['rollout_n']!=6 or config['actor_rollout_ref']['rollout']['n']!=6:raise ValueError('Each source must have six rollout attempts')
    if settings['max_tokens']!=config['data']['max_response_length']:raise ValueError('Engine generation cap must equal training response capacity')
    if settings['workers']>len(train)*settings['planner_flows_per_server']:raise ValueError('Global workers exceed per-server flow slots')
    cap=config['actor_rollout_ref']['rollout']['max_model_len']
    if config['data']['max_prompt_length']+settings['max_tokens']>cap:raise ValueError('Prompt and generation capacities exceed server context')
    if settings['planner_max_total_tokens']>cap:raise ValueError('Training transition token limit exceeds server context')
    return dict(total_gpus=8,train_gpus=train,fixed_gpus=fixed,fixed_tp=fixed_tp,
                workers=settings['workers'],microbatch=config['actor_rollout_ref']['actor']['ppo_micro_batch_size_per_gpu'],
                max_generated_tokens=settings['max_tokens'],planner_training_total_tokens=settings['planner_max_total_tokens'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--check-gpus',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    settings=json.loads(Path(os.environ.get('FLOW_SETTINGS',root/'configs/settings.json')).read_text())
    config=yaml.safe_load((root/'configs/graph_flow.yaml').read_text())
    report=validate(settings,config,os.environ.get('TRAIN_GPUS','0,1,2,3,4,5'),os.environ.get('FIXED_GPUS','6,7'),int(os.environ.get('FIXED_TP','2')))
    if a.check_gpus:
        import torch
        report['visible_gpus']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.device_count()<8:raise RuntimeError('This launcher requires eight visible GPUs before splitting training and fixed services')
        if not Path(settings['model']).is_dir():raise FileNotFoundError('Download the local Qwen2.5-7B-Instruct model first: '+settings['model'])
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
