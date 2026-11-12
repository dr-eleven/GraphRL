"""CPU-only contract checks; run stages in separate interpreters to isolate imports."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sample=dict(source_index=1,task_type='NC',target='node0',target_node=0,label_space=['A','B'],ground_truth='A',attack_budget=1,
    noisy_graph=dict(nodes={'0':'center','1':'neighbour','2':'noise'},edges=[dict(u=0,v=1),dict(u=0,v=2)]),
    clean_graph=dict(nodes={'0':'center','1':'neighbour'},edges=[dict(u=0,v=1)]),protected_nodes=[0],protected_edges=[],injected_nodes=[2])


def test_stage1():
    sys.path.insert(0,str(ROOT/'code/stage1'))
    import episode
    import generation_limits
    prompt=episode.build_restoration_prompt(sample)
    assert 'prediction' not in prompt.lower() and 'candidate' not in prompt.lower()
    assert 'ground_truth' not in prompt and 'label_space' not in prompt
    def msg(t):
        acts=[dict(op='DELETE_NODE',node=2)] if t=='del_node' else []
        return dict(role='assistant',tool_calls=[dict(type='function',function=dict(name=t,arguments=dict(analysis='Inspect actual graph.',actions=acts)))])
    env=episode.GraphEnvironment();env.reset(sample)
    rest=[msg(t) for t in episode.TOOL_ORDER[1:]]
    original_parser=generation_limits.parse_response
    generation_limits.parse_response=lambda tok,ids,**kw: rest[ids[0]-101]
    class Fake(generation_limits.TokenLimitToolsMixin):
        environments=[env];max_tool_calling_iterations=4;max_completion_length=8704
        _is_vlm=False;tools=list(episode.TOOL_ORDER);environment_factories=None;use_vllm=False
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768))
        _sync_tool_dicts=[{t:getattr(env,t) for t in episode.TOOL_ORDER}];_async_tool_dicts=[{}]
        _tokenizer=SimpleNamespace(decode=lambda ids,**kw:'unused')
        generated=0
        def _get_tool_suffix_ids(self,messages):return [0]
        def _generate_single_turn(self,*args,**kw):
            self.generated+=1
            assert self.generated<=3,'Fifth response was requested'
            return [[100+self.generated]],None,None
    fake=Fake()
    try:
        result=fake._tool_call_loop([[dict(role='user',content=prompt)]],[[1]],[[100]],[[msg('del_node')]],None,None,None,{})
    finally:generation_limits.parse_response=original_parser
    tool_mask,messages,ids=result[:3]
    assert fake.generated==3 and sum(m['role']=='assistant' for m in messages[0])==4
    assert len(messages[0])==8 and messages[0][-1]['role']=='tool'
    reward=env._score(messages[0]);metrics=env._record['metrics']
    assert abs(reward-1.9)<1e-8 and abs(metrics['structure_reward']-.8)<1e-8
    assert not any('prediction' in k or 'task_reward'==k for k in metrics)
    assert env._record['final_graph']==sample['clean_graph']
    assert sum(tool_mask[0])==4 and len(tool_mask[0])==len(ids[0])
    changed=deepcopy(sample);changed['ground_truth']='B';env2=episode.GraphEnvironment();env2.reset(changed)
    manual=[]
    for t in episode.TOOL_ORDER:
        m=msg(t);args=m['tool_calls'][0]['function']['arguments'];feedback=getattr(env2,t)(**args)
        manual.extend([m,dict(role='tool',name=t,content=feedback)])
    assert abs(env2._score(manual)-reward)<1e-8


def test_stage2():
    sys.path.insert(0,str(ROOT/'code/stage2'))
    import episode,reward
    prompt=episode.build_prediction_prompt(sample)
    assert 'del_node' not in prompt and 'tool' not in prompt.lower() and 'actions' not in prompt
    response=[dict(role='assistant',content=json.dumps(dict(analysis='Evidence from supplied graph',prediction='A')))]
    score,record=reward.score_episode(sample,response,terminated=True)
    assert score==2 and record['metrics']['prediction_correct']==1
    assert not any(k.startswith(('behavior_','hit_')) for k in record['metrics'])
    assert reward.score_episode(sample,response,terminated=False)[0]==0
    # Verify installed PEFT can continue the same adapter, retaining its learned tensors.
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import LoraConfig,get_peft_model,PeftModel
    config=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)
    first=get_peft_model(Qwen2ForCausalLM(config),LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj'],task_type='CAUSAL_LM'))
    for n,p in first.named_parameters():
        if 'lora_B' in n:p.data.fill_(.125)
    expected={n:p.detach().clone() for n,p in first.named_parameters() if 'lora_' in n}
    with tempfile.TemporaryDirectory() as td:
        first.save_pretrained(td,save_embedding_layers=False)
        continued=PeftModel.from_pretrained(Qwen2ForCausalLM(config),td,is_trainable=True)
        actual={n:p for n,p in continued.named_parameters() if p.requires_grad}
        assert set(actual)==set(expected) and len(continued.peft_config)==1
        assert all(torch.equal(actual[n].detach(),v) for n,v in expected.items())


def test_sampling():
    sys.path.insert(0,str(ROOT/'scripts'))
    from prepare_quarter_data import prepare
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);data=[]
        for budget in range(6):
            for i in range(8):data.append(dict(deepcopy(sample),source_index=8*budget+i,attack_budget=budget))
        source=root/'train.json';source.write_text(json.dumps(data))
        a=prepare(source,root/'a',42);b=prepare(source,root/'b',42)
        assert a['source_indices']==b['source_indices'] and len(a['source_indices'])==12
        assert all(v['selected']==2 for v in a['counts'].values())
        assert a['source_indices']==sorted(a['source_indices'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['stage1','stage2','sampling']);a=p.parse_args()
    {'stage1':test_stage1,'stage2':test_stage2,'sampling':test_sampling}[a.stage]()
    print('PASS '+a.stage)
