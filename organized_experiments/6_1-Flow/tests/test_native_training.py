import sys,unittest,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import bootstrap

class NativeTrainingTests(unittest.TestCase):
    def test_native_flow_grpo_lora_single_update_variable_flows(self):
        import numpy as np
        import torch,torch.distributed as dist
        from tensordict import TensorDict
        from omegaconf import open_dict
        from hydra import initialize_config_dir,compose
        import verl
        from verl import DataProto
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        from transformers import Qwen2Config,Qwen2ForCausalLM
        from peft import LoraConfig,get_peft_model
        from batching import install_actor_batching,align_transitions
        install_actor_batching()
        import verl.utils.torch_functional as functional
        functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE=False  # native PyTorch CPU fallback
        import verl.utils.profiler.performance as profiler
        profiler._get_current_mem_info=lambda:(0,0,0,1)  # disable CUDA-only memory telemetry on CPU
        with tempfile.TemporaryDirectory() as tmp:
            dist.init_process_group('gloo',init_method='file://'+tmp+'/group',rank=0,world_size=1)
            try:
                with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'),version_base=None):cfg=compose(config_name='ppo_trainer')
                actor_cfg=cfg.actor_rollout_ref.actor
                with open_dict(actor_cfg):
                    actor_cfg.use_remove_padding=False;actor_cfg.use_torch_compile=False;actor_cfg.ppo_mini_batch_size=2;actor_cfg.ppo_micro_batch_size_per_gpu=1;actor_cfg.ppo_epochs=1;actor_cfg.use_dynamic_bsz=False;actor_cfg.use_kl_loss=True;actor_cfg.kl_loss_coef=.001;actor_cfg.entropy_coeff=0.;actor_cfg.grad_clip=1.
                model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,attn_implementation='eager')).to(torch.bfloat16)
                model=get_peft_model(model,LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'),autocast_adapter_dtype=False)
                class CountingAdam(torch.optim.AdamW):
                    steps=0
                    def step(self,*args,**kwargs):
                        self.steps+=1
                        return super().step(*args,**kwargs)
                opt=CountingAdam([p for p in model.parameters() if p.requires_grad],lr=.01)
                actor=DataParallelPPOActor(actor_cfg,model,opt);actor.device_name='cpu'
                # Two sources x 6 flows; last flow has two decisions => 13 transitions.
                rewards=torch.tensor([1.,0,0,0,0,0,1,0,0,0,0,0,0]);uids=np.array(['source_a']*6+['source_b']*7)
                ids=torch.randint(0,32,(13,7));mask=torch.ones(13,3)
                scores=torch.zeros_like(mask);scores[:,-1]=rewards
                advantage,_=compute_grpo_outcome_advantage(scores,mask,uids)
                self.assertTrue((advantage[:6][0]>0).all())
                batch=DataProto(batch=TensorDict(dict(responses=ids[:,-3:],response_mask=mask,input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(7).expand(13,-1),advantages=advantage,token_level_scores=scores),batch_size=[13]),non_tensor_batch={'uid':uids},meta_info={'temperature':.7})
                with torch.no_grad():
                    _,logps=actor._forward_micro_batch(dict(batch.batch),temperature=.7)
                batch.batch['old_log_probs']=logps.clone();batch.batch['ref_log_prob']=logps.clone()
                aligned,padding=align_transitions(batch,2)
                self.assertEqual(padding,1);self.assertEqual(len(aligned),14)
                self.assertTrue((aligned.batch['response_mask'][-1]==0).all())
                self.assertEqual(len({x for x in batch.non_tensor_batch['uid']}),2)
                initial={n:p.detach().clone() for n,p in model.state_dict().items()}
                before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
                actor.update_policy(aligned)
                self.assertEqual(opt.steps,1)
                self.assertTrue(all(p.dtype==torch.bfloat16 for p in model.parameters()))
                self.assertTrue(all(v.dtype==torch.bfloat16 for state in opt.state.values() for key,v in state.items() if key in ['exp_avg','exp_avg_sq']))
                self.assertTrue(any(not torch.equal(before[n],p.detach()) for n,p in model.named_parameters() if n in before))
                self.assertEqual(actor_cfg.ppo_mini_batch_size,2)
                for width in [6,4,3,2]:
                    with self.subTest(microbatch=width):
                        model.load_state_dict(initial)
                        optimizer=CountingAdam([p for p in model.parameters() if p.requires_grad],lr=.01)
                        with open_dict(actor_cfg):actor_cfg.ppo_micro_batch_size_per_gpu=width
                        other=DataParallelPPOActor(actor_cfg,model,optimizer);other.device_name='cpu'
                        padded,_=align_transitions(batch,2*width)
                        other.update_policy(padded)
                        self.assertEqual(optimizer.steps,1)
                        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                        self.assertTrue(any(not torch.equal(before[n],p.detach()) for n,p in model.named_parameters() if n in before))
                        self.assertTrue(all(v.dtype==torch.bfloat16 for st in optimizer.state.values() for k,v in st.items() if k in ['exp_avg','exp_avg_sq']))
                report=dict(native_verl='0.5.0',real_tiny_qwen_forward_backward=True,lora_updated=True,source_questions=2,flows=12,real_transitions=13,padding_transitions=1,optimizer_steps=opt.steps,all_model_and_lora_bf16=True,optimizer_moments_bf16=True)
                (bootstrap.ROOT/'verification/native_cpu_grpo.json').write_text(json.dumps(report,indent=2)+'\n')
            finally:dist.destroy_process_group()
    def test_original_daemon_transition_tensorization(self):
        from agentflow.types import Rollout,Triplet
        from flow_transport import GraphFlowDaemon
        daemon=GraphFlowDaemon.__new__(GraphFlowDaemon)
        daemon.is_train=True;daemon.pad_token_id=0;daemon.reward_fillna_value=None
        daemon._completed_rollouts={};daemon._task_id_to_original_sample={};daemon._total_tasks_queued=12
        for source in range(2):
            for candidate in range(6):
                rid=f'{source}-{candidate}';reward=float(candidate==0)
                turns=2 if candidate==0 else 1
                traces=[Triplet(prompt={'token_ids':[1,2,3]},response={'token_ids':[4,5]}) for _ in range(turns)]
                daemon._completed_rollouts[rid]=Rollout(rollout_id=rid,final_reward=reward,triplets=traces)
                daemon._task_id_to_original_sample[rid]={'data_id':str(source)}
        batch,metrics=daemon.get_train_data_batch(16,8,'cpu')
        self.assertEqual(len(batch),14)
        self.assertEqual(batch.batch['input_ids'].shape,(14,5))
        self.assertEqual(set(batch.non_tensor_batch['data_id_list']),{'0','1'})
        self.assertEqual(float(batch.batch['token_level_scores'].sum()),4.)
        self.assertFalse(batch.batch['is_drop_mask'].any())

    def test_async_lora_sync_matches_peft_merge(self):
        import torch
        from transformers import Qwen2Config,Qwen2ForCausalLM
        from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
        from lora_sync import BaseWeightStore,merged_weights
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)
            base=Qwen2ForCausalLM(cfg).to(torch.bfloat16);base.save_pretrained(tmp)
            model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'),autocast_adapter_dtype=False)
            self.assertTrue(all(p.dtype==torch.bfloat16 for p in model.parameters() if p.requires_grad))
            for name,p in model.named_parameters():
                if 'lora_B' in name:torch.nn.init.normal_(p,std=.1)
            params=get_peft_model_state_dict(model)
            actual=dict(merged_weights(params,BaseWeightStore(tmp),4,2,'cpu'))
            expected=model.merge_and_unload().state_dict()
            self.assertEqual(len(actual),2)
            for key,value in actual.items():torch.testing.assert_close(value,expected[key])
if __name__=='__main__':unittest.main()
