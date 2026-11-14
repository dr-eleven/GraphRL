"""Real tiny-Qwen forward, reference, GRPO loss and optimizer; scripted generation only."""
from functools import partial
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from test_binary_controller import sample, tool


class TRLIntegration(unittest.TestCase):
    def test_real_grpo_update_masks_and_zero_variance_groups(self):
        import torch
        from trl import GRPOTrainer
        from peft import LoraConfig
        from transformers import AutoTokenizer,PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
        from tokenizers import Tokenizer,models,pre_tokenizers,decoders
        from train_grpo import (ThinkWithGraphTrainer,GraphRLTrainingArguments,
            GraphRLDataArguments,GraphRLDataset,graph_reward)
        from episode import GraphEnvironment
        from project_config import DEFAULT_MODE
        torch.set_num_threads(2)
        original=AutoTokenizer.from_pretrained(os.environ.get('TRAIN_MODEL','/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct'),local_files_only=True)
        vocab={c:i for i,c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend=Tokenizer(models.BPE(vocab=vocab,merges=[]))
        backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
        backend.decoder=decoders.ByteLevel()
        tok=PreTrainedTokenizerFast(tokenizer_object=backend,chat_template=original.chat_template,
            eos_token='<|im_end|>',pad_token='<|endoftext|>',
            additional_special_tokens=['<|im_start|>','<tool_call>','</tool_call>','<tool_response>','</tool_response>'])
        tok.padding_side='left'
        row=sample()
        data_args=GraphRLDataArguments(generation_micro_batch_size=1)
        budget=data_args.rollout_budget()
        data=GraphRLDataset([row],budget=budget).to_hf_dataset()
        fixture=[{'role':'system','content':'Restore the graph when necessary, then classify. Use analysis before actions and prediction.'},
                 {'role':'user','content':'Task: NC; target: node0; candidates: library, shop. '
                  'Nodes 0,1,2 exist. Edges 0-1 and 1-2. Node0 protected. Tools and protocol as declared.'}]
        data=data.remove_columns('prompt').add_column('prompt',[fixture])
        cfg=Qwen2Config(vocab_size=len(tok),hidden_size=8,intermediate_size=16,num_hidden_layers=1,
                       num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=32768,
                       eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,attention_dropout=0)
        model=Qwen2ForCausalLM(cfg)
        model.config._name_or_path='tiny-qwen-binary-integration'

        class AuditTrainer(ThinkWithGraphTrainer):
            def _generate_and_score_completions(self,inputs):
                self.captured=super()._generate_and_score_completions(inputs)
                return self.captured

        forced_label=[None]
        leaf_history=[]
        def scripted_leaf(self,prompt_ids,images,multimodal_fields,num_generations,has_tool_images=False):
            outputs=[]
            for i,ids in zip(self._active_rollout_indices,prompt_ids,strict=True):
                env=self.environments[i]
                leaf_history.append(dict(candidate=i,turn=env._turns,input=list(ids)))
                if env._turns==0:
                    response=tool('del_node',[dict(op='DELETE_NODE',node=0)])
                elif DEFAULT_MODE=='ordered_four_stage_retry' and (env._stage<3 or not env._stage_succeeded):
                    from episode import TOOL_ORDER
                    name=TOOL_ORDER[env._stage+1] if env._stage_succeeded else env._current_tool
                    response=tool(name)
                elif DEFAULT_MODE=='free' and env._turns < 4:
                    name=['del_edge','add_edge','rewrite'][env._turns-1]
                    actions=([dict(op='REWRITE',node=0,text='Library with books')] if name=='rewrite'
                             else [dict(op={'del_edge':'DEL_EDGE','add_edge':'ADD_EDGE'}[name],u=1,v=2)])
                    response=tool(name,actions)
                else:
                    response=json.dumps(dict(analysis="Reason",prediction=forced_label[0] or ('library' if i==0 else 'shop')))
                tokens=tok.encode(response+tok.eos_token,add_special_tokens=False)
                if len(tokens)>self.generation_config.max_new_tokens:raise AssertionError('Fixture exceeds actual generation cap')
                outputs.append(tokens)
            return outputs,None,None

        with tempfile.TemporaryDirectory() as tmp:
            args=GraphRLTrainingArguments(output_dir=tmp,use_cpu=True,bf16=False,fp16=False,tf32=False,
                deepspeed=None,use_liger_kernel=False,warmup_steps=0,per_device_train_batch_size=2,
                gradient_accumulation_steps=1,num_generations=2,num_train_epochs=1,max_steps=1,
                learning_rate=1e-3,report_to='none',save_strategy='no',logging_steps=1,disable_tqdm=True,
                beta=.01,gradient_checkpointing=False)
            trainer=AuditTrainer(model=model,args=args,train_dataset=data,processing_class=tok,
                reward_funcs=[graph_reward],environment_factory=partial(GraphEnvironment,budget=budget),
                peft_config=LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',lora_dropout=0),
                data_args=data_args)
            self.assertNotIn('one or more functions',trainer.chat_template)
            before={n:p.detach().clone() for n,p in trainer.model.named_parameters() if p.requires_grad}
            with patch.object(GRPOTrainer,'_generate_single_turn',scripted_leaf):trainer.train()
            self.assertEqual(trainer.state.global_step,1)
            self.assertTrue(any(not torch.equal(before[n],p.detach()) for n,p in trainer.model.named_parameters() if n in before))
            records=trainer.last_rollout_records
            self.assertEqual([r['reward'] for r in records],[1,0])
            self.assertTrue(records[0]['advantage']>0);self.assertTrue(records[1]['advantage']<0)
            for i,r in enumerate(records):
                self.assertEqual(r['errors'][0]['code'],'INVALID_ACTION')
                self.assertEqual(r['finish_reason'],'final_submitted')
                self.assertEqual(sum(x*y for x,y in zip(r['tool_mask'],r['completion_mask'])),r['metrics']['model_tokens'])
                second=next(x for x in leaf_history if x['candidate']==i and x['turn']==1)
                self.assertIn('ENVIRONMENT_FEEDBACK',tok.decode(second['input']))
                self.assertIn('DELETE_NODE',tok.decode(second['input']))
                self.assertNotIn('ground_truth',tok.decode(second['input']))
            inputs=trainer.captured
            targets=(inputs['completion_mask']*inputs['tool_mask']).bool()
            # All external tokens still have attention=1; tool mask alone excludes their direct loss.
            self.assertTrue((inputs['completion_mask'][inputs['tool_mask']==0]==1).all())
            current=torch.nn.Parameter(torch.zeros_like(inputs['completion_mask'],dtype=torch.float32))
            with patch.object(trainer,'_get_per_token_logps_and_entropies',return_value=(current,torch.ones_like(current),None)):
                trainer.compute_loss(trainer.model,inputs).backward()
            self.assertTrue((current.grad[~targets]==0).all())
            self.assertTrue((current.grad[targets]!=0).all())
            # Same real forward/reward/advantage pipeline: homogeneous binary groups are finite and zero.
            group_checks=[]
            for label in ['library','shop']:
                forced_label[0]=label
                with patch.object(GRPOTrainer,'_generate_single_turn',scripted_leaf):
                    output=trainer._generate_and_score_completions([dict(data[0]),dict(data[0])])
                self.assertTrue(torch.isfinite(output['advantages']).all())
                self.assertTrue((output['advantages']==0).all())
                group_checks.append([r['reward'] for r in trainer.last_rollout_records])
            # Exercise real HF generate and candidate-specific stopping, not the scripted leaf.
            from dataclasses import replace
            trainer.data_args=replace(data_args,generation_micro_batch_size=2)
            trainer._active_rollout_indices=[0,1]
            for e,cap in zip(trainer.environments,[3,7]):e._next_cap=cap
            generated,_,_=trainer._generate_single_turn([tok.encode('Hi'),tok.encode('Hi')],None,{},1)
            self.assertLessEqual(len(generated[0]),3);self.assertLessEqual(len(generated[1]),7)
            self.assertTrue(all(generated))
            report=dict(mode=DEFAULT_MODE,device='cpu',tiny_model=True,scripted_generation=True,
                        real_grpo_update=True,lora_updated=True,real_hf_bounded_generation=True,loss_mask_verified=True,feedback_attention_verified=True,
                        rewards=[1,0],homogeneous_groups=group_checks,
                        model_tokens=[r['metrics']['model_tokens'] for r in records],
                        feedback_tokens=[r['metrics']['feedback_tokens'] for r in records])
            (Path(__file__).resolve().parents[1]/'verification/cpu_grpo.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':unittest.main()
