"""真实 TRL/Trainer 回归：三段 mask、分组尾批、标准梯度累积、实际 LoRA 梯度。"""
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, Trainer

from config import ROOT, load_config
from train_grpo import (ThinkWithGraphTrainer, GraphRLTrainingArguments, MRRADataArguments,
                        GraphRLDataset, EpochCheckpointSnapshots)
from trl import GRPOTrainer
from model_backend import Segment
from episode import first_messages
from tail_batching import TailSafeRankBatchSampler
from test_protocol import sample


def tokenizer():
    raw = Tokenizer(WordLevel({'[PAD]':0,'[BOS]':1,'[EOS]':2,'[UNK]':3,'graph':4,'library':5,'sports':6},unk_token='[UNK]'))
    raw.pre_tokenizer=Whitespace()
    t=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]')
    t.chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '[EOS]' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    return t


class ControlledTrainer(ThinkWithGraphTrainer):
    # 只替代候选生成；优势、loss、backward、累积、优化器与 checkpoint 全部走真实 TRL/Trainer。
    def _generate_batch(self,prompt_ids,stage):
        out=[]
        for i,ids in enumerate(prompt_ids):
            good=i%2==0
            if stage in {'modify','rewrite','predict'}:
                reason='<reason>use graph evidence</reason>'
                if stage=='predict':
                    text=reason+'<answer>'+('library' if good else 'sports')+'</answer>'
                else:
                    actions=[{'op':'REWRITE','node':0,'text':'library books'}] if stage=='rewrite' and good else []
                    tag=stage+'_action'
                    text=reason+'<'+tag+'>'+json.dumps(actions)+'</'+tag+'>'
                tokens=self.processing_class(text,add_special_tokens=False)['input_ids']
                out.append(Segment(ids,tokens,text,stage))
                continue
            if stage=='answer':
                obj={'reason':'read actual feedback','answer':'library' if good else 'sports'}
            else:
                obj={'modify':{'reason':'retain structure','actions':[]},
                     'rewrite':{'reason':'restore text' if good else 'retain text',
                                'actions':[{'op':'REWRITE','node':0,'text':'library books'}] if good else []}}
                if stage=='llm':
                    graph=deepcopy(self._active_inputs[i]['episode']['noisy_graph'])
                    if good:graph['nodes']['0']='library books'
                    obj.update(restored_graph=graph,reason='use declared graph',answer='library' if good else 'sports')
            text=json.dumps(obj)
            tokens=self.processing_class(text,add_special_tokens=False)['input_ids']
            out.append(Segment(ids,tokens,text,stage))
        return out

    def compute_loss(self, *args, **kwargs):
        if not hasattr(self, 'accumulation_windows'):
            self.accumulation_windows = []
        self.accumulation_windows.append(self.current_gradient_accumulation_steps)
        return super().compute_loss(*args, **kwargs)


def build_trainer(output,strategy='executor',sources=5,batch=8,accum=4):
    torch.manual_seed(4)
    cfg=load_config(ROOT/'configs/mrra_executor.json'); cfg['strategy']=strategy
    t=tokenizer()
    rows=[]
    for i in range(sources):
        s=sample();s['sample_id']=f'toy:{i}';s['noisy_graph']['nodes']['0']+=' misleading';rows.append(s)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=16,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=4096,
        eos_token_id=2,pad_token_id=0,attention_dropout=0.0))
    args=GraphRLTrainingArguments(output_dir=str(output),use_cpu=True,bf16=False,fp16=False,
        per_device_train_batch_size=batch,per_device_eval_batch_size=batch,gradient_accumulation_steps=accum,
        num_generations=2,steps_per_generation=1,num_iterations=1,num_train_epochs=1,
        learning_rate=1e-3,warmup_steps=0,lr_scheduler_type='constant',beta=0.01,
        loss_type='grpo',use_liger_loss=False,use_liger_kernel=False,gradient_checkpointing=False,
        save_strategy='no',logging_strategy='no',report_to='none',disable_tqdm=True,
        remove_unused_columns=False,dataloader_pin_memory=False,lora_enable=True,disable_dropout=True)
    dataset=GraphRLDataset(rows,t,cfg)
    return ControlledTrainer(model=model,args=args,train_dataset=dataset,processing_class=t,
        reward_funcs=[lambda **kwargs:[0.]*len(kwargs.get('completions',[]))],
        peft_config=LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'),
        mrra_config=cfg,data_args=MRRADataArguments())


class TRLTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_inherits_real_trl(self):
        self.assertTrue(issubclass(ThinkWithGraphTrainer,GRPOTrainer))
        self.assertIs(ThinkWithGraphTrainer.training_step,GRPOTrainer.training_step)
        self.assertIs(ThinkWithGraphTrainer.train,GRPOTrainer.train)

    def test_rendered_prompt_is_not_misdetected_as_conversation(self):
        from trl.data_utils import is_conversational
        with tempfile.TemporaryDirectory() as output:
            trainer=build_trainer(output,sources=1,batch=2,accum=1)
            self.assertFalse(is_conversational(trainer.train_dataset[0]))

    def test_three_rank_groups_batch8_accum4_and_tail(self):
        for count in (1,3,4,17,2747,2748,5495):
            ranks=[list(TailSafeRankBatchSampler(count,6,3,r,8)) for r in range(3)]
            seen=Counter()
            for batches in zip(*ranks):
                ids=sum(batches,[])
                for start in range(0,len(ids),6):self.assertEqual(len(set(ids[start:start+6])),1)
                seen.update(ids)
            self.assertEqual(seen,Counter({i:6 for i in range(count)}))

    def test_real_trainer_A_B_updates_with_tail(self):
        for strategy in ('llm','executor'):
            with tempfile.TemporaryDirectory() as output:
                trainer=build_trainer(output,strategy,sources=17)
                self.assertTrue(isinstance(trainer,GRPOTrainer))
                initial={n:p.detach().clone() for n,p in trainer.model.named_parameters() if p.requires_grad}
                trainer.train()
                self.assertEqual(trainer.state.global_step,2)
                self.assertEqual(trainer.accumulation_windows,[4,4,4,4,1])
                rows=[json.loads(line) for line in (Path(output)/'rollouts_rank0.jsonl').read_text().splitlines()]
                self.assertEqual(Counter(r['sample_id'] for r in rows),Counter({f'toy:{i}':2 for i in range(17)}))
                self.assertEqual({r['reward'] for r in rows},{0.,2.} if strategy=='llm' else {1.,3.})
                self.assertTrue(any(not torch.equal(initial[n],p) for n,p in trainer.model.named_parameters() if n in initial))
                if strategy=='executor':
                    for row in rows:
                        self.assertGreater(row['environment_token_count'],0)
                        self.assertEqual(row['loss_token_count'],sum(len(s['completion_ids']) for s in row['segments']))
                        self.assertEqual(len(row['segments']),3)
                        self.assertEqual(row['metrics']['format_reward'],1.)

    def test_environment_attention_is_visible_loss_is_zero(self):
        with tempfile.TemporaryDirectory() as output:
            trainer=build_trainer(output,sources=1,batch=2,accum=1)
            trainer.model.train()
            inputs=trainer._generate_and_score_completions([trainer.train_dataset[0]]*2)
            loss_mask=inputs['completion_mask']; attention=inputs['completion_attention_mask']
            env=(attention==1)&(loss_mask==0)
            self.assertGreater(int(env.sum()),0)
            self.assertTrue(torch.all(loss_mask<=attention))
            # 两次环境反馈的 token loss 梯度为零；三段生成位置都有非零梯度。
            import trl.trainer.grpo_trainer as module
            captured=[]
            original=trainer._get_per_token_logps_and_entropies
            def capture(*args,**kwargs):
                out=original(*args,**kwargs)
                if torch.is_grad_enabled():
                    out[0].retain_grad();captured.append(out[0])
                return out
            trainer._get_per_token_logps_and_entropies=capture
            trainer.current_gradient_accumulation_steps=1
            loss=trainer.compute_loss(trainer.model,inputs)
            loss.backward()
            gradient=captured[0].grad
            self.assertEqual(float(gradient[env].abs().max()),0)
            for index,row in enumerate(trainer._records):
                base=len(row['segments'][0]['prompt_ids'])
                self.assertEqual(len(row['segments']),3)
                for segment in row['segments']:
                    start=len(segment['prompt_ids'])-base
                    end=start+len(segment['completion_ids'])
                    self.assertGreater(float(gradient[index,start:end].abs().sum()),0)

    def test_trainer_checkpoint_and_epoch_coverage(self):
        with tempfile.TemporaryDirectory() as output:
            trainer=build_trainer(output,'executor',sources=3,batch=4,accum=1)
            trainer.add_callback(EpochCheckpointSnapshots([f'toy:{i}' for i in range(3)],2))
            trainer.train()
            pinned=Path(output)/'epochs/epoch_1'
            for name in ('trainer_state.json','optimizer.pt','scheduler.pt','adapter_model.safetensors'):
                self.assertTrue((pinned/name).exists(),name)
            coverage=json.loads((Path(output)/'coverage_epoch_1.json').read_text())
            self.assertTrue(coverage['covered'])
            self.assertEqual(coverage['candidate_count'],6)
            trainer.save_model(str(Path(output)/'final_adapter'))
            self.assertTrue((Path(output)/'final_adapter/adapter_config.json').exists())


if __name__=='__main__':unittest.main()
