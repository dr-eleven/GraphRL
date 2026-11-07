"""Native TRL multi-turn tools, staged graph state, full-history loss and LoRA update."""
import os
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from episode import GraphEnvironment, TOOL_ORDER, FINAL_MARKER, initial_messages


def sample():
    graph={'nodes':{'0':'A public library','1':'A reading room'},'edges':[{'u':0,'v':1}]}
    return dict(source_index=7,task_type='nc',target='node0',target_node=0,
        label_space=['library','shop'],noisy_graph=graph,clean_graph=deepcopy(graph),
        ground_truth='library',protected_nodes=[0],protected_edges=[],hidden_gold_marker='HIDDEN_GOLD_NEVER_RENDER')


def invoke(env,name,analysis='Reason from the latest graph.',actions=None):
    actions=[] if actions is None else actions
    result=getattr(env,name)(analysis,actions)
    call={'role':'assistant','content':'','tool_calls':[{'type':'function','function':{
        'name':name,'arguments':{'analysis':analysis,'actions':actions}}}]}
    return [call,{'role':'tool','name':name,'content':result}]


def answer(reason='Books imply a library.'):
    return {'role':'assistant','content':json.dumps({'analysis':reason,'prediction':'library'})}


class EnvironmentTests(unittest.TestCase):
    def test_all_clean_stages_and_required_reasoning(self):
        env=GraphEnvironment();env.reset(sample());messages=[]
        self.assertFalse(hasattr(env,'finish'))
        self.assertNotIn('clean_graph',env._state.sample)
        for stage in TOOL_ORDER:
            self.assertEqual(env._state.current_tool,stage)
            messages+=invoke(env,stage)
            self.assertNotIn('<restored_graph>',messages[-1]['content'])
            self.assertIn('This stage used actions=[] and made no graph modifications.',messages[-1]['content'])
            self.assertNotIn('HIDDEN_GOLD_NEVER_RENDER',messages[-1]['content'])
        self.assertTrue(env._state.done)
        self.assertEqual(env._state.finish_reason,'four_stages_complete')
        self.assertEqual(env._state.turns,4)
        self.assertIn(FINAL_MARKER,messages[-1]['content'])
        self.assertEqual(env._score([*messages,answer()]),3)
        self.assertEqual(env._record['metrics']['structure_reward'],1)
        self.assertEqual(env._score([*messages,answer(reason='')]),1.8)
        with self.assertRaises(ValueError): GraphEnvironment(max_turns=8)

    def test_state_accumulates_invalid_action_skips_stage_without_retry(self):
        row=sample();row['noisy_graph']={'nodes':{'0':'Library advert','1':'Room','2':'Spam','3':'Books'},
            'edges':[{'u':0,'v':1},{'u':1,'v':2},{'u':1,'v':3}]}
        env=GraphEnvironment();env.reset(row);messages=[]
        messages+=invoke(env,'del_node',actions=[{'op':'DELETE_NODE','node':2}])
        self.assertNotIn('2',env._state.graph['nodes'])
        messages+=invoke(env,'del_edge',actions=[{'op':'DEL_EDGE','u':0,'v':1}])
        previous=deepcopy(env._state.graph)
        messages+=invoke(env,'add_edge',actions=[{'op':'ADD_EDGE','u':0,'v':999}])
        self.assertEqual(env._state.graph,previous)
        self.assertEqual(env._state.current_tool,'rewrite')
        self.assertIn('Edge endpoint missing',messages[-1]['content'])
        self.assertNotIn('<restored_graph>',messages[-1]['content'])
        self.assertIn('Continue the task using ONLY this current valid graph.',messages[-1]['content'])
        self.assertIn('Disregard the analysis',messages[-1]['content'])
        messages+=invoke(env,'rewrite',actions=[{'op':'REWRITE','node':0,'text':'A library with books and reading rooms.'}])
        self.assertEqual(env._state.graph['nodes']['0'],'A library with books and reading rooms.')
        self.assertEqual(len(env._state.actions),3)
        self.assertEqual(env._state.turns,4)
        self.assertEqual(env._score([*messages,answer()]),2.75)
        metrics=env._record['metrics']
        self.assertEqual((metrics['structure_reward'],metrics['behavior_reward'],metrics['task_reward']),(1,.75,1))

    def test_nonempty_edit_then_empty_stages_reuse_graph(self):
        from episode import CURRENT_GRAPH_GUIDANCE
        env=GraphEnvironment();env.reset(sample())
        messages=invoke(env,'del_node',actions=[{'op':'DELETE_NODE','node':1}])
        self.assertIn('<restored_graph>',messages[-1]['content'])
        latest=deepcopy(env._state.graph)
        version=env._state.version
        for stage in TOOL_ORDER[1:]:
            messages+=invoke(env,stage)
            self.assertNotIn('<restored_graph>',messages[-1]['content'])
            self.assertIn(CURRENT_GRAPH_GUIDANCE,messages[-1]['content'])
            self.assertIn('made no graph modifications',messages[-1]['content'])
            self.assertNotIn('Disregard the analysis',messages[-1]['content'])
            self.assertEqual(env._state.graph,latest)
            self.assertEqual(env._state.version,version)
        self.assertIn(FINAL_MARKER,messages[-1]['content'])
        self.assertEqual(env._score([*messages,answer()]),2.75)

    def test_atomic_failed_list_preserves_graph_and_advances(self):
        row=sample();row['noisy_graph']['nodes']['2']='Books'
        env=GraphEnvironment();env.reset(row);messages=[]
        for stage in TOOL_ORDER[:2]: messages+=invoke(env,stage)
        before=deepcopy(env._state.graph)
        messages+=invoke(env,'add_edge',actions=[{'op':'ADD_EDGE','u':0,'v':2},
                                                {'op':'ADD_EDGE','u':0,'v':999}])
        self.assertEqual(env._state.graph,before)
        self.assertEqual(env._state.current_tool,'rewrite')
        messages+=invoke(env,'rewrite')
        self.assertEqual(env._score([*messages,answer()]),2.5)

    def test_format_errors_terminate_without_action(self):
        cases = [
            ('del_edge', {'analysis':'Reason','actions':[]}),
            ('del_node', {'analysis':'','actions':[]}),
            ('del_node', {'actions':[]}),
            ('del_node', {'analysis':'Reason','actions':[],'extra':1}),
            ('del_node', {'analysis':'Reason','actions':[{'op':'DELETE_NODE','node':'1'}]}),
        ]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                env=GraphEnvironment();env.reset(sample())
                before=deepcopy(env._state.graph)
                result=getattr(env,name)(**arguments)
                self.assertIn('FORMAT_ERROR',result)
                self.assertNotIn('<restored_graph>',result)
                self.assertTrue(env._state.done)
                self.assertEqual(env._state.finish_reason,'format_error')
                self.assertEqual(env._state.turns,0)
                self.assertEqual(env._state.graph,before)
                self.assertEqual(env._score([answer()]),0)
                self.assertTrue(env._record['format_errors'])
                self.assertEqual([env._record['metrics'][k] for k in
                    ['structure_reward','behavior_reward','task_reward','total']], [0,0,0,0])
                env.del_node('No later dispatch',[{'op':'DELETE_NODE','node':1}])
                self.assertEqual(env._state.graph,before)

    def test_multicall_validation_precedes_all_execution(self):
        env=GraphEnvironment();env.reset(sample())
        calls=[{'type':'function','function':{'name':name,'arguments':{
            'analysis':'Reason','actions':[]}}} for name in ['del_node','del_edge']]
        self.assertFalse(env._validate_turn({'role':'assistant','tool_calls':calls}))
        self.assertEqual(env._state.turns,0)
        self.assertFalse(env._invocations)

    def test_dirty_noop_and_prediction_scores_are_independent(self):
        row=sample();row['noisy_graph']['nodes']['0']='Library with advertisement'
        env=GraphEnvironment();env.reset(row);messages=[]
        for stage in TOOL_ORDER: messages+=invoke(env,stage)
        self.assertEqual(env._score([*messages,answer()]),2.75)
        self.assertEqual(env._record['metrics']['behavior_reward'],.75)
        incorrect={'role':'assistant','content':json.dumps({'analysis':'Evidence','prediction':'shop'})}
        self.assertEqual(env._score([*messages,incorrect]),1.85)
        premature=GraphEnvironment();premature.reset(row)
        self.assertEqual(premature._score([answer()]),0)

    def test_prediction_tiers_and_accuracy(self):
        from evaluate_grpo_600 import summarize
        for label, expected in [('library',1),('shop',0.1),('outside',0)]:
            env=GraphEnvironment();env.reset(sample());messages=[]
            for stage in TOOL_ORDER: messages+=invoke(env,stage)
            final={'role':'assistant','content':json.dumps({'analysis':'Reason','prediction':label})}
            self.assertAlmostEqual(env._score([*messages,final]),2+expected)
            self.assertEqual(env._record['metrics']['task_reward'],expected)
            self.assertEqual(summarize([env._record])['task_accuracy'],float(label=='library'))

    def test_clean_edit_loses_only_its_stage_behavior_reward(self):
        env=GraphEnvironment();env.reset(sample());messages=[]
        for stage in TOOL_ORDER:
            actions=[{'op':'REWRITE','node':0,'text':'A public library with books'}] if stage=='rewrite' else []
            messages+=invoke(env,stage,actions=actions)
        self.assertEqual(env._score([*messages,answer()]),2.75)
        self.assertEqual(env._record['metrics']['unnecessary_clean_edit'],1)

    def test_baseline_only_receives_task_reward(self):
        env=GraphEnvironment();env.reset(sample(),bypass_actions=True)
        self.assertEqual(env._score([answer()]),1)
        self.assertEqual(env._record['metrics']['structure_metrics_applicable'],0)
        self.assertEqual(env._record['metrics']['behavior_reward'],0)

    def test_unobserved_tool_commit_retains_only_prior_stage_rewards(self):
        env=GraphEnvironment();env.reset(sample());messages=[]
        for stage in TOOL_ORDER: messages+=invoke(env,stage)
        self.assertEqual(env._score([*messages[:-1],answer()]),1.35)
        self.assertEqual(env._record['metrics']['observed_execution'],0)

    def test_schema_exposes_exactly_four_required_tools(self):
        import inspect
        from transformers.utils.chat_template_utils import get_json_schema
        env=GraphEnvironment()
        tools=[getattr(env,n) for n,f in inspect.getmembers(type(env),predicate=inspect.isfunction)
               if not n.startswith('_') and n!='reset']
        self.assertEqual({t.__name__ for t in tools},set(TOOL_ORDER))
        for tool in tools:
            schema=get_json_schema(tool)['function']
            self.assertEqual(schema['parameters']['required'],['analysis','actions'])


class NativeTrainerTests(unittest.TestCase):
    def test_reference_chunk_does_not_shrink_policy_batch(self):
        import torch
        from trl import GRPOTrainer
        from train_grpo import ThinkWithGraphTrainer, GraphRLDataArguments
        trainer=object.__new__(ThinkWithGraphTrainer)
        trainer.data_args=GraphRLDataArguments(logprob_micro_batch_size=1)
        ids=torch.ones((2,4),dtype=torch.long)
        with patch.object(GRPOTrainer,'_get_per_token_logps_and_entropies',return_value='scored') as parent:
            with torch.no_grad():
                self.assertEqual(trainer._get_per_token_logps_and_entropies(None,ids,ids,2,batch_size=2),'scored')
            self.assertEqual(parent.call_args.kwargs['batch_size'],1)
            with torch.enable_grad():
                trainer._get_per_token_logps_and_entropies(None,ids,ids,2,batch_size=2)
            self.assertEqual(parent.call_args.kwargs['batch_size'],2)

    def test_native_four_stages_invalid_action_and_loss(self):
        import torch
        import trl
        from trl import GRPOTrainer
        from peft import LoraConfig
        from transformers import AutoTokenizer,PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
        from tokenizers import Tokenizer,models,pre_tokenizers,decoders
        from train_grpo import (ThinkWithGraphTrainer,GraphRLTrainingArguments,
            GraphRLDataArguments,GraphRLDataset,graph_reward)
        torch.set_num_threads(2)
        original=AutoTokenizer.from_pretrained(os.environ.get('TRAIN_MODEL', '/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct'),local_files_only=True)
        vocab={c:i for i,c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend=Tokenizer(models.BPE(vocab=vocab,merges=[]))
        backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
        backend.decoder=decoders.ByteLevel()
        tok=PreTrainedTokenizerFast(tokenizer_object=backend,chat_template=original.chat_template,
            eos_token='<|im_end|>',pad_token='<|endoftext|>',
            additional_special_tokens=['<|im_start|>','<tool_call>','</tool_call>','<tool_response>','</tool_response>'])
        tok.padding_side='left'
        fixture_prompt=[{'role':'user','content':'Classify node0, a public library with books, from labels library or shop. '
            'Node1 is a reading room, node2 is an unrelated coin promotion connected to node0. '
            'Reason and visit del_node, del_edge, add_edge, rewrite once each in order. No retries. '
            'Retain all history, then reason and predict. CONTINUE_TOOL_STAGE: del_node.'}]
        row=sample()
        row['noisy_graph']['nodes']['2']='Unrelated coin promotion'
        row['noisy_graph']['edges'].append({'u':0,'v':2})
        dataset=GraphRLDataset([row]).to_hf_dataset()
        dataset=dataset.remove_columns('prompt').add_column('prompt',[fixture_prompt])
        cfg=Qwen2Config(vocab_size=len(tok),hidden_size=8,intermediate_size=16,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
            max_position_embeddings=32768,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,
            attention_dropout=0.0)
        model=Qwen2ForCausalLM(cfg)
        model.config._name_or_path='tiny-qwen2-native-multiturn-test'

        class AuditTrainer(ThinkWithGraphTrainer):
            def _generate_and_score_completions(self,inputs):
                value=super()._generate_and_score_completions(inputs)
                self.captured=value
                return value

        # 只替换生成叶子；生产子类的阶段预算、TRL工具循环、reference、loss和optimizer均实际运行。
        def scripted_leaf(self,prompt_ids,images,multimodal_fields,num_generations,has_tool_images=False):
            self.leaf_records=getattr(self,'leaf_records',[])
            outputs=[];finished=[]
            for i,ids in enumerate(prompt_ids):
                text=tok.decode(ids)
                found=re.search(r'candidate([01])',text)
                candidate=int(found.group(1)) if found else i
                env=self.environments[candidate]
                stage=env._state.current_tool
                finished.append(env._state.done)
                self.leaf_records.append(dict(candidate=candidate,stage=stage,
                    prompt_ids=list(ids),cap=self.generation_config.max_new_tokens))
                if env._state.done:
                    response=json.dumps({'analysis':f'candidate{candidate}: Books imply a library.', 'prediction':'library'})
                else:
                    actions=([{'op':'DELETE_NODE','node':2 if candidate==0 else 999}]
                             if not env._state.events else [])
                    response='<tool_call>\n'+json.dumps({'name':stage,'arguments':{
                        'analysis':f'candidate{candidate}: Reason about {stage} using the latest graph.',
                        'actions':actions}})+'\n</tool_call>'
                outputs.append(tok.encode(response+tok.eos_token,add_special_tokens=False))
            expected=128 if all(finished) else 256
            self.assert_generation_cap=self.generation_config.max_new_tokens
            if self.generation_config.max_new_tokens!=expected:
                raise AssertionError(f'Mixed-stage cap mismatch: {self.generation_config.max_new_tokens} != {expected}')
            return outputs,None,None

        with tempfile.TemporaryDirectory() as tmp:
            args=GraphRLTrainingArguments(output_dir=tmp,use_cpu=True,bf16=False,fp16=False,
                tf32=False,deepspeed=None,use_liger_kernel=False,warmup_steps=0,
                per_device_train_batch_size=2,gradient_accumulation_steps=1,num_generations=2,
                num_train_epochs=1,max_steps=1,learning_rate=1e-3,report_to='none',
                save_strategy='no',logging_steps=1,disable_tqdm=True,
                max_completion_length=8192,beta=.01,gradient_checkpointing=False)
            trainer=AuditTrainer(model=model,args=args,train_dataset=dataset,
                processing_class=tok,reward_funcs=[graph_reward],environment_factory=partial(GraphEnvironment,max_turns=4),
                peft_config=LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],
                    task_type='CAUSAL_LM',lora_dropout=0),
                data_args=GraphRLDataArguments(max_prompt_length=20000,action_max_new_tokens=256,prediction_max_new_tokens=128))
            before={n:p.detach().clone() for n,p in trainer.model.named_parameters() if p.requires_grad}
            with patch.object(GRPOTrainer,'_generate_single_turn',scripted_leaf): trainer.train()
            self.assertEqual(trainer.state.global_step,1)
            self.assertTrue(any(not torch.equal(before[n],p.detach()) for n,p in trainer.model.named_parameters() if n in before))
            records=trainer.last_rollout_records
            self.assertEqual([r['reward'] for r in records],[3.1,2.5])
            self.assertEqual([tuple(r['metrics'][key] for key in
                ('structure_reward','behavior_reward','task_reward')) for r in records],[(1,1,1),(1,.5,1)])
            self.assertEqual([len(r['events']) for r in records],[4,4])
            self.assertEqual([r['finish_reason'] for r in records],['four_stages_complete','four_stages_complete'])
            for candidate,record in enumerate(records):
                calls=[r for r in trainer.leaf_records if r['candidate']==candidate]
                self.assertEqual(len(calls),len(record['events'])+1)
                first=calls[0]['prompt_ids']
                for call in calls:
                    self.assertEqual(call['prompt_ids'][:len(first)],first)
                    self.assertIn(fixture_prompt[0]['content'],tok.decode(call['prompt_ids']))
                decoded=tok.decode(calls[-1]['prompt_ids'])
                self.assertIn(f'candidate{candidate}',decoded)
                graphs=re.findall(r'<restored_graph>\n(.*?)\n</restored_graph>',decoded,re.S)
                self.assertEqual(len(graphs),1 if candidate==0 else 0)
                if graphs:
                    self.assertNotIn('node2:',graphs[-1])
                else:
                    self.assertIn('original input graph G0',decoded)
                self.assertEqual(decoded.count('This stage used actions=[]'),3)
                self.assertNotIn('HIDDEN_GOLD_NEVER_RENDER',decoded)
                valid=record['completion_mask'];mask=record['tool_mask']
                sequence=[v for v,m in zip(mask,valid) if m]
                runs=sum(v==0 and (i==0 or sequence[i-1]==1) for i,v in enumerate(sequence))
                self.assertEqual(runs,len(record['events']))
                self.assertEqual(sequence[0],1);self.assertEqual(sequence[-1],1)
            inputs=trainer.captured
            current=torch.nn.Parameter(torch.zeros_like(inputs['completion_mask'],dtype=torch.float32))
            with patch.object(trainer,'_get_per_token_logps_and_entropies',return_value=(current,torch.ones_like(current),None)):
                trainer.compute_loss(trainer.model,inputs).backward()
            targets=(inputs['completion_mask']*inputs['tool_mask']).bool()
            self.assertTrue((current.grad[~targets]==0).all())
            self.assertTrue((current.grad[targets]!=0).all())
            report=dict(trl_version=trl.__version__,global_step=1,lora_updated=True,
                native_tool_loop=True,native_loss=True,tool_targets_zero_gradient=True,
                qwen_tool_template=True,test_vocab_size=len(tok),tool_turns=[4,4],invalid_action_skips_stage=True,
                rewards=[r['reward'] for r in records],model_tokens=targets.sum(dim=1).tolist(),
                reward_components=[r['metrics'] for r in records],
                events_by_candidate=[r['events'] for r in records],latest_graph_verified=True,
                external_tokens=(inputs['completion_mask']*(1-inputs['tool_mask'])).sum(dim=1).tolist(),sample=records[0])
            (Path(__file__).resolve().parents[1]/'verification/native_four_stage_chain.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__': unittest.main()
