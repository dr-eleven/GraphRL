"""Three-stage B protocol: actual state transitions, independent rewards, stopping and token tape."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from test_protocol import sample
from episode import public_observation, first_messages
from executor_protocol import (parse_stage, execute_stage, observation_after, aggregate_stages,
                               stage_messages, extract_tagged_answer)


def output(stage, actions=None, answer='library'):
    tag={'modify':'modify_action','rewrite':'rewrite_action','predict':'answer'}[stage]
    payload=answer if stage=='predict' else json.dumps(actions or [])
    return '<reason>Specific graph evidence supports this decision.</reason><'+tag+'>'+payload+'</'+tag+'>'


class ThreeStageProtocolTests(unittest.TestCase):
    def test_modify_failure_is_atomic_and_rewrite_can_continue(self):
        s=sample();o=public_observation(s)
        m=execute_stage(o,output('modify',[{'op':'DEL_EDGE','u':0,'v':1},
            {'op':'DELETE_NODE','node':0}]),'modify')
        self.assertIsNotNone(m['execution']['error'])
        self.assertEqual(m['execution']['graph'],o['graph'])
        r=execute_stage(observation_after(o,m),output('rewrite',[{'op':'REWRITE','node':0,'text':'restored library'}]),'rewrite')
        combined=aggregate_stages(m,r)[0]
        self.assertEqual(combined['graph']['nodes']['0'],'restored library')
        self.assertEqual(combined['graph']['edges'],o['graph']['edges'])
        self.assertEqual(combined['executed_actions'],r['execution']['actions'])
        self.assertIsNotNone(combined['error'])

    def test_rewrite_failure_preserves_modify_but_rolls_back_all_text(self):
        s=sample();o=public_observation(s)
        m=execute_stage(o,output('modify',[{'op':'DELETE_NODE','node':2}]),'modify')
        r=execute_stage(observation_after(o,m),output('rewrite',[{'op':'REWRITE','node':0,'text':'changed'},
            {'op':'REWRITE','node':1,'text':'out of scope'}]),'rewrite')
        combined=aggregate_stages(m,r)[0]
        self.assertNotIn('2',combined['graph']['nodes'])
        self.assertEqual(combined['graph']['nodes']['0'],o['graph']['nodes']['0'])
        self.assertEqual(combined['executed_actions'],[{'op':'DELETE_NODE','node':2}])

    def test_strict_formats_and_semantic_errors_are_distinct(self):
        good=output('modify')
        self.assertEqual(parse_stage(good,'modify')['actions'],[])
        for text in ['prefix'+good,good+'suffix',good+good,good.replace('<reason>','<reason><reason>',1),
                     good.replace('Specific graph evidence supports this decision.',''),
                     good.replace('<modify_action>[]','<modify_action>{}'),
                     good.replace('modify_action','rewrite_action'),
                     good.replace('[]','[NaN]'),good.replace('[]','[{"op":"X","op":"Y"}]')]:
            with self.assertRaises(ValueError,msg=text):parse_stage(text,'modify')
        wrong=execute_stage(public_observation(sample()),output('modify',[{'op':'DELETE_NODE','node':0}]),'modify')
        self.assertEqual(wrong['protocol_errors'],[])
        self.assertIsNotNone(wrong['execution']['error'])

    def test_format_failure_still_allows_later_prediction(self):
        self.assertEqual(extract_tagged_answer('<reason></reason><answer>library</answer>'),'library')
        self.assertEqual(extract_tagged_answer('<answer>library</answer><answer>sports</answer>'),'')
        self.assertEqual(extract_tagged_answer('<answer>library'),'')

    def test_task_specific_prompts_and_feedback_history(self):
        s=sample();messages=first_messages(s,'executor')
        self.assertIn('robust node classification',messages[0]['content'])
        execution=execute_stage(public_observation(s),output('modify'),'modify')['execution']
        second=stage_messages(messages,output('modify'),execution,'rewrite')
        self.assertEqual(second[:-2],messages)
        self.assertIn('<restored_graph>'+json.dumps(execution['graph'],ensure_ascii=False)+'</restored_graph>',second[-1]['content'])
        self.assertIn('使用最新修改图进行后续推理。',second[-1]['content'])
        self.assertIn('Current stage: Rewrite',second[-1]['content'])
        s.update(task_type='LC',protected_query_nodes=[0,1])
        self.assertIn('robust link classification',first_messages(s,'executor')[0]['content'])

    def test_end_to_end_three_calls_and_independent_B_T_F(self):
        from rollout import run_trajectory
        from model_backend import Segment
        class Policy:
            def __init__(self,texts):self.texts=iter(texts);self.calls=[]
            def generate(self,messages,stage,**kwargs):
                self.calls.append((deepcopy(messages),stage))
                return Segment([1],[2],next(self.texts),stage)
        s=sample()
        for modify,rewrite,predict,expected in [
            (output('modify'),output('rewrite'),output('predict'),(1,1,1)),
            (output('modify',[{'op':'DELETE_NODE','node':0}]),output('rewrite'),output('predict'),(0,1,1)),
            (output('modify'),output('rewrite'),'<reason></reason><answer>library</answer>',(1,1,0)),
            ('bad format',output('rewrite'),output('predict'),(0,1,0)),
            (output('modify'),output('rewrite'),output('predict',answer='sports'),(1,0,1))]:
            policy=Policy([modify,rewrite,predict]);segments,r=run_trajectory(policy,s,'executor')
            self.assertEqual([stage for _,stage in policy.calls],['modify','rewrite','predict'])
            self.assertEqual(tuple(r['metrics'][k] for k in ['behavior_reward','task_reward','format_reward']),expected)
            self.assertEqual(r['reward'],sum(expected))
            self.assertEqual(r['metrics']['model_calls'],3)
            self.assertEqual(len(policy.calls[1][0]),4)
            self.assertEqual(len(policy.calls[2][0]),6)
            self.assertNotIn('ground_truth',policy.calls[2][0][-1]['content'])

    def test_partial_success_reference_metrics_count_actual_actions(self):
        from reference_metrics import compute_reference_metrics
        s=sample();s['noisy_graph']['edges'].append({'u':0,'v':2});o=public_observation(s)
        m=execute_stage(o,output('modify',[{'op':'DEL_EDGE','u':0,'v':2}]),'modify')
        r=execute_stage(observation_after(o,m),output('rewrite',[{'op':'REWRITE','node':1,'text':'invalid'}]),'rewrite')
        combined=aggregate_stages(m,r)[0]
        values,_=compute_reference_metrics(s,combined)
        self.assertEqual(values['executed_edit_count'],1)
        self.assertEqual(values['structure_action_correct'],1)
        self.assertEqual(values['executed_edge_error_reduction'],1)

    def test_ablation_rejects_disabled_actions_without_format_penalty(self):
        o=public_observation(sample())
        for stage,ablation,actions in [('modify','rewrite_only',[{'op':'DELETE_NODE','node':2}]),
                ('rewrite','modify_only',[{'op':'REWRITE','node':0,'text':'changed'}])]:
            result=execute_stage(o,output(stage,actions),stage,ablation)
            self.assertEqual(result['protocol_errors'],[])
            self.assertIsNotNone(result['execution']['error'])
            self.assertEqual(result['execution']['graph'],o['graph'])

    def test_actual_Qwen_token_tape_preserves_EOS_and_both_feedback_masks(self):
        from transformers import AutoTokenizer
        from model_backend import Segment
        from config import ROOT
        from trajectory_tokens import render_messages,feedback_tokens,pack_segments
        tokenizer=AutoTokenizer.from_pretrained('/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct',local_files_only=True)
        s=sample();o=public_observation(s);messages=first_messages(s,'executor')
        prompt=tokenizer(render_messages(tokenizer,messages),add_special_tokens=False)['input_ids']
        first=Segment(prompt,tokenizer(output('modify'),add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id],output('modify'),'modify')
        m=execute_stage(o,first.text,'modify')
        messages2,suffix1,ids2=feedback_tokens(tokenizer,messages,first,m['execution'],'rewrite')
        second=Segment(ids2,tokenizer(output('rewrite'),add_special_tokens=False)['input_ids'],output('rewrite'),'rewrite')
        r=execute_stage(observation_after(o,m),second.text,'rewrite')
        messages3,suffix2,ids3=feedback_tokens(tokenizer,messages2,second,r['execution'],'predict')
        third=Segment(ids3,tokenizer(output('predict'),add_special_tokens=False)['input_ids'],output('predict'),'predict')
        tokens,mask=pack_segments([first,second,third],[suffix1,suffix2])
        self.assertEqual(sum(mask),sum(len(s.completion_ids) for s in [first,second,third]))
        self.assertEqual(mask.count(0),len(suffix1)+len(suffix2))
        self.assertGreater(len(suffix1),0);self.assertGreater(len(suffix2),0)
        self.assertEqual(tokens[:len(first.completion_ids)],first.completion_ids)
        self.assertNotEqual(suffix1[0],tokenizer.eos_token_id)
        self.assertEqual(len(messages3),6)
        third.prompt_ids=third.prompt_ids[:-1]
        with self.assertRaises(ValueError):pack_segments([first,second,third],[suffix1,suffix2])


class KeywordGenerationTests(unittest.TestCase):
    def test_actual_HF_generate_stops_each_row_and_excludes_padding_EOS(self):
        from contextlib import contextmanager
        from unittest.mock import patch
        import torch
        from transformers import Qwen2Config,Qwen2ForCausalLM,GenerationConfig,LogitsProcessor
        from train_grpo import ThinkWithGraphTrainer
        torch.set_num_threads(1)
        texts=[output('modify'),output('modify').replace('Specific graph evidence','More detailed specific graph evidence')]
        class Characters:
            pad_token_id=0;eos_token_id=0
            def decode(self,ids,skip_special_tokens=True):return ''.join(chr(int(i)) for i in ids if int(i)!=0)
        class ForceTokens(LogitsProcessor):
            def __call__(self,ids,scores):
                scores.fill_(-float('inf'));step=ids.shape[1]-1
                for i,text in enumerate(texts):scores[i,ord(text[step]) if step<len(text) else 0]=0
                return scores
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=256,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,eos_token_id=0,pad_token_id=0))
        original=model.generate
        model.generate=lambda **kw:original(**kw,logits_processor=[ForceTokens()])
        trainer=object.__new__(ThinkWithGraphTrainer)
        trainer.model_wrapped=model;trainer.accelerator=SimpleNamespace(device=torch.device('cpu'))
        trainer.processing_class=Characters();trainer.pad_token_id=trainer.eos_token_id=0
        trainer.mrra_config={'generation':{'modify_tokens':200,'edit_tokens':400,'prompt_tokens':512,
                                         'feedback_prompt_tokens':1024,'context_tokens':2048}}
        trainer.data_args=SimpleNamespace(memory_stress_tokens=0)
        trainer.args=SimpleNamespace(ds3_gather_for_generation=True)
        trainer.is_fsdp_enabled=False
        trainer.generation_config=GenerationConfig(do_sample=False,pad_token_id=0,eos_token_id=0)
        @contextmanager
        def unwrap(*args,**kw):yield model
        with patch('train_grpo.unwrap_model_for_generation',unwrap):
            rows=trainer._generate_batch([[65],[66]],'modify')
        self.assertEqual([r.text for r in rows],texts)
        for row,text in zip(rows,texts):
            self.assertEqual(row.completion_ids,[ord(c) for c in text])
            self.assertFalse(row.truncated)


if __name__=='__main__':unittest.main()
