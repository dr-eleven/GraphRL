"""Length termination must be candidate-local, precede tools, and preserve earned stage rewards."""
import os
from pathlib import Path
from functools import partial
from types import SimpleNamespace
import sys,json,types,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from episode import GraphEnvironment
from test_native_tools import sample,invoke,answer

class LengthLimitTests(unittest.TestCase):
 def test_terminal_prediction_retains_prior_rewards_even_when_answer_looks_correct(self):
  env=GraphEnvironment();env.reset(sample());messages=[]
  for stage in ['del_node','del_edge','add_edge','rewrite']:messages+=invoke(env,stage)
  env._abort_generation('prediction','stage_token_limit',512,512)
  self.assertEqual(env._score([*messages,answer()]),1.8)
  self.assertEqual(env._record['finish_reason'],'token_limit')
  self.assertEqual([env._record['metrics'][k] for k in ['structure_reward','behavior_reward','task_reward']], [.8,1,0])
  self.assertEqual(env._record['metrics']['token_limit_exceeded'],1)
  self.assertEqual(len(env._invocations),4)

 def test_per_row_criteria(self):
  import torch
  from generation_limits import CandidateTokenBudget
  stop=CandidateTokenBudget(10,[3,7])
  self.assertEqual(stop(torch.ones(2,13,dtype=torch.long),None).tolist(),[True,False])
  self.assertEqual(stop(torch.ones(2,17,dtype=torch.long),None).tolist(),[True,True])

 def test_hf_generate_stops_each_row_at_its_budget(self):
  import torch
  from transformers import Qwen2Config,Qwen2ForCausalLM,LogitsProcessor,LogitsProcessorList,StoppingCriteriaList
  from generation_limits import CandidateTokenBudget
  class ForceText(LogitsProcessor):
   def __call__(self,input_ids,scores):
    scores.fill_(-float('inf'));scores[:,1]=0;return scores
  torch.set_num_threads(2)
  model=Qwen2ForCausalLM(Qwen2Config(vocab_size=8,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,eos_token_id=2,pad_token_id=0))
  with torch.no_grad():
   result=model.generate(input_ids=torch.tensor([[3,4],[3,4]]),attention_mask=torch.ones(2,2,dtype=torch.long),max_new_tokens=20,do_sample=False,logits_processor=LogitsProcessorList([ForceText()]),stopping_criteria=StoppingCriteriaList([CandidateTokenBudget(2,[3,7])]))
  self.assertEqual(result[0,2:].tolist(),[1,1,1,0,0,0,0])
  self.assertEqual(result[1,2:].tolist(),[1]*7)

 def test_boundary_eos_is_not_failure_and_total_budget_stops(self):
  import torch
  from train_grpo import ThinkWithGraphTrainer,GraphRLDataArguments
  from trl import GRPOTrainer
  envs=[GraphEnvironment(),GraphEnvironment()]
  for env in envs:env.reset(sample())
  t=object.__new__(ThinkWithGraphTrainer);t.data_args=GraphRLDataArguments(action_max_new_tokens=3)
  t.environments=envs;t._initial_prompt_lengths=None;t._active_rollout_indices=None
  t.max_completion_length=9;t.eos_token_ids=[99]
  model=SimpleNamespace(generate=lambda **kw:None);t.model_wrapped=model
  t.accelerator=SimpleNamespace(unwrap_model=lambda m:m);t.generation_config=SimpleNamespace(max_new_tokens=77)
  with patch.object(GRPOTrainer,'_generate_single_turn',return_value=([[1,2,99],[1,2,3]],None,None)):
   t._generate_single_turn([[8],[8]],None,{},1)
  self.assertFalse(envs[0]._generation_limit_events)
  self.assertEqual(envs[1]._generation_limit_events[0]['reason'],'stage_token_limit')
  self.assertEqual(t.generation_config.max_new_tokens,77)
  t._active_rollout_indices=[0]
  with patch.object(GRPOTrainer,'_generate_single_turn',return_value=([[1,2]],None,None)):
   t._generate_single_turn([[8]*8],None,{},1)
  self.assertEqual(envs[0]._generation_limit_events[0]['reason'],'total_completion_limit')

 def test_native_loop_keeps_sibling_and_never_dispatches_limited_json(self):
  import torch
  from transformers import AutoTokenizer,PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
  from tokenizers import Tokenizer,models,pre_tokenizers,decoders
  from train_grpo import ThinkWithGraphTrainer,GraphRLTrainingArguments,GraphRLDataArguments,GraphRLDataset,graph_reward
  original=AutoTokenizer.from_pretrained(os.environ.get('TRAIN_MODEL', '/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct'),local_files_only=True)
  backend=Tokenizer(models.BPE(vocab={c:i for i,c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))},merges=[]))
  backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False);backend.decoder=decoders.ByteLevel()
  tok=PreTrainedTokenizerFast(tokenizer_object=backend,chat_template=original.chat_template,eos_token='<|im_end|>',pad_token='<|endoftext|>',additional_special_tokens=['<|im_start|>','<tool_call>','</tool_call>','<tool_response>','</tool_response>']);tok.padding_side='left'
  tool_text=lambda stage,analysis:'<tool_call>\n'+json.dumps({'name':stage,'arguments':{'analysis':analysis,'actions':[]}})+'\n</tool_call>'
  long=tok.encode(tool_text('del_node','A'*60),add_special_tokens=False);cap=len(long)
  model=Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok),hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=32768,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id))
  dataset=GraphRLDataset([sample()]).to_hf_dataset()
  with tempfile.TemporaryDirectory() as tmp:
   args=GraphRLTrainingArguments(output_dir=tmp,use_cpu=True,bf16=False,fp16=False,tf32=False,deepspeed=None,use_liger_kernel=False,gradient_checkpointing=False,per_device_train_batch_size=2,num_generations=2,beta=0,max_completion_length=8192,report_to='none',save_strategy='no')
   trainer=ThinkWithGraphTrainer(model=model,args=args,train_dataset=dataset,processing_class=tok,reward_funcs=[graph_reward],environment_factory=partial(GraphEnvironment,max_turns=4),data_args=GraphRLDataArguments(max_prompt_length=20000,action_max_new_tokens=cap,prediction_max_new_tokens=128))
   calls=[]
   def scripted(self,input_ids,**kwargs):
    outputs=[]
    for local,idx in enumerate(trainer._active_rollout_indices):
     env=trainer.environments[idx];calls.append((idx,env._state.current_tool))
     if idx==0:seq=long # Valid native JSON, but max-token termination without EOS.
     elif env._state.done:seq=tok.encode(json.dumps({'analysis':'Books.','prediction':'library'})+tok.eos_token,add_special_tokens=False)
     else:seq=tok.encode(tool_text(env._state.current_tool,'Books.')+tok.eos_token,add_special_tokens=False)
     outputs.append(seq)
    width=max(map(len,outputs));tail=torch.tensor([s+[tok.pad_token_id]*(width-len(s)) for s in outputs],device=input_ids.device)
    return torch.cat([input_ids,tail],dim=1)
   trainer.model.generate=types.MethodType(scripted,trainer.model)
   inputs=[dataset[0],dataset[0]]
   torch.set_num_threads(2)
   result=trainer._generate_and_score_completions(inputs)
   a,b=trainer.environments
   self.assertEqual(len(a._invocations),0)
   self.assertEqual(a._record['reward'],0)
   self.assertEqual(a._record['finish_reason'],'token_limit')
   self.assertEqual(len(b._invocations),4)
   self.assertEqual(b._record['reward'],3)
   self.assertEqual([x[0] for x in calls].count(0),1)
   self.assertEqual(result['advantages'].numel(),2)
   self.assertGreater(sum(a._record['generation_limit_events'][0]['generated_tokens'] for _ in [0]),0)

 def _native_format_case(self, mode):
  import torch
  from transformers import AutoTokenizer,PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
  from tokenizers import Tokenizer,models,pre_tokenizers,decoders
  from train_grpo import ThinkWithGraphTrainer,GraphRLTrainingArguments,GraphRLDataArguments,GraphRLDataset,graph_reward
  original=AutoTokenizer.from_pretrained(os.environ.get('TRAIN_MODEL', '/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct'),local_files_only=True)
  backend=Tokenizer(models.BPE(vocab={c:i for i,c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))},merges=[]))
  backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False);backend.decoder=decoders.ByteLevel()
  tok=PreTrainedTokenizerFast(tokenizer_object=backend,chat_template=original.chat_template,eos_token='<|im_end|>',pad_token='<|endoftext|>',additional_special_tokens=['<|im_start|>','<tool_call>','</tool_call>','<tool_response>','</tool_response>']);tok.padding_side='left'
  tool_text=lambda stage,analysis:'<tool_call>\n'+json.dumps({'name':stage,'arguments':{'analysis':analysis,'actions':[]}})+'\n</tool_call>'
  bad = {
    'multicall': tool_text('del_node','Books.')+'\n'+tool_text('del_edge','Books.'),
    'malformed': '<tool_call>{broken json}</tool_call>',
    'unknown': tool_text('unknown','Books.'),
    'prediction': json.dumps({'prediction':'library'}),
    'middle': 'No tool call in the add_edge stage.',
  }[mode]
  long=tok.encode(bad+tok.eos_token,add_special_tokens=False);cap=1024
  model=Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok),hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=32768,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id))
  dataset=GraphRLDataset([sample()]).to_hf_dataset()
  with tempfile.TemporaryDirectory() as tmp:
   args=GraphRLTrainingArguments(output_dir=tmp,use_cpu=True,bf16=False,fp16=False,tf32=False,deepspeed=None,use_liger_kernel=False,gradient_checkpointing=False,per_device_train_batch_size=2,num_generations=2,beta=0,max_completion_length=8192,report_to='none',save_strategy='no')
   trainer=ThinkWithGraphTrainer(model=model,args=args,train_dataset=dataset,processing_class=tok,reward_funcs=[graph_reward],environment_factory=partial(GraphEnvironment,max_turns=4),data_args=GraphRLDataArguments(max_prompt_length=20000,action_max_new_tokens=cap,prediction_max_new_tokens=128))
   calls=[]
   def scripted(self,input_ids,**kwargs):
    outputs=[]
    for local,idx in enumerate(trainer._active_rollout_indices):
     env=trainer.environments[idx];calls.append((idx,env._state.current_tool))
     if idx==0 and (mode not in ('prediction','middle') or
                   mode=='prediction' and env._state.done or
                   mode=='middle' and env._state.current_tool=='add_edge'):seq=long
     elif env._state.done:seq=tok.encode(json.dumps({'analysis':'Books.','prediction':'library'})+tok.eos_token,add_special_tokens=False)
     else:seq=tok.encode(tool_text(env._state.current_tool,'Books.')+tok.eos_token,add_special_tokens=False)
     outputs.append(seq)
    width=max(map(len,outputs));tail=torch.tensor([s+[tok.pad_token_id]*(width-len(s)) for s in outputs],device=input_ids.device)
    return torch.cat([input_ids,tail],dim=1)
   trainer.model.generate=types.MethodType(scripted,trainer.model)
   inputs=[dataset[0],dataset[0]]
   torch.set_num_threads(2)
   result=trainer._generate_and_score_completions(inputs)
   a,b=trainer.environments
   self.assertEqual(len(a._invocations),{'prediction':4,'middle':2}.get(mode,0))
   self.assertEqual(a._record['reward'],{'prediction':1.8,'middle':.9}.get(mode,0))
   self.assertEqual(a._record['finish_reason'],'format_error')
   self.assertIn('FORMAT_ERROR',a._record['format_feedback'])
   self.assertEqual(a._record['messages'][-1]['name'],'format_error')
   self.assertTrue(a._record['format_errors'])
   self.assertEqual(len(b._invocations),4)
   self.assertEqual(b._record['reward'],3)
   self.assertEqual([x[0] for x in calls].count(0),{'prediction':5,'middle':3}.get(mode,1))
   self.assertEqual(result['advantages'].numel(),2)
   self.assertFalse(a._record['generation_limit_events'])

 def test_native_format_errors_stop_only_failed_candidate(self):
  for mode in ['multicall','malformed','unknown','prediction','middle']:
   with self.subTest(mode=mode):
    self._native_format_case(mode)

if __name__=='__main__':unittest.main()
