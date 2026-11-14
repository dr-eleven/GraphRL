from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import json
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from test_binary_controller import sample, tool
from episode import GraphEnvironment
from budgets import RolloutBudget
from runtime import InfrastructureError


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok=AutoTokenizer.from_pretrained('/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct',local_files_only=True)

    def harness(self,tmp,script,sources=(7,7,8,8),budget=None):
        import torch
        from generation_limits import TokenLimitToolsMixin
        b=budget or RolloutBudget()
        class Harness(TokenLimitToolsMixin):
            def _tokenize_prompts(self,prompts):return [self._tokenizer.encode('task') for p in prompts],None,{}
            def _generate_single_turn(self,prompt_ids,*args,**kwargs):
                outputs=[]
                for i,ids in zip(self._active_rollout_indices,prompt_ids):
                    self.histories[i].append(list(ids))
                    response=script(self,i)
                    outputs.append(self._tokenizer.encode(response+self._tokenizer.eos_token,add_special_tokens=False))
                return outputs,None,None
        h=Harness();h._tokenizer=self.tok
        h.eos_token_ids=[self.tok.eos_token_id]
        h.model=SimpleNamespace(training=False,config=SimpleNamespace(model_type='qwen2',max_position_embeddings=32768))
        h._is_vlm=False;h.use_vllm=False
        h.accelerator=SimpleNamespace(device=torch.device('cpu'),gather=lambda x:x,process_index=0)
        h.state=SimpleNamespace(global_step=9,num_input_tokens_seen=0)
        h.args=SimpleNamespace(output_dir=tmp)
        h.max_completion_length=b.max_completion_tokens
        h.data_args=SimpleNamespace(rollout_budget=lambda *args:b,generation_micro_batch_size=1)
        h._metrics=defaultdict(lambda:defaultdict(list))
        h.histories=defaultdict(list);h.environments=[]
        for source in sources:
            e=GraphEnvironment(mode='free',budget=b);s=sample();s['source_index']=source;e.reset(s)
            h.environments.append(e)
        return h

    def test_infra_retries_entire_group_only(self):
        failed=[False]
        def script(h,i):
            if i==0 and not failed[0]:
                failed[0]=True;raise InfrastructureError('injected worker error')
            return '{"analysis":"Reason","prediction":"library"}'
        with tempfile.TemporaryDirectory() as tmp:
            h=self.harness(tmp,script);result=h._generate([[]]*4)
            self.assertEqual([e._infra_group_retries for e in h.environments],[1,1,0,0])
            self.assertEqual([len(h.histories[i]) for i in range(4)],[2,2,1,1])
            self.assertTrue(all(e._score(c)==1 for e,c in zip(h.environments,result[3])))
            log=json.loads((Path(tmp)/'infra_rank0.jsonl').read_text())
            self.assertFalse(log['valid_for_training']);self.assertEqual(log['failed_sources'],[7])

    def test_persistent_infra_raises_after_two_retries(self):
        def script(h,i):raise InfrastructureError('persistent error')
        with tempfile.TemporaryDirectory() as tmp:
            h=self.harness(tmp,script,sources=(7,7))
            with self.assertRaises(InfrastructureError):h._generate([[]]*2)
            self.assertEqual([len(h.histories[i]) for i in range(2)],[3,3])
            self.assertEqual(len((Path(tmp)/'infra_rank0.jsonl').read_text().splitlines()),3)

    def test_model_error_never_triggers_infra_resample(self):
        with tempfile.TemporaryDirectory() as tmp:
            h=self.harness(tmp,lambda h,i:'bad json',sources=(7,7))
            out=h._generate([[]]*2)
            self.assertEqual([e._infra_group_retries for e in h.environments],[0,0])
            for e,c in zip(h.environments,out[3]):
                self.assertEqual(e._turns,1);self.assertEqual(e._score(c),0)
                self.assertEqual(e._finish_reason,'parse_error')
            self.assertFalse((Path(tmp)/'infra_rank0.jsonl').exists())

    def test_first_and_later_schema_errors_both_return_feedback(self):
        sequence=[tool('unknown'),tool('del_node'),tool('unknown'),'{"analysis":"Reason","prediction":"library"}']
        with tempfile.TemporaryDirectory() as tmp:
            h=self.harness(tmp,lambda h,i:sequence[h.environments[i]._turns],sources=(7,))
            out=h._generate([[]])
            self.assertEqual(h.environments[0]._score(out[3][0]),1)
            self.assertEqual(len(h.histories[0]),4)
            self.assertIn('ENVIRONMENT_FEEDBACK',self.tok.decode(h.histories[0][1]))
            self.assertEqual(h.environments[0]._edit_errors,2)
            self.assertEqual(sum(out[2][0]),h.environments[0]._model_tokens)
            self.assertTrue(any(x==0 for x in out[2][0]))

    def test_later_parse_error_stops_without_error_obs(self):
        sequence=[tool('del_node'),'<tool_call>{']
        with tempfile.TemporaryDirectory() as tmp:
            h=self.harness(tmp,lambda h,i:sequence[h.environments[i]._turns],sources=(7,))
            out=h._generate([[]])
            e=h.environments[0]
            self.assertEqual(e._finish_reason,'parse_error')
            self.assertEqual(e._score(out[3][0]),0)
            self.assertEqual(len(h.histories[0]),2)
            self.assertEqual(len(out[3][0]),3)  # assistant, successful obs, final malformed assistant
            self.assertEqual(e._infra_group_retries,0)

    def test_feedback_snapshot_overflow_discards_graph(self):
        # A valid edit can grow the graph beyond feedback capacity; never commit it unseen.
        large='new text '*5000
        sequence=[tool('del_node',[dict(op='DELETE_NODE',node=2)]),'{"analysis":"Reason","prediction":"library"}']
        b=replace(RolloutBudget(),max_completion_tokens=2600)
        with tempfile.TemporaryDirectory() as tmp:
            h=self.harness(tmp,lambda h,i:sequence[h.environments[i]._turns],sources=(7,),budget=b)
            h.environments[0]._graph['nodes']['0']=large
            from copy import deepcopy
            before=deepcopy(h.environments[0]._graph)
            out=h._generate([[]])
            e=h.environments[0]
            self.assertEqual(e._graph,before)
            self.assertEqual(e._tool_successes,0)
            self.assertEqual(e._score(out[3][0]),0)
            self.assertEqual(len(h.histories[0]),1)
            self.assertLessEqual(len(out[1][0]),b.max_completion_tokens)
            self.assertIn('FEEDBACK_BUDGET',self.tok.decode(out[1][0]))

    def test_ten_schema_errors_no_resampling_or_extra_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            h=self.harness(tmp,lambda h,i:tool('unknown'),sources=(7,7))
            out=h._generate([[],[]])
            for i,e in enumerate(h.environments):
                self.assertEqual(e._turns,10);self.assertEqual(e._edit_errors,10)
                self.assertEqual(len(h.histories[i]),10)
                self.assertEqual(e._score(out[3][i]),0)
                self.assertEqual(e._infra_group_retries,0)
                decoded=self.tok.decode(out[1][i])
                self.assertTrue(decoded.endswith('<|im_end|>'))
                self.assertEqual(len([m for m in out[3][i] if m['role']=='assistant']),10)

    def test_qwen_suffix_matches_chat_template(self):
        from generation_limits import feedback_suffix
        for role in ['tool','user']:
            msg=dict(role=role,content='ENVIRONMENT_FEEDBACK\nhello')
            if role=='tool':msg['name']='del_node'
            prefix=[dict(role='user',content='task'),dict(role='assistant',content='',tool_calls=[
                dict(type='function',function=dict(name='del_node',arguments=dict(actions=[])))])]
            ids=self.tok.apply_chat_template(prefix,tokenize=True,add_generation_prompt=False,return_dict=False)
            # Qwen prints a newline following EOS; strip this template-only newline.
            eos=max(i for i,v in enumerate(ids) if v==self.tok.eos_token_id)
            ids=ids[:eos+1]
            full=self.tok.apply_chat_template(prefix+[msg],tokenize=True,add_generation_prompt=True,return_dict=False)
            self.assertEqual(ids+feedback_suffix(self.tok,msg,ids,[self.tok.eos_token_id]),full)
            boundary=feedback_suffix(self.tok,msg,[123],[self.tok.eos_token_id])
            self.assertEqual(boundary[0],self.tok.eos_token_id)  # external mask=0, never a sampled token
            other_eos=self.tok.convert_tokens_to_ids('<|endoftext|>')
            boundary=feedback_suffix(self.tok,msg,[other_eos],[self.tok.eos_token_id,other_eos])
            self.assertEqual(boundary[0],self.tok.eos_token_id)


if __name__=='__main__':unittest.main()
