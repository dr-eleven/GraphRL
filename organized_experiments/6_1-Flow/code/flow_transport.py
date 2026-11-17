"""In-process rollout transport; reuse AgentFlow's original transition/reward tensorization."""
import json, os, uuid
from threading import BoundedSemaphore
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import bootstrap
from engines import Engine
from workflow import run_workflow
from trajectory_limits import inspect_trace,trainable_groups
from agentflow.types import Rollout, Triplet
from agentflow.verl.daemon import AgentModeDaemon as OriginalDaemon

class GraphFlowDaemon(OriginalDaemon):
    def __init__(self,port,train_rollout_n,train_information,tokenizer,mini_batch_size,pad_token_id,**kwargs):
        self.train_rollout_n=train_rollout_n;self.train_information=train_information;self.tokenizer=tokenizer;self.mini_batch_size=mini_batch_size;self.pad_token_id=pad_token_id;self.reward_fillna_value=None
        self.config=json.loads(Path(os.environ['FLOW_SETTINGS']).read_text())
        self.output=Path(os.environ['FLOW_RUN_DIR']);self.step=0;self.clear_data_and_server()
    def start(self):pass
    def set_up_data_and_server(self,data,server_addresses,is_train=True):
        self.is_train=is_train;self.backend_llm_server_addresses=server_addresses;self.step+=int(is_train)
        count=len(next(iter(data.values())));self.tasks=[]
        if is_train and count!=2:raise ValueError('One update requires exactly two source samples.')
        for i in range(count):
            original={k:v[i] for k,v in data.items()};original['data_id']=str(uuid.uuid4())
            for j in range(self.train_rollout_n if is_train else 1):
                self.tasks.append((str(uuid.uuid4()),original,j))
        self._total_tasks_queued=len(self.tasks)
    def run_until_all_finished(self):
        model='/'.join(self.train_information['model'].split('/')[-2:])
        slots={address:BoundedSemaphore(self.config.get('planner_flows_per_server',2)) for address in self.backend_llm_server_addresses}
        def run(item):
            address=self.backend_llm_server_addresses[item[2]%len(self.backend_llm_server_addresses)]
            # Hold a flow slot, not the GPU: the server remains free to execute
            # the other flow's Planner while this flow waits for fixed roles.
            with slots[address]:return run_flow(item,address)
        def run_flow(item,address):
            rollout_id,original,candidate=item;sample=json.loads(original['episode'])
            trace=[]
            policy=Engine('http://'+address+'/v1',model,self.config['train_temperature'] if self.is_train else 0.,max_tokens=self.config.get('max_tokens',8192),trace=trace,role='planner',seed=self.config['seed']+candidate+self.step*100)
            fixed=Engine(self.config['fixed_url'],self.config['fixed_served_model'],0.,max_tokens=self.config.get('max_tokens',8192),role='fixed')
            try:result=run_workflow(sample,policy,fixed,max_steps=self.config['tool_steps'],max_time=self.config['agent_timeout'])
            finally:policy.http.close();fixed.http.close()
            if not trace:
                # A fixed-role capacity failure can occur before the first
                # Planner call. It supplies no policy tokens, so exclude the
                # entire flow instead of fabricating a transition or reward.
                if result.get('capacity_failure') is not None:
                    result.update(rollout_id=rollout_id,data_id=original['data_id'],candidate=candidate,step=self.step,
                        discarded=True,discard_reason='NO_PLANNER_TRACE_CAPACITY',planner_calls=policy.calls,fixed_calls=fixed.calls)
                    return rollout_id,original,None,result
                policy.diagnostic('missing_planner_trace',workflow_result=result,fixed_calls=fixed.calls)
                raise RuntimeError('No Planner transition without capacity failure; diagnostic saved.')
            audit=inspect_trace(trace,self.config.get('planner_max_total_tokens',8192))
            result.update(rollout_id=rollout_id,data_id=original['data_id'],candidate=candidate,step=self.step,token_limit_audit=audit,discarded=bool(audit['offending_turns']))
            if audit['offending_turns']:
                result.update(discard_reason='PLANNER_TOTAL_TOKEN_LIMIT',planner_calls=policy.calls,fixed_calls=fixed.calls)
                return rollout_id,original,None,result
            triplets=[Triplet(prompt={'token_ids':t['prompt_ids']},response={'token_ids':t['response_ids']},reward=result['reward']) for t in trace]
            result.update(rollout_id=rollout_id,data_id=original['data_id'],candidate=candidate,step=self.step)
            return rollout_id,original,Rollout(rollout_id=rollout_id,final_reward=result['reward'],triplets=triplets),result
        # Never resample based on reward. Infrastructure failures abort this batch.
        path=self.output/('train_rollouts.jsonl' if self.is_train else 'val_rollouts.jsonl')
        # Flush complete flows as they finish in task order. A partial batch is
        # evidence for diagnostics only; any failure still aborts the update.
        with path.open('a') as stream, ThreadPoolExecutor(max_workers=self.config['workers']) as pool:
            for rid,original,rollout,result in pool.map(run,self.tasks):
                if rollout is not None:
                    self._completed_rollouts[rid]=rollout;self._task_id_to_original_sample[rid]=original
                stream.write(json.dumps(result,ensure_ascii=True)+'\n');stream.flush()
        self.discarded_flows=self._total_tasks_queued-len(self._completed_rollouts)
    def batch_is_trainable(self):
        return trainable_groups(self.tasks,self._task_id_to_original_sample)
    def get_train_data_batch(self,max_prompt_length,max_response_length,device):
        # Keep the native per-decision Flow-GRPO grouping and terminal reward propagation.
        # Dynamic padding reduces memory; reject overlength rather than truncate the policy target.
        prompts=[len(t.prompt['token_ids']) for r in self._completed_rollouts.values() for t in r.triplets]
        responses=[len(t.response['token_ids']) for r in self._completed_rollouts.values() for t in r.triplets]
        if max(prompts)>max_prompt_length or max(responses)>max_response_length:raise RuntimeError('Trace exceeds configured capacity; refusing silent training truncation')
        limit=self.config.get('planner_max_total_tokens',8192) if hasattr(self,'config') else 8192
        if any(len(t.prompt['token_ids'])+len(t.response['token_ids'])>limit for r in self._completed_rollouts.values() for t in r.triplets):raise RuntimeError('Over-limit flow leaked into training batch')
        batch,metrics=super().get_train_data_batch(max(prompts),max(responses),device)
        if batch.batch['is_drop_mask'].any():raise RuntimeError('Unexpected dropped Planner transition')
        return batch,metrics
    def clear_data_and_server(self):
        self._completed_rollouts={};self._task_id_to_original_sample={};self._total_tasks_queued=0;self.is_train=True;self.tasks=[]
