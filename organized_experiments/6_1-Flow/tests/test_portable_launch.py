import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from resolve_settings import resolve
from check_launch import validate
from engines import Engine

class PortableLaunchTests(unittest.TestCase):
    def test_relocated_repository_and_explicit_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp)/'different location'
            cfg=resolve(ROOT/'configs/settings.json',repo,{})
            self.assertEqual(cfg['model'],str(repo/'models/Qwen2.5-7B-Instruct'))
            self.assertEqual(cfg['train_source'],str(repo/'1000_oneclass_sorted/train_grpo_episodes_redesign.json'))
            cfg=resolve(ROOT/'configs/settings.json',repo,{'TRAIN_MODEL':'/external/model','TEST_ONECLASS_EPISODES':'custom/test.json'})
            self.assertEqual(cfg['model'],'/external/model')
            self.assertEqual(cfg['test_sources']['test_data_600_oneclass'],str(repo/'custom/test.json'))

    def test_eight_gpu_split_and_no_overlapping_roles(self):
        cfg=json.loads((ROOT/'configs/settings.json').read_text())
        yaml_cfg=yaml.safe_load((ROOT/'configs/graph_flow.yaml').read_text())
        report=validate(cfg,yaml_cfg)
        self.assertEqual(report['total_gpus'],8)
        self.assertEqual(report['max_generated_tokens'],8192)
        self.assertEqual(report['workers'],12)
        with self.assertRaises(ValueError):validate(cfg,yaml_cfg,fixed_gpus='5,7')
        with self.assertRaises(ValueError):validate(cfg,yaml_cfg,train_gpus='0,1')

    def test_http_payload_really_uses_8192_generation_tokens(self):
        engine=Engine('http://localhost:1/v1','test')
        response=type('Response',(),{'status_code':200,'json':lambda self:{'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}]}})()
        with patch.object(engine.http,'post',return_value=response) as post:
            engine('test')
        self.assertEqual(post.call_args.kwargs['json']['max_tokens'],8192)
        engine.http.close()

    def test_twelve_flows_reach_all_six_planner_servers_with_8192_cap(self):
        from collections import Counter
        from flow_transport import GraphFlowDaemon
        class FakeEngine:
            def __init__(self,url,model,*args,**kwargs):
                self.url=url;self.trace=kwargs.get('trace');self.max_tokens=kwargs['max_tokens'];self.calls=[]
                self.http=type('HTTP',(),{'close':lambda self:None})()
        seen=[]
        def workflow(sample,policy,fixed,**kwargs):
            seen.append((policy.url,policy.max_tokens,fixed.max_tokens))
            policy.trace.append(dict(prompt_ids=[1],response_ids=[2]))
            return dict(reward=1.,capacity_failure=None)
        with tempfile.TemporaryDirectory() as tmp:
            daemon=GraphFlowDaemon.__new__(GraphFlowDaemon)
            daemon.config=json.loads((ROOT/'configs/settings.json').read_text())
            daemon.output=Path(tmp);daemon.train_information={'model':'/Qwen/Qwen2.5-7B-Instruct'}
            daemon.clear_data_and_server();daemon.step=1
            daemon.backend_llm_server_addresses=[f'host:{i}' for i in range(6)]
            daemon.tasks=[(str(i),dict(episode='{}',data_id=str(i//6)),i%6) for i in range(12)]
            daemon._total_tasks_queued=12
            with patch('flow_transport.Engine',FakeEngine),patch('flow_transport.run_workflow',side_effect=workflow):
                daemon.run_until_all_finished()
            self.assertEqual(Counter(x[0] for x in seen),{f'http://host:{i}/v1':2 for i in range(6)})
            self.assertTrue(all(x[1:]==(8192,8192) for x in seen))
            self.assertTrue(daemon.batch_is_trainable())
            self.assertEqual(daemon.discarded_flows,0)
