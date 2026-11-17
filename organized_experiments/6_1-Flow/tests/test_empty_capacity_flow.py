import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import bootstrap
from flow_transport import GraphFlowDaemon

class CapacityFlowTests(unittest.TestCase):
    def test_capacity_before_planner_is_logged_and_excluded_without_resampling(self):
        class FakeEngine:
            def __init__(self,*args,**kwargs):
                self.calls=[]
                self.http=type('HTTP',(),{'close':lambda self:None})()
        with tempfile.TemporaryDirectory() as tmp:
            daemon=GraphFlowDaemon.__new__(GraphFlowDaemon)
            daemon.config=dict(workers=4,planner_flows_per_server=2,train_temperature=.7,seed=42,fixed_url='http://fixed',fixed_served_model='fixed',tool_steps=10,agent_timeout=500)
            daemon.output=Path(tmp);daemon.train_information={'model':'/Qwen/Qwen2.5-3B-Instruct'}
            daemon.clear_data_and_server();daemon.step=1
            daemon.backend_llm_server_addresses=['a','b']
            daemon.tasks=[(str(i),dict(episode='{}',data_id=str(i//6)),i%6) for i in range(12)]
            daemon._total_tasks_queued=12
            with patch('flow_transport.Engine',FakeEngine),patch('flow_transport.run_workflow',return_value=dict(reward=0,capacity_failure='Module output hit max_tokens')) as workflow:
                daemon.run_until_all_finished()
            self.assertEqual(workflow.call_count,12)
            self.assertFalse(daemon.batch_is_trainable())
            self.assertEqual(daemon.discarded_flows,12)
            self.assertFalse(daemon._completed_rollouts)
            records=[json.loads(line) for line in (Path(tmp)/'train_rollouts.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),12)
            self.assertTrue(all(r['discard_reason']=='NO_PLANNER_TRACE_CAPACITY' for r in records))
