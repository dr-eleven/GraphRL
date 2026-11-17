import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from trajectory_limits import inspect_trace,trainable_groups

class TrajectoryLimitTests(unittest.TestCase):
    def test_exact_limit_and_whole_flow_overflow(self):
        safe={'prompt_ids':[1]*4096,'response_ids':[2]*2048}
        self.assertFalse(inspect_trace([safe],6144)['offending_turns'])
        over={'prompt_ids':[1]*6000,'response_ids':[2]*145}
        audit=inspect_trace([safe,over,safe],6144)
        self.assertEqual(audit['offending_turns'],[2])
        self.assertEqual(audit['max_total_tokens'],6145)
    def test_each_source_requires_two_retained_flows(self):
        tasks=[('a',{'data_id':'A'},0),('b',{'data_id':'B'},0)]
        originals={str(i):{'data_id':g} for i,g in enumerate(['A','A','B','B'])}
        self.assertTrue(trainable_groups(tasks,originals))
        originals.pop('3');self.assertFalse(trainable_groups(tasks,originals))
