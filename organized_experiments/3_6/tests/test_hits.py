import sys,json,unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'reference_tests')]
import test_stage_rewards as helpers
from hit_reward import stage_hits,rewrite_result

class HitTests(unittest.TestCase):
    def test_structural_hits(self):
        row=helpers.sample();row['noisy_graph']['nodes']['2']='noise'
        row['noisy_graph']['edges']=[{'u':0,'v':2}]
        env=helpers.run(row,{'del_node':[{'op':'DELETE_NODE','node':2}], 'add_edge':[{'op':'ADD_EDGE','u':0,'v':1}]})
        m=env._record['metrics']
        self.assertAlmostEqual(m['hit_reward'],.2)
        self.assertEqual(m['hit_del_edge_count'],0)
        self.assertAlmostEqual(env._record['reward'],3.2)

    def test_explicit_noise_edge(self):
        row=helpers.sample();row['clean_graph']['edges']=[]
        env=helpers.run(row,{'del_edge':[{'op':'DEL_EDGE','u':0,'v':1}]})
        self.assertEqual(env._record['metrics']['hit_reward'],.1)

    def test_self_repair_no_hits(self):
        env=helpers.run(helpers.sample(),{'del_edge':[{'op':'DEL_EDGE','u':0,'v':1}], 'add_edge':[{'op':'ADD_EDGE','u':0,'v':1}]})
        self.assertEqual(env._record['metrics']['hit_reward'],0)
        self.assertEqual(env._record['metrics']['behavior_add_edge'],.25)

    def test_duplicate_batch_rolls_back(self):
        row=helpers.sample();row['noisy_graph']['edges']=[]
        a={'op':'ADD_EDGE','u':0,'v':1}
        env=helpers.run(row,{'add_edge':[a,a]})
        self.assertEqual(env._record['metrics']['hit_reward'],0)
        self.assertEqual(env._state.graph['edges'],[])

    def test_rewrite_thresholds(self):
        for before,after,expected in [(.8,.85,True),(.81,.85,False),(.7,.84,False),(.9,.86,False)]:
            sim=lambda a,b: before if a=='before' else after
            self.assertEqual(rewrite_result('before','after','clean',sim)[0],expected)
        self.assertTrue(rewrite_result('before','clean','clean',lambda a,b:1 if a==b else .99)[0])
        self.assertFalse(rewrite_result('clean','other','clean',lambda a,b:1)[0])

    def test_rewrite_integrated(self):
        row=helpers.sample();row['noisy_graph']['nodes']['0']='corrupt'
        with patch('hit_reward.semantic',side_effect=lambda a,b: 1.0 if a==b else .99):
            env=helpers.run(row,{'rewrite':[{'op':'REWRITE','node':0,'text':row['clean_graph']['nodes']['0']}]})
        self.assertEqual(env._record['metrics']['hit_rewrite_reward'],.2)
        self.assertAlmostEqual(env._record['reward'],3.2)

    def test_failed_prediction_preserves_hits(self):
        row=helpers.sample();row['noisy_graph']['nodes']['2']='noise'
        env=helpers.GraphEnvironment();env.reset(row);messages=[]
        for name in helpers.TOOL_ORDER:
            messages+=helpers.invoke(env,name,actions=[{'op':'DELETE_NODE','node':2}] if name=='del_node' else [])
        reward=env._score(messages+[{'role':'assistant','content':'bad format'}])
        self.assertAlmostEqual(reward,1.9)
        self.assertEqual(env._record['metrics']['hit_reward'],.1)

if __name__=='__main__':unittest.main()
