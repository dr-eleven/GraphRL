"""Behavioral cases for stage-local rewards and termination credit."""
from copy import deepcopy
import unittest
from test_native_tools import sample, invoke, answer, GraphEnvironment, TOOL_ORDER
from reward import operation_needed


def run(row, edits=None):
    env = GraphEnvironment(); env.reset(row); messages = []
    for name in TOOL_ORDER:
        messages += invoke(env, name, actions=(edits or {}).get(name, []))
    env._score([*messages, answer()])
    return env


class StageRewardsTests(unittest.TestCase):
    def test_need_action_truth_table_for_each_tool(self):
        edits = {
            'del_node': [{'op':'DELETE_NODE', 'node':1}],
            'del_edge': [{'op':'DEL_EDGE', 'u':0, 'v':1}],
            'add_edge': [{'op':'ADD_EDGE', 'u':0, 'v':1}],
            'rewrite': [{'op':'REWRITE', 'node':0, 'text':'Restored prose'}],
        }
        for tool in TOOL_ORDER:
            for needed in [False, True]:
                for nonempty in [False, True]:
                    with self.subTest(tool=tool, needed=needed, nonempty=nonempty):
                        row=sample()
                        if tool=='del_node' and needed:
                            row['clean_graph']['nodes'].pop('1');row['clean_graph']['edges']=[]
                        elif tool=='del_edge' and needed:
                            row['clean_graph']['edges']=[]
                        elif tool=='add_edge':
                            row['noisy_graph']['edges']=[]
                            if not needed:row['clean_graph']['edges']=[]
                        elif tool=='rewrite' and needed:
                            row['noisy_graph']['nodes']['0']='Noisy prose'
                        env=run(row,{tool:edits[tool] if nonempty else []})
                        m=env._record['metrics']
                        self.assertEqual(m[f'operation_needed_{tool}'],needed)
                        self.assertEqual(m[f'behavior_{tool}'],.25 if needed==nonempty else 0)
                        self.assertEqual(m[f'structure_{tool}'],.2)

    def test_stage_entry_graph_changes_need(self):
        row=sample();row['noisy_graph']['nodes']['2']='Injected'
        row['noisy_graph']['edges'].append({'u':0,'v':2})
        env=run(row,{'del_node':[{'op':'DELETE_NODE','node':2}]})
        self.assertEqual(env._record['reward'],3)
        self.assertEqual(env._record['metrics']['operation_needed_del_edge'],0)
        missed=run(row)
        self.assertEqual(missed._record['metrics']['operation_needed_del_edge'],1)
        self.assertEqual(missed._record['metrics']['behavior_del_edge'],0)

    def test_earlier_wrong_deletion_creates_add_edge_need(self):
        env=run(sample(),{'del_edge':[{'op':'DEL_EDGE','u':0,'v':1}]})
        m=env._record['metrics']
        self.assertEqual(m['behavior_del_edge'],0)
        self.assertEqual(m['operation_needed_add_edge'],1)
        self.assertEqual(m['behavior_add_edge'],0)
        self.assertEqual(m['behavior_reward'],.5)

    def test_repair_earlier_edge_deletion(self):
        env=run(sample(), {'del_edge':[{'op':'DEL_EDGE','u':0,'v':1}],
            'add_edge':[{'op':'ADD_EDGE','u':0,'v':1}]})
        self.assertEqual(env._record['metrics']['behavior_add_edge'], .25)
        self.assertEqual(env._state.graph, sample()['noisy_graph'])

    def test_deleted_endpoint_is_not_actionable(self):
        env=run(sample(), {'del_node':[{'op':'DELETE_NODE','node':1}]})
        m=env._record['metrics']
        self.assertEqual(m['residual_count_add_edge'],1)
        self.assertEqual(m['actionable_count_add_edge'],0)
        self.assertEqual(m['behavior_add_edge'],.25)

    def test_format_and_length_failure_at_every_stage_preserve_prefix(self):
        for index,name in enumerate([*TOOL_ORDER,'prediction']):
            for length in [False,True]:
                with self.subTest(stage=name,length=length):
                    env=GraphEnvironment();env.reset(sample());messages=[]
                    for prior in TOOL_ORDER[:index]:messages+=invoke(env,prior)
                    if length:env._abort_generation(name,'stage_token_limit',512,512)
                    else:env._validate_turn({'role':'assistant','content':'not a tool or JSON'})
                    # Even a correct-looking answer after abort must not earn T.
                    value=env._score([*messages,answer()])
                    self.assertAlmostEqual(value,index*.45)
                    m=env._record['metrics']
                    self.assertAlmostEqual(m['structure_reward'],index*.2)
                    self.assertAlmostEqual(m['behavior_reward'],index*.25)
                    self.assertEqual(m['task_reward'],0)
                    self.assertIn('retained',env._record['format_feedback'])

    def test_illegal_action_only_loses_own_behavior_credit(self):
        env=run(sample(),{'del_node':[{'op':'DELETE_NODE','node':999}]})
        m=env._record['metrics']
        self.assertEqual((m['structure_reward'],m['behavior_reward'],m['task_reward']),(1,.75,1))

    def test_partial_or_wrong_legal_edit_still_counts_as_nonempty(self):
        row=sample();row['noisy_graph']['nodes'].update({'2':'Injected','3':'Injected'})
        env=run(row,{'del_node':[{'op':'DELETE_NODE','node':1}]})
        self.assertEqual(env._record['metrics']['behavior_del_node'],.25)
        self.assertIn('2',env._record['final_graph']['nodes'])

    def test_lc_rewrite_need_only_checks_query_endpoints(self):
        row=sample();row.update(task_type='LC',target_node=None,protected_query_nodes=[0,1])
        row['noisy_graph']['nodes']['2']='Non-query changed'
        row['clean_graph']['nodes']['2']='Original'
        self.assertFalse(operation_needed(row,row['noisy_graph'],'rewrite'))
        row['noisy_graph']['nodes']['1']='Changed endpoint'
        self.assertTrue(operation_needed(row,row['noisy_graph'],'rewrite'))

    def test_hidden_need_flags_not_in_feedback(self):
        row=sample();row['noisy_graph']['nodes']['0']='Changed'
        env=GraphEnvironment();env.reset(row)
        messages=invoke(env,'del_node')
        self.assertNotIn('operation_needed',messages[-1]['content'])
        self.assertNotIn('clean_graph',env._state.sample)


if __name__=='__main__':unittest.main()
