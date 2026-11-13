import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from episode import build_prediction_prompt
from graph_state import clean_episode,load_episodes
from task_utils import prepare_episode,lp_label_map
from reward import score_episode
from evaluation_metrics import summarize

def lp():
    return {'source_index':7,'task_type':'LP','target':'query pair node0, node1',
        'protected_query_nodes':[0,1],'target_node':None,
        'label_space':['Yes, they have citation relationships','No, they do not have citation relationships'],
        'ground_truth':'Yes, they have citation relationships',
        'clean_graph':{'nodes':{'0':'HIDDEN_CLEAN_TEXT','1':'Other paper','2':'Neighbor'},
            'edges':[{'u':0,'v':1},{'u':1,'v':0},{'u':0,'v':2}]},
        'noisy_graph':{'nodes':{'0':'Observed target paper','1':'Other paper','2':'Neighbor'},
            'edges':[{'u':0,'v':1,'relation':'LEAKING_LABEL'}, {'u':1,'v':0}, {'u':0,'v':2}]},
        'prompt':'HIDDEN_STORED_PROMPT','oracle_completion':'HIDDEN_ORACLE','attack_budget':99}

class LPConstraints(unittest.TestCase):
    def test_both_graph_views_mask_query_in_both_directions_without_mutating_source(self):
        row=lp();before=copy.deepcopy(row);prepared=prepare_episode(row)
        for graph in ['noisy_graph','clean_graph']:
            self.assertEqual(prepared[graph]['edges'],[{'u':0,'v':2}])
        self.assertEqual(row,before)
        self.assertEqual(prepared['ground_truth'],row['ground_truth'])
        self.assertEqual(clean_episode(row)['noisy_graph']['edges'],[{'u':0,'v':2}])

    def test_prompt_masks_explicit_graph_override(self):
        row=lp();text=build_prediction_prompt(row,graph=row['clean_graph'])
        self.assertNotIn('<node0, node1>',text)
        self.assertNotIn('<node1, node0>',text)

    def test_directed_query_only_masks_requested_direction(self):
        row=lp();row['query_directed']=True
        self.assertEqual(prepare_episode(row)['noisy_graph']['edges'],[{'u':1,'v':0},{'u':0,'v':2}])
        self.assertNotIn('Query direction:',build_prediction_prompt(row))

    def test_positive_and_negative_examples_have_same_public_prompt(self):
        row=lp();other=copy.deepcopy(row);other['ground_truth']=row['label_space'][1]
        other['clean_graph']['nodes']['0']='DIFFERENT_PRIVATE_REFERENCE'
        self.assertEqual(build_prediction_prompt(row),build_prediction_prompt(other))
        text=build_prediction_prompt(row)
        for secret in ['HIDDEN_','LEAKING_LABEL','attack_budget','ground_truth']:
            self.assertNotIn(secret,text)
        self.assertIn('absence from the observed graph alone does not imply',text)

    def test_target_string_is_rebuilt_from_structured_nodes(self):
        row=lp();row['target']='node0, node1; HIDDEN_GROUND_TRUTH_HINT'
        self.assertNotIn('HIDDEN_',build_prediction_prompt(row))

    def test_missing_text_is_evidence_absence_not_a_loader_error(self):
        row=lp();row['noisy_graph']['nodes']['1']='Missing.'
        self.assertIn('Missing.',build_prediction_prompt(row))

    def test_invalid_task_nodes_and_graph_references_fail(self):
        cases=[]
        row=lp();row['task_type']='GC';cases.append(row)
        row=lp();row['protected_query_nodes']=[0,0];cases.append(row)
        row=lp();row['protected_query_nodes']=[0,9];cases.append(row)
        row=lp();row['target_nodes']=[0,2];cases.append(row)
        row=lp();row['noisy_graph']['edges'].append({'u':0,'v':9});cases.append(row)
        row=lp();row['noisy_graph']['nodes']['node0']='Conflicting duplicate';cases.append(row)
        row=lp();row['source_index']=True;cases.append(row)
        for row in cases:
            with self.subTest(row=row),self.assertRaises(ValueError):prepare_episode(row)

    def test_lp_candidate_constraints(self):
        for labels in [['yes'],['yes','no','maybe'],['yes','YES'],['yes','yes indeed'],['connected','unconnected']]:
            row=lp();row['label_space']=labels
            with self.subTest(labels=labels),self.assertRaises(ValueError):prepare_episode(row)
        row=lp();row['label_space']=['connected','unconnected'];row['ground_truth']='connected'
        row['lp_label_polarity']={'connected':True,'unconnected':False}
        self.assertEqual(lp_label_map(row['label_space'],row['lp_label_polarity']),{'connected':True,'unconnected':False})
        prepare_episode(row)

    def test_yes_and_no_must_match_complete_candidate(self):
        row=lp()
        for prediction,expected in [(row['label_space'][0],2),(row['label_space'][1],1.1),('Yes',1),('No',1)]:
            score,record=score_episode(row,json.dumps({'analysis':'Evidence','prediction':prediction}))
            self.assertEqual(score,expected)
        row['ground_truth']=row['label_space'][1]
        self.assertEqual(score_episode(row,json.dumps({'analysis':'Evidence','prediction':row['label_space'][1]}))[0],2)

    def test_loader_applies_masking_and_preserves_order(self):
        a=lp();b=lp();b['source_index']=2
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'rows.json';p.write_text(json.dumps([a,b]));rows=load_episodes(p)
        self.assertEqual([r['source_index'] for r in rows],[7,2])
        self.assertEqual(rows[0]['noisy_graph']['edges'],[{'u':0,'v':2}])

    def test_invalid_prediction_does_not_count_as_negative(self):
        rows=[]
        for truth,prediction in [(True,True),(True,None),(False,False),(False,True)]:
            row=lp();row['ground_truth']=row['label_space'][0 if truth else 1]
            label='other' if prediction is None else row['label_space'][0 if prediction else 1]
            _,record=score_episode(row,json.dumps({'analysis':'Evidence','prediction':label}));rows.append(record)
        report=summarize(rows)['lp']
        self.assertEqual(report['count'],4)
        self.assertEqual(report['invalid_predictions'],1)
        self.assertEqual(report['balanced_accuracy'],.5)
        self.assertEqual(report['confusion']['positive']['invalid'],1)

    def test_missing_lp_test_coverage_is_explicit(self):
        self.assertFalse(summarize([])['lp']['available'])
        self.assertIsNone(summarize([])['lp']['balanced_accuracy'])

if __name__=='__main__':unittest.main()
