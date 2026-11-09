import json
import unittest
from copy import deepcopy

from episode import (public_observation, execute_actions, parse_completion, first_messages,
                     normalize_graph)
from executor_protocol import stage_messages
from prepare_data import normalize_episode
from reward import score_episode
from reference_metrics import compute_reference_metrics


def sample():
    graph = {'nodes': {'0': 'library books', '1': 'library reading', '2': 'sports'},
             'edges': [{'u': 0, 'v': 1}]}
    return normalize_episode({'source_index': 1, 'dataset': 'toy', 'task_type': 'NC', 'target_node': 0,
        'protected_nodes': [0], 'protected_edges': [], 'label_space': ['library', 'sports'],
        'ground_truth': 'library', 'clean_graph': deepcopy(graph), 'noisy_graph': deepcopy(graph),
        'injected_nodes': [], 'attack_budget': 0, 'attack_kind': 'clean', 'prompt': 'SECRET_OLD_PROMPT'})


class ProtocolTests(unittest.TestCase):
    def test_hidden_fields_do_not_affect_policy_or_executor(self):
        a = sample()
        b = deepcopy(a)
        b.update(ground_truth='SECRET', attack_budget=999, injected_nodes=[2], oracle_completion='SECRET')
        b['clean_graph']['nodes']['0'] = 'SECRET'
        self.assertEqual(first_messages(a, 'executor'), first_messages(b, 'executor'))
        action = [{'op': 'DELETE_NODE', 'node': 2}]
        ea = execute_actions(public_observation(a), action, [])
        eb = execute_actions(public_observation(b), action, [])
        self.assertEqual(ea, eb)
        self.assertIsNone(ea['error'])
        self.assertNotIn('2', ea['graph']['nodes'])

    def test_all_actions_and_single_transaction(self):
        s = sample()
        e = execute_actions(public_observation(s), [{'op': 'DEL_EDGE', 'u': 0, 'v': 1},
            {'op': 'ADD_EDGE', 'u': 0, 'v': 2}], [{'op': 'REWRITE', 'node': 0, 'text': 'books and reading'}])
        self.assertIsNone(e['error'])
        self.assertEqual(e['graph']['edges'], [{'u': 0, 'v': 2}])
        self.assertEqual(e['graph']['nodes']['0'], 'books and reading')
        self.assertEqual(s['noisy_graph']['nodes']['0'], 'library books')

    def test_invalid_text_rolls_back_structure(self):
        s = sample()
        e = execute_actions(public_observation(s), [{'op': 'DEL_EDGE', 'u': 0, 'v': 1}],
                            [{'op': 'REWRITE', 'node': 2, 'text': 'changed'}])
        self.assertIsNotNone(e['error'])
        self.assertEqual(e['graph'], s['noisy_graph'])
        self.assertEqual(e['operation_status'][0]['status'], 'rolled_back')
        self.assertEqual(score_episode(s, e, 'library')[0], 1)

    def test_target_and_query_protection(self):
        s = sample()
        e = execute_actions(public_observation(s), [{'op': 'DELETE_NODE', 'node': 0}], [])
        self.assertIsNotNone(e['error'])
        s.update(task_type='LC', target_node=None, protected_query_nodes=[0, 1])
        s = normalize_episode(s)
        self.assertEqual(s['noisy_graph']['edges'], [])
        for node in [0, 1]:
            self.assertIsNotNone(execute_actions(public_observation(s), [{'op': 'DELETE_NODE', 'node': node}], [])['error'])
            self.assertIsNone(execute_actions(public_observation(s), [], [{'op': 'REWRITE', 'node': node, 'text': 'changed text'}])['error'])
        self.assertIsNotNone(execute_actions(public_observation(s), [{'op': 'ADD_EDGE', 'u': 0, 'v': 1}], [])['error'])

    def test_conflicts_strict_ids_and_noop(self):
        o = public_observation(sample())
        bad = [[{'op': 'ADD_EDGE', 'u': True, 'v': 2}], [{'op': 'ADD_EDGE', 'u': '0', 'v': 2}],
               [{'op': 'DEL_EDGE', 'u': 0, 'v': 1}, {'op': 'DELETE_NODE', 'node': 1}],
               [{'op': 'ADD_EDGE', 'u': 0, 'v': 2}, {'op': 'DEL_EDGE', 'u': 0, 'v': 2}],
               [{'op': 'NO_OP'}, {'op': 'DELETE_NODE', 'node': 2}]]
        for actions in bad:
            self.assertIsNotNone(execute_actions(o, actions, [])['error'])
        self.assertEqual(execute_actions(o, [{'op': 'NO_OP'}], [{'op': 'NO_OP'}])['actions'], [])

    def test_reward_is_exactly_binary_B_plus_T(self):
        s = sample()
        noop = execute_actions(public_observation(s), [], [])
        self.assertEqual(score_episode(s, noop, 'library')[0], 2)
        self.assertEqual(score_episode(s, noop, 'sports')[0], 1)
        s['noisy_graph']['nodes']['0'] += ' misleading'
        noop = execute_actions(public_observation(s), [], [])
        self.assertEqual(score_episode(s, noop, 'library')[0], 1)
        wrong_but_legal = execute_actions(public_observation(s), [{'op': 'DELETE_NODE', 'node': 1}], [])
        self.assertEqual(score_episode(s, wrong_but_legal, 'library')[0], 2)

    def test_declared_graph_is_checked_but_not_rewarded(self):
        s = sample()
        e = execute_actions(public_observation(s), [], [])
        declared = deepcopy(e['graph'])
        declared['nodes']['0'] = 'hallucinated'
        metrics, _ = compute_reference_metrics(s, e, declared)
        self.assertEqual(metrics['declared_execution_consistent'], 0)
        self.assertEqual(score_episode(s, e, 'library')[0], 2)
        self.assertEqual(e['graph']['nodes']['0'], 'library books')

    def test_strict_stage_order_and_no_early_answer(self):
        good = {'modify': {'reason': 'keep', 'actions': []}, 'rewrite': {'reason': 'keep', 'actions': []}}
        self.assertEqual(parse_completion(json.dumps(good), 'edit'), good)
        for text in [json.dumps(dict(reversed(list(good.items())))), json.dumps({**good, 'answer': 'library'}),
                     '{"modify":{},"modify":{},"rewrite":{}}', '```json\n' + json.dumps(good) + '\n```']:
            with self.assertRaises(ValueError):
                parse_completion(text, 'edit')

    def test_feedback_is_actual_graph_and_preserves_history(self):
        s = sample()
        messages = first_messages(s, 'executor')
        e = execute_actions(public_observation(s), [{'op': 'DELETE_NODE', 'node': 2}], [])
        feedback = stage_messages(messages, 'EDIT_DECISION', e, 'rewrite')
        self.assertEqual(feedback[:-2], messages)
        self.assertEqual(feedback[-2], {'role': 'assistant', 'content': 'EDIT_DECISION'})
        self.assertIn('EXECUTION FEEDBACK (from Python)', feedback[-1]['content'])
        self.assertIn(json.dumps(e['graph'], ensure_ascii=False), feedback[-1]['content'])

    def test_existing_selfloop_preserved_new_selfloop_rejected(self):
        s = sample()
        s['noisy_graph']['edges'].append({'u': 1, 'v': 1, 'relation': 'existing'})
        s = normalize_episode(s)
        self.assertEqual(execute_actions(public_observation(s), [], [])['graph'], s['noisy_graph'])
        self.assertIsNotNone(execute_actions(public_observation(s), [{'op': 'ADD_EDGE', 'u': 2, 'v': 2}], [])['error'])


if __name__ == '__main__':
    unittest.main()
