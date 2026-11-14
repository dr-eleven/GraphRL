from copy import deepcopy
from dataclasses import replace
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from budgets import RolloutBudget
from episode import GraphEnvironment, initial_messages, label_map, TOOL_ORDER
from protocol import parse_event, validate_tool, ProtocolError
from runtime import InfrastructureError


def sample():
    graph={'nodes':{'0':'A library','1':'Reading room','2':'Noise'},'edges':[{'u':0,'v':1},{'u':1,'v':2}]}
    return dict(source_index=7,task_type='NC',target='node0',target_node=0,
                label_space=['library','shop'],noisy_graph=graph,clean_graph=deepcopy(graph),
                ground_truth='library',protected_nodes=[0],protected_edges=[],secret='HIDDEN_GOLD')


def tool(name, actions=None, **extra):
    return '<tool_call>'+json.dumps(dict(name=name,arguments=dict(analysis=extra.pop("analysis","Reason from latest graph."),actions=actions or [],**extra)))+'</tool_call>'


def env(mode='free',budget=None):
    e=GraphEnvironment(mode=mode,budget=budget);e.reset(sample());return e


def consume(e,text,**kwargs):
    return e._consume(text,list(range(len(text))),**kwargs)


def finish(e,prediction='library'):
    if e._mode=='ordered_four_stage_retry':
        while e._stage<3 or not e._stage_succeeded:
            name=TOOL_ORDER[e._stage+1] if e._stage_succeeded else e._current_tool
            consume(e,tool(name))
    consume(e,json.dumps(dict(analysis="Reason",prediction=prediction)))
    return e._score([])


class ProtocolTests(unittest.TestCase):
    def test_invalid_events_and_duplicate_keys(self):
        for text in ['', 'hello','[]','null','{}','{"analysis":"Reason","prediction":','{"analysis":"Reason","prediction":"a","prediction":"b"}',
                     tool('del_node')+tool('add_edge'),tool('del_node')+'{"analysis":"Reason","prediction":"library"}',
                     '<tool_call>{"name":"del_node"}', '<|im_start|>user\nhi']:
            with self.subTest(text=text): self.assertEqual(parse_event(text).kind,'INVALID_EVENT')

    def test_json_parsable_schema_errors(self):
        invalid=[[],None,{},dict(name='wat',arguments={}),dict(name=[],arguments={}),
                 dict(name='del_node',arguments='[]'),dict(name='del_node',arguments='null'),
                 dict(name='del_node',arguments=dict(analysis="Reason",actions=[],extra=1)),
                 dict(name='del_node',arguments=dict(analysis="Reason",actions=[{'op':'DELETE_NODE','node':True}])),
                 dict(name='del_node',arguments=dict(analysis="Reason",actions=[{'op':'DELETE_NODE','node':'2'}])),
                 dict(name='del_node',arguments=dict(analysis="Reason",actions=[{'op':'DELETE_NODE','node':2.0}])),
                 dict(name='del_node',arguments=dict(analysis="Reason",actions=[{'op':'DEL_EDGE','u':1,'v':2}])),
                 dict(name='rewrite',arguments=dict(analysis="Reason",actions=[{'op':'REWRITE','node':0,'text':' '}]))]
        for obj in invalid:
            with self.subTest(obj=obj),self.assertRaises(ProtocolError): validate_tool(obj)

    def test_analysis_required_nonempty_before_actions(self):
        for args in [{'actions':[]},{'actions':[],'analysis':'reason'},
                     {'analysis':'','actions':[]},{'analysis':'  ','actions':[]}]:
            with self.assertRaises(ProtocolError): validate_tool(dict(name='del_node',arguments=args))
        self.assertEqual(validate_tool(dict(name='del_node',arguments={'analysis':'Reason','actions':[]}))[0],'del_node')

    def test_truncated_vs_complete_event(self):
        self.assertEqual(parse_event('<tool_call>{','length').code,'GENERATION_TRUNCATED')
        self.assertEqual(parse_event(tool('del_node'),'length').kind,'TOOL_EVENT')
        self.assertEqual(parse_event('{"analysis":"Reason","prediction":"library"}','length').kind,'FINAL_EVENT')

    def test_extreme_json_is_model_error_not_infrastructure(self):
        for text in ['{"analysis":"Reason","prediction":'+('1'*5000)+'}',
                     '<tool_call>{"name":"rewrite","arguments":{"actions":[],"analysis":"\\ud800"}}</tool_call>']:
            self.assertEqual(parse_event(text).kind,'INVALID_EVENT')

    def test_public_label_collision(self):
        s=sample();s['label_space']=['Library',' library. ']
        with self.assertRaises(ValueError): label_map(s)

class StateTests(unittest.TestCase):
    def test_parse_error_terminates_without_feedback(self):
        for mode in ['free','ordered_four_stage_retry']:
            for raw in ['<tool_call>{','plain prose',tool('del_node')+tool('del_node'),
                        '<tool_call>{"name":"del_node","arguments":"{"}</tool_call>']:
                e=env(mode);before=deepcopy(e._graph)
                _,feedback=consume(e,raw)
                self.assertIsNone(feedback);self.assertEqual(e._graph,before)
                self.assertEqual(e._phase,'TERMINAL');self.assertEqual(e._finish_reason,'parse_error')
                self.assertEqual(e._turns,1);self.assertEqual(e._score([]),0)
                with self.assertRaises(RuntimeError):consume(e,tool('del_node'))

    def test_all_errors_collected_atomically_and_counted_once(self):
        e=env('ordered_four_stage_retry');before=deepcopy(e._graph)
        raw=tool('del_node',[dict(op='DELETE_NODE',node=0),dict(op='DELETE_NODE',node=999),
                             dict(op='DELETE_NODE',node=0),dict(op='DELETE_NODE',node='2')],analysis='')
        _,feedback=consume(e,raw)
        text=feedback['content']
        for expected in ['nonempty','protected','999','Repeated','JSON integer','actions[3].node']:
            self.assertIn(expected,text)
        self.assertEqual(e._graph,before);self.assertEqual(e._version,0)
        self.assertEqual(e._edit_errors,1);self.assertEqual(e._current_tool,'del_node')
        self.assertGreaterEqual(len(e._errors),6)
        self.assertNotIn('Next:',text)
        self.assertEqual(finish(e),1)

    def test_multi_event_atomic_rejection(self):
        e=env();before=deepcopy(e._graph)
        consume(e,tool('del_node',[dict(op='DELETE_NODE',node=2)])+'<tool_call>{bad}</tool_call>')
        self.assertEqual(e._graph,before);self.assertEqual(e._tool_successes,0)

    def test_action_batch_atomic_rejection(self):
        e=env();before=deepcopy(e._graph)
        consume(e,tool('add_edge',[dict(op='ADD_EDGE',u=0,v=2),dict(op='ADD_EDGE',u=0,v=999)]))
        self.assertEqual(e._graph,before);self.assertEqual(e._version,0)
        self.assertEqual(finish(e),1)

    def test_unknown_and_bad_args_recover(self):
        for raw in [tool('unknown'),'<tool_call>{"name":"del_node","arguments":"null"}</tool_call>',
                    tool('del_node',[dict(op='DELETE_NODE',node=0)])]:
            e=env();consume(e,raw)
            self.assertTrue(e._active);self.assertEqual(finish(e),1)

    def test_free_repeat_undo_redo(self):
        e=env()
        for name in ['del_edge','add_edge','del_edge','add_edge']:
            consume(e,tool(name,[dict(op={'del_edge':'DEL_EDGE','add_edge':'ADD_EDGE'}[name],u=1,v=2)]))
        for text in ['New library text','Another library text']:
            consume(e,tool('rewrite',[dict(op='REWRITE',node=0,text=text)]))
        self.assertEqual(e._tool_successes,6);self.assertIsNone(e._current_tool)
        self.assertEqual(finish(e),1)

    def test_legal_harmful_edit_still_binary(self):
        e=env();consume(e,tool('del_node',[dict(op='DELETE_NODE',node=1)]))
        self.assertEqual(finish(e),1)
        self.assertNotIn('1',e._graph['nodes'])

    def test_zero_tool_correct_and_valid_wrong_terminal(self):
        for label,reward in [('library',1),('shop',0)]:
            e=env();self.assertEqual(finish(e,label),reward)
            self.assertEqual(e._tool_attempts,0)
            with self.assertRaises(RuntimeError): consume(e,'{"analysis":"Reason","prediction":"library"}')

    def test_final_outside_label_and_format_repair(self):
        for first in ['{"analysis":"Reason","prediction":"outside"}','{"analysis":"Reason","prediction":["library"]}',
                      '{"analysis":"Reason","prediction":"library","extra":1}']:
            e=env();consume(e,first)
            self.assertEqual(e._phase,'ANSWER_ONLY');self.assertEqual(e._final_attempts,1)
            self.assertEqual(finish(e),1)

    def test_system_user_split_and_compact_tool_descriptions(self):
        from transformers.utils.chat_template_utils import get_json_schema
        messages=initial_messages(sample())
        self.assertEqual([m['role'] for m in messages],['system','user'])
        self.assertIn('Think-with-Graph',messages[0]['content'])
        self.assertIn('At most 10 generation turns',messages[0]['content'])
        metadata,graph=messages[1]['content'].split('\n<initial_graph>\n',1)
        public=json.loads(metadata)
        self.assertEqual(public['label_space'],sample()['label_space'])
        self.assertEqual(public['target'],'node0')
        self.assertNotIn('analysis',public)
        self.assertNotIn('ground_truth',public);self.assertNotIn('clean_graph',public)
        self.assertIn('node0: A library',graph)
        self.assertNotIn('<restored_graph>',messages[1]['content'])
        self.assertIn('or <initial_graph> if no update exists',messages[0]['content'])
        expected_descriptions={'del_node': 'Delete injected noise nodes.', 'del_edge': 'Delete edges added by structural noise.', 'add_edge': 'Restore edges removed by structural noise.', 'rewrite': 'Rewrite target text corrupted by textual noise.'}
        e=env()
        for tool_name in TOOL_ORDER:
            schema=get_json_schema(getattr(e,tool_name))['function']
            self.assertEqual(schema['description'],expected_descriptions[tool_name])
            self.assertEqual(schema['parameters']['required'],['analysis','actions'])
            self.assertNotIn('return',schema)
            self.assertEqual(schema['parameters']['properties']['analysis']['description'],'Reasoning.')
            self.assertNotIn('[]',schema['parameters']['properties']['actions']['description'])

    def test_final_schema_only_and_no_hidden_supervision_feedback(self):
        e=env();s=sample();s['secret']='DO_NOT_RENDER';e.reset(s)
        self.assertNotIn('clean_graph',e._public)
        self.assertNotIn('ground_truth',''.join(m['content'] for m in initial_messages(s)))
        _,f=consume(e,tool('del_node'))
        self.assertNotIn('clean_graph',f['content']);self.assertNotIn('ground_truth',f['content'])
        self.assertNotIn('DO_NOT_RENDER',f['content'])

    def test_infra_is_not_reward_zero(self):
        e=env()
        with patch('episode.execute_actions',side_effect=RuntimeError('broken worker')):
            with self.assertRaises(InfrastructureError):consume(e,tool('del_node'))
        self.assertEqual(e._phase,'ABORTED_INFRA')
        with self.assertRaises(InfrastructureError):e._score([])

    def test_complete_length_stop_executes_incomplete_never_executes(self):
        e=env();consume(e,tool('del_node'),finish_reason='length')
        self.assertEqual(e._tool_successes,1)
        consume(e,'<tool_call>{',finish_reason='length')
        self.assertEqual(e._tool_successes,1)
        self.assertEqual(e._errors[-1]['code'],'GENERATION_TRUNCATED')

    def test_tool_schema_required_actions_only(self):
        from transformers.utils.chat_template_utils import get_json_schema
        e=env()
        names=[n for n,f in inspect.getmembers(type(e),inspect.isfunction) if not n.startswith('_') and n!='reset']
        self.assertEqual(set(names),set(TOOL_ORDER))
        for n in names:self.assertEqual(get_json_schema(getattr(e,n))['function']['parameters']['required'],['analysis','actions'])

    def test_no_stage_or_error_limit(self):
        e=env('ordered_four_stage_retry')
        for _ in range(9):
            consume(e,tool('del_node',[dict(op='DELETE_NODE',node=999)]))
            self.assertEqual(e._phase,'EDITING');self.assertEqual(e._current_tool,'del_node')
        self.assertEqual(e._edit_errors,9)
        consume(e,tool('del_node'))
        self.assertEqual(e._turns,10);self.assertEqual(e._finish_reason,'max_model_turns')
        self.assertEqual(e._score([]),0)

    def test_success_allows_repeat_or_immediate_next(self):
        e=env('ordered_four_stage_retry')
        for _ in range(4):
            _,f=consume(e,tool('del_node'))
            self.assertEqual(e._current_tool,'del_node')
            self.assertIn('next_tool: del_node or del_edge',f['content'])
        consume(e,tool('rewrite'))
        self.assertEqual(e._current_tool,'del_node')
        self.assertFalse(e._stage_succeeded)
        _,f=consume(e,tool('del_edge'))
        self.assertIn('WRONG_STAGE',f['content'])
        consume(e,tool('del_node'));consume(e,tool('del_edge'))
        self.assertEqual(e._current_tool,'del_edge')
        _,f=consume(e,tool('del_node'))
        self.assertIn('WRONG_STAGE',f['content'])

    def test_noop_uses_previous_graph(self):
        e=env('ordered_four_stage_retry')
        _,f=consume(e,tool('del_node'))
        self.assertIn('Graph unchanged; use <initial_graph>.',f['content'])
        self.assertNotIn('<restored_graph>',f['content'])
        _,f=consume(e,tool('del_node',[dict(op='DELETE_NODE',node=999)]))
        self.assertIn('Graph unchanged; use <initial_graph>.',f['content'])
        _,f=consume(e,tool('del_node',[dict(op='DELETE_NODE',node=2)]))
        self.assertIn('<restored_graph>\n',f['content'])
        _,f=consume(e,tool('del_node'))
        self.assertNotIn('<restored_graph>\n',f['content'])
        self.assertIn('Graph unchanged; use the latest <restored_graph>.',f['content'])
        self.assertIn('Completed successfully.',f['content'])

    def test_final_on_tenth_turn_is_scored(self):
        for mode in ['free','ordered_four_stage_retry']:
            e=env(mode)
            if mode=='ordered_four_stage_retry':
                for name in TOOL_ORDER:consume(e,tool(name))
            while e._turns<9:consume(e,tool('rewrite' if mode!='free' else 'del_node'))
            self.assertEqual(finish(e),1)
            self.assertEqual(e._turns,10);self.assertEqual(e._finish_reason,'final_submitted')

    def test_tenth_tool_executes_then_zero_no_eleventh(self):
        for mode in ['free','ordered_four_stage_retry']:
            e=env(mode)
            for _ in range(9):consume(e,tool('del_node'))
            _,f=consume(e,tool('del_node',[dict(op='DELETE_NODE',node=2)]))
            self.assertNotIn('2',e._graph['nodes'])
            self.assertIn('Completed successfully.',f['content'])
            self.assertNotIn('next_tool:',f['content'])
            self.assertEqual(e._score([]),0)
            with self.assertRaises(RuntimeError):consume(e,'{"analysis":"Reason","prediction":"library"}')

    def test_final_schema_repair_has_no_separate_quota(self):
        e=env()
        for _ in range(9):
            consume(e,'{"analysis":"Reason","prediction":"outside"}')
            self.assertTrue(e._active)
        self.assertEqual(finish(e),1)

    def test_fixed_cannot_answer_before_rewrite_success(self):
        e=env('ordered_four_stage_retry')
        consume(e,'{"analysis":"Reason","prediction":"library"}')
        self.assertIsNone(e._prediction);self.assertEqual(e._current_tool,'del_node')
        self.assertEqual(finish(e),1)

    def test_feedback_capacity_failure_never_requests_extra_answer(self):
        e=env('ordered_four_stage_retry');before=deepcopy(e._graph)
        _,f=consume(e,tool('del_node',[dict(op='DELETE_NODE',node=2)]),feedback_fits=lambda _:False)
        self.assertEqual(e._graph,before);self.assertEqual(e._phase,'TERMINAL')
        self.assertEqual(e._score([]),0)
        self.assertNotIn('submit final',f['content'])


if __name__=='__main__':unittest.main()
