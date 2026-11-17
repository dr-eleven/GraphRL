import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from test_graph_flow import sample, Scripted
from graph_task import GraphSession
from graph_core.graph_state import serialize_graph
from prompt_memory import PromptMemory
from agentflow.models.memory import Memory
from agentflow.models.formatters import MemoryVerification, NextStep, ToolCommand
from workflow import run_workflow


class PromptMemoryTests(unittest.TestCase):
    def test_latest_graph_once_with_complete_independent_audit_history(self):
        session=GraphSession(sample());memory=Memory();initial=serialize_graph(session.graph)
        for turn in range(1,10):
            result=session.apply('del_edge','reason',[dict(op='DEL_EDGE',u=0,v=0)])
            memory.add_action(turn,'del_edge','remove loop','attempt',result)
        result=session.apply('del_node','reason',[dict(op='DELETE_NODE',node=2)])
        memory.add_action(10,'del_node','remove noise','delete node2',result)
        view=PromptMemory(memory,session).get_actions()
        text=json.dumps(view)
        self.assertEqual(text.count('current_graph'),1)
        self.assertEqual(len(view['history']),10)
        self.assertEqual(view['current_graph'],serialize_graph(session.graph))
        self.assertNotEqual(view['current_graph'],initial)
        self.assertIn('errors',view['history']['Action Step 1']['result'])
        self.assertEqual(memory.get_actions()['Action Step 1']['result']['current_graph'],initial)
        self.assertEqual(len(memory.get_actions()),10)
        view['history']['Action Step 1']['result']['errors'].clear()
        self.assertTrue(memory.get_actions()['Action Step 1']['result']['errors'])

    def test_ten_turn_original_solver_uses_views_but_logs_full_snapshots(self):
        class Fixed(Scripted):
            def __call__(self,content,response_format=None,**kwargs):
                if response_format is MemoryVerification:
                    self.calls.append(dict(content=content,schema=str(response_format)))
                    return MemoryVerification(analysis='continue',stop_signal=False)
                if response_format is ToolCommand:
                    self.calls.append(dict(content=content,schema=str(response_format)))
                    return ToolCommand(analysis='reason',explanation='invalid loop',command="execution = tool.execute(analysis='reason', actions=[{'op':'DEL_EDGE','u':0,'v':0}])")
                return super().__call__(content,response_format,**kwargs)
        planner,fixed=Scripted('planner'),Fixed('fixed')
        result=run_workflow(sample(),planner,fixed,max_steps=10)
        self.assertEqual(len(planner.calls),10)
        self.assertEqual(len(fixed.calls),22)
        self.assertEqual(result['reward'],1)
        self.assertEqual(len(result['solver_result']['memory']),10)
        for action in result['solver_result']['memory'].values():
            self.assertIn('current_graph',action['result'])
        for call in planner.calls:
            self.assertEqual(str(call['content']).count("'current_graph':"),1)
        for call in fixed.calls:
            if call['schema']==str(MemoryVerification) or call['schema']=='None':
                content=call['content'] if isinstance(call['content'],str) else '\n'.join(call['content'])
                self.assertEqual(content.count("'current_graph':"),1)
        self.assertIn('INVALID_ACTION',str(planner.calls[-1]['content']))
        self.assertIn('initial_graph',str(planner.calls[-1]['content']))
        json.dumps(result)
