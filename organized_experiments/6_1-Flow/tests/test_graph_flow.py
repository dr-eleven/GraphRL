import sys,json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import bootstrap
from graph_task import GraphSession,build_question,execute_literal_command,extract_prediction,score
from workflow import run_workflow
from agentflow.models.formatters import QueryAnalysis,NextStep,ToolCommand,MemoryVerification


def sample():
    clean=dict(nodes={'0':'Target factual text','1':'Related fact'},edges=[dict(u=0,v=1)])
    noisy=deepcopy(clean);noisy['nodes']['2']='Injected irrelevant';noisy['edges'].append(dict(u=1,v=2))
    return dict(source_index=1,task_type='NC',dataset='fixture',target='node0',target_node=0,label_space=['class_a','class_b'],ground_truth='class_a',clean_graph=clean,noisy_graph=noisy,protected_nodes=[0],protected_edges=[],protected_query_nodes=[0],secret='secret_clean')

class Scripted:
    def __init__(self,role):self.role=role;self.calls=[]
    def __call__(self,content,response_format=None,**kwargs):
        self.calls.append(dict(content=content,schema=str(response_format)))
        if response_format is QueryAnalysis:return QueryAnalysis(concise_summary='classify',required_skills='graph',relevant_tools='del_node',additional_considerations='protect target')
        if response_format is NextStep:return NextStep(justification='remove injected',context='node2 is irrelevant',sub_goal='remove node2',tool_name='del_node')
        if response_format is ToolCommand:return ToolCommand(analysis='noise',explanation='remove',command='execution = tool.execute(analysis="noise", actions=[{"op":"DELETE_NODE","node":2}])')
        if response_format is MemoryVerification:return MemoryVerification(analysis='enough',stop_signal=True)
        return '{"analysis":"factual target","prediction":"class_a"}'

class GraphFlowTests(unittest.TestCase):
    def test_public_whitelist(self):
        q=build_question(sample())
        for name in ['secret_clean','ground_truth','clean_graph','injected_nodes']:self.assertNotIn(name,q)
    def test_atomic_rejection(self):
        state=GraphSession(sample());before=deepcopy(state.graph)
        result=state.apply('del_node','reason',[dict(op='DELETE_NODE',node=2),dict(op='DELETE_NODE',node=0)])
        self.assertFalse(result['ok']);self.assertEqual(state.graph,before)
    def test_free_order_and_candidate_isolation(self):
        a,b=GraphSession(sample()),GraphSession(sample())
        self.assertTrue(a.apply('del_edge','reason',[dict(op='DEL_EDGE',u=1,v=2)])['ok'])
        self.assertTrue(a.apply('add_edge','reason',[dict(op='ADD_EDGE',u=1,v=2)])['ok'])
        self.assertTrue(a.apply('rewrite','reason',[dict(op='REWRITE',node=0,text='Repaired factual text')])['ok'])
        self.assertNotEqual(a.graph,b.graph);self.assertEqual(b.graph,sample()['noisy_graph'])
    def test_empty_actions_rejected_for_every_tool(self):
        from graph_task import metadata
        for tool in ['del_node','del_edge','add_edge','rewrite']:
            with self.subTest(tool=tool):
                state=GraphSession(sample());before=deepcopy(state.graph)
                result=execute_literal_command(state,tool,'execution = tool.execute(analysis="reason", actions=[])')
                self.assertFalse(result['ok']);self.assertTrue(result['graph_unchanged'])
                self.assertEqual(state.graph,before)
                self.assertIn('EMPTY_ACTIONS',[e['code'] for e in result['errors']])
                json.dumps(state.audit)
                self.assertIn('nonempty actions list',build_question(sample()))
                self.assertNotIn('actions=[]',metadata()[tool]['command_example'])
    def test_literal_execution(self):
        state=GraphSession(sample())
        self.assertTrue(execute_literal_command(state,'del_node','execution = tool.execute(analysis="reason", actions=[{"op":"DELETE_NODE","node":2}])')['ok'])
        self.assertNotIn('2',state.graph['nodes'])
    def test_reject_arbitrary_python(self):
        for command in ['import os\nexecution = tool.execute(analysis="a", actions=[])','execution = tool.execute(analysis=str(1), actions=[])','execution = tool.execute(**dict(analysis="a",actions=[]))']:
            self.assertFalse(execute_literal_command(GraphSession(sample()),'del_node',command)['ok'])
    def test_candidate_matching(self):
        self.assertEqual(extract_prediction('Summary\n{"prediction":"class_a"}',['class_a','class_b']),'class_a')
        self.assertIsNone(extract_prediction('{"prediction":"class_a"} {"prediction":"class_b"}',['class_a','class_b']))
        self.assertIsNone(extract_prediction('class_a',['class_a','class_b']))
        self.assertEqual(score(None,'class_a'),0.)

    def test_python_only_literals_rejected_and_audit_serializable(self):
        values=["[{'DELETE_NODE', 2}]", "[{'op':'DELETE_NODE','node':{2}}]", "(1, 2)", "[b'bad']", "[{1:'bad'}]", "[1e999]", "[1+2j]"]
        for value in values:
            with self.subTest(value=value):
                state=GraphSession(sample());before=deepcopy(state.graph)
                command="execution = tool.execute(analysis='reason', actions="+value+")"
                result=execute_literal_command(state,'del_node',command)
                self.assertFalse(result['ok']);self.assertEqual(state.graph,before)
                self.assertEqual(state.audit[-1]['raw_command'],command)
                json.dumps(state.audit,allow_nan=False)

    def test_invalid_set_does_not_abort_original_workflow_logging(self):
        class InvalidTool(Scripted):
            def __call__(self,content,response_format=None,**kwargs):
                if response_format is ToolCommand:
                    return ToolCommand(analysis='noise',explanation='remove',command="execution = tool.execute(analysis='reason', actions=[{'DELETE_NODE', 2}])")
                return super().__call__(content,response_format,**kwargs)
        result=run_workflow(sample(),Scripted('planner'),InvalidTool('fixed'))
        self.assertEqual(result['reward'],1.)
        self.assertFalse(result['actions'][0]['result']['ok'])
        self.assertEqual(result['final_graph'],sample()['noisy_graph'])
        json.dumps(result,allow_nan=False)
    def test_local_engine_structured_response_and_raw_trace(self):
        from engines import Engine
        raw='```json\n'+json.dumps(dict(analysis='reason',explanation='no edit',command='execution = tool.execute(analysis="reason", actions=[])'))+'\n```'
        class Response:
            status_code=200
            def json(self):return dict(choices=[dict(message=dict(content=raw),finish_reason='stop')],prompt_token_ids=[1,2],response_token_ids=[[3,4]])
        class HTTP:
            payload=None
            def post(self,url,json,timeout):self.payload=json;return Response()
        for role in ['fixed','planner']:
            trace=[];engine=Engine('http://127.0.0.1:1/v1','fixture',trace=trace if role=='planner' else None,role=role);engine.http=HTTP()
            output=engine('fixture',response_format=ToolCommand)
            self.assertIsInstance(output,ToolCommand)
            self.assertEqual(engine.calls[0]['raw_output'],raw)
            self.assertEqual('response_format' in engine.http.payload,role=='fixed')
            if role=='planner':self.assertEqual(trace[0]['response_ids'],[3,4])

    def test_original_solver_all_roles(self):
        planner,fixed=Scripted('planner'),Scripted('fixed')
        result=run_workflow(sample(),planner,fixed)
        self.assertEqual(result['reward'],1.)
        self.assertEqual(len(planner.calls),1)
        self.assertEqual(len(fixed.calls),4) # analysis, executor, verifier, generator
        self.assertEqual(result['solver_result']['step_count'],1)
        self.assertNotIn('2',result['final_graph']['nodes'])
        self.assertTrue(any('Memory' in str(call['content']) for call in fixed.calls))
        self.assertEqual(result['actions'][0]['tool'],'del_node')

if __name__=='__main__':unittest.main()
