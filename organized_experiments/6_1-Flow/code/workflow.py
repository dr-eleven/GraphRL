"""Run the original AgentFlow Solver and modules with GraphRL tools and local engines."""
import json, time
from pathlib import Path
from copy import deepcopy
import bootstrap
from graph_core.graph_state import serialize_graph
from graph_task import GraphSession, metadata, execute_literal_command, build_question, extract_prediction, score, graph_key
from engines import Engine, CapacityError

# Only replace engine construction. AgentFlow's original module methods are reused.
import agentflow.engine.factory as factory
factory.create_llm_engine=lambda *args,**kwargs:None
from agentflow.models.planner import Planner
from agentflow.models.verifier import Verifier
from agentflow.models.executor import Executor
from agentflow.models.memory import Memory
from agentflow.solver import Solver
from prompt_memory import PromptMemory

class GraphPlanner(Planner):
    def generate_next_step(self,question,image,query_analysis,memory,step_count,max_step_count,json_data=None):
        return super().generate_next_step(question,image,query_analysis,
            PromptMemory(memory,self.session),step_count,max_step_count,json_data)
    def generate_direct_output(self,question,image,memory):
        return super().generate_direct_output(question,image,PromptMemory(memory,self.session))
    def generate_final_output(self,question,image,memory):
        return super().generate_final_output(question,image,PromptMemory(memory,self.session))

class GraphVerifier(Verifier):
    def verificate_context(self,question,image,query_analysis,memory,step_count=0,json_data=None):
        return super().verificate_context(question,image,query_analysis,
            PromptMemory(memory,self.session),step_count,json_data)

class GraphExecutor(Executor):
    def generate_tool_command(self,question,image,context,sub_goal,tool_name,tool_metadata,step_count=0,json_data=None):
        context=str(context)+'\nCURRENT GRAPH (authoritative)\n'+serialize_graph(self.session.graph)
        return super().generate_tool_command(question,image,context,sub_goal,tool_name,tool_metadata,step_count,json_data)
    def execute_tool_command(self,tool_name,command):return execute_literal_command(self.session,tool_name,command)


def run_workflow(sample,planner_engine,fixed_engine,max_steps=10,max_time=500,verbose=False):
    session=GraphSession(sample);meta=metadata();tools=list(meta)
    planner=GraphPlanner('local','local',meta,tools,verbose=verbose);planner.session=session;planner.llm_engine=planner_engine;planner.llm_engine_fixed=fixed_engine
    executor=GraphExecutor('local',verbose=verbose);executor.session=session;executor.llm_generate_tool_command=fixed_engine
    verifier=GraphVerifier('local','local',meta,tools,verbose=verbose);verifier.session=session;verifier.llm_engine=fixed_engine;verifier.llm_engine_fixed=fixed_engine
    solver=Solver(planner,verifier,Memory(),executor,output_types='direct',max_steps=max_steps,max_time=max_time,max_tokens=getattr(planner_engine,'max_tokens',8192),root_cache_dir=str(bootstrap.ROOT/'runs/.solver_cache'),verbose=verbose)
    question=build_question(sample);started=time.monotonic();failure=None
    try:result=solver.solve(question);prediction=extract_prediction(result.get('direct_output',''),sample['label_space'])
    except CapacityError as exc:
        result={};prediction=None;failure=str(exc)
    return dict(source_index=sample['source_index'],task_type=sample['task_type'],dataset=sample.get('dataset'),prediction=prediction,ground_truth=sample['ground_truth'],reward=score(prediction,sample['ground_truth']),input_dirty=graph_key(sample['noisy_graph'])!=graph_key(sample['clean_graph']),graph_exact=graph_key(session.graph)==graph_key(sample['clean_graph']),capacity_failure=failure,final_graph=deepcopy(session.graph),actions=deepcopy(session.audit),solver_result=result,seconds=time.monotonic()-started,planner_calls=deepcopy(planner_engine.calls),fixed_calls=deepcopy(fixed_engine.calls))
