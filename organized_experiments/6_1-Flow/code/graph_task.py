"""Candidate-local GraphRL state and four AgentFlow tool.execute interfaces."""
import ast, json, math
from copy import deepcopy
from graph_core.graph_state import public_episode, serialize_graph, execute_actions, TOOL_ORDER, TOOL_OP, ACTION_FIELDS
from graph_core.validation import inspect_tool
from graph_core.label_utils import _label_key

RULES='''Solve the public graph classification task. The initial graph may contain noise. Use only graph structure, node text and public constraints; noise locations and amounts are unknown. Use the CURRENT graph in the latest successful tool result; failed calls do not change it. Select any of del_node, del_edge, add_edge, rewrite in any order; repeats are allowed. Select a tool only when there is at least one supported applicable action. Every tool call MUST contain a nonempty actions list; actions=[] is forbidden. Never delete protected nodes or edit protected edges. No self-loops, missing endpoints, duplicate/conflicting actions. DEL_EDGE requires a present edge, ADD_EDGE an absent edge. REWRITE only target/query endpoints; preserve supported facts and never invent labels or answer hints. Submit the final answer as one JSON object {"analysis":"prediction reasoning","prediction":"one exact candidate label"}. Do not use a ground-truth label or hidden clean graph. A tool command must be exactly execution = tool.execute(analysis="reason", actions=[...]), using literal parameter values. The generator decides the final label from the current graph.'''

def build_question(sample):
    p=public_episode(sample);graph=p.pop('noisy_graph')
    return RULES+'\nPUBLIC TASK\n'+json.dumps(p,ensure_ascii=False)+'\n<initial_graph>\n'+serialize_graph(graph)+'\n</initial_graph>'

class GraphSession:
    def __init__(self,sample):
        self.public=public_episode(sample)
        self.graph=deepcopy(self.public['noisy_graph'])
        self.audit=[]
    def apply(self,name,analysis,actions):
        payload={'name':name,'arguments':{'analysis':analysis,'actions':actions}}
        _,_,errors=inspect_tool(payload,self.public,self.graph,None,64)
        before=deepcopy(self.graph)
        if errors:
            result=dict(ok=False,errors=errors,graph_unchanged=True)
        else:
            updated,error=execute_actions({**self.public,'noisy_graph':self.graph},actions)
            if error:raise RuntimeError('Validator/executor inconsistency: '+error)
            self.graph=updated;result=dict(ok=True,changed=updated!=before,actions=actions)
        # Full current graph lets every unchanged upstream role inspect real state.
        result['current_graph']=serialize_graph(self.graph)
        self.audit.append(dict(tool=name,analysis=analysis,actions=deepcopy(actions),result=deepcopy(result)))
        return result

class GraphTool:
    def __init__(self,session,name):self.session,self.name=session,name
    def execute(self,analysis,actions):return self.session.apply(self.name,analysis,actions)

def metadata():
    purposes={'del_node':'Delete unsupported noise nodes.','del_edge':'Delete unsupported edges.','add_edge':'Restore missing supported edges.','rewrite':'Repair noise in target/query node text.'}
    examples={'del_node':{'op':'DELETE_NODE','node':2},'del_edge':{'op':'DEL_EDGE','u':0,'v':1},'add_edge':{'op':'ADD_EDGE','u':0,'v':2},'rewrite':{'op':'REWRITE','node':0,'text':'Full repaired text grounded in supported facts'}}
    return {name:dict(tool_name=name,description=purposes[name],parameters={'analysis':'nonempty evidence-based reasoning','actions':f'Nonempty list (1-64 actions) with op={TOOL_OP[name]} and exactly fields {sorted(ACTION_FIELDS[TOOL_OP[name]])}; empty actions are forbidden'},command_example='execution = tool.execute(analysis="reason", actions='+repr([examples[name]])+')',example_note='Illustrative IDs only; use existing permitted nodes and supported edits in the CURRENT graph.') for name in TOOL_ORDER}

def validate_json_literal(value):
    """Reject Python-only literals before they can enter graph state or audit data."""
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:validate_json_literal(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():validate_json_literal(item)
        return
    raise ValueError('Tool arguments must use JSON-compatible literals; invalid type: '+type(value).__name__)

def execute_literal_command(session,name,command):
    """Keep AgentFlow's command interface, execute only one literal graph call."""
    try:
        tree=ast.parse(command.strip())
        if len(tree.body)!=1:raise ValueError('Exactly one tool call is allowed.')
        assignment=tree.body[0]
        if not isinstance(assignment,ast.Assign) or len(assignment.targets)!=1 or not isinstance(assignment.targets[0],ast.Name) or assignment.targets[0].id!='execution':raise ValueError('Assign the tool result to execution.')
        call=assignment.value
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or not isinstance(call.func.value,ast.Name) or call.func.value.id!='tool' or call.func.attr!='execute' or call.args:raise ValueError('Use tool.execute with keyword literals.')
        if any(k.arg is None for k in call.keywords):raise ValueError('No keyword expansion.')
        args={k.arg:ast.literal_eval(k.value) for k in call.keywords}
        if len(args)!=len(call.keywords) or set(args)!={'analysis','actions'}:raise ValueError('Expected analysis and actions exactly once.')
        validate_json_literal(args)
        if name not in TOOL_ORDER:raise ValueError('Unknown graph tool.')
        return GraphTool(session,name).execute(**args)
    except (SyntaxError,ValueError,TypeError,KeyError) as exc:
        result=dict(ok=False,errors=[dict(code='COMMAND_SCHEMA',message=str(exc))],graph_unchanged=True,current_graph=serialize_graph(session.graph))
        session.audit.append(dict(tool=name,analysis=None,actions=None,raw_command=command,result=deepcopy(result)))
        return result

def extract_prediction(text,labels):
    # GraphRL final JSON is task-specific; retain AgentFlow's final Generator call.
    raw=str(text).strip()
    candidates=[]
    try:candidates.append(json.loads(raw))
    except (ValueError,TypeError):pass
    # Upstream Generator can surround the required final JSON with its process summary.
    decoder=json.JSONDecoder()
    for i,c in enumerate(raw):
        if c=='{':
            try:obj,_=decoder.raw_decode(raw[i:]);candidates.append(obj)
            except ValueError:pass
    valid=[o['prediction'] for o in candidates if isinstance(o,dict) and isinstance(o.get('prediction'),str)]
    keys={_label_key(v) for v in valid}
    if len(keys)!=1:return None
    lookup={_label_key(v):v for v in labels}
    return lookup.get(next(iter(keys)))

def score(prediction,truth):return float(prediction is not None and _label_key(prediction)==_label_key(truth))


def graph_key(graph):
    nodes=tuple(sorted((str(k),str(v)) for k,v in graph['nodes'].items()))
    edges=tuple(sorted((min(int(e['u']),int(e['v'])),max(int(e['u']),int(e['v'])),str(e.get('relation',''))) for e in graph['edges']))
    return nodes,edges
