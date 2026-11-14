"""Collect independent request errors without executing or consulting hidden supervision."""
from graph_state import TOOL_ORDER, TOOL_OP, ACTION_FIELDS, text_nodes, _node_id, _edge_key, _edge_map
from label_utils import _label_key
from protocol import strict_json


def inspect_tool(payload, sample, graph, expected_tool, max_actions=64):
    errors = []
    def add(code, path, message):
        errors.append(dict(code=code,path=path,message=message))
    if not isinstance(payload,dict):
        add('TOOL_SCHEMA','$','Tool payload must be an object.')
        return None,None,errors
    missing={'name','arguments'}-set(payload)
    extra=set(payload)-{'name','arguments'}
    if missing:add('TOOL_SCHEMA','$','Missing fields: '+', '.join(sorted(missing)))
    if extra:add('TOOL_SCHEMA','$','Unexpected fields: '+', '.join(sorted(extra)))
    name=payload.get('name')
    known=isinstance(name,str) and name in TOOL_ORDER
    if 'name' in payload and not known:
        add('UNKNOWN_TOOL','name','Unknown tool; allowed: '+', '.join(TOOL_ORDER))
    if known and expected_tool and name not in ([expected_tool] if isinstance(expected_tool,str) else expected_tool):
        add('WRONG_STAGE','name','Tool is out of order; only the current stage, or its immediate successor after success, is allowed.')
    arguments=payload.get('arguments')
    if isinstance(arguments,str):arguments=strict_json(arguments)  # parsing errors remain terminal
    if not isinstance(arguments,dict):
        if 'arguments' in payload:add('ARGUMENT_SCHEMA','arguments','Must be an object with analysis then actions.')
        return name,None,errors
    if list(arguments)!=['analysis','actions']:
        add('ARGUMENT_SCHEMA','arguments','Required fields/order: analysis, actions; no extra fields.')
    if 'analysis' in arguments and (not isinstance(arguments['analysis'],str) or not arguments['analysis'].strip()):
        add('ARGUMENT_SCHEMA','arguments.analysis','Must be a nonempty string.')
    actions=arguments.get('actions')
    if not isinstance(actions,list):
        if 'actions' in arguments:add('ACTION_SCHEMA','arguments.actions','Must be a list.')
        return name,arguments,errors
    if len(actions)>max_actions:add('ACTION_SCHEMA','arguments.actions',f'At most {max_actions} actions are allowed.')
    typed=[]
    for index,action in enumerate(actions):
        path=f'actions[{index}]'
        if not isinstance(action,dict):
            add('ACTION_SCHEMA',path,'Must be an object.');continue
        op=action.get('op')
        if not isinstance(op,str) or op not in ACTION_FIELDS:
            add('ACTION_SCHEMA',path+'.op','Unknown operation.')
            # Still report independently checkable ID/text errors.
        elif known and op!=TOOL_OP[name]:
            add('ACTION_SCHEMA',path+'.op',f'{name} requires {TOOL_OP[name]}.')
        fields=ACTION_FIELDS.get(op) if isinstance(op,str) else None
        if fields is not None and set(action)!=fields:
            add('ACTION_SCHEMA',path,'Required fields: '+', '.join(sorted(fields)))
        for key in ['node','u','v']:
            if key in action and type(action[key]) is not int:
                add('ACTION_SCHEMA',path+'.'+key,'Must be a JSON integer, not bool/string/float.')
        if op=='REWRITE' and 'text' in action and (not isinstance(action['text'],str) or not action['text'].strip()):
            add('ACTION_SCHEMA',path+'.text','Must be nonempty full text.')
        if fields and fields<=set(action) and all(type(action[k]) is int for k in ['node','u','v'] if k in fields):
            typed.append((index,action))
    nodes=set(map(int,graph['nodes']))
    edges=_edge_map(graph)
    protected={_node_id(n) for n in sample.get('protected_nodes',[])}|set(text_nodes(sample))
    if sample.get('target_node') is not None:protected.add(_node_id(sample['target_node']))
    protected_edges={_edge_key(*x) for x in sample.get('protected_edges',[])}
    editable=set(text_nodes(sample))
    deletes={a['node'] for _,a in typed if a['op']=='DELETE_NODE'}
    seen_nodes,seen_edges,seen_text=set(),set(),set()
    for index,a in typed:
        path=f'actions[{index}]';op=a['op']
        if op=='DELETE_NODE':
            node=a['node']
            if node not in nodes:add('INVALID_ACTION',path,f'Node {node} does not exist.')
            if node in protected:add('INVALID_ACTION',path,f'Node {node} is protected.')
            if node in seen_nodes:add('INVALID_ACTION',path,f'Repeated deletion of node {node}.')
            seen_nodes.add(node)
        elif op in {'ADD_EDGE','DEL_EDGE'}:
            u,v=a['u'],a['v'];key=_edge_key(u,v)
            if u not in nodes:add('INVALID_ACTION',path,f'Endpoint {u} does not exist.')
            if v not in nodes and v!=u:add('INVALID_ACTION',path,f'Endpoint {v} does not exist.')
            if u==v:add('INVALID_ACTION',path,'Self-loop is forbidden.')
            if key in protected_edges:add('INVALID_ACTION',path,'Edge is protected.')
            if key in seen_edges:add('INVALID_ACTION',path,'Repeated or conflicting edge action.')
            if u in deletes or v in deletes:add('INVALID_ACTION',path,'Edge conflicts with node deletion in this batch.')
            if op=='ADD_EDGE' and key in edges:add('INVALID_ACTION',path,'ADD_EDGE requires an absent edge.')
            if op=='DEL_EDGE' and key not in edges:add('INVALID_ACTION',path,'DEL_EDGE requires a present edge.')
            seen_edges.add(key)
        elif op=='REWRITE':
            node=a['node']
            if node not in nodes:add('INVALID_ACTION',path,f'Node {node} does not exist.')
            if node not in editable:add('INVALID_ACTION',path,f'Node {node} is not an editable target/query endpoint.')
            if node in deletes:add('INVALID_ACTION',path,'Rewrite conflicts with node deletion in this batch.')
            if node in seen_text:add('INVALID_ACTION',path,f'Repeated rewrite of node {node}.')
            if node in nodes and isinstance(a['text'],str):
                current=graph['nodes'].get(str(node),graph['nodes'].get(node,''))
                norm=lambda value:' '.join(str(value).split()).casefold()
                if norm(a['text'])==norm(current):add('INVALID_ACTION',path,'REWRITE must change the current text.')
            seen_text.add(node)
    return name,arguments,errors


def inspect_final(payload, labels):
    errors=[]
    def add(code,path,message):errors.append(dict(code=code,path=path,message=message))
    if not isinstance(payload,dict):
        add('FINAL_SCHEMA','$','Final payload must be an object.');return None,errors
    if list(payload)!=['analysis','prediction']:add('FINAL_SCHEMA','$','Required fields/order: analysis, prediction.')
    if 'analysis' in payload and (not isinstance(payload['analysis'],str) or not payload['analysis'].strip()):
        add('FINAL_SCHEMA','analysis','Must be a nonempty string.')
    value=payload.get('prediction')
    if not isinstance(value,str) or _label_key(value) not in labels:
        add('PUBLIC_CANDIDATE','prediction','Must be one public candidate label.')
    return (labels.get(_label_key(value)) if isinstance(value,str) else None),errors
