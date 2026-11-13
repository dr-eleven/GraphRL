"""Task validation and leakage-free graph views for NC, LC and binary LP."""
from copy import deepcopy
import re
from label_utils import _label_key

TASKS={'NC','LC','LP'}

def node_id(value):
    if isinstance(value,bool):raise ValueError('Boolean is not a node ID')
    match=re.fullmatch(r'(?:node)?(\d+)',str(value).strip(),re.I)
    if match is None:raise ValueError(f'Invalid node ID: {value!r}')
    return int(match.group(1))

def lp_label_map(labels,explicit=None):
    """Map exact candidate labels to polarity; never score by yes/no substring."""
    if len(labels)!=2:raise ValueError('LP requires exactly two candidate labels')
    if explicit is not None:
        if not isinstance(explicit,dict) or any(type(v) is not bool for v in explicit.values()):
            raise ValueError('lp_label_polarity must map candidate labels to JSON booleans')
        mapping={_label_key(k):v for k,v in explicit.items()}
        if len(mapping)!=2 or set(mapping)!={_label_key(v) for v in labels}:
            raise ValueError('LP polarity mapping must cover both exact candidate labels')
    else:
        mapping={}
        for label in labels:
            key=_label_key(label)
            if key in {'1','true','positive','exists','edge exists','link exists'} or re.match(r'^yes\b',key):
                mapping[key]=True
            elif key in {'0','false','negative','absent','no edge','no link','edge absent','link absent'} or re.match(r'^no\b',key):
                mapping[key]=False
            else:
                raise ValueError('Ambiguous LP label; provide lp_label_polarity explicitly: '+label)
    if set(mapping.values())!={False,True}:
        raise ValueError('LP labels must contain one positive and one negative candidate')
    return mapping

def validate_labels(sample,supervision=True):
    labels=sample.get('label_space')
    if not isinstance(labels,list) or not labels or any(not isinstance(v,str) or not _label_key(v) for v in labels):
        raise ValueError('Candidate labels must be a nonempty list of nonempty strings')
    keys=[_label_key(v) for v in labels]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate candidate labels after normalization')
    task=str(sample.get('task_type','')).upper()
    if task not in TASKS:raise ValueError('Supported tasks are NC, LC and LP')
    if supervision and (not isinstance(sample.get('ground_truth'),str) or _label_key(sample['ground_truth']) not in keys):
        raise ValueError('ground_truth must match a candidate label')
    if task=='LP':return lp_label_map(labels,sample.get('lp_label_polarity'))
    return None

def graph_view(graph,targets,mask_query,directed):
    if not isinstance(graph,dict) or not isinstance(graph.get('nodes'),dict):
        raise ValueError('Graph requires a nodes dictionary')
    nodes={}
    for key,value in graph['nodes'].items():
        n=node_id(key)
        if str(n) in nodes:raise ValueError('Duplicate normalized node ID')
        if not isinstance(value,str):raise ValueError('Node text must be a string; missing evidence may be an explicit placeholder')
        nodes[str(n)]=value
    if any(str(n) not in nodes for n in targets):raise ValueError('Target node missing from graph')
    if not isinstance(graph.get('edges',[]),list):raise ValueError('Graph edges must be a list')
    edges=[];removed=0
    for source in graph.get('edges',[]):
        if not isinstance(source,dict) or not {'u','v'}<=source.keys():raise ValueError('Edge requires u and v')
        edge=deepcopy(source);u,v=node_id(edge['u']),node_id(edge['v'])
        if str(u) not in nodes or str(v) not in nodes:raise ValueError('Edge endpoint missing from graph')
        edge.update(u=u,v=v)
        is_query=mask_query and ((u,v)==tuple(targets) if directed else frozenset((u,v))==frozenset(targets))
        if is_query:removed+=1
        else:edges.append(edge)
    return {'nodes':nodes,'edges':edges},removed

def prepare_episode(sample,*,supervision=True,graph=None):
    """Copy and sanitize; preserve sample order/labels and never derive gold from a masked edge."""
    result=deepcopy(sample)
    if supervision and (type(result.get('source_index')) is not int or result['source_index']<0):
        raise ValueError('source_index must be a nonnegative integer')
    task=str(result.get('task_type','')).upper();result['task_type']=task
    validate_labels(result,supervision=supervision)
    if task=='NC':
        target=result.get('target_node')
        if target is None:
            values=result.get('protected_query_nodes',result.get('target_nodes',[]))
            if len(values)==1:target=values[0]
            else:target=result.get('target','')
        targets=[node_id(target)];result['target_node']=targets[0]
        result['target']=f'node{targets[0]}'
    else:
        values=result.get('protected_query_nodes',result.get('target_nodes'))
        if values is None:values=re.findall(r'node(\d+)',str(result.get('target','')),re.I)
        if not isinstance(values,(list,tuple)) or len(values)!=2:raise ValueError('LC/LP require exactly two query endpoints')
        targets=[node_id(n) for n in values]
        if task=='LP' and targets[0]==targets[1]:raise ValueError('LP endpoints must be distinct')
        result['target']=f'query pair node{targets[0]}, node{targets[1]}'
    for name in ['protected_query_nodes','target_nodes']:
        if name in result and list(map(node_id,result[name]))!=targets:raise ValueError('Conflicting target-node fields')
    result['protected_query_nodes']=targets
    directed=result.get('query_directed',False)
    if type(directed) is not bool:raise ValueError('query_directed must be a JSON boolean')
    result['query_directed']=directed
    noisy,removed=graph_view(result['noisy_graph'] if graph is None else graph,targets,task!='NC',directed)
    result['noisy_graph']=noisy
    if supervision:
        if 'clean_graph' not in result:raise ValueError('Missing clean_graph reward reference')
        result['clean_graph'],clean_removed=graph_view(result['clean_graph'],targets,task!='NC',directed)
        audit=result.get('_query_edge_masking',{})
        result['_query_edge_masking']={'noisy':int(audit.get('noisy',0))+removed,'clean':int(audit.get('clean',0))+clean_removed}
    else:
        result.pop('clean_graph',None)
    return result

def dataset_statistics(rows):
    from collections import Counter
    counts=Counter(r['task_type'] for r in rows)
    lp=Counter()
    for row in rows:
        if row['task_type']=='LP':
            mapping=lp_label_map(row['label_space'],row.get('lp_label_polarity'))
            lp['positive' if mapping[_label_key(row['ground_truth'])] else 'negative']+=1
    return {'count':len(rows),'by_task':dict(counts),'by_dataset_and_task':dict(Counter(
        f"{r.get('dataset','unknown')}/{r['task_type']}" for r in rows)),
        'lp_polarity_counts':dict(lp),
        'query_edges_removed':{name:sum(r.get('_query_edge_masking',{}).get(name,0) for r in rows) for name in ['clean','noisy']}}
