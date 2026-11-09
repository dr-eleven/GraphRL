"""Sample 3500 NC + 1500 LC; mix clean/budgets and stably sort by target degree."""
from __future__ import annotations
import argparse
import collections
import copy
import csv
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'organized_experiments/runtime/workspace/GraphRL'
if not (RUNTIME/'graph_rl/episode.py').is_file():
    RUNTIME=ROOT/'tools/oneclass_dataset_helpers'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def complete(a):
    return len(a)==5 and all(x['budget']==b and x['metadata']['status']=='attacked'
        and not x['metadata'].get('error') and x['metadata'].get('completed_joint_rounds')==b
        and len(x['metadata'].get('text_changes',[]))==b
        and len(x['metadata'].get('added_edges',[]))==b for b,x in enumerate(a,1))

def degree(episode):
    nodes=([episode['target_node']] if episode['task_type']=='NC' else episode['protected_query_nodes'])
    neighbors=collections.defaultdict(set)
    for edge in episode['noisy_graph']['edges']:
        u,v=int(edge['u']),int(edge['v'])
        if u!=v:neighbors[u].add(v);neighbors[v].add(u)
    degrees=[len(neighbors[int(n)]) for n in nodes]
    return list(map(int,nodes)),degrees,sum(degrees)/len(degrees)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'data/train_data_15785_oneclass')
    parser.add_argument('--output',type=Path,default=ROOT/'data/train_data_oneclass_5000_degree_sorted')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();src=args.source.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError('Preserve existing dataset: '+str(out))
    before={n:sha(src/n) for n in ['train_rl.parquet','manifest.json']}
    clean=pq.read_table(src/'train_rl.parquet').to_pylist()
    clean_by_id={int(row['sample_id']):row for row in clean}
    source_position={int(row['sample_id']):i for i,row in enumerate(clean)}
    assert len(clean_by_id)==len(clean)
    rng=random.Random(args.seed);assignment={};task_group_counts={};selected_by_task={}
    for task,count in [('nc',3500),('lc',1500)]:
        pool=sorted(sid for sid,row in clean_by_id.items() if row['task']==task)
        chosen=rng.sample(pool,count);selected_by_task[task]=sorted(chosen)
        clean_ids=set(rng.sample(chosen,count//5))
        attacked=[sid for sid in chosen if sid not in clean_ids];rng.shuffle(attacked)
        for sid in clean_ids:assignment[sid]=0
        assert len(attacked)%5==0
        step=len(attacked)//5
        for b in range(1,6):
            for sid in attacked[(b-1)*step:b*step]:assignment[sid]=b
        task_group_counts[task]={'clean':len(clean_ids),**{f'badgraph_budget_{b}':step for b in range(1,6)}}
    initial_order=sorted(assignment);rng.shuffle(initial_order)
    assert len(assignment)==5000
    # Reuse the previous dataset's episode/executor implementation. The label
    # list comes from validated structured source fields, including apostrophes.
    legacy=load('episode',RUNTIME/'graph_rl/episode.py')
    builder=load('oneclass_legacy_builder',RUNTIME/'graph_rl/prepare_rl_data.py')
    rows=[];legacy_episodes=[];metadata={};info={}
    source_full_5_checked=0;query_edges_removed=0
    for rank,sid in enumerate(initial_order):
        original=clean_by_id[sid];b=assignment[sid]
        if b:
            path=src/f'sample_results/{sid:06d}.json';bundle=json.loads(path.read_text())
            assert complete(bundle),sid
            item=bundle[b-1];row=copy.deepcopy(item['data']);meta=copy.deepcopy(item['metadata'])
            assert int(row['sample_id'])==sid
            source_full_5_checked+=1
        else:
            row=copy.deepcopy(original);meta={'sample_id':sid,'source_index':sid,'budget':0,'status':'clean',
                'dataset':original['dataset'],'task':original['task'],'ground_truth':original['ground_truth']}
        for k in ['dataset','task','ground_truth','target_nodes','label_list']:
            assert row[k]==original[k],(sid,k)
        labels=list(original['label_list']);assert labels and len(labels)==len(set(labels))
        assert original['ground_truth'] in labels,sid
        episode=builder.make_episode(original,row['prompt'],meta,'badgraph' if b else 'clean',sid)
        episode.update(sample_id=sid,source_row_index=source_position[sid],label_space=labels)
        assert list(episode['protected_query_nodes'])==list(original['target_nodes']),sid
        episode['prompt']=legacy.build_restoration_prompt(episode)
        if original['task']=='lc':
            pair=frozenset(original['target_nodes'])
            raw_edges=legacy.parse_graph_prompt(row['prompt'])['edges']
            query_edges_removed+=sum(frozenset((e['u'],e['v']))==pair for e in raw_edges)
            assert all(frozenset((e['u'],e['v']))!=pair for e in episode['noisy_graph']['edges'])
        nodes,degrees,value=degree(episode)
        row.update(source_index=sid,source_row_index=source_position[sid],data_type='badgraph' if b else 'clean',
            budget=b,target_degree=value)
        metadata[sid]=meta
        info[sid]={'source_index':sid,'sample_id':sid,'source_row_index':source_position[sid],
            'original_mixed_position':rank+1,'task':original['task'].upper(),'dataset':original['dataset'],
            'group':'clean' if b==0 else f'badgraph_budget_{b}','budget':b,
            'query_nodes':json.dumps(nodes),'endpoint_degrees':json.dumps(degrees),'target_degree':value}
        rows.append(row);legacy_episodes.append(episode)
        if len(rows)%250==0:print('BUILT',len(rows),'/ 5000',flush=True)
    order=sorted(range(5000),key=lambda i:rows[i]['target_degree'])
    rows=[rows[i] for i in order];legacy_episodes=[legacy_episodes[i] for i in order]
    ids=[row['sample_id'] for row in rows]
    assert len(ids)==len(set(ids))==5000
    assert all(a['target_degree']<=b['target_degree'] for a,b in zip(rows,rows[1:]))
    counts=collections.Counter('clean' if r['budget']==0 else f'badgraph_budget_{r["budget"]}' for r in rows)
    assert counts=={'clean':1000,**{f'badgraph_budget_{b}':800 for b in range(1,6)}}
    assert collections.Counter(r['task'] for r in rows)=={'nc':3500,'lc':1500}
    out.mkdir(parents=True);(out/'groups').mkdir()
    schema=pa.Table.from_pylist(rows).schema
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),out/'train.parquet',compression='zstd')
    with (out/'train.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    selected_clean=[clean_by_id[s] for s in ids]
    pq.write_table(pa.Table.from_pylist(selected_clean),out/'clean.parquet',compression='zstd')
    groups={name:[r['sample_id'] for r in rows if ('clean' if not r['budget'] else f'badgraph_budget_{r["budget"]}')==name] for name in counts}
    for name,group_ids in groups.items():
        members=set(group_ids)
        pq.write_table(pa.Table.from_pylist([r for r in rows if r['sample_id'] in members],schema=schema),out/'groups'/f'{name}.parquet',compression='zstd')
    dump(out/'group_indices.json',groups)
    with (out/'attack_metadata.jsonl').open('w') as f:
        for sid in ids:f.write(json.dumps(metadata[sid],ensure_ascii=False)+'\n')
    for name,episodes in [('train_grpo_episodes.json',legacy_episodes)]:
        (out/name).write_text(json.dumps(episodes,ensure_ascii=False))
    records=[{'position':i+1,**info[sid]} for i,sid in enumerate(ids)]
    with (out/'target_degree_order.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    # Re-read saved files and compare their stable IDs and payloads.
    saved=pq.read_table(out/'train.parquet').to_pylist();assert saved==rows
    for filename in ['train_grpo_episodes.json']:
        restored=json.loads((out/filename).read_text());assert [e['sample_id'] for e in restored]==ids
    assert [r['sample_id'] for r in pq.read_table(out/'clean.parquet').to_pylist()]==ids
    assert all(sha(src/n)==value for n,value in before.items())
    files={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.rglob('*') if p.is_file()}
    manifest={'rows':5000,'seed':args.seed,'source':str(src),'source_hashes':before,'source_rows':len(clean),
        'task_counts':{'nc':3500,'lc':1500},'group_counts':dict(counts),'task_group_counts':task_group_counts,
        'clean_fraction':0.2,'attack_fraction':0.8,'each_budget_fraction_within_attack':0.2,
        'selected_ids_by_task':selected_by_task,'group_indices_semantics':'stable source sample_id; source_index aliases sample_id; source_row_index is the source clean parquet row position',
        'selection':'Uniform random sampling within task, then task-stratified clean/budget assignment; no replacement; one version per sample',
        'sorting':{'graph':'episode noisy_graph after LC query-edge removal','nc':'target degree','lc':'mean degree of the two query endpoints',
            'degree':'undirected unique neighbors; ignore self-loops and relation types','ties':'stable seeded mixed input order',
            'min':min(r['target_degree'] for r in rows),'max':max(r['target_degree'] for r in rows)},
        'validation':{'unique_ids':5000,'full_5_round_attack_sources_checked':source_full_5_checked,
            'candidate_labels_valid':True,'task_and_group_quotas_exact':True,'degree_nondecreasing':True,'all_formats_same_id_order':True,
            'source_unchanged':True,'lc_query_edges_removed_in_episodes':query_edges_removed},
        'builder':str(Path(__file__)),'builder_sha256':sha(Path(__file__)),'files':files,
        'created_at':datetime.datetime.now().astimezone().isoformat()}
    dump(out/'manifest.json',manifest)
    dump(out/'train_grpo_episodes_summary.json',{'episodes':5000,'groups':dict(counts),'task_counts':{'NC':3500,'LC':1500},'seed':args.seed,'sorting':manifest['sorting']})
    (out/'README.md').write_text('''# 5,000条同类候选攻击混合训练数据\n\n来源：`../train_data_15785_oneclass`；随机种子42；NC 3,500条、LC 1,500条，无LP。每个原始sample_id只选一个版本。\n\n| 任务 | clean | budget1 | budget2 | budget3 | budget4 | budget5 | 合计 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n| NC | 700 | 560 | 560 | 560 | 560 | 560 | 3,500 |\n| LC | 300 | 240 | 240 | 240 | 240 | 240 | 1,500 |\n| 合计 | 1,000 | 800 | 800 | 800 | 800 | 800 | 5,000 |\n\n按episode实际输入的noisy_graph计算目标度并升序排列：NC取目标节点度，LC取两个查询端点度数的平均值。邻居无向、去重，忽略自环和关系类型；LC查询边先移除。同度保留种子42生成的混合顺序。\n\n- `train.parquet`、`train.jsonl`：选定clean或budget版本的5,000条记录。\n- `clean.parquet`：相同5,000个sample_id对应的攻击前clean记录，按训练数据顺序排列；它是奖励参考，不是额外训练样本。\n- `train_grpo_episodes.json`：旧格式episode，候选标签取结构化label_list。\n- `groups/`：六个互斥组的Parquet，各自维持目标度排序。\n- `attack_metadata.jsonl`：按训练顺序保存对应版本的攻击元信息。\n- `target_degree_order.csv`：排序位置、原始ID、目标端点及其度数。\n- `group_indices.json`：组内原始sample_id；source_index同样使用稳定sample_id，不是新文件行号。\n- `manifest.json`：来源、抽样配额、ID、排序规则、完整性验证和文件哈希。\n\n所有选中的攻击样本均来自完整完成5轮的源记录，但每条训练样本只使用分配到的budget版本。未重新调用模型或生成攻击。原始prompt和图字段在train.parquet中保留；episode额外移除LC查询边，目标度使用episode口径。\n''')
    print(json.dumps({k:manifest[k] for k in ['rows','task_counts','group_counts','sorting','validation']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
