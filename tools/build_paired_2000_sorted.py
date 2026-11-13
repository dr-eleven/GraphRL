"""Paired one-class/control datasets using the stratified 900-sample protocol."""
from __future__ import annotations
import argparse
import collections
import copy
import csv
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'900_train_data_clean20_badgraph80_degree_sorted'
REFERENCE_BUILDER=ROOT/'organized_experiments/3_5_2/code/build_900_data.py'
RUNTIME=ROOT/'organized_experiments/runtime/workspace/GraphRL'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def complete(a):
    return len(a)==5 and all(x['budget']==b and x['metadata']['status']=='attacked'
        and not x['metadata'].get('error') and x['metadata'].get('completed_joint_rounds')==b
        and len(x['metadata'].get('text_changes',[]))==b and len(x['metadata'].get('added_edges',[]))==b
        for b,x in enumerate(a,1))

def degree(e):
    neighbors=collections.defaultdict(set)
    for edge in e['noisy_graph']['edges']:
        u,v=int(edge['u']),int(edge['v'])
        if u!=v:neighbors[u].add(v);neighbors[v].add(u)
    targets=[e['target_node']] if e['task_type']=='NC' else e['protected_query_nodes']
    values=[len(neighbors[int(n)]) for n in targets]
    return sum(values)/len(values),values

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output-root',type=Path,default=ROOT)
    args=parser.parse_args();outroot=args.output_root.resolve()
    names={'oneclass':'2000_oneclass_sorted','notoneclass':'2000_notoneclass_sorted'}
    for name in names.values():
        if (outroot/name).exists():raise FileExistsError('Preserve existing output: '+str(outroot/name))
    source={k:ROOT/f'data/train_data_15785_{k}' for k in names}
    files=[source['oneclass']/'train_rl.parquet',REFERENCE/'manifest.json',REFERENCE_BUILDER]
    for path in source.values():
        files += [path/f'budget{b}/train_rl_attacked_badgraph_budget{b}.parquet' for b in range(1,6)]
    hashes={str(path):sha(path) for path in files}
    clean_rows=pq.read_table(source['oneclass']/'train_rl.parquet').to_pylist()
    clean={int(row['sample_id']):row for row in clean_rows};assert len(clean)==len(clean_rows)
    common=set(clean)
    for path in source.values():
        for b in range(1,6):
            ids=pq.read_table(path/f'budget{b}/train_rl_attacked_badgraph_budget{b}.parquet',columns=['sample_id']).column(0).to_pylist()
            assert len(ids)==len(set(ids));common.intersection_update(ids)
    strata=collections.defaultdict(list)
    for row in clean_rows:
        sid=int(row['sample_id'])
        if sid in common:strata[(row['dataset'],row['task'].upper())].append(sid)
    assert len(strata)==19 and collections.Counter(t for d,t in strata)=={'NC':10,'LC':1,'LP':8}
    rng=random.Random(args.seed);assignments={};chosen_by_stratum={};quotas={}
    for (dataset,task),ids in sorted(strata.items()):
        count=200 if task=='LC' else 100
        chosen=rng.sample(ids,count);rng.shuffle(chosen)
        key=f'{dataset}/{task}';chosen_by_stratum[key]=chosen
        clean_count=count//5;per_budget=(count-clean_count)//5
        quotas[key]={'total':count,'clean':clean_count,**{f'budget{b}':per_budget for b in range(1,6)}}
        for j,sid in enumerate(chosen):assignments[sid]=0 if j<clean_count else 1+(j-clean_count)//per_budget
    assert len(assignments)==2000
    assert collections.Counter(assignments.values())=={0:400,**{b:320 for b in range(1,6)}}
    assert collections.Counter(clean[s]['task'] for s in assignments)=={'nc':1000,'lc':200,'lp':800}
    legacy=load('episode',RUNTIME/'graph_rl/episode.py')
    builder=load('paired2000_episode_builder',RUNTIME/'graph_rl/prepare_rl_data.py')
    stage=Path(tempfile.mkdtemp(prefix='.paired2000_',dir=outroot))
    exports={};reports={}
    for kind,name in names.items():
        out=stage/name;out.mkdir();rows=[];episodes=[];status_counts=collections.Counter()
        sources_checked=0
        for b in range(6):
            if b:
                table=pq.read_table(source[kind]/f'budget{b}/train_rl_attacked_badgraph_budget{b}.parquet')
                wanted={s for s,selected_budget in assignments.items() if selected_budget==b}
                variants={int(row['sample_id']):row for row in table.to_pylist() if row['sample_id'] in wanted}
                assert len(variants)==320
                del table
            else:variants=clean
            for sid,selected_budget in assignments.items():
                if selected_budget!=b:continue
                row=copy.deepcopy(variants[sid]);original=clean[sid]
                for field in ['dataset','task','ground_truth','target_nodes','label_list']:
                    assert row[field]==original[field],(kind,sid,field)
                if b:
                    path=source[kind]/f'sample_results/{sid:06d}.json'
                    bundle=json.loads(path.read_text());assert complete(bundle),(kind,sid)
                    meta=bundle[b-1]['metadata'];status_counts[meta['status']]+=1;sources_checked+=1
                    for field in ['node_information','connection_information']:
                        assert bundle[b-1]['data'][field]==row[field],(kind,sid,field)
                else:meta={'budget':0,'status':'clean'}
                # Extend the reference's pair-task handling to LP while retaining
                # the actual LP task and binary prediction labels in the result.
                clean_view=copy.deepcopy(original)
                if original['task']=='lp':clean_view['task']='lc'
                e=builder.make_episode(clean_view,row['prompt'],meta,'badgraph' if b else 'clean',sid)
                task=original['task'].upper();e['task_type']=task
                e['label_space']=list(original['label_list'])
                assert e['ground_truth'] in e['label_space']
                assert len(e['label_space'])==len(set(e['label_space']))
                assert e['protected_query_nodes']==original['target_nodes']
                e['text_edit_nodes']=e['protected_query_nodes'] if task in ['LC','LP'] else [e['target_node']]
                e.pop('prompt',None);e.pop('oracle_completion',None)
                e=json.loads(json.dumps(e)) # Same JSON-node-key convention as the reference.
                value,endpoint_degrees=degree(e)
                e.update(sample_id=sid,target_degree=value)
                row.update(sample_id=sid,source_index=sid,budget=b,data_type='clean' if b==0 else 'badgraph',target_degree=value)
                rows.append(row);episodes.append(e)
        order=sorted(range(2000),key=lambda i:(rows[i]['target_degree'],rows[i]['source_index']))
        rows=[rows[i] for i in order];episodes=[episodes[i] for i in order]
        ids=[e['source_index'] for e in episodes]
        assert len(ids)==len(set(ids))==2000 and set(ids)==set(assignments)
        assert collections.Counter((e['dataset'],e['task_type'],e['attack_budget']) for e in episodes)=={
            (dataset,task,b):(40 if b==0 else 32) if task=='LC' else (20 if b==0 else 16)
            for dataset,task in strata for b in range(6)}
        pq.write_table(pa.Table.from_pylist(rows),out/'train.parquet',compression='zstd')
        with (out/'train.jsonl').open('w') as handle:
            for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        (out/'train_grpo_episodes_redesign.json').write_text(json.dumps(episodes,ensure_ascii=False)+'\n')
        pq.write_table(pa.Table.from_pylist([clean[s] for s in ids]),out/'clean.parquet',compression='zstd')
        group_ids={'clean' if b==0 else f'badgraph_budget_{b}':[s for s in ids if assignments[s]==b] for b in range(6)}
        dump(out/'group_indices.json',group_ids)
        records=[]
        for position,e in enumerate(episodes,1):
            _,ds=degree(e);sid=e['source_index']
            records.append({'position':position,'sample_id':sid,'source_index':sid,'dataset':e['dataset'],
                'task':e['task_type'],'budget':e['attack_budget'],'target_degree':e['target_degree'],
                'query_nodes':json.dumps(e['protected_query_nodes']),'endpoint_degrees':json.dumps(ds)})
        write_csv(out/'target_degree_order.csv',records)
        # Validate the saved schema and the ordering in both primary formats.
        assert pq.read_table(out/'train.parquet').to_pylist()==rows
        restored=json.loads((out/'train_grpo_episodes_redesign.json').read_text())
        assert [e['source_index'] for e in restored]==ids
        assert all((a['target_degree'],a['source_index'])<=(b['target_degree'],b['source_index']) for a,b in zip(rows,rows[1:]))
        for e in episodes:
            nodes=set(map(int,e['noisy_graph']['nodes']))
            assert all(int(n) in nodes for n in e['protected_query_nodes'])
            assert all(int(edge['u']) in nodes and int(edge['v']) in nodes for edge in e['noisy_graph']['edges'])
            if e['task_type'] in ['LC','LP']:
                pair=frozenset(e['protected_query_nodes'])
                assert all(frozenset((edge['u'],edge['v']))!=pair for edge in e['noisy_graph']['edges'])
        manifest={'seed':args.seed,'total':2000,'clean':400,'badgraph':1600,'budgets':{str(b):320 for b in range(1,6)},
            'task_counts':{'NC':1000,'LC':200,'LP':800},'group_counts':{k:v['total'] for k,v in quotas.items()},
            'stratum_quotas':quotas,'selected_ids':chosen_by_stratum,'sample_budget_assignment':{str(k):v for k,v in assignments.items()},
            'source':str(source[kind]),'clean_reference':str(source['oneclass']/'train_rl.parquet'),
            'eligible_common_pool':len(common),'source_sha256':hashes,'reference_dataset':str(REFERENCE),
            'reference_builder':str(REFERENCE_BUILDER),'builder':str(Path(__file__)),'builder_sha256':sha(Path(__file__)),
            'sort':'noisy NC target degree / LC+LP mean endpoint degree ascending, source_index ascending tie-break',
            'pair_tasks':'LC/LP query edge removed in clean/noisy episodes; raw source graph/prompt fields preserved in parquet',
            'source_index_definition':'stable original sample_id, shared between both variants; not a new row position',
            'episode_format':'same core schema as 900: structured clean/noisy graphs and text_edit_nodes; prompt/oracle omitted; LP retains task_type=LP',
            'degree_min':min(e['target_degree'] for e in episodes),'degree_max':max(e['target_degree'] for e in episodes),
            'actual_unchanged_by_budget':dict(collections.Counter(str(e['attack_budget']) for e in episodes if e['noisy_graph']==e['clean_graph'])),
            'validation':{'unique_source_ids':2000,'attack_sources_full_five_rounds_checked':sources_checked,
                'task_stratum_budget_quotas_exact':True,'degree_then_id_nondecreasing':True,'labels_valid':True,
                'pair_query_edges_masked':True,'all_targets_and_edge_endpoints_exist':True,'primary_formats_same_id_order':True},
            'created_at':datetime.datetime.now().astimezone().isoformat()}
        exports[kind]={'rows':rows,'episodes':episodes,'out':out}
        reports[kind]=manifest
        print('BUILT_VARIANT',kind,2000,flush=True)
    a={r['sample_id']:r for r in exports['oneclass']['rows']};b={r['sample_id']:r for r in exports['notoneclass']['rows']}
    assert set(a)==set(b)
    for sid in a:
        for key in ['dataset','task','ground_truth','target_nodes','label_list','budget','data_type']:
            assert a[sid][key]==b[sid][key],(sid,key)
        if assignments[sid]==0:
            for key in ['prompt','node_information','connection_information','target_degree']:assert a[sid][key]==b[sid][key]
    positions={k:{r['sample_id']:i+1 for i,r in enumerate(v['rows'])} for k,v in exports.items()}
    pairing=[{'sample_id':sid,'dataset':a[sid]['dataset'],'task':a[sid]['task'].upper(),'budget':assignments[sid],
        'oneclass_position':positions['oneclass'][sid],'notoneclass_position':positions['notoneclass'][sid],
        'oneclass_degree':a[sid]['target_degree'],'notoneclass_degree':b[sid]['target_degree']} for sid in sorted(a)]
    assert all(sha(Path(path))==value for path,value in hashes.items()),'Source data changed'
    for kind,value in exports.items():
        out=value['out'];write_csv(out/'pair_alignment.csv',pairing)
        report=reports[kind];report['validation'].update(paired_ids_and_budget_assignments_identical=True,paired_clean_400_payloads_identical=True,source_files_unchanged=True)
        dump(out/'manifest.json',report)
        quota_rows='\n'.join(f'| {key} | {q["clean"]} | '+ ' | '.join(str(q[f'budget{x}']) for x in range(1,6))+f' | {q["total"]} |' for key,q in quotas.items())
        (out/'README.md').write_text(f'''# {names[kind]}\n\n随机种子{args.seed}；沿用900条数据的分层抽样方法，在两类攻击完整结果共有的{len(common)}个sample_id中，19种数据集+任务组合各100条，WN18RR/LC额外100条。每个源ID只使用一个版本。\n\n| 数据集/任务 | clean | b1 | b2 | b3 | b4 | b5 | 合计 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n{quota_rows}\n| 合计 | 400 | 320 | 320 | 320 | 320 | 320 | 2000 |\n\nNC 1000条、LC 200条、LP 800条。两套数据的sample_id、任务、标签、目标节点及clean/budget分配完全相同，clean的400条图文本也相同；攻击版本来自各自来源目录。\n\n以episode的noisy_graph计算无向、去重邻居度，忽略自环及关系类型：NC取目标节点度，LC/LP移除查询边后取两端度数平均。按degree升序，同degree按source_index升序。两套图的degree可能不同，排序后行号不必相同；按sample_id/budget配对，见pair_alignment.csv。\n\ntrain.parquet、train.jsonl为选中的原始任务格式；train_grpo_episodes_redesign.json沿用900条版本的结构化episode核心字段，并保留LP任务类型及二元候选标签。该episode不含预生成prompt或oracle，由训练协议的数据准备代码生成。clean.parquet是同2000个ID的奖励参考，不是另外的训练样本。原始图/任务prompt在Parquet中保留，查询边屏蔽作用于episode。\n\nmanifest.json记录ID、分组、哈希、来源和验证；target_degree_order.csv记录各样本排序度；pair_alignment.csv给出两套数据的行号与degree对应关系。构造代码：tools/build_paired_2000_sorted.py。\n''')
        report['output_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        dump(out/'manifest.json',report)
    for kind,name in names.items():os.replace(exports[kind]['out'],outroot/name)
    stage.rmdir()
    print(json.dumps({'outputs':{k:str(outroot/n) for k,n in names.items()},'rows_each':2000,'task_counts':{'NC':1000,'LC':200,'LP':800},'clean_each':400,'each_budget':320,'paired_ids_and_assignments':True},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
