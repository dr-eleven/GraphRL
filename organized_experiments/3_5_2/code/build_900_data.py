"""原始5495条分层抽样；每层20 clean、各budget16；按noisy目标度数升序。"""
import collections,copy,hashlib,importlib.util,json,random,sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3]
RUNTIME=ROOT/'organized_experiments/runtime/workspace/GraphRL'
OUT=ROOT/'900_train_data_clean20_badgraph80_degree_sorted'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
 if OUT.exists():raise FileExistsError(OUT)
 source=ROOT/'data/train_data_5k';cp=source/'train_sft_data_badgraph_input.parquet'
 clean=pq.read_table(cp).to_pylist();rng=random.Random(42);groups=collections.defaultdict(list)
 for i,row in enumerate(clean):groups[(row['dataset'],row['task'].upper())].append(i)
 assert len(groups)==9
 assignments={};selected={}
 for key,ids in sorted(groups.items()):
  chosen=rng.sample(ids,100);rng.shuffle(chosen);selected['/'.join(key)]=chosen
  for j,sid in enumerate(chosen):assignments[sid]=0 if j<20 else 1+(j-20)//16
 legacy=load('episode',RUNTIME/'graph_rl/episode.py');builder=load('legacy_builder900',RUNTIME/'graph_rl/prepare_rl_data.py')
 templates={e['source_index']:e for e in json.loads((ROOT/'GraphRL_train_data_clean20_badgraph80_degree_sorted/train_grpo_episodes_redesign.json').read_text())}
 rows=[];episodes=[];sources={str(cp):hashlib.sha256(cp.read_bytes()).hexdigest()}
 for b in range(6):
  ids=[i for i,v in assignments.items() if v==b]
  if b:
   folder=source/f'badgraph_attack_train_data/budget{b}/badgraph';pp=next(folder.glob('*.parquet'));mp=next(folder.glob('*_metadata.jsonl'))
   variants=pq.read_table(pp).to_pylist();meta=[json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
   for p in (pp,mp):sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  else:variants=clean;meta=[{}]*len(clean)
  for sid in ids:
   row=copy.deepcopy(variants[sid]);e=builder.make_episode(clean[sid],row['prompt'],meta[sid],'badgraph' if b else 'clean',sid)
   e['label_space']=templates[sid]['label_space'];e['text_edit_nodes']=e['protected_query_nodes'] if e['task_type']=='LC' else [e['target_node']]
   assert e['ground_truth'] in e['label_space']
   e.pop('prompt',None);e.pop('oracle_completion',None)
   # JSON object keys are strings, matching serialized source episodes.
   e=json.loads(json.dumps(e));neighbors=collections.defaultdict(set)
   for edge in e['noisy_graph']['edges']:
    u,v=int(edge['u']),int(edge['v'])
    if u!=v:neighbors[u].add(v);neighbors[v].add(u)
   targets=[e['target_node']] if e['task_type']=='NC' else e['protected_query_nodes']
   degree=sum(len(neighbors[int(n)]) for n in targets)/len(targets)
   e['target_degree']=degree;row.update(source_index=sid,budget=b,data_type='clean' if b==0 else 'badgraph',target_degree=degree)
   rows.append(row);episodes.append(e)
 order=sorted(range(900),key=lambda i:(rows[i]['target_degree'],rows[i]['source_index']))
 rows=[rows[i] for i in order];episodes=[episodes[i] for i in order]
 assert len(set(e['source_index'] for e in episodes))==900
 counts=collections.Counter((e['dataset'],e['task_type'],e['attack_budget']) for e in episodes)
 assert all(counts[(d,t,b)]==(20 if b==0 else 16) for d,t in groups for b in range(6))
 OUT.mkdir();pq.write_table(pa.Table.from_pylist(rows),OUT/'train.parquet',compression='zstd')
 (OUT/'train_grpo_episodes_redesign.json').write_text(json.dumps(episodes,ensure_ascii=False)+'\n')
 manifest=dict(seed=42,total=900,clean=180,badgraph=720,budgets={str(b):144 for b in range(1,6)},group_counts={'/'.join(k):len(v[:100]) for k,v in groups.items()},selected_ids=selected,source_sha256=sources,sort='noisy NC target degree / LC mean endpoint degree ascending, source_index tie-break',actual_unchanged_by_budget=dict(collections.Counter(str(e['attack_budget']) for e in episodes if e['noisy_graph']==e['clean_graph'])),note='Existing original BadGraph variants retained; partial/ineffective attacks not silently replaced.')
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k not in ('selected_ids','source_sha256')},indent=2))
if __name__=='__main__':main()
