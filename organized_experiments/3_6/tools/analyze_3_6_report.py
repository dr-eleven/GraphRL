import json,pathlib,hashlib,collections,statistics,math,datetime,os,shutil,ast,re
B=pathlib.Path('/home/temptest/GraphRL/organized_experiments/3_6')
RUNS=B/'runs'
names=['3_6_gpu1_cpuoffload_20261108_021730','3_6_gpu1_bgefix_resume40_20261108_073206','3_6_gpu1_z2_b12_a1_cap5120_resume320_20261109_122155']
current=RUNS/names[-1]
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
progress=json.loads(pathlib.Path(os.environ.get('TASK_REPORT_SNAPSHOT',str(current/'progress.json'))).read_text()); cutoff=progress['global_step']
stamp=now.strftime('%Y%m%d_%H%M%S'); out=B/'reports'/'analysis'/stamp; out.mkdir(parents=True, exist_ok=False)
data_path=pathlib.Path('/home/temptest/GraphRL/data/3_5_1_3_6/train_episodes.json')
data=json.loads(data_path.read_text()); samples={s['source_index']:s for s in data}
stages=['del_node','del_edge','add_edge','rewrite','prediction']; tools=stages[:4]
def nodes(g): return {int(k):v for k,v in g['nodes'].items()}
def edges(g): return {tuple(sorted((int(e['u']),int(e['v'])))) for e in g['edges']}
def text(v): return v if isinstance(v,str) else json.dumps(v,sort_keys=True,ensure_ascii=False)
def distance(g,c): return len(set(nodes(g))^set(nodes(c)))+len(edges(g)^edges(c))
def full(g,c): return nodes(g)==nodes(c) and edges(g)==edges(c)
perf={}
for line in (current/'grpo/performance_steps.jsonl').read_text().splitlines():
 p=json.loads(line)
 if p['global_step']<=cutoff: perf[p['global_step']]=p
rows=[]; manifest=[]; examples={}; anomalies=collections.Counter(); stage_error_sets=collections.defaultdict(set)
example_raw={}
detail=(out/'trajectory_summaries.jsonl').open('w')
for name,lo,hi in [(names[0],0,40),(names[1],40,320),(names[2],320,cutoff)]:
 path=RUNS/name/'grpo/rollouts_rank0.jsonl'; limit=path.stat().st_size; digest=hashlib.sha256(); used=0; usedbytes=0; counts=collections.Counter()
 with path.open('rb') as f:
  while f.tell()<limit:
   offset=f.tell(); raw=f.readline()
   if not raw.endswith(b'\n'): break
   a=json.loads(raw); step=a['step']
   if step>=hi: break
   digest.update(raw); usedbytes=f.tell()
   if step<lo: continue
   used+=1; counts[step]+=1
   s=samples[a['source_index']]; m=a['metrics']; ev={e['expected_tool']:e for e in a['events']}; calls=[e for e in a['events'] if e.get('format_valid')]
   # Completion masks distinguish generated tokens from tool feedback and padding.
   segments=[]; chunk=[]
   for tid,cm,tm in zip(a['completion_ids'],a['completion_mask'],a['tool_mask']):
    if cm and tm: chunk.append(tid)
    elif chunk: segments.append(chunk); chunk=[]
   if chunk: segments.append(chunk)
   if len(segments)>5: anomalies['more_than_5_generated_segments']+=1
   token_available=not a.get('training_discarded',False)
   generated_count=sum(msg['role']=='assistant' for msg in a['messages'])
   if token_available and len(segments)!=generated_count: anomalies['generated_segment_count_mismatch']+=1
   if not (len(a['completion_ids'])==len(a['completion_mask'])==len(a['tool_mask'])): anomalies['mask_length_mismatch']+=1
   fs=next((t for t in stages if m.get('structure_'+t,0)<0.199),None)
   fmterr=bool(a.get('format_errors'))
   # A length cap stop is separate from protocol errors, even when no stage is reached.
   ferr={t:[] for t in stages}
   if fmterr and fs:
    ferr[fs]=a['format_errors']
   illegal={t:ev[t]['error'] for t in tools if t in ev and ev[t].get('error') and ev[t].get('format_valid')}
   for t,e in illegal.items(): stage_error_sets[t].add(e)
   before_dist=distance(s['noisy_graph'],s['clean_graph']); after_dist=distance(a['final_graph'],s['clean_graph'])
   dirty=bool(m.get('actual_corrupt'))
   st={}
   for t in tools:
    e=ev.get(t); valid=bool(e and e.get('format_valid')); acts=(e.get('call') or {}).get('actions',[]) if e else []
    executed=acts if valid and not e.get('error') else []
    clean=s['clean_graph']; cn=nodes(clean); ce=edges(clean)
    ref=sum((x.get('node') not in cn) if t=='del_node' else (tuple(sorted((x['u'],x['v']))) not in ce) if t=='del_edge' else (tuple(sorted((x['u'],x['v']))) in ce) if t=='add_edge' else (x.get('node') in cn and text(x.get('text'))==text(cn[x['node']])) for x in executed)
    st[t]={'dispatch':int(valid),'empty':int(valid and not e.get('error') and not acts),'nonempty':int(valid and not e.get('error') and bool(acts)),'illegal':int(t in illegal),'error':illegal.get(t),'submitted':len(acts),'executed':len(executed),'reference':ref,'hit':m.get('hit_'+t+'_count',0),'gt':m.get('actionable_count_'+t,0),'unactionable':m.get('unactionable_count_'+t,0),'need':bool(m.get('operation_needed_'+t,0)),'has':bool(acts),'B':m.get('behavior_'+t,0),'S':m.get('structure_'+t,0)}
   r={'step':step,'window':step+1,'source':a['source_index'],'run':name,'offset':offset,'task':s['task_type'],'dataset':s['dataset'],'dirty':dirty,'discarded':bool(a.get('training_discarded',False)),'S':m['structure_reward'],'B':m['behavior_reward'],'T':m['task_reward'],'H':m.get('hit_reward',0),'R':a['reward'],'correct':m['prediction_correct'],'format_valid':m.get('format_valid',0),'prediction_format_valid':m.get('prediction_format_valid',0),'format_error':fmterr,'format_errors':ferr,'illegal':bool(illegal),'stages':st,'stageS':{t:m.get('structure_'+t,0) for t in stages},'tokens':[len(x) for x in segments],'full':full(a['final_graph'],s['clean_graph']),'structure_exact':after_dist==0,'dist_before':before_dist,'dist_after':after_dist,'text_exact':m.get('reference_text_exact',0),'advantage':a.get('advantage'),'limit_events':a.get('generation_limit_events',[]),'finish':a.get('finish_reason'),'metrics':{k:v for k,v in m.items() if k.startswith('rewrite_node_')}}
   r['generated_count']=generated_count
   r['token_available']=token_available
   r['observed_execution']=m.get('observed_execution',0)
   r['without_visible_feedback']=bool(a['executed_actions']) and not bool(m.get('observed_execution',0))
   if abs(r['R']-sum(r[k] for k in ['S','B','T','H']))>1e-6: anomalies['reward_sum_mismatch']+=1
   if abs(r['S']-sum(r['stageS'].values()))>1e-6: anomalies['stage_S_sum_mismatch']+=1
   if abs(r['B']-sum(r['stages'][t]['B'] for t in tools))>1e-6: anomalies['stage_B_sum_mismatch']+=1
   if abs(r['H']-sum(r['stages'][t]['hit']*(.2 if t=='rewrite' else .1) for t in tools))>1e-6: anomalies['stage_H_sum_mismatch']+=1
   if fmterr and r['T']!=0: anomalies['format_error_T_nonzero']+=1
   for e in calls:
    if e.get('error') and e['before_hash']!=e['after_hash']: anomalies['illegal_changed_hash']+=1
   rows.append(r); detail.write(json.dumps(r,ensure_ascii=False)+'\n')
   if s['task_type']=='NC':
    conditions={'最早完整NC':not fmterr and len(segments)==5,'格式错误终止':fmterr,'非法action后继续':bool(illegal) and len(segments)==5,'B满分但图未完整恢复':r['B']>=.999 and not r['full'] and dirty,'候选内错误得0.1':abs(r['T']-.1)<1e-6,'改写获得H':st['rewrite']['hit']>0,'长度丢弃':r['discarded'],'末段完整NC':not fmterr and len(segments)==5}
    for label,condition in conditions.items():
     if condition and (label not in example_raw or label in ['末段完整NC','改写获得H']):
      example_raw[label]={'row':a,'segments':segments,'summary':r}
 manifest.append({'path':str(path),'step_range':[lo,hi-1],'file_size_snapshot':limit,'hashed_prefix_bytes':usedbytes,'sha256':digest.hexdigest(),'included_rows':used,'window_counts':dict(counts)})
detail.close()
groups=collections.defaultdict(list)
for r in rows: groups[(r['step'],r['source'])].append(r)
assert set(r['step'] for r in rows)==set(range(cutoff)), 'missing windows'
assert all(len(g)==6 for g in groups.values()), 'group count !=6'
assert all(len({r['discarded'] for r in g})==1 for g in groups.values()), 'partial group discarded'
assert all(sum(1 for r in rows if r['step']==s)==12 for s in range(cutoff)), 'window count !=12'
for w,p in perf.items():
 rr=[r for r in rows if r['window']==w]
 if sum(not r['discarded'] for r in rr)!=p['kept_candidates']: anomalies['kept_count_mismatch']+=1
assert not anomalies, dict(anomalies)

# Rendering is separate from log extraction; only the renderer writes the report.
from render_3_6_report import render
render(globals())
