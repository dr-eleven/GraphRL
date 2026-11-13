import os,sys,json,subprocess,tempfile,itertools
from pathlib import Path
# This validates orchestration only: the recorder never loads or trains a model.
script=Path(__file__).resolve().parents[1]/'scripts/run_portable.sh'
with tempfile.TemporaryDirectory() as td:
 t=Path(td); recorder=t/'record';recorder.write_text('#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ["CHECK_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n');recorder.chmod(0o755)
 env=dict(os.environ,TRAIN_MODEL=str(t/'base'),PYTHON_BIN=str(recorder),CHECK_LOG=str(t/'calls'))
 for k in ['TEST_EPISODES','TRAIN_BATCH_SIZE','GRAD_ACCUM','NUM_GENERATIONS','WARMUP_STEPS']:env.pop(k,None)
 valid=[]
 for n,batch,acc,g in [(8,48,1,6),(8,24,2,6),(8,24,1,6),(8,72,1,6),(8,96,1,6),(3,4,2,6)]:
  run=t/f'train_{n}_{batch}_{acc}_{g}'
  e=dict(env,CUDA_VISIBLE_DEVICES=','.join(map(str,range(n))),TRAIN_BATCH_SIZE=str(batch),GRAD_ACCUM=str(acc),NUM_GENERATIONS=str(g),RUN_DIR=str(run))
  r=subprocess.run(['bash',str(script),'train'],env=e,capture_output=True,text=True);assert r.returncode==0,r.stderr
  args=json.loads((t/'calls').read_text().splitlines()[-1])
  assert int(args[args.index('--per_device_train_batch_size')+1])==batch
  assert int(args[args.index('--gradient_accumulation_steps')+1])==acc
  assert int(args[args.index('--num_generations')+1])==g
  assert args[args.index('--warmup_steps')+1]=='0.05'
  valid.append({'gpus':n,'batch':batch,'accum':acc,'G':g,'normal_update_trajectories':n*batch*acc,'normal_update_sources':n*batch*acc//g})
 for b,acc,g in [('48','1','8'),('32','1','6'),('0','1','6'),('48','0','6')]:
  e=dict(env,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',TRAIN_BATCH_SIZE=b,GRAD_ACCUM=acc,NUM_GENERATIONS=g,RUN_DIR=str(t/f'invalid_{b}_{acc}_{g}'))
  r=subprocess.run(['bash',str(script),'train'],env=e,capture_output=True,text=True);assert r.returncode!=0
 # Simulate rank concatenation and reward grouping, including G=6 split across ranks.
 sys.path.insert(0,str(script.parents[1]/'code'))
 from tail_batching import TailSafeRankBatchSampler
 from collections import Counter
 sampler_checks=[]
 for batch in [3,24,48,72,96]:
  ranks=[list(TailSafeRankBatchSampler(2000,6,8,rank,batch)) for rank in range(8)]
  assert len({len(batches) for batches in ranks})==1
  occurrences=Counter()
  for step in range(len(ranks[0])):
   assert len({len(rank_batches[step]) for rank_batches in ranks})==1
   gathered=[i for rank_batches in ranks for i in rank_batches[step]]
   assert len(gathered)%6==0
   for offset in range(0,len(gathered),6):assert gathered[offset:offset+6]==[gathered[offset]]*6
   occurrences.update(gathered)
  assert occurrences==Counter({i:6 for i in range(2000)})
  sampler_checks.append({'per_device_batch':batch,'eight_rank_microbatches':len(ranks[0]),'unique_sources':2000,'trajectories':12000,'last_microbatch_sources':len(gathered)//6,'equal_rank_lengths':True,'no_dropped_or_duplicated_source_groups':True})
 try:TailSafeRankBatchSampler(2001,6,8,0,48)
 except ValueError:pass
 else:raise AssertionError('Uneven tail must fail rather than drop or duplicate trajectories')
 matrix={}
 for training in ['oneclass','notoneclass']:
  run=t/('test_'+training);(run/'data').mkdir(parents=True)
  for variant in ['oneclass','notoneclass']:(run/f'data/test_{variant}_episodes.json').write_text('[]')
  (t/'calls').write_text('')
  e=dict(env,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',RUN_DIR=str(run),TRAIN_EPISODES='/train/'+training)
  r=subprocess.run(['bash',str(script),'test'],env=e,capture_output=True,text=True);assert r.returncode==0,r.stderr
  calls=[json.loads(line) for line in (t/'calls').read_text().splitlines()];assert len(calls)==2
  pairs=set()
  for args in calls:
   source=Path(args[args.index('--data')+1]).name
   variant=source[len('test_'):-len('_episodes.json')];cond=args[args.index('--condition')+1]
   out=Path(args[args.index('--output-dir')+1]);assert out==run/'test'/variant/cond
   assert args[args.index('--model')+1]==str(run/'merged_model')
   pairs.add((variant,cond))
  assert pairs==set(itertools.product(['oneclass','notoneclass'],['noisy_policy']))
  matrix[training]=len(calls)
 # A partial old preparation must fail before any evaluation begins.
 (t/'calls').write_text('');(run/'data/test_notoneclass_episodes.json').unlink()
 r=subprocess.run(['bash',str(script),'test'],env=e,capture_output=True,text=True);assert r.returncode!=0 and not (t/'calls').read_text()
 print(json.dumps({'valid_batch_layouts':valid,'eight_rank_sampler_checks':sampler_checks,'uneven_tail_rejected':True,'invalid_layouts_rejected':4,'evaluation_calls_per_training':matrix,'missing_second_test_rejected_before_evaluation':True},indent=2))
