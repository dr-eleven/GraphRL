import os,sys,json,subprocess,tempfile,itertools
from pathlib import Path
# This validates orchestration only: the recorder never loads or trains a model.
script=Path(__file__).resolve().parents[1]/'scripts/run_portable.sh'
with tempfile.TemporaryDirectory() as td:
 t=Path(td); recorder=t/'record';recorder.write_text('#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ["CHECK_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n');recorder.chmod(0o755)
 env=dict(os.environ,TRAIN_MODEL=str(t/'base'),PYTHON_BIN=str(recorder),CHECK_LOG=str(t/'calls'))
 for k in ['TEST_EPISODES','TRAIN_BATCH_SIZE','GRAD_ACCUM','NUM_GENERATIONS']:env.pop(k,None)
 valid=[]
 for n in [1,2,3,6]:
  for mb in [6,12]:
   run=t/f'train_{n}_{mb}';e=dict(env,CUDA_VISIBLE_DEVICES=','.join(map(str,range(n))),TRAIN_BATCH_SIZE=str(mb//n),RUN_DIR=str(run))
   r=subprocess.run(['bash',str(script),'train'],env=e,capture_output=True,text=True);assert r.returncode==0,r.stderr
   args=json.loads((t/'calls').read_text().splitlines()[-1]);batch=int(args[args.index('--per_device_train_batch_size')+1]);acc=int(args[args.index('--gradient_accumulation_steps')+1]);assert n*batch*acc==12
   valid.append({'gpus':n,'batch':batch,'accum':acc})
 for b,acc in [('2','1'),('1','4'),('4','2'),('0','2')]:
  e=dict(env,CUDA_VISIBLE_DEVICES='0,1,2',TRAIN_BATCH_SIZE=b,GRAD_ACCUM=acc,RUN_DIR=str(t/f'invalid_{b}_{acc}'))
  r=subprocess.run(['bash',str(script),'train'],env=e,capture_output=True,text=True);assert r.returncode!=0
 matrix={}
 for training in ['oneclass','notoneclass']:
  run=t/('test_'+training);(run/'data').mkdir(parents=True)
  for variant in ['oneclass','notoneclass']:(run/f'data/test_{variant}_episodes.json').write_text('[]')
  (t/'calls').write_text('')
  e=dict(env,CUDA_VISIBLE_DEVICES='0,1,2',RUN_DIR=str(run),TRAIN_EPISODES='/train/'+training)
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
 print(json.dumps({'valid_batch_layouts':valid,'invalid_layouts_rejected':4,'evaluation_calls_per_training':matrix,'missing_second_test_rejected_before_evaluation':True},indent=2))
