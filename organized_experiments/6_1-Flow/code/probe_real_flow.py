"""Real base-7B Solver smoke; not a trained-model accuracy measurement."""
import json,time,requests
from pathlib import Path
import bootstrap
from workflow import run_workflow
from engines import Engine
cfg=json.loads((bootstrap.ROOT/'configs/settings.json').read_text())
http=requests.Session();http.trust_env=False
for _ in range(180):
    try:
        if http.get(cfg['fixed_url']+'/models',timeout=2).status_code==200:break
    except requests.RequestException:pass
    time.sleep(2)
else:raise RuntimeError('Fixed model did not become ready')
sample=json.loads(Path(cfg['train_source']).read_text())[0]
trace=[];planner=Engine(cfg['fixed_url'],cfg['fixed_served_model'],.7,trace=trace,role='planner')
fixed=Engine(cfg['fixed_url'],cfg['fixed_served_model'],0.,role='fixed')
result=run_workflow(sample,planner,fixed,cfg['tool_steps'],cfg['agent_timeout'])
result['verification_only']=True;result['planner_uses_original_base_model']=True
(bootstrap.ROOT/'verification/real_7b_flow.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['source_index','prediction','reward','capacity_failure','seconds']},ensure_ascii=False))
print('Planner calls',len(result['planner_calls']),'fixed calls',len(result['fixed_calls']))
assert trace
