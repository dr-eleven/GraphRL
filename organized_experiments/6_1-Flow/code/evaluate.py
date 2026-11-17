import argparse,json,os
from pathlib import Path
from collections import Counter
import bootstrap
from engines import Engine
from workflow import run_workflow
from graph_core.graph_state import load_episodes

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--planner-url',default='http://127.0.0.1:8622/v1');p.add_argument('--planner-model',default='graphrl-planner');p.add_argument('--limit',type=int);a=p.parse_args()
    cfg=json.loads(Path(os.environ.get('FLOW_SETTINGS',bootstrap.ROOT/'configs/settings.json')).read_text());rows=load_episodes(a.data,a.limit);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);results=[]
    pred=out/'predictions.jsonl'
    if pred.exists():raise FileExistsError('Refusing to overwrite existing predictions: '+str(pred))
    with pred.open('w') as stream:
        for sample in rows:
            planner=Engine(a.planner_url,a.planner_model,0.,max_tokens=cfg.get('max_tokens',8192),role='planner')
            fixed=Engine(cfg['fixed_url'],cfg['fixed_served_model'],0.,max_tokens=cfg.get('max_tokens',8192),role='fixed')
            try:result=run_workflow(sample,planner,fixed,cfg['tool_steps'],cfg['agent_timeout'])
            finally:planner.http.close();fixed.http.close()
            results.append(result)
            stream.write(json.dumps(result,ensure_ascii=True)+'\n');stream.flush()
    stats=dict(count=len(results),correct=sum(r['reward'] for r in results),accuracy=sum(r['reward'] for r in results)/len(results),valid_final=sum(r['prediction'] is not None for r in results),capacity_failures=sum(r['capacity_failure'] is not None for r in results),task_counts=dict(Counter(r['task_type'] for r in results)))
    stats['tool_attempts']=sum(len(r['actions']) for r in results)
    stats['tool_errors']=sum(not a['result']['ok'] for r in results for a in r['actions'])
    stats['graph_exact']=sum(r['graph_exact'] for r in results)
    stats['dirty_graph_exact']=sum(r['graph_exact'] and r['input_dirty'] for r in results)
    stats['dirty_inputs']=sum(r['input_dirty'] for r in results)
    stats['per_task']={task:dict(count=len(a),correct=sum(r['reward'] for r in a),accuracy=sum(r['reward'] for r in a)/len(a)) for task in stats['task_counts'] for a in [[r for r in results if r['task_type']==task]]}
    (out/'summary.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
