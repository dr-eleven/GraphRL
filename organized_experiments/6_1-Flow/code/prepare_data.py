import argparse, hashlib, json
from pathlib import Path
import pandas as pd
import bootstrap
from graph_core.graph_state import load_episodes
from graph_core.label_utils import _label_key
from graph_task import build_question

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    out=Path(a.run_dir)/'data';out.mkdir(parents=True,exist_ok=False)
    import os
    cfg=json.loads(Path(os.environ.get('FLOW_SETTINGS',bootstrap.ROOT/'configs/settings.json')).read_text());report={}
    splits={'train':cfg['train_source'],**cfg['test_sources']}
    for name,file in splits.items():
        rows=load_episodes(file,a.limit if name=='train' else None)
        if name=='train' and (len(rows)%2 or not rows):raise ValueError('Training must contain a positive even number of sources')
        if name!='train' and len(rows)!=600:raise ValueError('Each final test set must contain 600 sources')
        for row in rows:
            if _label_key(row['ground_truth']) not in {_label_key(x) for x in row['label_space']}:raise ValueError('Ground truth outside label space')
        (out/f'{name}.json').write_text(json.dumps(rows,ensure_ascii=False)+'\n')
        if name=='train':
            converted=[dict(data_source='graphrl',prompt=[dict(role='user',content=build_question(row))],episode=json.dumps(row,ensure_ascii=False),question=build_question(row),result=str(row['ground_truth']),reward_model=dict(style='rule',ground_truth=str(row['ground_truth'])),extra_info=dict(index=row['source_index'])) for row in rows]
            pd.DataFrame(converted).to_parquet(out/'train.parquet',index=False)
            # Separate two-sample validation input; actual two test sets run after merge.
            pd.DataFrame(converted[:2]).to_parquet(out/'validation.parquet',index=False)
        report[name]=dict(count=len(rows),source=file,source_sha256=hashlib.sha256(Path(file).read_bytes()).hexdigest())
    report['updates']=report['train']['count']//2
    (out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
