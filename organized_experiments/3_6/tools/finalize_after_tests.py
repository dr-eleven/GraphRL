"""Wait for both complete policy evaluations, rebuild the report, then open the training gate."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/3_6_gpu1_z2_b12_a1_cap5120_resume320_20261109_122155'
PYTHON='/home/temptest/.venvs/graphrl-3_5_1-3_6/bin/python'


def main():
    pointer=RUN/'final_evaluation_pointer.json'
    folder=Path(json.loads(pointer.read_text())['path'])
    def tests_ready():
        try:
            return json.loads((folder/'status.json').read_text()).get('state')=='completed' and all(json.loads((folder/c/'summary.json').read_text()).get('count')==600 for c in ['restored','clean_policy'])
        except (FileNotFoundError,ValueError):return False
    while not tests_ready():
        (folder/'analysis_status.json').write_text(json.dumps(dict(state='waiting_policy_tests',time=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
        time.sleep(30)
    # Analyze only the trajectory prefix corresponding to the evaluated checkpoint.
    step=json.loads((folder/'manifest.json').read_text())['checkpoint_step']
    perf=[json.loads(l) for l in (RUN/'grpo/performance_steps.jsonl').open()]
    snapshot=next(p for p in perf if p['global_step']==step)
    (folder/'training_snapshot.json').write_text(json.dumps(snapshot,indent=2))
    env=os.environ.copy();env.update(TASK_REPORT_SNAPSHOT=str(folder/'training_snapshot.json'),CUDA_VISIBLE_DEVICES='')
    with (folder/'analysis.log').open('w') as log:
        subprocess.run([PYTHON,str(ROOT/'tools/analyze_3_6_report.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=ROOT/'reports/3_6.md'
    marker=dict(state='completed',checkpoint_step=step,report=str(report),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        time=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (folder/'analysis_completed.json').write_text(json.dumps(marker,indent=2))
    (folder/'status.json').write_text(json.dumps(dict(state='completed',count_per_condition=600),indent=2))
    print(json.dumps(marker),flush=True)


if __name__=='__main__':main()
