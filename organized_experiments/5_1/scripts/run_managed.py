"""Run prepare, train, merge and the two test sets with persisted status and a GPU lock."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROOT = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    run = args.run or ROOT / 'runs' / ('5_1_1000_oneclass_' + stamp)
    if run.exists():
        raise FileExistsError(run)
    logs = ROOT / 'setup_logs'
    logs.mkdir(exist_ok=True)
    guard = (ROOT / '.managed.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shared = GRAPH_ROOT / 'setup_logs'
    shared.mkdir(exist_ok=True)
    gpu_lock = (shared / f'gpu{args.gpu}_experiment.lock').open('a')

    def status(state, **more):
        obj = dict(state=state, time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   run=str(run), gpu=args.gpu, train_count=1000, test_sets=['600_test_data_oneclass', '600_test_data_notoneclass'], **more)
        path = ROOT / 'latest_run.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(obj, indent=2) + '\n'); temporary.replace(path)
        if run.is_dir():
            (run / 'status.json').write_text(json.dumps(obj, indent=2) + '\n')
        print(json.dumps(obj), flush=True)

    while True:
        locked = False
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            used = int(subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True))
            if used <= 1024:
                break
        except BlockingIOError:
            pass
        if locked:
            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
        status('waiting_gpu_release')
        if not args.wait:
            raise RuntimeError('Selected GPU is occupied')
        time.sleep(30)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), RUN_DIR=str(run), PYTHONUNBUFFERED='1')
    try:
        for stage in ['prepare', 'train', 'merge', 'test']:
            status(stage)
            with (logs / (run.name + '_' + stage + '.log')).open('w') as stream:
                subprocess.run(['bash', str(ROOT / 'scripts/run_jhh_1000.sh'), stage], env=env,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            if stage == 'prepare':
                report = json.loads((run / 'data/preflight.json').read_text())
                assert report['train_count'] == 1000 and report['fraction'] == 1
                assert report['test_counts'] == {'600_test_data_oneclass':600, '600_test_data_notoneclass':600}
                (run / 'deployment_receipt.json').write_text((ROOT / 'deployment_receipt.json').read_text())
        completed = json.loads((run / 'grpo/completed.json').read_text())
        assert completed['epoch'] >= 1 and completed['train_count'] == 1000
        summaries = {}
        for name in ['600_test_data_oneclass', '600_test_data_notoneclass']:
            folder = run / 'test' / name / 'restored'
            summary = json.loads((folder / 'summary.json').read_text())
            rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
            assert summary['count'] == len(rows) == len({r['source_index'] for r in rows}) == 600
            summaries[name] = summary
        (run / 'test_results.json').write_text(json.dumps(summaries, indent=2) + '\n')
        status('completed', summaries=str(run / 'test_results.json'))
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
