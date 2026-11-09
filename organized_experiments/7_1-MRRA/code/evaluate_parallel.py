"""Independent one-GPU evaluation workers; resume and merge in original source order."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

from config import load_config, resolve, write_json
from prepare_data import load_episodes


def read_records(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # An interrupted writer may leave only its last line incomplete.
            if index == len(lines)-1 and not line.endswith(b'\n'):
                continue
            raise
    return rows


def collect_records(paths, expected_ids, strategy, condition):
    expected = set(expected_ids)
    records = {}
    for path in paths:
        for row in read_records(path):
            sid = row['sample_id']
            if sid not in expected or row['strategy'] != strategy or row['condition'] != condition:
                raise ValueError(f'Incompatible prediction in {path}: {sid}')
            if sid in records and records[sid] != row:
                raise ValueError(f'Conflicting duplicate prediction: {sid}')
            records[sid] = row
    return records


def split_pending(raw_samples, ids, completed, workers):
    pending = [raw for raw, sid in zip(raw_samples, ids) if sid not in completed]
    return [pending[i::workers] for i in range(workers)]


def evaluate(args):
    cfg = load_config(args.config)
    episodes = resolve(args.episodes or cfg['data']['test'])
    samples = load_episodes(episodes, limit=args.limit)
    raw = json.loads(episodes.read_text())[:len(samples)]
    ids = [s['sample_id'] for s in samples]
    strategy = args.strategy or cfg['strategy']
    gpu_ids = args.gpus.split(',')
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('GPU IDs must be nonempty and distinct')
    output = resolve(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError('Existing output requires --resume')
    output.mkdir(parents=True, exist_ok=True)
    plan = {'config':cfg, 'model':str(resolve(args.model or cfg['model']['path'])),
            'adapter':str(resolve(args.adapter)) if args.adapter else None,
            'strategy':strategy, 'condition':args.condition, 'source_ids':ids,
            'semantic':bool(args.semantic or cfg['evaluation']['use_semantic'])}
    previous_plan = output/'parallel_plan.json'
    if previous_plan.exists():
        if json.loads(previous_plan.read_text()) != plan:
            raise ValueError('Resume plan differs from existing parallel evaluation')
    elif (output/'evaluation_config.json').exists():
        old = json.loads((output/'evaluation_config.json').read_text())
        if (old['config'] != cfg or old['strategy'] != strategy or old['condition'] != args.condition
                or str(resolve(old['policy_model'])) != plan['model'] or old.get('adapter') != plan['adapter']
                or bool(old['arguments'].get('semantic') or cfg['evaluation']['use_semantic']) != plan['semantic']):
            raise ValueError('Existing single-GPU results do not match the requested evaluation')
    write_json(previous_plan, plan)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if (output/'predictions.jsonl').exists():
        shutil.copy2(output/'predictions.jsonl', output/f'predictions.before_parallel_{stamp}.jsonl')
    paths = [output/'predictions.jsonl'] + sorted(output.glob('parallel_workers/attempt_*/gpu_*/predictions.jsonl'))
    done = collect_records(paths, ids, strategy, args.condition)
    chunks = split_pending(raw, ids, done, len(gpu_ids))
    attempt = output/'parallel_workers'/f'attempt_{stamp}'
    attempt.mkdir(parents=True)
    jobs = []
    print(f'Preserved {len(done)}/{len(ids)} records; pending {len(ids)-len(done)} on GPUs {gpu_ids}', flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for gpu, chunk in zip(gpu_ids, chunks):
            if not chunk:
                continue
            input_path = attempt/f'episodes_gpu_{gpu}.json'
            write_json(input_path, chunk)
            worker_output = attempt/f'gpu_{gpu}'
            command = [sys.executable, str(Path(__file__).with_name('evaluate_grpo_600.py')),
                       '--config',str(Path(args.config).resolve()), '--episodes',str(input_path),
                       '--output',str(worker_output),'--strategy',strategy,'--condition',args.condition,'--device','cuda:0']
            if args.model:
                command += ['--model',str(resolve(args.model))]
            if args.adapter:
                command += ['--adapter',str(resolve(args.adapter))]
            if args.semantic:
                command += ['--semantic']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
            log = (attempt/f'gpu_{gpu}.log').open('w')
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            jobs.append((gpu, process, worker_output, len(chunk)))
        last_count = -1
        while True:
            worker_progress = []
            for gpu, process, folder, size in jobs:
                code = process.poll()
                count = len(read_records(folder/'predictions.jsonl'))
                worker_progress.append({'gpu':gpu,'pid':process.pid,'completed':count,'assigned':size,'exit_code':code})
                if code is not None and code != 0:
                    raise RuntimeError(f'GPU {gpu} worker failed ({code}); see {attempt}/gpu_{gpu}.log')
            count = len(done)+sum(w['completed'] for w in worker_progress)
            write_json(output/'parallel_status.json', {'stage':'evaluating','completed':count,'total':len(ids),
                       'preserved':len(done),'workers':worker_progress,'updated_at':datetime.now().isoformat()})
            if count != last_count:
                print(f'{count}/{len(ids)} total; '+', '.join(f"GPU {w['gpu']}: {w['completed']}/{w['assigned']}" for w in worker_progress),flush=True)
                last_count = count
            if all(process.poll() is not None for _,process,_,_ in jobs):
                break
            time.sleep(5)
        final_paths = paths + [folder/'predictions.jsonl' for _,_,folder,_ in jobs]
        merged = collect_records(final_paths, ids, strategy, args.condition)
        if set(merged) != set(ids):
            raise RuntimeError(f'Incomplete coverage: {len(merged)}/{len(ids)}')
        ordered = [merged[sid] for sid in ids]
        temporary = output/'predictions.merged.tmp'
        with temporary.open('w',encoding='utf-8') as handle:
            for row in ordered:
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        temporary.replace(output/'predictions.jsonl')
        from evaluate_grpo_600 import summarize
        write_json(output/'summary.json',summarize(ordered))
        write_json(output/'parallel_status.json',{'stage':'complete','completed':len(ordered),'total':len(ids),
                   'preserved':len(done),'workers':worker_progress,'updated_at':datetime.now().isoformat()})
        print(f'Complete: {len(ordered)} unique predictions, original source order, global summary saved.',flush=True)
    except BaseException:
        for _,process,_,_ in jobs:
            if process.poll() is None:
                process.terminate()
        for _,process,_,_ in jobs:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--episodes')
    parser.add_argument('--model')
    parser.add_argument('--adapter')
    parser.add_argument('--strategy',choices=['llm','executor','direct','ssr'])
    parser.add_argument('--condition',choices=['noisy','clean'],default='noisy')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--semantic',action='store_true')
    parser.add_argument('--gpus',default='0,1,2')
    parser.add_argument('--resume',action='store_true')
    evaluate(parser.parse_args())
