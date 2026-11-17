"""Read run evidence without loading models or contacting inference services."""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


def inspect_run(run):
    state_path = run / 'pipeline_status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    pid = state.get('pipeline_pid')
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            # A defunct parent is not an active experiment.
            alive = Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
        except (ProcessLookupError, FileNotFoundError):
            pass
    totals = Counter()
    steps = Counter()
    path = run / 'train_rollouts.jsonl'
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A concurrent append may be incomplete.
                totals['completed_flows'] += 1
                steps[row['step']] += 1
                for role in ['planner', 'fixed']:
                    for call in row[role + '_calls']:
                        totals[role + '_calls'] += 1
                        usage = call.get('usage', {})
                        totals[role + '_input_tokens'] += usage.get('prompt_tokens', 0)
                        totals[role + '_generated_tokens'] += usage.get('completion_tokens', 0)
    log = run / 'train.log'
    # Only trainer metrics emitted after a successful update count as updates.
    updates = [int(v) for v in re.findall(r'training/global_step[:=]\s*(\d+)', log.read_text(errors='replace'))] if log.exists() else []
    audits = {p.name: json.loads(p.read_text()) for p in run.glob('*dtype_rank*.json')}
    evaluations = {str(p.relative_to(run)): json.loads(p.read_text()) for folder in ['test', 'smoke_test'] for p in (run / folder).glob('*/restored/summary.json')}
    return dict(run_dir=str(run), pipeline=state, pipeline_alive=alive,
                completed_updates=max(updates, default=0), recorded_flow_counts_by_step=dict(steps),
                token_counts_scope='completed flows written to train_rollouts.jsonl; excludes in-flight calls',
                **totals, dtype_audits=audits, evaluations=evaluations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_run(args.run_dir.resolve()), ensure_ascii=False, indent=2))
