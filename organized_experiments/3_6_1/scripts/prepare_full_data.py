"""Preserve every source episode and its degree-sorted order, with byte-exact input."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def prepare(source,output):
    source=Path(source);output=Path(output)
    if source.is_dir():
        candidates=[p for p in [source/'train_grpo_episodes_redesign.json',source/'train_episodes.json'] if p.is_file()]
        if len(candidates)!=1:raise ValueError('Expected one unambiguous training episode file')
        source=candidates[0]
    raw=source.read_bytes();data=json.loads(raw)
    if not isinstance(data,list) or not data:raise ValueError('Empty or invalid training data')
    if len({s['source_index'] for s in data})!=len(data):raise ValueError('Duplicate source_index')
    counts=Counter()
    for s in data:
        if type(s['attack_budget']) is not int or s['attack_budget'] not in range(6):raise ValueError('Expected attack_budget 0..5')
        for k in ['clean_graph','noisy_graph','ground_truth','label_space','task_type','target']:
            if k not in s:raise ValueError('Missing episode field '+k)
        counts[s['attack_budget']]+=1
    if set(counts)!=set(range(6)):raise ValueError('Missing clean or budget stratum')
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    # Exact bytes keep both the full sample coverage and the original ordering.
    (output/'train_900.json').write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    manifest=dict(source=str(source.resolve()),source_sha256=digest,selected_sha256=digest,
        original_count=len(data),selected_count=len(data),fraction=1.0,sampling=False,
        counts={('clean' if b==0 else 'budget'+str(b)):dict(original=counts[b],selected=counts[b]) for b in range(6)},
        source_indices=[s['source_index'] for s in data],order='all episodes in original degree-sorted order',
        tasks=dict(Counter(s['task_type'] for s in data)))
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();m=prepare(a.source,a.output);print(json.dumps({k:v for k,v in m.items() if k!='source_indices'},ensure_ascii=False,indent=2))
