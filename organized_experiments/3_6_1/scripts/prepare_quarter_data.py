"""Sample one quarter of each original clean/budget1..5 training stratum."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random


def prepare(source,output,seed=42):
    source=Path(source);output=Path(output)
    if source.is_dir():
        choices=[source/'train_grpo_episodes_redesign.json',source/'train_episodes.json']
        found=[p for p in choices if p.is_file()]
        if len(found)!=1:raise ValueError('Expected exactly one train episode file: '+str(choices))
        source=found[0]
    if not source.is_file():raise FileNotFoundError('Required new 900 dataset is not available: '+str(source))
    raw=source.read_bytes();data=json.loads(raw);strata=defaultdict(list)
    assert isinstance(data,list) and len({s['source_index'] for s in data})==len(data)
    for i,s in enumerate(data):
        b=s['attack_budget']
        if type(b) is not int or b not in range(6):raise ValueError('Expected attack_budget integer 0..5')
        for k in ['clean_graph','noisy_graph','ground_truth','label_space','task_type','target']:
            if k not in s:raise ValueError('Missing episode field '+k)
        strata[b].append(i)
    if set(strata)!=set(range(6)):raise ValueError('Missing clean or budget stratum')
    rng=random.Random(seed);indices=set();counts={}
    for budget in range(6):
        count=len(strata[budget])//4
        if count<1:raise ValueError('Stratum too small to retain one quarter')
        selected=rng.sample(strata[budget],count);indices.update(selected)
        counts['clean' if budget==0 else 'budget'+str(budget)]=dict(original=len(strata[budget]),selected=count)
    chosen=[s for i,s in enumerate(data) if i in indices]
    manifest=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),seed=seed,
        fraction='1/4',rounding='floor separately per original attack_budget stratum',
        actual_clean_graph_does_not_override_original_budget=True,counts=counts,
        original_count=len(data),selected_count=len(chosen),source_indices=[s['source_index'] for s in chosen],
        order='retain original degree-sorted source order after random within-stratum selection',
        tasks=dict(Counter(s['task_type'] for s in chosen)))
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    text=json.dumps(chosen,ensure_ascii=False,indent=2)+'\n'
    (output/'train_quarter.json').write_text(text)
    manifest['selected_sha256']=hashlib.sha256(text.encode()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.seed),ensure_ascii=False,indent=2))
