"""Whole-flow admission rule, based on exact sampled Planner token IDs."""
from collections import Counter

def inspect_trace(trace,limit):
    lengths=[len(t['prompt_ids'])+len(t['response_ids']) for t in trace]
    return dict(limit=limit,max_total_tokens=max(lengths,default=0),
                offending_turns=[i+1 for i,n in enumerate(lengths) if n>limit])

def trainable_groups(tasks,originals,min_flows=2):
    expected={item[1]['data_id'] for item in tasks}
    counts=Counter(row['data_id'] for row in originals.values())
    return len(expected)==2 and all(counts[group]>=min_flows for group in expected)
