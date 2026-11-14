"""Deterministic 32-row controller smoke; scripted gold submissions are NOT model accuracy."""
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from episode import GraphEnvironment, load_episodes, initial_messages
from graph_state import _edge_key
from project_config import DEFAULT_MODE
from test_binary_controller import tool


def main():
    root=Path(__file__).resolve().parents[4]
    source=root/'GraphRL/1000_oneclass_sorted/train_grpo_episodes_redesign.json'
    data=load_episodes(source)
    indices=[i*(len(data)-1)//31 for i in range(32)]
    records=[]
    for idx in indices:
        row=data[idx];env=GraphEnvironment();env.reset(row)
        # Fixed sampling by source position; no filtering using reward or corruption quality.
        env._consume(tool('unknown'),[1,2,3])
        protected={_edge_key(*x) for x in row.get('protected_edges',[])}
        edge=next((x for x in row['noisy_graph']['edges'] if x['u'] != x['v'] and _edge_key(x['u'],x['v']) not in protected),None)
        for name in ['del_node','del_edge','add_edge','rewrite']:
            actions=[]
            if edge and name in {'del_edge','add_edge'}:
                actions=[dict(op={'del_edge':'DEL_EDGE','add_edge':'ADD_EDGE'}[name],u=int(edge['u']),v=int(edge['v']))]
            raw=tool(name,actions);env._consume(raw,[1]*len(raw))
        env._consume(json.dumps(dict(analysis="Reason",prediction=row['ground_truth'])),[1,2,3])
        assert env._score([])==1
        assert env._errors[0]['code']=='UNKNOWN_TOOL'
        assert env._tool_successes==4
        records.append(dict(source_index=row['source_index'],task_type=row['task_type'],
                            dataset=row.get('dataset'),tool_successes=env._tool_successes))
    report=dict(mode=DEFAULT_MODE,count=32,source=str(source),source_count=len(data),selection=f'32 evenly spaced positions in unchanged {len(data)}-row source order',
                scripted_gold_submission=True,not_model_accuracy=True,records=records)
    (Path(__file__).resolve().parents[1]/'verification/data_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print('32 real-source controller cases passed; this is not model inference accuracy.')


if __name__=='__main__':main()
