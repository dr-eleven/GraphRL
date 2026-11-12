"""Validate the real 900 data and simulate the automatic gates without starting training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_quarter_data import prepare
import run_pipeline as pipeline


def main(source):
    expected=json.loads((ROOT/'configs/data_source.json').read_text())
    source=Path(source)
    if source.is_dir():source=source/expected['training_file']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected['training_sha256']
    episodes=json.loads(source.read_text());assert len(episodes)==900
    for s in episodes:
        nodes=set(map(int,s['noisy_graph']['nodes']))
        assert all(int(e['u']) in nodes and int(e['v']) in nodes for e in s['noisy_graph']['edges'])
        assert s['ground_truth'] in s['label_space']
    import pyarrow.parquet as pq
    parquet=pq.read_table(source.parent/'train.parquet',columns=['source_index','budget']).to_pylist()
    assert [r['source_index'] for r in parquet]==[r['source_index'] for r in episodes]
    assert [r['budget'] for r in parquet]==[r['attack_budget'] for r in episodes]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);manifest=prepare(source,root/'quarter',42)
        assert manifest['selected_count']==225 and manifest['selected_sha256']==expected['quarter_training_sha256']
        previous=root/'previous';evaluation=root/'evaluation';previous.mkdir();evaluation.mkdir()
        pipeline.PREVIOUS=previous
        (previous/'final_evaluation_pointer.json').write_text(json.dumps(dict(path=str(evaluation))))
        (evaluation/'status.json').write_text(json.dumps(dict(state='evaluating')))
        (evaluation/'manifest.json').write_text(json.dumps(dict(checkpoint_step=2200)))
        assert pipeline.prerequisite(source)==(False,'waiting_3_6_test')
        (evaluation/'status.json').write_text(json.dumps(dict(state='completed')))
        assert pipeline.prerequisite(source)==(False,'waiting_3_6_analysis')
        report=root/'report.md';report.write_text('CPU-only test report')
        (evaluation/'analysis_completed.json').write_text(json.dumps(dict(state='completed',checkpoint_step=2200,
            report=str(report),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())))
        assert pipeline.prerequisite(source)==(True,'ready')
        wrong=root/'wrong.json';wrong.write_bytes(source.read_bytes()+b' ')
        assert pipeline.prerequisite(wrong)==(False,'waiting_verified_900_sha256')
        report.write_text('modified')
        assert pipeline.prerequisite(source)==(False,'waiting_verified_3_6_report')
    print(json.dumps(dict(real_data=900,quarter_data=225,parquet_json_aligned=True,gates='PASS',training_launched=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);a=parser.parse_args();main(a.source)
