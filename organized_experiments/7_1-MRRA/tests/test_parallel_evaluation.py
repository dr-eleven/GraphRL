"""Coverage/resume guards for independent evaluation shards (no GPU required)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from evaluate_parallel import collect_records, read_records, split_pending


class ParallelEvaluationTests(unittest.TestCase):
    def test_resume_partitions_only_missing_sources_without_overlap(self):
        ids=list('abcdefg')
        chunks=split_pending([{'id':s} for s in ids],ids,{'a':{},'b':{}},3)
        self.assertEqual(chunks,[[{'id':'c'},{'id':'f'}],[{'id':'d'},{'id':'g'}],[{'id':'e'}]])

    def test_merge_preserves_results_and_rejects_conflicting_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.jsonl'; b=Path(tmp)/'b.jsonl'
            row={'sample_id':'x','strategy':'llm','condition':'noisy','prediction':'A'}
            a.write_text(json.dumps(row)+'\n');b.write_text(json.dumps(row)+'\n')
            self.assertEqual(collect_records([a,b],['x'],'llm','noisy'),{'x':row})
            b.write_text(json.dumps(dict(row,prediction='B'))+'\n')
            with self.assertRaises(ValueError):collect_records([a,b],['x'],'llm','noisy')
            with self.assertRaises(ValueError):collect_records([a],['x'],'llm','clean')
            with self.assertRaises(ValueError):collect_records([a],['other'],'llm','noisy')

    def test_only_interrupted_trailing_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'predictions.jsonl'
            p.write_text('{"sample_id":"x"}\n{"sample_id":')
            self.assertEqual(read_records(p),[{'sample_id':'x'}])
            p.write_text('{"sample_id":\n')
            with self.assertRaises(json.JSONDecodeError):read_records(p)


if __name__=='__main__':unittest.main()
