"""7B 可能采用 batch2/累积2；验证同一有效训练组和真实 Trainer 的短尾缩放。"""
import json,tempfile,unittest
from pathlib import Path
from collections import Counter
import torch
from test_trl import build_trainer
from tail_batching import TailSafeRankBatchSampler

class SevenBBatchingTests(unittest.TestCase):
 def test_three_rank_effective_groups_match_3b(self):
  for count in (1,2,3,17,5495):
   profiles=[]
   for batch,accum in [(4,1),(2,2)]:
    ranks=[list(TailSafeRankBatchSampler(count,6,3,r,batch)) for r in range(3)]
    batches=[sum(items,[]) for items in zip(*ranks)]
    windows=[]
    for start in range(0,len(batches),accum):windows.append(sum(batches[start:start+accum],[]))
    profiles.append([Counter(window) for window in windows])
   self.assertEqual(profiles[0],profiles[1])

 def test_real_trainer_batch2_accum2_and_tail(self):
  torch.set_num_threads(1)
  with tempfile.TemporaryDirectory() as output:
   trainer=build_trainer(output,'llm',sources=5,batch=2,accum=2)
   trainer.train()
   self.assertEqual(trainer.state.global_step,3)
   self.assertEqual(trainer.accumulation_windows,[2,2,2,2,1])
   rows=[json.loads(line) for line in (Path(output)/'rollouts_rank0.jsonl').read_text().splitlines()]
   self.assertEqual(Counter(r['sample_id'] for r in rows),Counter({f'toy:{i}':2 for i in range(5)}))


class SevenBCheckpointingTests(unittest.TestCase):
 def test_generation_restores_explicit_checkpointing_options(self):
  from types import SimpleNamespace
  from unittest.mock import patch
  import train_grpo
  from train_grpo_7b import SevenBThinkWithGraphTrainer
  calls=[]
  trainer=object.__new__(SevenBThinkWithGraphTrainer)
  trainer.args=SimpleNamespace(gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':True})
  trainer.model=SimpleNamespace(gradient_checkpointing_enable=lambda **kw:calls.append(kw))
  with patch.object(train_grpo.ThinkWithGraphTrainer,'_generate_batch',return_value=['segment']):
   self.assertEqual(trainer._generate_batch([[1]],'llm'),['segment'])
  self.assertEqual(calls,[{'gradient_checkpointing_kwargs':{'use_reentrant':True}}])

 def test_restored_options_reach_actual_qwen_layers(self):
  import tempfile
  from train_grpo_7b import restore_gradient_checkpointing
  with tempfile.TemporaryDirectory() as output:
   trainer=build_trainer(output,'llm',sources=1,batch=2,accum=1)
   trainer.args.gradient_checkpointing=True
   trainer.args.gradient_checkpointing_kwargs={'use_reentrant':True}
   trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
   restore_gradient_checkpointing(trainer)
   layers=trainer.model.get_base_model().model.layers
   self.assertTrue(all(layer._gradient_checkpointing_func.keywords['use_reentrant'] for layer in layers))

if __name__=='__main__':unittest.main()
