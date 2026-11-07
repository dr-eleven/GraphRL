"""Verify the copied A-7B profile and full-source / accumulation tail layout."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def defaults(class_name):
    tree=ast.parse((ROOT/'code/train_grpo.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==class_name)
    result={}
    for node in cls.body:
        if isinstance(node,ast.AnnAssign) and node.value is not None:
            try: result[node.target.id]=ast.literal_eval(node.value)
            except (ValueError,TypeError): pass
    return result


class ASevenBProfileTests(unittest.TestCase):
    def test_launcher_deepspeed_path_parses_as_string(self):
        import sys
        sys.path.insert(0,str(ROOT/'code'))
        from transformers import HfArgumentParser
        from train_grpo import ModelArguments,GraphRLDataArguments,GraphRLTrainingArguments
        parser=HfArgumentParser((ModelArguments,GraphRLDataArguments,GraphRLTrainingArguments))
        config=str(ROOT/'configs/zero3_7b.json')
        parsed=parser.parse_args(['--deepspeed',config,'--output_dir','/tmp/graphrl-parse-only',
                                  '--gradient_checkpointing_kwargs','{"use_reentrant":true}'])
        self.assertEqual(parsed.deepspeed,config)
        self.assertEqual(parsed.gradient_checkpointing_kwargs,'{"use_reentrant":true}')

    def test_actual_completed_profile_matches_training_defaults(self):
        reference=json.loads((ROOT/'configs/a7b_reference.json').read_text())
        expected=reference['training_arguments']
        actual=defaults('GraphRLTrainingArguments')
        keys=['per_device_train_batch_size','per_device_eval_batch_size','gradient_accumulation_steps',
              'num_train_epochs','learning_rate','warmup_steps','lr_scheduler_type','optim','weight_decay',
              'max_grad_norm','bf16','tf32','gradient_checkpointing','use_liger_kernel','disable_dropout',
              'num_generations','max_completion_length','steps_per_generation','num_iterations',
              'temperature','top_p','top_k','beta','loss_type','shuffle_dataset','save_steps','save_total_limit']
        for key in keys:
            self.assertEqual(actual[key],expected[key],key)
        model=defaults('ModelArguments')
        for key in ['lora_rank','lora_alpha','lora_dropout','disable_flash_attn2']:
            self.assertEqual(model[key],expected[key],key)
        data=defaults('GraphRLDataArguments')
        self.assertEqual(data['action_max_new_tokens'],512)
        self.assertEqual(data['prediction_max_new_tokens'],512)
        self.assertEqual(data['logprob_micro_batch_size'],1)
        self.assertEqual(data['max_tool_turns'],4)
        self.assertEqual(actual['max_tool_calling_iterations'],4)
        config=json.loads((ROOT/'configs/zero3_7b.json').read_text())
        self.assertEqual(config['zero_optimization']['stage'],3)
        self.assertEqual(reference['selected_profile']['profile'],'zero3_reentrant_batch2_accum2')
        text=(ROOT/'scripts/run_direct_batch4_1epoch.sh').read_text()
        for fragment in ['GRAD_ACCUM:-2','configs/zero3_7b.json','--loss_type grpo',
                         '--use_liger_kernel True','--disable_flash_attn2 False',
                         '--gradient_checkpointing_kwargs \'{"use_reentrant":true}\'']:
            self.assertIn(fragment,text)

    def test_full_generation_groups_and_final_update_tail(self):
        spec=importlib.util.spec_from_file_location('tail_batching',ROOT/'code/tail_batching.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        ranks=[list(module.TailSafeRankBatchSampler(5495,6,3,rank,2)) for rank in range(3)]
        self.assertEqual([len(rows) for rows in ranks],[5495]*3)
        for index in range(5495):
            combined=[source for rank in ranks for source in rank[index]]
            self.assertEqual(combined,[index]*6)
        self.assertEqual(math.ceil(len(ranks[0])/2),2748)
        self.assertEqual(len(ranks[0])%2,1)
        self.assertEqual(sum(len(rank[-1]) for rank in ranks),6)


if __name__=='__main__': unittest.main()
