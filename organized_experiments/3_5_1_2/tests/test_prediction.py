import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from episode import build_prediction_prompt, parse_prediction_completion
from reward import score_episode


def sample(task='NC'):
    if task=='LP':
        return dict(source_index=2,task_type='LP',target='query pair node0, node1',target_node=None,
            protected_query_nodes=[0,1],label_space=['Yes, they have citation relationships','No, they do not have citation relationships'],
            ground_truth='Yes, they have citation relationships',
            clean_graph={'nodes':{'0':'A paper on sensors.','1':'A related paper.'},'edges':[{'u':0,'v':1}]},
            noisy_graph={'nodes':{'0':'A paper on sensors.','1':'A related paper.'},'edges':[{'u':0,'v':1},{'u':1,'v':0}]})
    return dict(source_index=1, task_type='NC', target='node0', target_node=0,
        label_space=['library', 'shop'], ground_truth='library',
        clean_graph={'nodes': {'0': 'HIDDEN_CLEAN_SENTINEL'}, 'edges': []},
        noisy_graph={'nodes': {'0': 'Books and a reading room.'}, 'edges': []},
        prompt='OLD_PROMPT_SENTINEL', injected_nodes=['HIDDEN_ATTACK_SENTINEL'])


class PredictionTests(unittest.TestCase):
    def test_reward_table(self):
        for label, expected in [('library', 2), ('shop', 1.1), ('other', 1)]:
            with self.subTest(label=label):
                score, record = score_episode(sample(), json.dumps(dict(analysis='Evidence.', prediction=label)))
                self.assertEqual(score, expected)
                self.assertEqual(record['metrics']['structure_reward'], 1)
                self.assertNotIn('behavior_reward', record['metrics'])

    def test_invalid_formats(self):
        cases = ['not json', '{"prediction":"library","analysis":"yes"}',
            '{"analysis":"","prediction":"library"}',
            '{"analysis":"yes","prediction":""}',
            '{"analysis":"yes","prediction":1}',
            '{"analysis":"yes","prediction":"library","actions":[]}',
            '{"analysis":"yes","analysis":"again","prediction":"library"}',
            '```json\n{"analysis":"yes","prediction":"library"}\n```']
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(score_episode(sample(), text)[0], 0)

    def test_truncation_overrides_complete_json(self):
        text = '{"analysis":"yes","prediction":"library"}'
        self.assertEqual(score_episode(sample(), text, terminated=False)[0], 0)

    def test_messages_and_no_extra_turns(self):
        good = {'role':'assistant', 'content':'{"analysis":"yes","prediction":"library"}'}
        self.assertEqual(score_episode(sample(), [good])[0], 2)
        self.assertEqual(score_episode(sample(), [good,good])[0], 0)
        self.assertEqual(score_episode(sample(), [dict(good,tool_calls=[{}])])[0], 0)

    def test_prompt_public_input(self):
        row = sample()
        row['ground_truth'] = 'HIDDEN_GOLD_SENTINEL'
        prompt = build_prediction_prompt(row)
        for forbidden in ['HIDDEN_', 'OLD_PROMPT', 'tool_call', 'del_node', 'actions', 'restore', 'protected']:
            self.assertNotIn(forbidden, prompt)
        for expected in ['Books and a reading room.', 'library', 'shop', 'node0', 'NC', 'CONSTRAINTS', 'OUTPUT']:
            self.assertIn(expected, prompt)
        row['task_type'], row['target'] = 'LC', 'query pair node0, node1'
        row['protected_query_nodes']=[0,1]
        row['noisy_graph']['nodes']['1']='Second query entity.'
        self.assertIn('query pair node0, node1', build_prediction_prompt(row))

    def test_cpu_native_lp_grpo_lora_update(self):
        import torch
        from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        from peft import LoraConfig
        from trl import GRPOTrainer
        from train_grpo import ThinkWithGraphTrainer, GraphRLTrainingArguments, GraphRLDataArguments, GraphRLDataset
        torch.set_num_threads(2)
        original = AutoTokenizer.from_pretrained(os.environ.get('TRAIN_MODEL', '/opt/data/private/Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct'), local_files_only=True)
        vocab = {c:i for i,c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend = Tokenizer(models.BPE(vocab=vocab,merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
        backend.decoder = decoders.ByteLevel()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, chat_template=original.chat_template,
            eos_token='<|im_end|>', pad_token='<|endoftext|>', additional_special_tokens=['<|im_start|>'])
        tok.padding_side = 'left'
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=4096, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id))
        model.config._name_or_path = 'tiny-direct-prediction-test'
        row=sample('LP')
        dataset = GraphRLDataset([row]).to_hf_dataset()
        self.assertEqual(GraphRLDataset([row])[0]['episode']['noisy_graph']['edges'],[])
        calls = []
        def scripted_leaf(trainer, prompt_ids, *args, **kwargs):
            calls.append(len(prompt_ids))
            texts = [json.dumps(dict(analysis='The neighborhood supports this link decision.', prediction=label)) for label in row['label_space']]
            return [tok.encode(text+tok.eos_token,add_special_tokens=False) for text in texts], None, None
        with tempfile.TemporaryDirectory() as tmp:
            args = GraphRLTrainingArguments(output_dir=tmp, use_cpu=True, bf16=False, fp16=False,
                tf32=False, deepspeed=None, use_liger_kernel=False, warmup_steps=0,
                per_device_train_batch_size=2, gradient_accumulation_steps=1, num_generations=2, num_generations_eval=1,
                max_steps=1, learning_rate=1e-3, report_to='none', save_strategy='no',
                disable_tqdm=True, beta=.01, gradient_checkpointing=False)
            trainer = ThinkWithGraphTrainer(model=model,args=args,train_dataset=dataset,processing_class=tok,
                peft_config=LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'),
                data_args=GraphRLDataArguments())
            self.assertFalse(trainer.tools)
            self.assertIsNone(trainer.environment_factories)
            before = {n:p.detach().clone() for n,p in trainer.model.named_parameters() if p.requires_grad}
            with patch.object(GRPOTrainer, '_generate_single_turn', scripted_leaf):
                trainer.train()
            self.assertEqual(trainer.state.global_step, 1)
            self.assertEqual(calls, [2])
            self.assertEqual([r['reward'] for r in trainer.last_rollout_records], [2,1.1])
            self.assertTrue(any(not torch.equal(before[n],p.detach()) for n,p in trainer.model.named_parameters() if n in before))
            # Real generation exercises the unpatched single-turn path and EOS/truncation tracking.
            trainer.model.eval()
            trainer.generation_config.max_new_tokens = 2
            trainer._generate_and_score_completions([GraphRLDataset([row])[0]])
            self.assertEqual(len(trainer.last_rollout_records), 1)
            self.assertEqual(trainer.last_rollout_records[0]['reward'], 0)
            self.assertLessEqual(len(trainer.last_rollout_records[0]['completion_ids']), 2)


if __name__ == '__main__':
    unittest.main()
