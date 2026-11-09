"""MRRA on Graph-R1/TRL GRPOTrainer: real batched rollout, joint token mask, Trainer updates."""
from __future__ import annotations

import copy
import json
import math
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
from accelerate.data_loader import DataLoaderShard
from accelerate.utils import gather_object
from peft import LoraConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainerCallback, set_seed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import pad

from graph_r1_grpo_trainer import GraphR1GRPOTrainer
from graph_r1_params import GRPOArguments, ModelArguments
from config import ROOT, load_config, resolve, write_json, digest
from prepare_data import load_episodes
from episode import first_messages
from model_backend import Segment, StopAfterJson, StopAfterStage, generation_limits, segment_complete
from rollout import decide, finish_record
from trajectory_tokens import render_messages, feedback_tokens, pack_segments
from executor_protocol import execute_stage, observation_after, aggregate_stages, STAGE_TAGS, PROTOCOL_VERSION
from episode import public_observation
from tail_batching import TailSafeRankBatchSampler


@dataclass
class GraphRLTrainingArguments(GRPOArguments):
    # 与 degree-sorted 相同：每个 loader 微批生成一次，不能按累积步缓存并跳过新输入。
    steps_per_generation: int | None = 1


@dataclass
class MRRADataArguments:
    mrra_config: str = field(default=str(ROOT / 'configs/mrra_llm.json'))
    data_path: str | None = None
    logprob_micro_batch_size: int = 1
    validation_only: bool = False
    memory_stress_tokens: int = 0
    limit_data_sources: int | None = None
    continuation_checkpoint: str | None = None


def append_jsonl(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')


class GraphRLDataset(Dataset):
    def __init__(self, samples, tokenizer, cfg):
        self.samples = samples
        self.messages = [first_messages(s, cfg['strategy'], cfg['ablation']) for s in samples]
        self.prompts = [render_messages(tokenizer, messages) for messages in self.messages]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {'prompt': self.prompts[index], 'episode': self.samples[index], 'task_messages': self.messages[index]}


class EpochCheckpointSnapshots(TrainerCallback):
    # 与原实验相同，epoch 结束强制保存并固定完整 checkpoint；保留 optimizer/scheduler/RNG。
    def __init__(self, expected_ids, generations):
        self.pending_epoch = None
        self.expected_ids, self.generations = set(expected_ids), generations

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        if abs(state.epoch - epoch) < 1e-6:
            self.pending_epoch = epoch
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if self.pending_epoch is None:
            return control
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if state.is_world_process_zero:
            source = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            destination = Path(args.output_dir) / 'epochs' / f'epoch_{self.pending_epoch}'
            shutil.copytree(source, destination)
            write_json(destination / 'epoch_snapshot.json', {'epoch': self.pending_epoch, 'global_step': state.global_step})
            from collections import Counter
            counts = Counter()
            for path in Path(args.output_dir).glob('rollouts_rank*.jsonl'):
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row['epoch'] == self.pending_epoch:
                        counts[row['sample_id']] += 1
            covered = set(counts) == self.expected_ids and set(counts.values()) == {self.generations}
            write_json(Path(args.output_dir) / f'coverage_epoch_{self.pending_epoch}.json',
                       {'covered': covered, 'unique_sources': len(counts), 'candidate_count': sum(counts.values()),
                        'candidate_count_distribution': dict(Counter(counts.values()))})
            if not covered:
                raise RuntimeError('Incomplete source/candidate coverage at epoch boundary')
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.pending_epoch = None
        return control


class ThinkWithGraphTrainer(GraphR1GRPOTrainer):
    """TRL/Graph-R1 管理优势、损失、累积、优化和保存；本类适配 MRRA 轨迹与环境 mask。"""
    def __init__(self, *args, mrra_config, data_args=None, **kwargs):
        self.mrra_config = mrra_config
        self.data_args = data_args or MRRADataArguments()
        self._loss_attention_mask = None
        model = kwargs.get('model', args[0] if args else None)
        # 同 degree-sorted 的 Transformers 兼容处理。
        if model is not None and not hasattr(model, 'warnings_issued'):
            model.warnings_issued = {}
        super().__init__(*args, **kwargs)
        if self.args.steps_per_generation != 1 or self.num_iterations != 1:
            raise ValueError('MRRA tail-safe loader requires steps_per_generation=num_iterations=1')
        if self.loss_type != 'grpo':
            raise ValueError('Joint trajectory uses GRPO per-trajectory generated-token normalization')
        if self.use_vllm or self.mask_truncated_completions:
            raise ValueError('This implementation uses HF rollout and scores actual generated tokens including truncations')
        self._micro_step = 0

    def _set_signature_columns_if_needed(self):
        self._signature_columns = ['prompt', 'episode', 'task_messages']

    # 完整候选组/尾批沿用 degree-sorted；这是实际每卡 batch，不是手工更新窗口。
    def get_train_dataloader(self):
        sampler = TailSafeRankBatchSampler(len(self.train_dataset), self.num_generations,
            self.accelerator.num_processes, self.accelerator.process_index, self.args.per_device_train_batch_size)
        return DataLoaderShard(self.train_dataset, batch_sampler=sampler, collate_fn=self.data_collator,
                               num_workers=0, pin_memory=self.args.dataloader_pin_memory, device=None, rng_types=None)

    # 参考/old 概率按原实验分块，policy loss 仍使用完整本卡微批和 Liger fused loss。
    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, **kwargs):
        if self._loss_attention_mask is not None:
            attention_mask = self._loss_attention_mask
        chunk = min(batch_size or input_ids.shape[0], self.data_args.logprob_micro_batch_size)
        return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask,
            logits_to_keep, batch_size=chunk, **kwargs)

    # 真正批量生成：输入本 rank 的全部候选，左侧 padding，不截断图或反馈。
    def _generate_batch(self, prompt_ids, stage):
        device, tokenizer = self.accelerator.device, self.processing_class
        cfg = self.mrra_config['generation']
        maximum, limit = generation_limits(cfg,stage)
        for ids in prompt_ids:
            if not ids or len(ids) > limit or len(ids) + maximum > self.mrra_config['generation']['context_tokens']:
                raise ValueError(f'{stage} prompt exceeds configured context; no silent truncation')
        tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in prompt_ids]
        input_ids = pad(tensors, padding_value=self.pad_token_id, padding_side='left')
        attention = pad([torch.ones_like(row) for row in tensors], padding_value=0, padding_side='left')
        generation = copy.deepcopy(self.generation_config)
        generation.max_new_tokens = maximum
        generation.num_return_sequences = 1
        if self.data_args.memory_stress_tokens:
            # 只在 LR=0 的独立探针中造最大长度 token；不复用探针权重。
            token = tokenizer(' graph', add_special_tokens=False)['input_ids'][0]
            count = min(maximum,self.data_args.memory_stress_tokens)
            tokens = [[token] * count for _ in prompt_ids]
            return [Segment(ids, row, tokenizer.decode(row), stage, 0.0, True) for ids, row in zip(prompt_ids, tokens)]
        start = time.perf_counter()
        with (unwrap_model_for_generation(self.model_wrapped, self.accelerator,
                  gather_deepspeed3_params=self.args.ds3_gather_for_generation) as model,
              torch.no_grad(),
              FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext()):
            was_training, use_cache = model.training, model.config.use_cache
            try:
                model.eval()
                model.config.use_cache = True
                from transformers import StoppingCriteriaList
                stop = (StopAfterStage(tokenizer,input_ids.shape[1],stage) if stage in STAGE_TAGS else
                        StopAfterJson(tokenizer,input_ids.shape[1]))
                result = model.generate(input_ids=input_ids, attention_mask=attention,
                    generation_config=generation, use_cache=True, synced_gpus=False,
                    stopping_criteria=StoppingCriteriaList([stop]))
            finally:
                model.config.use_cache = use_cache
                model.train(was_training)
        elapsed = time.perf_counter() - start
        output = []
        for index,(prompt,row) in enumerate(zip(prompt_ids, result[:, input_ids.shape[1]:].tolist())):
            if isinstance(stop,StopAfterStage) and stop.stop_lengths is not None and stop.stop_lengths[index] is not None:
                row = row[:stop.stop_lengths[index]]
            if self.eos_token_id in row:
                row = row[:row.index(self.eos_token_id) + 1]
            else:
                while row and row[-1] == self.pad_token_id:
                    row.pop()
            if not row:
                raise ValueError('Empty generated trajectory segment')
            text = tokenizer.decode(row, skip_special_tokens=True)
            truncated = len(row) >= maximum and not segment_complete(text,stage) and row[-1] != self.eos_token_id
            output.append(Segment(prompt, row, text, stage, elapsed / len(prompt_ids), truncated))
        return output

    # Graph-R1 的 _generate 继续负责全局 token 统计；本方法只替换实际轨迹生成。
    def _generate_single_turn(self, prompts):
        tokenizer = self.processing_class
        prompt_ids = [tokenizer(prompt, add_special_tokens=False)['input_ids'] for prompt in prompts]
        strategy = self.mrra_config['strategy']
        if strategy == 'executor':
            return self._generate_executor_trajectory(prompt_ids)
        first = self._generate_batch(prompt_ids, 'llm')
        decisions = [decide(row['episode'], segment, strategy, self.mrra_config['ablation'])
                     for row, segment in zip(self._active_inputs, first)]
        completions, self._trajectory_masks, self._records = [], [], []
        for i, (row, segment, decision) in enumerate(zip(self._active_inputs, first, decisions)):
            segments = [segment]
            tokens, mask = pack_segments(segments,[])
            completions.append(tokens)
            self._trajectory_masks.append(mask)
            record = finish_record(row['episode'], strategy, row['task_messages'], segments, decision,
                                   metrics=self.mrra_config['training']['record_reference_metrics'])
            self._records.append(record)
        return prompt_ids, completions, None, {}

    def _generate_executor_trajectory(self,prompt_ids):
        tokenizer, ablation = self.processing_class, self.mrra_config['ablation']
        observations = [public_observation(row['episode']) for row in self._active_inputs]
        modify = self._generate_batch(prompt_ids,'modify')
        structural = [execute_stage(o,s.text,'modify',ablation) for o,s in zip(observations,modify)]
        rewrite_messages, first_suffixes, rewrite_prompts = [], [], []
        for row,segment,result in zip(self._active_inputs,modify,structural):
            messages,suffix,ids = feedback_tokens(tokenizer,row['task_messages'],segment,result['execution'],'rewrite')
            rewrite_messages.append(messages);first_suffixes.append(suffix);rewrite_prompts.append(ids)
        rewrite = self._generate_batch(rewrite_prompts,'rewrite')
        textual = [execute_stage(observation_after(o,m),r.text,'rewrite',ablation)
                   for o,m,r in zip(observations,structural,rewrite)]
        final_messages, second_suffixes, answer_prompts = [], [], []
        for messages,segment,result in zip(rewrite_messages,rewrite,textual):
            messages,suffix,ids = feedback_tokens(tokenizer,messages,segment,result['execution'],'predict')
            final_messages.append(messages);second_suffixes.append(suffix);answer_prompts.append(ids)
        answers = self._generate_batch(answer_prompts,'predict')
        completions, self._trajectory_masks, self._records = [], [], []
        for i,row in enumerate(self._active_inputs):
            segments = [modify[i],rewrite[i],answers[i]]
            tokens,mask = pack_segments(segments,[first_suffixes[i],second_suffixes[i]])
            record = finish_record(row['episode'],'executor',row['task_messages'],segments,
                aggregate_stages(structural[i],textual[i]),final_messages[i],
                metrics=self.mrra_config['training']['record_reference_metrics'],
                stage_results=[structural[i],textual[i]])
            completions.append(tokens);self._trajectory_masks.append(mask);self._records.append(record)
        return prompt_ids,completions,None,{}

    # A 的 B+T / B 的 B+T+F 作为整条轨迹奖励；组内优势计算由 Graph-R1 父类完成。
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        values = torch.tensor([[row['reward']] for row in self._records], device=self.accelerator.device)
        return self.accelerator.gather(values)

    def _generate_and_score_completions(self, inputs):
        if not isinstance(inputs, list):
            raise TypeError('MRRA uses the same identity collator as degree-sorted')
        self._active_inputs = inputs
        source_ids = gather_object([row['episode']['sample_id'] for row in inputs])
        if any(len(set(source_ids[i:i + self.num_generations])) != 1
               for i in range(0, len(source_ids), self.num_generations)):
            raise RuntimeError('GRPO candidates from different sources entered the same group')
        output = super()._generate_and_score_completions(inputs)
        # 父类已用完整 attention 计算 ref/old logps；在此单独替换学习 mask。
        output['completion_attention_mask'] = output['completion_mask'].clone()
        masks = [torch.tensor(mask, device=self.accelerator.device) for mask in self._trajectory_masks]
        output['completion_mask'] = pad(masks, padding_value=0, padding_side='right')
        output['num_items_in_batch'] = self.accelerator.gather(output['completion_mask'].sum()).sum()
        rank = self.accelerator.process_index
        for i, record in enumerate(self._records):
            record.update(step=self.state.global_step, micro_step=self._micro_step,
                epoch=math.floor(self.state.epoch or 0) + 1, rank=rank,
                candidate=(rank * len(inputs) + i) % self.num_generations,
                advantage=float(output['advantages'][i]),
                loss_mask=self._trajectory_masks[i],
                loss_token_count=sum(self._trajectory_masks[i]),
                environment_token_count=len(self._trajectory_masks[i]) - sum(self._trajectory_masks[i]))
            append_jsonl(Path(self.args.output_dir) / f'rollouts_rank{rank}.jsonl', record)
        append_jsonl(Path(self.args.output_dir) / f'progress_rank{rank}.jsonl',
            {'step': self.state.global_step, 'micro_step': self._micro_step, 'batch_size': len(inputs),
             'source_ids': [row['episode']['sample_id'] for row in inputs],
             'reward_mean': sum(row['reward'] for row in self._records) / len(inputs)})
        print(f'MRRA rank={rank} step={self.state.global_step} micro={self._micro_step} '
              f'batch={len(inputs)} strategy={self.mrra_config["strategy"]}', flush=True)
        self._micro_step += 1
        return output

    # TRL 的 GRPO/Liger 目标及缩放保持不变；唯一区别是 attention 不屏蔽环境，loss 屏蔽环境。
    def compute_liger_loss(self, unwrapped_model, inputs):
        ids = torch.cat([inputs['prompt_ids'], inputs['completion_ids']], dim=1)
        attention = torch.cat([inputs['prompt_mask'], inputs['completion_attention_mask']], dim=1)
        hidden = self._get_last_hidden_state(unwrapped_model, ids, attention, inputs['completion_ids'].size(1))
        loss, metrics = self.liger_grpo_loss(_input=hidden, lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=inputs['completion_ids'], attention_mask=inputs['completion_mask'],
            advantages=inputs['advantages'], bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get('old_per_token_logps'), ref_per_token_logps=inputs.get('ref_per_token_logps'))
        if self.beta:
            self._metrics['train']['kl'].append(self.accelerator.gather(metrics[0]).mean().item())
        self._metrics['train']['clip_ratio'].append(self.accelerator.gather(metrics[-1]).mean().item())
        return loss / self.current_gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # CPU/非 Liger 验证仍调用 TRL 原生 _compute_loss，仅修正前向 attention mask。
        self._loss_attention_mask = torch.cat([inputs['prompt_mask'], inputs['completion_attention_mask']], dim=1)
        try:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        finally:
            self._loss_attention_mask = None


def train():
    parser = HfArgumentParser((ModelArguments, MRRADataArguments, GraphRLTrainingArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses()
    cfg = load_config(data_args.mrra_config)
    if cfg['strategy']=='executor' and cfg.get('protocol') != PROTOCOL_VERSION:
        raise ValueError('B config must select the current three-stage tagged protocol')
    if Path(model_args.model_id).resolve() != resolve(cfg['model']['path']).resolve():
        raise ValueError('Direct GRPO must use the original base specified by the experiment config')
    if args.num_iterations != 1 or args.steps_per_generation != 1:
        raise ValueError('Use one iteration and one generation microbatch, matching degree-sorted')
    if data_args.validation_only and args.learning_rate != 0:
        raise ValueError('Validation probes must have LR=0; their weights are never reused')
    if data_args.memory_stress_tokens and (not data_args.validation_only or args.max_steps != 1):
        raise ValueError('Memory stress requires validation_only and max_steps=1')
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite output: ' + str(output))
    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_id, local_files_only=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation='sdpa' if args.disable_flash_attn2 else 'flash_attention_2')
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    data = cfg['data']
    samples = load_episodes(data_args.data_path or data['train'], data['partition'], data['fraction'], data_args.limit_data_sources)
    if data_args.continuation_checkpoint:
        checkpoint = Path(data_args.continuation_checkpoint)
        previous_config = checkpoint.parent / 'experiment_config.json'
        previous_state = checkpoint / 'trainer_state.json'
        if not previous_config.exists() or not previous_state.exists():
            raise ValueError('Resume requires this project\'s full Trainer checkpoint and experiment config')
        previous = json.loads(previous_config.read_text())
        state = json.loads(previous_state.read_text())
        for key in ('per_device_train_batch_size', 'gradient_accumulation_steps', 'num_generations'):
            if previous['training_arguments'][key] != getattr(args, key):
                raise ValueError('Incompatible continuation layout: ' + key)
        if previous['mrra']['strategy'] != cfg['strategy']:
            raise ValueError('A/B cannot resume each other\'s LoRA')
        if cfg['strategy']=='executor' and previous['mrra'].get('protocol') != PROTOCOL_VERSION:
            raise ValueError('Old two-stage B checkpoints cannot resume the new three-stage protocol/reward')
        old_ids, new_ids = set(previous['source_ids']), {s['sample_id'] for s in samples}
        if old_ids != new_ids:
            if old_ids & new_ids or not args.ignore_data_skip or args.max_steps <= state['global_step']:
                raise ValueError('Disjoint suffix continuation requires ignore_data_skip=True and an extended max_steps horizon')
    dataset = GraphRLDataset(samples, tokenizer, cfg)
    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=cfg['lora']['target_modules'], bias='none', task_type='CAUSAL_LM')
    if not args.lora_enable:
        raise ValueError('Direct MRRA uses LoRA on the original frozen base')
    def trajectory_reward(**kwargs):
        return [0.] * len(kwargs.get('completions', []))
    callbacks = [] if data_args.validation_only else [EpochCheckpointSnapshots([s['sample_id'] for s in samples], args.num_generations)]
    trainer = ThinkWithGraphTrainer(model=model, args=args, train_dataset=dataset, eval_dataset=None,
        processing_class=tokenizer, reward_funcs=[trajectory_reward], peft_config=lora, callbacks=callbacks,
        mrra_config=cfg, data_args=data_args)
    if any(p.requires_grad for n, p in trainer.model.named_parameters() if 'lora_' not in n):
        raise ValueError('Base weights must remain frozen')
    if trainer.accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'experiment_config.json', {'mrra': cfg, 'training_arguments': args.to_dict(),
            'trainer_mro': [f'{c.__module__}.{c.__name__}' for c in type(trainer).__mro__],
            'source_ids': [s['sample_id'] for s in samples], 'validation_only': data_args.validation_only})
        snapshot = output / 'code_snapshot'
        snapshot.mkdir()
        for folder in ('code', 'configs', 'scripts'):
            shutil.copytree(ROOT / folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__'))
        (output / 'status.txt').write_text('training\n')
    trainer.accelerator.wait_for_everyone()
    trainer.train(resume_from_checkpoint=data_args.continuation_checkpoint)
    trainer.save_state()
    memory = {'rank': trainer.accelerator.process_index, 'max_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
              'max_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
              'completed_steps': trainer.state.global_step, 'validation_only': data_args.validation_only}
    write_json(output / f'memory_rank{trainer.accelerator.process_index}.json', memory)
    if not data_args.validation_only:
        trainer.save_model(str(output / 'final_adapter'))
        if trainer.accelerator.is_main_process:
            tokenizer.save_pretrained(output / 'final_adapter')
    if trainer.accelerator.is_main_process:
        write_json(output / 'summary.json', {'global_step': trainer.state.global_step,
            'adapter': str(output / 'final_adapter'), 'epochs': trainer.state.epoch,
            'trainer': 'ThinkWithGraphTrainer -> GraphR1GRPOTrainer -> TRL GRPOTrainer'})
        (output / 'status.txt').write_text('probe_complete\n' if data_args.validation_only else 'complete\n')


if __name__ == '__main__':
    train()
