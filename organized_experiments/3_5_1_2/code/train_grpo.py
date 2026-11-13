"""3_5_1_2: NC/LC/LP single-turn GRPO prediction without tools."""
from __future__ import annotations
import os

import json
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path

import torch
import trl
import transformers
from accelerate.data_loader import DataLoaderShard
from peft import LoraConfig
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed, StoppingCriteriaList
from trl import GRPOConfig, GRPOTrainer
from trl.distributed import DistributedBackend
from trl.models import unwrap_model_for_generation

from episode import initial_messages, load_episodes
from reward import score_episode
from tail_batching import TailSafeRankBatchSampler
from task_utils import prepare_episode, dataset_statistics


WORKSPACE = Path(__file__).resolve().parents[4]
BASE_MODEL = Path(os.environ.get('TRAIN_MODEL', str(WORKSPACE / 'Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct')))


@dataclass
class ModelArguments:
    model_id: str = str(BASE_MODEL)
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    disable_flash_attn2: bool = False


@dataclass
class GraphRLDataArguments:
    data_path: str = str(WORKSPACE / 'GraphRL/2000_oneclass_sorted/train_grpo_episodes_redesign.json')
    max_prompt_length: int = 8192
    prediction_max_new_tokens: int = 512
    logprob_micro_batch_size: int = 1
    continuation_checkpoint: str | None = None
    limit: int | None = None


@dataclass
class GraphRLTrainingArguments(GRPOConfig):
    # 7_1-MRRA已完成7B A策略的实际参数（selected_profile / training_arguments）。
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-6
    warmup_steps: float = 55
    lr_scheduler_type: str = 'cosine'
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = 'adamw_torch'
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict | str | None = field(default_factory=lambda: {'use_reentrant': True})
    deepspeed: str | None = str(Path(__file__).resolve().parents[1]/'configs/zero3_7b.json')
    use_liger_kernel: bool = True
    disable_dropout: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    save_steps: int = 100
    save_total_limit: int | None = 3
    logging_steps: float = 1
    # 本实验sampler每次提供一个候选微批次，保持3号完整G组和尾批不丢弃。
    steps_per_generation: int | None = 1
    num_iterations: int = 1
    num_generations: int = 6
    beta: float = 0.01
    loss_type: str = 'grpo'
    shuffle_dataset: bool | None = False
    max_tool_calling_iterations: int | None = 0
    # One prediction response, including analysis and label.
    max_completion_length: int | None = 512
    use_vllm: bool = False
    mask_truncated_completions: bool = False
    use_bias_correction_kl: bool = False  # 保持3号原KL目标，工具循环使用新版实现。


class GraphRLDataset(Dataset):
    """JSON整体读取，候选复制和分卡仍由3号sampler完成。"""
    def __init__(self, path, limit=None):
        self.samples = load_episodes(path, limit) if isinstance(path,(str,Path)) else [prepare_episode(s) for s in path]
        if not self.samples:
            raise ValueError('Empty episode dataset')
        if len({s['source_index'] for s in self.samples})!=len(self.samples):
            raise ValueError('Duplicate source_index')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {'prompt': initial_messages(sample), 'episode': sample, 'source_index': sample['source_index']}

    def to_hf_dataset(self):
        # 新版TRL要求HF Dataset。图字典存为JSON字符串，避免Arrow展开所有节点ID。
        rows = []
        for i in range(len(self)):
            row = self[i]
            rows.append(dict(row,episode=json.dumps(row['episode'],ensure_ascii=False)))
        return HFDataset.from_list(rows)


def graph_reward(completions, episode, **kwargs):
    return [score_episode(sample, completion)[0]
            for sample, completion in zip(episode, completions, strict=True)]


class ThinkWithGraphTrainer(GRPOTrainer):
    """Keep the source sampler and LoRA loss; generate one prediction per candidate."""
    def __init__(self, *args, data_args=None, **kwargs):
        self.data_args = data_args or GraphRLDataArguments()
        if self.data_args.logprob_micro_batch_size < 1:
            raise ValueError('logprob_micro_batch_size must be positive')
        self.last_rollout_records = []
        kwargs['reward_funcs'] = [self._score_predictions]
        super().__init__(*args, **kwargs)
        if self.tools or self.environment_factories is not None:
            raise ValueError('Direct prediction must not register tools or environments')
        if self.max_completion_length != self.data_args.prediction_max_new_tokens:
            raise ValueError('Completion and prediction token budgets must match')
        if self.args.steps_per_generation != 1 or self.num_iterations != 1:
            raise ValueError('Tail-safe loader requires steps_per_generation=num_iterations=1')
        if self.use_vllm:
            raise ValueError('This launcher uses the HF backend for per-turn generation budgets')

    def _set_signature_columns_if_needed(self):
        self._signature_columns = ['prompt','episode','source_index']

    def get_train_dataloader(self):
        sampler = TailSafeRankBatchSampler(len(self.train_dataset), self.num_generations,
            self.accelerator.num_processes, self.accelerator.process_index,
            self.args.per_device_train_batch_size)
        return DataLoaderShard(self.train_dataset, batch_sampler=sampler,
            collate_fn=self.data_collator, num_workers=0, pin_memory=self.args.dataloader_pin_memory,
            device=None, rng_types=None)

    def _tokenize_prompts(self, prompts):
        result = super()._tokenize_prompts(prompts)
        if any(len(ids)>self.data_args.max_prompt_length for ids in result[0]):
            raise ValueError('Initial prompt exceeds budget; no silent truncation')
        return result

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                                         logits_to_keep, batch_size=None, **kwargs):
        # 与A策略相同：reference/old的no_grad评分逐候选分块，policy前向保留真实每卡batch。
        if not torch.is_grad_enabled():
            batch_size = min(batch_size or input_ids.shape[0], self.data_args.logprob_micro_batch_size)
        return super()._get_per_token_logps_and_entropies(model,input_ids,attention_mask,
            logits_to_keep,batch_size=batch_size,**kwargs)

    def _generate(self, prompts):
        # Gather ZeRO-3 parameters once for the single prediction generation.
        if not DistributedBackend(self.accelerator).is_zero3:
            return super()._generate(prompts)
        if not self.args.ds3_gather_for_generation:
            raise ValueError('ZeRO-3 generation requires ds3_gather_for_generation=True')
        with unwrap_model_for_generation(self.model_wrapped,self.accelerator,
                gather_deepspeed3_params=True,
                gradient_checkpointing_kwargs=self.args.gradient_checkpointing_kwargs) as model:
            had_override = 'generate' in model.__dict__
            previous_override = model.__dict__.get('generate')
            original_generate = model.generate
            @wraps(original_generate)
            def local_generate(*args, **kwargs):
                # Full weights are resident throughout this outer context.
                kwargs['synced_gpus'] = False
                return original_generate(*args, **kwargs)
            model.generate = local_generate
            self.args.ds3_gather_for_generation = False
            try:
                return super()._generate(prompts)
            finally:
                self.args.ds3_gather_for_generation = True
                if had_override:
                    model.generate = previous_override
                else:
                    delattr(model,'generate')

    def _generate_single_turn(self, *args, **kwargs):
        ids, logps, sampling = super()._generate_single_turn(*args, **kwargs)
        self._terminated = [bool(row) and row[-1] in self.eos_token_ids for row in ids]
        return ids, logps, sampling

    def _score_predictions(self, completions, episode, **kwargs):
        terminated = self._terminated
        self._scored_records = []
        rewards = []
        for sample, completion, ended in zip(episode, completions, terminated, strict=True):
            sample = json.loads(sample) if isinstance(sample, str) else sample
            reward, record = score_episode(sample, completion, terminated=ended)
            self._scored_records.append(dict(record, source_index=sample['source_index'],
                messages=completion, reward=reward))
            rewards.append(reward)
        logger = kwargs.get('log_metric')
        if logger:
            for name in self._scored_records[0]['metrics']:
                logger(name, sum(r['metrics'][name] for r in self._scored_records)/len(self._scored_records))
        return rewards

    def _generate_and_score_completions(self, inputs):
        # Standard single-turn TRL generation, reward, advantages and loss.
        if self.model.training:
            ids = torch.tensor([row['source_index'] for row in inputs],device=self.accelerator.device)
            groups = self.accelerator.gather(ids).view(-1,self.num_generations)
            if not torch.all(groups == groups[:,:1]):
                raise RuntimeError('Broken GRPO source grouping across ranks')
        output = super()._generate_and_score_completions(inputs)
        self.last_rollout_records = []
        for i,(row,scored) in enumerate(zip(inputs,self._scored_records,strict=True)):
            record = dict(scored, prompt=row['prompt'],
                step=self.state.global_step, micro_step=self._step,
                prompt_ids=output['prompt_ids'][i].detach().cpu().tolist(),
                prompt_mask=output['prompt_mask'][i].detach().cpu().tolist(),
                completion_ids=output['completion_ids'][i].detach().cpu().tolist(),
                completion_mask=output['completion_mask'][i].detach().cpu().tolist(),
                advantage=float(output['advantages'][i].detach().cpu()))
            self.last_rollout_records.append(record)
        if self.model.training:
            path = Path(self.args.output_dir)
            path.mkdir(parents=True,exist_ok=True)
            with (path/f'rollouts_rank{self.accelerator.process_index}.jsonl').open('a') as stream:
                for record in self.last_rollout_records:
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        return output


def train():
    """与3号同样按模型/数据/训练三组参数构造，再创建Trainer、train、保存。"""
    parser = HfArgumentParser((ModelArguments,GraphRLDataArguments,GraphRLTrainingArguments))
    model_args,data_args,training_args = parser.parse_args_into_dataclasses()
    output = Path(training_args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing existing output: {output}')
    if Path(model_args.model_id).resolve()!=BASE_MODEL.resolve():
        raise ValueError('Initialize 3_5_1_2 from the original Qwen2.5-7B-Instruct')
    set_seed(training_args.seed)
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_args.model_id,dtype=dtype,
        local_files_only=True,attn_implementation='sdpa' if model_args.disable_flash_attn2 else 'flash_attention_2')
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id,local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='left'
    lora_config = LoraConfig(r=model_args.lora_rank,lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,bias='none',task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    dataset = GraphRLDataset(data_args.data_path,data_args.limit)
    trainer = ThinkWithGraphTrainer(model=model,args=training_args,train_dataset=dataset.to_hf_dataset(),
        processing_class=tokenizer,peft_config=lora_config,reward_funcs=[graph_reward],
        data_args=data_args)
    if trainer.is_world_process_zero():
        output.mkdir(parents=True,exist_ok=True)
        config=dict(experiment_id='3_5_1_2',implementation='native_trl_direct_prediction',
            trl_version=trl.__version__,transformers_version=transformers.__version__,
            model=asdict(model_args),data=asdict(data_args),training=training_args.to_dict(),
            train_count=len(dataset),supported_tasks=['NC','LC','LP'],train_statistics=dataset_statistics(dataset.samples),
            query_edge_masking='LC and LP clean/noisy views; both directions unless query_directed=True',
            reward_version='3_5_1_2_structure_prediction',
            reward='R=S+T; S=1 for valid prediction format else 0; T=1 correct, 0.1 candidate incorrect, 0 otherwise; invalid/truncated prediction earns 0')
        config['hyperparameter_reference'] = json.loads(
            (Path(__file__).resolve().parents[1]/'configs/a7b_reference.json').read_text())
        (output/'experiment_config.json').write_text(json.dumps(config,indent=2)+'\n')
        (output/'chat_template.jinja').write_text(trainer.chat_template or tokenizer.chat_template)
    trainer.train(resume_from_checkpoint=data_args.continuation_checkpoint)
    trainer.save_state()
    trainer.save_model(str(output/'final_adapter'))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output/'final_adapter')
        (output/'completed.json').write_text(json.dumps(dict(global_step=trainer.state.global_step,
            epoch=trainer.state.epoch,train_count=len(dataset),trl_version=trl.__version__))+'\n')


if __name__=='__main__':
    train()
