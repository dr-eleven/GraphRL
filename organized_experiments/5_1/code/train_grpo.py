"""Binary outcome GRPO; custom finite rollout controller, native TRL policy loss."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from functools import partial, wraps
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

from episode import GraphEnvironment, initial_messages, load_episodes
from budgets import RolloutBudget
from prompt_template import single_call_template
from project_config import EXPERIMENT_ID, DEFAULT_MODE
from tail_batching import TailSafeRankBatchSampler
from generation_limits import TokenLimitToolsMixin, CandidateTokenBudget

WORKSPACE = Path(__file__).resolve().parents[4]
BASE_MODEL = Path(os.environ.get('TRAIN_MODEL', str(WORKSPACE / 'Graph-R1-main/locals/Qwen/Qwen2.5-7B-Instruct')))


@dataclass
class ModelArguments:
    model_id: str = str(BASE_MODEL)
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    disable_flash_attn2: bool = False


@dataclass(frozen=True)
class GraphRLDataArguments(RolloutBudget):
    data_path: str = str(WORKSPACE / 'GraphRL/1000_oneclass_sorted/train_grpo_episodes_redesign.json')
    logprob_micro_batch_size: int = 1
    generation_micro_batch_size: int = 1
    continuation_checkpoint: str | None = None
    limit: int | None = None

    def __post_init__(self):
        # Validate only budget fields, not path/optional trainer fields.
        RolloutBudget(**{f.name:getattr(self,f.name) for f in fields(RolloutBudget)})

    def rollout_budget(self, completion_length=None):
        values = {f.name:getattr(self,f.name) for f in fields(RolloutBudget)}
        if completion_length is not None:
            if completion_length != self.max_completion_tokens:
                raise ValueError('max_completion_length must equal max_completion_tokens')
        return RolloutBudget(**values)


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
    save_steps: int = 50
    save_total_limit: int | None = 3
    logging_steps: float = 1
    # 本实验sampler每次提供一个候选微批次，保持3号完整G组和尾批不丢弃。
    steps_per_generation: int | None = 1
    num_iterations: int = 1
    num_generations: int = 6
    beta: float = 0.01
    loss_type: str = 'grpo'
    shuffle_dataset: bool | None = False
    max_tool_calling_iterations: int | None = 10
    # 包含所有工具轮的推理、调用、工具反馈和最终推理/答案。
    max_completion_length: int | None = 16384
    use_vllm: bool = False
    mask_truncated_completions: bool = False
    use_bias_correction_kl: bool = False  # 保持3号原KL目标，工具循环使用新版实现。


class GraphRLDataset(Dataset):
    """JSON整体读取，候选复制和分卡仍由3号sampler完成。"""
    def __init__(self, path, limit=None, bypass_actions=False, budget=None):
        self.samples = load_episodes(path, limit) if isinstance(path,(str,Path)) else path
        self.bypass_actions = bypass_actions
        self.budget = budget or RolloutBudget()
        if not self.samples:
            raise ValueError('Empty episode dataset')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = initial_messages(sample,self.budget,bypass_actions=self.bypass_actions)
        return {'prompt':prompt, 'episode':sample, 'source_index':sample['source_index'],
                'bypass_actions':self.bypass_actions}

    def to_hf_dataset(self):
        # 新版TRL要求HF Dataset。图字典存为JSON字符串，避免Arrow展开所有节点ID。
        rows = []
        for i in range(len(self)):
            row = self[i]
            rows.append(dict(row,episode=json.dumps(row['episode'],ensure_ascii=False)))
        return HFDataset.from_list(rows)


def graph_reward(completions, environments, **kwargs):
    """Strict 0/1 terminal correctness; no intermediate reward or generation."""
    return [env._score(completion) for env,completion in zip(environments,completions,strict=True)]


class ThinkWithGraphTrainer(TokenLimitToolsMixin, GRPOTrainer):
    """Custom raw-event rollout; native TRL group advantage, reference and policy loss."""
    def __init__(self, *args, data_args=None, **kwargs):
        self.data_args = data_args or GraphRLDataArguments()
        if self.data_args.logprob_micro_batch_size < 1:
            raise ValueError('logprob_micro_batch_size must be positive')
        if self.data_args.generation_micro_batch_size < 1:
            raise ValueError('generation_micro_batch_size must be positive')
        self.last_rollout_records = []
        super().__init__(*args, **kwargs)
        self.chat_template = single_call_template(self.chat_template or self._tokenizer.chat_template)
        budget = self.data_args.rollout_budget(self.max_completion_length)
        if self.args.max_tool_calling_iterations < budget.max_model_turns:
            raise ValueError('Framework iteration cap must cover max_model_turns')
        if self.mask_truncated_completions:
            raise ValueError('Policy budget failures must remain in GRPO; truncated masking is forbidden')
        if self.args.loss_type != 'grpo' or self.args.scale_rewards != 'group':
            raise ValueError('Main experiment requires standard group-normalized GRPO')
        if self.args.steps_per_generation != 1 or self.num_iterations != 1:
            raise ValueError('Tail-safe loader requires steps_per_generation=num_iterations=1')
        if self.use_vllm:
            raise ValueError('This launcher uses the HF backend for per-turn generation budgets')

    def _init_training_state(self, max_steps, num_update_steps_per_epoch,
                             num_train_epochs, resume_from_checkpoint, trial):
        result = super()._init_training_state(max_steps, num_update_steps_per_epoch,
            num_train_epochs, resume_from_checkpoint, trial)
        # Checkpoint state contains old intervals; explicit current settings win.
        self.state.compute_steps(self.args, max_steps)
        if self.is_world_process_zero():
            path = Path(self.args.output_dir)/'effective_resume_state.json'
            path.write_text(json.dumps(dict(global_step=self.state.global_step,
                max_steps=max_steps, save_steps=self.state.save_steps,
                generation_micro_batch_size=self.data_args.generation_micro_batch_size,
                skipped_batches=result[1]), indent=2)+'\n')
        return result

    def _set_signature_columns_if_needed(self):
        self._signature_columns = ['prompt','episode','source_index','bypass_actions']

    def get_train_dataloader(self):
        sampler = TailSafeRankBatchSampler(len(self.train_dataset), self.num_generations,
            self.accelerator.num_processes, self.accelerator.process_index,
            self.args.per_device_train_batch_size)
        return DataLoaderShard(self.train_dataset, batch_sampler=sampler,
            collate_fn=self.data_collator, num_workers=0, pin_memory=self.args.dataloader_pin_memory,
            device=None, rng_types=None)

    def _tokenize_prompts(self, prompts):
        result = super()._tokenize_prompts(prompts)
        if any(len(ids)>self.data_args.max_initial_prompt_tokens for ids in result[0]):
            raise ValueError('Initial prompt including native tool schema exceeds budget; no silent truncation')
        return result

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                                         logits_to_keep, batch_size=None, **kwargs):
        # 与A策略相同：reference/old的no_grad评分逐候选分块，policy前向保留真实每卡batch。
        if not torch.is_grad_enabled():
            batch_size = min(batch_size or input_ids.shape[0], self.data_args.logprob_micro_batch_size)
        return super()._get_per_token_logps_and_entropies(model,input_ids,attention_mask,
            logits_to_keep,batch_size=batch_size,**kwargs)

    def _generate(self, prompts):
        self._active_rollout_indices = None
        self._initial_prompt_lengths = None
        # 各rank的候选可以提前结束，原生tool loop的本地轮数因此不一致。
        # ZeRO-3在整条rollout外只聚参一次，期间移除DS hooks；本地generate
        # 不做跨rank同步，避免某卡下一轮聚参与另一卡统计collective错位。
        # 生成控制由本项目负责；GRPO优势和loss继续使用原生TRL。
        if not DistributedBackend(self.accelerator).is_zero3:
            return super()._generate(prompts)
        if not self.args.ds3_gather_for_generation:
            raise ValueError('ZeRO-3 native tools require ds3_gather_for_generation=True')
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

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
        """Generate only active candidates using controller-computed per-row caps."""
        indices = list(self._active_rollout_indices)
        width = self.data_args.generation_micro_batch_size
        if len(prompt_ids) <= width:
            return self._generate_bounded_turn(prompt_ids, images, multimodal_fields,
                num_generations, has_tool_images=has_tool_images)
        if (images and any(x is not None for x in images)) or multimodal_fields or has_tool_images:
            raise ValueError('Generation microbatching supports the text-only graph protocol')
        chunks = []
        try:
            for start in range(0, len(prompt_ids), width):
                stop = min(start+width, len(prompt_ids))
                self._active_rollout_indices = indices[start:stop]
                value = self._generate_bounded_turn(prompt_ids[start:stop], None, {},
                    num_generations, has_tool_images=False)
                if len(value[0]) != stop-start:
                    raise RuntimeError('Generation microbatch lost candidate alignment')
                chunks.append(value)
        finally:
            self._active_rollout_indices = indices
        merged = []
        for column in range(3):
            values = [chunk[column] for chunk in chunks]
            if all(value is None for value in values):
                merged.append(None)
            elif any(value is None for value in values):
                raise RuntimeError('Inconsistent generation microbatch metadata')
            else:
                merged.append([row for value in values for row in value])
        return tuple(merged)

    def _generate_bounded_turn(self, prompt_ids, images, multimodal_fields,
                               num_generations, has_tool_images=False):
        """Generate a text microbatch while retaining candidate-local stage budgets."""
        indices = self._active_rollout_indices
        context = self.model.config.max_position_embeddings
        padded_room = context-max(map(len,prompt_ids))
        budgets = [min(self.environments[i]._next_cap,padded_room) for i in indices]
        if min(budgets) < 1:
            raise RuntimeError('No capacity for a requested generation')

        model = self.accelerator.unwrap_model(self.model_wrapped)
        had_override = 'generate' in model.__dict__
        saved_override = model.__dict__.get('generate')
        original_generate = model.generate
        @wraps(original_generate)
        def bounded_generate(*args, **kwargs):
            input_ids = kwargs.get('input_ids', args[0] if args else None)
            criteria = StoppingCriteriaList(list(kwargs.get('stopping_criteria') or []))
            criteria.append(CandidateTokenBudget(input_ids.shape[1], budgets))
            kwargs['stopping_criteria'] = criteria
            return original_generate(*args, **kwargs)
        previous = self.generation_config.max_new_tokens
        model.generate = bounded_generate
        try:
            self.generation_config.max_new_tokens = max(1, max(budgets))
            ids, logps, sampling = super()._generate_single_turn(
                prompt_ids, images, multimodal_fields, num_generations,
                has_tool_images=has_tool_images)
        finally:
            self.generation_config.max_new_tokens = previous
            if had_override: model.generate = saved_override
            else: delattr(model, 'generate')
        for local, original in enumerate(indices):
            # Native HF batches can right-pad an already stopped row; retain only its budget.
            ids[local] = ids[local][:budgets[local]]
            if logps is not None: logps[local] = logps[local][:budgets[local]]
            if sampling is not None: sampling[local] = sampling[local][:budgets[local]]
            normal_end = bool(ids[local]) and ids[local][-1] in self.eos_token_ids
            self.environments[original]._last_finish = (
                'stop' if normal_end or len(ids[local]) < budgets[local] else 'length')
        return ids, logps, sampling

    def _generate_and_score_completions(self, inputs):
        # TRL creates per-candidate environments; our _generate handles all turns, then TRL scores GRPO.
        if self.model.training:
            ids = torch.tensor([row['source_index'] for row in inputs],device=self.accelerator.device)
            groups = self.accelerator.gather(ids).view(-1,self.num_generations)
            if not torch.all(groups == groups[:,:1]):
                raise RuntimeError('Broken GRPO source grouping across ranks')
        output = super()._generate_and_score_completions(inputs)
        self.last_rollout_records = []
        for i,(row,env) in enumerate(zip(inputs,self.environments,strict=True)):
            record = dict(env._record, infra_group_retries=env._infra_group_retries, prompt=row['prompt'],
                step=self.state.global_step, micro_step=self._step,
                prompt_ids=output['prompt_ids'][i].detach().cpu().tolist(),
                prompt_mask=output['prompt_mask'][i].detach().cpu().tolist(),
                completion_ids=output['completion_ids'][i].detach().cpu().tolist(),
                completion_mask=output['completion_mask'][i].detach().cpu().tolist(),
                tool_mask=output['tool_mask'][i].detach().cpu().tolist() if 'tool_mask' in output else None,
                advantage=float(output['advantages'][i].detach().cpu()))
            self.last_rollout_records.append(record)
        if self.model.training:
            rewards = self.accelerator.gather(torch.tensor([e._record['reward'] for e in self.environments],
                                                          device=self.accelerator.device)).view(-1,self.num_generations)
            self._metrics['train']['binary/all_zero_groups'].append((rewards==0).all(1).float().mean().item())
            self._metrics['train']['binary/all_one_groups'].append((rewards==1).all(1).float().mean().item())
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
        raise ValueError('Initialize the main binary experiment from the original Qwen2.5-7B-Instruct')
    budget = data_args.rollout_budget(training_args.max_completion_length)
    if data_args.continuation_checkpoint:
        checkpoint = Path(data_args.continuation_checkpoint).resolve()
        if not (checkpoint/'trainer_state.json').is_file():
            raise ValueError(f'Invalid continuation checkpoint: {checkpoint}')
        prior = json.loads((checkpoint.parent/'experiment_config.json').read_text())
        if (prior.get('experiment_id') != EXPERIMENT_ID or prior.get('mode') != DEFAULT_MODE
                or prior.get('reward_version') != 'binary_terminal_v1'
                or prior.get('rollout_budget') != asdict(budget)):
            raise ValueError('Resume requires the same project, binary reward and rollout budget')
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
    dataset = GraphRLDataset(data_args.data_path,data_args.limit,budget=budget)
    trainer = ThinkWithGraphTrainer(model=model,args=training_args,train_dataset=dataset.to_hf_dataset(),
        processing_class=tokenizer,peft_config=lora_config,reward_funcs=[graph_reward],
        environment_factory=partial(GraphEnvironment,budget=budget),data_args=data_args)
    if trainer.is_world_process_zero():
        output.mkdir(parents=True,exist_ok=True)
        config=dict(experiment_id=EXPERIMENT_ID,implementation='binary_raw_event_controller',mode=DEFAULT_MODE,
            trl_version=trl.__version__,transformers_version=transformers.__version__,
            model=asdict(model_args),data=asdict(data_args),training=training_args.to_dict(),
            train_count=len(dataset),reward_version='binary_terminal_v1',
            reward='float(accepted_final_prediction matches canonical ground_truth); otherwise zero',
            rollout_budget=asdict(budget))
        config['continuation_checkpoint'] = data_args.continuation_checkpoint
        config['generation_micro_batch_size'] = data_args.generation_micro_batch_size
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
