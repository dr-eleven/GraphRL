"""HF/PEFT 生成与分块 token log-prob；生成与训练共享同一个 LoRA。"""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, PeftModel, get_peft_model

from config import resolve
from episode import complete_json
from executor_protocol import STAGE_TAGS


@dataclass
class Segment:
    prompt_ids: list[int]
    completion_ids: list[int]
    text: str
    stage: str
    elapsed_seconds: float = 0.0
    truncated: bool = False


class StopAfterJson(StoppingCriteria):
    def __init__(self, tokenizer, width):
        self.tokenizer, self.width = tokenizer, width

    def __call__(self, input_ids, scores, **kwargs):
        return torch.tensor([complete_json(self.tokenizer.decode(row[self.width:], skip_special_tokens=True))
                             for row in input_ids], device=input_ids.device)


class StopAfterStage(StoppingCriteria):
    """Per-row stop lengths exclude padding added while other rows keep generating."""
    def __init__(self, tokenizer, width, stage):
        self.tokenizer, self.width = tokenizer, width
        self.keyword = '</'+STAGE_TAGS[stage]+'>'
        self.stop_lengths = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.stop_lengths is None:
            self.stop_lengths = [None]*len(input_ids)
        stopped = []
        for index,row in enumerate(input_ids):
            if self.stop_lengths[index] is None:
                text = self.tokenizer.decode(row[self.width:],skip_special_tokens=True)
                if self.keyword in text:
                    self.stop_lengths[index] = len(row)-self.width
            stopped.append(self.stop_lengths[index] is not None)
        return torch.tensor(stopped,device=input_ids.device)


def generation_limits(generation, stage):
    keys = {'llm':'llm_tokens','edit':'edit_tokens','answer':'answer_tokens','direct':'answer_tokens',
            'ssr':'llm_tokens','modify':'modify_tokens','rewrite':'rewrite_tokens','predict':'answer_tokens'}
    key = keys[stage]
    maximum = generation.get(key, generation['edit_tokens']//2) if stage in {'modify','rewrite'} else generation[key]
    limit = generation['feedback_prompt_tokens'] if stage in {'answer','rewrite','predict'} else generation['prompt_tokens']
    return maximum, limit


def segment_complete(text, stage):
    return ('</'+STAGE_TAGS[stage]+'>' in text) if stage in STAGE_TAGS else complete_json(text)


class Policy:
    def __init__(self, model, tokenizer, cfg, device):
        self.model, self.tokenizer, self.cfg, self.device = model, tokenizer, cfg, torch.device(device)
        # Dropout 关闭，使 rollout/old/reference/current 概率可比。
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    @classmethod
    def load(cls, cfg, device, *, train=False, adapter=None, model_path=None):
        path = str(resolve(model_path or cfg['model']['path']))
        dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float32}[cfg['model']['dtype']]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=dtype,
                                                    attn_implementation=cfg['model']['attention']).to(device)
        if adapter:
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=train)
        elif train:
            settings = cfg['lora']
            model = get_peft_model(model, LoraConfig(r=settings['rank'], lora_alpha=settings['alpha'],
                lora_dropout=0.0, target_modules=settings['target_modules'], task_type='CAUSAL_LM', bias='none'))
        if train and cfg['training']['gradient_checkpointing']:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.enable_input_require_grads()
        return cls(model, tokenizer, cfg, device)

    def encode(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)

    # 每次生成一条候选，保留原始生成 token；不得 decode 后重编码代替策略概率目标。
    # 触发完整 JSON 停止条件不强制插入 EOS，避免把人为强制 token 当成策略动作。
    def generate(self, messages, stage, *, sample):
        ids = self.encode(messages)
        generation = self.cfg['generation']
        maximum, prompt_limit = generation_limits(generation, stage)
        context_limit = min(generation['context_tokens'], self.model.config.max_position_embeddings)
        if len(ids) > prompt_limit or len(ids) + maximum > context_limit:
            raise ValueError(f'{stage}: prompt {len(ids)} + output {maximum} exceeds configured context; no silent truncation')
        if not ids:
            raise ValueError('Chat template produced an empty token sequence')
        inputs = torch.tensor([ids], device=self.device, dtype=torch.long)
        # 不继承 Qwen generation_config 中的 repetition_penalty=1.05 等变换，
        # 否则实际采样概率与 GRPO 所评分的温度缩放 logits 不一致。
        generation_config = GenerationConfig(max_new_tokens=maximum, do_sample=sample, use_cache=True,
                      pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                      bos_token_id=self.tokenizer.bos_token_id, repetition_penalty=1.0,
                      temperature=generation['temperature'] if sample else 1.0,
                      top_p=generation['top_p'] if sample else 1.0,
                      top_k=max(0,generation.get('top_k',0)) if sample else 0)
        kwargs = {'generation_config': generation_config}
        if stage != 'ssr':
            stop = StopAfterStage(self.tokenizer,len(ids),stage) if stage in STAGE_TAGS else StopAfterJson(self.tokenizer,len(ids))
            kwargs['stopping_criteria'] = StoppingCriteriaList([stop])
        self.model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            result = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **kwargs)
        tokens = result[0, len(ids):].tolist()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        ended = bool(tokens and tokens[-1] == self.tokenizer.eos_token_id) or (stage != 'ssr' and segment_complete(text,stage))
        return Segment(ids, tokens, text, stage, time.perf_counter() - start, len(tokens) == maximum and not ended)

