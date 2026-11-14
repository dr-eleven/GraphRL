"""Qwen2.5/TRL 1.17.1 rollout: raw parsing, exact budgets, feedback and loss masks.

Overrides the whole generation entry, including turn zero. TRL still owns reward
normalization, policy/reference log probabilities and the GRPO loss.
"""
from copy import deepcopy
import json
from pathlib import Path
import time
import torch
from transformers import StoppingCriteria
from runtime import InfrastructureError, deadline


class CandidateTokenBudget(StoppingCriteria):
    def __init__(self, prompt_width, budgets):
        self.prompt_width, self.budgets = prompt_width, list(budgets)

    def __call__(self, input_ids, scores, **kwargs):
        return torch.tensor([input_ids.shape[1]-self.prompt_width >= n for n in self.budgets],
                            dtype=torch.bool,device=input_ids.device)


def feedback_suffix(tokenizer, message, previous_ids, eos_ids, add_generation_prompt=True):
    """Qwen tool/user framing, appended without re-tokenizing any model output.

    If generation ended without a message terminator, the controller closes the
    message in the EXTERNAL span (mask=0); it never fabricates a sampled EOS.
    """
    message_end = tokenizer.encode('<|im_end|>',add_special_tokens=False)[0]
    ended = bool(previous_ids) and previous_ids[-1] == message_end
    boundary = '' if ended else '<|im_end|>'
    content = message['content']
    if message['role'] == 'tool':
        content = '<tool_response>\n'+content+'\n</tool_response>'
    text = boundary+'\n<|im_start|>user\n'+content+'<|im_end|>'
    if add_generation_prompt:text += '\n<|im_start|>assistant\n'
    return tokenizer.encode(text,add_special_tokens=False)


class TokenLimitToolsMixin:
    def _generate(self, prompts):
        """Collect complete groups; retry infrastructure failures only, at fixed policy."""
        prompt_ids, images, multimodal = self._tokenize_prompts(deepcopy(prompts))
        if images or multimodal or self._is_vlm or self.use_vllm:
            raise ValueError('Binary graph rollout supports text-only Qwen HF generation')
        self._initial_prompt_lengths = [len(p) for p in prompt_ids]
        config = self.model.config
        context = config.max_position_embeddings
        if config.model_type != 'qwen2':
            raise ValueError('This raw-event controller requires the Qwen2.5 chat protocol')
        for marker in ('<|im_start|>','<|im_end|>'):
            if len(self._tokenizer.encode(marker,add_special_tokens=False)) != 1:
                raise ValueError('Tokenizer does not support the Qwen message delimiters')
        completion_ids = [[] for _ in prompts]
        completions = [[] for _ in prompts]
        masks = [[] for _ in prompts]
        pending = list(range(len(prompts)))
        b = self.data_args.rollout_budget(self.max_completion_length)
        retries = [0]*len(prompts)
        for attempt in range(b.infra_retries+1):
            failures = {}
            self._collect_candidates(pending,prompt_ids,completion_ids,completions,masks,context,failures)
            # Ranks finish their local candidate loops before this single collective.
            # Full G groups may cross ranks; gather source IDs, never resample by reward.
            status = torch.tensor([[int(e._sample['source_index']),int(i in failures)]
                                   for i,e in enumerate(self.environments)],device=self.accelerator.device)
            statuses = self.accelerator.gather(status).cpu().tolist()
            failed_sources = {source for source,failed in statuses if failed}
            if not failed_sources:
                break
            self._write_infra_audit(attempt,failed_sources,failures)
            if attempt == b.infra_retries:
                raise InfrastructureError(f'Complete groups failed after {attempt+1} collections: {sorted(failed_sources)}')
            pending = [i for i,e in enumerate(self.environments) if int(e._sample['source_index']) in failed_sources]
            for i in pending:
                env = self.environments[i]
                env.reset(env._sample,bypass_actions=env._bypass)
                completion_ids[i],completions[i],masks[i] = [],[],[]
                retries[i] += 1
        for i,env in enumerate(self.environments):
            env._infra_group_retries = retries[i]
            if len(completion_ids[i]) > b.max_completion_tokens:
                raise AssertionError('Rollout exceeded completion budget; never truncate committed history')
            if len(completion_ids[i]) != len(masks[i]) or sum(masks[i]) != env._model_tokens:
                raise AssertionError('Generated token/loss mask accounting mismatch')
        self._log_rollout_metrics(prompt_ids,masks)
        # Same nine-field interface as TRL _generate. HF sampling logps are computed downstream.
        return prompt_ids,completion_ids,masks,completions,None,None,{},None,[[] for _ in prompts]

    def _write_infra_audit(self, attempt, failed_sources, failures):
        path = Path(self.args.output_dir)
        path.mkdir(parents=True,exist_ok=True)
        payload = dict(attempt=attempt,policy_step=self.state.global_step,valid_for_training=False,
                       failed_sources=sorted(failed_sources),local_errors=failures,
                       discarded=[dict(source_index=e._sample['source_index'],raw_turns=e._raw_turns,
                                       errors=e._errors,phase=e._phase)
                                  for e in self.environments if int(e._sample['source_index']) in failed_sources])
        with (path/f'infra_rank{self.accelerator.process_index}.jsonl').open('a') as stream:
            stream.write(json.dumps(payload,ensure_ascii=False)+'\n')

    def _collect_candidates(self, pending, prompt_ids, completion_ids, completions, masks, context, failures):
        b = self.data_args.rollout_budget(self.max_completion_length)
        started = {i:time.monotonic() for i in pending}
        active = list(pending)
        while active:
            ready = []
            for i in active:
                env = self.environments[i]
                if time.monotonic()-started[i] > b.worker_watchdog_sec:
                    env._phase = 'ABORTED_INFRA'
                    failures[i] = 'candidate watchdog expired'
                    continue
                cap,feedback = env._capacity(len(prompt_ids[i]),len(completion_ids[i]),context)
                if feedback is not None:
                    self._append_feedback(i,feedback,prompt_ids,completion_ids,completions,masks,context)
                    cap,_ = env._capacity(len(prompt_ids[i]),len(completion_ids[i]),context)
                if cap > 0 and env._active:
                    env._next_cap = cap
                    ready.append(i)
            if not ready:
                break
            width = self.data_args.generation_micro_batch_size
            for start in range(0,len(ready),width):
                indices = ready[start:start+width]
                self._active_rollout_indices = indices
                inputs = [prompt_ids[i]+completion_ids[i] for i in indices]
                try:
                    remaining_time = min(b.worker_watchdog_sec-(time.monotonic()-started[i]) for i in indices)
                    if remaining_time <= 0:
                        raise InfrastructureError('candidate watchdog expired')
                    with deadline(remaining_time,'model generation'):
                        outputs,logps,sampling = self._generate_single_turn(inputs,None,{},num_generations=1)
                    if len(outputs) != len(indices):
                        raise InfrastructureError('Generation lost candidate alignment')
                    if any(len(ids)>self.environments[i]._next_cap for i,ids in zip(indices,outputs)):
                        raise InfrastructureError('Generation backend violated the requested token cap')
                except Exception as exc:
                    for i in indices:
                        self.environments[i]._phase = 'ABORTED_INFRA'
                        failures[i] = f'{type(exc).__name__}: {exc}'
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                for i,ids in zip(indices,outputs,strict=True):
                    env = self.environments[i]
                    # Every emitted token is retained before parsing, including invalid turns.
                    completion_ids[i].extend(ids)
                    masks[i].extend([1]*len(ids))
                    raw_ids = ids[:-1] if ids and ids[-1] in self.eos_token_ids else ids
                    raw = self._tokenizer.decode(raw_ids,skip_special_tokens=False)
                    def feedback_fits(message):
                        suffix = feedback_suffix(self._tokenizer,message,completion_ids[i],self.eos_token_ids,
                                                 add_generation_prompt=env._turns<b.max_model_turns)
                        used = len(completion_ids[i])+len(suffix)
                        return (used <= b.max_completion_tokens
                                and len(prompt_ids[i])+used <= context)
                    try:
                        assistant,feedback = env._consume(raw,ids,env._last_finish,feedback_fits)
                        completions[i].append(assistant)
                        if feedback is not None:
                            self._append_feedback(i,feedback,prompt_ids,completion_ids,completions,masks,context)
                    except Exception as exc:
                        env._phase = 'ABORTED_INFRA'
                        failures[i] = f'{type(exc).__name__}: {exc}'
            active = [i for i in ready if self.environments[i]._active and i not in failures]

    def _append_feedback(self, i, message, prompt_ids, completion_ids, completions, masks, context):
        env = self.environments[i]
        b = env._budget
        suffix = feedback_suffix(self._tokenizer,message,completion_ids[i],self.eos_token_ids,
                                 add_generation_prompt=env._active)
        available = min(b.max_completion_tokens-len(completion_ids[i]),
                        context-len(prompt_ids[i])-len(completion_ids[i]))
        if len(suffix) > available:
            env._finish('feedback_capacity_exhausted')
            env._events.append(dict(status='feedback_omitted',reason='absolute_budget',message=message))
            return
        completion_ids[i].extend(suffix)
        masks[i].extend([0]*len(suffix))
        env._feedback_tokens += len(suffix)
        completions[i].append(message)

    def _log_rollout_metrics(self, prompt_ids, masks):
        mode = 'train' if self.model.training else 'eval'
        device = self.accelerator.device
        lengths = self.accelerator.gather(torch.tensor([sum(m) for m in masks],device=device))
        prompts = self.accelerator.gather(torch.tensor([len(p) for p in prompt_ids],device=device))
        if mode == 'train':
            self.state.num_input_tokens_seen += int(lengths.sum()+prompts.sum())
        self._metrics[mode]['num_tokens'] = [self.state.num_input_tokens_seen]
        for name, value in [('mean',lengths.float().mean()),('min',lengths.min()),('max',lengths.max())]:
            self._metrics[mode][f'completions/{name}_length'].append(value.item())
