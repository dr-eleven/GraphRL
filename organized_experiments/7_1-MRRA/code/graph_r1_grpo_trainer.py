"""Pure-text Graph-R1 GRPO trainer built on TRL."""

from __future__ import annotations

import functools
import inspect
import os
from collections import defaultdict
from copy import deepcopy
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from accelerate.utils import gather_object, is_peft_model
from packaging.version import Version
import transformers.utils.hub as _transformers_hub
from transformers.trainer import (
    PREFIX_CHECKPOINT_DIR,
    Trainer,
    get_parameter_names,
    is_sagemaker_mp_enabled,
    logger,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

try:
    from graph_r1_train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
except Exception:  # pragma: no cover - fallback for standalone imports
    get_peft_state_maybe_zero_3 = None
    get_peft_state_non_lora_maybe_zero_3 = None

try:
    from rewards.graph_r1_reward_funcs import compute_score as _compute_score
except Exception:  # pragma: no cover
    _compute_score = None

if not hasattr(_transformers_hub, "TRANSFORMERS_CACHE"):
    _transformers_hub.TRANSFORMERS_CACHE = getattr(_transformers_hub, "HF_HUB_CACHE", None) or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub"
    )

import trl.import_utils as trl_import_utils


def _normalize_trl_optional_flags():
    for name in dir(trl_import_utils):
        if not (name.startswith("_") and name.endswith("_available")):
            continue
        value = getattr(trl_import_utils, name)
        if isinstance(value, tuple):
            setattr(trl_import_utils, name, value[0])


_normalize_trl_optional_flags()

from trl import GRPOTrainer
from trl.data_utils import is_conversational
from trl.trainer.utils import pad


def _identity_collator(features):
    return features


def _iter_generate_models(model):
    seen = set()
    stack = [model]
    while stack:
        candidate = stack.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        if hasattr(candidate, "module"):
            stack.append(candidate.module)
        if is_peft_model(candidate):
            try:
                stack.append(candidate.get_base_model())
            except Exception:
                pass
        base_model = getattr(candidate, "base_model", None)
        if base_model is not None:
            stack.append(base_model)


def _liger_exposes_mm_token_type_ids() -> bool:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return False
    try:
        return Version(version("liger_kernel")) >= Version("0.8.0")
    except PackageNotFoundError:
        return False


def _ensure_mm_token_type_ids_generate_compat(model):
    if _liger_exposes_mm_token_type_ids():
        return
    for candidate in _iter_generate_models(model):
        config = getattr(candidate, "config", None)
        model_type = getattr(config, "model_type", None)
        if model_type not in {"qwen2_vl", "qwen2_5_vl", "qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe"}:
            continue
        try:
            forward_sig = inspect.signature(candidate.forward)
        except (TypeError, ValueError):
            continue
        if "mm_token_type_ids" in forward_sig.parameters:
            continue
        original_forward = candidate.forward

        def forward_with_mm_token_type_ids(self, *args, mm_token_type_ids=None, **kwargs):
            if mm_token_type_ids is not None:
                kwargs["mm_token_type_ids"] = mm_token_type_ids
            return original_forward(*args, **kwargs)

        candidate.forward = MethodType(forward_with_mm_token_type_ids, candidate)


def _patch_prepare_deepspeed_for_ref_model():
    import trl.trainer.grpo_trainer as trl_grpo

    original = trl_grpo.prepare_deepspeed

    def patched_prepare_deepspeed(model, accelerator):
        import deepspeed

        deepspeed_plugin = accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        stage = config_kwargs["zero_optimization"]["stage"]

        if stage == 3:
            config_kwargs["zero_optimization"]["offload_param"] = {"device": "none"}
            config_kwargs["zero_optimization"]["offload_optimizer"] = {"device": "none"}
            config_kwargs.pop("optimizer", None)
            if model is not None:
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )
        else:
            config_kwargs["zero_optimization"]["stage"] = 0

        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    trl_grpo.prepare_deepspeed = patched_prepare_deepspeed
    return original


def _restore_prepare_deepspeed(original):
    import trl.trainer.grpo_trainer as trl_grpo

    trl_grpo.prepare_deepspeed = original


class GraphR1GRPOTrainer(GRPOTrainer):
    """Pure-text Graph-R1 GRPO trainer."""

    def __init__(self, *args, **kwargs):
        _original_prepare = _patch_prepare_deepspeed_for_ref_model()
        try:
            super().__init__(*args, **kwargs)
        finally:
            _restore_prepare_deepspeed(_original_prepare)

        self.data_collator = _identity_collator
        _ensure_mm_token_type_ids_generate_compat(self.model)
        _ensure_mm_token_type_ids_generate_compat(getattr(self, "model_wrapped", None))
        _ensure_mm_token_type_ids_generate_compat(getattr(self, "ref_model", None))
        self._apply_liger_grpo_loss_type_override()

        if self.args.gradient_checkpointing:
            gck_kwargs = getattr(self.args, "gradient_checkpointing_kwargs", None) or {}
            gck_kwargs.setdefault("use_reentrant", True)
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gck_kwargs)
            for module in self.model.modules():
                if hasattr(module, "gradient_checkpointing_func"):
                    module.gradient_checkpointing_func = functools.partial(
                        torch.utils.checkpoint.checkpoint, **gck_kwargs
                    )
                if hasattr(module, "_gradient_checkpointing_func"):
                    module._gradient_checkpointing_func = functools.partial(
                        torch.utils.checkpoint.checkpoint, **gck_kwargs
                    )

    def _apply_liger_grpo_loss_type_override(self) -> None:
        desired = getattr(self.args, "liger_grpo_loss_type", None)
        if not desired:
            return
        liger_grpo_loss = getattr(self, "liger_grpo_loss", None)
        if liger_grpo_loss is None or not hasattr(liger_grpo_loss, "loss_type"):
            return
        previous = liger_grpo_loss.loss_type
        liger_grpo_loss.loss_type = desired
        if getattr(self, "accelerator", None) is None or self.accelerator.is_main_process:
            print(f"[GraphR1GRPOTrainer] liger_grpo_loss.loss_type: {previous!r} -> {desired!r}")

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "assistant", "ground_truth", "data_source", "task", "split"]

    def _generate_single_turn(self, prompts: list):
        from contextlib import nullcontext
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from trl.models.utils import unwrap_model_for_generation

        device = self.accelerator.device
        processor_kwargs = {
            "text": prompts,
            "return_tensors": "pt",
            "padding": True,
            "padding_side": "left",
            "max_length": self.max_prompt_length,
            "truncation": True,
            "add_special_tokens": False,
        }
        generate_inputs = self.processing_class(**processor_kwargs)
        generate_inputs = Trainer._prepare_inputs(self, generate_inputs)

        with (
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            use_cache = getattr(unwrapped_model.config, "use_cache", None)
            if use_cache is not None:
                unwrapped_model.config.use_cache = True
            try:
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs,
                    generation_config=self.generation_config,
                    disable_compile=True,
                    use_cache=True,
                )
            finally:
                if use_cache is not None:
                    unwrapped_model.config.use_cache = use_cache

        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
        return prompt_ids, completion_ids, None, {}

    def _generate(self, prompts: list):
        if self.use_vllm:
            return super()._generate(prompts)

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts)

        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        local_completion_tokens = completion_lengths.sum()
        num_items_in_batch = self.accelerator.gather(local_completion_tokens).sum().clamp(min=1)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor(
            [(ids[-1] if ids else self.eos_token_id) not in eos_and_pad for ids in completion_ids],
            device=device,
        )
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) > 0:
            self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
            self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())
        else:
            self._metrics[mode]["completions/mean_terminated_length"].append(float("nan"))
            self._metrics[mode]["completions/max_terminated_length"].append(float("nan"))

        return prompt_ids, completion_ids, num_items_in_batch, logprobs, extra_fields

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """Pure-text override that handles ``_identity_collator`` output.

        Mirrors the non-visual parts of ``QwenGRPOTrainer._generate_and_score_completions``
        so that list-of-dicts inputs are correctly unpacked before generation,
        reward calculation, and advantage computation.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # ── Normalise input format ──
        # With _identity_collator we receive list[dict]; TRL's default collator
        # would produce dict-of-lists.  Handle both defensively.
        if isinstance(inputs, dict):
            if "prompt" in inputs:
                bsz = len(inputs["prompt"])
                inputs = [
                    {k: (v[i] if v is not None else None) for k, v in inputs.items()}
                    for i in range(bsz)
                ]
            else:
                raise ValueError(
                    f"Received pre-processed inputs with keys {list(inputs.keys())}. "
                    f"Expected raw inputs with 'prompt' key. Ensure data_collator=identity_collator."
                )
        elif not isinstance(inputs, list):
            raise TypeError(
                f"Expected inputs to be list[dict] or dict, got {type(inputs).__name__}."
            )

        prompts = [x["prompt"] for x in inputs]

        # ── Generate ──
        prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
            self._generate(prompts)
        )

        # ── Pad tensors ──
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # ── Mask truncated completions ──
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor(
                [(ids[-1] if ids else self.eos_token_id) not in eos_and_pad for ids in completion_ids_list],
                device=device,
            )
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
        num_items_in_batch = self.accelerator.gather(completion_mask.sum()).sum().clamp(min=1)

        # ── Concatenate ──
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        # ── Forward kwargs (pure text — no pixel_values etc.) ──
        forward_kwargs: dict[str, Any] = {}

        # ── Old / ref per-token logps ──
        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    **forward_kwargs,
                )
            else:
                old_per_token_logps = None

            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        **forward_kwargs,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            **forward_kwargs,
                        )
            else:
                ref_per_token_logps = None

        # ── Decode ──
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # ── Merge extra_fields into inputs ──
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # ── Rewards ──
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # ── Advantages (GRPO group normalisation) ──
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid scale_rewards: {self.scale_rewards}. Must be 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        # ── Metrics ──
        for i, reward_func_name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(
                torch.nanmean(rewards_per_func[:, i]).item()
            )
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # ── Logs ──
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        # ── Build output ──
        output: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

        return output

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        return self.optimizer

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # ── Descriptive symlink (safe — doesn't break HF/TRL internals) ──
        self._add_descriptive_checkpoint_link(run_dir, checkpoint_folder)

        if not self.args.lora_enable or get_peft_state_non_lora_maybe_zero_3 is None:
            return

        non_lora = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=True)

        if self.args.should_save:
            torch.save(non_lora, os.path.join(output_dir, "non_lora_state_dict.bin"))
            self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))

    def _add_descriptive_checkpoint_link(self, run_dir: str, checkpoint_folder: str) -> None:
        """Create a human-readable symlink next to the standard checkpoint dir."""
        try:
            label_parts = []

            if getattr(self.args, "lora_enable", False):
                r = getattr(self.args, "lora_rank", 0)
                a = getattr(self.args, "lora_alpha", 0)
                label_parts.append(f"lora_r{r}_a{a}")

            lr = getattr(self.args, "learning_rate", None)
            if lr is not None:
                lr_str = f"{float(lr):.0e}".replace("e-0", "e-")
                label_parts.append(f"lr{lr_str}")

            bs = getattr(self.args, "per_device_train_batch_size", None)
            if bs is not None:
                label_parts.append(f"bs{int(bs)}")

            ngen = getattr(self.args, "num_generations", None)
            if ngen is not None:
                label_parts.append(f"n{int(ngen)}")

            label = "ckpt-" + "_".join(label_parts) + f"-step{self.state.global_step}"
            link_path = os.path.join(run_dir, label)

            if not os.path.lexists(link_path):
                os.symlink(checkpoint_folder, link_path, target_is_directory=True)
        except Exception:
            pass


