"""TRL 1.17.1 native tool loop with candidate-local token-limit termination.

The loop below is adapted from trl.trainer.grpo_trainer (Apache-2.0).
Only candidate indexing and suppression of length-aborted calls are added;
tool execution, feedback tokens, masks and parsing retain upstream behavior.
"""
import asyncio
import torch
from transformers import StoppingCriteria
from trl import GRPOTrainer
from trl.chat_template_utils import parse_response
from trl.trainer.grpo_trainer import split_pixel_values_by_grid, unsplit_pixel_values_by_grid


class CandidateTokenBudget(StoppingCriteria):
    """Stop each row at its own remaining budget without stopping sibling rows."""
    def __init__(self, prompt_width, budgets):
        self.prompt_width = prompt_width
        self.budgets = list(budgets)

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids.shape[1] - self.prompt_width
        return torch.tensor([generated >= n for n in self.budgets],
                            dtype=torch.bool, device=input_ids.device)


class TokenLimitToolsMixin:
    def _suppress_limited_calls(self, completions, completion_ids):
        for i, messages in enumerate(completions):
            if self.environments[i]._generation_limit_events:
                messages[-1] = {'role': 'assistant', 'content':
                    self._tokenizer.decode(completion_ids[i], skip_special_tokens=True)}

    def _tool_call_loop(
        self, prompts, prompt_ids, completion_ids, completions, logprobs, sampling_mask, images, multimodal_fields
    ):
        # Tool execution loop: execute tools, then regenerate completions with tool results appended to the prompt
        self._suppress_limited_calls(completions, completion_ids)
        tool_calls = [completion[0].get("tool_calls")
                      if self.environments[i]._validate_turn(completion[0]) else None
                      for i, completion in enumerate(completions)]
        idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        tool_mask = [[1] * len(ids) for ids in completion_ids]  # 0 for tool result tokens, 1 elsewhere
        # Collect images from multimodal tool responses for the forward pass
        tool_images = [[] for _ in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0

        while idxs_with_tool and iteration_num < self.max_tool_calling_iterations:
            prompt_completion_tools = [prompts[i] for i in idxs_with_tool]  # select only prompts that need tool calls
            # Snapshot state so we can rollback tool results that would exceed max_completion_length
            completions_len_before = [len(completions[i]) for i in idxs_with_tool]
            tool_images_len_before = [len(tool_images[i]) for i in idxs_with_tool]
            prompts_len_before = [len(prompts[i]) for i in idxs_with_tool]

            # Call the tools, and build the new prompt for generation
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call_list = tool_calls[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                sync_tool_dict = self._sync_tool_dicts[idx_with_tool]
                async_tool_dict = self._async_tool_dicts[idx_with_tool]
                # Append the last assistant message (which triggered tool_calls) to the prompt
                prompt_completion_tool.append(completions[idx_with_tool][-1])
                async_coros = []
                tool_call_results = []
                for tool_call in tool_call_list:
                    tool_call_count += 1
                    if tool_call["type"] == "function":
                        function = tool_call["function"]
                        name = function["name"]
                        try:
                            if name in sync_tool_dict:
                                tool_call_results.append((name, sync_tool_dict[name](**function["arguments"])))
                            elif name in async_tool_dict:
                                async_coros.append((name, async_tool_dict[name](**function["arguments"])))
                            else:
                                raise ValueError(f"Tool {name} not found.")
                        except Exception as e:
                            tool_failure_count += 1
                            result = {"error": str(e)}
                            tool_call_results.append((name, result))
                    else:
                        tool_failure_count += 1
                        name = tool_call.get("name", "unknown")
                        tool_call_results.append((name, {"error": f"Unsupported tool call type: {tool_call['type']}"}))

                if async_coros:

                    async def _run_async_tools(async_coros):
                        coros = [coro for _, coro in async_coros]
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        return [(name, result) for (name, _), result in zip(async_coros, results, strict=False)]

                    async_results = asyncio.run_coroutine_threadsafe(
                        _run_async_tools(async_coros), self.async_loop
                    ).result()

                    for name, result in async_results:
                        if isinstance(result, Exception):
                            tool_failure_count += 1
                            tool_call_results.append((name, {"error": str(result)}))
                        else:
                            tool_call_results.append((name, result))

                for name, result in tool_call_results:
                    # Support multimodal tool responses: if the tool returns a list of content blocks
                    # (e.g., [{"type": "image", "image": ...}, {"type": "text", "text": "..."}]),
                    # pass them through directly so _tokenize_prompts can extract images for VLMs.
                    content = result if isinstance(result, list) else str(result)
                    tool_message = {"role": "tool", "name": name, "content": content}
                    # Collect images from multimodal tool responses
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image":
                                tool_images[idx_with_tool].append(part["image"])
                    prompt_completion_tool.append(tool_message)
                    completions[idx_with_tool].append(tool_message)

            # Build token IDs by concatenation: prompt + completion + tool_suffix.
            prompt_completion_tool_ids = []
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                # Extract trailing tool messages from completions
                tool_messages = []
                for message in reversed(completions[idx_with_tool]):
                    if message["role"] == "tool":
                        tool_messages.insert(0, message)
                    else:
                        break
                suffix_ids = self._get_tool_suffix_ids(tool_messages)
                prompt_completion_tool_ids.append(
                    prompt_ids[idx_with_tool] + completion_ids[idx_with_tool] + suffix_ids
                )

            # Drop tool results whose addition would push the sequence past max_completion_length (the completion
            # budget) or past the backend context ceiling (vLLM and transformers will error out on inputs longer than
            # the model's max length). The sample exits the loop with its completion as-is, and the tool
            # messages/images appended this iteration are rolled back so completions and tool_images stay consistent
            # with completion_ids.
            if self.use_vllm and self.vllm_mode == "colocate":
                max_model_len = self.vllm_generation.llm.llm_engine.model_config.max_model_len
            else:
                config = self.model.config.text_config if self._is_vlm else self.model.config
                max_model_len = config.max_position_embeddings
            overlong = [
                len(pct) - len(prompt_ids[i]) > self.max_completion_length or len(pct) >= max_model_len
                for i, pct in zip(idxs_with_tool, prompt_completion_tool_ids, strict=True)
            ]
            for idx in range(len(idxs_with_tool)):
                if overlong[idx]:
                    idx_with_tool = idxs_with_tool[idx]
                    self.environments[idx_with_tool]._abort_format(
                        'Tool feedback exceeds remaining completion or model context budget')
                    del completions[idx_with_tool][completions_len_before[idx] :]
                    del tool_images[idx_with_tool][tool_images_len_before[idx] :]
                    del prompts[idx_with_tool][prompts_len_before[idx] :]
            # Keep only non-overlong items for further processing
            idxs_with_tool = [idx for idx, o in zip(idxs_with_tool, overlong, strict=True) if not o]
            prompt_completion_tool_ids = [
                pct for pct, o in zip(prompt_completion_tool_ids, overlong, strict=True) if not o
            ]
            if not idxs_with_tool:
                break  # all overlong, exit tool loop

            # Filter images and multimodal fields to match the current subset (index into full batch).
            # Merge tool response images so the model can see visual feedback during generation.
            merged_images = images
            if any(imgs for imgs in tool_images):
                if merged_images is None:
                    merged_images = [imgs if imgs else None for imgs in tool_images]
                else:
                    merged_images = [
                        (existing or []) + new for existing, new in zip(merged_images, tool_images, strict=True)
                    ]
            loop_images = [merged_images[i] for i in idxs_with_tool] if merged_images else None
            if self._is_vlm and self.tools and any(imgs for imgs in tool_images):
                flat_loop_images = [img for img_list in loop_images if img_list for img in img_list]
                if flat_loop_images:
                    image_inputs = self.processing_class.image_processor(images=flat_loop_images, return_tensors="pt")
                    image_inputs = super(GRPOTrainer, self)._prepare_inputs(image_inputs)
                    loop_multimodal_fields = dict(image_inputs)
                else:
                    loop_multimodal_fields = {}
            elif multimodal_fields:
                if "num_images" not in multimodal_fields and images is not None:
                    multimodal_fields = {
                        **multimodal_fields,
                        "num_images": [len(img_list) if img_list else 0 for img_list in images],
                    }
                split_fields = split_pixel_values_by_grid(multimodal_fields)
                loop_multimodal_fields = {}
                for k, v in split_fields.items():
                    selected = [v[i] for i in idxs_with_tool]
                    # Per-token type-id fields may arrive as tensors (e.g. via the padding workaround);
                    # convert them to lists so they follow the same left-padding path as list-valued
                    # per-token fields below, keeping them aligned with the left-padded input_ids.
                    if k in ("token_type_ids", "mm_token_type_ids") and isinstance(selected[0], torch.Tensor):
                        selected = [s.tolist() for s in selected]
                    # Per-token fields (e.g. token_type_ids) need zero-padding to match extended prompt length
                    if isinstance(selected[0], list):
                        selected = [
                            s + [0] * (len(pct) - len(s))
                            for s, pct in zip(selected, prompt_completion_tool_ids, strict=True)
                        ]
                    elif isinstance(v, torch.Tensor):
                        selected = torch.stack(selected)
                    loop_multimodal_fields[k] = selected
                loop_multimodal_fields = unsplit_pixel_values_by_grid(loop_multimodal_fields)
                loop_multimodal_fields.pop("num_images", None)
                if "pixel_values" in loop_multimodal_fields and loop_multimodal_fields["pixel_values"].numel() == 0:
                    for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
                        loop_multimodal_fields.pop(key, None)
            else:
                loop_multimodal_fields = {}

            # Generate new completions after tool execution (using concatenated IDs, no re-tokenization)
            self._active_rollout_indices = list(idxs_with_tool)
            post_tool_ids, post_tool_logprobs, post_tool_sampling_mask = self._generate_single_turn(
                prompt_completion_tool_ids,
                loop_images,
                loop_multimodal_fields,
                num_generations=1,  # each sample has its own history, so generate one completion per sample
                has_tool_images=any(imgs for imgs in tool_images),
            )

            # Truncate so that pct[len(prompt_ids[idx]) :] + post_tool does not exceed max_completion_length.
            # The pre-regen check guarantees len(completion_tool_ids) <= max_completion_length, so any
            # excess can only come from post_tool_ids. post_tool_ids is model-generated text and never
            # contains image tokens, so a plain slice is safe.
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                completion_tool_length = len(prompt_completion_tool_ids[idx]) - len(prompt_ids[idx_with_tool])
                excess_length = completion_tool_length + len(post_tool_ids[idx]) - self.max_completion_length
                if excess_length > 0:
                    new_len = len(post_tool_ids[idx]) - excess_length
                    post_tool_ids[idx] = post_tool_ids[idx][:new_len]
                    if logprobs is not None:
                        post_tool_logprobs[idx] = post_tool_logprobs[idx][:new_len]
                    if sampling_mask is not None:
                        post_tool_sampling_mask[idx] = post_tool_sampling_mask[idx][:new_len]

            # Update tool_mask: the tool result should be 0 and the post-tool 1
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool_length = len(prompt_completion_tool_ids[idx])
                prompt_length = len(prompt_ids[idx_with_tool])
                completion_length = len(completion_ids[idx_with_tool])
                post_tool_length = len(post_tool_ids[idx])
                tool_length = prompt_completion_tool_length - prompt_length - completion_length
                tool_mask[idx_with_tool] += [0] * tool_length + [1] * post_tool_length
                if logprobs is not None:
                    logprobs[idx_with_tool] += [0.0] * tool_length + post_tool_logprobs[idx]
                if sampling_mask is not None:
                    # Tool tokens weren't sampled: each is its own candidate set, which keeps its log-prob finite
                    tool_ids = prompt_completion_tool_ids[idx][prompt_length + completion_length :]
                    sampling_mask[idx_with_tool] += [[t] for t in tool_ids] + post_tool_sampling_mask[idx]

            # Update completion_ids with the new completions (after tool execution)
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_length = len(prompt_ids[idx_with_tool])
                pct = prompt_completion_tool_ids[idx]  # = prompt-completion-tool
                completion_ids[idx_with_tool] = pct[prompt_length:] + post_tool_ids[idx]

            # Decode post-tool completions, each with the tools its prompt was rendered with
            if self.environment_factories is not None:
                per_prompt_tools = [self._env_tools[self._batch_environments[i]] for i in idxs_with_tool]
            else:
                per_prompt_tools = [self.tools] * len(idxs_with_tool)
            post_tool_completions = [
                parse_response(
                    self._tokenizer, ids, prefix=prompt_completion_tool_ids[idx], tools=per_prompt_tools[idx]
                )
                if ids
                else {}
                for idx, ids in enumerate(post_tool_ids)
            ]

            # A length-aborted turn must never dispatch a tool, even if its JSON parses.
            for local, original in enumerate(idxs_with_tool):
                if self.environments[original]._generation_limit_events:
                    post_tool_completions[local] = {
                        'role': 'assistant',
                        'content': self._tokenizer.decode(post_tool_ids[local], skip_special_tokens=True),
                    }

            # Add post-tool completions to the existing completions
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if post_tool_completions[idx]:  # {} if post-tool completions completely truncated
                    completions[idx_with_tool].append(post_tool_completions[idx])

            # Check for further tool calls
            tool_calls = [completion.get("tool_calls")
                          if self.environments[original]._validate_turn(completion) else None
                          for original, completion in zip(idxs_with_tool, post_tool_completions, strict=True)]
            idxs_with_tool = [idx for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True) if tool_call]
            tool_calls = [tool_call for tool_call in tool_calls if tool_call]
            iteration_num += 1

        # Terminal feedback is external text (mask=0), never another generation request.
        # If no context budget remains, retain the full reason in the rollout audit.
        for i, env in enumerate(self.environments):
            if not env._format_errors:
                continue
            feedback = {'role':'tool', 'name':'format_error', 'content':env._format_feedback()}
            suffix = self._get_tool_suffix_ids([feedback])
            config = self.model.config.text_config if self._is_vlm else self.model.config
            if (len(completion_ids[i]) + len(suffix) <= self.max_completion_length
                    and len(prompt_ids[i]) + len(completion_ids[i]) + len(suffix) < config.max_position_embeddings):
                completions[i].append(feedback)
                completion_ids[i].extend(suffix)
                tool_mask[i].extend([0] * len(suffix))
                if logprobs is not None:
                    logprobs[i].extend([0.0] * len(suffix))
                if sampling_mask is not None:
                    sampling_mask[i].extend([[token] for token in suffix])

        return (
            tool_mask,
            completions,
            completion_ids,
            logprobs,
            sampling_mask,
            tool_call_count,
            tool_failure_count,
            tool_images,
        )
