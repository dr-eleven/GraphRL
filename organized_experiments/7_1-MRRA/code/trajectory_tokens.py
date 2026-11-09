"""多段轨迹的真实 token 拼接；环境可见性与策略 loss mask 分开。"""
from executor_protocol import stage_messages


def render_messages(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# 保留第一段实际采样的原始 token，不用 decode→encode 替代学习目标。
# 只编码新增环境消息；若第一段自然生成 EOS，避免重复添加同一聊天结束标记。
def feedback_tokens(tokenizer, messages, first, execution, next_stage):
    feedback = stage_messages(messages, first.text, execution, next_stage)
    rendered = render_messages(tokenizer, feedback)
    prefix = render_messages(tokenizer, messages) + first.text
    if not rendered.startswith(prefix):
        raise ValueError('Chat template must preserve the initial prompt and assistant decision prefix')
    suffix = tokenizer(rendered[len(prefix):], add_special_tokens=False)['input_ids']
    if first.completion_ids[-1:] == [tokenizer.eos_token_id] and suffix[:1] == [tokenizer.eos_token_id]:
        suffix = suffix[1:]
    ids = first.prompt_ids + first.completion_ids + suffix
    return feedback, suffix, ids


def pack_segments(segments, suffixes):
    if len(suffixes) != len(segments)-1:
        raise ValueError('Each transition requires exactly one environment suffix')
    first = segments[0]
    tokens = list(first.completion_ids)
    mask = [1]*len(tokens)
    for segment,suffix in zip(segments[1:],suffixes):
        expected = first.prompt_ids+tokens+suffix
        if segment.prompt_ids != expected:
            raise ValueError('Generation did not read the exact preceding tokens and actual environment feedback')
        tokens += suffix+segment.completion_ids
        mask += [0]*len(suffix)+[1]*len(segment.completion_ids)
    return tokens, mask
