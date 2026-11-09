"""A 使用 B+T；新 B 使用 B+T+F。隐藏参考不进入模型观察或执行器。"""
import re
import json
from episode import _edge_map
from label_utils import _label_key


def actual_corrupt(sample):
    a, b = sample['noisy_graph'], sample['clean_graph']
    return a['nodes'] != b['nodes'] or set(_edge_map(a)) != set(_edge_map(b))


# 答案独立提取：前段非法仍可得到 T，但多个 answer 字段或非字符串答案不能得分。
def extract_answer(text):
    matches = re.findall(r'"answer"\s*:\s*("(?:\\.|[^"\\])*")', str(text))
    if len(matches) != 1:
        return ''
    try:
        return json.loads(matches[0])
    except ValueError:
        return ''


def score_episode(sample, execution, answer, *, format_valid=None):
    dirty = actual_corrupt(sample)
    valid, has = execution['error'] is None, bool(execution['actions'])
    behavior = float(valid and (has if dirty else not has))
    task = float(_label_key(answer) == _label_key(sample['ground_truth']))
    values = {
        'behavior_reward': behavior, 'task_reward': task, 'total': behavior + task,
        'legal_execution': float(valid), 'illegal': float(not valid), 'has_actions': float(has),
        'actual_corrupt': float(dirty), 'correct_clean_noop': float(valid and not dirty and not has),
        'unnecessary_clean_edit': float(not dirty and has), 'attacked_noop': float(dirty and valid and not has)}
    if format_valid is not None:
        values['format_reward'] = float(bool(format_valid))
        values['total'] += values['format_reward']
    return values['total'], values
