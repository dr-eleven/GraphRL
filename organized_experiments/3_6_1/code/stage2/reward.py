"""R=S+T. S is binary format validity; T is 0 / 0.1 / 1."""
import json
from label_utils import _label_key


def score_episode(sample, completion, *, terminated=True):
    from episode import parse_prediction_completion, message_text
    sample = json.loads(sample) if isinstance(sample, str) else sample
    errors = []
    raw = completion if isinstance(completion, str) else ''
    try:
        if not terminated:
            raise ValueError('Generation reached its token budget without EOS')
        if not isinstance(completion, str):
            if len(completion) != 1 or completion[0].get('role') != 'assistant' or completion[0].get('tool_calls'):
                raise ValueError('Exactly one prediction message is required')
            raw = message_text(completion[0])
        parsed = parse_prediction_completion(raw)
        valid = True
    except (ValueError, TypeError, KeyError) as exc:
        parsed, valid = {}, False
        errors.append(str(exc))
    predicted = _label_key(parsed.get('prediction', ''))
    correct = valid and bool(predicted) and predicted == _label_key(sample['ground_truth'])
    in_candidates = valid and bool(predicted) and predicted in {_label_key(v) for v in sample['label_space']}
    structure = float(valid)
    task = 1.0 if correct else 0.1 if in_candidates else 0.0
    metrics = dict(structure_reward=structure, task_reward=task, total=structure+task,
        prediction_correct=float(correct), prediction_in_candidates=float(in_candidates),
        format_valid=float(valid), prediction_format_valid=float(valid),
        token_limit_exceeded=float(not terminated))
    return structure+task, dict(prediction_raw=raw, errors=errors, metrics=metrics)
