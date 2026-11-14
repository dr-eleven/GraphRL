"""Only terminal correctness enters GRPO; no structure, action or cost rewards."""
from label_utils import _label_key


def terminal_reward(prediction, ground_truth):
    return float(prediction is not None and _label_key(prediction) == _label_key(ground_truth))
