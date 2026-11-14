"""Public, finite budgets. Defaults implement Binary_reward.md, not tuned results."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class RolloutBudget:
    max_model_turns: int = 10
    max_new_tokens_per_turn: int = 2048
    max_total_model_tokens: int = 12288
    max_completion_tokens: int = 16384
    max_initial_prompt_tokens: int = 8192
    max_actions_per_call: int = 64
    tool_timeout_sec: int = 10
    worker_watchdog_sec: int = 900
    infra_retries: int = 2

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or value < (0 if key == 'infra_retries' else 1):
                raise ValueError(f'Invalid positive integer budget: {key}')
