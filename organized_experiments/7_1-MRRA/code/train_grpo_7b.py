"""7B 入口：沿用 MRRA/Graph-R1/TRL，保留生成前后的 checkpointing 配置。"""
import train_grpo as base


def restore_gradient_checkpointing(trainer):
    if trainer.args.gradient_checkpointing:
        # TRL unwrap_model_for_generation 会用默认参数重新启用 checkpointing。
        # 对 ZeRO-3 必须恢复显式 use_reentrant=True，避免重算时读到已分片的空权重。
        trainer.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=dict(trainer.args.gradient_checkpointing_kwargs or {}))


class SevenBThinkWithGraphTrainer(base.ThinkWithGraphTrainer):
    def _generate_batch(self, prompt_ids, stage):
        try:
            return super()._generate_batch(prompt_ids, stage)
        finally:
            restore_gradient_checkpointing(self)


if __name__ == '__main__':
    # 复用相同 HfArgumentParser、数据、模型、LoRA、callback 和 Trainer.train 入口。
    base.ThinkWithGraphTrainer = SevenBThinkWithGraphTrainer
    base.train()
