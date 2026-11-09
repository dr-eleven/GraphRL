"""Lazy, frozen BGE text embeddings for optional reward shaping."""
from functools import lru_cache


class FrozenTextEncoder:
    def __init__(self, model_path: str, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device

    @lru_cache(maxsize=8192)
    def encode(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            vector = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.torch.nn.functional.normalize(vector.float(), dim=-1).cpu()[0]

    def similarity(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        return max(0.0, min(1.0, float(self.torch.dot(self.encode(a), self.encode(b)))))
