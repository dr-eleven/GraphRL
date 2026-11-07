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
        ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        vectors, weights = [], []
        # Encode the full text; fixed non-overlapping chunks only beyond model capacity.
        width = min(8192, self.tokenizer.model_max_length)-self.tokenizer.num_special_tokens_to_add(False)
        for start in range(0, max(1,len(ids)), width):
            chunk = ids[start:start+width]
            tokens = [self.tokenizer.cls_token_id, *chunk, self.tokenizer.sep_token_id]
            inputs = {'input_ids':self.torch.tensor([tokens],device=self.device),
                      'attention_mask':self.torch.ones((1,len(tokens)),dtype=self.torch.long,device=self.device)}
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                vector = hidden.float().mean(dim=1)[0].cpu()
            vectors.append(vector); weights.append(max(1,len(chunk)))
        vector = sum(v*w for v,w in zip(vectors,weights))/sum(weights)
        return self.torch.nn.functional.normalize(vector, dim=-1)

    def similarity(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        return max(0.0, min(1.0, float(self.torch.dot(self.encode(a), self.encode(b)))))
