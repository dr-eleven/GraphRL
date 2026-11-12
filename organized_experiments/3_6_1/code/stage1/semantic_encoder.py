"""Lazy, frozen BGE text embeddings for optional reward shaping."""
from functools import lru_cache
from contextlib import contextmanager


@contextmanager
def _standalone_model_loading():
    """Keep this frozen inference model out of the training model's ZeRO-3 init."""
    from transformers.integrations import deepspeed
    config_ref = getattr(deepspeed, '_hf_deepspeed_config_weak_ref', None)
    config = config_ref() if config_ref is not None else None
    if config is None or not config.is_zero3():
        yield
        return
    deepspeed.unset_hf_deepspeed_config()
    try:
        yield
    finally:
        deepspeed.set_hf_deepspeed_config(config)


class FrozenTextEncoder:
    def __init__(self, model_path: str, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        with _standalone_model_loading():
            self.model = AutoModel.from_pretrained(
                model_path, local_files_only=True, dtype=torch.float32).to(device).eval()
        self.model.requires_grad_(False)
        if any(hasattr(p, 'ds_id') for p in self.model.parameters()):
            raise RuntimeError('Frozen semantic encoder must have complete standalone parameters')
        embedding = self.model.get_input_embeddings()
        if embedding.weight.ndim != 2 or embedding.weight.shape[0] != embedding.num_embeddings:
            raise RuntimeError('Frozen semantic encoder has an incomplete embedding table')
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
