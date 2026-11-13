# Persistent prompt-node BGE-M3 cache

`bge_m3_prompt_embeddings.sqlite` stores the BGE-M3 text embeddings used to
map prompt-subgraph nodes to the organized complete-graph catalogue during
fullgraph BadGraph attacks.

- Cache key: SHA-1 of the exact UTF-8 node text.
- Stored vector: 1024-dimensional BGE-M3 embedding, serialized as float16.
- Scope: prompt-node texts from the 600-row GraphSSR/GOFA test input and any
  additional exact texts encountered by later reruns.
- This cache is deliberately outside individual experiment output folders so
  budget or sample-subset reruns can reuse it.

The runner's default `--embedding-cache` now points here.  The catalogue node
features remain in the sibling `data/test_data/<dataset>/node_features.pt`
files and are not duplicated in this cache.

## Rendered 600-row evaluation corpus

This directory also holds the complete rendered prompts used by the full-graph
evaluation protocol:

- `clean/clean_qwen_rendered.json`: the shared 600 clean prompts;
- `badgraph/budget{1..5}/attacked_qwen_rendered.json`: the 600 paired
  BadGraph prompts at each budget;
- `injection/budget{1..5}/attacked_qwen_rendered.json`: the 600 paired
  injection prompts at each budget.

Every JSON file has the same ordered 600 sample IDs as the clean corpus.
Each sibling `indices_{0,1,2}.txt` partitions indices 0--599 for the three-GPU
evaluator.  The cached embeddings above are auxiliary data and are not a
replacement for these prompt/label records.
