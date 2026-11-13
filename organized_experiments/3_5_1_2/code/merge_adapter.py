import argparse
from pathlib import Path
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite merged model: {out}')
    base = PeftConfig.from_pretrained(a.adapter).base_model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cpu')
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(a.output, safe_serialization=True, max_shard_size='5GB')
    AutoTokenizer.from_pretrained(base).save_pretrained(a.output)
    print('Merged', a.adapter, 'into', a.output)
if __name__ == '__main__':
    main()
