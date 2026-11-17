import argparse
import json
from collections import Counter
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
    model = PeftModel.from_pretrained(model, a.adapter, autocast_adapter_dtype=False).to(torch.bfloat16).merge_and_unload()
    dtypes = Counter(str(p.dtype) for p in model.parameters())
    if any(p.dtype != torch.bfloat16 for p in model.parameters() if p.is_floating_point()):
        raise RuntimeError('Merged model must contain only BF16 floating parameters')
    model.save_pretrained(a.output, safe_serialization=True, max_shard_size='5GB')
    AutoTokenizer.from_pretrained(base).save_pretrained(a.output)
    (out / 'dtype_report.json').write_text(json.dumps(dict(all_model_parameters_bf16=True, parameter_dtypes=dict(dtypes), adapter=str(Path(a.adapter).resolve())), indent=2) + '\n')
    print('Merged', a.adapter, 'into', a.output)
if __name__ == '__main__':
    main()
