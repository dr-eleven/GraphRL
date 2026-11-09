"""将本项目 LoRA 合并为独立推理权重，不覆盖已有输出。"""
import argparse
from pathlib import Path
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import resolve


def merge(adapter, output):
    adapter, output = resolve(adapter), resolve(output)
    if output.exists():
        raise FileExistsError(output)
    config = PeftConfig.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
        torch_dtype=torch.bfloat16, local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(output)
    AutoTokenizer.from_pretrained(adapter, local_files_only=True).save_pretrained(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    merge(args.adapter, args.output)
