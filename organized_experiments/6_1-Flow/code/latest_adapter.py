import sys
from pathlib import Path
r=Path(sys.argv[1]);checkpoints=sorted(r.glob('global_step_*'),key=lambda p:int(p.name.rsplit('_',1)[1]))
if not checkpoints:raise SystemExit('No completed training checkpoint')
p=checkpoints[-1]/'actor/lora_adapter'
if not (p/'adapter_model.safetensors').is_file():raise SystemExit('Latest checkpoint has no exported LoRA adapter: '+str(p))
print(p)
