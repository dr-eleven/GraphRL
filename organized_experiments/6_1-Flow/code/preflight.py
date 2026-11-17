import bootstrap,json,os,importlib,importlib.metadata as metadata,hashlib
from pathlib import Path
cfg=json.loads(Path(os.environ.get('FLOW_SETTINGS',bootstrap.ROOT/'configs/settings.json')).read_text())
model_path=Path(cfg['model'])
if not model_path.is_absolute():model_path=bootstrap.ROOT.parents[1]/model_path
report={'versions':{x:metadata.version(x) for x in ['torch','transformers','peft','verl','vllm','ray','flask','safetensors','pyarrow']},'model_exists':model_path.is_dir()}
for mod in ['workflow','flow_transport','agentflow.verl.entrypoint','verl.workers.fsdp_workers','verl.workers.rollout.vllm_rollout.vllm_async_server']:
    m=importlib.import_module(mod);report[mod]=str(m.__file__)
assert report['versions']['verl']=='0.5.0'
assert report['model_exists']
# Record every changed upstream file, rather than describing a rewrite as unchanged.
manifest=json.loads((bootstrap.ROOT/'source_manifest.json').read_text());changed=[]
for entry in manifest['files']:
    file=bootstrap.ROOT/'upstream'/entry['path']
    actual=hashlib.sha256(file.read_bytes()).hexdigest()
    if actual!=entry['sha256']:changed.append(dict(path=entry['path'],source_sha256=entry['sha256'],actual_sha256=actual))
report['changed_upstream_files']=changed
(bootstrap.ROOT/'verification/preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
