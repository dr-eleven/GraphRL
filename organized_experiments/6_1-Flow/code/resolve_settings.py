"""Resolve portable settings once per run; protect a run from configuration drift."""
import argparse,json,os
from pathlib import Path


def resolve(source,repo,environ):
    cfg=json.loads(Path(source).read_text())
    cfg['model']=environ.get('TRAIN_MODEL',cfg['model'])
    cfg['train_source']=environ.get('TRAIN_EPISODES',cfg['train_source'])
    overrides={'test_data_600_oneclass':'TEST_ONECLASS_EPISODES',
               'test_data_600_notoneclass':'TEST_NOTONECLASS_EPISODES'}
    for key,variable in overrides.items():
        cfg['test_sources'][key]=environ.get(variable,cfg['test_sources'][key])
    def absolute(value):
        path=Path(value).expanduser()
        return str((path if path.is_absolute() else repo/path).resolve())
    for key in ['model','train_source']:cfg[key]=absolute(cfg[key])
    cfg['test_sources']={key:absolute(value) for key,value in cfg['test_sources'].items()}
    return cfg


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    cfg=resolve(a.source,repo,os.environ);out=Path(a.output)
    if out.exists() and json.loads(out.read_text())!=cfg:
        raise RuntimeError('Settings differ from this run. Restore the original settings or choose a new RUN_DIR.')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
