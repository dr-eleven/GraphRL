"""项目内 JSON 配置；相对路径一律以 7-MRRA 为基准。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path)
    cfg = json.loads(path.read_text())
    if 'extends' in cfg:
        parent = load_config(path.parent / cfg.pop('extends'))
        for key, value in cfg.items():
            if isinstance(value, dict) and isinstance(parent.get(key), dict):
                parent[key].update(value)
            else:
                parent[key] = value
        cfg = parent
    return cfg


def resolve(path):
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
