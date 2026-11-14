"""Download this experiment and its datasets from an exact GitHub commit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--commit', required=True)
    p.add_argument('--destination', type=Path, required=True, help='GraphRL repository directory')
    a = p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', a.commit):
        raise ValueError('Use the exact full commit SHA')
    base = f'https://raw.githubusercontent.com/dr-eleven/GraphRL/{a.commit}/'
    manifest = json.loads(fetch(base + 'organized_experiments/5_1/deployment_manifest.json'))
    destination = a.destination.resolve()
    backup = destination / '.deployment_backups' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')

    def transfer(entry):
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe path')
        target = destination / relative
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == entry['sha256']:
            return 'verified ' + str(relative)
        raw = fetch(base + str(relative))
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Downloaded content mismatch: ' + str(relative))
        if target.exists():
            old = backup / relative
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, old)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.download')
        temporary.write_bytes(raw)
        temporary.replace(target)
        return 'downloaded ' + str(relative)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(transfer, manifest['files']):
            print(result, flush=True)
    receipt = dict(commit=a.commit, repository='https://github.com/dr-eleven/GraphRL',
                   time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   destination=str(destination), verified_files=len(manifest['files']), manifest=manifest)
    root = destination / 'organized_experiments/5_1'
    (root / 'deployment_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (root / 'deployment_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('VERIFIED', len(manifest['files']), 'files from commit', a.commit, flush=True)


if __name__ == '__main__':
    main()
