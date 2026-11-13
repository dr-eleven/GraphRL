#!/usr/bin/env python3
"""Verify and restore the four exact dataset directories for 3_5_1_2.
Uses Python standard library and the system zstd executable. No network access.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

TARGETS = ('2000_oneclass_sorted', '2000_notoneclass_sorted',
           'data/test_data_600_oneclass', 'data/test_data_600_notoneclass')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in name:
        raise ValueError('Unsafe archive path: ' + name)
    return p


def matches(path, info):
    return (path.is_file() and not path.is_symlink() and
            path.stat().st_size == info['bytes'] and digest(path) == info['sha256'])


def restore(repo, output, target, verify_only):
    package = repo / target
    manifest = json.loads((package / 'archive_manifest.json').read_text())
    if manifest['target'] != target:
        raise ValueError('Manifest target mismatch')
    entries = {str(safe_name(x['path'])): x for x in manifest['files']}
    if len(entries) != len(manifest['files']):
        raise ValueError('Duplicate manifest file')
    destination = output / target
    for name, info in entries.items():
        existing = destination / name
        # Never follow existing symlinks, including a symlink in a parent directory.
        for ancestor in (existing, *existing.parents):
            if ancestor.is_symlink():
                raise ValueError('Refusing symlink: ' + str(ancestor))
        if existing.exists() and not matches(existing, info):
            raise ValueError('Existing file differs; refusing overwrite: ' + str(existing))
        if verify_only and not matches(existing, info):
            raise ValueError('Missing restored file: ' + str(existing))
    with tempfile.TemporaryDirectory(prefix='graphrl-restore-') as temp:
        stage = Path(temp)
        archive = stage / 'bundle.tar.zst'
        h = hashlib.sha256()
        total = 0
        with archive.open('wb') as out:
            for part in manifest['parts']:
                path = package / str(safe_name(part['path']))
                if not matches(path, part):
                    raise ValueError('Part checksum mismatch: ' + str(path))
                with path.open('rb') as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b''):
                        out.write(chunk)
                        h.update(chunk)
                        total += len(chunk)
        if total != manifest['archive_bytes'] or h.hexdigest() != manifest['archive_sha256']:
            raise ValueError('Combined archive checksum mismatch')
        if not verify_only:
            if not shutil.which('zstd'):
                raise RuntimeError('Install the zstd executable first')
            process = subprocess.Popen(['zstd', '-dq', '-c', str(archive)], stdout=subprocess.PIPE)
            seen = set()
            try:
                with tarfile.open(fileobj=process.stdout, mode='r|') as tf:
                    for member in tf:
                        name = str(safe_name(member.name))
                        if not member.isfile() or name not in entries or name in seen:
                            raise ValueError('Unexpected archive entry: ' + name)
                        extracted = stage / 'files' / name
                        extracted.parent.mkdir(parents=True, exist_ok=True)
                        with tf.extractfile(member) as src, extracted.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
                        if not matches(extracted, entries[name]):
                            raise ValueError('Extracted file checksum mismatch: ' + name)
                        seen.add(name)
                process.stdout.close()
                if process.wait() or seen != set(entries):
                    raise ValueError('Incomplete archive')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            for name, info in entries.items():
                dst = destination / name
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    # Files are validated before installation; existing matching files are preserved.
                    shutil.copy2(stage / 'files' / name, dst)
            if not all(matches(destination / name, info) for name, info in entries.items()):
                raise ValueError('Installed data verification failed')
    print(('Verified' if verify_only else 'Restored') + ': ' + target +
          ' (' + str(len(entries)) + ' exact files)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, help='Optional separate restoration root')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_root.resolve() if args.output_root else repo
    for target in TARGETS:
        restore(repo, output, target, args.verify_only)


if __name__ == '__main__':
    main()
