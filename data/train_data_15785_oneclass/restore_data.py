"""Verify the uploaded parts and restore the complete one-class dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true',help='Verify all part and archive hashes without extracting')
    parser.add_argument('--output-dir',type=Path,default=HERE.parent,help='Parent data directory to restore into')
    args=parser.parse_args()
    manifest=json.loads((HERE/'archive_manifest.json').read_text())
    combined=hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix='graphrl-oneclass-') as tmp:
        archive=Path(tmp)/'dataset.tar.zst'
        with archive.open('wb') as target:
            for item in manifest['parts']:
                source=HERE/'archive'/item['name'];digest=hashlib.sha256();size=0
                with source.open('rb') as handle:
                    while block:=handle.read(1024*1024):
                        digest.update(block);combined.update(block);target.write(block);size+=len(block)
                if size!=item['bytes'] or digest.hexdigest()!=item['sha256']:
                    raise RuntimeError('Part checksum failed: '+item['name'])
        if archive.stat().st_size!=manifest['archive_bytes'] or combined.hexdigest()!=manifest['archive_sha256']:
            raise RuntimeError('Combined archive checksum failed')
        print('All part and archive SHA256 checks passed.',flush=True)
        if args.verify_only:return
        if not shutil.which('zstd'):
            raise RuntimeError('Install zstd first (for example: conda install -c conda-forge zstd, or apt-get install zstd)')
        output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
        process=subprocess.Popen(['zstd','-q','-d','-c',str(archive)],stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout,mode='r|') as tf:
                for member in tf:
                    path=output/member.name
                    if not path.resolve().is_relative_to(output) or not (member.isfile() or member.isdir()):
                        raise RuntimeError('Unsafe archive member: '+member.name)
                    tf.extract(member,path=output)
        finally:
            process.stdout.close()
        if process.wait()!=0:raise RuntimeError('zstd decompression failed')
        restored=output/'train_data_15785_oneclass'
        info=json.loads((restored/'manifest.json').read_text())
        assert info['rows']==manifest['rows']==15364
        assert len(list((restored/'sample_results').glob('*.json')))==15364
        print(f'Restored {info["rows"]} complete five-round samples to {restored}')

if __name__=='__main__':main()
