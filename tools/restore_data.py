"""Restore exact dataset bytes from the small Git-tracked gzip files."""
from pathlib import Path
import hashlib,gzip,json
R=Path(__file__).resolve().parents[1]/'data/3_5_1_3_6'
def main():
    metadata=json.loads((R/'manifest.json').read_text())
    for split,item in metadata['splits'].items():
        source=R/item['compressed_file'];target=R/item['uncompressed_file']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==item['compressed_sha256']
        data=gzip.decompress(source.read_bytes())
        assert hashlib.sha256(data).hexdigest()==item['sha256']
        assert len(json.loads(data))==item['rows']
        if target.exists():assert target.read_bytes()==data, 'Refusing to overwrite changed data: '+str(target)
        else:target.write_bytes(data)
        print(split,item['rows'],'rows; SHA256 verified:',target)
if __name__=='__main__':main()
