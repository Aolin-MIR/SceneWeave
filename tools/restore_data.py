"""Restore exact prepared data bytes after validating the committed catalog."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def restore(group=None, verify_only=False):
    catalog = json.loads((ROOT / 'data/catalog.json').read_text())
    selected = []
    for item in catalog['data']:
        if group and item['path'].split('/')[1] != group:
            continue
        archive = (ROOT / item['path']).resolve()
        dest = (ROOT / item['restored_path']).resolve()
        if not archive.is_relative_to(ROOT / 'data') or not dest.is_relative_to(ROOT / 'data/restored'):
            raise ValueError('Catalog path escapes the data directories')
        zipped = archive.read_bytes()
        if sha(zipped) != item['archive_sha256']:
            raise ValueError('Compressed dataset changed: ' + item['path'])
        raw = gzip.decompress(zipped)
        if sha(raw) != item['uncompressed_sha256']:
            raise ValueError('Restored dataset checksum mismatch: ' + item['path'])
        if not verify_only:
            if dest.exists() and dest.read_bytes() != raw:
                raise ValueError('Preserve a different existing data file: ' + str(dest.relative_to(ROOT)))
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(raw)
        selected.append(item['path'])
    if not selected:
        raise ValueError('No data matched the requested group')
    return {'verified_datasets': len(selected), 'restored': not verify_only,
            'labels_or_partitions_changed': False, 'training_started': False}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=('writer', 'sources', 'preferences', 'auxiliary'))
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    print(json.dumps(restore(args.group, args.verify_only)))
