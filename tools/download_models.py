"""Download an explicitly selected public base model outside version control."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('writer', 'preference_comparator', 'preference_base'), required=True)
    p.add_argument('--revision', help='Required explicit revision when absent from the registry')
    args = p.parse_args()
    spec = json.loads((ROOT / 'configs/models.json').read_text())[args.model]
    revision = args.revision or spec.get('revision')
    if not revision:
        p.error('Pin an actual --revision before downloading this model')
    destination = ROOT / spec['local_directory']
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=spec['repo_id'], revision=revision, local_dir=destination)
    (destination / 'source_revision.json').write_text(json.dumps(
        {'repo_id': spec['repo_id'], 'revision': revision}, indent=2) + '\n')
    print(json.dumps({'repo_id': spec['repo_id'], 'revision': revision, 'weights_tracked_by_git': False}))
