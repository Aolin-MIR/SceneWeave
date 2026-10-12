"""Inspect candidate snapshots without qualifying aesthetics or creating targets."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'data/writer/candidates/snapshot_588'

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def inspect():
    manifest = json.loads(gzip.decompress((SNAPSHOT / 'manifest.json.gz').read_bytes()))
    stages = {}
    delta = []
    physical_rows = 0
    for path in sorted(SNAPSHOT.glob('*.jsonl.gz')):
        rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines() if line.strip()]
        physical_rows += len(rows)
        if path.name.startswith('new_increment113.'):
            delta = rows
            continue
        for row in rows:
            identifier = row['id']
            if identifier in stages:
                raise ValueError('Repeated candidate ID across primary stages: ' + identifier)
            stages[identifier] = row
    if len(stages) != manifest['total_actual_candidates']:
        raise ValueError('Primary candidate count differs from the immutable source manifest')
    for row in delta:
        if row['id'] not in stages or fingerprint(row) != fingerprint(stages[row['id']]):
            raise ValueError('Delta record differs from its primary-stage candidate')
    if len(delta) != manifest['new_increment_records']:
        raise ValueError('Delta count differs from the immutable source manifest')
    return {'primary_candidate_records': len(stages), 'physical_rows_including_delta_view': physical_rows,
            'overlapping_delta_view_records': len(delta), 'delta_is_not_new_independent_data': True,
            'qualified_new_train_records': manifest['final_TRAIN_new'],
            'training_authorized': manifest['training_authorized'],
            'source_protection_confirmed': manifest['current_owner_protection_confirmed'],
            'new_token_check_pending': manifest['new_increment_complete_token_check_pending'],
            'aesthetic_qualification_performed': False}

if __name__ == '__main__':
    print(json.dumps(inspect(), indent=2))
