"""Verify actual independent tie artifacts without generating annotations."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
KIND = "assistant_independent_dual_review"
def digest(value):
    return original.digest(value)
def input_digest(row):
    """Hash the full original slots and request without normalizing text."""
    return digest([row['pair_id'], row['prompt'], row['chosen'], row['rejected']])

def validate_review(review):
    """Validate a reader record without repairing missing judgments."""
    if not isinstance(review, dict) or not isinstance(review.get('pair_id'), str):
        raise ValueError('A review requires its original pair identity')
    if review.get('winner') not in ('A', 'B', 'tie'):
        raise ValueError('The reader must choose A, B or tie')
    if any(type(review.get(key)) is not bool for key in ('form_a', 'form_b')):
        raise ValueError('Both actual form flags are required')
    if not isinstance(review.get('reason'), str) or not review['reason'].strip():
        raise ValueError('The actual reading requires a comparative reason')
    if not isinstance(review.get('input_sha256'), str) or len(review['input_sha256']) != 64:
        raise ValueError('The full original input hash is required')
    return review

def validate_document(document):
    """Keep separate reader identities and declared pre-comparison independence."""
    if (not isinstance(document, dict) or not isinstance(document.get('reader_id'), str)
            or not document['reader_id'].strip()
            or document.get('reviewed_before_other_reader_results') is not True
            or not isinstance(document.get('reviews'), list)):
        raise ValueError('A reader artifact requires its identity and independent-reading declaration')
    by_id = {}
    for review in document['reviews']:
        validate_review(review)
        if review['pair_id'] in by_id:
            raise ValueError('A reader artifact contains duplicate pair identities')
        if review.get('reviewed_before_other_reader_results', True) is not True:
            raise ValueError('An individual review was made after seeing the other reader results')
        if review.get('reader_id', document['reader_id']) != document['reader_id']:
            raise ValueError('An individual record has a different reader identity')
        by_id[review['pair_id']] = review
    return by_id

def _verified_document(path, expected_sha256, mtime_ns, ctime_ns, size):
    """Cache immutable content while invalidating on a changed artifact stat."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('The original independently read artifact changed')
    document = json.loads(raw)
    return document, validate_document(document)

def load_verified_document(path, expected_sha256):
    path = Path(path).resolve()
    stat = path.stat()
    return _verified_document(str(path), expected_sha256,
        stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)

def _verified_anchors(path, expected_sha256, mtime_ns, ctime_ns, size):
    """Verify the exact shared real-preference anchor material."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('The shared real human anchor manifest changed')
    manifest = json.loads(raw)
    if not isinstance(manifest.get('examples'), list) or not manifest['examples']:
        raise ValueError('The human anchor examples are missing')
    return manifest

def verify_anchors(path, expected_sha256):
    path = Path(path).resolve()
    stat = path.stat()
    return _verified_anchors(str(path), expected_sha256, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)

def assistant_genuine_tie(row):
    """Require two distinct real records and their verified frozen artifacts."""
    if (row.get('kind') != KIND or not row.get('synthetic')
            or float(row.get('preference_probability', -1)) != .5):
        return False
    entries = row.get('assistant_dual_reviews')
    if not isinstance(entries, list) or len(entries) != 2:
        return False
    readers = set()
    full_hash = input_digest(row)
    try:
        for entry in entries:
            review = validate_review(entry['review'])
            reader_id = entry['reader_id']
            if (not isinstance(reader_id, str) or not reader_id.strip() or reader_id in readers
                    or entry.get('reviewed_before_other_reader_results') is not True
                    or review['pair_id'] != row['pair_id'] or review['input_sha256'] != full_hash
                    or review['winner'] != 'tie' or review['form_a'] is not True or review['form_b'] is not True):
                return False
            document, by_id = load_verified_document(entry['artifact_path'], entry['artifact_sha256'])
            if document['reader_id'] != reader_id or by_id.get(row['pair_id']) != review:
                return False
            basis = entry.get('human_anchor_basis')
            if not isinstance(basis, str) or not basis.strip():
                return False
            verify_anchors(entry['human_anchor_manifest_path'], entry['human_anchor_manifest_sha256'])
            if document.get('human_anchor_manifest_sha256') != entry['human_anchor_manifest_sha256']:
                return False
            expected_basis = review.get('human_anchor_basis', document.get('human_anchor_basis'))
            if expected_basis != basis:
                return False
            readers.add(reader_id)
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return len(readers) == 2
