"""Read unchanged TRAIN/DEV while sealing held-out TEST lexically."""
import hashlib
import json
from pathlib import Path
def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)

def skip_space(raw, offset):
    while offset < len(raw) and raw[offset] in b' \r\n\t':
        offset += 1
    return offset

def value_end(raw, offset):
    """Locate a JSON value lexically without decoding any protected target."""
    start = offset
    quote = False
    escaped = False
    stack = []
    for offset in range(start, len(raw)):
        byte = raw[offset]
        if quote:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quote = False
                if not stack and raw[start] == 34:
                    return offset + 1
            continue
        if byte == 34:
            quote = True
        elif byte in (91, 123):
            stack.append(byte)
        elif byte in (93, 125):
            if not stack:
                return offset
            if (stack.pop(), byte) not in ((91, 93), (123, 125)):
                raise ValueError('Mismatched JSON container')
            if not stack:
                return offset + 1
        elif not stack and byte in b', \r\n\t':
            return offset
    if quote or stack:
        raise ValueError('Incomplete JSON value')
    return len(raw)

def normalized_raw_digest(raw):
    """Ignore formatting whitespace, retaining all protected JSON content."""
    result = hashlib.sha256()
    quote = escaped = False
    block = bytearray()
    for byte in raw:
        if quote or byte not in b' \r\n\t':
            block.append(byte)
        if quote:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quote = False
        elif byte == 34:
            quote = True
        if len(block) >= 65536:
            result.update(block)
            block.clear()
    result.update(block)
    return result.hexdigest()

def array_size(raw):
    offset = skip_space(raw, 0)
    if raw[offset] != 91:
        raise ValueError('Protected split must remain a JSON array')
    count = 0
    offset = skip_space(raw, offset + 1)
    while raw[offset] != 93:
        offset = skip_space(raw, value_end(raw, offset))
        count += 1
        if raw[offset] == 44:
            offset = skip_space(raw, offset + 1)
        elif raw[offset] != 93:
            raise ValueError('Malformed protected JSON array')
    return count

def read_train_dev(path):
    """Decode TRAIN/DEV only; seal TEST by lexical hash and row count."""
    raw = Path(path).read_bytes()
    offset = skip_space(raw, 0)
    if raw[offset] != 123:
        raise ValueError('Expected an object containing preserved data splits')
    result = {'data_sha256': hashlib.sha256(raw).hexdigest()}
    seen = set()
    offset = skip_space(raw, offset + 1)
    while raw[offset] != 125:
        end = value_end(raw, offset)
        key = json.loads(raw[offset:end])
        if key in seen:
            raise ValueError('Duplicate data split key')
        seen.add(key)
        offset = skip_space(raw, end)
        if raw[offset] != 58:
            raise ValueError('Expected a JSON member separator')
        start = skip_space(raw, offset + 1)
        end = value_end(raw, start)
        if key in ('train', 'dev'):
            result[key] = json.loads(raw[start:end])
        elif key == 'test':
            result['test_content_sha256'] = normalized_raw_digest(raw[start:end])
            result['test_rows'] = array_size(raw[start:end])
        offset = skip_space(raw, end)
        if raw[offset] == 44:
            offset = skip_space(raw, offset + 1)
        elif raw[offset] != 125:
            raise ValueError('Malformed top-level dataset object')
    if skip_space(raw, offset + 1) != len(raw):
        raise ValueError('Unexpected text after dataset object')
    if not {'train', 'dev', 'test_content_sha256', 'test_rows'}.issubset(result):
        raise ValueError('TRAIN, DEV and protected TEST must all be present')
    return result
