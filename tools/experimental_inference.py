"""Explicit Wiki transport with archived attempts and no automatic paid retry."""
from pathlib import Path
import hashlib
import json
import os
import re
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'runs/raw'
MODEL = 'gpt-6-luna'

def reserve_request(name):
    """Allow the controller to inject its own request ledger."""

def settle_request(name, response):
    """Allow the controller to inject its own receipt ledger."""

def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def inference(name, system, prompt, effort='low', maximum=None, model=MODEL,
              overrides=None, call_category='experiment', prompt_cache_key=None):
    """Send one authorized request; retain ambiguous failures without reissuing."""
    if model != MODEL or os.environ.get('WIKI_MODEL', MODEL) != MODEL:
        raise ValueError('Only the explicitly configured Wiki gpt-6-luna model is supported')
    if maximum is not None or call_category not in ('experiment', 'editing'):
        raise ValueError('Use the native output allowance and an explicit call category')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('Unsafe request identity')
    if effort not in ('none', 'low', 'medium', 'high'):
        raise ValueError('Unknown reasoning effort')
    payload = {'model': MODEL, 'messages': [{'role': 'system', 'content': system},
               {'role': 'user', 'content': prompt}], 'temperature': 1,
               'reasoning_effort': effort, 'response_format': {'type': 'json_object'}}
    if overrides:
        if set(overrides) & {'model', 'messages', 'stream', 'max_tokens', 'max_completion_tokens'}:
            raise ValueError('Overrides cannot change model, source content or output allowance')
        payload.update(overrides)
    if prompt_cache_key:
        payload['prompt_cache_key'] = prompt_cache_key
    binding = {'model': MODEL, 'system': system, 'prompt': prompt,
               'reasoning_effort': effort, 'overrides': overrides or {},
               'call_category': call_category, 'prompt_cache_key': prompt_cache_key}
    request_sha = hashlib.sha256(json.dumps(binding, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = RAW / (name + '.json')
    intent = RAW / (name + '.request_intent.json')
    if path.exists():
        row = json.loads(path.read_text())
        if row.get('request_sha256') != request_sha or row.get('model') != MODEL:
            raise ValueError('An existing request receipt belongs to different inputs or model')
        return row
    if intent.exists():
        raise RuntimeError('A previously admitted request lacks a valid receipt; inspect it before another call')
    base = os.environ.get('WIKI_API_BASE', '').rstrip('/')
    key = os.environ.get('WIKI_API_KEY', '')
    if not base.startswith('https://') or not key or '.example' in base:
        raise ValueError('Set the authorized HTTPS WIKI_API_BASE and WIKI_API_KEY explicitly')
    reserve_request(name)
    RAW.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects admission across concurrent processes.
    with intent.open('x') as stream:
        json.dump({'request_sha256': request_sha, 'model': MODEL, 'source_requested': 'wiki'}, stream)
    started = time.monotonic()
    with httpx.Client(timeout=300, trust_env=False) as client:
        response = client.post(base + '/chat/completions', json=payload,
            headers={'Authorization': 'Bearer ' + key,
                     'X-Litellm-Spend-Logs-Metadata': json.dumps({'source': 'wiki',
                        'purpose': 'sceneweave-' + call_category, 'research_request_identity': name})})
        response.raise_for_status()
        body = response.json()
    choice = body['choices'][0]
    row = {'request_sha256': request_sha, 'model_requested': MODEL, 'model': body.get('model'),
           'source_requested': 'wiki', 'text': choice['message'].get('content') or '',
           'finish_reason': choice.get('finish_reason'), 'usage': body.get('usage') or {},
           'gateway_response_id': body.get('id'), 'seconds': time.monotonic() - started,
           'call_category': call_category}
    _save(path, row)
    settle_request(name, row)
    if row['model'] != MODEL:
        raise ValueError('Actual served model differs from the authorized model; receipt preserved')
    return row
