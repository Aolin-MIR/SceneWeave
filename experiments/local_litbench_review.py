"""Score complete existing screenplays with the pinned official local verifier."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import subprocess
import threading

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "runs"
MODEL = REPO / 'models/litbench_verifier_81e7bc'
OUT = ARCHIVE / 'local_litbench_review_v1'
REVISION = '81e7bc150b17012c856eb39172c1b15fccd239b5'
FROZEN_LOCK=threading.Lock()
FROZEN_MODEL=None
FROZEN_TOKENIZER=None


def frozen_story_logit(text):
    """Return an internal raw-text feature with the official frozen input format."""
    global FROZEN_MODEL,FROZEN_TOKENIZER
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    with FROZEN_LOCK:
        if FROZEN_MODEL is None:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if free<24000:raise RuntimeError('Insufficient measured memory for frozen local inference')
            index=json.loads((MODEL/'model.safetensors.index.json').read_text())
            for shard in set(index['weight_map'].values()):
                metadata=MODEL/'.cache/huggingface/download'/(shard+'.metadata')
                if metadata.read_text().splitlines()[0]!=REVISION:raise ValueError('Frozen model revision differs')
            FROZEN_TOKENIZER=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
            FROZEN_MODEL=AutoModelForSequenceClassification.from_pretrained(MODEL,local_files_only=True,
                torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval()
            FROZEN_TOKENIZER.pad_token=FROZEN_TOKENIZER.eos_token
            FROZEN_MODEL.config.pad_token_id=FROZEN_TOKENIZER.pad_token_id
        inputs=FROZEN_TOKENIZER(text.strip(),return_tensors='pt',truncation=False).to('cuda:0')
        length=inputs['input_ids'].shape[1]
        if length>FROZEN_MODEL.config.max_position_embeddings:raise ValueError('Frozen model context exceeded; no truncation')
        with torch.inference_mode():value=float(FROZEN_MODEL(**inputs).logits.reshape(()).float())
        if not __import__('math').isfinite(value):raise ValueError('Nonfinite frozen preference feature')
        return {'logit':value,'tokens':length,'truncation':False,'outside_model_card_4096_example':length>4096}


def release_frozen_model():
    """Free the temporary frozen verifier before the main comparator trains."""
    global FROZEN_MODEL,FROZEN_TOKENIZER
    with FROZEN_LOCK:
        FROZEN_MODEL=None;FROZEN_TOKENIZER=None
        import torch
        torch.cuda.empty_cache()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(p)


def ready():
    index = MODEL / 'model.safetensors.index.json'
    if not index.exists() or not (MODEL / 'tokenizer.json').exists():
        return False
    value = json.loads(index.read_text())
    return all((MODEL / f).exists() for f in set(value['weight_map'].values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-model', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    while not ready():
        if not args.wait_model or time.monotonic()-started > 7200:
            raise RuntimeError('Pinned download not complete; no fallback.')
        time.sleep(180)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    if free < 24000:
        raise RuntimeError('Insufficient measured GPU memory; preserve other workloads.')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    candidates = []
    for stage in ('practice_reconstruction_v10', 'adaptation_transfer_v1', 'adaptation_whole_context_v1'):
        for path in sorted((ARCHIVE / stage / 'selected').glob('*.json')):
            row = json.loads(path.read_text())
            for index, script in enumerate(row['scripts']):
                candidates.append((str(path), index, script))
    for path in sorted((ARCHIVE / 'paper_dramaturge_api_v1/selected').glob('*.json')):
        candidates.append((str(path), 0, json.loads(path.read_text())['script']))
    for path in sorted((ARCHIVE / 'official_superwriter_api_v1/readable').glob('*_refine.md')):
        candidates.append((str(path), 0, path.read_text()))
    diagnostic = ARCHIVE / 'adaptation_single_reading_v1/protocol.json'
    if diagnostic.exists():
        for case in json.loads(diagnostic.read_text())['cases']:
            candidates.append((str(diagnostic) + '#' + case['id'], 0, case['script']))
    results, cache = [], {}
    prior = OUT / 'results.json'
    if prior.exists():
        for row in json.loads(prior.read_text())['results']:
            cache[row['sha256']] = {key: row[key] for key in ('score', 'tokens', 'beyond_model_card_4096_example')}
    save('protocol.json', {'model': 'SAA-Lab/Llama8B-CreativeWritingVerifier', 'revision': REVISION, 'input': 'Raw complete script only; no chat template, source novel or method prompt.', 'truncation': False, 'sample_count': len(candidates), 'dtype': 'BF16', 'human_calibrated': False, 'selector_changed': False})
    for path, index, script in candidates:
        text = script.strip()
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest not in cache:
            inputs = tokenizer(text, return_tensors='pt', truncation=False).to('cuda:0')
            length = inputs['input_ids'].shape[1]
            if length > model.config.max_position_embeddings:
                raise RuntimeError('Complete screenplay exceeds model context; no truncation.')
            with torch.inference_mode():
                score = model(**inputs).logits.squeeze().float().item()
            if not __import__('math').isfinite(score):
                raise RuntimeError('Non-finite reward.')
            cache[digest] = {'score': score, 'tokens': length, 'beyond_model_card_4096_example': length > 4096}
        results.append({'path': path, 'realization': index, 'sha256': digest, **cache[digest]})
        save('results.json', {'results': results, 'unique_scripts': len(cache), 'complete': len(results)==len(candidates), 'limits': 'Uncalibrated story reward transferred to screenplay format; no source-adaptation judgement or paper win claim.'})
    print('Local complete-text scoring finished:', len(results), 'outputs;', len(cache), 'unique scripts.', flush=True)


if __name__ == '__main__':
    main()
