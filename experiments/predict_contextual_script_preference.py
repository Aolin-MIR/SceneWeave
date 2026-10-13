"""Predict a direct script preference with the trained, antisymmetric comparator."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from train_contextual_script_comparator import build_model, pair_text, symmetrize_logits


def selected_evaluator():
    """Resolve the explicitly retained comparator without adopting a training candidate."""
    root = Path(__file__).resolve().parents[1]
    selected = json.loads((root / 'configs/evaluator_release.json').read_text())
    checkpoint = root / 'models' / selected['export_filename']
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != selected['export_sha256']:
        raise ValueError('Released evaluator differs from its pinned export')
    return checkpoint, {'checkpoint_sha256': digest,
        'evaluator_id': 'local_script_comparator_sha256:' + selected['source_checkpoint_sha256']}


def verify_revision(source, expected_revision):
    """Verify full checkpoints and official head-plus-adapter releases."""
    source = Path(source)
    provenance = json.loads((source/'source_revision.json').read_text())
    if provenance['revision'] != expected_revision:
        raise ValueError('Base model revision differs from the trained checkpoint')
    index_path = source/'model.safetensors.index.json'
    if index_path.exists():
        files = set(json.loads(index_path.read_text())['weight_map'].values()) | {'config.json'}
    else:
        config_name = ('strands_decider_config.json' if (source/'strands_decider_config.json').exists()
                       else 'hobson_config.json')
        files = {'head.safetensors', 'lora/adapter_model.safetensors', config_name}
    for filename in files:
        metadata = source/'.cache/huggingface/download'/(filename+'.metadata')
        if metadata.read_text().splitlines()[0] != expected_revision:
            raise ValueError('Base model shard provenance differs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Optional explicit checkpoint; defaults to the pinned retained evaluator')
    parser.add_argument('--pair', required=True, help='JSON containing prompt, a and b')
    args = parser.parse_args()
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        identity = {'evaluator_id': 'local_script_comparator_sha256:' + digest,
                    'checkpoint_sha256': digest}
    else:
        checkpoint, selected = selected_evaluator()
        identity = {'evaluator_id': selected['evaluator_id'], 'checkpoint_sha256': selected['checkpoint_sha256']}
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    protocol = state['protocol']
    if protocol.get('classes') in [('A','B','tie'), ['A','B','tie']]:
        predict_three_way(state, args.pair, identity=identity)
        return
    from transformers import AutoTokenizer
    from script_preference_backends import encode_pair
    source=Path(protocol['model'])
    verify_revision(source, protocol['model_revision'])
    if protocol.get('backend') == 'decider':
        base = Path(__file__).resolve().parents[1]/'models/qwen3_5_2b_base'
        verify_revision(base, protocol['architecture']['base_model_revision'])
    tokenizer = AutoTokenizer.from_pretrained(protocol['model'], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    query = json.loads(Path(args.pair).read_text())
    row = {'prompt': query.get('prompt', ''), 'chosen': query['a'], 'rejected': query['b']}
    encoded=[encode_pair(tokenizer,row,reverse,protocol.get('backend','scalar')) for reverse in (False,True)]
    if max(item['input_ids'].shape[1] for item in encoded)>protocol['max_tokens']:
        raise ValueError('Pair exceeds untruncated trained context allowance')
    model = build_model(protocol['model'],protocol.get('backend','scalar'))
    model.config.pad_token_id = tokenizer.pad_token_id
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state['adapter']):
        raise ValueError('Checkpoint parameter schema differs')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state['adapter'][name].to(parameter.device))
    model.eval()
    logits = []
    with torch.no_grad():
        for inputs in encoded:
            logits.append(model(**{k:v.cuda() for k,v in inputs.items()}).logits.reshape(()).float())
    p = float(symmetrize_logits(*logits).sigmoid())
    print(json.dumps({**identity, 'probability_a_preferred': p, 'probability_b_preferred': 1-p,
                      'winner': 'A' if p>.5 else 'B' if p<.5 else 'tie',
                      'raw_probability_a_in_ab': float(logits[0].sigmoid()),
                      'raw_probability_b_in_ba': float(logits[1].sigmoid()),
                      'raw_order_agreement': bool((logits[0]>0)!=(logits[1]>0)) or bool((logits[0]==0)&(logits[1]==0)),
                      'swap_consistency_by_construction': True,
                      'probability_calibration_established': False,
                      'interpretation': 'Relative request-conditioned preference; no absolute manuscript score'}))


def predict_three_way(state, pair_path, identity=None):
    """Read the learned indifference option without equating it to uncertainty."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from transformers import AutoTokenizer
    from train_tie_script_comparator import encode_three, swap_options, CLASSES
    protocol=state['protocol'];source=Path(protocol['parent_model'])
    if not source.is_absolute():
        source = Path(__file__).resolve().parents[1] / source
    verify_revision(source,protocol['parent_revision'])
    verify_revision(Path(__file__).resolve().parents[1]/'models/qwen3_5_2b_base',
                    protocol['architecture']['base_model_revision'])
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    query=json.loads(Path(pair_path).read_text())
    if not all(isinstance(query.get(k),str) and query[k].strip() for k in ('prompt','a','b')):
        raise ValueError('A shared writing request and both complete supplied drafts are required')
    row={'prompt':query['prompt'],'chosen':query['a'],'rejected':query['b']}
    encoded=[encode_three(tokenizer,row,r) for r in (False,True)]
    if max(i['input_ids'].shape[1] for i in encoded)>protocol['max_tokens']:
        raise ValueError('Input exceeds untruncated trained context allowance')
    model=build_model(source,'decider')
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(parameters)!=set(state['adapter']):raise ValueError('Checkpoint parameter schema differs')
    with torch.no_grad():
        for name,p in parameters.items():p.copy_(state['adapter'][name].to(p.device))
    model.eval()
    with torch.no_grad():
        z=[model.decision(**{k:v.cuda() for k,v in i.items()},temperature=1.)['logits'][0].float() for i in encoded]
        p=((z[0]+swap_options(z[1]))/2).softmax(-1)
    print(json.dumps({**(identity or {}), 'winner':CLASSES[int(p.argmax())],'probabilities':dict(zip(CLASSES,p.cpu().tolist())),
        'raw_order_agreement':int(z[0].argmax())==int(swap_options(z[1]).argmax()),
        'swap_consistency_by_construction':True,'probability_calibration_established':False,
        'interpretation':'Request-conditioned A/B/indifference prediction; no absolute manuscript score'}))


if __name__ == '__main__':
    main()
