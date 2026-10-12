"""Fine-tune the released native pointer head on A, B and genuine indifference."""
import argparse
import gc
import hashlib
import json
import itertools
import math
from pathlib import Path
import random
import shutil
import subprocess
import time

import torch
from torch.nn import functional as F
from run_repaired_script_preference_training import read_train_dev, save, file_digest, value_end, skip_space
from train_adaptation_reward import save_checkpoint, restore_training_state
from train_contextual_script_comparator import pair_text, padded_pair_inputs, fitting_microbatch_count
from script_preference_backends import official_decider, build_backend
from assistant_script_tie_reviews import assistant_genuine_tie, KIND as ASSISTANT_REVIEW_KIND

CLASSES = ('A', 'B', 'tie')


def genuine_tie(row):
    """Distinguish a released indifference vote from an aggregate split vote."""
    if float(row['preference_probability']) != .5:
        return False
    if row.get('synthetic'):
        if row.get('kind') == ASSISTANT_REVIEW_KIND:
            return assistant_genuine_tie(row)
        reviews = row.get('reviews', {})
        votes = [reviews.get('gpt-6-luna:' + order, {}) for order in ('AB', 'BA')]
        return (len(reviews) >= 2 and all(v.get('winner') == 'tie' and
                v.get('form_a') is True and v.get('form_b') is True for v in votes))
    strengths = row.get('original_signed_strengths')
    if strengths:
        return all(float(value) == 0 for value in strengths)
    return row.get('winner', row.get('original_winner')) == 'tie'


def target_distribution(row, epsilon=0.):
    """Keep human annotations intact; soften only synthetic fitting targets."""
    if not math.isfinite(epsilon) or not 0 <= epsilon < .5:
        raise ValueError('Smoothing must be finite and in [0, .5)')
    q = float(row['preference_probability'])
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('Invalid preference probability')
    strengths = row.get('original_signed_strengths')
    if not row.get('synthetic') and strengths:
        values = [float(value) for value in strengths]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Invalid original individual human annotation')
        result = [sum(value < 0 for value in values)/len(values),
                  sum(value > 0 for value in values)/len(values),
                  sum(value == 0 for value in values)/len(values)]
        if abs(result[0]+result[2]/2-q)>1e-8:
            raise ValueError('Original human annotation directions disagree with stored target')
        return result
    if genuine_tie(row):
        return [epsilon/2, epsilon/2, 1-epsilon] if row.get('synthetic') else [0., 0., 1.]
    if row.get('synthetic') and q == .5:
        raise ValueError('Synthetic half-probability lacks supported indifference judgments')
    return [(1-2*epsilon)*q+epsilon, (1-2*epsilon)*(1-q)+epsilon, 0.] if row.get('synthetic') else [q, 1-q, 0.]


def swap_options(value):
    """Map the reversed presentation back to the original candidate identity."""
    return value[..., [1, 0, 2]]


def encode_three(tokenizer, row, reverse=False, option_order=None):
    """Use all three native pointer slots and preserve complete manuscripts."""
    official_decider()
    from strands_decider.schema import ChoiceQuestion
    from strands_decider.prompting import build_prompt
    from strands_decider.data.collate import SystemOneCollator
    text = pair_text(row, reverse).replace('Return the preference for candidate A over candidate B.',
        'Choose A, B or tie. Tie means comparable overall quality at the requested scope; '
        'small stylistic differences do not require a winner. Missing material is invalid input.')
    question = ChoiceQuestion(instructions='Which preference is best supported by both complete drafts?',
        criteria={'A': 'Candidate A is preferable', 'B': 'Candidate B is preferable',
                  'tie': 'The candidates have comparable overall quality for the request'})
    prompt, rendered = build_prompt(text, question, option_order=option_order)
    enc = tokenizer(prompt, return_tensors='pt', truncation=False, return_offsets_mapping=True)
    offsets = enc.pop('offset_mapping')[0].tolist()
    indices = SystemOneCollator.option_token_index(offsets, rendered.option_spans,
                                                   len(prompt)-len(rendered.text))
    canonical_indices = [indices[rendered.slot_labels.index(label)] for label in CLASSES]
    enc['opt_idx'] = torch.tensor([canonical_indices], dtype=torch.long)
    enc['n_slots'] = torch.tensor([3], dtype=torch.long)
    return enc


def train_option_order(pair_id):
    """Choose a label-independent option permutation using a fixed TRAIN seed."""
    permutations=list(itertools.permutations(range(3)))
    key=hashlib.sha256(('script_tie_option_order_42:'+pair_id).encode()).hexdigest()
    return permutations[int(key[:16],16)%len(permutations)]


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1)


def weighted_training_loss(logits, targets, weights, update_pairs):
    """Keep the original two-order, eight-pair update normalization."""
    return (soft_ce(logits, targets) * weights).sum() / (2 * update_pairs)


def profile_three_way_batches(model, encoded, tokenizer, parameters, targets, args, available_bytes):
    """Validate native pointer padding and actual TRAIN gradients without updating."""
    longest = sorted(encoded['train'], key=lambda item: max(v['input_ids'].shape[1] for v in item[1]), reverse=True)
    shortest = min(longest, key=lambda item: max(v['input_ids'].shape[1] for v in item[1]))
    trials = []
    for count in getattr(args, 'profile_counts', (8, 4, 2, 1)):
        if len(longest) < count:
            continue
        rows = longest[:count]
        width = max(v['input_ids'].shape[1] for _, pair in rows for v in pair)
        if count * width > args.maximum_padded_tokens:
            continue
        for parameter in parameters:
            parameter.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trial = {'pairs': count, 'padded_tokens': count * width, 'optimizer_updates': 0}
        started = time.perf_counter()
        try:
            # Include unequal lengths, so the test exercises actual padding.
            check_rows = rows if count == 1 else rows[:-1] + [shortest]
            model.eval()
            maximum_difference = 0.
            with torch.no_grad():
                for reverse in (False, True):
                    single = torch.cat([model.decision(**{k: v.cuda() for k, v in pair[reverse].items()},
                                        temperature=1.)['logits'].float() for _, pair in check_rows])
                    batch = padded_pair_inputs([pair[reverse] for _, pair in check_rows], tokenizer.pad_token_id)
                    together = model.decision(**{k: v.cuda() for k, v in batch.items()}, temperature=1.)['logits'].float()
                    difference = float((single.softmax(-1) - together.softmax(-1)).abs().max())
                    maximum_difference = max(maximum_difference, difference)
                    if not torch.allclose(single.softmax(-1), together.softmax(-1), atol=.005, rtol=.02):
                        raise ValueError('Complete single and padded native-pointer probabilities disagree')
            del single, together, batch
            model.train()
            for reverse in (False, True):
                batch = padded_pair_inputs([pair[reverse] for _, pair in rows], tokenizer.pad_token_id)
                logits = model.decision(**{k: v.cuda() for k, v in batch.items()}, temperature=1.)['logits'].float()
                target = torch.tensor([targets[row['pair_id']] for row, _ in rows], device='cuda')
                soft_ce(logits, swap_options(target) if reverse else target).mean().backward()
                del logits, batch, target
            if not all(p.grad is None or torch.isfinite(p.grad).all().item() for p in parameters):
                raise RuntimeError('Nonfinite actual TRAIN batching gradients')
            reserved = torch.cuda.max_memory_reserved()
            required = reserved + sum(p.numel() * 12 for p in parameters) + 3072 * 2**20
            trial.update(maximum_probability_difference=maximum_difference,
                         peak_reserved_bytes=reserved, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         required_bytes_including_optimizer_and_headroom=required,
                         elapsed_seconds=time.perf_counter() - started)
            if required > available_bytes:
                trial['status'] = 'insufficient_optimizer_and_memory_headroom'
                trials.append(trial)
                continue
            trial['status'] = 'passed'
            trials.append(trial)
            return count, trials
        except torch.OutOfMemoryError:
            trial.update(status='out_of_memory_without_optimizer_update', elapsed_seconds=time.perf_counter() - started)
            trials.append(trial)
        except ValueError as error:
            trial.update(status='padding_numerical_mismatch', reason=str(error), elapsed_seconds=time.perf_counter() - started)
            trials.append(trial)
        finally:
            for parameter in parameters:
                parameter.grad = None
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError('No valid complete-input native three-way batch passed')


def length_adaptive_groups(indices, lengths, capabilities):
    """Reorder only within one optimizer update and use verified batch widths."""
    pending = sorted(indices, key=lambda index: lengths[index], reverse=True)
    groups = []
    while pending:
        width = lengths[pending[0]]
        count = next((count for count in sorted(capabilities, reverse=True)
                      if count <= len(pending) and width <= capabilities[count]), None)
        if count is None:
            raise ValueError('Complete input has no validated physical batch capability')
        groups.append(pending[:count])
        pending = pending[count:]
    return groups


def profile_length_adaptive_batches(model, encoded, tokenizer, parameters, targets, args, available_bytes):
    """Verify each supported batch count on its longest eligible TRAIN inputs."""
    capabilities, trials = {}, []
    for count in (8, 4, 2, 1):
        rows = [item for item in encoded['train']
                if count == 1 or count * max(v['input_ids'].shape[1] for v in item[1]) <= args.maximum_padded_tokens]
        if len(rows) < count:
            continue
        args.profile_counts = (count,)
        try:
            selected, records = profile_three_way_batches(model, {'train': rows}, tokenizer,
                                                         parameters, targets, args, available_bytes)
        except RuntimeError as error:
            if 'No valid complete-input' not in str(error):
                raise
            trials.append({'pairs': count, 'status': 'not_qualified', 'reason': str(error), 'optimizer_updates': 0})
            continue
        width = max(v['input_ids'].shape[1] for _, pair in rows for v in pair)
        capabilities[selected] = width
        trials.extend(records)
    del args.profile_counts
    if 1 not in capabilities:
        raise RuntimeError('The longest complete TRAIN input is not memory-safe')
    return capabilities, trials


def benchmark_length_adaptive_batches(model, encoded, tokenizer, parameters, targets, capabilities):
    """Compare matched random TRAIN update groups without changing any weights."""
    indices = random.Random(20261012).sample(range(len(encoded['train'])), min(64, len(encoded['train'])))
    lengths = [max(v['input_ids'].shape[1] for v in pair) for _, pair in encoded['train']]
    rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    timings, padded_counts = {}, {}
    try:
        model.train()
        for name in ('current_fixed_two', 'length_adaptive'):
            for repeat in range(2):
                torch.set_rng_state(rng); torch.cuda.set_rng_state_all(cuda_rng)
                torch.cuda.synchronize(); started = time.perf_counter()
                padding, actual_groups = 0, {}
                for offset in range(0, len(indices), 8):
                    chunk = indices[offset:offset + 8]
                    for parameter in parameters:
                        parameter.grad = None
                    groups = ([chunk[start:start + 2] for start in range(0, len(chunk), 2)]
                              if name == 'current_fixed_two' else length_adaptive_groups(chunk, lengths, capabilities))
                    for group in groups:
                        micro = [encoded['train'][index] for index in group]
                        target = torch.tensor([targets[row['pair_id']] for row, _ in micro], device='cuda')
                        weights = torch.ones(len(group), device='cuda')
                        actual_groups[len(group)] = actual_groups.get(len(group), 0) + 1
                        for reverse in (False, True):
                            inputs = padded_pair_inputs([pair[reverse] for _, pair in micro], tokenizer.pad_token_id)
                            padding += inputs['input_ids'].numel()
                            logits = model.decision(**{k: v.cuda() for k, v in inputs.items()}, temperature=1.)['logits'].float()
                            weighted_training_loss(logits, swap_options(target) if reverse else target,
                                                   weights, len(chunk)).backward()
                torch.cuda.synchronize()
                timings[name] = time.perf_counter() - started
                padded_counts[name] = {'padded_tokens_both_orders': padding, 'physical_batch_counts': actual_groups}
        return {'pairs': len(indices), 'train_pair_ids': [encoded['train'][index][0]['pair_id'] for index in indices],
                'optimizer_updates': 0, 'two_orders_per_pair': True,
                'current_fixed_two_seconds': timings['current_fixed_two'],
                'length_adaptive_seconds': timings['length_adaptive'],
                'measured_speedup': timings['current_fixed_two'] / timings['length_adaptive'],
                'padded_token_counts': padded_counts,
                'scope': 'Same seeded random TRAIN update groups; full-epoch throughput remains pending'}
    finally:
        for parameter in parameters:
            parameter.grad = None
        torch.set_rng_state(rng); torch.cuda.set_rng_state_all(cuda_rng)
        gc.collect(); torch.cuda.empty_cache()


def verify_batching_continuation(saved, current):
    """Allow only an explicitly requested execution amendment on unchanged data."""
    allowed = {'implementation_sha256', 'physical_microbatch_pairs', 'maximum_padded_tokens_per_microbatch',
               'length_adaptive_microbatch', 'length_adaptive_capabilities', 'warm_start_policy',
               'author_batching_amendment'}
    current = json.loads(json.dumps(current))
    differences = {key for key in saved if saved[key] != current.get(key)}
    if differences - allowed:
        raise ValueError('Continuation changes protected configuration: ' + ', '.join(sorted(differences - allowed)))
    amendment = current.get('author_batching_amendment', {})
    if amendment.get('author_approved') is not True or amendment.get('original_data_sha256') != saved['data_sha256']:
        raise ValueError('A changed batching continuation requires the actual author amendment')


def benchmark_three_way_batches(model, encoded, tokenizer, parameters, targets, count, disable_checkpointing):
    """Time the previous serial recipe and the verified batching on the same TRAIN inputs."""
    ordered = sorted(encoded['train'], key=lambda item: max(v['input_ids'].shape[1] for v in item[1]))
    middle = len(ordered) // 2
    rows = ordered[middle:middle + 8]
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    timings = {}
    try:
        for name, physical, checkpointing in [('previous_serial', 1, True),
                                               ('verified_batched', count, not disable_checkpointing)]:
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            else:
                model.decision.torso.gradient_checkpointing_disable()
            torch.set_rng_state(rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            for parameter in parameters:
                parameter.grad = None
            model.train()
            # Exclude one-time kernel initialization from the measured pass.
            for _ in range(2):
                for parameter in parameters:
                    parameter.grad = None
                torch.cuda.synchronize()
                started = time.perf_counter()
                for offset in range(0, len(rows), physical):
                    micro = rows[offset:offset + physical]
                    target = torch.tensor([targets[row['pair_id']] for row, _ in micro], device='cuda')
                    weights = torch.ones(len(micro), device='cuda')
                    for reverse in (False, True):
                        inputs = padded_pair_inputs([pair[reverse] for _, pair in micro], tokenizer.pad_token_id)
                        logits = model.decision(**{k: v.cuda() for k, v in inputs.items()}, temperature=1.)['logits'].float()
                        weighted_training_loss(logits, swap_options(target) if reverse else target,
                                               weights, len(rows)).backward()
                torch.cuda.synchronize()
                timings[name] = time.perf_counter() - started
        return {'train_pair_ids': [row['pair_id'] for row, _ in rows], 'pairs': len(rows),
                'two_orders_per_pair': True, 'optimizer_updates': 0,
                'previous_serial_seconds': timings['previous_serial'],
                'verified_batched_seconds': timings['verified_batched'],
                'measured_speedup': timings['previous_serial'] / timings['verified_batched'],
                'scope': 'Same median-length TRAIN inputs; this is not a full-epoch speed estimate'}
    finally:
        for parameter in parameters:
            parameter.grad = None
        if disable_checkpointing:
            model.decision.torso.gradient_checkpointing_disable()
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.set_rng_state(rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        gc.collect()
        torch.cuda.empty_cache()


def metrics(rows):
    """Retain every eligible observation, including false ties on clear choices."""
    explicit = [r for r in rows if r['human_label'] is not None]
    ties = [r for r in explicit if r['human_label'] == 'tie']
    decisive = [r for r in explicit if r['human_label'] != 'tie']
    predicted_ties = [r for r in explicit if r['prediction'] == 'tie']
    correct_ties = sum(r['prediction'] == 'tie' for r in ties)
    return {'n': len(rows), 'explicit_label_n': len(explicit),
        'three_way_correct': sum(r['prediction'] == r['human_label'] for r in explicit),
        'three_way_accuracy': sum(r['prediction'] == r['human_label'] for r in explicit)/max(1,len(explicit)),
        'decisive_n': len(decisive), 'decisive_correct': sum(r['prediction'] == r['human_label'] for r in decisive),
        'decisive_accuracy': sum(r['prediction'] == r['human_label'] for r in decisive)/max(1,len(decisive)),
        'human_ties': len(ties), 'correct_ties': correct_ties,
        'tie_recall': correct_ties/len(ties) if ties else None,
        'tie_precision': correct_ties/len(predicted_ties) if predicted_ties else None,
        'false_ties_on_decisive': sum(r['prediction'] == 'tie' for r in decisive),
        'false_tie_rate_on_decisive': sum(r['prediction'] == 'tie' for r in decisive)/max(1,len(decisive))}


def opaque_member(path, member):
    """Decode one named member while leaving all other JSON values opaque."""
    raw=Path(path).read_bytes();offset=skip_space(raw,1)
    while raw[offset]!=125:
        end=value_end(raw,offset);key=json.loads(raw[offset:end]);offset=skip_space(raw,end)
        if raw[offset]!=58:raise ValueError('Invalid JSON object')
        start=skip_space(raw,offset+1);end=value_end(raw,start)
        if key==member:return json.loads(raw[start:end])
        offset=skip_space(raw,end)
        if raw[offset]==44:offset=skip_space(raw,offset+1)
    raise KeyError(member)


def frozen_external(base, used_groups):
    """Reuse the exact previous development cohort without opening final TEST."""
    path=base/'open_weight_comparison_v1/strands_decider_2b/external_assessment_protocol.json'
    fixed=json.loads(path.read_text())
    qualifier=json.loads((base/'contextual_calibration_protocol.json').read_text())
    source=Path(qualifier['human_training_data_path'])
    if file_digest(source)!=qualifier['human_training_data_sha256']:
        raise ValueError('External human source changed')
    native=source.parent.parent/'contextual_fragment_human_votes_v2/released_contextual_preferences.json'
    rows=opaque_member(source,'train')+opaque_member(native,'additional_source_validation')
    by_id={r['pair_id']:r for r in rows}
    selected=[by_id[pair_id] for pair_id in fixed['primary_pair_ids']]
    if any(r['work_id'] in used_groups for r in selected):
        raise ValueError('External assessment source group leaked into TRAIN/DEV')
    return selected,{'frozen_old_protocol_sha256':file_digest(path),'pair_ids':fixed['primary_pair_ids'],
        'source_sha256':file_digest(source),'native_source_sha256':file_digest(native),
        'limits':'Previously used teacher-development groups plus fifteen native human validation pairs; '
                 'this is not fully independent final TEST. Only the real DEV selects the checkpoint.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--selection-protocol', type=Path, required=True)
    parser.add_argument('--additional-human-dev', type=Path)
    parser.add_argument('--initial-checkpoint', type=Path)
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--early-stopping-patience', type=int, default=3)
    parser.add_argument('--auto-microbatch', action='store_true')
    parser.add_argument('--maximum-padded-tokens', type=int, default=65536)
    parser.add_argument('--disable-gradient-checkpointing', action='store_true')
    parser.add_argument('--length-adaptive-microbatch', action='store_true')
    parser.add_argument('--continue-from-run', type=Path)
    args = parser.parse_args()
    if args.early_stopping_patience < 1 or args.maximum_padded_tokens < 1:
        raise ValueError('Invalid stopping or complete-input batching allowance')
    if args.continue_from_run and args.resume:
        raise ValueError('An execution amendment and exact same-run resume are separate modes')
    if args.length_adaptive_microbatch and (not args.auto_microbatch or not args.disable_gradient_checkpointing):
        raise ValueError('Length-adaptive batching requires an actual no-recomputation memory probe')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'latest.pt').exists() and not args.resume:
        raise ValueError('Do not overwrite an existing trained run')
    data = read_train_dev(args.data)
    selection = json.loads(args.selection_protocol.read_text())
    selection_ids = set(selection['selection_dev_ids'])
    original_selection_ids=set(selection_ids)
    added_dev=[]
    if args.additional_human_dev:
        added_dev=json.loads(args.additional_human_dev.read_text())['dev']
        if any(r.get('synthetic') for r in added_dev):raise ValueError('Additional DEV must have original human labels')
        if {r['pair_id'] for r in data['dev']} & {r['pair_id'] for r in added_dev}:
            raise ValueError('Additional human DEV contains original observations')
        data['dev']=data['dev']+added_dev
        selection_ids.update(r['pair_id'] for r in added_dev)
    if not selection_ids.issubset({r['pair_id'] for r in data['dev']}):
        raise ValueError('Primary human DEV identities changed')
    if set(r['work_id'] for r in data['train']) & set(r['work_id'] for r in data['dev']):
        raise ValueError('TRAIN/DEV group overlap')
    # Freeze validated targets once; do not reread immutable reader files per step.
    train_targets = {row['pair_id']: target_distribution(row, .1) for row in data['train']}
    dev_targets = {row['pair_id']: target_distribution(row) for row in data['dev']}
    genuine = {row['pair_id']: genuine_tie(row) for row in data['train'] + data['dev']}
    # Bind optimized operators before Transformers resolves its fallback decorator.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = {split: [(r, [encode_three(tokenizer,r,reverse,
                            train_option_order(r['pair_id']) if split=='train' else None) for reverse in (False,True)])
                        for r in data[split]] for split in ('train','dev')}
    lengths = {split: [max(i['input_ids'].shape[1] for i in pair) for _,pair in rows]
               for split,rows in encoded.items()}
    if any(max(v) > args.max_tokens for v in lengths.values()):
        raise ValueError('Full input exceeds token allowance; no silent exclusion or truncation')
    protocol = {'data_sha256': file_digest(args.data), 'test_scored': False, 'test_tokenized': False,
        'protected_test_sha256': data['test_content_sha256'], 'test_rows': data['test_rows'],
        'classes': CLASSES, 'parent_model': str(args.model.resolve()),
        'parent_revision': json.loads((args.model/'source_revision.json').read_text())['revision'],
        'objective': 'Bidirectional soft three-class cross entropy with the published native pointer head',
        'option_order': 'TRAIN-only fixed seeded random permutation independent of labels; '
                        'canonical semantic pointer indices and canonical DEV/inference options',
        'inference': 'softmax((logits_AB + logits_BA[A/B swapped, tie unchanged])/2); argmax',
        'synthetic_label_smoothing': .1, 'synthetic_tie_target': [.05,.05,.9],
        'synthetic_decisive_target': '[.8*q+.1,.8*(1-q)+.1,0]',
        'human_target_policy': 'Explicit indifference -> tie; individual split votes retain A/B distribution',
        'train_pairs': len(encoded['train']), 'dev_pairs': len(encoded['dev']),
        'selection_dev_ids': sorted(selection_ids), 'max_tokens': args.max_tokens, 'truncation': False,
        'original_selection_dev_ids':sorted(original_selection_ids),
        'additional_human_dev_sha256':file_digest(args.additional_human_dev) if args.additional_human_dev else None,
        'additional_human_dev_ids':[r['pair_id'] for r in added_dev],
        'initial_checkpoint':str(args.initial_checkpoint.resolve()) if args.initial_checkpoint else None,
        'initial_checkpoint_sha256':file_digest(args.initial_checkpoint) if args.initial_checkpoint else None,
        'warm_start_policy':'Load selected prior adapters only; reset optimizer, scheduler and RNG for changed-data stage' if args.initial_checkpoint else None,
        'primary_selection': 'Minimum completed-epoch real DEV three-class soft cross entropy',
        'synthetic_total_weight': 'Equal to total human TRAIN weight',
        'lr': 1e-5, 'minimum_epochs': 4, 'patience_full_epochs': args.early_stopping_patience, 'maximum_epochs': 200,
        'scheduler': 'ReduceLROnPlateau patience1 factor.5 minimum1e-7',
        'gradient_accumulation_pairs': 8, 'physical_microbatch_pairs': 1,
        'auto_microbatch': args.auto_microbatch, 'maximum_padded_tokens_per_microbatch': args.maximum_padded_tokens,
        'gradient_checkpointing': not args.disable_gradient_checkpointing,
        'length_adaptive_microbatch': args.length_adaptive_microbatch,
        'verified_targets_cached_before_optimization': True,
        'implementation_sha256': {name: file_digest(Path(__file__).parent/name) for name in
            ('train_tie_script_comparator.py','assistant_script_tie_reviews.py','script_preference_backends.py','train_adaptation_reward.py')},
        'assistant_dual_review_policy': 'Two distinct independently read assistant artifacts verified against exact full inputs; synthetic labels',
        'known_pretraining_source_policy': selection.get('evaluation_source_policy'),
        'genuine_train_ties': sum(genuine[r['pair_id']] for r in data['train']),
        'synthetic_train_ties': sum(genuine[r['pair_id']] for r in data['train'] if r.get('synthetic'))}
    if args.preflight:
        save(out/'encoding_preflight.json', {**protocol,'maximum_encoded_tokens': {k:max(v) for k,v in lengths.items()}})
        print(json.dumps({'status':'encoding_preflight_passed','train_pairs':len(data['train'])}),flush=True)
        return
    free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if free < 16000:
        raise RuntimeError('Insufficient measured GPU memory for complete-input training')
    torch.cuda.set_per_process_memory_fraction(min(1.,(free-3072)*2**20/torch.cuda.get_device_properties(0).total_memory))
    random.seed(42); torch.manual_seed(42)
    model = build_backend(args.model,'decider')
    from predict_contextual_script_preference import verify_revision
    verify_revision(args.model, protocol['parent_revision'])
    verify_revision(Path(__file__).resolve().parents[1]/'models/qwen3_5_2b_base',
                    model.backend_details['base_model_revision'])
    if not args.disable_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else:
        model.decision.torso.gradient_checkpointing_disable()
    protocol['architecture'] = model.backend_details
    base=Path(json.loads((args.data.parent/'protocol.json').read_text())['base_release'])
    external,external_protocol=frozen_external(base,{r['work_id'] for r in data['train']+data['dev']})
    protocol['external_development_assessment']=external_protocol
    if args.resume:
        saved_protocol = json.loads((out/'protocol.json').read_text())
        for field in ('physical_microbatch_pairs', 'length_adaptive_microbatch', 'length_adaptive_capabilities',
                      'author_batching_amendment', 'warm_start_policy'):
            if field in saved_protocol:
                protocol[field] = saved_protocol[field]
    continuation = None
    if args.continue_from_run:
        previous_protocol = json.loads((args.continue_from_run/'protocol.json').read_text())
        continuation = torch.load(args.continue_from_run/'latest.pt', map_location='cpu', weights_only=False)
        if json.loads(json.dumps(continuation['protocol'])) != previous_protocol or any(r['train_pairs_covered'] != len(encoded['train']) for r in continuation['history']):
            raise ValueError('Continuation lacks an unchanged atomic full-epoch source state')
        protocol['author_batching_amendment'] = {'author_approved': True, 'original_data_sha256': previous_protocol['data_sha256'],
            'previous_run': str(args.continue_from_run.resolve()), 'previous_protocol_sha256': file_digest(args.continue_from_run/'protocol.json'),
            'complete_checkpoint_sha256': file_digest(args.continue_from_run/'latest.pt'),
            'completed_full_epochs_restored': continuation['epoch'], 'unchanged_execution_claimed': False,
            'preserved_state': ['adapter', 'optimizer', 'scheduler', 'python_rng', 'torch_rng', 'cuda_rng', 'history', 'best_dev', 'patience_elapsed']}
        protocol['warm_start_policy'] = 'Restore complete optimizer, scheduler and RNG from the previous full epoch; explicit batching amendment'
        verify_batching_continuation(previous_protocol, protocol)
    if args.resume and json.loads((out/'protocol.json').read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError('Saved run protocol changed before resume')
    save(out/'protocol.json', protocol)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if (args.initial_checkpoint and not args.resume) or continuation is not None:
        initial = continuation if continuation is not None else torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False)
        if initial['protocol']['parent_revision']!=protocol['parent_revision'] or initial['protocol']['architecture']!=model.backend_details:
            raise ValueError('Warm-start model provenance or trainable architecture differs')
        target_parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
        if set(target_parameters)!=set(initial['adapter']):raise ValueError('Warm-start trainable parameters differ')
        with torch.no_grad():
            for name,p in target_parameters.items():p.copy_(initial['adapter'][name].to(p.device))
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=.5, min_lr=1e-7)
    if args.auto_microbatch and not args.resume:
        if args.length_adaptive_microbatch:
            capabilities, trials = profile_length_adaptive_batches(model, encoded, tokenizer, parameters, train_targets,
                                                                  args, free * 2**20)
            count = max(capabilities)
            protocol['length_adaptive_capabilities'] = capabilities
        else:
            count, trials = profile_three_way_batches(model, encoded, tokenizer, parameters, train_targets,
                                                      args, free * 2**20)
        protocol['physical_microbatch_pairs'] = count
        save(out/'batching_preflight.json', {'selected_pairs': count, 'trials': trials,
             'optimizer_updates': 0, 'eight_pairs_per_update_unchanged': True,
             'native_three_pointer_slots_checked': True, 'test_scored': False})
        save(out/'protocol.json', protocol)
        print(json.dumps({'status': 'native_training_batch_verified', 'physical_pairs': count}), flush=True)
        if args.length_adaptive_microbatch:
            benchmark = benchmark_length_adaptive_batches(model, encoded, tokenizer, parameters, train_targets, capabilities)
            # Retain the current fixed-two recipe if the candidate gives no useful speed gain.
            if benchmark['measured_speedup'] < 1.10:
                protocol['length_adaptive_microbatch'] = False
                protocol['physical_microbatch_pairs'] = 2
                args.maximum_padded_tokens = 65536
                protocol['maximum_padded_tokens_per_microbatch'] = args.maximum_padded_tokens
                benchmark['adopted'] = False
            else:
                benchmark['adopted'] = True
        else:
            benchmark = benchmark_three_way_batches(model, encoded, tokenizer, parameters, train_targets,
                                                    count, args.disable_gradient_checkpointing)
        save(out/'batching_speed_benchmark.json', benchmark)
        save(out/'protocol.json', protocol)
        print(json.dumps({'status': 'training_speed_measured', 'speedup': benchmark['measured_speedup']}), flush=True)
    def logits(inputs):
        return model.decision(**{k:v.cuda() for k,v in inputs.items()},temperature=1.)['logits'][0].float()
    def evaluate():
        model.eval(); predictions=[]; losses=[]
        with torch.no_grad():
            for row,inputs in encoded['dev']:
                ab,ba = logits(inputs[0]),logits(inputs[1])
                z = (ab+swap_options(ba))/2
                p = z.softmax(-1)
                target = dev_targets[row['pair_id']]
                if row['pair_id'] in selection_ids:
                    losses.append(float(soft_ce(z,torch.tensor(target,device='cuda'))))
                label = 'tie' if genuine[row['pair_id']] else ('A' if target[0]>.5 else 'B' if target[1]>.5 else None)
                predictions.append({'pair_id':row['pair_id'], 'human_label':label,
                    'human_distribution':target, 'probabilities':p.cpu().tolist(),
                    'prediction':CLASSES[int(p.argmax())],
                    'raw_order_agreement':int(ab.argmax())==int(swap_options(ba).argmax()),
                    'scope':row.get('scope',row.get('script_extent_a'))})
        primary = [r for r in predictions if r['pair_id'] in selection_ids]
        return sum(losses)/len(losses), predictions, metrics(primary)
    if not args.resume and continuation is None:
        loss,predictions,summary = evaluate()
        save(out/'before_training_dev_report.json', {'loss':loss,'metrics':summary,'predictions':predictions,
            'test_scored':False,'calibration_established':False})
    longest = max(encoded['train'],key=lambda r:max(i['input_ids'].shape[1] for i in r[1]))
    model.train()
    for reverse in (False,True):
        target=torch.tensor(train_targets[longest[0]['pair_id']],device='cuda')
        soft_ce(logits(longest[1][reverse]),swap_options(target) if reverse else target).backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
        raise RuntimeError('Nonfinite longest-TRAIN preflight gradients')
    optimizer.zero_grad(set_to_none=True)
    save(out/'model_preflight.json', {'longest_train_pair_id':longest[0]['pair_id'],
         'finite_gradients':True,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'optimizer_updates':0})
    real_weight = sum(float(r.get('weight',1)) for r,_ in encoded['train'] if not r.get('synthetic'))
    syn_weight = sum(float(r.get('weight',1)) for r,_ in encoded['train'] if r.get('synthetic'))
    scale = min(1.,real_weight/syn_weight) if syn_weight else 1.
    mean_weight=(real_weight+syn_weight*scale)/len(encoded['train'])
    best,bad,history,best_epoch,start = math.inf,0,[],0,1
    if args.resume:
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if state['protocol'] != protocol or any(r['train_pairs_covered']!=len(encoded['train']) for r in state['history']):
            raise ValueError('Resume protocol or complete-epoch coverage changed')
        restore_training_state(model,optimizer,state,protocol['data_sha256'],protocol)
        scheduler.load_state_dict(state['scheduler'])
        best,bad,history,best_epoch,start=state['best_dev'],state['patience_elapsed'],state['history'],state['best_epoch'],state['epoch']+1
    elif continuation is not None:
        restore_training_state(model, optimizer, continuation, protocol['data_sha256'], continuation['protocol'])
        scheduler.load_state_dict(continuation['scheduler'])
        best,bad,history,best_epoch,start = continuation['best_dev'],continuation['patience_elapsed'],continuation['history'],continuation['best_epoch'],continuation['epoch']+1
        shutil.copy2(args.continue_from_run/'best.pt', out/'best.pt')
        shutil.copy2(args.continue_from_run/'before_training_dev_report.json', out/'before_training_dev_report.json')
        save(out/'history.json', history)
        save(out/'continuation_record.json', {**protocol['author_batching_amendment'],
             'restored_complete_epochs': continuation['epoch'], 'next_epoch': start,
             'optimizer_scheduler_rng_restored': True, 'test_scored': False})
        del continuation
    train_lengths = [max(v['input_ids'].shape[1] for v in pair) for _, pair in encoded['train']]
    for epoch in range(start,201):
        epoch_started = time.perf_counter()
        model.train(); order=list(range(len(encoded['train']))); random.shuffle(order)
        train_loss=0.; updates=0
        for offset in range(0,len(order),8):
            chunk=order[offset:offset+8]; optimizer.zero_grad(set_to_none=True)
            if protocol['length_adaptive_microbatch']:
                capabilities = {int(k): v for k, v in protocol['length_adaptive_capabilities'].items()}
                groups = length_adaptive_groups(chunk, train_lengths, capabilities)
            else:
                groups = None
            position = 0
            while position < len(chunk):
                if groups is not None:
                    indices = groups.pop(0)
                    count = len(indices)
                else:
                    count = min(protocol['physical_microbatch_pairs'], len(chunk) - position)
                    lengths = [train_lengths[index] for index in chunk[position:position + count]]
                    count = fitting_microbatch_count(lengths, count, args.maximum_padded_tokens)
                    indices = chunk[position:position + count]
                micro = [encoded['train'][index] for index in indices]
                weights = torch.tensor([float(row.get('weight', 1)) * (scale if row.get('synthetic') else 1.)
                                        / mean_weight for row, _ in micro], device='cuda')
                target = torch.tensor([train_targets[row['pair_id']] for row, _ in micro], device='cuda')
                for reverse in (False,True):
                    inputs = padded_pair_inputs([pair[reverse] for _, pair in micro], tokenizer.pad_token_id)
                    values = model.decision(**{k: v.cuda() for k, v in inputs.items()}, temperature=1.)['logits'].float()
                    losses = soft_ce(values, swap_options(target) if reverse else target)
                    if not torch.isfinite(losses).all(): raise RuntimeError('Nonfinite fitting loss')
                    loss = weighted_training_loss(values, swap_options(target) if reverse else target,
                                                  weights, len(chunk))
                    loss.backward(); train_loss += float((losses.detach() * weights).sum()) / 2
                position += count
            torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();updates+=1
            if updates == 1 or updates % 40 == 0:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - epoch_started
                save(out/'progress.json', {'phase': 'training', 'epoch': epoch, 'completed_full_epochs': epoch - 1,
                     'pairs_covered_in_current_epoch': min(offset + len(chunk), len(order)),
                     'pairs_per_epoch': len(order), 'optimizer_updates_in_current_epoch': updates,
                     'physical_microbatch_pairs': protocol['physical_microbatch_pairs'],
                     'length_adaptive_microbatch': protocol['length_adaptive_microbatch'],
                     'epoch_elapsed_seconds': elapsed, 'seconds_per_update_so_far': elapsed / updates,
                     'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                     'patience_full_epochs': args.early_stopping_patience, 'test_scored': False})
        dev_loss,predictions,summary=evaluate();scheduler.step(dev_loss)
        improved=dev_loss<best-1e-5
        if improved:best,bad,best_epoch=dev_loss,0,epoch
        else:bad+=1
        record={'epoch':epoch,'train_pairs_covered':len(order),'optimizer_updates':updates,
            'train_loss':train_loss/len(order),'dev_three_class_ce':dev_loss,
            'lr':optimizer.param_groups[0]['lr'],'metrics':summary,'predictions':predictions,
            'epoch_elapsed_seconds': time.perf_counter() - epoch_started}
        history.append(record)
        state={'adapter':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'history':history,
            'best_dev':best,'best_epoch':best_epoch,'patience_elapsed':bad,'protocol':protocol,
            'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        save_checkpoint(state,out/'latest.pt')
        if improved:save_checkpoint(state,out/'best.pt')
        save(out/'history.json',history)
        print(json.dumps({k:v for k,v in record.items() if k!='predictions'}),flush=True)
        if bad>=args.early_stopping_patience and epoch>=4:break
    selected=next(r for r in history if r['epoch']==best_epoch)
    original_rows=[r for r in selected['predictions'] if r['pair_id'] in original_selection_ids]
    new_rows=[r for r in selected['predictions'] if r['pair_id'] in {x['pair_id'] for x in added_dev}]
    selected_state=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:p.copy_(selected_state['adapter'][name].to(p.device))
    model.eval();external_predictions=[]
    with torch.no_grad():
        for row in external:
            inputs=[encode_three(tokenizer,row,r) for r in (False,True)]
            if max(i['input_ids'].shape[1] for i in inputs)>args.max_tokens:
                raise ValueError('Frozen external input exceeds complete-context limit')
            ab,ba=logits(inputs[0]),logits(inputs[1]);p=((ab+swap_options(ba))/2).softmax(-1)
            target=target_distribution(row)
            label='tie' if genuine_tie(row) else 'A' if row['preference_probability']>.5 else 'B' if row['preference_probability']<.5 else None
            external_predictions.append({'pair_id':row['pair_id'],'human_label':label,
                'human_distribution':target,'probabilities':p.cpu().tolist(),
                'prediction':CLASSES[int(p.argmax())], 'scope':row.get('scope',row.get('script_extent_a')),
                'raw_order_agreement':int(ab.argmax())==int(swap_options(ba).argmax())})
    save(out/'external_development_report.json',{'metrics':metrics(external_predictions),
        'predictions':external_predictions,'selected_epoch':best_epoch,**external_protocol,'test_scored':False})
    save(out/'dev_report.json',{'selected_epoch':best_epoch,'metrics':selected['metrics'],
        'original_primary_dev_metrics':metrics(original_rows),
        'additional_human_dev_metrics':metrics(new_rows) if new_rows else None,
        'predictions':selected['predictions'],'selection':'Minimum full-epoch primary human DEV three-class loss',
        'human_tie_support_limit':f"Original primary DEV has two explicit human ties; additional human DEV contains {sum(genuine_tie(r) for r in added_dev)} ties. These are development observations, not final TEST.",
        'test_scored':False})
    save(out/'completion.json',{'completed_full_epochs':epoch,'selected_best_epoch':best_epoch,
        'total_optimizer_updates':sum(r['optimizer_updates'] for r in history),'best_dev_loss':best,
        'stop_reason':'full_epoch_dev_plateau' if bad>=args.early_stopping_patience and epoch>=4 else 'epoch_budget',
        'patience_full_epochs': args.early_stopping_patience,
        'optimization_convergence_claimed':False,'test_scored':False})
    threshold_path = args.data.parent/'threshold_comparison.json'
    if threshold_path.exists():
        baseline = json.loads(threshold_path.read_text())
        if baseline['primary_human_dev_n'] != len(original_rows):
            raise ValueError('Threshold comparison has a different human DEV denominator')
        save(out/'threshold_vs_label_retraining.json', {'existing_binary_thresholds':baseline['thresholds'],
            'explicit_label_retraining':metrics(original_rows), 'selected_epoch':best_epoch,
            'selection_used_human_dev':True, 'test_scored':False,
            'limits':'Two human DEV ties; development comparison cannot establish broad tie recall. '
                     'Binary and three-class losses have different targets and are not comparable quality scores.'})


if __name__ == '__main__':
    main()
