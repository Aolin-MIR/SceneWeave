"""Train an open-weight script comparator using ordinary LoRA and soft preferences."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import gc
import torch
from torch.nn import functional as F
from train_adaptation_reward import save_checkpoint, stopping_ready, validate_data, restore_training_state


def pair_text(row, reverse=False):
    """The candidates share the same request and may represent any supplied scope."""
    a, b = row['chosen'], row['rejected']
    if reverse:
        a, b = b, a
    return ('Compare the supplied dramatic writing for its common writing request. '
            'Consider request fulfillment and literary and dramatic quality. '
            'Return the preference for candidate A over candidate B.\n'
            '<writing_request>\n'+row.get('prompt', '')+'\n</writing_request>\n'
            '<candidate_A>\n'+a+'\n</candidate_A>\n<candidate_B>\n'+b+'\n</candidate_B>')


def symmetrize_logits(ab, ba):
    """Antisymmetric logits guarantee complementary probabilities after swapping."""
    return (ab - ba) / 2


def training_target(row, synthetic_label_smoothing=0.0):
    """Soften synthetic TRAIN targets without changing stored or human labels."""
    epsilon = float(synthetic_label_smoothing)
    if not math.isfinite(epsilon) or not 0 <= epsilon < .5:
        raise ValueError('Synthetic label smoothing must be finite and in [0, 0.5)')
    target = float(row['preference_probability'])
    if not math.isfinite(target) or not 0 <= target <= 1:
        raise ValueError('Training preference target must be a finite probability')
    return (1 - 2 * epsilon) * target + epsilon if row.get('synthetic') else target


def padded_pair_inputs(items,pad_id):
    """Right-pad complete independent comparisons; preserve every attention mask."""
    length=max(item['input_ids'].shape[1] for item in items)
    output = {'input_ids':torch.cat([F.pad(item['input_ids'],(0,length-item['input_ids'].shape[1]),value=pad_id) for item in items]),
            'attention_mask':torch.cat([F.pad(item['attention_mask'],(0,length-item['input_ids'].shape[1]),value=0) for item in items])}
    for key in ('opt_idx','n_slots'):
        if key in items[0]:
            output[key] = torch.cat([item[key] for item in items])
    return output


def fitting_microbatch_count(lengths, maximum_pairs, token_budget):
    """Keep every input complete and bound only the padded batch size."""
    count = min(len(lengths), maximum_pairs)
    while count > 1 and count * max(lengths[:count]) > token_budget:
        count -= 1
    return count


def profile_auto_microbatch(model, encoded, tokenizer, parameters, optimizer, args):
    """Choose the largest safe actual-TRAIN batch without changing update size."""
    candidates = sorted(encoded['train'],
                        key=lambda item: max(v['input_ids'].shape[1] for v in item[1]), reverse=True)
    trials = []
    maximum = min(8, args.gradient_accumulation, len(candidates))
    for count in [value for value in (8, 4, 2, 1) if value <= maximum]:
        rows = candidates[:count]
        width = max(inputs['input_ids'].shape[1] for _, pair in rows for inputs in pair)
        if count * width > args.maximum_padded_tokens:
            continue
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = {'pairs': count, 'padded_tokens': count * width, 'optimizer_updates': 0}
        try:
            difference = probe_training_microbatch(model, rows, tokenizer, args)
            if not all(p.grad is None or torch.isfinite(p.grad).all().item() for p in parameters):
                raise RuntimeError('Nonfinite automatic TRAIN microbatch gradients')
            result.update(status='passed', maximum_logit_difference=difference,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            available = getattr(args, 'profile_available_bytes', None)
            if available is not None:
                required = (result['peak_reserved_bytes'] + sum(p.numel()*12 for p in parameters)
                            + args.memory_safety_mib*2**20)
                result['required_bytes_including_optimizer_and_headroom'] = required
                if required > available:
                    result['status'] = 'insufficient_optimizer_and_safety_headroom'
                    trials.append(result)
                    continue
            trials.append(result)
            optimizer.zero_grad(set_to_none=True)
            return count, trials
        except torch.OutOfMemoryError:
            result['status'] = 'out_of_memory_no_optimizer_update'
            trials.append(result)
        except MicrobatchNumericalMismatch as error:
            result.update(status='padding_numerical_mismatch', reason=str(error))
            trials.append(result)
        finally:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError('No numerically valid complete TRAIN microbatch fits available memory')


class MicrobatchNumericalMismatch(RuntimeError):
    """A padded batch changed actual comparison logits beyond tolerance."""


def probe_training_microbatch(model, rows, tokenizer, args):
    """Check complete padded inputs, then measure both-order forward/backward."""
    model.eval()
    with torch.no_grad():
        single = torch.stack([model(**{k:v.cuda() for k,v in inputs[0].items()}).logits.reshape(()).float()
                              for _, inputs in rows])
        batch = padded_pair_inputs([inputs[0] for _, inputs in rows], tokenizer.pad_token_id)
        together = model(**{k:v.cuda() for k,v in batch.items()}).logits.reshape(-1).float()
        difference = float((single-together).abs().max())
        if not torch.allclose(single, together, atol=.005, rtol=.05):
            raise MicrobatchNumericalMismatch('Actual TRAIN single and padded logits exceed tolerance')
    del single, together, batch
    model.train()
    for reverse in (False, True):
        batch = padded_pair_inputs([inputs[reverse] for _, inputs in rows], tokenizer.pad_token_id)
        logits = model(**{k:v.cuda() for k,v in batch.items()}).logits.reshape(-1).float()
        target = torch.tensor([training_target(row, args.synthetic_label_smoothing) for row,_ in rows], device='cuda')
        F.binary_cross_entropy_with_logits(logits, 1-target if reverse else target).backward()
    return difference


def validate_comparator_checkpoint(checkpoint, protocol):
    """Reject changed protocols or incomplete histories before touching existing files."""
    saved = checkpoint['protocol']
    fields = ('data_sha256', 'model_revision', 'model', 'lora', 'max_tokens', 'truncation',
              'eligible_pairs', 'exclusions', 'objective', 'inference', 'max_epochs',
              'patience_full_epochs', 'minimum_epochs', 'lr', 'min_lr', 'scheduler',
              'gradient_accumulation_pairs', 'microbatch_pairs', 'maximum_padded_tokens_per_microbatch',
              'auto_microbatch',
              'synthetic_total_weight', 'synthetic_label_smoothing', 'software_versions', 'backend', 'shared_eligibility_sha256',
              'selection_dev_ids', 'pretraining_exposed_evaluation_sources','implementation_sha256')
    changed = {key for key in fields if saved.get(key, 0.0 if key == 'synthetic_label_smoothing' else None)
               != protocol.get(key, 0.0 if key == 'synthetic_label_smoothing' else None)}
    if changed:
        change = protocol.get('author_stopping_policy_change', {})
        allowed = {'patience_full_epochs', 'implementation_sha256'}
        stopping_change_valid = (change and changed.issubset(allowed)
            and change.get('data_sha256') == protocol['data_sha256']
            and saved.get('patience_full_epochs') == change.get('previous_patience_full_epochs')
            and protocol.get('patience_full_epochs') == change.get('patience_full_epochs')
            and saved.get('minimum_epochs') == change.get('minimum_epochs')
            and saved.get('implementation_sha256') == change.get('original_implementation_sha256')
            and protocol.get('implementation_sha256') == change.get('updated_implementation_sha256'))
        schedule_change = protocol.get('author_scheduler_policy_change', {})
        schedule_change_valid = (schedule_change.get('author_approved') is True
            and changed.issubset({'scheduler', 'implementation_sha256'})
            and schedule_change.get('data_sha256') == protocol['data_sha256']
            and saved.get('scheduler') == schedule_change.get('previous_scheduler')
            and protocol.get('scheduler') == schedule_change.get('scheduler')
            and saved.get('implementation_sha256') == schedule_change.get('original_implementation_sha256')
            and protocol.get('implementation_sha256') == schedule_change.get('updated_implementation_sha256')
            and saved.get('patience_full_epochs') == protocol.get('patience_full_epochs')
            and saved.get('minimum_epochs') == protocol.get('minimum_epochs'))
        if not (stopping_change_valid or schedule_change_valid):
            raise ValueError('Resume changes the recorded data or training protocol')
    if 'scheduler' not in checkpoint:
        raise ValueError('Resume requires the original scheduler state')
    history = checkpoint['history']
    if ([record['epoch'] for record in history] != list(range(1, checkpoint['epoch'] + 1))
            or any(record['train_pairs_covered'] != protocol['eligible_pairs']['train'] for record in history)):
        raise ValueError('Resume history does not establish complete TRAIN epochs')
    selected = checkpoint.get('best_epoch')
    if selected is None:
        matches = [record['epoch'] for record in history if abs(record['dev_soft_bce'] - checkpoint['best_dev']) < 1e-12]
        if not matches:
            raise ValueError('Saved best validation loss has no matching complete epoch')
        selected = matches[0]
    if selected not in {record['epoch'] for record in history}:
        raise ValueError('Saved best checkpoint epoch is outside the completed history')
    return selected


def restore_comparator_checkpoint(model, optimizer, scheduler, checkpoint, protocol):
    """Restore only complete epochs with the same data, objective and optimizer policy."""
    selected = validate_comparator_checkpoint(checkpoint, protocol)
    saved = checkpoint['protocol']
    restore_training_state(model, optimizer, checkpoint, protocol['data_sha256'], saved)
    scheduler.load_state_dict(checkpoint['scheduler'])
    change = protocol.get('author_scheduler_policy_change')
    if change and saved.get('scheduler') != protocol['scheduler']:
        scheduler.patience = change['scheduler_patience']
        # Apply a reduction already due at the boundary without counting DEV twice.
        if scheduler.num_bad_epochs > scheduler.patience:
            for group in optimizer.param_groups:
                group['lr'] = max(group['lr'] * scheduler.factor, protocol['min_lr'])
            scheduler.num_bad_epochs = 0
            scheduler.cooldown_counter = scheduler.cooldown
            scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
    return checkpoint['best_dev'], checkpoint['patience_elapsed'], checkpoint['history'], selected


def development_report(data, history, teacher_report=None, bootstrap_draws=2000, selected_epoch=None):
    """Evaluate the selected checkpoint on every real DEV pair, retaining failures."""
    from collections import defaultdict
    from calibrate_contextual_script_preferences import summarize
    best = (next(record for record in history if record['epoch']==selected_epoch) if selected_epoch is not None
            else min(history, key=lambda epoch: epoch['dev_soft_bce']))
    by_id = {row['pair_id']:row for row in best['predictions']}
    rows=[]
    for source in data['dev']:
        prediction=by_id.get(source['pair_id'])
        p=prediction['probability_a'] if prediction else .5
        rows.append({'pair_id':source['pair_id'],'work_id':source['work_id'],
            'human_p_a':source['preference_probability'],'probability_a':p,
            'decided':prediction is not None and p!=.5,'order_stable':prediction is not None,
            'dataset':source.get('source_dataset',source.get('dataset','unknown')),
            'language':source.get('language','unknown'),
            'scope':source.get('script_extent_a','unknown') if source.get('script_extent_a')==source.get('script_extent_b') else 'mixed',
            'missing_prediction':prediction is None,
            'raw_order_agreement':prediction.get('raw_order_agreement') if prediction else None,
            'position_offset':prediction.get('position_offset') if prediction else None})
    groups=defaultdict(list)
    for row in rows:groups[row['work_id']].append(row)
    rng=random.Random(619)
    keys=sorted(groups)
    samples=[[r for key in rng.choices(keys,k=len(keys)) for r in groups[key]]
             for _ in range(bootstrap_draws)]
    def interval(values):
        values=sorted(values)
        return [values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]]
    def accuracy(sample):
        decisive=[r for r in sample if r['human_p_a']!=.5]
        return sum(r['decided'] and (r['probability_a']>.5)==(r['human_p_a']>.5)
                   for r in decisive)/max(1,len(decisive))
    def brier(sample):
        return sum((r['probability_a']-r['human_p_a'])**2 for r in sample)/len(sample)
    report={'selected_epoch':best['epoch'],'selection':'Minimum full-epoch real DEV soft BCE',
        'metrics':summarize(rows),'rows':rows,'test_scored':False,
        'uncertainty':{'unit':'Original work_id group','bootstrap_draws':bootstrap_draws,
            'accuracy_interval95':interval([accuracy(sample) for sample in samples]),
            'brier_interval95':interval([brier(sample) for sample in samples]),
            'interpretation':'Descriptive DEV uncertainty; repeated development and checkpoint selection are not independent TEST evidence'},
        'scope_metrics':{},'source_metrics':{},'language_metrics':{},'comparisons':{}}
    native = [row for row in rows if row['raw_order_agreement'] is not None]
    calibration=[]
    for index in range(5):
        lo,hi=index/5,(index+1)/5
        subset=[r for r in rows if lo<=r['probability_a']<hi or index==4 and r['probability_a']==1]
        if subset:
            calibration.append({'probability_bin':[lo,hi],'pairs':len(subset),
                'mean_predicted_preference':sum(r['probability_a'] for r in subset)/len(subset),
                'mean_observed_human_preference':sum(r['human_p_a'] for r in subset)/len(subset),
                'missing_predictions':sum(r['missing_prediction'] for r in subset)})
    report['probability_diagnostics']={'reliability_bins':calibration,
        'mean_absolute_binned_gap':sum(bin['pairs']*abs(bin['mean_predicted_preference']-bin['mean_observed_human_preference'])
                                      for bin in calibration)/len(rows),
        'probability_calibration_established':False,
        'interpretation':'Descriptive soft-human-target reliability; small bins and DEV selection limit certainty'}
    report['order_diagnostics'] = {
        'returned_probability_swap_consistency': 'Guaranteed by the specified antisymmetric inference rule; not aesthetic validation',
        'raw_order_agreement_pairs': len(native),
        'raw_order_agreement_rate': sum(row['raw_order_agreement'] for row in native)/len(native) if native else None,
        'mean_absolute_position_offset': sum(abs(row['position_offset']) for row in native)/len(native) if native else None}
    for key,destination in [('scope','scope_metrics'),('dataset','source_metrics'),('language','language_metrics')]:
        for value in sorted({r[key] for r in rows}):
            subset=[r for r in rows if r[key]==value]
            if any(r['human_p_a']!=.5 for r in subset):report[destination][value]=summarize(subset)
    if teacher_report:
        teachers={r['pair_id']:r for r in teacher_report['rows']}
        corrected = any('teacher_probability_a' in row for row in teachers.values())
        names = ('human_preference_corrected_teacher', 'raw_api_teacher') if corrected else ('qualified_api_teacher',)
        for name in names:
            comparison={}
            for row in rows:
                teacher=teachers.get(row['pair_id'],{})
                p=teacher.get('teacher_probability_a',teacher.get('probability_a',.5)) if name=='raw_api_teacher' else teacher.get('probability_a',.5)
                comparison[row['pair_id']]={**row,'probability_a':p,'decided':'probability_a' in teacher and not teacher.get('error') and p!=.5,
                                           'order_stable':teacher.get('order_stable',False)}
            paired_accuracy=[];paired_brier=[]
            for sample in samples:
                other=[comparison[r['pair_id']] for r in sample]
                paired_accuracy.append(accuracy(sample)-accuracy(other))
                paired_brier.append(brier(sample)-brier(other))
            other=list(comparison.values())
            report['comparisons'][name]={'metrics':summarize(other),
                'accuracy_difference':accuracy(rows)-accuracy(other),
                'accuracy_difference_interval95':interval(paired_accuracy),
                'brier_difference':brier(rows)-brier(other),
                'brier_difference_interval95':interval(paired_brier),
                'identical_real_dev_denominator':True,
                'teacher_kind':'Frozen API judgments with real human demonstrations' if name=='qualified_api_teacher' else name}
    metrics=report['metrics']
    report['development_gate_passed']=(metrics['n']>=32 and not any(r['missing_prediction'] for r in rows)
        and metrics['accuracy_all_pairs']>=.8 and metrics['accuracy_wilson95_lower']>=.7
        and metrics['balanced_accuracy']>=.75 and metrics['paired_majority_baseline_one_sided_exact_p']<.05)
    report['development_gate_scope']='Model development only; full frozen real cohort, Wilson lower bound >=.7. The 96-retained-case teacher production gate remains unchanged.'
    return report


def build_model(model_path, backend='scalar'):
    """Load the recorded released model and its comparison adapters for train or inference."""
    from script_preference_backends import build_backend
    return build_backend(model_path, backend)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=['scalar','decider'], default='scalar')
    parser.add_argument('--minimum-free-mib', type=int, default=32000)
    parser.add_argument('--shared-eligibility', help='Frozen cross-model full-input eligibility JSON')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--model-preflight', action='store_true')
    parser.add_argument('--teacher-report')
    parser.add_argument('--gradient-accumulation',type=int,default=8)
    parser.add_argument('--microbatch-pairs',type=int,default=2)
    parser.add_argument('--auto-microbatch', action='store_true')
    parser.add_argument('--maximum-padded-tokens', type=int)
    parser.add_argument('--memory-safety-mib', type=int, default=3072)
    parser.add_argument('--synthetic-label-smoothing', type=float, default=0.0)
    parser.add_argument('--resume', help='Complete full-epoch checkpoint from this exact run')
    parser.add_argument('--early-stopping-patience', type=int, default=3)
    parser.add_argument('--minimum-epochs', type=int, default=4)
    parser.add_argument('--stopping-policy-change', help='Recorded author authorization for a stopping-only change')
    parser.add_argument('--scheduler-patience', type=int, default=3)
    parser.add_argument('--scheduler-policy-change', help='Recorded author authorization for a scheduler-only change')
    parser.add_argument('--finalize-only', action='store_true', help='Assess an already eligible full-epoch checkpoint without updates')
    args = parser.parse_args()
    training_target({'preference_probability': 1., 'synthetic': True}, args.synthetic_label_smoothing)
    if args.gradient_accumulation<1 or args.microbatch_pairs<1:raise ValueError('Training batch sizes must be positive')
    args.maximum_padded_tokens = args.maximum_padded_tokens or args.max_tokens
    if args.maximum_padded_tokens < args.max_tokens or args.memory_safety_mib < 0:
        raise ValueError('Complete inputs and nonnegative GPU headroom are required')
    if args.auto_microbatch:
        args.microbatch_pairs = min(8, args.gradient_accumulation)
        # Import optimized operators before lazy Transformers model imports.
        # The fallback decorator freezes its choice when the model module loads.
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        import inspect
        import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_module
        bindings = inspect.getclosurevars(qwen_module.torch_chunk_gated_delta_rule).nonlocals
        if bindings.get('is_new_implementation') is not True:
            raise RuntimeError('Qwen linear attention did not bind the installed optimized FLA operators')
    if args.early_stopping_patience < 1 or args.minimum_epochs < 1:
        raise ValueError('Stopping policy requires positive complete-epoch counts')
    if args.scheduler_patience < 0:
        raise ValueError('Scheduler patience must be nonnegative')
    if args.finalize_only and not args.resume:
        raise ValueError('Final assessment requires a complete-epoch checkpoint')
    from transformers import AutoTokenizer
    from script_preference_backends import encode_pair
    data = validate_data(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'latest.pt').exists() and not args.resume:
        raise RuntimeError('Do not overwrite a trained run')
    resume = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if args.resume and Path(args.resume).resolve() != (out/'latest.pt').resolve():
        raise ValueError('Resume must use this run\'s latest complete-epoch checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded, exclusions = {}, []
    eligibility = json.loads(Path(args.shared_eligibility).read_text()) if args.shared_eligibility else None
    if eligibility and eligibility['data_sha256'] != hashlib.sha256(Path(args.data).read_bytes()).hexdigest():
        raise ValueError('Shared eligibility does not match the training data')
    exposed_sources=set(eligibility.get('pretraining_exposed_evaluation_sources',[])) if eligibility else set()
    selection_ids=set(eligibility['selection_dev_ids']) if eligibility and 'selection_dev_ids' in eligibility else {r['pair_id'] for r in data['dev']}
    primary_data={'dev':[r for r in data['dev'] if r['pair_id'] in selection_ids]}
    if not primary_data['dev']:
        raise ValueError('No unexposed real DEV observations for selection')
    assessment=None
    qualification=Path(args.data).parent/'contextual_calibration_protocol.json'
    if qualification.exists():
        qualifier=json.loads(qualification.read_text())
        if qualifier.get('domain_audit_blind_to_human_votes_and_previous_judgments'):
            source=Path(qualifier['human_training_data_path'])
            if hashlib.sha256(source.read_bytes()).hexdigest()!=qualifier['human_training_data_sha256']:
                raise ValueError('Qualified domain data changed before external assessment')
            original=json.loads(source.read_text())
            ids=set(qualifier['confirmation_ids'])
            if 'assessment_confirmation_ids' in globals():
                ids=assessment_confirmation_ids(qualifier)
            assessment=[{**r,'assessment_role':'Teacher-qualified groups excluded from student TRAIN/DEV'}
                        for r in original['train'] if r['pair_id'] in ids]
            extra=source.parent.parent/'contextual_fragment_human_votes_v2/released_contextual_preferences.json'
            if extra.exists():
                assessment.extend({**r,'assessment_role':'Native human source validation; not used for teacher or student selection'}
                    for r in json.loads(extra.read_text())['additional_source_validation'])
            if len({r['pair_id'] for r in assessment})!=len(assessment) or not ids.issubset({r['pair_id'] for r in assessment}):
                raise ValueError('External human assessment identities differ')
            used={r['work_id'] for split in ('train','dev') for r in data[split]}
            if any(r['work_id'] in used for r in assessment):raise ValueError('External assessment group leaked to TRAIN/DEV')
            assessment_protocol = {'pair_ids':[r['pair_id'] for r in assessment],
                'primary_excluded_pretraining_sources':sorted(exposed_sources),
                'primary_pair_ids':[r['pair_id'] for r in assessment if r.get('source_dataset',r.get('dataset')) not in exposed_sources],
                'selection':'Frozen before training; evaluate best real DEV checkpoint only after training',
                'previous_teacher_access':'128 groups used for teacher qualification; no claim of fully independent final TEST',
                'source_validation_count':sum(r['assessment_role'].startswith('Native') for r in assessment),
                'test_scored':False}
            assessment_path = out/'external_assessment_protocol.json'
            if assessment_path.exists() and json.loads(assessment_path.read_text()) != assessment_protocol:
                raise ValueError('Previously frozen external assessment differs')
            assessment_path.write_text(json.dumps(assessment_protocol,indent=2)+'\n')
    for split in ('train', 'dev'):
        encoded[split] = []
        for row in data[split]:
            if eligibility and row['pair_id'] not in eligibility['eligible_ids'][split]:
                exclusions.append({'split':split,'pair_id':row['pair_id'],'reason':'Common cross-model full-input eligibility'})
                continue
            inputs = [encode_pair(tokenizer,row,r,args.backend) for r in (False, True)]
            lengths = [i['input_ids'].shape[1] for i in inputs]
            if max(lengths) > args.max_tokens:
                exclusions.append({'split': split, 'pair_id': row['pair_id'], 'token_lengths': lengths,
                                   'reason': 'Cannot fit both untruncated comparison orders'})
                continue
            encoded[split].append((row, inputs))
    if not encoded['train'] or not encoded['dev']:
        raise ValueError('No usable training or development pairs')
    protocol = {'data_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                'objective': 'Bidirectional soft binary cross entropy; direct contextual pair prediction',
                'inference': 'sigmoid((g(request,A,B)-g(request,B,A))/2)',
                'model': args.model, 'backend': args.backend,
                'lora': {'implementation': 'Ordinary LoRA; preserve pretrained decision adapters and heads where provided'},
                'shared_eligibility_sha256':hashlib.sha256(Path(args.shared_eligibility).read_bytes()).hexdigest() if args.shared_eligibility else None,
                'selection_dev_ids':sorted(selection_ids),
                'pretraining_exposed_evaluation_sources':sorted(exposed_sources),
                'evaluation_source_policy':'Exclude known released decision-model post-training sources from common primary DEV/external; retain as diagnostics',
                'max_tokens': args.max_tokens, 'truncation': False, 'exclusions': exclusions,
                'eligible_pairs': {k: len(v) for k, v in encoded.items()}, 'test_scored': False,
                'max_epochs': 200, 'patience_full_epochs': args.early_stopping_patience, 'minimum_epochs': args.minimum_epochs,
                'stopping_basis': 'Author-selected consecutive full-epoch DEV stagnation; best checkpoint retained',
                'lr': 1e-5, 'min_lr': 1e-7,
                'scheduler': f'ReduceLROnPlateau, patience{args.scheduler_patience}, factor0.5',
                'gradient_accumulation_pairs':args.gradient_accumulation,
                'microbatch_pairs':args.microbatch_pairs,'maximum_padded_tokens_per_microbatch':args.maximum_padded_tokens,
                'auto_microbatch':args.auto_microbatch,
                'batching':'Independent fully masked sequences; same per-pair weighted objective and full epoch coverage',
                'software_versions': {name:importlib.metadata.version(name) for name in
                    ('torch','transformers','peft','triton','flash-linear-attention','causal-conv1d')},
                'implementation_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                    for name in ('train_contextual_script_comparator.py','script_preference_backends.py','train_adaptation_reward.py')},
                'synthetic_total_weight': 'Equal to total real TRAIN weight; unique pairs covered every epoch',
                'synthetic_label_smoothing': args.synthetic_label_smoothing,
                'target_policy': 'Synthetic TRAIN q -> (1-2*epsilon)*q+epsilon; real targets and all evaluation targets unchanged'}
    if args.stopping_policy_change:
        change = json.loads(Path(args.stopping_policy_change).read_text())
        if (change['data_sha256'] != protocol['data_sha256']
            or change['patience_full_epochs'] != args.early_stopping_patience
            or change['minimum_epochs'] != args.minimum_epochs
            or change['updated_implementation_sha256'] != protocol['implementation_sha256']):
            raise ValueError('Author stopping change differs from the actual data or implementation')
        protocol['author_stopping_policy_change'] = change
    if args.scheduler_policy_change:
        change = json.loads(Path(args.scheduler_policy_change).read_text())
        if (change.get('author_approved') is not True
            or change.get('data_sha256') != protocol['data_sha256']
            or change.get('scheduler_patience') != args.scheduler_patience
            or change.get('scheduler') != protocol['scheduler']
            or change.get('updated_implementation_sha256') != protocol['implementation_sha256']):
            raise ValueError('Author scheduler change differs from the actual data or implementation')
        protocol['author_scheduler_policy_change'] = change
    provenance = json.loads((Path(args.model)/'source_revision.json').read_text())
    index_path = Path(args.model)/'model.safetensors.index.json'
    files = (set(json.loads(index_path.read_text())['weight_map'].values()) | {'config.json'}
             if index_path.exists() else {'head.safetensors','lora/adapter_model.safetensors',
                'strands_decider_config.json' if (Path(args.model)/'strands_decider_config.json').exists() else 'hobson_config.json'})
    for shard in files:
        metadata = Path(args.model)/'.cache/huggingface/download'/(shard+'.metadata')
        if metadata.read_text().splitlines()[0] != provenance['revision']:
            raise ValueError('Local base-model revision differs from provenance')
    protocol['model_revision'] = provenance['revision']
    if resume:
        validate_comparator_checkpoint(resume, protocol)
    if not resume:
        (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    if args.preflight:
        print(json.dumps(protocol))
        return
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    if free < args.minimum_free_mib:
        raise RuntimeError('Insufficient measured free GPU memory')
    args.profile_available_bytes = free*2**20
    if args.auto_microbatch:
        total = torch.cuda.get_device_properties(0).total_memory
        allowed = max(0, free-args.memory_safety_mib)*2**20
        if allowed <= 0:
            raise RuntimeError('No GPU capacity remains after the declared safety headroom')
        torch.cuda.set_per_process_memory_fraction(min(1., allowed/total))
        (out/'resource_memory_policy.json').write_text(json.dumps({
            'measured_initial_free_mib': free, 'safety_mib': args.memory_safety_mib,
            'process_allocation_limit_bytes': allowed, 'unrelated_processes_managed': False})+'\n')
    random.seed(42)
    torch.manual_seed(42)
    model = build_model(args.model, args.backend)
    protocol['architecture'] = model.backend_details
    if resume and resume['protocol'].get('architecture') != model.backend_details:
        raise ValueError('Resume changes the pretrained head or adaptation architecture')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=args.scheduler_patience, min_lr=1e-7)
    real_weight = sum(float(r.get('weight', 1)) for r, _ in encoded['train'] if not r.get('synthetic'))
    synthetic_weight = sum(float(r.get('weight', 1)) for r, _ in encoded['train'] if r.get('synthetic'))
    synthetic_scale = min(1., real_weight / synthetic_weight) if synthetic_weight else 1.
    mean_weight = (real_weight + synthetic_weight*synthetic_scale)/len(encoded['train'])
    best, bad, history, best_epoch = math.inf, 0, [], 0
    if resume:
        args.microbatch_pairs = resume['protocol'].get('effective_microbatch_pairs', args.microbatch_pairs)
    def logit(inputs):
        return model(**{k: v.cuda() for k, v in inputs.items()}).logits.reshape(()).float()
    if not resume and not args.model_preflight:
        model.eval()
        if model.config.model_type == 'llama' and model.backend_details['pretrained_head_preserved']:
            native_predictions=[];native_losses=[]
            with torch.no_grad():
                for row, _ in encoded['dev']:
                    raw=[tokenizer(row[key].strip(),return_tensors='pt',truncation=False)
                         for key in ('chosen','rejected')]
                    if max(item['input_ids'].shape[1] for item in raw)>args.max_tokens:
                        continue
                    z=logit(raw[0])-logit(raw[1])
                    native_predictions.append({'pair_id':row['pair_id'],'probability_a':float(z.sigmoid())})
                    if row['pair_id'] in selection_ids:
                        native_losses.append(float(F.binary_cross_entropy_with_logits(z,
                            torch.tensor(float(row['preference_probability']),device='cuda'))))
            native=development_report(primary_data,[{'epoch':0,'dev_soft_bce':sum(native_losses)/len(native_losses),
                'predictions':native_predictions}],selected_epoch=0)
            native.update(selection='Published single-story scoring: raw candidate text, no prompt wrapper or chat template',
                inference='sigmoid(reward(A)-reward(B))',common_request_used=False,
                probability_calibration_established=False)
            (out/'native_reward_before_training_dev_report.json').write_text(json.dumps(native,indent=2)+'\n')
        baseline_predictions=[];baseline_losses=[]
        with torch.no_grad():
            for row, inputs in encoded['dev']:
                ab,ba=logit(inputs[0]),logit(inputs[1]);z=symmetrize_logits(ab,ba)
                baseline_predictions.append({'pair_id':row['pair_id'],'probability_a':float(z.sigmoid()),
                    'raw_order_agreement':bool((ab>0)!=(ba>0)) or bool((ab==0)&(ba==0)),
                    'position_offset':float((ab+ba)/2)})
                if row['pair_id'] in selection_ids:
                    baseline_losses.append(float(F.binary_cross_entropy_with_logits(z,
                        torch.tensor(float(row['preference_probability']),device='cuda'))))
        baseline=development_report(primary_data,[{'epoch':0,'dev_soft_bce':sum(baseline_losses)/len(baseline_losses),
            'predictions':baseline_predictions}],selected_epoch=0)
        baseline.update(selection='No task-finetuning; pretrained heads retained when present; Qwen has a new random task head',
                        probability_calibration_established=False)
        (out/'before_training_dev_report.json').write_text(json.dumps(baseline,indent=2)+'\n')
    longest = max(encoded['train'], key=lambda item: max(v['input_ids'].shape[1] for v in item[1]))
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for reverse, inputs in enumerate(longest[1]):
        target = training_target(longest[0], args.synthetic_label_smoothing)
        preflight_loss = F.binary_cross_entropy_with_logits(logit(inputs), torch.tensor(1-target if reverse else target, device='cuda'))
        preflight_loss.backward()
    finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in parameters)
    if not finite:
        raise RuntimeError('Nonfinite longest-TRAIN gradients')
    batching_check={'requested_microbatch_pairs':args.microbatch_pairs,'effective_microbatch_pairs':args.microbatch_pairs}
    if args.auto_microbatch and not resume:
        args.microbatch_pairs, trials = profile_auto_microbatch(model, encoded, tokenizer, parameters, optimizer, args)
        batching_check.update(effective_microbatch_pairs=args.microbatch_pairs, actual_train_capacity_trials=trials,
                              same_untruncated_inputs=True, finite_microbatch_gradients=True)
    elif args.microbatch_pairs>1:
        candidates=sorted([item for item in encoded['train'] if max(v['input_ids'].shape[1] for v in item[1])*args.microbatch_pairs<=args.maximum_padded_tokens],
                          key=lambda item:max(v['input_ids'].shape[1] for v in item[1]),reverse=True)[:args.microbatch_pairs]
        if len(candidates)==args.microbatch_pairs:
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad():
                single=torch.stack([logit(inputs[0]) for _,inputs in candidates])
                batch=padded_pair_inputs([inputs[0] for _,inputs in candidates],tokenizer.pad_token_id)
                together=model(**{k:v.cuda() for k,v in batch.items()}).logits.reshape(-1).float()
            batching_check['maximum_logit_difference']=float((single-together).abs().max())
            batching_check['same_untruncated_inputs']=True
            if not torch.allclose(single,together,atol=.005,rtol=.05):
                args.microbatch_pairs=1
                batching_check.update(effective_microbatch_pairs=1,reason='Actual TRAIN padding preflight exceeded numerical agreement tolerance')
            else:
                model.train()
                for reverse in (False,True):
                    batch=padded_pair_inputs([inputs[reverse] for _,inputs in candidates],tokenizer.pad_token_id)
                    z=model(**{k:v.cuda() for k,v in batch.items()}).logits.reshape(-1).float()
                    target=torch.tensor([training_target(r,args.synthetic_label_smoothing) for r,_ in candidates],device='cuda')
                    F.binary_cross_entropy_with_logits(z,1-target if reverse else target).backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all().item() for p in parameters):
                    raise RuntimeError('Nonfinite actual TRAIN microbatch gradients')
                batching_check['finite_microbatch_gradients']=True
        else:
            args.microbatch_pairs=1
            batching_check.update(effective_microbatch_pairs=1,reason='Insufficient TRAIN sequences fitting the padded token allowance')
    protocol['effective_microbatch_pairs']=args.microbatch_pairs
    if resume and args.microbatch_pairs != resume['protocol'].get('effective_microbatch_pairs', resume['protocol']['microbatch_pairs']):
        raise ValueError('Resume would change the effective training microbatch')
    if not resume:
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    elif protocol.get('author_stopping_policy_change') or protocol.get('author_scheduler_policy_change'):
        previous = out/('protocol_before_author_scheduler_change.json'
                        if protocol.get('author_scheduler_policy_change') else 'protocol_before_author_stopping_change.json')
        if not previous.exists():
            previous.write_text((out/'protocol.json').read_text())
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    preflight_path = out/('resume_preflight.json' if resume else 'model_preflight.json')
    preflight_path.write_text(json.dumps({'finite_gradients': finite,'batching_check':batching_check,
        'longest_train_pair_id': longest[0]['pair_id'], 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'optimizer_updates': 0, 'test_scored': False})+'\n')
    optimizer.zero_grad(set_to_none=True)
    if args.model_preflight:
        print(json.dumps({'status':'model_preflight_passed','architecture':model.backend_details,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'test_scored':False}),flush=True)
        return
    start_epoch = 1
    if resume:
        best, bad, history, best_epoch = restore_comparator_checkpoint(model, optimizer, scheduler, resume, protocol)
        best_path = out/'best.pt'
        if resume['epoch'] == best_epoch:
            save_checkpoint(resume, best_path)
        elif not best_path.exists():
            raise ValueError('Exact resume cannot recover missing selected-best weights')
        else:
            selected = torch.load(best_path,map_location='cpu',weights_only=False)
            if (selected['epoch'] != best_epoch or selected['protocol']['data_sha256'] != protocol['data_sha256']):
                raise ValueError('Selected-best weights differ from saved stopping state')
        start_epoch = resume['epoch'] + 1
        (out/'resume_record.json').write_text(json.dumps({'checkpoint':str(Path(args.resume).resolve()),
            'completed_full_epochs_restored':resume['epoch'],'next_epoch':start_epoch,
            'optimizer_scheduler_and_rng_restored':True,
            'partial_epoch_policy':'Discard and replay any interrupted partial epoch from the saved complete-epoch state; do not claim partial-update recovery',
             'author_stopping_policy_change':protocol.get('author_stopping_policy_change'),
             'author_scheduler_policy_change':protocol.get('author_scheduler_policy_change'),
             'unchanged_optimization_protocol': not bool(protocol.get('author_scheduler_policy_change')),
            'finalize_only':args.finalize_only,
            'test_scored':False})+'\n')
    epoch = start_epoch - 1
    terminal = resume is not None and stopping_ready(bad,args.early_stopping_patience,epoch,args.minimum_epochs,optimizer.param_groups[0]['lr'],1e-7,False)
    if args.finalize_only and not terminal:
        raise ValueError('Complete checkpoint has not met the recorded stopping policy')
    for epoch in range(start_epoch, 201) if not terminal else ():
        model.train()
        order = list(range(len(encoded['train'])))
        random.shuffle(order)
        train_loss = 0.
        updates=0
        for start in range(0,len(order),args.gradient_accumulation):
            chunk=order[start:start+args.gradient_accumulation]
            optimizer.zero_grad(set_to_none=True)
            position=0
            while position<len(chunk):
                count=min(args.microbatch_pairs,len(chunk)-position)
                lengths = [max(inputs['input_ids'].shape[1] for inputs in encoded['train'][index][1])
                           for index in chunk[position:position+count]]
                count = fitting_microbatch_count(lengths, count, args.maximum_padded_tokens)
                micro=[encoded['train'][index] for index in chunk[position:position+count]]
                weights=torch.tensor([float(row.get('weight',1))*(synthetic_scale if row.get('synthetic') else 1.)/mean_weight
                                      for row,_ in micro],device='cuda')
                targets=torch.tensor([training_target(row,args.synthetic_label_smoothing) for row,_ in micro],device='cuda')
                for reverse in (False,True):
                    inputs=padded_pair_inputs([items[reverse] for _,items in micro],tokenizer.pad_token_id)
                    logits=model(**{k:v.cuda() for k,v in inputs.items()}).logits.reshape(-1).float()
                    losses=F.binary_cross_entropy_with_logits(logits,1-targets if reverse else targets,reduction='none')
                    if not torch.isfinite(losses).all():
                        raise RuntimeError('Nonfinite training loss')
                    weighted=(losses*weights).sum()
                    (weighted/(2*len(chunk))).backward()
                    train_loss+=float(weighted.detach())/2
                position+=count
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            updates+=1
        model.eval()
        predictions, losses = [], []
        with torch.no_grad():
            for row, inputs in encoded['dev']:
                ab, ba = logit(inputs[0]), logit(inputs[1])
                z = symmetrize_logits(ab, ba)
                target = float(row['preference_probability'])
                if row['pair_id'] in selection_ids:
                    losses.append(float(F.binary_cross_entropy_with_logits(z, torch.tensor(target, device='cuda'))))
                predictions.append({'pair_id': row['pair_id'], 'human_p_a': target,
                                    'probability_a': float(z.sigmoid()), 'source': row.get('source_dataset'),
                                    'raw_order_agreement': bool((ab > 0) != (ba > 0)) or bool((ab == 0) & (ba == 0)),
                                    'position_offset': float((ab + ba)/2)})
        dev_loss = sum(losses)/len(losses)
        scheduler.step(dev_loss)
        improved = dev_loss < best-1e-5
        if improved:
            best, bad = dev_loss, 0
            best_epoch = epoch
        else:
            bad += 1
        record = {'epoch': epoch, 'train_pairs_covered': len(order), 'train_loss': train_loss/len(order),
                  'optimizer_updates':updates,
                  'dev_soft_bce': dev_loss, 'lr': optimizer.param_groups[0]['lr'],
                  'dev_majority_accuracy': sum((r['probability_a']>.5)==(r['human_p_a']>.5) for r in predictions
                    if r['human_p_a'] != .5 and r['pair_id'] in selection_ids)
                  /max(1, sum(r['human_p_a'] != .5 and r['pair_id'] in selection_ids for r in predictions)),
                  'primary_selection_dev_pairs':len(losses),'predictions': predictions}
        history.append(record)
        state = {'adapter': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch,
                 'history': history, 'best_dev': best, 'best_epoch': best_epoch, 'patience_elapsed': bad,
                 'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all(), 'protocol': protocol}
        save_checkpoint(state, out/'latest.pt')
        if improved:
            save_checkpoint(state, out/'best.pt')
        (out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k != 'predictions'}), flush=True)
        if stopping_ready(bad, args.early_stopping_patience, epoch, args.minimum_epochs, optimizer.param_groups[0]['lr'], 1e-7, False):
            break
    (out/'completion.json').write_text(json.dumps({'completed_full_epochs': epoch, 'best_dev_loss': best,
        'total_optimizer_updates':sum(record['optimizer_updates'] for record in history),'selected_best_epoch':best_epoch,
        'stop_reason': 'full_epoch_dev_plateau' if epoch < 200 else 'epoch_budget',
        'optimization_convergence_claimed': False, 'test_scored': False})+'\n')
    completion=json.loads((out/'completion.json').read_text())
    completion.update(patience_full_epochs=args.early_stopping_patience,minimum_epochs=args.minimum_epochs,
                      author_stopping_policy_change=protocol.get('author_stopping_policy_change'),
                      author_scheduler_policy_change=protocol.get('author_scheduler_policy_change'),
                      final_assessment_without_optimizer_updates=args.finalize_only)
    (out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    teacher_path=Path(args.teacher_report) if args.teacher_report else out.parent/'dev_report.json'
    teacher=json.loads(teacher_path.read_text()) if teacher_path.exists() else None
    report=development_report(primary_data,history,teacher,selected_epoch=best_epoch)
    (out/'dev_report.json').write_text(json.dumps(report,indent=2)+'\n')
    all_source_report=development_report(data,history,teacher,selected_epoch=best_epoch)
    all_source_report['scope']='Diagnostic only: includes sources used by the released decision-model post-training'
    (out/'all_sources_dev_diagnostic.json').write_text(json.dumps(all_source_report,indent=2)+'\n')
    external_report=None
    if assessment:
        selected=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        with torch.no_grad():
            for name,parameter in model.named_parameters():
                if parameter.requires_grad:parameter.copy_(selected['adapter'][name].to(parameter.device))
        model.eval();predictions=[]
        with torch.no_grad():
            for row in assessment:
                inputs=[encode_pair(tokenizer,row,r,args.backend) for r in (False,True)]
                if max(item['input_ids'].shape[1] for item in inputs)>args.max_tokens:continue
                ab,ba=logit(inputs[0]),logit(inputs[1])
                p=float(symmetrize_logits(ab,ba).sigmoid())
                predictions.append({'pair_id':row['pair_id'],'probability_a':p,
                    'raw_order_agreement':bool((ab>0)!=(ba>0)) or bool((ab==0)&(ba==0)),
                    'position_offset':float((ab+ba)/2)})
        frozen_epoch=[{'epoch':best_epoch,'dev_soft_bce':best,'predictions':predictions}]
        comparison_file=qualification.parent/'confirmation_report.json'
        external_teacher=json.loads(comparison_file.read_text()) if comparison_file.exists() else None
        primary_assessment=[r for r in assessment if r.get('source_dataset',r.get('dataset')) not in exposed_sources]
        external_report=development_report({'dev':primary_assessment},frozen_epoch,selected_epoch=best_epoch)
        external_report['selection']='Checkpoint selected by original real DEV only; external assessment never selected an epoch'
        external_report['limits']='Teacher qualification already inspected128 groups. Known decision pretraining sources excluded from the primary report. Native source-validation subgroup is separate and small. Original TEST is still sealed.'
        external_report['pretraining_exposed_sources_excluded']=sorted(exposed_sources)
        external_report['full_assessment_pairs']=len(assessment)
        external_report['primary_assessment_pairs']=len(primary_assessment)
        external_report['assessment_roles']={role:development_report({'dev':[r for r in primary_assessment if r['assessment_role']==role]},
            frozen_epoch,selected_epoch=best_epoch,bootstrap_draws=1000)['metrics'] for role in {r['assessment_role'] for r in primary_assessment}}
        if external_teacher:
            common=[r for r in primary_assessment if not r['assessment_role'].startswith('Native')]
            matched=development_report({'dev':common},frozen_epoch,external_teacher,selected_epoch=best_epoch)
            external_report['qualified_group_teacher_comparison']=matched['comparisons']
            external_report['native_validation_teacher_comparison']='No teacher predictions collected for these15 pairs; excluded from paired teacher comparison'
        (out/'external_assessment_report.json').write_text(json.dumps(external_report,indent=2)+'\n')
        diagnostic=development_report({'dev':assessment},frozen_epoch,selected_epoch=best_epoch)
        diagnostic['scope']='Diagnostic only: includes known decision-model post-training sources'
        (out/'all_sources_external_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    (out/'consumer_readiness.json').write_text(json.dumps({'development_gate_passed':report['development_gate_passed'],
        'external_assessment_gate_passed':external_report['development_gate_passed'] if external_report else None,
        'best_checkpoint':str(out/'best.pt'),'metrics':report['metrics'],'test_scored':False,
        'probability_calibration_established':False,'scope':f'Within {args.max_tokens} tokens, with the common request and both whole supplied scripts',
        'limits':'DEV-qualified comparator if the gate passes; long-script ability and final TEST generalization remain unestablished'},indent=2)+'\n')


if __name__ == '__main__':
    main()
