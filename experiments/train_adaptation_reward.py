"""Local LoRA reward training with work-disjoint data and full-epoch validation."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class LowRankLinear(nn.Module):
    """Keep the pretrained projection frozen and learn a zero-initialized update."""
    def __init__(self, original, rank=8):
        super().__init__()
        self.original = original
        self.a = nn.Parameter(torch.empty(rank, original.in_features, device=original.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(original.out_features, rank, device=original.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        self.scale = 2.0

    def forward(self, x):
        update = F.linear(F.linear(x.float(), self.a), self.b)
        return self.original(x) + (self.scale * update).to(x.dtype)


class Float32Score(nn.Linear):
    """Keep the newly trained scalar head and its input in FP32."""
    def __init__(self, original):
        super().__init__(original.in_features, 1, bias=False,
                         device=original.weight.device, dtype=torch.float32)
        nn.init.normal_(self.weight, mean=0., std=.001)

    def forward(self, x):
        return F.linear(x.float(), self.weight)


def restore_training_state(model, optimizer, checkpoint, expected_data_sha, source_protocol):
    """Restore a full-epoch checkpoint after verifying its data and parameter set."""
    if source_protocol['data_sha256'] != expected_data_sha:
        raise ValueError('Resume data differs from the recorded source protocol.')
    required = {'adapter', 'optimizer', 'epoch', 'history', 'best_dev',
                'patience_elapsed', 'python_rng', 'torch_rng', 'cuda_rng'}
    if not required.issubset(checkpoint):
        raise ValueError('Resume requires complete optimizer, history and RNG state.')
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(checkpoint['adapter']):
        raise ValueError('Resume adapter parameter names differ.')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(checkpoint['adapter'][name].to(parameter.device))
    optimizer.load_state_dict(checkpoint['optimizer'])
    random.setstate(checkpoint['python_rng'])
    torch.set_rng_state(checkpoint['torch_rng'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])


def stopping_ready(bad, patience, epoch, minimum_epochs, lr, minimum_lr, plateau_lr):
    """Require full-epoch stagnation and the declared learning-rate floor."""
    return (bad >= patience and epoch >= minimum_epochs
            and (not plateau_lr or lr <= minimum_lr * (1 + 1e-6)))


def save_checkpoint(state, path):
    """Publish a checkpoint only after its complete serialization succeeds."""
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def validate_data(path):
    data = json.loads(Path(path).read_text())
    works, hashes = {}, {}
    for split in ('train', 'dev', 'test'):
        rows = data[split]
        if not rows:
            raise ValueError('Every work-disjoint split must contain observations.')
        works[split] = {r['work_id'] for r in rows}
        hashes[split] = set()
        for row in rows:
            if not row.get('label_provenance') or not row.get('supported_preference'):
                raise ValueError('Each pair needs explicit preference provenance and support.')
            probability = row.get('preference_probability',1.0)
            if not 0 <= probability <= 1 or not 0 < row.get('weight',1.0) <= 1:
                raise ValueError('Invalid preference target or observation weight.')
            for side in ('chosen', 'rejected'):
                if not isinstance(row[side], str) or not row[side].strip():
                    raise ValueError('Empty screenplay.')
                hashes[split].add(hashlib.sha256(row[side].strip().encode()).hexdigest())
            if row['chosen'] == row['rejected']:
                raise ValueError('Identical pair cannot be preference training.')
    for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        if works[a] & works[b] or hashes[a] & hashes[b]:
            raise ValueError('Work or text leakage across splits.')
    protected = {'pride', 'jane', 'alice', 'carol'}
    if works['train'] & protected:
        raise ValueError('Existing diagnostic works are excluded from training.')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--model-preflight', action='store_true')
    parser.add_argument('--memory-floor-mib', type=int, default=32000)
    parser.add_argument('--max-epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--macro-dev-by-dataset', action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--plateau-lr', action='store_true')
    parser.add_argument('--lr-patience', type=int, default=3)
    parser.add_argument('--min-lr', type=float, default=1e-7)
    parser.add_argument('--minimum-epochs', type=int, default=0)
    parser.add_argument('--model-path')
    parser.add_argument('--model-revision')
    parser.add_argument('--train-score-head', action='store_true')
    args = parser.parse_args()
    if args.patience < 8:
        raise ValueError('Use adequate full-epoch patience.')
    data = validate_data(args.data)
    if args.preflight:
        print({s: len(data[s]) for s in data})
        return
    if any(row.get('requires_candidate_history') for rows in data.values() for row in rows):
        raise RuntimeError('This trainer scores final text only; implement and validate candidate-history inputs before training this stage.')
    import subprocess
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from local_litbench_review import MODEL, REVISION
    model_path = args.model_path or MODEL
    revision = args.model_revision or REVISION
    if args.model_path and not args.model_revision:
        raise ValueError('An explicit local model requires recorded revision provenance.')
    if int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0]) < args.memory_floor_mib:
        raise RuntimeError('Training memory preflight failed; preserve other workloads.')
    random.seed(42)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_kwargs = {'num_labels': 1} if args.train_score_head else {}
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa', **model_kwargs).to('cuda:0')
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names = []
    for name, layer in list(model.named_modules()):
        if name.endswith(('q_proj', 'v_proj')) and isinstance(layer, nn.Linear):
            parent, leaf = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), leaf, LowRankLinear(layer))
            names.append(name)
    if not names:
        raise RuntimeError('No compatible attention projections.')
    if args.train_score_head:
        model.score = Float32Score(model.score)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.model_preflight and (output / 'best.pt').exists():
        raise RuntimeError('Use a new output directory; preserve existing checkpoints.')
    def tokens(text):
        batch = tokenizer(text, return_tensors='pt', truncation=False).to('cuda:0')
        if batch['input_ids'].shape[1] > model.config.max_position_embeddings:
            raise ValueError('Full screenplay exceeds context; no truncation.')
        return batch
    def pair_loss(row):
        chosen = model(**tokens(row['chosen'])).logits.float().squeeze()
        rejected = model(**tokens(row['rejected'])).logits.float().squeeze()
        target = torch.tensor(row.get('preference_probability', 1.0), device=chosen.device)
        return F.binary_cross_entropy_with_logits(chosen-rejected, target)
    def dev_loss():
        model.eval()
        with torch.no_grad():
            totals = {}
            for row in data['dev']:
                group = row.get('dataset', 'default') if args.macro_dev_by_dataset else 'default'
                totals.setdefault(group, [0.0, 0.0])
                weight = row.get('weight', 1.0)
                totals[group][0] += pair_loss(row).item() * weight
                totals[group][1] += weight
            return sum(value / weight for value, weight in totals.values()) / len(totals)
    if args.model_preflight:
        model.train()
        preflight_row = max(data['train'], key=lambda r: max(len(r['chosen']),len(r['rejected'])))
        loss = pair_loss(preflight_row)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite model preflight loss.')
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Non-finite or absent adapter gradients.')
        record = {'loss':loss.item(), 'peak_cuda_bytes':torch.cuda.max_memory_allocated(), 'finite_gradients':True, 'test_accessed':False, 'selection':'longest TRAIN pair by characters', 'pair_id':preflight_row.get('pair_id')}
        (output/'model_preflight.json').write_text(json.dumps(record,indent=2))
        print(record,flush=True)
        return
    data_sha = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    start_epoch = 0
    resume_metadata = None
    if args.resume:
        source = Path(args.resume).resolve()
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        source_protocol = json.loads((source.parent / 'protocol.json').read_text())
        restore_training_state(model, optimizer, checkpoint, data_sha, source_protocol)
        start_epoch = checkpoint['epoch']
        best = checkpoint['best_dev']
        history = list(checkpoint['history'])
        bad = checkpoint['patience_elapsed']
        resume_metadata = {'checkpoint': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                           'source_epoch': start_epoch, 'optimizer_and_rng_restored': True,
                           'stage_type': 'Continuation with explicitly new scheduling and stopping rules'}
    else:
        best = dev_loss()
        bad, history = 0, [{'epoch': 0, 'dev_loss': best}]
    scheduler = None
    if args.plateau_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=.5, patience=args.lr_patience,
            threshold=0.0, threshold_mode='abs', min_lr=args.min_lr, eps=0.0)
        scheduler.step(best)
    (output/'protocol.json').write_text(json.dumps({'revision': revision, 'model_path': str(model_path),
        'train_score_head': args.train_score_head, 'score_head_initialization_std': .001 if args.train_score_head else None,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'input': 'raw full text; no model/source labels', 'adapter_rank': 8,
        'adapter_scale': 2, 'projections': names, 'seed': 42, 'lr': args.lr, 'max_epochs': args.max_epochs,
        'patience_full_epochs': args.patience, 'macro_dev_by_dataset': args.macro_dev_by_dataset, 'test_accessed': False, 'objective':'weighted soft-target Bradley-Terry; unsure votes excluded', 'memory_floor_mib':args.memory_floor_mib, 'data_sha256': data_sha,
        'resume': resume_metadata, 'plateau_lr': args.plateau_lr, 'lr_patience_full_epochs': args.lr_patience,
        'minimum_lr': args.min_lr, 'minimum_full_epochs': args.minimum_epochs,
        'stopping': 'Full-epoch DEV stagnation; require LR floor when plateau scheduling is enabled'}, indent=2))
    save_checkpoint({'adapter':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
        'optimizer':optimizer.state_dict(),'epoch':start_epoch,'history':history,'best_dev':best,'patience_elapsed':bad,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},output/'best.pt')
    stopped = False
    for epoch in range(start_epoch+1, args.max_epochs+1):
        model.train()
        order = list(range(len(data['train'])))
        random.shuffle(order)
        total = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            row = data['train'][index]
            loss = pair_loss(row) * row.get('weight',1.0)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training objective.')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad and p.grad is not None):
                raise RuntimeError('Non-finite adapter gradients.')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total += loss.item()
        value = dev_loss()
        history.append({'epoch': epoch, 'train_pairs_visited': len(order), 'unique_pairs_visited': len(set(order)),
                        'train_loss': total/len(order), 'dev_loss': value,
                        'learning_rate': optimizer.param_groups[0]['lr']})
        if scheduler:
            scheduler.step(value)
        state = {'adapter': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'history': history,
                 'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        improved = value < best
        best, bad = (value, 0) if improved else (best, bad+1)
        state.update(best_dev=best, patience_elapsed=bad,
                     scheduler=scheduler.state_dict() if scheduler else None)
        save_checkpoint(state, output/'latest.pt')
        if improved:
            save_checkpoint(state, output/'best.pt')
        (output/'history.json').write_text(json.dumps(history, indent=2))
        print('Completed epoch', epoch, 'DEV loss', value, flush=True)
        if stopping_ready(bad, args.patience, epoch, args.minimum_epochs,
                          optimizer.param_groups[0]['lr'], args.min_lr, args.plateau_lr):
            stopped = True
            break
    (output/'completion.json').write_text(json.dumps({'completed_epochs': epoch,
        'stop_reason': 'dev_plateau_at_lr_floor' if stopped and scheduler else 'dev_patience' if stopped else 'epoch_budget',
        'new_stage_completed_full_epochs': epoch - start_epoch,
        'best_dev_loss': best, 'final_lr': optimizer.param_groups[0]['lr'],
        'better_than_pretrained_dev': best < history[0]['dev_loss'], 'test_accessed': False,
        'convergence_scope': 'DEV generalization plateau under declared LR schedule; not proof of a global optimum'}, indent=2))


if __name__ == '__main__':
    main()
