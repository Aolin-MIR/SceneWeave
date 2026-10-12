"""A trained checkpoint's own answer distributions, as a teacher file (replay toward a parent).

A later model trained on new rows can lose what an earlier one did on old ones even
though the old rows stay in its corpus: v15 and v16 both kept v14's multi-step rows and
both regressed on MuSiQue (and v16 on HotpotQA). decider-2b protects its parent by
training replayed rows toward the parent's own distribution, KL(p_parent || p_model),
instead of their labels (its 4B v2.1 ablation: replay on hard labels sharpened every
answer). The trainer's teacher term is exactly that loss, so replay needs only the
parent's distributions in the teacher-file format `train.py` reads:

    {"i": <row index in the concatenated train_files>, "probs": [...]}

Probabilities are the softmax of the checkpoint's raw logits (temperature 1, no
calibration), over each row's options in canonical order; the collator permutes them
with the options at training time.

    python -m strands_decider.data.replay CHECKPOINT --src data/multistep_v14.jsonl \\
        --out data/replay_v14_multistep.jsonl --shift-by data/train_v5.jsonl

On N GPUs: one process per GPU with `--num-shards N --shard-index i` (each with the same
`--shift-by`), then the same command with `--merge` in place of `--shard-index` to write
`--out` (see data/shards.py).
"""
from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    from ..evaluate import collect_logits
    from ..modeling import StrandsDeciderModel, masked_log_softmax
    from . import shards
    from .format import load_examples

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--src", required=True, help="the rows to label, one train_files entry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shift-by", nargs="*", default=[],
                    help="train_files that come before --src; indices are offset past their rows")
    ap.add_argument("--batch-size", type=int, default=8)
    shards.add_args(ap)
    args = ap.parse_args(argv)
    sharded = shards.sharded(ap, args)

    offset = 0
    for path in args.shift_by:
        with open(path, encoding="utf-8") as fh:
            offset += sum(1 for line in fh if line.strip())
    rows = load_examples([args.src])
    # The rows this process writes: all of them, or for a shard the single-process run's
    # --batch-size batches i, i + N, ..., so that each row is computed in the same batch.
    keep = [j for j in range(len(rows))
            if not sharded or j // args.batch_size % args.num_shards == args.shard_index]
    out = shards.path(args.out, args.shard_index, args.num_shards) if sharded else args.out
    if args.merge:
        got = shards.merge(args.out, args.num_shards)
        if set(got) != set(range(offset, offset + len(rows))):
            raise SystemExit(f"shards cover {len(got):,} rows, not the {len(rows):,} of "
                             f"{args.src} from index {offset:,} -- wrong --src or --shift-by?")
        found = [got[offset + j] for j in keep]
    else:
        model = StrandsDeciderModel.load(args.checkpoint)
        logits, _, n_slots, _ = collect_logits(model, [rows[j] for j in keep],
                                              max_length=model.config.max_length,
                                              batch_size=args.batch_size)
        probs = masked_log_softmax(logits, n_slots).exp()
        found = [[round(float(x), 6) for x in p[:n]]
                 for p, n in zip(probs, n_slots.tolist(), strict=True)]
    agree = 0
    with open(out, "w", encoding="utf-8") as fh:
        for j, row in zip(keep, found, strict=True):
            ex, n = rows[j], len(row)
            assert n == ex.n_options, f"row {j}: {n} logits for {ex.n_options} options"
            agree += int(max(range(n), key=row.__getitem__) == ex.label)
            fh.write(json.dumps({"i": offset + j, "probs": row}) + "\n")
    if sharded:
        shards.mark_done(out)
    print(f"{out}: {len(keep):,} rows from {args.checkpoint}, indices from {offset:,}; "
          f"its answer equals the label on {agree / max(len(keep), 1):.3f}")


if __name__ == "__main__":
    main()
