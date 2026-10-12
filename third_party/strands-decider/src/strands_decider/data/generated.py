"""Generated document questions (v16): realistic workplace documents, each with 3-4 hard
typed questions, written by Qwen3.6-27B and kept only when two answers by
Qwen3.5-397B-A17B agreed with the writer's (data/generators/gen_documents_openrouter.py).

The generator exports strands-decider Examples already; this module turns its output into the
training and evaluation files:

  train  documents outside the held-out domains, with yes/no questions balanced to equal
         "yes" and "no" within each skill. The generator asks for a 50/50 split, but the
         verifier filter keeps some answers more often than others, and an unbalanced
         skill would let the skill alone predict the answer. Choice questions are
         thinned until the answer is the longest option no more often than chance
         (`balance_length`); their option order is shuffled at collate time.
  eval   documents in the held-out domains (insurance claims, municipal permits,
         laboratory safety, payroll), at their natural label distribution, options in a
         fixed shuffled order (`shuffle_options`).

Rows over the window are dropped, never truncated (`multistep.fits`).

v18 combines further exports into a file of its own:
    python -m strands_decider.data.generated --src data/generators/gen_pilot_qwen data/generators/gen_mixed_pilot data/generators/gen_weak \
        --out data/generated_v18.jsonl --eval-out data/generated_v18_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

from .format import Example
from .multistep import fits
from .policy import balance


def load(path: str) -> list[Example]:
    with open(path, encoding="utf-8") as fh:
        return [Example.from_dict(json.loads(line)) for line in fh if line.strip()]


def balance_noul(examples: list[Example], seed: int = 0) -> list[Example]:
    """Equal "yes" and "no" within each skill for yes/no rows; choice rows untouched."""
    noul = [e for e in examples if e.kind == "noul"]
    rest = [e for e in examples if e.kind != "noul"]
    return rest + balance(noul, key=lambda e: e.task, min_count=1, seed=seed)


def _longest(ex: Example) -> int:
    return max(range(len(ex.options)), key=lambda i: len(ex.options[i][0]) + len(ex.options[i][1]))


def balance_length(examples: list[Example], seed: int = 0) -> list[Example]:
    """Choice rows thinned until the answer is the longest option no more often than chance.

    The writer gives the right answer the fullest description: in the v16 export the
    correct option (name + description) is the longest in 34% of choice rows against a
    23% chance rate, a cue the pointer readout can see. Rows where it is are dropped at
    random until that rate equals the mean of 1/n_options over the choice rows that
    remain; the others and all yes/no rows are kept.
    """
    rng = random.Random(seed)
    choice = [e for e in examples if e.kind == "choice"]
    rest = [e for e in examples if e.kind != "choice"]
    cued = [e for e in choice if _longest(e) == e.label]
    clean = [e for e in choice if _longest(e) != e.label]
    rng.shuffle(cued)
    while cued:
        kept = cued + clean
        if len(cued) / len(kept) <= sum(1 / len(e.options) for e in kept) / len(kept):
            break
        cued.pop()
    return rest + cued + clean


def shuffle_options(examples: list[Example], seed: int = 0) -> list[Example]:
    """A fixed random option order per row, the label moved with it.

    Training shuffles options at collate time; evaluation reads them as stored, and the
    writer put the answer second in 42% of choice questions. Shuffling once, with a
    seed, keeps the eval file fixed without that artefact.
    """
    rng = random.Random(seed)
    out = []
    for e in examples:
        order = list(range(len(e.options)))
        if e.kind == "choice":  # noul and score keep their canonical order
            rng.shuffle(order)
        out.append(Example(kind=e.kind, state=e.state, instructions=e.instructions,
                           options=[e.options[i] for i in order], label=order.index(e.label),
                           task=e.task, weight=e.weight, instruction_variants=e.instruction_variants))
    return out


def load_paraphrases(path: str) -> dict:
    """{question text: [checked paraphrases]} from data/generators/gen_paraphrases_openrouter.py."""
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {r["question"].strip(): r["paraphrases"] for r in rows if r.get("paraphrases")}


def attach_paraphrases(examples: list[Example], table: dict) -> list[Example]:
    """Each question's checked paraphrases as instruction variants, sampled per epoch at
    collate time (v20). The original wording stays one of the variants; rows keep their
    order, so a build with and without paraphrases differs only in the variants."""
    out = []
    for e in examples:
        ps = table.get(e.instructions.strip(), [])
        variants = [e.instructions] + [p for p in ps if p != e.instructions] if ps else list(e.instruction_variants)
        out.append(Example(kind=e.kind, state=e.state, instructions=e.instructions, options=e.options,
                           label=e.label, task=e.task, weight=e.weight, instruction_variants=variants))
    return out


def paraphrase_pairs(examples: list[Example], table: dict) -> list[Example]:
    """For the consistency eval: each row that has a paraphrase, followed by the same row
    asked with its first paraphrase. Same state, same options in the same order."""
    out = []
    for e in examples:
        ps = table.get(e.instructions.strip())
        if ps:
            out += [e, Example(kind=e.kind, state=e.state, instructions=ps[0], options=e.options,
                               label=e.label, task=e.task, weight=e.weight)]
    return out


def attach_main(args: argparse.Namespace) -> None:
    """`--attach`: paraphrases onto an already built file, rows and order untouched, and
    with `--attach-eval`, the paired consistency eval from an already built eval file.
    This is how v20's files were made from the exact files v16-v19 used."""
    from .format import read_jsonl, write_jsonl

    table = load_paraphrases(args.paraphrases)
    train = attach_paraphrases(list(read_jsonl(args.attach)), table)
    write_jsonl(args.out, train)
    print(f"{args.out}: paraphrases attached to {sum(len(e.instruction_variants) > 1 for e in train):,} "
          f"of {len(train):,} rows from {args.attach}")
    if args.attach_eval and args.pairs_out:
        pairs = paraphrase_pairs(list(read_jsonl(args.attach_eval)), table)
        write_jsonl(args.pairs_out, pairs)
        print(f"{args.pairs_out}: {len(pairs) // 2:,} eval rows from {args.attach_eval}, each with one paraphrase")


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .format import write_jsonl

    ap = argparse.ArgumentParser(description="Build the generated-document training and eval sets (v16).")
    ap.add_argument("--src", nargs="+", default=["data/generators/gen_v16"],
                    help="the generator's output directories, combined")
    ap.add_argument("--out", default="data/generated_v16.jsonl")
    ap.add_argument("--eval-out", default="data/generated_v16_eval.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paraphrases", default="", help="checked paraphrases to attach as instruction variants")
    ap.add_argument("--pairs-out", default="", help="with --paraphrases: the paired consistency eval")
    ap.add_argument("--attach", default="", help="an already built training file to attach --paraphrases to "
                                                 "(no rebuild; written to --out)")
    ap.add_argument("--attach-eval", default="", help="with --attach: an already built eval file for --pairs-out")
    args = ap.parse_args(argv)
    if args.attach:
        if not args.paraphrases:
            ap.error("--attach needs --paraphrases")
        return attach_main(args)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    raw = [e for d in args.src for e in load(f"{d}/gen_train.jsonl")]
    train = fits(balance_length(balance_noul(raw, seed=args.seed), seed=args.seed), tok, args.max_tokens)
    evals = fits(shuffle_options([e for d in args.src for e in load(f"{d}/gen_eval.jsonl")],
                                 seed=args.seed), tok, args.max_tokens)
    random.Random(args.seed).shuffle(train)
    if args.paraphrases:
        table = load_paraphrases(args.paraphrases)
        train = attach_paraphrases(train, table)
        print(f"paraphrases attached to {sum(len(e.instruction_variants) > 1 for e in train):,} of "
              f"{len(train):,} training rows")
        if args.pairs_out:
            pairs = paraphrase_pairs(evals, table)
            print(f"consistency eval: {len(pairs) // 2:,} eval rows, each with one paraphrase")
            write_jsonl(args.pairs_out, pairs)
    print(f"train {len(train):,} rows (from {len(raw):,} kept; yes/no balanced per skill, "
          f"answer-is-longest thinned to chance), eval {len(evals):,} rows (options shuffled)")
    write_jsonl(args.out, train)
    write_jsonl(args.eval_out, evals)


if __name__ == "__main__":
    main()
