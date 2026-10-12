"""Catch-all options (v20): choice rows where "other" / "none of these" is an option, and
half the time the right one.

The classification corpus names every category it asks about, so the model has never
had to pick a catch-all because the true category is missing, or to pass one over
because it is present. JevBench's routing and intent questions carry such options, and
v19 misses one of each with high confidence. This adds rows built from the corpus's own
choice tasks with named categories (ag_news, banking77, clinc150, dbpedia,
yahoo_topics):

  absent   the gold option is removed and a catch-all is added: the catch-all is right
  present  a catch-all is added beside the gold option: the gold option is right

in equal numbers, so the presence of a catch-all never predicts the answer. Rows whose
options already include CLINC150's out-of-scope class are skipped. The catch-all goes
at the end of the canonical order; options are shuffled at collate time.

The eval applies the same transform to the test half of two held-out tasks never
trained on (`massive_intent`, `emotion` in data/holdout_v5_norule.jsonl).

    python -m strands_decider.data.catchall --out data/catchall_v20.jsonl --eval-out data/catchall_v20_eval.jsonl
"""
from __future__ import annotations

import random
from collections.abc import Iterable

from .format import Example

SOURCES = ("ag_news", "banking77", "clinc150", "dbpedia", "yahoo_topics")
HELDOUT = ("massive_intent", "emotion")
NAMES = ["other", "none of these", "something else", "none of the listed categories"]
DESCRIPTION = "the text fits none of the other options"


def _oos(ex: Example) -> bool:
    return any("oos" in n.lower() or "out of scope" in n.lower() or "out_of_scope" in n.lower()
               for n, _ in ex.options)


def transform(ex: Example, mode: str, rng: random.Random, max_options: int = 24) -> Example | None:
    """One catch-all row from `ex`; None if the row cannot take one. A row already at
    `max_options` (the model's num_slots, the width of its training KL reference) gives
    up one wrong option at random to make room for the catch-all."""
    if ex.kind != "choice" or _oos(ex) or any(n.lower() in NAMES for n, _ in ex.options):
        return None
    catch = [rng.choice(NAMES), DESCRIPTION]
    if mode == "absent":
        rest = [o for i, o in enumerate(ex.options) if i != ex.label]
        if len(rest) < 2:
            return None
        options, label = [*rest, catch], len(rest)
    elif mode == "present":
        options, label = list(ex.options), ex.label
        if len(options) >= max_options:
            drop = rng.choice([i for i in range(len(options)) if i != label])
            options = [o for i, o in enumerate(options) if i != drop]
            label = label - (drop < label)
        options = [*options, catch]
    else:
        raise ValueError(mode)
    return Example(kind="choice", state=ex.state, instructions=ex.instructions,
                   options=[list(o) for o in options], label=label, task=f"{ex.task}+catchall:{mode}",
                   weight=ex.weight, instruction_variants=list(ex.instruction_variants))


def build(examples: list[Example], tasks: Iterable[str], per_task: int, seed: int = 0) -> list[Example]:
    """`per_task` rows from each task, half absent and half present."""
    rng = random.Random(seed)
    out: list[Example] = []
    for task in tasks:
        pool = [e for e in examples if e.task == task and e.kind == "choice"]
        rng.shuffle(pool)
        want = {"absent": per_task // 2, "present": per_task - per_task // 2}
        for e in pool:
            mode = "absent" if want["absent"] >= want["present"] and want["absent"] > 0 else "present"
            if want[mode] == 0:
                break
            row = transform(e, mode, rng)
            if row is not None:
                out.append(row)
                want[mode] -= 1
    rng.shuffle(out)
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    from ..evaluate import partition_examples
    from .format import read_jsonl, write_jsonl

    ap = argparse.ArgumentParser(description="Build the catch-all option rows (v20).")
    ap.add_argument("--src", default="data/train_v5.jsonl")
    ap.add_argument("--heldout", default="data/holdout_v5_norule.jsonl")
    ap.add_argument("--out", default="data/catchall_v20.jsonl")
    ap.add_argument("--eval-out", default="data/catchall_v20_eval.jsonl")
    ap.add_argument("--per-task", type=int, default=1000)
    ap.add_argument("--eval-per-task", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    train = build(list(read_jsonl(args.src)), SOURCES, args.per_task, seed=args.seed)
    test_half = partition_examples(list(read_jsonl(args.heldout)), "test")
    evals = build(test_half, HELDOUT, args.eval_per_task, seed=args.seed)
    for name, rows in (("train", train), ("eval", evals)):
        absent = sum(r.task.endswith(":absent") for r in rows)
        print(f"{name}: {len(rows):,} rows ({absent:,} catch-all right, {len(rows) - absent:,} catch-all a distractor)")
    write_jsonl(args.out, train)
    write_jsonl(args.eval_out, evals)


if __name__ == "__main__":
    main()
