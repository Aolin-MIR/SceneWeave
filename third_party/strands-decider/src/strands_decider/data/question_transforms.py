"""Question-varied rows: the same real state, asked a question whose answer differs.

v7 learned that the question never changes the answer, because in its corpus it never
did: each state is asked one fixed question per task, so the state and the option set
determine the label on their own. evaluation/question_sensitivity.py measured the result --
change the question and v7 gives the same answer 94-99% of the time, where the frozen
base follows the new question.

These transforms keep the states (real, varied, the part v9 and v10 lacked) and change
only the question, so the question is the one thing that explains the label:

  is_answer   choice -> noul   'Is "X" the correct answer?' / 'Is "X" the wrong answer?'
                               X is the gold option or another, polarity is random, so
                               the same state and X get opposite labels.
  complement  choice -> choice two options, gold and one other; asked for the right one
                               or the wrong one.
  threshold   score  -> noul   'Is the rating k or higher?' / 'Is it lower than k?'
  polarity    noul   -> noul   the row's own question inside a wrapper that keeps or
                               flips it ("...the answer is positive" / "...is negative").

Every label is derived from the source row's gold label, never estimated.
tests/test_question_transforms.py re-derives each one from the rendered text.

The probe's question forms are deliberately absent -- "listed first/last", "which option
does this text NOT fit" over all options, 'Is the answer to the following question
"no"?' -- so the probe measures whether the model learned to read questions, not
whether it learned these templates. PROBE_FORMS lists them and the tests enforce it.

Transformed noul rows use the generic true/false descriptions. A task's own ("true --
promotional, scam, or bulk message") would contradict a flipped question.

Rows keep no instruction variants: the collator samples a phrasing from them, and the
source row's variants phrase the *original* question, whose answer is different.
"""
from __future__ import annotations

import argparse
import json
import random

from ..prompting import NOUL_DEFAULT_CRITERIA, NOUL_SLOT_LABELS
from .format import Example, read_jsonl, write_jsonl

GENERIC_NOUL = [[lbl, NOUL_DEFAULT_CRITERIA[lbl]] for lbl in NOUL_SLOT_LABELS]

# (template, polarity). Positive: true when X is the gold option; negative: when it is not.
IS_ANSWER = [
    ('Question: "{q}" Is "{x}" the correct answer?', True),
    ('Someone asked: "{q}" Would "{x}" be a right answer?', True),
    ('For the question "{q}", does "{x}" apply?', True),
    ('"{q}" Is the answer {x}?', True),
    ('Question: "{q}" Is "{x}" the wrong answer?', False),
    ('Someone asked: "{q}" Would "{x}" be a mistaken answer?', False),
    ('For the question "{q}", is "{x}" incorrect?', False),
    ('"{q}" Would answering {x} be an error?', False),
]
# Positive: pick the option that answers the question; negative: the one that does not.
COMPLEMENT = [
    ('Question: "{q}" Which of these two is the correct answer?', True),
    ('For "{q}", pick the option that applies.', True),
    ('Of these two options, which one answers "{q}"?', True),
    ('Question: "{q}" Which of these two is NOT a correct answer?', False),
    ('One of these options is wrong for the question "{q}". Which one?', False),
    ('For "{q}", pick the option that does not apply.', False),
]
# Positive: "k or higher"; negative: "lower than k".
THRESHOLD = [
    ('{q} Scale: {scale}. Is the rating {k} or higher?', True),
    ('{q} Using the scale {scale}: does it reach at least {k}?', True),
    ('{q} Scale: {scale}. Is the rating lower than {k}?', False),
    ('{q} Using the scale {scale}: does it fall short of {k}?', False),
]
# Positive keeps the row's answer; negative flips it.
POLARITY = [
    ('True or false: the honest answer to "{q}" is positive.', True),
    ('Confirm or deny: the answer to "{q}" is yes.', True),
    ('Is it right to answer yes here? {q}', True),
    ('True or false: the honest answer to "{q}" is negative.', False),
    ('Confirm or deny: the answer to "{q}" is no.', False),
    ('Is it wrong to answer yes here? {q}', False),
]

# The probe's forms (evaluation/question_sensitivity.py); never generated here.
PROBE_FORMS = [
    "listed first",
    "listed last",
    "clearly NOT fit",
    'Is the answer to the following question "no"?',
]

CHOICE_SPLIT = 0.6  # share of transformed choice rows that become is_answer, not complement


def _pick(rng: random.Random, templates: list[tuple[str, bool]], polarity: bool) -> str:
    return rng.choice([t for t, p in templates if p == polarity])


def _clean(q: str) -> str:
    # Quotes inside a quoted question would make the rendered text ambiguous to parse.
    return " ".join(q.replace('"', "'").split())


def is_answer(ex: Example, rng: random.Random) -> tuple[Example, dict]:
    gold = ex.label
    use_gold = rng.random() < 0.5
    k = gold if use_gold else rng.choice([i for i in range(ex.n_options) if i != gold])
    positive = rng.random() < 0.5
    x = ex.options[k][0]
    q = _pick(rng, IS_ANSWER, positive).format(q=_clean(ex.instructions), x=_clean(x))
    label = int(use_gold == positive)
    out = Example(kind="noul", state=ex.state, instructions=q, options=GENERIC_NOUL,
                  label=label, task=f"{ex.task}+is_answer", weight=ex.weight)
    return out, {"transform": "is_answer", "x": x, "positive": positive}


def complement(ex: Example, rng: random.Random) -> tuple[Example, dict]:
    gold = ex.label
    other = rng.choice([i for i in range(ex.n_options) if i != gold])
    pair = [ex.options[gold], ex.options[other]]
    rng.shuffle(pair)
    positive = rng.random() < 0.5
    q = _pick(rng, COMPLEMENT, positive).format(q=_clean(ex.instructions))
    want = ex.options[gold] if positive else ex.options[other]
    out = Example(kind="choice", state=ex.state, instructions=q, options=pair,
                  label=pair.index(want), task=f"{ex.task}+complement", weight=ex.weight)
    return out, {"transform": "complement", "gold_name": ex.options[gold][0],
                 "positive": positive}


def threshold(ex: Example, rng: random.Random) -> tuple[Example, dict]:
    gold, n = ex.label, ex.n_options
    scale = "; ".join(f"{i} = {_clean(desc)}" for i, (_, desc) in enumerate(ex.options))
    # Choose the label first, then a (polarity, k) that produces it, so labels stay
    # balanced whatever the gold distribution; fall back if none can.
    want = rng.random() < 0.5
    cands = [(p, k) for p in (True, False) for k in range(1, n)
             if ((gold >= k) if p else (gold < k)) == want]
    if not cands:
        cands = [(p, k) for p in (True, False) for k in range(1, n)]
    positive, k = rng.choice(cands)
    label = int((gold >= k) if positive else (gold < k))
    q = _pick(rng, THRESHOLD, positive).format(q=_clean(ex.instructions), scale=scale, k=k)
    out = Example(kind="noul", state=ex.state, instructions=q, options=GENERIC_NOUL,
                  label=label, task=f"{ex.task}+threshold", weight=ex.weight)
    return out, {"transform": "threshold", "k": k, "positive": positive}


def polarity(ex: Example, rng: random.Random) -> tuple[Example, dict]:
    positive = rng.random() < 0.5
    q = _pick(rng, POLARITY, positive).format(q=_clean(ex.instructions))
    label = ex.label if positive else 1 - ex.label
    out = Example(kind="noul", state=ex.state, instructions=q, options=GENERIC_NOUL,
                  label=label, task=f"{ex.task}+polarity", weight=ex.weight)
    return out, {"transform": "polarity", "positive": positive}


def transform(ex: Example, rng: random.Random) -> tuple[Example, dict]:
    """One question-varied row from one source row."""
    if ex.kind == "choice":
        return (is_answer if rng.random() < CHOICE_SPLIT else complement)(ex, rng)
    if ex.kind == "score":
        return threshold(ex, rng)
    return polarity(ex, rng)


def build(
    examples: list[Example], fraction: float, seed: int = 0
) -> tuple[list[Example], list[tuple[int, Example, dict]]]:
    """Pick `fraction` of rows (the same rows for a given seed) and transform each.

    Returns (untouched rows, [(source index, transformed row, info)]). Arm A trains on
    untouched + transformed; arm B keeps every source row and adds the transformed ones
    as KL-only prompts -- so both arms see exactly the same changed-question prompts.
    """
    rng = random.Random(seed)
    chosen = set(rng.sample(range(len(examples)), round(fraction * len(examples))))
    kept, made = [], []
    for i, ex in enumerate(examples):
        if i in chosen:
            new, info = transform(ex, rng)
            made.append((i, new, info))
        else:
            kept.append(ex)
    return kept, made


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build question-varied corpora (v11 arms).")
    ap.add_argument("--src", default="data/train_v5.jsonl")
    ap.add_argument("--fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replace-out", default="data/train_v11a.jsonl",
                    help="arm A: source rows with the chosen fraction replaced")
    ap.add_argument("--kl-out", default="data/kl_v11b.jsonl",
                    help="arm B: the transformed rows alone, for kl_only_files")
    args = ap.parse_args(argv)

    examples = list(read_jsonl(args.src))
    kept, made = build(examples, args.fraction, args.seed)
    rows = kept + [m[1] for m in made]
    random.Random(args.seed + 1).shuffle(rows)
    n_a = write_jsonl(args.replace_out, rows)
    n_b = write_jsonl(args.kl_out, [m[1] for m in made])
    counts: dict[str, list[int]] = {}
    for _, ex, info in made:
        counts.setdefault(info["transform"], []).append(ex.label if ex.kind == "noul" else -1)
    print(f"{args.replace_out}: {n_a:,} rows ({len(kept):,} kept, {len(made):,} transformed)")
    print(f"{args.kl_out}: {n_b:,} rows")
    for name, labels in sorted(counts.items()):
        noul = [x for x in labels if x >= 0]
        share = f", share true {sum(noul) / len(noul):.3f}" if noul else ""
        print(f"  {name:<11} {len(labels):>6,}{share}")
    print(json.dumps({"src": args.src, "fraction": args.fraction, "seed": args.seed}))


if __name__ == "__main__":
    main()
