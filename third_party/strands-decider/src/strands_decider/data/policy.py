"""Rules applied to a user's situation, from public datasets, as typed questions (v15).

decider-2b v11 leads v14 by 14 JevBench tasks, and its stage-2 data -- questions over
policies and business documents -- is not released. These are the nearest public sets:

  sharc          short rules from government websites (ShARC, CC BY-SA 3.0), a user's
                 question and situation, and the follow-up questions answered so far:
                 yes, no, or the answer depends on a fact the user has not stated.
  conditionalqa  long gov.uk guidance pages (ConditionalQA): yes, no, yes-or-no only
                 under a condition the scenario does not establish, or not answerable
                 from the page.

Every label comes from the dataset. Both have shortcuts that reading the text would not
need, removed at assembly (see `balance`): ShARC's labels follow whether a scenario and
a dialogue are present and what the last follow-up answer was, and ConditionalQA is 68%
"yes".
"""
from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Iterable, Iterator

from .format import Example
from .multistep import CHAR_BUDGET, _window, fits

YES, NO, DEPENDS, NOT_SAID = 0, 1, 2, 3
POLICY_OPTIONS = [
    ["yes", "the text and the facts given establish that the answer is yes"],
    ["no", "the text and the facts given establish that the answer is no"],
    ["depends", "the answer turns on a fact or condition the user has not established"],
    ["not stated", "the text does not answer this question"],
]
POLICY_Q = ("Based only on the text and what the user has said, what is the answer to "
            'the user\'s question: "{q}"')


# ---- ShARC ------------------------------------------------------------------

def sharc_label(answer: str) -> int | None:
    """Yes / No / a follow-up question (the answer depends on something unsaid).

    "Irrelevant" rows are dropped: they are questions paired with an unrelated rule, all
    with no scenario and no dialogue, so that emptiness alone identifies them.
    """
    if answer == "Irrelevant":
        return None
    return {"Yes": YES, "No": NO}.get(answer, DEPENDS)


def sharc_cell(r: dict) -> tuple:
    """What the answer follows without reading: scenario present, dialogue length, and the
    last follow-up answer (the final answer often echoes it: 0.53 on dev from that alone)."""
    last = r["history"][-1]["follow_up_answer"].strip().lower() if r["history"] else "-"
    return bool(r["scenario"].strip()), min(len(r["history"]), 3), last


def sharc(rows: Iterable[dict]) -> Iterator[Example]:
    for r in rows:
        label = sharc_label(r["answer"])
        if label is None:
            continue
        parts = [f"Rule:\n{r['snippet'].strip()}"]
        if r["scenario"].strip():
            parts.append(f"User's situation: {r['scenario'].strip()}")
        if r["history"]:
            parts.append("Follow-up so far:\n" + "\n".join(
                f"Q: {h['follow_up_question']}\nA: {h['follow_up_answer']}" for h in r["history"]))
        yield Example(kind="choice", state="\n\n".join(parts),
                      instructions=POLICY_Q.format(q=r["question"].strip()),
                      options=POLICY_OPTIONS[:3], label=label, task="sharc")


# ---- ConditionalQA ----------------------------------------------------------

def _plain(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).strip()


def conditionalqa_label(q: dict) -> int | None:
    """Span answers are skipped; a yes/no that holds only under conditions is DEPENDS."""
    if q["not_answerable"]:
        return NOT_SAID
    answers = {a.lower() for a, _ in q["answers"]}
    if not answers or not answers <= {"yes", "no"}:
        return None
    if any(conds for _, conds in q["answers"]):
        return DEPENDS
    return {frozenset({"yes"}): YES, frozenset({"no"}): NO}.get(frozenset(answers))


def conditionalqa(questions: Iterable[dict], documents: Iterable[dict],
                  budget: int = CHAR_BUDGET) -> Iterator[Example]:
    docs = {d["url"]: d for d in documents}
    for q in questions:
        label = conditionalqa_label(q)
        if label is None:
            continue
        doc = docs[q["url"]]
        elems = [_plain(c) for c in doc["contents"]]
        text, spans, pos = "", [], 0
        for e in elems:
            spans.append([pos, pos + len(e)])
            text += e + "\n"
            pos += len(e) + 1
        # the window keeps the evidence and, for DEPENDS, the conditions the answer turns on
        want = {_plain(e) for e in q["evidences"]} | {_plain(c) for _, cs in q["answers"] for c in cs}
        evidence = sorted({elems.index(e) for e in want if e in elems})
        if len(evidence) < len(want):
            continue  # an evidence or condition sentence is not in the page as published
        body = _window(text, spans, evidence, budget - len(doc["title"]) - len(q["scenario"]) - 50)
        if body is None:
            continue
        state = f"{doc['title']}\n\n{body}\n\nUser's situation: {q['scenario'].strip()}"
        yield Example(kind="choice", state=state,
                      instructions=POLICY_Q.format(q=q["question"].strip()),
                      options=POLICY_OPTIONS, label=label, task="conditionalqa")


# ---- corpus assembly --------------------------------------------------------

def balance(examples: list[Example], key: Callable[[Example], object], *, min_count: int = 25,
            seed: int = 0) -> list[Example]:
    """Equal counts of each label within each `key` cell, so the cell predicts nothing.

    The per-claim balancing of ContractNLI (`multistep.balance_by_claim`), generalised:
    labels with fewer than `min_count` rows in a cell are dropped, cells left with one
    label are dropped whole, and the rest are subsampled to their rarest label.
    """
    rng = random.Random(seed)
    by: dict = {}
    for ex in examples:
        by.setdefault(key(ex), {}).setdefault(ex.label, []).append(ex)
    out: list[Example] = []
    for cell in sorted(by, key=repr):
        groups = [g for _, g in sorted(by[cell].items()) if len(g) >= min_count]
        if len(groups) < 2:
            continue
        n = min(len(g) for g in groups)
        for g in groups:
            out += rng.sample(g, n)
    return out


def cap_per(examples: list[Example], key: Callable[[Example], object], cap: int,
            seed: int = 0) -> list[Example]:
    """At most `cap` rows per `key` value (ShARC asks one rule up to 234 times)."""
    rng = random.Random(seed)
    by: dict = {}
    for ex in examples:
        by.setdefault(key(ex), []).append(ex)
    return [ex for k in sorted(by, key=repr) for ex in rng.sample(by[k], min(cap, len(by[k])))]


def _sharc_examples(rows: Iterable[dict]) -> list[Example]:
    """ShARC rows with their cell and rule attached, for balancing and capping."""
    out = []
    for r in rows:
        for ex in sharc([r]):
            ex._cell, ex._rule = sharc_cell(r), r["snippet"]  # type: ignore[attr-defined]
            out.append(ex)
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    from transformers import AutoTokenizer

    from .format import write_jsonl

    ap = argparse.ArgumentParser(description="Build the policy training and eval sets (v15).")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/policy_v15.jsonl")
    ap.add_argument("--eval-out", default="data/policy_v15_eval.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--sharc-per-rule", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def load(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as fh:
            data: list[dict] = json.load(fh)
            return data

    sh = f"{args.raw}/sharc/sharc1-official/json/sharc_{{}}.json"
    cq = f"{args.raw}/conditionalqa/{{}}.json"
    documents = load(cq.format("documents"))

    # cap first: capping after balancing would undo it
    sharc_train = cap_per(_sharc_examples(load(sh.format("train"))), lambda e: e._rule,  # type: ignore[attr-defined]
                          args.sharc_per_rule, seed=args.seed)
    sharc_train = balance(sharc_train, lambda e: e._cell, seed=args.seed)  # type: ignore[attr-defined]
    cq_train = list(conditionalqa(load(cq.format("train")), documents))
    # yes and no to equal counts; DEPENDS and NOT_SAID are kept whole (all are rarer)
    yes = [e for e in cq_train if e.label == YES]
    no = [e for e in cq_train if e.label == NO]
    rest = [e for e in cq_train if e.label not in (YES, NO)]
    rng = random.Random(args.seed)
    cq_train = rng.sample(yes, min(len(yes), len(no))) + rng.sample(no, min(len(yes), len(no))) + rest

    train = {"sharc": sharc_train, "conditionalqa": cq_train}
    evals = {  # natural label distributions; ShARC dev and ConditionalQA dev share no rule or page with train
        "sharc": list(sharc(load(sh.format("dev")))),
        "conditionalqa": list(conditionalqa(load(cq.format("dev")), documents)),
    }
    rows, eval_rows = [], []
    for name, exs in train.items():
        kept = fits(exs, tok, args.max_tokens)
        print(f"train {name:<14} {len(kept):>6,} rows ({len(exs) - len(kept)} over the window)")
        rows += kept
    for name, exs in evals.items():
        kept = fits(exs, tok, args.max_tokens)
        print(f"eval  {name:<14} {len(kept):>6,} rows ({len(exs) - len(kept)} over the window)")
        eval_rows += kept
    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    write_jsonl(args.eval_out, eval_rows)
    print(f"{args.out}: {len(rows):,}   {args.eval_out}: {len(eval_rows):,}")


if __name__ == "__main__":
    main()
