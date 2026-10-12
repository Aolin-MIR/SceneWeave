"""Answer adequacy from HelpSteer2 (nvidia/HelpSteer2, CC-BY-4.0): a request, a response,
and "is the response adequate?".

JevBench's `adequacy` family asks this question, and nothing in the training corpus
does: v17 answers "adequate" on 11 of that family's 12 public tasks. HelpSteer2 has
human ratings (Scale AI, 3-5 annotators per response, 0-4 scales) of responses written
by NVIDIA's in-house models, two responses per prompt.

  adequate    helpfulness >= 3 and correctness >= 3
  inadequate  helpfulness <= 1 or correctness <= 1
  (the middle band is dropped as ambiguous)

Single-turn prompts only. Both splits are balanced to equal "yes" and "no": every
inadequate response is kept, and the adequate ones are drawn first from the same
prompts (the other response to a prompt whose first is inadequate), so that the
request alone does not predict the answer. The eval comes from HelpSteer2's
validation split and training rows only from its train split.

    python -m strands_decider.data.adequacy --raw data/raw/helpsteer2 \
        --out data/adequacy_hs2.jsonl --eval-out data/adequacy_hs2_eval.jsonl
"""
from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict

from .format import Example

TASK = "adequacy_hs2"
QUESTION = "Does the response adequately answer the request: correct, complete and on point?"
VARIANTS = [
    QUESTION,
    "Is this response good enough to send as the answer to the request?",
    "Would a careful reviewer accept this response as an adequate answer to the request?",
    "Does the response do what the request asks, without significant errors or omissions?",
]
OPTIONS = [["false", "the response is not adequate"], ["true", "the response is adequate"]]


def verdict(r: dict) -> bool | None:
    if r["helpfulness"] >= 3 and r["correctness"] >= 3:
        return True
    if r["helpfulness"] <= 1 or r["correctness"] <= 1:
        return False
    return None


def load(path: str) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if "<extra_id_1>" not in r["prompt"] and verdict(r) is not None]


def balanced(rows: list[dict], seed: int = 0) -> list[dict]:
    """Every inadequate response, and as many adequate ones, pair-mates first."""
    rng = random.Random(seed)
    bad = [r for r in rows if not verdict(r)]
    bad_prompts = {r["prompt"] for r in bad}
    good = [r for r in rows if verdict(r)]
    mates = [r for r in good if r["prompt"] in bad_prompts]
    others = [r for r in good if r["prompt"] not in bad_prompts]
    rng.shuffle(mates)
    rng.shuffle(others)
    return bad + (mates + others)[:len(bad)]


def to_example(r: dict) -> Example:
    state = f"Request:\n{r['prompt'].strip()}\n\nResponse:\n{r['response'].strip()}"
    return Example(kind="noul", state=state, instructions=QUESTION, options=[list(o) for o in OPTIONS],
                   label=int(bool(verdict(r))), task=TASK, instruction_variants=list(VARIANTS))


def main(argv: list[str] | None = None) -> None:
    import argparse

    from transformers import AutoTokenizer

    from .format import write_jsonl
    from .multistep import fits

    ap = argparse.ArgumentParser(description="Build the HelpSteer2 answer-adequacy rows.")
    ap.add_argument("--raw", default="data/raw/helpsteer2")
    ap.add_argument("--out", default="", help="training rows (train split); omitted if empty")
    ap.add_argument("--eval-out", default="data/adequacy_hs2_eval.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    splits = {"eval": (f"{args.raw}/validation.jsonl.gz", args.eval_out)}
    if args.out:
        splits["train"] = (f"{args.raw}/train.jsonl.gz", args.out)
    prompts = defaultdict(set)
    for name, (src, dst) in splits.items():
        rows = balanced(load(src), seed=args.seed)
        prompts[name] = {r["prompt"] for r in rows}
        examples = fits([to_example(r) for r in rows], tok, args.max_tokens)
        # fits() may drop rows over the window unevenly; rebalance what remains
        yes = [e for e in examples if e.label == 1]
        no = [e for e in examples if e.label == 0]
        n = min(len(yes), len(no))
        examples = yes[:n] + no[:n]
        random.Random(args.seed).shuffle(examples)
        print(f"{name}: {len(examples):,} rows ({n:,} adequate, {n:,} not), "
              f"{len(prompts[name]):,} distinct prompts")
        write_jsonl(dst, examples)
    if "train" in prompts:
        print(f"prompts shared by train and eval: {len(prompts['train'] & prompts['eval'])}")


if __name__ == "__main__":
    main()
