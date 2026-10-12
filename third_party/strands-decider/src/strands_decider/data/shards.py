"""Labelling a corpus on N GPUs: one process per shard, then a merge (teacher.py, replay.py).

Shard i of N writes `<out stem>.shard<i>of<N>.jsonl`, and `<that>.done` once it has
finished. Each script gives shard i whole batches i, i + N, i + 2N, ... of the batching
the single-process run uses, so every row is computed in the batch it would be in anyway
and the merged file is the single-process file. `--merge` loads no model: it checks that
every shard finished and that none overlap, and the script writes `--out` from the rows.
"""
from __future__ import annotations

import argparse
import json
import os


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the rows over N processes (one per GPU); see --shard-index")
    ap.add_argument("--shard-index", type=int, default=None,
                    help="label shard i of --num-shards into <out stem>.shard<i>of<N>.jsonl")
    ap.add_argument("--merge", action="store_true",
                    help="combine the --num-shards finished shards into --out (no model)")


def sharded(ap: argparse.ArgumentParser, args: argparse.Namespace) -> bool:
    """Whether this process labels one shard; checks the three flags agree."""
    # --shard-index selects shard mode even for one shard, so a runner can always
    # shard then merge, whatever its GPU count.
    if args.merge and args.shard_index is not None:
        ap.error("--merge and --shard-index are exclusive")
    if args.num_shards > 1 and not (args.merge or args.shard_index is not None):
        ap.error("--num-shards > 1 needs --shard-index (or --merge)")
    if args.shard_index is not None and not 0 <= args.shard_index < args.num_shards:
        ap.error(f"--shard-index must be in [0, {args.num_shards})")
    return args.shard_index is not None


def path(out: str, shard_index: int, num_shards: int) -> str:
    return f"{out.removesuffix('.jsonl')}.shard{shard_index}of{num_shards}.jsonl"


def read(file: str) -> dict[int, list[float]]:
    """{row index: probs} from a labels file; skips a last line an interruption cut off."""
    rows: dict[int, list[float]] = {}
    with open(file, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[d["i"]] = d["probs"]
    return rows


def mark_done(file: str) -> None:
    open(file + ".done", "w").close()


def merge(out: str, num_shards: int) -> dict[int, list[float]]:
    """Every shard's rows, after checking that each finished and that none overlap."""
    rows: dict[int, list[float]] = {}
    for k in range(num_shards):
        file = path(out, k, num_shards)
        if not os.path.exists(file + ".done"):
            raise SystemExit(f"{file} is missing or unfinished (no {file}.done); rerun shard {k}")
        part = read(file)
        overlap = rows.keys() & part.keys()
        if overlap:
            raise SystemExit(f"{file} repeats rows {sorted(overlap)[:5]} -- mixed shard counts?")
        rows.update(part)
    return rows
