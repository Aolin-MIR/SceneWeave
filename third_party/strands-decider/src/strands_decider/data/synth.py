"""Synthesise stated-rule execution tasks: a document defines a computation, a record
supplies the case, and the answer is computed in code.

The structure is taken from JevBench's hard tier in the abstract -- "a precisely stated
rule plus a specific case, answer by executing the rule" -- not from its scenarios.
Domains, field names and prose here are deliberately unrelated to anything in the
benchmark.

Two design rules do the real work:

1. **Ground truth is computed, never annotated.** Label noise is the documented ceiling
   on the existing corpus; here there is none by construction.

2. **Every distractor is the output of a named reasoning error** -- forgot an exclusion,
   rounded the wrong way, used `>` for `>=`, skipped the aggregation step. So accuracy
   measures whether the rule was executed correctly, and the *chosen* distractor says
   which mistake was made. That makes the eval diagnostic rather than just a score.

Surface form is varied on purpose. RuleTaker transferred inside its own family and
nowhere else (see v5), and its narrow templated surface is the best explanation we
have for why.
"""
from __future__ import annotations

import calendar
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any


@dataclass
class Item:
    kind: str                      # noul | choice | score
    state: str
    instructions: str
    options: list[list[str]]
    label: int
    task: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("meta")
        d["weight"] = 1.0
        return d


# --------------------------------------------------------------------------
# Surface variation. Drawn per item so no two documents read the same way.
# --------------------------------------------------------------------------
ORGS = ["Calder & Voss", "Meridian Foundry", "Thistlewood Group", "Okonjo Analytics",
        "Braunstein Rail", "Ferrante Textiles", "Nakagawa Logistics", "Ravensmere Ltd",
        "Quillon Systems", "Ashgrove Dairy", "Petrosyan Marine", "Delacroix Optics"]
DOCKIND = ["OPERATING PROCEDURE", "INTERNAL STANDARD", "CONTROL NOTE", "SCHEDULE",
           "WORKING RULE", "DESK INSTRUCTION", "POLICY EXTRACT"]
HEADSTYLE = ["=" * 60, "-" * 60, "", "* * *"]


def _ref(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"


def _header(rng: random.Random, title: str) -> str:
    org = rng.choice(ORGS)
    kind = rng.choice(DOCKIND)
    rule = rng.randint(1, 9)
    style = rng.choice(HEADSTYLE)
    lines = [f"{org.upper()} - {kind} {rule}.{rng.randint(1, 9)}: {title}"]
    if style:
        lines.append(style)
    return "\n".join(lines)


def _shuffled(rng: random.Random, correct: str, wrong: list[tuple[str, str]],
              describe: Callable[[str], str]) -> tuple[list[list[str]], int, list[str]]:
    """Build an option list, shuffle it, and report which error each option encodes."""
    entries = [(correct, "correct", describe(correct))]
    seen = {correct}
    for val, err in wrong:
        if val in seen:          # an error that happens to land on the right answer
            continue             # would make the item unanswerable -- drop it
        seen.add(val)
        entries.append((val, err, describe(val)))
    rng.shuffle(entries)
    options = [[v, d] for v, _, d in entries]
    label = next(i for i, (v, e, _) in enumerate(entries) if e == "correct")
    return options, label, [e for _, e, _ in entries]


# --------------------------------------------------------------------------
# G1: running total with exclusions, first crossing of a threshold
# --------------------------------------------------------------------------
def gen_threshold(rng: random.Random) -> Item:
    n_days = rng.randint(8, 14)
    start = date(2026, rng.randint(1, 11), 1)
    counted = rng.choice(["service", "usage", "handling", "carriage"])
    excluded = rng.choice(["tax", "deposit", "levy", "surcharge"])
    limit = rng.choice([500, 800, 1000, 1200, 2000])
    pct = rng.choice([60, 75, 80, 90])
    trigger = limit * pct / 100

    rows, run, answer, naive_run, naive_answer = [], 0.0, None, 0.0, None
    for i in range(n_days):
        d = start + timedelta(days=i)
        amt = round(rng.uniform(20, limit / 4), 2)
        cat = counted if rng.random() < 0.7 else excluded
        rows.append((d, amt, cat))
        if cat == counted:
            run += amt
        naive_run += amt
        if answer is None and run >= trigger:
            answer = d
        if naive_answer is None and naive_run >= trigger:
            naive_answer = d
    if answer is None:                                  # ensure the rule can fire
        return gen_threshold(rng)
    assert naive_answer is not None  # naive_run >= run, so it fires no later than answer

    strict = None                                       # '>' instead of '>='
    run2 = 0.0
    for d, amt, cat in rows:
        if cat == counted:
            run2 += amt
        if strict is None and run2 > trigger:
            strict = d

    body = "\n".join(f"  {d.isoformat()}   {amt:>8.2f}   {cat}" for d, amt, cat in rows)
    state = (
        f"{_header(rng, 'ALERT THRESHOLD')}\n"
        f"Account {_ref(rng, 'AC')}. Ceiling for the period: {limit:.2f} units.\n"
        f"Rule. The alert is raised at the end of the first day on which the "
        f"period-to-date {counted} total is at or above {pct}% of the ceiling "
        f"({trigger:.2f} units).\n"
        f"Only {counted} entries count toward the total. {excluded.title()} entries are "
        f"recorded for audit and are excluded from the total.\n\n"
        f"  DATE         AMOUNT   TYPE\n{body}\n"
    )
    wrong = [(naive_answer.isoformat(), "counted_excluded_entries")]
    if strict:
        wrong.append((strict.isoformat(), "used_strict_inequality"))
    nxt = answer + timedelta(days=1)
    wrong.append((nxt.isoformat(), "off_by_one_day"))

    options, label, errs = _shuffled(
        rng, answer.isoformat(), wrong, lambda v: f"the alert is raised at the end of {v}")
    return Item("choice", state, "On which date is the alert raised?", options, label,
                "syn_threshold", {"rule": "threshold", "errors": errs,
                                  "edge": "exclusions"})


# --------------------------------------------------------------------------
# G2: unit conversion, stated rounding, comparison against a limit
# --------------------------------------------------------------------------
LB = 0.45359237


def gen_units(rng: random.Random) -> Item:
    limit = rng.choice([500, 750, 1000, 1500, 2500])
    rounding = rng.choice(["up", "down", "nearest"])
    include_optional = rng.random() < 0.5

    def to_kg(v: float, u: str) -> float:
        return v * LB if u == "lb" else v

    def rnd(x: float) -> float:
        if rounding == "up":
            return -(-x // 1)
        if rounding == "down":
            return x // 1
        return float(round(x))

    # Land the total within a few kg of the limit. Sampling freely puts almost every
    # item hundreds of kg inside the permit, which is both trivially answerable and
    # badly class-imbalanced -- the model could score well by always saying "within".
    # Near the boundary the conversion and the rounding direction actually decide it.
    total = limit + rng.uniform(-2.5, 2.5)

    tare_u, gear_u = rng.choice(["kg", "lb"]), rng.choice(["kg", "lb"])
    tare_kg = rng.uniform(20, 120)
    gear_kg = rng.uniform(5, 60)
    wrap = round(rng.uniform(3, 25), 2)
    net = total - tare_kg - gear_kg - (wrap if include_optional else 0.0)
    if net <= 0:
        return gen_units(rng)

    parts = [
        ("net load", round(net, 2), "kg"),
        ("carrier tare", round(tare_kg / LB if tare_u == "lb" else tare_kg, 2), tare_u),
        ("securing gear", round(gear_kg / LB if gear_u == "lb" else gear_kg, 2), gear_u),
    ]
    optional = ("protective wrap", wrap, "kg")

    # Recompute from the rounded figures the document actually shows, so the stated
    # numbers and the ground truth cannot drift apart.
    total = sum(to_kg(v, u) for _, v, u in parts)
    if include_optional:
        total += optional[1]
    final = rnd(total)
    ok = final <= limit

    listed = "\n".join(f"  {n:<16}{v:>10.2f} {u}" for n, v, u in parts)
    listed += f"\n  {optional[0]:<16}{optional[1]:>10.2f} {optional[2]}"
    state = (
        f"{_header(rng, 'LOADED MASS CHECK')}\n"
        f"Consignment {_ref(rng, 'CN')}. Permitted mass: {limit} kg.\n"
        f"Rule. Loaded mass = net load + carrier tare + securing gear"
        f"{' + protective wrap' if include_optional else ''}. "
        f"{'Protective wrap is excluded from loaded mass.' if not include_optional else ''}\n"
        f"Convert using 1 lb = {LB} kg, then round the total {rounding} to a whole "
        f"kilogram. Compare the rounded total against the permitted mass; equal is "
        f"within the permit.\n\n  COMPONENT           MASS\n{listed}\n"
    )
    return Item(
        "noul", state,
        f"Is consignment loaded mass within the permitted {limit} kg?",
        [["false", "the loaded mass exceeds the permit"],
         ["true", "the loaded mass is within the permit"]],
        1 if ok else 0, "syn_units",
        {"rule": "units", "edge": f"round_{rounding}",
         "optional_included": include_optional, "margin_kg": round(final - limit, 2)},
    )


# --------------------------------------------------------------------------
# G3: a date window with month-end clamping
# --------------------------------------------------------------------------
def _add_months(d: date, n: int) -> tuple[date, bool]:
    """Add n months, clamping to the last valid day. Returns (date, clamped?)."""
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    last = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last)), d.day > last


def gen_window(rng: random.Random) -> Item:
    # Whole-year terms land in the same month and never clamp, so the month-end rule
    # would be stated and never exercised. Odd terms from a 29th/30th/31st reach
    # February and the 30-day months, where the clamp actually decides the answer.
    months = rng.choice([1, 3, 5, 6, 7, 11, 13, 17, 18])
    inclusive = rng.random() < 0.5
    start_month = rng.choice([1, 3, 5, 7, 8, 10, 12])
    start = date(2025, start_month, rng.choice([29, 30, 31]))
    end, clamped = _add_months(start, months)

    # Sample an event near the boundary, where the clamping rule actually bites.
    offset = rng.choice([-2, -1, 0, 1, 2])
    event = end + timedelta(days=offset)
    inside = event <= end if inclusive else event < end
    if event < start:
        return gen_window(rng)

    state = (
        f"{_header(rng, 'COVER PERIOD')}\n"
        f"Agreement {_ref(rng, 'AG')}.\n"
        f"Rule. Cover starts on the commencement date and runs for {months} months. "
        f"Where the month in which cover ends has no day bearing the same number as "
        f"the commencement date, cover ends on the last day of that month.\n"
        f"The final day of cover is {'included in' if inclusive else 'excluded from'} "
        f"the period.\n\n"
        f"  Commencement date : {start.isoformat()}\n"
        f"  Event date        : {event.isoformat()}\n"
    )
    return Item(
        "noul", state, "Did the event fall within the cover period?",
        [["false", "the event fell outside the cover period"],
         ["true", "the event fell within the cover period"]],
        1 if inside else 0, "syn_window",
        {"rule": "window", "edge": "month_end_clamp" if clamped else "plain",
         "inclusive_end": inclusive, "offset_days": offset},
    )


# --------------------------------------------------------------------------
# G4: banded lookup where an aggregation step changes the band
# --------------------------------------------------------------------------
def gen_bands(rng: random.Random) -> Item:
    edges = sorted(rng.sample([2_000, 5_000, 10_000, 25_000, 50_000, 100_000], 3))
    names = ["routine", "supervised", "committee", "board"]
    window = rng.choice([30, 60, 90])

    def band(x: float) -> str:
        return names[sum(1 for e in edges if x >= e)]

    # Sample the AGGREGATE uniformly across the whole banded range first, then split
    # it into this request plus its siblings. Sampling the primary instead left the
    # lowest band unreachable and put 63% of items in one band, which a model can
    # exploit without reading the table.
    aggregate = round(rng.uniform(edges[0] * 0.3, edges[-1] * 1.6), 2)
    n_sib = rng.randint(1, 3)
    sib_share = rng.uniform(0.05, 0.6)
    sib_total = aggregate * sib_share
    cuts = sorted(rng.uniform(0, 1) for _ in range(n_sib - 1))
    parts, prev = [], 0.0
    for c in [*cuts, 1.0]:
        parts.append(round(sib_total * (c - prev), 2))
        prev = c
    siblings = [max(1.0, x) for x in parts]
    primary = round(aggregate - sum(siblings), 2)
    if primary <= 0:
        return gen_bands(rng)
    aggregate = primary + sum(siblings)
    answer, naive = band(aggregate), band(primary)
    # Aggregation must NOT always change the band. If it did, "never take the naive
    # band" would score as well as doing the arithmetic, and the item would measure a
    # shortcut instead of the rule. Keep a minority where the naive answer is right.
    if answer == naive and rng.random() < 0.55:
        return gen_bands(rng)

    # band(x) = names[number of edges x reaches], so names[i+1] begins at edges[i].
    # Rendering names[i] against edges[i] shifted every row by one, left the span
    # between the first two edges undefined, and made the last two rows contradict
    # each other -- 49% of labels then disagreed with the table as written.
    rows_out = [f"  below {edges[0]:>10,.0f}              {names[0]}"]
    for i, e in enumerate(edges[:-1]):
        rows_out.append(f"  {e:>10,.0f} to below {edges[i + 1]:>10,.0f}   {names[i + 1]}")
    rows_out.append(f"  {edges[-1]:>10,.0f} and above              {names[-1]}")
    table = "\n".join(rows_out)
    sib = "\n".join(f"    {_ref(rng, 'RQ')}   {s:>10,.2f}" for s in siblings)
    state = (
        f"{_header(rng, 'AUTHORISATION LEVEL')}\n"
        f"Rule. The authorisation level is set by the aggregated value: the value of "
        f"this request plus every related request raised by the same originator within "
        f"the preceding {window} days.\n"
        f"Thresholds are inclusive of the lower bound.\n\n"
        f"  AGGREGATED VALUE        LEVEL\n{table}\n\n"
        f"  This request   : {primary:>12,.2f}\n"
        f"  Related requests raised in the preceding {window} days:\n{sib}\n"
    )
    wrong = [(naive, "skipped_aggregation")]
    idx = names.index(answer)
    if idx + 1 < len(names):
        wrong.append((names[idx + 1], "band_off_by_one_high"))
    if idx:
        wrong.append((names[idx - 1], "band_off_by_one_low"))
    options, label, errs = _shuffled(
        rng, answer, wrong, lambda v: f"authorise at the {v} level")
    return Item("choice", state, "Which authorisation level applies to this request?",
                options, label, "syn_bands",
                {"rule": "bands", "errors": errs, "edge": "aggregation"})


# --------------------------------------------------------------------------
# G5: verify someone else's work against the stated requirements
# --------------------------------------------------------------------------
def gen_verify(rng: random.Random) -> Item:
    # A proportional step and an absolute step, in that order. Addition and
    # subtraction commute, so an earlier version's "wrong order" fault produced the
    # *identical* number and the item asked whether a correct answer was wrong --
    # exactly the ambiguous labelling this generator exists to avoid. A percentage
    # followed by an addition does not commute, so order genuinely changes the value.
    a = round(rng.uniform(40, 400), 1)
    pct = round(rng.uniform(5, 40), 1)
    added = round(rng.uniform(5, 60), 1)
    dp = rng.choice([1, 2, 3])
    truth = round(a * (1 - pct / 100) + added, dp)

    # Half the responses are correct. Drawing uniformly over the fault types made
    # 80% of items "does not satisfy", which a model can exploit without reading.
    fault = "none" if rng.random() < 0.5 else rng.choice(
        ["arithmetic", "precision", "order", "sign"])
    if fault == "none":
        shown = truth
    elif fault == "arithmetic":
        shown = round(truth + rng.choice([-1, 1]) * round(rng.uniform(0.5, 4), dp), dp)
    elif fault == "precision":
        shown = truth
    elif fault == "order":                    # adds first, then applies the reduction
        shown = round((a + added) * (1 - pct / 100), dp)
    else:                                     # applies the percentage as a gain
        shown = round(a * (1 + pct / 100) + added, dp)
    if fault in ("arithmetic", "order", "sign") and abs(shown - truth) < 10 ** -dp:
        return gen_verify(rng)                # the fault made no difference here

    # Render to the requested precision. Python drops trailing zeros, so a correct
    # 230.130 printed as "230.13" silently failed the "to 3 decimal places" half of
    # the request while still being labelled correct.
    shown_dp = max(0, dp - 1) if fault == "precision" else dp
    reduction = round(a * pct / 100, 3)
    # Every response states its working the same way; only the number differs. An
    # earlier version had the wrong-order response announce "applied the addition
    # first", which let v7 score 0.941 on that split by reading the prose instead of
    # doing any arithmetic -- a surface cue, not a reasoning test.
    steps = (f"Reduction: {pct}% of {a} = {reduction}. "
             f"Starting quantity {a}, final quantity {shown:.{shown_dp}f}.")
    request = (f"A vessel holds {a} units. Its contents are reduced by {pct}%, after "
               f"which {added} units are added. Report the final quantity to {dp} "
               f"decimal place{'s' if dp > 1 else ''}, applying the events in "
               f"chronological order.")
    state = json.dumps({"request": request, "response": steps}, ensure_ascii=False)
    return Item(
        "noul", state,
        "Does the response fully and correctly satisfy the request?",
        [["false", "the response does not satisfy the request as stated"],
         ["true", "the response fully and correctly satisfies the request"]],
        1 if fault == "none" else 0, "syn_verify",
        {"rule": "verify", "edge": fault},
    )


GENERATORS = {
    "syn_threshold": gen_threshold,
    "syn_units": gen_units,
    "syn_window": gen_window,
    "syn_bands": gen_bands,
    "syn_verify": gen_verify,
}


# Edge cases held out of training, so the synthetic eval measures transfer to an
# unseen variant of a rule rather than recall of a seen one. Each is a parameter the
# rule text still states explicitly -- the model has to read it, not remember it.
HELD_OUT_EDGES = {
    "syn_units": {"round_nearest"},      # trains on round up / round down
    "syn_window": {"month_end_clamp"},   # trains on windows that need no clamping
    "syn_verify": {"order"},             # trains on arithmetic / precision / sign
}


# Edges the held-out eval keeps alongside the held-out ones, so it is not single-class.
# `syn_verify`'s faults are all labelled "does not satisfy", so an eval of nothing but
# `order` items has a constant label -- v7 scored 1.000 on it by answering "false"
# every time, measuring nothing. Pairing them with correct responses makes it a real
# discrimination: is this right, or was it computed in the wrong order?
EVAL_COMPANION_EDGES = {"syn_verify": {"none"}}


def generate(n: int, seed: int = 0, only: list[str] | None = None,
             split: str = "all", tries_per_item: int = 60) -> list[Item]:
    """`split`: 'all', 'train' (drop held-out edges), or 'heldout' (those + companions).

    Rejection-samples rather than post-filtering so the requested count is exact and
    the generator mix stays even.
    """
    rng = random.Random(seed)
    names = only or list(GENERATORS)
    out: list[Item] = []
    i = 0
    while len(out) < n:
        name = names[i % len(names)]
        i += 1
        held = HELD_OUT_EDGES.get(name, set())
        wanted = held | EVAL_COMPANION_EDGES.get(name, set())
        for _ in range(tries_per_item):
            it = GENERATORS[name](rng)
            edge = it.meta.get("edge")
            if split == "all" or not held:
                break
            if split == "train" and edge not in held:
                break
            if split == "heldout" and edge in wanted:
                break
        else:
            continue                      # this generator cannot serve this split
        edge = it.meta.get("edge")
        if split == "heldout" and held and edge not in wanted:
            continue
        if split == "train" and edge in held:
            continue
        out.append(it)
    return out


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for it in generate(n, seed=1):
        print("=" * 78)
        print(f"[{it.task}] {it.kind}  meta={it.meta}")
        print(it.state)
        print("ASK:", it.instructions)
        for k, (nm, desc) in enumerate(it.options):
            print(f"   {'*' if k == it.label else ' '} {nm} - {desc}")
