"""Minimal-pair synthesis for cross-reference and requirement checking.

Second attempt at synthetic training data. The first (`synth.py`, v9) taught the model
its generators completely -- 0.436 to 0.972 on held-out items -- and moved the JevBench
families it targeted by exactly zero tasks. Three things are different here, each taken
from what v9 and the rest of the leaderboard showed:

**No arithmetic.** Single-pass models of every size in the JevBench table stall on
`temporal_numeric` (Jev itself scores 0.27; only reasoning models pass ~0.5), and in v9
the held-out variants that did not transfer were exactly the arithmetic ones -- rounding
mode and order of operations -- while reading an unseen clamping rule did. So these
generators ask the model to *find and apply* a stated rule: follow a lookup chain,
honour an override, count, compare. `multi_hop` and `judge_hard` are the targets, and
non-reasoning systems reach 0.88 on both.

**Minimal pairs.** Every item is half of a pair whose two documents are identical except
for one decisive fact, and whose answers differ. A model that learned the surface of the
generator cannot tell the halves apart; only one that reads the decisive fact can. kev
0.6B trains on pairs like this and beats our 1.7B at a third of the size. Rendering is a
pure function of a `world` dict, so the halves of a pair differ in exactly the perturbed
field and nothing else.

**Realistic documents.** v9's documents were ~400 characters of clean tables; JevBench's
hard states average ~1,300 characters of prose with sections that do not matter. These
carry a purpose section, a superseded table explicitly marked do-not-use, a glossary, a
revision history, section order that varies, and sometimes a requester's note that names
a plausible wrong answer.

Held out of training entirely, for the transfer eval: one whole domain skin (`claim`)
for cross-reference, and one requirement type (`date`) for requirement checking.
"""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict


@dataclass
class Item:
    kind: str
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


ORGS = ["Harrowgate Mutual", "Kestrel Freight", "Oduya Health Network", "Linfield Utilities",
        "Marchetti Group", "Vantadour Systems", "Sørhaug Energi", "Pemberton & Achebe",
        "Tidewell Retail", "Castellan Insurance", "Ilori Logistics", "Brightmoor Estates",
        "Ngata Water", "Faulkes Engineering", "Yarrow Public Library", "Delphine Air"]
PEOPLE = ["A. Okafor", "B. Lindqvist", "C. Ramírez", "D. Haddad", "E. Novak", "F. Mensah",
          "G. Tanaka", "H. Oyelaran", "I. Petrov", "J. Castillo", "K. Brennan", "L. Achterberg",
          "M. Sato", "N. Delacroix", "O. Mwangi", "P. Varga", "Q. Holloway", "R. Iyer",
          "S. Kowalczyk", "T. Adeyemi", "U. Bergström", "V. Moreau", "W. Chukwu", "X. Laine",
          "Y. Fontaine", "Z. Obi", "Ab. Rossi", "Ba. Nakamura", "Ca. Dubois", "Da. Eriksen"]


def _ref(rng: random.Random, p: str) -> str:
    return f"{p}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"


# ==========================================================================
# 1. Cross-reference: entity -> category (with override) -> assignee by group
# ==========================================================================
SKINS: dict[str, dict[str, Any]] = {
    "incident": dict(
        title="ALERT ROUTING HANDBOOK",
        entity="component", entity_pl="components",
        entities=["ledger-sync", "auth-gateway", "search-indexer", "billing-export",
                  "media-transcoder", "report-builder", "cache-warmer", "webhook-relay",
                  "geo-resolver", "queue-drainer", "token-minter", "audit-shipper"],
        category="owning team",
        cats=["Payments", "Identity", "Discovery", "Data", "Media", "Reliability"],
        override_cats=["Platform", "Core Operations"],
        override_rule="Components on the legacy list are routed to the {oc} team, "
                      "whatever the ownership map says.",
        override_head="Legacy list",
        condition="day of the incident",
        values={"Monday": 0, "Tuesday": 0, "Wednesday": 0, "Thursday": 0,
                "Friday": 1, "Saturday": 2, "Sunday": 2},
        groups=["Mon-Thu", "Fri", "Sat-Sun"],
        assignee="on-call engineer", pool="people",
        record="INCIDENT", ask="Who must be paged for this incident?",
        verb="is paged",
    ),
    "shipment": dict(
        title="INBOUND HANDLING STANDARD",
        entity="product line", entity_pl="product lines",
        entities=["solvent drums", "lithium packs", "bulk flour", "glass panels",
                  "aerosol cartons", "seed stock", "paint tins", "fertiliser sacks",
                  "cold vaccines", "steel coil", "resin pellets", "cotton bales"],
        category="handling class",
        cats=["Class A", "Class B", "Class C", "Class D", "Class E"],
        override_cats=["Class R", "Class Q"],
        override_rule="Product lines under an active recall are handled as {oc}, "
                      "whatever the class table says.",
        override_head="Active recalls",
        condition="receiving site",
        values={"Leeds": 0, "York": 0, "Hull": 0, "Derby": 1, "Leicester": 1,
                "Coventry": 1, "Bristol": 2, "Exeter": 2, "Plymouth": 2},
        groups=["North", "Midlands", "South-West"],
        assignee="receiving bay", pool="bays",
        record="INBOUND CONSIGNMENT", ask="Which bay must receive this consignment?",
        verb="receives it",
    ),
    "facility": dict(
        title="CLEANING SCHEDULE RULES",
        entity="room type", entity_pl="room types",
        entities=["operating suite", "consulting room", "staff kitchen", "server hall",
                  "archive store", "lecture theatre", "changing room", "loading dock",
                  "prayer room", "plant room", "laboratory", "atrium"],
        category="cleaning regime",
        cats=["Regime 1", "Regime 2", "Regime 3", "Regime 4"],
        override_cats=["Regime Z", "Regime X"],
        override_rule="Rooms under renovation follow {oc}, whatever the regime table says.",
        override_head="Under renovation",
        condition="shift",
        values={"Early": 0, "Day": 0, "Late": 1, "Evening": 1, "Night": 2, "Overnight": 2},
        groups=["Morning", "Afternoon", "Night"],
        assignee="crew", pool="crews",
        record="WORK ORDER", ask="Which crew is assigned to this work order?",
        verb="is assigned",
    ),
    # Held out of training: the transfer eval asks whether the lookup skill survives an
    # entirely unseen vocabulary.
    "claim": dict(
        title="CLAIMS ALLOCATION PROCEDURE",
        entity="claim type", entity_pl="claim types",
        entities=["water ingress", "theft from vehicle", "storm roof", "subsidence",
                  "accidental glass", "fire in kitchen", "pet injury", "travel delay",
                  "lost baggage", "frozen food loss", "garden structure", "tenant damage"],
        category="handling unit",
        cats=["Unit North", "Unit South", "Unit Motor", "Unit Travel", "Unit Home"],
        override_cats=["Specialist Unit", "Review Unit"],
        override_rule="Claim types on the watch list are allocated to the {oc}, "
                      "whatever the unit table says.",
        override_head="Watch list",
        condition="policy tier",
        values={"Basic": 0, "Standard": 0, "Plus": 1, "Extra": 1, "Premier": 2, "Elite": 2},
        groups=["Core", "Enhanced", "Premium"],
        assignee="adjuster", pool="people",
        record="CLAIM", ask="Which adjuster is allocated this claim?",
        verb="is allocated",
    ),
}
TRAIN_SKINS = ["incident", "shipment", "facility"]
HELDOUT_SKINS = ["claim"]


def _pool(rng: random.Random, kind: str, n: int) -> list[str]:
    if kind == "people":
        return rng.sample(PEOPLE, n)
    if kind == "bays":
        return [f"Bay {c}" for c in rng.sample(
            [f"{i}{L}" for i in range(1, 10) for L in "ABCD"], n)]
    return [f"Crew {c}" for c in rng.sample(
        [f"{L}{i}" for L in "KLMNPRST" for i in range(1, 9)], n)]


def xref_world(rng: random.Random, skin: str) -> dict[str, Any]:
    s = SKINS[skin]
    ents = rng.sample(s["entities"], rng.randint(6, 8))
    cats = rng.sample(s["cats"], rng.randint(3, 4))
    ocat = rng.choice(s["override_cats"])
    ent_cat = {e: rng.choice(cats) for e in ents}
    # Every category must own something, or its row in the assignee table is dead text.
    for i, c in enumerate(cats):
        ent_cat[ents[i]] = c
    override = rng.sample(ents, rng.randint(1, 2))
    groups = s["groups"]
    names = _pool(rng, s["pool"], (len(cats) + 1) * len(groups))
    assign, k = {}, 0
    for c in [*cats, ocat]:
        for g in groups:
            assign[f"{c}|{g}"] = names[k]
            k += 1
    # A superseded ownership table: same entities, some mapped differently. Retained
    # "for audit" and explicitly marked do-not-use -- the realistic distractor.
    old_cat = {e: (rng.choice(cats) if rng.random() < 0.6 else ent_cat[e]) for e in ents}

    return dict(
        skin=skin, org=rng.choice(ORGS), ref=_ref(rng, "DOC"), rev=rng.randint(3, 19),
        entities=ents, cats=cats, ocat=ocat, ent_cat=ent_cat, override=override,
        groups=groups, assign=assign, old_cat=old_cat,
        record_entity=rng.choice(ents),
        record_value=rng.choice(list(s["values"])),
        record_id=_ref(rng, "REC"),
        note=(rng.choice(cats) if rng.random() < 0.5 else None),
        order=rng.sample(["purpose", "glossary", "superseded", "history"], 4),
        table_style=rng.choice(["pipe", "aligned", "colon"]),
        superseded_date=f"20{rng.randint(22, 25)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
    )


def xref_answer(w: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """The assignee, plus what each named reasoning error would have produced."""
    s = SKINS[w["skin"]]
    e = w["record_entity"]
    cat = w["ocat"] if e in w["override"] else w["ent_cat"][e]
    g = w["groups"][s["values"][w["record_value"]]]
    right = w["assign"][f"{cat}|{g}"]
    errs: dict[str, str] = {}
    errs["ignored_override"] = w["assign"][f"{w['ent_cat'][e]}|{g}"]
    errs["used_superseded_table"] = w["assign"][f"{w['old_cat'][e]}|{g}"]
    other_g = w["groups"][(s["values"][w["record_value"]] + 1) % len(w["groups"])]
    errs["wrong_group"] = w["assign"][f"{cat}|{other_g}"]
    if w["note"]:
        errs["followed_requester_note"] = w["assign"][f"{w['note']}|{g}"]
    return right, errs


def _table(style: str, head: tuple[str, str], rows: list[tuple[str, str]]) -> str:
    if style == "pipe":
        out = [f"  | {head[0]} | {head[1]} |", "  | --- | --- |"]
        out += [f"  | {a} | {b} |" for a, b in rows]
    elif style == "aligned":
        w0 = max(len(head[0]), *(len(a) for a, _ in rows)) + 3
        out = [f"  {head[0]:<{w0}}{head[1]}"] + [f"  {a:<{w0}}{b}" for a, b in rows]
    else:
        out = [f"  {a}: {b}" for a, b in rows]
    return "\n".join(out)


def render_xref(w: dict[str, Any]) -> tuple[str, str]:
    s = SKINS[w["skin"]]
    st = w["table_style"]
    E, C, A, G = s["entity"], s["category"], s["assignee"], s["condition"]

    own = _table(st, (E, C), [(e, w["ent_cat"][e]) for e in w["entities"]])
    grid_head = f"  {C:<22}" + "".join(f"{g:<18}" for g in w["groups"])
    grid = "\n".join(
        f"  {c:<22}" + "".join(f"{w['assign'][f'{c}|{g}']:<18}" for g in w["groups"])
        for c in [*w["cats"], w["ocat"]])
    groups_tbl = _table(st, (G, "group"),
                        [(v, w["groups"][gi]) for v, gi in s["values"].items()])
    over = "\n".join(f"  - {e}" for e in w["override"])

    extras = {
        "purpose": (f"PURPOSE\nThis document fixes how each {s['record'].lower()} is "
                    f"assigned. Where a line of this document conflicts with an older "
                    f"instruction, this document prevails. Figures in the revision "
                    f"history are informational."),
        "glossary": (f"GLOSSARY\n  {C.capitalize()}: the grouping that decides the "
                     f"{A}.\n  Group: the band a {G} falls into, set in the group table.\n"
                     f"  Requester note: free text from whoever raised the record; it is "
                     f"not an input to assignment."),
        "superseded": (f"SUPERSEDED {C.upper()} TABLE (withdrawn {w['superseded_date']}; "
                       f"retained for audit only, do not use)\n" +
                       _table(st, (E, C), [(e, w["old_cat"][e]) for e in w["entities"]])),
        "history": (f"REVISION HISTORY\n  r{w['rev']}: current tables issued.\n"
                    f"  r{w['rev'] - 1}: {C} table revised; previous table withdrawn.\n"
                    f"  r{w['rev'] - 2}: {A} rota reorganised by {G} group."),
    }
    core = [
        f"{C.upper()} TABLE\nEach {E} belongs to exactly one {C}.\n{own}",
        f"OVERRIDE\n{s['override_rule'].format(oc=w['ocat'])}\n"
        f"{s['override_head']}:\n{over}",
        f"{G.upper()} GROUPS\n{groups_tbl}",
        f"{A.upper()} BY {C.upper()} AND GROUP\n{grid_head}\n{grid}",
    ]
    # Distractor sections are interleaved with the rules, in an order fixed per world.
    blocks = core[:2] + [extras[w["order"][0]]] + core[2:] + [extras[k] for k in w["order"][1:]]

    rec = [f"  Reference : {w['record_id']}",
           f"  {E.capitalize():<10}: {w['record_entity']}",
           f"  {G.capitalize()} : {w['record_value']}"]
    if w["note"]:
        rec.append(f"  Requester note: \"I believe {w['note']} should take this.\"")

    doc = (f"{w['org'].upper()} - {s['title']} ({w['ref']}, revision {w['rev']})\n\n"
           + "\n\n".join(blocks)
           + f"\n\n{s['record']}\n" + "\n".join(rec) + "\n")
    return doc, s["ask"]


def _flip_xref(rng: random.Random, w: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """One decisive fact changed, answer changed. None if no perturbation flips it."""
    s = SKINS[w["skin"]]
    right, _ = xref_answer(w)
    moves = ["override", "value", "entity"]
    rng.shuffle(moves)
    for mv in moves:
        for _ in range(12):
            w2 = copy.deepcopy(w)
            if mv == "override":
                e = w2["record_entity"]
                if e in w2["override"]:
                    w2["override"] = [x for x in w2["override"] if x != e]
                    if not w2["override"]:
                        continue
                else:
                    w2["override"] = [*w2["override"], e]
            elif mv == "value":
                w2["record_value"] = rng.choice(list(s["values"]))
            else:
                w2["record_entity"] = rng.choice(w2["entities"])
            if xref_answer(w2)[0] != right:
                return w2, mv
    return None


def xref_pair(rng: random.Random, skin: str) -> list[Item]:
    for _ in range(50):
        w = xref_world(rng, skin)
        flip = _flip_xref(rng, w)
        if flip is None:
            continue
        w2, move = flip
        pair_id = _ref(rng, "PAIR")
        out = []
        for world, half in ((w, "a"), (w2, "b")):
            right, errs = xref_answer(world)
            doc, ask = render_xref(world)
            seen, opts = {right}, [(right, "correct")]
            for name, val in errs.items():
                if val not in seen:
                    seen.add(val)
                    opts.append((val, name))
            if len(opts) < 2:
                break
            rng.shuffle(opts)
            verb = SKINS[skin]["verb"]
            out.append(Item(
                "choice", doc, ask, [[v, f"{v} {verb}"] for v, _ in opts],
                next(i for i, (_, e) in enumerate(opts) if e == "correct"),
                "syn_xref",
                {"skin": skin, "pair": pair_id, "half": half, "flip": move,
                 "errors": [e for _, e in opts]}))
        if len(out) == 2:
            return out
    raise RuntimeError("could not build a flipping cross-reference pair")


# ==========================================================================
# 2. Requirement checking: does this response meet every stated requirement?
# ==========================================================================
class ReqSkin(TypedDict):
    what: str
    unit: str
    alt: str
    scale: int | None
    noun: str


REQ_SKINS: list[ReqSkin] = [
    {"what": "sample masses", "unit": "kg", "alt": "g", "scale": 1000, "noun": "sample"},
    {"what": "tank volumes", "unit": "L", "alt": "mL", "scale": 1000, "noun": "tank"},
    {"what": "meter readings", "unit": "kWh", "alt": "Wh", "scale": 1000, "noun": "meter"},
    {"what": "parcel weights", "unit": "kg", "alt": "lb", "scale": None, "noun": "parcel"},
]
REQ_TYPES = ["count", "unit", "decimals", "sort", "prefix", "countline", "date"]
TRAIN_REQS = ["count", "unit", "decimals", "sort", "prefix", "countline"]
HELDOUT_REQS = ["date"]

REQ_TEXT = {
    "count": "List exactly {n} entries.",
    "unit": "Give every quantity in {unit}.",
    "decimals": "Report every quantity to {dp} decimal place{dps}.",
    "sort": "Order the entries by quantity, {order}.",
    "prefix": "Begin each entry with its reference code.",
    "countline": "Finish with a line reading 'Entries: <number of entries>'.",
    "date": "Write every date as YYYY-MM-DD.",
}


def req_world(rng: random.Random, allowed: list[str], must: str | None = None) -> dict[str, Any]:
    sk = rng.choice(REQ_SKINS)
    k = rng.randint(3, 5)
    reqs = rng.sample(allowed, min(k, len(allowed)))
    if must and must not in reqs:
        reqs[0] = must
    n = rng.randint(4, 7)
    dp = rng.choice([1, 2, 3])
    order = rng.choice(["ascending", "descending"])
    entries = []
    for _ in range(n):
        entries.append(dict(
            code=f"{rng.choice('ABCDEFGHJKMNPR')}{rng.choice('ABCDEFGHJKMNPR')}-{rng.randint(100, 999)}",
            name=f"{sk['noun']} {rng.randint(1, 60)}",
            day=(rng.randint(1, 28), rng.randint(1, 12), 2026),
            qty=round(rng.uniform(1, 90), 4),
        ))
    return dict(
        skin=sk, reqs=reqs, n=n, dp=dp, order=order, entries=entries,
        org=rng.choice(ORGS), ref=_ref(rng, "REQ"),
        context=rng.choice([
            f"We are consolidating the {sk['what']} for the quarterly return. The "
            f"figures come from three sites and have already been reconciled, so do not "
            f"recompute anything; just present them as asked.",
            f"Please prepare the {sk['what']} for the auditor. Accuracy of the underlying "
            f"figures is not in question here -- the auditor checks the presentation "
            f"against the list below and nothing else.",
            f"The {sk['what']} below go into an automated import, which rejects any file "
            f"that departs from the format. Content was checked upstream.",
        ]),
        violation=None,
        # properties NOT required are allowed to vary; the pair shares these so they
        # cannot be what separates a satisfying response from a failing one.
        free_unsorted=rng.random() < 0.5,
        free_date=rng.choice(["iso", "dmy", "long"]),
        free_prefix=rng.random() < 0.5,
        free_countline=rng.random() < 0.5,
        free_dp=rng.choice([1, 2, 3]),
    )


def _fmt_date(d: tuple[int, int, int], style: str) -> str:
    day, m, y = d
    if style == "iso":
        return f"{y:04d}-{m:02d}-{day:02d}"
    if style == "dmy":
        return f"{day:02d}/{m:02d}/{y:04d}"
    mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{day} {mon[m - 1]} {y}"


def render_req(w: dict[str, Any]) -> tuple[str, str]:
    sk, reqs, v = w["skin"], w["reqs"], w["violation"]
    ents = list(w["entries"])
    if "count" in reqs and v == "count":
        ents = [*ents, dict(ents[0], code=ents[0]["code"][:-1] + "0",
                            name=ents[0]["name"] + "b")]

    if "sort" in reqs:
        ents = sorted(ents, key=lambda e: e["qty"], reverse=(w["order"] == "descending"))
        if v == "sort":
            i = len(ents) // 2
            ents[i - 1], ents[i] = ents[i], ents[i - 1]
    elif w["free_unsorted"]:
        ents = sorted(ents, key=lambda e: e["code"])

    dp = w["dp"] if "decimals" in reqs else w["free_dp"]
    date_style = "iso" if "date" in reqs else w["free_date"]
    lines = []
    for j, e in enumerate(ents):
        q, unit = e["qty"], sk["unit"]
        this_dp = dp
        if v == "unit" and j == len(ents) - 2:
            unit = sk["alt"]
            q = q * sk["scale"] if sk["scale"] else q / 0.45359237
        if v == "decimals" and j == len(ents) - 2:
            this_dp = dp + 1 if dp < 3 else dp - 1
        ds = date_style
        if v == "date" and j == 1:
            ds = "dmy"
        prefix = ("prefix" in reqs and not (v == "prefix" and j == 1)) or \
                 ("prefix" not in reqs and w["free_prefix"])
        head = f"{e['code']}  " if prefix else ""
        lines.append(f"{head}{e['name']:<12} {_fmt_date(e['day'], ds):<12} "
                     f"{q:.{this_dp}f} {unit}")
    shown_n = len(ents)
    if "countline" in reqs:
        lines.append(f"Entries: {shown_n + (1 if v == 'countline' else 0)}")
    elif w["free_countline"]:
        lines.append(f"Entries: {shown_n}")

    req_lines = []
    for r in reqs:
        req_lines.append("  - " + REQ_TEXT[r].format(
            n=w["n"], unit=sk["unit"], dp=w["dp"], dps="s" if w["dp"] > 1 else "",
            order=w["order"]))
    doc = (f"{w['org'].upper()} - PRESENTATION REQUEST {w['ref']}\n\n"
           f"REQUEST\n{w['context']}\nRequirements:\n" + "\n".join(req_lines) +
           "\n\nRESPONSE\n" + "\n".join(lines) + "\n")
    return doc, ""


def req_sentence(w: dict[str, Any], r: str) -> str:
    return REQ_TEXT[r].format(n=w["n"], unit=w["skin"]["unit"], dp=w["dp"],
                              dps="s" if w["dp"] > 1 else "", order=w["order"])


def _displayed_qty(w: dict[str, Any], e: dict[str, Any]) -> str:
    dp = w["dp"] if "decimals" in w["reqs"] else w["free_dp"]
    return f"{e['qty']:.{dp}f}"


def req_pair(rng: random.Random, allowed: list[str], must: str | None = None) -> list[Item]:
    for _ in range(50):
        w = req_world(rng, allowed, must)
        v = must if must else rng.choice(w["reqs"])
        # A unit violation writes one quantity in the alternative unit (25000 Wh among
        # kWh values). The order is still physically correct, but it *looks* broken, so
        # with sorting also required "which requirement fails?" has two defensible
        # answers. That is label noise -- exactly what synthesis is meant to avoid.
        if v == "unit" and "sort" in w["reqs"]:
            continue
        # A sort violation swaps two adjacent entries; if they display identically at
        # the chosen precision the swap is invisible and the label is unreadable.
        if v == "sort":
            ents = sorted(w["entries"], key=lambda e: e["qty"])
            shown = [_displayed_qty(w, e) for e in ents]
            i = len(ents) // 2
            if shown[i - 1] == shown[i]:
                continue
        break
    else:
        raise RuntimeError("could not build an unambiguous requirement pair")

    bad = copy.deepcopy(w)
    bad["violation"] = v
    pair_id = _ref(rng, "PAIR")
    as_choice = rng.random() < 0.5
    out = []
    for world, half in ((w, "a"), (bad, "b")):
        doc, _ = render_req(world)
        v = world["violation"]
        if as_choice:
            # Which requirement fails? Options are the listed requirements plus "none".
            names = list(world["reqs"])
            opts = [[r, "fails: " + req_sentence(world, r)] for r in names] + \
                   [["none", "the response meets every requirement"]]
            label = [o[0] for o in opts].index(v if v else "none")
            out.append(Item("choice", doc,
                            "Which listed requirement, if any, does the response fail?",
                            opts, label, "syn_reqcheck",
                            {"pair": pair_id, "half": half, "violation": v or "none",
                             "reqs": names}))
        else:
            out.append(Item("noul", doc,
                            "Does the response meet every requirement in the request?",
                            [["false", "at least one requirement is not met"],
                             ["true", "every requirement is met"]],
                            0 if v else 1, "syn_reqcheck",
                            {"pair": pair_id, "half": half, "violation": v or "none",
                             "reqs": list(world["reqs"])}))
    return out


# ==========================================================================
def generate_pairs(n_pairs: int, seed: int, split: str) -> list[Item]:
    """`split`: 'train' (training skins and requirement types) or 'heldout'."""
    rng = random.Random(seed)
    out: list[Item] = []
    for i in range(n_pairs):
        if i % 2 == 0:
            skins = TRAIN_SKINS if split == "train" else HELDOUT_SKINS
            pair = xref_pair(rng, rng.choice(skins))
        elif split == "train":
            pair = req_pair(rng, TRAIN_REQS)
        else:
            # Held-out requirement type, listed in every item and violated in half.
            pair = req_pair(rng, TRAIN_REQS + HELDOUT_REQS, must="date")
        # Deterministic and unique. Random reference-style ids collide at corpus scale
        # (birthday bound), which silently merged unrelated pairs in the analysis.
        for it in pair:
            it.meta["pair"] = f"{split}-{seed}-{i}"
        out += pair
    return out


if __name__ == "__main__":
    import sys

    for it in generate_pairs(int(sys.argv[1]) if len(sys.argv) > 1 else 2, seed=5,
                             split=sys.argv[2] if len(sys.argv) > 2 else "train"):
        print("=" * 80)
        print(f"[{it.task}] {it.kind}  meta={ {k: v for k, v in it.meta.items() if k != 'errors'} }")
        print(it.state)
        print("ASK:", it.instructions)
        for k, (v, d) in enumerate(it.options):
            print(f"   {'*' if k == it.label else ' '} {v} - {d}")
