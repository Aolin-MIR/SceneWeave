"""Building a training corpus from public classification datasets.

The model has to learn a single skill: *read the option list in the prompt, and
bind head k to option k*. That skill only emerges if training sweeps across many
different label sets, phrasings and domains. A corpus drawn from one task teaches
the heads to memorise that task instead.

So each recipe converts an off-the-shelf dataset into (state, question, answer)
triples, and the mix is deliberately varied: topic, sentiment, intent, entailment,
ordinal ratings, and yes/no. Add your own by appending to RECIPES.

Large label sets (banking77 has 77 intents) are handled by subsampling the option
list per example down to at most `max_options`, always keeping the gold option.
That is not just a workaround for the K-slot budget -- varying N across examples is
itself the signal that stops the head from assuming a fixed number of live slots.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .format import Example


@dataclass
class RecipeSpec:
    """How to pull one dataset and what to ask about it."""

    name: str
    path: str  # HF dataset id
    config: str | None = None
    split: str = "train"
    max_examples: int = 4000
    trust_remote_code: bool = False
    # Applied to raw rows BEFORE the example cap. Some datasets pack several variants
    # into one split (RuleTaker carries every reasoning depth in `config`), and
    # filtering after truncation would leave a fraction of the requested count.
    row_filter: Callable[[Any], bool] | None = None


# Smallest option list we will cut a large label set down to. Below this the task
# stops resembling the original one.
MIN_SUBSAMPLED_OPTIONS = 3


def _subsample_options(
    options: list[list[str]],
    gold: int,
    max_options: int,
    rng: random.Random,
) -> tuple[list[list[str]], int]:
    """Keep the gold option plus a random sample of distractors.

    The kept count is drawn uniformly from [MIN_SUBSAMPLED_OPTIONS, max_options]
    rather than fixed at max_options. Two reasons:

    1. Slot coverage. With a fixed N, only slots 0..N-1 ever receive a gradient and
       the remaining heads stay at their initialisation -- so a serving request with
       an option count the corpus never contained would hit untrained slots.
    2. It strengthens the signal that N is a property of the prompt, not a constant.
    """
    if len(options) <= max_options:
        return options, gold
    n_keep = rng.randint(min(MIN_SUBSAMPLED_OPTIONS, max_options), max_options)
    others = [i for i in range(len(options)) if i != gold]
    keep = [*rng.sample(others, n_keep - 1), gold]
    keep.sort()
    return [options[i] for i in keep], keep.index(gold)


def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " ..."


def _label_names(ds: Any, column: str) -> list[str] | None:
    feat = ds.features.get(column)
    names = getattr(feat, "names", None)
    return list(names) if names else None


def _humanise(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip()


# --------------------------------------------------------------------------------
# Converters. Each takes a loaded HF dataset and yields Examples.
# --------------------------------------------------------------------------------


def _choice_from_labelled(
    ds: Any,
    *,
    task: str,
    label_column: str,
    instructions: str,
    text_column: str | None = None,
    text_fn: Callable[[Any], str] | None = None,
    descriptions: dict[str, str] | None = None,
    max_options: int,
    max_chars: int,
    rng: random.Random,
) -> list[Example]:
    """`text_fn` builds the state from several columns (title + body, say); otherwise
    `text_column` is used directly."""
    names = _label_names(ds, label_column)
    if names is None:
        raise ValueError(f"{task}: column {label_column!r} has no class names")
    descriptions = descriptions or {}
    canonical = [[_humanise(n), descriptions.get(n, "")] for n in names]

    out: list[Example] = []
    for row in ds:
        gold = int(row[label_column])
        opts, new_gold = _subsample_options(
            [list(o) for o in canonical], gold, max_options, rng
        )
        out.append(
            Example(
                kind="choice",
                state=_clip(text_fn(row) if text_fn else row[text_column], max_chars),
                instructions=instructions,
                options=opts,
                label=new_gold,
                task=task,
            )
        )
    return out


def _choice_from_string_labels(
    ds: Any,
    *,
    task: str,
    text_column: str,
    label_column: str,
    instructions: str,
    max_options: int,
    max_chars: int,
    rng: random.Random,
) -> list[Example]:
    """Like _choice_from_labelled, but for datasets whose label column holds the
    class *name* as a string rather than a ClassLabel index.

    The label set is derived from the data and sorted, so the canonical option order
    is stable across runs even though the sample order is not.
    """
    names = sorted({str(row[label_column]) for row in ds})
    index = {n: i for i, n in enumerate(names)}
    canonical = [[_humanise(n), ""] for n in names]

    out: list[Example] = []
    for row in ds:
        gold = index[str(row[label_column])]
        opts, new_gold = _subsample_options(
            [list(o) for o in canonical], gold, max_options, rng
        )
        out.append(
            Example(
                kind="choice",
                state=_clip(row[text_column], max_chars),
                instructions=instructions,
                options=opts,
                label=new_gold,
                task=task,
            )
        )
    return out


def _noul_from_claim_evidence(
    ds: Iterable[dict],
    *,
    task: str,
    evidence_column: str,
    claim_column: str,
    label_column: str,
    true_values: Collection[str],
    false_values: Collection[str],
    templates: Sequence[str],
    true_desc: str,
    false_desc: str,
    max_chars: int,
    rng: random.Random,
) -> list[Example]:
    """Does this evidence support this claim?

    The judgement the docs single out -- checking a claim against a source, for
    citation and hallucination checking. Distinct from entailment in framing: the
    state is retrieved evidence rather than a premise, and the claim is something a
    system asserted.

    Rows outside `true_values`/`false_values` are dropped rather than forced. A
    "not enough info" verdict is genuinely neither true nor false, and labelling it
    either way teaches noise -- the same reasoning as dropping MNLI's neutral class.
    """
    out: list[Example] = []
    for row in ds:
        label = str(row[label_column]).strip().upper()
        if label in true_values:
            truth = True
        elif label in false_values:
            truth = False
        else:
            continue
        claim = _clip(str(row[claim_column]), 400)
        out.append(
            Example(
                kind="noul",
                state=_clip(str(row[evidence_column]), max_chars),
                instructions=templates[0].format(claim=claim),
                instruction_variants=[t.format(claim=claim) for t in templates[1:]],
                options=[["false", false_desc], ["true", true_desc]],
                label=1 if truth else 0,
                task=task,
            )
        )
    rng.shuffle(out)
    return out



def _noul_from_pair(
    ds: Iterable[dict],
    *,
    task: str,
    state_column: str,
    claim_column: str,
    label_column: str,
    template: str,
    true_label: int,
    true_desc: str,
    false_desc: str,
    max_chars: int,
) -> list[Example]:
    """Yes/no over a (passage, claim) pair where the claim varies per example.

    Distinct from _noul_from_binary, where the question is constant and only the
    state moves. Here the instruction is built from a column, so the model sees a
    different question every time -- the framing real serving traffic uses.
    """
    out: list[Example] = []
    for row in ds:
        out.append(
            Example(
                kind="noul",
                state=_clip(row[state_column], max_chars),
                instructions=template.format(claim=_clip(str(row[claim_column]), 400)),
                options=[["false", false_desc], ["true", true_desc]],
                label=1 if int(row[label_column]) == true_label else 0,
                task=task,
            )
        )
    return out



def _score_from_ordinal(
    ds: Any,
    *,
    task: str,
    text_column: str,
    label_column: str,
    instructions: str,
    levels: Sequence[str],
    max_chars: int,
) -> list[Example]:
    """Ordinal targets keep their order: score slots are semantic, never shuffled."""
    canonical = [[str(i), desc] for i, desc in enumerate(levels)]
    out: list[Example] = []
    for row in ds:
        lvl = int(row[label_column])
        if not 0 <= lvl < len(levels):
            continue
        out.append(
            Example(
                kind="score",
                state=_clip(row[text_column], max_chars),
                instructions=instructions,
                options=[list(o) for o in canonical],
                label=lvl,
                task=task,
            )
        )
    return out


def _noul_from_binary(
    ds: Any,
    *,
    task: str,
    text_column: str,
    label_column: str,
    instructions: str,
    true_desc: str,
    false_desc: str,
    true_label: int,
    max_chars: int,
) -> list[Example]:
    out: list[Example] = []
    for row in ds:
        is_true = int(row[label_column]) == true_label
        out.append(
            Example(
                kind="noul",
                state=_clip(row[text_column], max_chars),
                instructions=instructions,
                # Canonical noul order is always [false, true].
                options=[["false", false_desc], ["true", true_desc]],
                label=1 if is_true else 0,
                task=task,
            )
        )
    return out


def _noul_from_entailment(
    ds: Any,
    *,
    task: str,
    instructions_template: str,
    max_chars: int,
) -> list[Example]:
    """MNLI-style pairs, rephrased as: given the premise, is the hypothesis true?

    Only entailment (0) and contradiction (2) are used; `neutral` is genuinely
    neither true nor false, so labelling it either way would teach the model noise.
    """
    out: list[Example] = []
    for row in ds:
        lab = int(row["label"])
        if lab not in (0, 2):
            continue
        out.append(
            Example(
                kind="noul",
                state=_clip(row["premise"], max_chars),
                instructions=instructions_template.format(
                    hypothesis=_clip(row["hypothesis"], 400)
                ),
                options=[
                    ["false", "the state does not support the statement"],
                    ["true", "the state supports the statement"],
                ],
                label=1 if lab == 0 else 0,
                task=task,
            )
        )
    return out


def _rank_levels(
    values: Sequence[float], n_levels: int, dead_zone: float
) -> list[tuple]:
    """Assign ordinal levels by quantile rank, dropping points that sit near a cut.

    Quantile binning rather than fixed thresholds: these scores (formality, aggregated
    hate-speech severity) are on no documented scale, so guessing cut points would give
    wildly unbalanced levels. Ranking guarantees every level is populated.

    Points whose percentile falls within `dead_zone / n_levels` of a cut are dropped.
    Near a boundary the human annotators themselves disagree, so a hard level there is
    label noise rather than signal -- the same reasoning behind dropping the ambiguous
    middle band of civil_comments.

    Returns [(original_index, level), ...] for the surviving points.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    n = max(1, len(order))
    margin = dead_zone / n_levels
    out = []
    for rank, idx in enumerate(order):
        p = (rank + 0.5) / n
        if any(abs(p - c / n_levels) < margin for c in range(1, n_levels)):
            continue
        out.append((idx, min(n_levels - 1, int(p * n_levels))))
    return out


def _score_from_continuous(
    ds: Any,
    *,
    task: str,
    text_column: str,
    value_column: str,
    instructions: str,
    levels: Sequence[str],
    max_chars: int,
    target: int,
    rng: random.Random,
    group_column: str | None = None,
    dead_zone: float = 0.15,
) -> list[Example]:
    """Ordinal task from a continuous annotation.

    `group_column` aggregates repeated rows before binning. measuring-hate-speech is
    annotator-level -- the same comment appears once per annotator -- so without this
    the corpus would contain duplicate states carrying conflicting labels, and the
    train/val split could put the same text on both sides.
    """
    texts: list[str] = []
    values: list[float] = []

    if group_column:
        agg: dict[Any, list[float]] = {}
        text_of: dict[Any, str] = {}
        for row in ds:
            key = row[group_column]
            agg.setdefault(key, []).append(float(row[value_column]))
            text_of.setdefault(key, row[text_column])
        for key, vals in agg.items():
            texts.append(text_of[key])
            values.append(sum(vals) / len(vals))
    else:
        for row in ds:
            texts.append(row[text_column])
            values.append(float(row[value_column]))

    canonical = [[str(i), desc] for i, desc in enumerate(levels)]
    picked = _rank_levels(values, len(levels), dead_zone)
    rng.shuffle(picked)

    out: list[Example] = []
    for idx, level in picked[:target]:
        out.append(
            Example(
                kind="score",
                state=_clip(texts[idx], max_chars),
                instructions=instructions,
                options=[list(o) for o in canonical],
                label=level,
                task=task,
            )
        )
    return out


def _score_from_stars(
    ds: Any,
    *,
    task: str,
    text_column: str,
    star_column: str,
    instructions: str,
    levels: Sequence[str],
    max_chars: int,
    target: int,
    rng: random.Random,
) -> list[Example]:
    """Ordinal task from an existing 1..L star rating, balanced across levels.

    App-store ratings are heavily skewed to 5 stars; left unbalanced the task
    degenerates into "predict 5" and teaches the head nothing about the rubric.
    """
    by_level: dict[int, list[str]] = {}
    for row in ds:
        try:
            lvl = int(float(row[star_column])) - 1  # stars are 1-based
        except (TypeError, ValueError):
            continue
        if 0 <= lvl < len(levels):
            by_level.setdefault(lvl, []).append(row[text_column])

    if not by_level:
        return []
    per_level = max(1, target // len(levels))
    cap = min(per_level, min(len(v) for v in by_level.values()))

    canonical = [[str(i), desc] for i, desc in enumerate(levels)]
    out: list[Example] = []
    for lvl, items in sorted(by_level.items()):
        rng.shuffle(items)
        for text in items[:cap]:
            out.append(
                Example(
                    kind="score",
                    state=_clip(text, max_chars),
                    instructions=instructions,
                    options=[list(o) for o in canonical],
                    label=lvl,
                    task=task,
                )
            )
    rng.shuffle(out)
    return out


def _noul_from_passage_question(
    ds: Any,
    *,
    task: str,
    passage_column: str,
    question_column: str,
    answer_column: str,
    max_chars: int,
) -> list[Example]:
    """Yes/no about a passage -- the framing closest to real serving traffic.

    The other noul recipes are text-classification reframed as yes/no, where the
    question is constant and only the state varies. Here the question varies per
    example and genuinely interrogates the state, which is what the API is for.
    """
    out: list[Example] = []
    for row in ds:
        ans = row[answer_column]
        truth = ans if isinstance(ans, bool) else str(ans).strip().lower() in ("true", "1", "yes")
        out.append(
            Example(
                kind="noul",
                state=_clip(row[passage_column], max_chars),
                instructions=str(row[question_column]).strip().rstrip("?") + "?",
                options=[
                    ["false", "the passage does not support this"],
                    ["true", "the passage supports this"],
                ],
                label=1 if truth else 0,
                task=task,
            )
        )
    return out



# --------------------------------------------------------------------------------
# Instruction pools.
#
# Twelve of the original fourteen training tasks carried exactly one instruction
# string, so the model never saw a task asked two ways. Measured cost: held-out
# accuracy moves by up to 0.15 on rewording alone, concentrated on the tasks the
# model finds hard. For a system whose premise is that you write your own question,
# that is the wrong thing to be brittle about.
#
# Pools span registers on purpose -- terse, plain, verbose -- because a user will not
# match whatever phrasing this corpus happens to use. The first entry is canonical and
# is what evaluation uses, so runs stay comparable; the rest are sampled per epoch by
# the collator, exactly as option order is.
#
# Tasks whose instruction already varies per example (mnli_entail, boolq, and the
# pair-based noul recipes) are absent by design -- they have thousands of distinct
# phrasings already.
# --------------------------------------------------------------------------------

INSTRUCTION_POOLS: dict[str, list[str]] = {
    "ag_news": [
        "Which section of the newspaper does this article belong to?",
        "What is this article about?",
        "Pick the desk that would run this story.",
        "Classify the topic of this news item.",
        "Which category?",
        "Reading this piece, which part of the paper would you expect to find it in?",
    ],
    "banking77": [
        "Which customer-service intent does this message express?",
        "What is this customer asking about?",
        "Identify the banking intent.",
        "Route this support message to the right topic.",
        "What does the customer want?",
        "Which of these best describes the reason this person got in touch?",
    ],
    "clinc150": [
        "Which assistant intent does this request express?",
        "What is the user asking the assistant to do?",
        "Identify the intent.",
        "Match this utterance to an action.",
        "What does this person want done?",
        "Which of the following actions is being requested here?",
    ],
    "dbpedia": [
        "Which kind of entity does this article describe?",
        "What sort of thing is this about?",
        "Classify the subject of this entry.",
        "What type of entity is described?",
        "Pick the category this encyclopedia entry falls under.",
    ],
    "lang_id": [
        "Which language is this text written in?",
        "What language is this?",
        "Identify the language.",
        "Which of these languages is the passage in?",
        "Name the language used here.",
    ],
    "yahoo_topics": [
        "Which category does this question belong to?",
        "What is this question about?",
        "File this question under one of these topics.",
        "Pick the best category.",
        "Which subject area does this belong to?",
    ],
    "spam": [
        "Is this message unsolicited spam?",
        "Is this spam?",
        "Would you consider this an unwanted bulk message?",
        "Junk or genuine?",
        "Decide whether this message was sent without the recipient asking for it.",
    ],
    "toxicity": [
        "Is this comment toxic, insulting, or abusive?",
        "Is this comment rude?",
        "Would this comment make someone want to leave the conversation?",
        "Is the author being abusive?",
        "Assess whether this message attacks or demeans someone.",
    ],
    "yelp_stars": [
        "How positive is this review of the business?",
        "How happy was this customer?",
        "Rate the sentiment of this review.",
        "How well did this go for the reviewer?",
        "On balance, how favourable is this review?",
    ],
    "sst5_sentiment": [
        "How positive is the sentiment of this sentence?",
        "How positive is this?",
        "Rate the sentiment.",
        "Where does this sentence fall on a negative-to-positive scale?",
        "How favourable is the tone here?",
    ],
    "app_reviews": [
        "How satisfied is this app reviewer?",
        "How happy is this user with the app?",
        "Rate the reviewer's satisfaction.",
        "Did the app work out for this person?",
        "How well is this app serving the reviewer?",
    ],
    "formality": [
        "How formal is the register of this sentence?",
        "How formal is this?",
        "Rate the formality.",
        "Would this sound out of place in a business letter?",
        "Where does this sit between casual chat and formal prose?",
    ],
    "hate_severity": [
        "How severe is the hostility directed at a person or group here?",
        "How hostile is this?",
        "Rate the level of hostility.",
        "Assess how aggressive this message is toward someone.",
        "Where does this sit between harmless and hateful?",
    ],
}


# --------------------------------------------------------------------------------
# The default mix.
# --------------------------------------------------------------------------------

RECIPES: dict[str, Callable[..., list[Example]]] = {}
SPECS: dict[str, RecipeSpec] = {}


def register(spec: RecipeSpec, fn: Callable[..., list[Example]]) -> None:
    SPECS[spec.name] = spec
    RECIPES[spec.name] = fn


register(
    RecipeSpec("ag_news", "fancyzhx/ag_news", max_examples=6000),
    lambda ds, *, max_options, max_chars, rng: _choice_from_labelled(
        ds,
        task="ag_news",
        text_column="text",
        label_column="label",
        instructions="Which section of the newspaper does this article belong to?",
        descriptions={
            "World": "international news, politics, conflict",
            "Sports": "athletes, matches, results, teams",
            "Business": "companies, markets, economics, finance",
            "Sci/Tech": "science, technology, research, computing",
        },
        max_options=max_options,
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    RecipeSpec("emotion", "dair-ai/emotion", max_examples=6000),
    lambda ds, *, max_options, max_chars, rng: _choice_from_labelled(
        ds,
        task="emotion",
        text_column="text",
        label_column="label",
        instructions="Which emotion is the author of this message expressing?",
        max_options=max_options,
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    RecipeSpec("banking77", "legacy-datasets/banking77", max_examples=8000),
    lambda ds, *, max_options, max_chars, rng: _choice_from_labelled(
        ds,
        task="banking77",
        text_column="text",
        label_column="label",
        instructions="Which customer-service intent does this message express?",
        max_options=max_options,
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    RecipeSpec(
        "massive_intent", "mteb/amazon_massive_intent", config="en", max_examples=6000
    ),
    lambda ds, *, max_options, max_chars, rng: _choice_from_string_labels(
        ds,
        task="massive_intent",
        text_column="text",
        label_column="label_text",
        instructions="Which assistant intent does this utterance request?",
        max_options=max_options,
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    RecipeSpec("yelp_stars", "Yelp/yelp_review_full", max_examples=6000),
    lambda ds, *, max_options, max_chars, rng: _score_from_ordinal(
        ds,
        task="yelp_stars",
        text_column="text",
        label_column="label",
        instructions="How positive is this review of the business?",
        levels=[
            "very negative — the reviewer is angry or warns others away",
            "negative — clearly dissatisfied",
            "mixed — some praise and some complaint",
            "positive — clearly satisfied",
            "very positive — enthusiastic recommendation",
        ],
        max_chars=max_chars,
    ),
)

register(
    RecipeSpec("sst5_sentiment", "SetFit/sst5", max_examples=6000),
    lambda ds, *, max_options, max_chars, rng: _score_from_ordinal(
        ds,
        task="sst5_sentiment",
        text_column="text",
        label_column="label",
        instructions="How positive is the sentiment of this sentence?",
        levels=[
            "very negative",
            "negative",
            "neutral",
            "positive",
            "very positive",
        ],
        max_chars=max_chars,
    ),
)

register(
    RecipeSpec("toxicity", "google/civil_comments", max_examples=6000),
    lambda ds, *, max_options, max_chars, rng: [
        Example(
            kind="noul",
            state=_clip(row["text"], max_chars),
            instructions="Is this comment toxic, insulting, or abusive?",
            options=[
                ["false", "civil, even if critical or blunt"],
                ["true", "rude, disrespectful, insulting, or hateful"],
            ],
            label=1 if float(row["toxicity"]) >= 0.5 else 0,
            task="toxicity",
        )
        for row in ds
        # Drop the ambiguous middle: crowd scores near 0.5 mean annotators disagreed,
        # so a hard 0/1 target there is label noise, not signal.
        if float(row["toxicity"]) <= 0.2 or float(row["toxicity"]) >= 0.5
    ],
)

register(
    RecipeSpec("spam", "ucirvine/sms_spam", max_examples=4000),
    lambda ds, *, max_options, max_chars, rng: _noul_from_binary(
        ds,
        task="spam",
        text_column="sms",
        label_column="label",
        instructions="Is this message unsolicited spam?",
        true_desc="promotional, scam, or bulk unsolicited message",
        false_desc="a genuine personal or transactional message",
        true_label=1,
        max_chars=max_chars,
    ),
)

register(
    RecipeSpec("mnli_entail", "nyu-mll/glue", config="mnli", max_examples=8000),
    lambda ds, *, max_options, max_chars, rng: _noul_from_entailment(
        ds,
        task="mnli_entail",
        instructions_template="Given only the state, is this statement true? {hypothesis}",
        max_chars=max_chars,
    ),
)



# --------------------------------------------------------------------------------
# Expansion mix. The original nine recipes covered 7 tasks, five of which had a
# single fixed label set -- thin ground for a model whose whole job is binding to
# option lists it has not seen. These add rubric variety to `score` (which had two
# near-identical 5-point sentiment sets) and label-set variety to `choice`.
# --------------------------------------------------------------------------------

register(
    # Oversampled: the dead zone drops ~20% and annotator aggregation collapses
    # roughly 3.4 rows into one comment.
    RecipeSpec("hate_severity", "ucberkeley-dlab/measuring-hate-speech", max_examples=40000),
    lambda ds, *, max_options, max_chars, rng: _score_from_continuous(
        ds,
        task="hate_severity",
        text_column="text",
        value_column="hate_speech_score",
        group_column="comment_id",
        instructions="How severe is the hostility directed at a person or group here?",
        levels=[
            "benign — no hostility toward any person or group",
            "edgy — rude or dismissive, but not targeting a group",
            "hostile — demeaning toward a person or group",
            "hateful — dehumanising or inciting harm",
        ],
        max_chars=max_chars,
        target=5000,
        rng=rng,
    ),
)

register(
    RecipeSpec("app_reviews", "sealuzh/app_reviews", max_examples=60000),
    lambda ds, *, max_options, max_chars, rng: _score_from_stars(
        ds,
        task="app_reviews",
        text_column="review",
        star_column="star",
        instructions="How satisfied is this app reviewer?",
        levels=[
            "very dissatisfied — app is broken or unusable",
            "dissatisfied — significant problems",
            "mixed — works but with real complaints",
            "satisfied — works well, minor gripes",
            "delighted — enthusiastic praise",
        ],
        max_chars=max_chars,
        target=5000,
        rng=rng,
    ),
)

register(
    RecipeSpec("formality", "osyvokon/pavlick-formality-scores", max_examples=9274),
    lambda ds, *, max_options, max_chars, rng: _score_from_continuous(
        ds,
        task="formality",
        text_column="sentence",
        value_column="avg_score",
        instructions="How formal is the register of this sentence?",
        levels=[
            "very informal — slang, abbreviations, no punctuation",
            "casual — conversational but readable",
            "neutral — plain standard prose",
            "formal — careful, professional or literary register",
        ],
        max_chars=max_chars,
        target=4000,
        rng=rng,
    ),
)

register(
    RecipeSpec("clinc150", "clinc/clinc_oos", config="plus", max_examples=6000),
    lambda ds, *, max_options, max_chars, rng: _choice_from_labelled(
        ds,
        task="clinc150",
        text_column="text",
        label_column="intent",
        instructions="Which assistant intent does this request express?",
        max_options=max_options,
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    # Language ID is the purest training signal for slot binding: the label set has no
    # semantic relationship to what the text is about, so there is no shortcut but to
    # read the options.
    RecipeSpec("lang_id", "papluca/language-identification", max_examples=4000),
    lambda ds, *, max_options, max_chars, rng: _choice_from_string_labels(
        ds,
        task="lang_id",
        text_column="text",
        label_column="labels",
        instructions="Which language is this text written in?",
        max_options=max_options,
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    RecipeSpec("dbpedia", "fancyzhx/dbpedia_14", max_examples=4000),
    lambda ds, *, max_options, max_chars, rng: _choice_from_labelled(
        ds,
        task="dbpedia",
        text_fn=lambda r: f"{r['title']}\n{r['content']}",
        label_column="label",
        instructions="Which kind of entity does this article describe?",
        max_options=max_options,
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    RecipeSpec(
        "yahoo_topics", "community-datasets/yahoo_answers_topics", max_examples=4000
    ),
    lambda ds, *, max_options, max_chars, rng: _choice_from_labelled(
        ds,
        task="yahoo_topics",
        text_fn=lambda r: f"{r['question_title']}\n{r['question_content']}",
        label_column="topic",
        instructions="Which category does this question belong to?",
        max_options=max_options,
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    RecipeSpec("boolq", "google/boolq", max_examples=4000),
    lambda ds, *, max_options, max_chars, rng: _noul_from_passage_question(
        ds,
        task="boolq",
        passage_column="passage",
        question_column="question",
        answer_column="answer",
        max_chars=max_chars,
    ),
)

register(
    RecipeSpec("sarcasm", "raquiba/Sarcasm_News_Headline", max_examples=4000),
    lambda ds, *, max_options, max_chars, rng: _noul_from_binary(
        ds,
        task="sarcasm",
        text_column="headline",
        label_column="is_sarcastic",
        instructions="Is this headline sarcastic or satirical rather than straight news?",
        true_desc="written to mock, exaggerate, or joke",
        false_desc="a sincere, literal news headline",
        true_label=1,
        max_chars=max_chars,
    ),
)



# --------------------------------------------------------------------------------
# Held-out probes. None of these belong in a training mix -- they exist to widen the
# evidence base for transfer, which otherwise rests on two choice tasks and one noul
# task. They deliberately span a range of distance from the training corpus:
#
#   adjacent -- close to something trained on, so transfer is expected
#   novel    -- a kind of judgement nothing in training makes
#
# If adjacent transfers and novel does not, semantic proximity is confirmed as the
# governing variable rather than anything about the architecture.
# --------------------------------------------------------------------------------

register(
    # adjacent: topic classification like ag_news/dbpedia/yahoo, different domain
    RecipeSpec("newsgroups", "SetFit/20_newsgroups", max_examples=4000),
    lambda ds, *, max_options, max_chars, rng: _choice_from_string_labels(
        ds,
        task="newsgroups",
        text_column="text",
        label_column="label_text",
        instructions="Which newsgroup was this message posted to?",
        max_options=max_options,
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    # novel: classifies what KIND of answer the question wants, not its topic
    RecipeSpec("trec_qc", "SetFit/TREC-QC", max_examples=4000),
    lambda ds, *, max_options, max_chars, rng: _choice_from_string_labels(
        ds,
        task="trec_qc",
        text_column="text",
        label_column="label_text",
        instructions="What kind of answer is this question asking for?",
        max_options=max_options,
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    # novel: scientific subject areas over long technical text
    RecipeSpec("arxiv_class", "ccdv/arxiv-classification", config="no_ref", max_examples=3000),
    lambda ds, *, max_options, max_chars, rng: _choice_from_labelled(
        ds,
        task="arxiv_class",
        text_column="text",
        label_column="label",
        instructions="Which research area does this paper belong to?",
        max_options=max_options,
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    # adjacent: entailment, the same judgement mnli_entail trains on
    RecipeSpec("rte", "nyu-mll/glue", config="rte", max_examples=2490),
    lambda ds, *, max_options, max_chars, rng: _noul_from_pair(
        ds,
        task="rte",
        state_column="sentence1",
        claim_column="sentence2",
        label_column="label",
        template="Given only the state, is this statement true? {claim}",
        true_label=0,  # glue rte: 0 = entailment
        true_desc="the state supports the statement",
        false_desc="the state does not support the statement",
        max_chars=max_chars,
    ),
)

register(
    # adjacent: does the passage answer the question -- close to boolq
    RecipeSpec("qnli", "nyu-mll/glue", config="qnli", max_examples=4000),
    lambda ds, *, max_options, max_chars, rng: _noul_from_pair(
        ds,
        task="qnli",
        state_column="sentence",
        claim_column="question",
        label_column="label",
        template="Does this passage contain the answer to: {claim}",
        true_label=0,  # glue qnli: 0 = entailment
        true_desc="the passage answers the question",
        false_desc="the passage does not answer the question",
        max_chars=max_chars,
    ),
)

register(
    # novel: grammaticality, a linguistic judgement unlike anything in training
    RecipeSpec("cola", "nyu-mll/glue", config="cola", max_examples=4000),
    lambda ds, *, max_options, max_chars, rng: _noul_from_binary(
        ds,
        task="cola",
        text_column="sentence",
        label_column="label",
        instructions="Is this a grammatically acceptable English sentence?",
        true_desc="well formed; a native speaker would accept it",
        false_desc="ungrammatical or malformed",
        true_label=1,
        max_chars=max_chars,
    ),
)

register(
    # novel: a stance about the text rather than a claim about its content
    RecipeSpec("subjectivity", "SetFit/subj", max_examples=4000),
    lambda ds, *, max_options, max_chars, rng: _noul_from_binary(
        ds,
        task="subjectivity",
        text_column="text",
        label_column="label",
        instructions="Is this sentence subjective rather than a statement of fact?",
        true_desc="opinion, judgement, or interpretation",
        false_desc="objective description of events or facts",
        true_label=1,
        max_chars=max_chars,
    ),
)



# --------------------------------------------------------------------------------
# Noul breadth. The original four covered logical entailment, two content properties
# and factual QA over a passage -- narrow for a primitive meant to answer arbitrary
# yes/no questions. These add judgement types nothing else in the corpus makes.
#
# `cola`, `subjectivity` and `sarcasm` are deliberately NOT here: they serve as
# held-out probes, and training on them would destroy the measurement.
# --------------------------------------------------------------------------------

register(
    # semantic equivalence -- adversarial, so high word overlap does not imply agreement
    RecipeSpec("paws", "google-research-datasets/paws", config="labeled_final",
               max_examples=5000),
    lambda ds, *, max_options, max_chars, rng: _noul_from_claim_evidence(
        ds,
        task="paws",
        evidence_column="sentence1",
        claim_column="sentence2",
        label_column="label",
        true_values={"1"},
        false_values={"0"},
        templates=[
            "Does this sentence mean the same thing as the state? {claim}",
            "Is this a paraphrase of the state? {claim}",
            "Same meaning? {claim}",
            "Would these two be interchangeable: {claim}",
        ],
        true_desc="the two say the same thing",
        false_desc="the two differ in meaning, despite similar wording",
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    # fact verification against retrieved evidence -- the citation-checking use case
    RecipeSpec("vitaminc", "tals/vitaminc", max_examples=12000),
    lambda ds, *, max_options, max_chars, rng: _noul_from_claim_evidence(
        ds,
        task="vitaminc",
        evidence_column="evidence",
        claim_column="claim",
        label_column="label",
        true_values={"SUPPORTS"},
        false_values={"REFUTES"},
        templates=[
            "Does the evidence support this claim? {claim}",
            "Is this claim borne out by the state? {claim}",
            "Check this against the source: {claim}",
            "Given only this evidence, is the following true? {claim}",
            "Would you say the source backs this up? {claim}",
        ],
        true_desc="the evidence supports the claim",
        false_desc="the evidence contradicts the claim",
        max_chars=max_chars,
        rng=rng,
    ),
)

register(
    # coreference -- does the pronoun resolve the way the claim assumes
    RecipeSpec("wnli", "nyu-mll/glue", config="wnli", max_examples=635),
    lambda ds, *, max_options, max_chars, rng: _noul_from_pair(
        ds,
        task="wnli",
        state_column="sentence1",
        claim_column="sentence2",
        label_column="label",
        template="Does the state imply this? {claim}",
        true_label=1,  # glue wnli: 1 = entailment
        true_desc="the state implies the statement",
        false_desc="the state does not imply the statement",
        max_chars=max_chars,
    ),
)

register(
    # boolq framing in a biomedical register -- tests domain shift, not a new judgement
    RecipeSpec("pubmed_qa", "qiaojin/PubMedQA", config="pqa_labeled", max_examples=1000),
    lambda ds, *, max_options, max_chars, rng: _noul_from_claim_evidence(
        ds,
        task="pubmed_qa",
        evidence_column="long_answer",
        claim_column="question",
        label_column="final_decision",
        true_values={"YES"},
        false_values={"NO"},
        templates=[
            "Based on this abstract: {claim}",
            "Does the research support a yes to: {claim}",
            "According to these findings: {claim}",
        ],
        true_desc="the findings support a yes",
        false_desc="the findings do not support a yes",
        max_chars=max_chars,
        rng=rng,
    ),
)



# --------------------------------------------------------------------------------
# RuleTaker: apply a stated ruleset to a case.
#
# JevBench exposed the gap these address. We score 1.000 on fact, routing and
# tool_selection -- all single-step classification -- against 0.167 on tradeoff,
# 0.278 multi_hop and 0.316 long_policy. Every one of the other training tasks
# answers in one step, so the corpus has never asked the model to compose anything.
#
# RuleTaker is the only source found that ships *graded* compositional depth: the
# same task at 0, 1, 2, 3 and 5 inference steps. That makes it an instrument as well
# as training data -- hold the deep splits out and the accuracy-versus-depth curve
# shows exactly where one forward pass stops being enough.
#
# Registered per depth so `--holdout` can route them individually. The intended
# experiment trains on d0/d1/d2 and holds out d3/d5, which asks whether shallow
# composition generalises to deeper chains.
#
# `natlang` is a separate axis: the same logic phrased naturally rather than
# templated, which separates "cannot compose" from "cannot read formal notation".
# --------------------------------------------------------------------------------

_RULETAKER_TEMPLATES = [
    "Given only the rules and facts in the state, does this follow? {claim}",
    "Is this entailed by the state? {claim}",
    "Applying the rules above, is this the case? {claim}",
    "Does the state establish this? {claim}",
    "Following the stated rules, is the following true? {claim}",
]


def _ruletaker(name: str, config_value: str, n: int) -> None:
    def row_filter(r: Mapping[str, Any]) -> bool:
        return bool(r["config"] == config_value)
    register(
        RecipeSpec(
            name,
            "tasksource/ruletaker",
            max_examples=n,
            row_filter=row_filter,
        ),
        lambda ds, *, max_options, max_chars, rng, _t=name: _noul_from_claim_evidence(
            ds,
            task=_t,
            evidence_column="context",
            claim_column="question",
            label_column="label",
            # The label is a string, and "not entailment" must not be matched by a
            # prefix test against "entailment" -- exact membership only.
            true_values={"ENTAILMENT"},
            false_values={"NOT ENTAILMENT"},
            templates=_RULETAKER_TEMPLATES,
            true_desc="the rules and facts establish this",
            false_desc="the rules and facts do not establish this",
            max_chars=max_chars,
            rng=rng,
        ),
    )


_ruletaker("ruletaker_d0", "depth-0", 4000)
_ruletaker("ruletaker_d1", "depth-1", 4000)
_ruletaker("ruletaker_d2", "depth-2", 4000)
_ruletaker("ruletaker_d3", "depth-3", 4000)
_ruletaker("ruletaker_d5", "depth-5", 4000)
_ruletaker("ruletaker_natlang", "NatLang", 4000)


def build_recipe(
    name: str,
    *,
    max_options: int = 16,
    max_chars: int = 2000,
    seed: int = 0,
    max_examples: int | None = None,
) -> list[Example]:
    """Load one dataset and convert it. Requires `datasets` and network access."""
    from datasets import load_dataset

    if name not in SPECS:
        raise KeyError(f"unknown recipe {name!r}; known: {sorted(SPECS)}")
    spec = SPECS[name]
    rng = random.Random(seed)

    ds = load_dataset(
        spec.path,
        spec.config,
        split=spec.split,
        trust_remote_code=spec.trust_remote_code,
    )
    if spec.row_filter is not None:
        ds = ds.filter(spec.row_filter)

    limit = max_examples if max_examples is not None else spec.max_examples
    # Shuffle before truncating: many HF splits are ordered by label, and taking a
    # head slice of those would yield a corpus missing most classes.
    if limit and len(ds) > limit:
        ds = ds.shuffle(seed=seed).select(range(limit))

    examples = RECIPES[name](ds, max_options=max_options, max_chars=max_chars, rng=rng)

    # Attach alternative phrasings where the task has a fixed instruction. Done here
    # rather than inside each converter so recipes stay unaware of it, and skipped for
    # tasks whose instruction already varies per example.
    pool = INSTRUCTION_POOLS.get(name)
    if pool:
        canonical, variants = pool[0], pool[1:]
        for ex in examples:
            ex.instructions = canonical
            ex.instruction_variants = variants
    return examples


def available_recipes() -> list[str]:
    return sorted(SPECS)
