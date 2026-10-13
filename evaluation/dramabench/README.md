# Full-script content diagnostics based on DramaBench

The related research was accepted at the ICLR 2026 ICBINB Workshop as
[I Can't Believe LLMs Still Can't Write Drama: Multi-Dimensional Failures in Script Continuation](https://openreview.net/forum?id=9sVGZgUIFz),
as listed on the [official accepted-paper page](https://sites.google.com/view/icbinb-2026/papers).
The original longer framework paper and pinned prompts remain provenance sources.

## Complete adaptations

### Anchored psychological and conflict treatment

The binary supported-treatment checks indicate basic plausibility and may saturate.
`graded_treatment.py` separately assesses three criteria for each dimension using
fixed 0--3 anchors. Emotion uses particularity, realization and development. Conflict
uses tension, development and integration under the original adaptation brief.
The per-reading grade is the median of the three criterion levels; report the mean
of two repeated readings and retain both sets of labels. Report distributions and
repeatability with method means: levels are ordinal, and arithmetic summaries are
descriptive, not calibrated audience utility. Do not force dispersion or select
rubrics because a particular method wins.

```sh
python evaluation/dramabench/evaluate_treatment.py \
  --inputs selected_inputs.json --briefs original_briefs.json \
  --human-train-examples human_train_examples.json --output treatment_results \
  --gateway-command 'python your_wiki_adapter.py' --workers 8
```

Original briefs are keyed by `work_id`. Human examples contain the original
`writing_request`, `a`, `b`, `human_probability_a`, and
`released_human_explanations`. Use verified TRAIN material and protect evaluation
works. Overall human votes provide examples of reader priorities; do not fabricate
dimension scores from them. A local or secondary episode does not veto the whole
script. First identify principal, supporting and background tensions; an on-screen
decision may complete a local exchange without a later outcome. Both resolved and
open endings, direct expression and subtext, quiet and confrontational treatments
use the same quality anchors. Fix and qualify these rules on independent development
material before method scoring. Keep older binary checks in their original archive.

Use `--complete-script` for the fixed-unit complete-adaptation protocol:

```sh
python evaluation/dramabench/evaluate.py \
  --inputs selected_inputs.json --output complete_results --complete-script \
  --gateway-command 'python your_wiki_adapter.py' --workers 8
```

`complete_script.py` fixes action paragraphs, dialogue turns and scenes before
inference. Narrative and character annotations must cover every eligible unit;
emotional annotations must cover every scene. The reported rates measure redundant
units, unsupported character breaks, distinctive dialogue, supported psychological
treatment, continuity breaks and supported conflict treatment. Emotion and conflict
labels distinguish supported, unsupported, inapplicable and uncertain treatment.
Uncertain/inapplicable items remain excluded from the relevant denominator.
Full-script emotional development includes intermediate turns and meaningful
sustained contradictions; a different endpoint is not required. Resolved and open
endings use the same preparation and realization criterion.

Driver-beat ENR, emotional endpoint shifts, raw emotion counts and the original
conflict-category weights are omitted from this protocol because they do not
measure complete-adaptation quality adequately. Format is also omitted. Rates
describe specific content properties; they do not combine into a single quality
score or certify human preference. Fix and verify criteria on independent
development material before evaluating any method, then apply the same criteria
and missing-value rules to all frozen candidates. Use a separate output directory;
the earlier rubrics, annotations and results remain unchanged.

## Earlier diagnostic protocol

These task-adapted rubrics evaluate complete short screenplays. They use the
content dimensions of [DramaBench](https://arxiv.org/abs/2512.19012), with explicit
instructions for coverage, situated character change, emotional subtext,
reported speech and internal continuity, and supported endings. The upstream
prompt reference is commit `bcc38b546a8bb9883c5e61f77ffbfd27cc45a70e` of
[IIIIQIIII/DramaBench](https://github.com/IIIIQIIII/DramaBench).

The scripts are evaluated in full: `CONTEXT` followed by `CONTINUATION` is the
complete input. Narrative efficiency, emotional shift and complex-emotion counts
are descriptive statistics. Character change, quiet passages and resolved
endings receive no automatic penalty. The original conflict weights remain
available as a descriptive diagnostic; treatment fit is reported separately.
No overall aesthetic score is computed. Results are not comparable with the
original DramaBench leaderboard, which uses a different task and annotation model.

Zero `ooc_rate` or `logic_break_rate` means no violation was detected among the
annotated eligible units; lower rates indicate fewer flagged errors. Missing or
undefined values remain missing and are never filled with zero. Zero `arc_score`
means no opening-to-ending shift, even when intermediate psychological development
exists. Zero conflict weight denotes `RESOLUTION`, not poor writing. Under the
current character rubric, the historical `voice_distinctiveness` output key is the
fraction labeled `IN_CHARACTER` (contextually motivated dialogue). Report it as
motivated-dialogue share; it does not establish distinctive character voice.

## Input

Supply a JSON object with `records` and `pairs`. A record contains `work_id`,
`author_group`, `method`, `script`, and its UTF-8 SHA-256 in `script_sha256`.
A pair contains `work_id`, `author_group`, `baseline`, `ours_hash`, and
`baseline_hash`. Freeze candidate selection and the eligible work cohort before
annotation. Do not select texts using these diagnostic outcomes.

## Annotation

```sh
python evaluation/dramabench/evaluate.py \
  --inputs selected_inputs.json --output results \
  --gateway-command 'python your_wiki_adapter.py' --workers 8
```

The transport adapter reads a JSON request from standard input. It passes `system`
and the serialized `input` to the authorized Wiki gateway using `gpt-6-luna`, low
reasoning effort, and the model's native output allowance. Its output is a JSON
object containing `status`, `served_model`, `source_requested`, `finish_reason`,
and parsed `result`. Required successful values are `complete`, `gpt-6-luna`,
`wiki`, and `stop`, respectively. The adapter must obtain these identities from
actual inference receipts. Credentials and organization-specific gateway code
are supplied locally and are not part of this repository.

Identical texts share annotations. Existing annotation files are reused,
including failures; no silent retries, fallback models, or missing-value ties
are used. All dimension labels are retained. Counts are calculated in code,
independently of any model-reported totals. Undefined denominators stay missing.
World-continuity rates use only constraints annotated as `WORLD_FACT`. Character
claims and beliefs remain in the raw labels, with their contradictions recorded,
but they do not enter the world-continuity error rate.

```sh
python evaluation/dramabench/evaluate.py \
  --inputs selected_inputs.json --output results --summarize-only
```

The `per_item` output reports each method's mean separately for every content
statistic. Within each statistic, all methods use the same work intersection;
the work identifiers and sample size are retained. Missing annotations and
undefined denominators are excluded from that statistic for every method.
No format score or combined content score is added.

Reported contrasts are paired mean differences, with author-cluster bootstrap
intervals (5,000 samples, seed 20261013). Metric-specific sample sizes are retained.
Intervals reflect sampling uncertainty conditional on annotations, not annotation
error or human aesthetic agreement. Structural controls in `rubrics.fixtures()`
check narrow category distinctions and do not establish human taste validity.

## Task-conditioned pairwise comparisons

This optional tool is separate from the paper's trained human-preference
evaluator and per-item content results.

`compare.py` uses qualitative content dimensions to compare the realized dramatic
effect of two drafts. Supply `cases` (each with `id`, `request`, `a`, and `b`) and
`human_train_demonstrations` in a JSON file. Demonstrations retain actual human
preferences and explanations. Optional request retrieval uses prepared original
human TRAIN records only, with eight TF-IDF-nearest requests; candidate text and
comparison outcomes do not select examples.

```sh
python evaluation/dramabench/compare.py \
  --inputs comparison_cases.json --output pairwise_results \
  --human-train human_train.json --retrieved-examples 8 \
  --gateway-command 'python your_wiki_adapter.py' --workers 8
```

Prepared human TRAIN records use `prompt`, `chosen`, `rejected`,
`preference_probability`, `pair_id`, and `label_provenance`. Here `chosen` and
`rejected` are the input A/B slots; `preference_probability` is the original
human probability of choosing slot A. Preserve any original human explanations
in `original_human_annotations`. TRAIN and evaluation groups must be disjoint.

Both presentation orders are used. Reported relative class confidences are
aligned and averaged: `q(A) = (p_AB(A) + p_BA(B))/2`, with analogous expressions
for B and tie. They are not measured audience vote frequencies. A unique largest
aggregate supplies the predicted class. Equal largest values remain unresolved;
an aggregate tie requires both raw semantic choices to agree on tie. Unavailable
or conflicting dimension judgments remain missing. Exact text identity is a
deterministic tie and creates no inference call. All raw order judgments remain
available for sensitivity analysis.

Validate the resulting comparator against real human judgments before drawing
method-quality conclusions. Report human agreement, coverage and raw order
agreement separately. These predictions are distinct from human votes, the trained
human-preference model, and the original DramaBench content statistics.
