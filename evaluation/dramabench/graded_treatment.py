"""Anchored ordinal assessment of psychological and conflict treatment."""

from statistics import median
import complete_script

VERSION = "graded-dramatic-treatment-v4"
CRITERIA = {
    "emotional_depth": ("particularity", "realization", "development"),
    "conflict_handling": ("tension", "development", "integration"),
}
LEVELS = {
    0: "A concrete substantive failure: incoherent, unmotivated or unrealized treatment.",
    1: "Legible, basically coherent treatment, but generic, repetitive or thinly realized.",
    2: "Convincing, specific treatment of these people and their circumstances.",
    3: "Finely differentiated and integrated treatment: precise responses and implications deepen or reframe the encounter.",
}
COMMON = """Assess this entire anonymous short screenplay under its ORIGINAL BRIEF.
The supplied source is a bounded passage; external novel/chapter outcomes are not
required. Fidelity and deviation receive no automatic reward or penalty. Assess
the realized screenplay, without inventing missing support or unseen source facts.
The REAL HUMAN TRAIN EXAMPLES contain original overall human preferences and
explanations, not gold dimension scores. Learn applicable distinctions from them;
ignore format remarks and task-specific constraints absent from the current brief.
Direct speech, voice-over, silence, action and explicit feeling can all be effective.
Do not systematically prefer understatement, ambiguity, novelty, loudness, tragedy,
complexity, unresolved endings or closed endings. Simplicity can be finely realized.
Use fixed SCRIPT and UNITS as the actual submitted material. Evaluate each named
criterion independently using ANCHORS; no forced score spread, quotas or comparison
with imagined masterpieces. Read modest strengths and real limitations together.
Distinguish useful economical treatment from underdevelopment and ornament from
meaningful detail. A single local flaw affects the relevant criterion; it does not
automatically invalidate every criterion or the entire draft. Use null if a
criterion is inapplicable or cannot be judged, with an explicit explanation.
Return {"attributes":{"criterion_name":{"level":0,"reason":"specific strength and limitation",
"unit_ids":["U0001"]}},"scope_interpretation":"what the brief asks this draft to accomplish"}.
Use each criterion exactly once. Levels must be integers 0,1,2,3 or null.
"""
RUBRICS = {
    "emotional_depth": COMMON + """
PARTICULARITY: Are feelings or psychological contradictions particular to these
people, relationships and lived circumstances? 0: unmotivated/incoherent attribution;
1: understandable but interchangeable feeling; 2: clearly situated, differentiated
feeling; 3: precise, individual nuances materially sharpen how the person is understood.
REALIZATION: Does expression make the experience perceptible through the encounter?
0: bare unsupported claims of depth; 1: readable but mostly stock or summarized
expression; 2: convincing specific speech, conduct and responses; 3: finely responsive
expression whose changing implications matter. Explicit dialogue can earn 3 when
its particular formulation, cost and response realize the psychological experience.
DEVELOPMENT: Does the experience acquire shape, pressure or altered meaning within
the requested scope? 0: arbitrary turns or missing treatment; 1: coherent but mainly
repeated labels or thin beats; 2: motivated development or a sustained contradiction
deepened by interactions; 3: economical, cumulative or concentrated development
that meaningfully reinterprets earlier expression. A single-scene exchange may earn
3; an ending may return to the original feeling. More emotions are not better.
Do not equate the absence of an obvious error with level3. A scene that is merely
plausible typically establishes level1, and specific convincing realization level2.
Criteria keys: particularity, realization, development.
""",
    "conflict_handling": COMMON + """
TENSION: Are incompatible needs, loyalties, understandings, positions or practical
limits enacted in these particular encounters? 0: incoherent or merely proclaimed
opposition; 1: legible but generic opposition; 2: specific, credible competing demands
and responses; 3: finely differentiated tension with consequential nuances. Warm,
comic or low-stakes encounters may earn3; hostility is not required.
DEVELOPMENT: Do actual responses modify choices, relations, practical possibilities
or the meaning of the encounter? 0: arbitrary reversal or abandoned principal
tension; 1: repetitive objection with little consequence; 2: motivated developments
and perceptible consequences; 3: economical, mutually responsive turns that deepen
or reframe the opposition. Neither increasing danger nor extra twists earn credit.
INTEGRATION: Are relevant preparation, scene connections and final treatment coherent
and effective for this brief? 0: a consequential structural break; 1: legible but
weakly prepared or connected treatment; 2: convincingly prepared consequences and
ending; 3: precise orchestration gives the final treatment richer implications.
Classify each tension's role as PRINCIPAL, SUPPORTING or BACKGROUND before judging.
An on-screen decision can complete a local exchange without showing its future
outcome. Background, parallel or remembered episodes need not have separate later
endings; check their perceptible function in the submitted draft. Ambiguous role
does not prove a dropped principal conflict. A genuine disconnected opening can
lower integration, without automatically destroying tension and development.
Apply the same role and consequence criterion to near-equivalent exchanges.
Record roles in "tension_roles":[{"role":"PRINCIPAL|SUPPORTING|BACKGROUND|UNCERTAIN",
"description":"tension and its function","unit_ids":["U0001"]}].
Criteria keys: tension, development, integration.
""",
}


def payload(script, dimension, brief, examples):
    return {"rubric": RUBRICS[dimension], "ANCHORS": LEVELS,
            "REAL_HUMAN_TRAIN_EXAMPLES": examples, "ORIGINAL_BRIEF": brief,
            "SCRIPT": script, "UNITS": complete_script.units(script),
            "script_id": "anonymous", "model": "anonymous"}


def metrics(dimension, labels, script):
    attributes = labels["attributes"]
    if set(attributes) != set(CRITERIA[dimension]):
        raise ValueError("Missing or extra treatment criteria")
    ids = {u["unit_id"] for u in complete_script.units(script)}
    values = []
    result = {}
    for criterion in CRITERIA[dimension]:
        item = attributes[criterion]
        level = item["level"]
        if level is not None and (type(level) is not int or level not in LEVELS):
            raise ValueError("Invalid ordinal level")
        if not set(item["unit_ids"]) <= ids or (level is not None and not item["unit_ids"]):
            raise ValueError("Untraceable criterion annotation")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValueError("Missing criterion rationale")
        result[dimension + "_" + criterion] = level
        values.append(level)
    result["emotion_grade" if dimension == "emotional_depth" else "conflict_grade"] = (
        median(values) if all(v is not None for v in values) else None)
    return result
