"""Fixed-input content annotations for complete adaptations, without format scores."""

import re
from collections import Counter


VERSION = "complete-script-content-v3.2"
DIMENSIONS = ("narrative_efficiency", "character_consistency", "emotional_depth",
              "logic_consistency", "conflict_handling")
METRICS = ("redundant_unit_rate", "ooc_rate", "distinctive_dialogue_rate",
           "supported_emotional_treatment_rate", "logic_break_rate",
           "supported_conflict_treatment_rate")
COMMON = """Read the ENTIRE anonymous complete screenplay, including its ending.
Use the fixed UNITS; do not split, merge or omit eligible units. Read each unit
in its surrounding scene and whole-script context. Method identity is hidden.
Source fidelity, length, loudness, novelty, suffering, explanation, ambiguity,
conflict escalation and a resolved ending earn no automatic credit or penalty.
Judge observable treatment of particular people and situations. A quiet scene
can develop meaning; a dramatic-looking gesture can fail to do so. Do not invent
unstated support. A possible explanation needs actual contextual support.
Return only JSON with the specified fields. Keep each reason under 25 words.
UNCERTAIN and NOT_APPLICABLE are separate from success, failure, or equality.
"""
RUBRICS = {
    "narrative_efficiency": COMMON + """
For EVERY ACTION and DIALOGUE unit, classify NONREDUNDANT, REDUNDANT, or UNCERTAIN.
REDUNDANT requires unnecessary repetition of an already established function,
with no changed implication, reaction, rhythm, preparation or subsequent use.
Do not call a first atmospheric detail, routine transition, silence, relational
hesitation or a prepared closing echo redundant just because no plot state changes.
Repetition gains a new function only when timing, response, circumstance or later
use concretely changes its effect. Reworded repeated exposition without new use
is still redundant. NONREDUNDANT only means no gratuitous functional repetition;
it is not an assertion that every unit is excellent. State which earlier unit
is repeated for every REDUNDANT finding.
Return {"units":[{"unit_id":"U0001","classification":"NONREDUNDANT|REDUNDANT|UNCERTAIN",
"repeats_unit_id":null,"reason":"specific contribution or unnecessary repetition"}]}.
""",
    "character_consistency": COMMON + """
For EVERY DIALOGUE unit, classify IN_CHARACTER, NEUTRAL, OOC, or UNCERTAIN. OOC means an
unsupported break in an established disposition, knowledge, ability or relation;
contextually prepared change, deception and different conduct with different
listeners are legitimate. Do not invent preparation to excuse a break.
Separately classify voice as SPECIFIC, GENERIC, or UNCERTAIN. SPECIFIC needs
recognizable diction, speech tactic, rhythm or evasions particular to this person
in this relationship. Being logical or well motivated alone is not a distinct voice.
Useful generic dialogue is legitimate. A new speaker without prior characterization
cannot be judged against an invented personality.
Return {"dialogue_analysis":[{"unit_id":"U0001","classification":"IN_CHARACTER|NEUTRAL|OOC|UNCERTAIN",
"voice":"SPECIFIC|GENERIC|UNCERTAIN","reason":"context and expression checked"}]}.
""",
    "emotional_depth": COMMON + """
For EVERY scene, trace any substantial psychological development or sustained
psychological contradiction, including intermediate turns, subtext, and a return
to the opening feeling. Check how concrete speech, action, withholding or changes
in interaction make the experience perceptible. SUPPORTED means an interpretable
psychological course or contradiction is realized and contextually motivated.
UNSUPPORTED means attempted psychological treatment is merely asserted, incoherent,
unprepared, or relies on stock gestures without contextual realization.
NOT_APPLICABLE means no substantial psychological treatment is attempted in this
scene; practical or transitional scenes need not demonstrate psychological depth.
Neither a changed endpoint nor more emotional moments means better treatment.
Do not require mixed feelings, a confession, opacity, or a transformation.
Return {"scenes":[{"scene_id":"S001","classification":"SUPPORTED|UNSUPPORTED|NOT_APPLICABLE|UNCERTAIN",
"course":"specific psychological course or contradiction","unit_ids":["U0001"],
"reason":"realization through the actual encounter"}]}.
""",
    "logic_consistency": COMMON + """
Check continuity across the complete script using time, location, possession,
identity, physical capability and character knowledge. Report the categories
checked even when no relevant constraints arise. Track initial and later states.
An observable change, flashback, dream, deception or acknowledged correction can
explain a difference; do not assume an unstated explanation. A person's established
knowledge is a world-state fact even though what that person claims may be false.
Separate WORLD_FACT, CLAIM and BELIEF. A report is not automatically world truth.
VIOLATED requires an unresolved contradiction at the same time/reference; an
unused fact is IRRELEVANT. Extract every substantial cross-unit constraint you
can actually check, without padding the list with arbitrary minimum counts.
An omitted ordinary action is not a contradiction: an object can be retrieved,
carried or unfolded between depicted moments without showing every movement.
Flag incompatible states only when timing, access or explicit conditions rule
out that feasible transition. Missing information is UNCERTAIN, not VIOLATED.
Return {"checked_categories":["time","location","possession","identity","capability","knowledge"],
"facts":[{"fact_id":1,"fact_kind":"WORLD_FACT|CLAIM|BELIEF",
"initial_unit_id":"U0001","later_unit_id":"U0002",
"description":"specific constraint","status":"MAINTAINED|VIOLATED|IRRELEVANT|UNCERTAIN",
"reason":"actual continuity or unexplained contradiction"}]}.
""",
    "conflict_handling": COMMON + """
Trace the principal tension and other consequential tensions through the complete
script, including incompatible relational needs and psychological contradictions.
Assess preparation, actual interaction, consequences and ending together.
SUPPORTED means the treatment is prepared and realized through specific actions,
responses and consequences, whether it resolves, escalates, pauses or stays open.
UNSUPPORTED means a concrete unprepared reversal, disposal or unexplained omission
of a consequential established tension. An unresolved ending needs a perceptible
consequence; deliberate ambiguity alone is insufficient. A resolved ending is
valid when the encounter prepares it. NOT_APPLICABLE means no substantial tension
is established. Give one whole-script classification without category weights.
Distinguish present-action tensions from memories, illustrative episodes and
background. A remembered family situation can prepare the present encounter
without requiring its own subsequent outcome. Do not require every secondary
episode to be closed; check its function in the actual scope of the complete draft.
Return {"classification":"SUPPORTED|UNSUPPORTED|NOT_APPLICABLE|UNCERTAIN",
"tensions":[{"description":"specific tension","unit_ids":["U0001"],
"development":"how treatment and consequence occur"}],"reason":"whole treatment and ending"}.
""",
}

HEADING = re.compile(r"^(?:\d+[.)]\s*)?(?:INT\.?|EXT\.?|INT\.?/EXT\.?|I/E\.?)\b", re.I)
TRANSITION = re.compile(r"^(?:FADE IN|FADE OUT|CUT TO(?: BLACK)?|DISSOLVE TO|END|THE END)\s*[:.]?$", re.I)
INLINE_CUE = re.compile(r"^([A-Z][A-Z0-9 .’'()/-]{0,58}):\s*(.*)$")
DIRECTIVE = re.compile(r"^(?:SUPER|TITLE(?: CARD)?|INTERTITLE|ON[ -]SCREEN|FLASHBACK|(?:A )?MEMORY|INSERT|CLOSE[ -]ON|(?:MON|TUES|WEDNES|THURS|FRI|SATUR|SUN)DAY)$")


def units(script):
    """Fix action paragraphs and complete dialogue turns before model inference."""
    script = script.replace("\r\n", "\n").replace("\r", "\n")
    script = re.sub(r"\n[ \t]+(?=\n)", "\n", script)
    blocks = [m.group().strip() for m in re.finditer(r"[^\n]+(?:\n[^\n]+)*", script)
              if m.group().strip()]
    first_heading = next((i for i, b in enumerate(blocks) if HEADING.match(b.splitlines()[0])), 0)
    result, scene, pending = [], 0, None
    def add(kind, text, speaker=None):
        result.append({"unit_id": f"U{len(result)+1:04d}", "scene_id": f"S{max(scene,1):03d}",
                       "kind": kind, "text": text, "speaker": speaker})
    def is_cue(text):
        return (len(text) <= 60 and bool(re.search(r"[A-Z]", text)) and text == text.upper()
                and not HEADING.match(text) and not TRANSITION.match(text)
                and ":" not in text and len(text.split()) <= 8)
    for block in blocks[first_heading:]:
        lines = block.splitlines()
        if HEADING.match(lines[0]):
            if pending:
                add("ACTION", pending); pending = None
            scene += 1
            add("HEADING", lines[0])
            if len(lines) > 1:
                block = "\n".join(lines[1:]).strip(); lines = block.splitlines()
            else:
                continue
        if TRANSITION.match(block):
            continue
        inline = INLINE_CUE.match(lines[0])
        if pending:
            add("DIALOGUE", pending + "\n" + block, pending); pending = None
        elif inline and not DIRECTIVE.match(inline.group(1)):
            speaker, first_line = inline.groups()
            content = "\n".join([first_line] + lines[1:]).strip()
            if content:
                add("DIALOGUE", speaker + "\n" + content, speaker)
            else:
                pending = speaker
        elif is_cue(lines[0]):
            if len(lines) > 1:
                add("DIALOGUE", block, lines[0])
            else:
                pending = block
        else:
            add("ACTION", block)
    if pending:
        add("ACTION", pending)
    return result


def payload(script, dimension):
    return {"rubric": RUBRICS[dimension], "SCRIPT": script, "UNITS": units(script),
            "script_id": "anonymous", "model": "anonymous"}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(dimension, labels, script):
    """Validate full eligible-unit coverage and calculate category rates."""
    expected = units(script)
    if dimension in {"narrative_efficiency", "character_consistency"}:
        kind = {"ACTION", "DIALOGUE"} if dimension == "narrative_efficiency" else {"DIALOGUE"}
        ids = [u["unit_id"] for u in expected if u["kind"] in kind]
        rows = labels["units" if dimension == "narrative_efficiency" else "dialogue_analysis"]
        found = [r["unit_id"] for r in rows]
        if len(found) != len(set(found)) or set(found) != set(ids):
            raise ValueError("Incomplete or duplicated fixed-unit coverage")
        c = Counter(r["classification"] for r in rows)
        if dimension == "narrative_efficiency":
            if set(c) - {"NONREDUNDANT", "REDUNDANT", "UNCERTAIN"}:
                raise ValueError("Invalid narrative classification")
            for r in rows:
                if r["classification"] == "REDUNDANT" and r["repeats_unit_id"] not in ids:
                    raise ValueError("Redundancy requires an actual earlier unit")
            return {"redundant_unit_rate": ratio(c["REDUNDANT"], c["REDUNDANT"]+c["NONREDUNDANT"]),
                    "eligible_units": len(ids), "uncertain_units": c["UNCERTAIN"]}
        if set(c) - {"IN_CHARACTER", "NEUTRAL", "OOC", "UNCERTAIN"}:
            raise ValueError("Invalid character classification")
        v = Counter(r["voice"] for r in rows)
        if set(v) - {"SPECIFIC", "GENERIC", "UNCERTAIN"}:
            raise ValueError("Invalid voice classification")
        return {"ooc_rate": ratio(c["OOC"], len(rows)-c["UNCERTAIN"]),
                "distinctive_dialogue_rate": ratio(v["SPECIFIC"], v["SPECIFIC"]+v["GENERIC"]),
                "eligible_dialogues": len(rows), "uncertain_voice_units": v["UNCERTAIN"]}
    if dimension == "emotional_depth":
        scene_ids = {u["scene_id"] for u in expected}
        rows = labels["scenes"]
        if len(rows) != len(scene_ids) or {r["scene_id"] for r in rows} != scene_ids:
            raise ValueError("Incomplete or duplicated scene coverage")
        c = Counter(r["classification"] for r in rows)
        if set(c) - {"SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "UNCERTAIN"}:
            raise ValueError("Invalid emotional treatment")
        return {"supported_emotional_treatment_rate": ratio(c["SUPPORTED"],c["SUPPORTED"]+c["UNSUPPORTED"]),
                "eligible_emotional_scenes": c["SUPPORTED"]+c["UNSUPPORTED"],
                "emotional_not_applicable": c["NOT_APPLICABLE"], "emotional_uncertain": c["UNCERTAIN"]}
    if dimension == "logic_consistency":
        categories = {"time", "location", "possession", "identity", "capability", "knowledge"}
        if set(labels["checked_categories"]) != categories:
            raise ValueError("Incomplete continuity category inspection")
        rows = labels["facts"]
        if any(r["fact_kind"] not in {"WORLD_FACT", "CLAIM", "BELIEF"}
               or r["status"] not in {"VIOLATED", "MAINTAINED", "IRRELEVANT", "UNCERTAIN"} for r in rows):
            raise ValueError("Invalid continuity classification")
        c = Counter(r["status"] for r in rows if r["fact_kind"] == "WORLD_FACT")
        return {"logic_break_rate": ratio(c["VIOLATED"], c["VIOLATED"]+c["MAINTAINED"]),
                "eligible_world_constraints": c["VIOLATED"]+c["MAINTAINED"]}
    state = labels["classification"]
    if state not in {"SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "UNCERTAIN"}:
        raise ValueError("Invalid conflict treatment")
    return {"supported_conflict_treatment_rate": {"SUPPORTED":1,"UNSUPPORTED":0}.get(state)}
