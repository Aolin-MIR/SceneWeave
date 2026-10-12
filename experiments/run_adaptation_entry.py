"""Execute four source-conditioned adaptation entries through one writer contract.

Preparation is offline. Runtime requires the existing actual evaluator gate.
The provisional selector has no observed route-utility training labels. Distinct
entry materials are process implementations, not established quality gains.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from adaptation_system_contract import fingerprint, validate_program
import adaptation_entry_selection as selection
import adaptation_entry_algorithms as algorithms
import scene_entry_mechanism_controls as scene_controls
import run_scene_first_matched as matched


VERSION = "adaptation-entry-execution-v11"
EXECUTION_MODES = ("native", "native_complete_v2", "native_complete_v3", "native_complete_v4", "prompt_materials_v6")
WRITER_PHASES = ("official", "develop", "explore", "revise_program", "realize", "review", "final")
SCENE_CONTROLS = ("freeze_prefix", *sorted(scene_controls.CONDITIONS))
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "runs/entries"
DEFAULT_MATERIAL_OUT = DEFAULT_OUT.parent / "native_entries_material_development_v4"
DRAMATURGE_PATH = Path(__file__).resolve().parents[1] / "third_party/dramaturge/appendix_prompts.json"
PRACTICE = {
    "sciamma-2019-lecture": {"url": "https://www.bafta.org/media-centre/press-releases/screenwriters-lecture-series-2019-celine-sciamma/", "extent": "Her personal distinction between scenes needed for continuity and scenes desired for expression; conditional scene reconsideration here, not a universal rule to delete transitions"},
    "gerwig-august-2020-scriptnotes": {"url": "https://johnaugust.com/2020/scriptnotes-ep-433-the-one-with-greta-gerwig", "extent": "Gerwig's account of how adult absence and preceding/following scenes recontextualize childhood warmth; conditional order comparison here, not a rule to use nonlinear structure"},
    "focus-2024-straughan": {"url": "https://www.focusfeatures.com/article/conclave_interview_peter_straughan", "extent": "Straughan's public ritual/private interaction and institutional/ordinary human juxtaposition; conditional cross-situation relationship reading here, not a required public/private pair or a rule against dialogue"},
    "birch-2024-lecture": {"url": "https://www.bafta.org/media-centre/press-releases/screenwriters-lecture-series-2024-alice-birch/", "extent": "Birch's form/content alignment and experimenting with biographies, timelines and simultaneous scenes; alternative concrete arrangements here, not a claim to reproduce her entire writing practice"},
}
PROGRAM_SCHEMA = "{intent:string,choices:{plot:string,character_actions:string,relationships:string,ending:string,background:string,props:string,scene_organization:string,expression:string},scenes:[{scene_id:string,development:string}]}"
DEVELOPMENT_CHECKS = (
    " Treat the supplied initial plan and draft as fallible writing, not as "
    "an authoritative account of the source. Correct mistaken participants, "
    "locations, actions and prior relationships while developing from its "
    "useful passages. Source limits describe what has been supplied; they do "
    "not prohibit new adaptation choices authorized by the brief. Distinguish "
    "a stated prior action from an unknown motive or reaction. Preserving "
    "uncertainty about a motive does not require hiding the action itself. "
    "Identify a concrete unresolved adaptation decision for the two treatments. "
    "Test different ways the scene or arrangement expresses the particular "
    "relationship and experience; stylistic variation alone may be insufficient. "
    "Neither restraint nor a larger change is a default preference."
)
DRAFT_CHECKS = (
    " Read the actual source and complete current script independently of "
    "earlier summaries and advice. Check where each person is, occupied and "
    "unoccupied places, departures and returns, possession of objects, and "
    "the sequence or framing of time. Separate contradictions from deliberate "
    "ellipsis. Check whether a viewer unfamiliar with the source can follow "
    "the particular relationship and the prior actions relevant to this "
    "encounter. If a known action was removed, assess the dramatic gain and "
    "loss rather than calling it unavailable. Preserving a fact, withholding "
    "it, changing it, or replacing its role can all serve the brief. Uncertainty "
    "about motives, unresolved endings and less explanatory speech earn no "
    "automatic advantage. Authorized invention can support a fuller scene. "
    "Check earlier development decisions against what the script actually "
    "realizes, including discoveries in the exploratory materials; a recorded "
    "decision or a claimed applied change is not proof it appears on screen."
)
DEVELOP = {
    "scene_first": matched.PLAN,
    "relationships_first": (
        "Read the source and brief and the available complete initial script. "
        "Use the supplied published Character Evaluator role as an analytical "
        "starting point. Return JSON {program:" + PROGRAM_SCHEMA + ",character_reading:"
        "{people:[{person:string,inner_contradiction:string}],relationship:string,"
        "movement:string},exploration:{focus:string,approaches:[string,string]}}. "
        "Identify particular people, their shared past, reciprocal perceptions "
        "and changes of relation. Propose two different linked encounter "
        "sequences around this relationship, not two abstract character profiles. "
        "The supplied published role is task-adapted: unexpected revelations, "
        "growth, boldness and greater conflict are optional, not objectives. "
        "This output contract overrides its original response format."
    ),
    "structure_first": (
        "Execute the supplied pinned official planning prompt for this source "
        "and brief. Return JSON {official_outline:string,program:" + PROGRAM_SCHEMA + ","
        "exploration:{focus:string,approaches:[string,string]}}. Retain the "
        "complete public outline in official_outline and organize its decisions "
        "into the structured program. Propose two different complete scene "
        "arrangements, including selection, emphasis, order and the beginning "
        "and ending. Standard act counts, escalation and closed endings are "
        "optional. This structured output contract replaces the upstream "
        "transport format, not its planning operation."
    ),
    "expression_first": (
        "Read what the source's narration, inner perspective, rhythm or imagery "
        "contributes under the requested dramatic medium. Return JSON {program:"
        + PROGRAM_SCHEMA + ",expression_reading:{source_feature:string,"
        "dramatic_opportunity:string},exploration:{focus:string,approaches:[string,string]}}. "
        "Choose one concrete source feature or passage and propose two ways "
        "to recreate its experience as actual dramatic expression. Make the "
        "two treatments comparable in focus. Voice, monologue, silence, action, "
        "space, sound, image and temporal cuts may all work; no mode is "
        "intrinsically superior. Keep whole-work choices provisional."
    ),
}
DEVELOP = {entry: prompt + DEVELOPMENT_CHECKS
           for entry, prompt in DEVELOP.items()}
EXPLORE = {
    "scene_first": matched.ACTUAL,
    "relationships_first": (
        "Develop the supplied approach into a concrete sequence of at least "
        "two linked encounters involving the focal relationship. Return JSON "
        "{text:string,encounters:[{encounter_id:string,participants:[string],"
        "situation:string,outward_response:string,other_response:string,"
        "relationship_before:string,action:string,relationship_after:string,"
        "motive_uncertainty:string}],cross_context_reading:{status:examined|not_applicable,"
        "person:string|null,encounter_ids:[string],observation:string,uncertainty:string}}. "
        "Each encounter needs particular circumstances and interaction; the "
        "later encounter must respond to what earlier interaction changed or "
        "left unresolved. Express ambivalence, shared habits and lived "
        "relations where relevant. Encounters may preserve a relation or "
        "leave interpretation open. Make the person's actual outward response "
        "and the other person's reply concrete in their situation. If relevant, "
        "compare how one person responds in different settings or to different "
        "people, retaining uncertainty about motives. If unsuitable, explain "
        "not_applicable instead of inventing such a contrast. Biographies "
        "do not prove fixed traits; no growth or public/private pairing is "
        "required. Do not merely list traits or goals."
    ),
    "structure_first": (
        "Develop the supplied approach into an entire adapted work arranged "
        "as ordered scenes. Return JSON {text:string,ordered_scenes:"
        "[{scene_id:string,development:string,continuity_role:string,expressive_reason:string}],"
        "arrangement_decisions:[{operation:retain|cut|merge|move|modify|add|invent|expand|compress,scene_ids:[string],"
        "intended_gain:string,possible_loss:string}],scene_relations:"
        "[{earlier_scene_id:string,later_scene_id:string,character_understanding:string,"
        "rhythm_effect:string,possible_loss:string}]}. State concrete selected, "
        "omitted, expanded or invented developments and their placement. "
        "Include the opening, development and ending of this adapted unit. "
        "Characters, plot and ending can change within the brief. Do not "
        "fully draft the dialogue or force a conventional act structure. "
        "For every scene distinguish what continuity requires from the "
        "particular experience its expression offers; a necessary quiet "
        "transition may be justified without a separate expressive flourish. "
        "Record concrete retained or changed arrangement decisions, considering "
        "cuts, merges, moves or modifications where useful and their possible "
        "losses. Examine at least one adjacency or earlier/later resonance "
        "for its effect on character understanding and rhythm. For a single "
        "scene, this may concern two moments within it. Linear order, bridges, "
        "unchanged scenes and unresolved interpretation are legitimate."
    ),
    "expression_first": (
        "Write an actual dramatic sample realizing the supplied approach to "
        "the selected source feature. Return JSON {text:string,mode:string,"
        "source_feature:string}. Write usable speech, action, voice, silence, "
        "space, sound or image as the target medium requires, with a natural "
        "local endpoint. A description of a possible sample is insufficient. "
        "The sample tests how the source experience changes in performance; "
        "do not score it or assume that less narration is better."
    ),
}
UPDATE = matched.UPDATE.replace(
    "Read the original source, provisional program, and BOTH explored treatments of the same scene.",
    "Read the original source, provisional program, and BOTH concrete exploratory materials. "
    "Their representation field explains whether they are actual scenes, encounter sequences, "
    "whole-work arrangements or actual expression samples. Compare their concrete gains, "
    "losses and implications for the whole; the entry name carries no quality meaning. "
    "The initial entry limits no later edit axis. Source narration order supplies context "
    "without requiring that screenplay order. Use the existing reason fields to explain "
    "intended gain and possible loss to relationships, psychological layers and lived meaning.") + DEVELOPMENT_CHECKS
REALIZE = matched.REALIZE + (
    " Interpret each exploration through its representation and actual content: "
    "actual_screenplay supplies a written scene; encounter_sequence supplies "
    "linked interactions and their relationship reading; whole_work_arrangement "
    "supplies selected and ordered developments rather than finished dialogue; "
    "expression_sample supplies an actual sample of medium-specific expression. "
    "These materials have different scope and do not each constitute a complete "
    "screenplay. Use their particular discoveries under the revised program; "
    "fully write the complete adaptation in the same requested medium."
) + DRAFT_CHECKS
REVIEW = matched.REVIEW + DRAFT_CHECKS + (
    " Put concrete continuity or audience-understanding concerns in "
    "revision_opportunities alongside aesthetic suggestions. Confirm every "
    "premise of your advice against the supplied text and current draft. "
    "Do not describe an expressly supplied action as unspecified. Do not "
    "recommend withholding merely to keep an interpretation open."
)
FINAL = matched.FINAL + DRAFT_CHECKS + (
    " Recheck the premises of each suggestion before applying it. Correct "
    "unintentional physical and temporal contradictions even if the advice "
    "missed them. Make each applied_changes item describe a change visible "
    "in this final script; preserve useful specificity when rejecting a "
    "conclusive explanation."
)
PROMPTS = {"selection": selection.ROUTE_INSTRUCTION, "develop": DEVELOP,
           "explore": EXPLORE, "update": UPDATE, "realize": REALIZE,
           "review": REVIEW, "final": FINAL, "official": matched.OFFICIAL}


def borrowed_methods():
    """Read verified local primary artifacts without importing upstream services."""
    official = matched.official_provenance()
    raw = DRAMATURGE_PATH.read_bytes()
    data = json.loads(raw)
    role = data["sections"]["A1.SS2"]
    if not isinstance(role, str) or "Character Evaluator" not in role:
        raise ValueError("The archived published Character Evaluator prompt is unavailable")
    return {
        "scene_first": {"extent": "Proposed actual focal-scene trials followed by whole-program reconsideration; no quality gain established"},
        "relationships_first": {"extent": "Exact archived Character Evaluator text supplied to a task-adapted character-reading stage; encounter-sequence generation is our implementation, not the full Dramaturge pipeline",
                                "source_url": data["source"], "artifact_path": str(DRAMATURGE_PATH),
                                "artifact_sha256": matched.sha256_bytes(raw), "section": "A1.SS2",
                                "role_sha256": fingerprint(role),
                                "practice": {k: PRACTICE[k] for k in ("focus-2024-straughan", "birch-2024-lecture")}},
        "structure_first": {"extent": "Pinned official SuperWriter plan function with a source-to-screenplay query and structured output wrapper; not Re3/DOC controller reproduction or official trained weights", **official,
                            "practice": {k: PRACTICE[k] for k in ("sciamma-2019-lecture", "gerwig-august-2020-scriptnotes", "birch-2024-lecture")}},
        "expression_first": {"extent": "Our implementation of actual target-medium expression samples; no borrowed complete strong-baseline route claimed", "practice": {"birch-2024-lecture": PRACTICE["birch-2024-lecture"]}},
    }


def frozen_code(*, scene_control=False):
    """Bind the shared writer and input contracts as well as this executor."""
    paths = (Path(__file__), Path(selection.__file__), Path(algorithms.__file__), Path(matched.__file__),
             Path(__file__).with_name("adaptation_system_contract.py"),
             Path(__file__).with_name("local_adaptation_writer_interface.py"))
    if scene_control:
        paths += (Path(scene_controls.__file__),)
    return {str(p.resolve()): matched.sha256_bytes(p.read_bytes()) for p in paths}


def stage_phase(stage):
    """Map distinct entry operations to shared configurable writer task roles."""
    if stage in {v[0] for v in selection.ENTRY_STAGE_NAMES.values()}:
        return "develop"
    if stage in {v[1] for v in selection.ENTRY_STAGE_NAMES.values()}:
        return "explore"
    if stage in WRITER_PHASES:
        return stage
    raise ValueError("Unknown writer execution phase")


def bind_stage_backends(backend, stage_backends=None):
    """Freeze explicit assignments; absent roles use the declared writer base."""
    matched.validate_backend(backend)
    requested = {} if stage_backends is None else stage_backends
    if not isinstance(requested, dict) or not set(requested) <= set(WRITER_PHASES):
        raise ValueError("Stage backend assignments must use the declared writer phases")
    result = {phase: deepcopy(requested.get(phase, backend)) for phase in WRITER_PHASES}
    for value in result.values():
        matched.validate_backend(value)
    return result


def check_stage_callbacks(backend, assignments, stage_generate):
    """Reject unavailable heterogeneous backends before any generation call."""
    callbacks = {} if stage_generate is None else stage_generate
    if not isinstance(callbacks, dict) or not set(callbacks) <= set(WRITER_PHASES):
        raise ValueError("Stage callbacks must use the declared writer phases")
    if any(not callable(callback) for callback in callbacks.values()):
        raise ValueError("Each registered stage backend requires an inference callback")
    for phase, value in assignments.items():
        if value != backend and phase not in callbacks:
            raise ValueError("An explicit callback is required for a different backend: " + phase)
    return callbacks


def backend_billing(records):
    """Keep real API and local receipts separate in mixed writer runs."""
    groups = {}
    for row in records:
        key = fingerprint(row["backend"])
        groups.setdefault(key, []).append(row)
    return {key: {"backend": rows[0]["backend"], "billing": matched.billing(rows)}
            for key, rows in groups.items()}


def check_development(entry, result):
    program = validate_program(result.get("program", {}))
    exploration = result.get("exploration", {})
    approaches = exploration.get("approaches")
    if (not isinstance(approaches, list) or len(approaches) != 2
            or any(not isinstance(a, str) or not a.strip() for a in approaches)):
        raise ValueError("Every entry must develop two concrete alternative treatments")
    if approaches[0].strip() == approaches[1].strip():
        raise ValueError("The two alternative treatments cannot be the same instruction")
    if entry == "scene_first":
        if exploration.get("scene_id") not in {s["scene_id"] for s in program["scenes"]}:
            raise ValueError("Focal scene must occur in the provisional program")
    else:
        matched.text(exploration, "focus")
        if entry == "relationships_first":
            reading = result.get("character_reading", {})
            matched.text(reading, "relationship")
            matched.text(reading, "movement")
            people = reading.get("people")
            if not isinstance(people, list) or not people:
                raise ValueError("Relationship development requires particular people")
            for person in people:
                matched.text(person, "person")
                matched.text(person, "inner_contradiction")
        elif entry == "structure_first":
            matched.text(result, "official_outline")
        else:
            reading = result.get("expression_reading", {})
            matched.text(reading, "source_feature")
            matched.text(reading, "dramatic_opportunity")
    return program, exploration


def bind_material(entry, result, trial_id, source, program, exploration):
    """Preserve distinct actual materials and their source/program identities."""
    matched.text(result, "text")
    if entry == "scene_first":
        if result.get("scene_id") != exploration["scene_id"]:
            raise ValueError("Actual scene trial changed its focal scene identifier")
    elif entry == "relationships_first":
        encounters = result.get("encounters")
        if not isinstance(encounters, list) or len(encounters) < 2:
            raise ValueError("Relationship entry requires linked encounters, not a profile")
        ids = []
        for row in encounters:
            for key in ("encounter_id", "situation", "outward_response", "other_response",
                        "relationship_before", "action", "relationship_after", "motive_uncertainty"):
                matched.text(row, key)
            if (not isinstance(row.get("participants"), list) or not row["participants"]
                    or any(not isinstance(v, str) or not v.strip() for v in row["participants"])):
                raise ValueError("Encounter participants must be explicit")
            ids.append(row["encounter_id"])
        if len(set(ids)) != len(ids):
            raise ValueError("Encounter identifiers must be unique")
        reading = result.get("cross_context_reading", {})
        if reading.get("status") not in {"examined", "not_applicable"}:
            raise ValueError("Record whether cross-situation comparison was applicable")
        matched.text(reading, "observation")
        matched.text(reading, "uncertainty")
        if not isinstance(reading.get("encounter_ids"), list) or not set(reading["encounter_ids"]) <= set(ids):
            raise ValueError("Cross-situation reading refers to unavailable encounters")
        if reading["status"] == "examined":
            matched.text(reading, "person")
            if len(set(reading["encounter_ids"])) < 2:
                raise ValueError("A claimed cross-situation comparison needs multiple encounters")
            if any(reading["person"] not in row["participants"] for row in encounters
                   if row["encounter_id"] in reading["encounter_ids"]):
                raise ValueError("Compared person must be present in the recorded encounters")
    elif entry == "structure_first":
        validate_program({**program, "scenes": result.get("ordered_scenes", [])})
        for row in result["ordered_scenes"]:
            matched.text(row, "continuity_role")
            matched.text(row, "expressive_reason")
        current_ids = {row["scene_id"] for row in result["ordered_scenes"]}
        possible_ids = current_ids | {row["scene_id"] for row in program["scenes"]}
        decisions = result.get("arrangement_decisions")
        if not isinstance(decisions, list) or not decisions:
            raise ValueError("Concrete arrangement decisions and possible losses are required")
        for row in decisions:
            if row.get("operation") not in {"retain", "cut", "merge", "move", "modify", "add", "invent", "expand", "compress"}:
                raise ValueError("Unknown arrangement operation")
            if not isinstance(row.get("scene_ids"), list) or not row["scene_ids"] or not set(row["scene_ids"]) <= possible_ids:
                raise ValueError("Arrangement decision refers to unavailable scenes")
            matched.text(row, "intended_gain")
            matched.text(row, "possible_loss")
        relations = result.get("scene_relations")
        if not isinstance(relations, list) or not relations:
            raise ValueError("At least one concrete scene relationship is required")
        for row in relations:
            if row.get("earlier_scene_id") not in current_ids or row.get("later_scene_id") not in current_ids:
                raise ValueError("Scene relationship refers to unavailable scenes")
            for field in ("character_understanding", "rhythm_effect", "possible_loss"):
                matched.text(row, field)
    elif entry == "expression_first":
        matched.text(result, "mode")
        matched.text(result, "source_feature")
    else:
        raise ValueError("Unknown material representation")
    return {**deepcopy(result), "trial_id": trial_id,
            "representation": selection.ENTRY_MATERIAL_KINDS[entry],
            "source_sha256": fingerprint(source), "program_sha256": fingerprint(program)}


def check_material_pair(materials, source, program):
    """Check material identity and literal duplicates, not artistic diversity."""
    if not isinstance(materials, list) or len(materials) != 2:
        raise ValueError("The common writer requires both complete explorations")
    if {m.get("trial_id") for m in materials} != {"0", "1"}:
        raise ValueError("Explorations require the two distinct recorded trial identifiers")
    if len({m.get("representation") for m in materials}) != 1:
        raise ValueError("The two explorations must use the same entry-specific material type")
    for material in materials:
        if (material.get("representation") not in selection.ENTRY_MATERIAL_KINDS.values()
                or material.get("source_sha256") != fingerprint(source)
                or material.get("program_sha256") != fingerprint(program)):
            raise ValueError("Exploration identity differs from its actual source or provisional program")
        matched.text(material, "text")
    if materials[0]["text"].strip() == materials[1]["text"].strip():
        raise ValueError("The two explorations returned identical material text")
    return {"identities_verified": True, "literal_duplicate": False,
            "semantic_difference_assessed": False,
            "representation": materials[0]["representation"],
            "text_sha256": [fingerprint(m["text"]) for m in materials]}


def validate_development_assessment(value, unit):
    """Permit an explicitly requested DEV reading without certifying human taste."""
    if (not isinstance(value, dict) or value.get("mode") != "codex_development_reading"
            or not isinstance(value.get("authorization"), str) or not value["authorization"].strip()
            or value.get("source_sha256") != fingerprint(unit["source"])
            or any(value.get(key) is not False for key in
                   ("judgments_are_human", "training_label_release", "paper_result_release", "test_accessed"))):
        raise ValueError("Development reading requires its explicit limited assessment contract")
    return deepcopy(value)


def validate_shared_initialization(value, source, backend):
    """Reuse one actual generated plan and draft across different entry branches."""
    if not isinstance(backend, dict):
        raise ValueError("Shared initialization requires an explicit authorized backend")
    matched.validate_backend(backend)
    if (not isinstance(value, dict)
            or value.get("origin") != "shared_generated_official_plan_write"
            or value.get("source_sha256") != fingerprint(source)
            or value.get("brief_sha256") != fingerprint(source["brief"])
            or value.get("backend") != backend):
        raise ValueError("Shared initialization belongs to a different source, brief or backend")
    for name in ("initial_plan", "initial_script"):
        text = matched.text(value, name)
        if value.get(name + "_sha256") != fingerprint(text):
            raise ValueError("The complete shared initialization text changed")
    return {name: value[name] for name in ("initial_plan", "initial_script")}


def _scene_control_parameters(unit, initialization, assignments, binding):
    return {"task_id": unit["id"], "work_id": unit["source"]["work_id"],
            "author_group": unit["author_group"], "split": unit["split"],
            "writer_stage_backends": deepcopy(assignments),
            "shared_initialization_sha256": fingerprint(initialization),
            "evaluator_validation": deepcopy(binding),
            "control_module_sha256": matched.sha256_bytes(Path(scene_controls.__file__).read_bytes())}


def validate_scene_control_prefix(value, unit, assignments, *, expected_sha256, evaluator_binding=None):
    """Validate frozen content against actual authorized five-argument receipts."""
    if (not isinstance(value, dict) or fingerprint(value) != expected_sha256
            or value.get("status") != "prefix_frozen" or value.get("scene_control") != "freeze_prefix"
            or value.get("scene_control_version") != scene_controls.VERSION
            or value.get("writer_workflow_version") != scene_controls.VERSION + ":freeze_prefix"
            or value.get("task_id") != unit["id"] or value.get("work_id") != unit["source"]["work_id"]
            or value.get("author_group") != unit["author_group"]
            or value.get("split") != unit["split"]
            or value.get("source_sha256") != fingerprint(unit["source"])
            or value.get("writer_stage_backends") != assignments
            or value.get("execution_mode") != "native_complete_v4"
            or value.get("requested_entry") != "scene_first" or value.get("selected_entry") != "scene_first"
            or value.get("selector_assessments_to_writer") is not False or value.get("selector_execution") != "none"
            or value.get("development_assessment") is not None
            or value.get("initialization_backend") != assignments["official"]
            or value.get("training_eligible") is not False):
        raise ValueError("Frozen scene prefix differs from its actual task, backend or artifact SHA")
    initialization = value.get("shared_initialization")
    from sceneweave_audience_selection import validate_common_initialization
    initial = validate_common_initialization(initialization, unit["source"], assignments["official"])
    if value.get("shared_initialization_sha256") != fingerprint(initialization):
        raise ValueError("Frozen scene prefix initialization changed")
    prior_binding = matched.validate_evaluator(value.get("evaluator_validation") or {})
    if evaluator_binding is not None and prior_binding != evaluator_binding:
        raise ValueError("Scene conditions must use the same frozen evaluator gate")
    bank = scene_controls.validate_frozen_scene_exploration(
        value["frozen_scene_prefix"], expected_bank_sha256=value["frozen_scene_prefix_bank_sha256"])
    expected_available = {"source": unit["source"], **initial, "development_entry": "scene_first"}
    if (bank["available"] != expected_available
            or bank["prompts"] != {**PROMPTS, "trial": matched.ACTUAL}
            or bank["parameters"] != _scene_control_parameters(unit, initialization, assignments, prior_binding)):
        raise ValueError("Frozen prefix source, initial writing, prompts or parameters changed")
    calls = value.get("calls", [])
    for row in calls:
        phase = row.get("phase")
        backend = assignments.get(phase)
        response = row.get("response", {})
        if (backend is None or backend["transport"] != "api" or row.get("backend") != backend
                or row.get("status") != "complete" or row.get("selector") is not False
                or row.get("request_sha256") != fingerprint({"stage": row.get("stage"), "system": row.get("system"),
                    "input": row.get("input"), "backend": backend})
                or row.get("output") != response.get("result") or response.get("status", "complete") != "complete"
                or any(item.get("served_model") != matched.MODEL or item.get("source_requested") != "wiki"
                       or item.get("finish_reason") != "stop" for item in (row, response))):
            raise ValueError("Frozen prefix requested/served backend receipt is invalid")
    official_rows = [row for row in calls if row.get("phase") == "official"]
    if len(official_rows) != (2 if value.get("shared_superwriter_start") is True else 0):
        raise ValueError("Frozen common initialization call inventory differs")
    for index, (stage, row) in enumerate(zip(("plan", "write"), official_rows)):
        if (row.get("request_identity") != unit["id"] + "__entry_v1__official_" + stage
                or row.get("stage") != "official" or row.get("input", {}).get("official_stage") != stage
                or row["input"].get("source") != unit["source"]
                or row.get("response") != initialization["receipts"][index]
                or (stage == "write" and row["input"].get("initial_plan") != initial["initial_plan"])):
            raise ValueError("Common initialization receipt differs from its actual task or response")
    rows = [row for row in calls if row.get("phase") != "official"]
    if len(rows) != len(bank["prefix_records"]):
        raise ValueError("Frozen prefix needs every actual proposal and trial receipt")
    for row, content in zip(rows, bank["prefix_records"]):
        request = content["request"]
        phase = request["phase"]
        backend = assignments[phase]
        matched.validate_backend(backend)
        stage = {"develop": "plan", "explore": "trial"}[phase]
        system = request["prompt"] + "\n" + matched.common.THEORY + "\n" + selection.COMMON_CONTRACT
        identity = unit["id"] + "__entry_v1__scene_first__scene_control__freeze_prefix__" + request["key"]
        expected_request = fingerprint({"stage": stage, "system": system,
                                        "input": request["input"], "backend": backend})
        response = row.get("response", {})
        if (backend["transport"] != "api" or row.get("status") != "complete" or row.get("selector") is not False
                or row.get("phase") != phase or row.get("stage") != stage or row.get("backend") != backend
                or row.get("request_identity") != identity or row.get("request_sha256") != expected_request
                or row.get("system") != system or row.get("input") != request["input"]
                or row.get("output") != content["output"] or response.get("result") != content["output"]
                or row.get("served_model") != matched.MODEL or response.get("served_model") != matched.MODEL
                or row.get("source_requested") != "wiki" or response.get("source_requested") != "wiki"
                or row.get("finish_reason") != "stop" or response.get("finish_reason") != "stop"
                or response.get("status", "complete") != "complete"):
            raise ValueError("Frozen scene prefix has an invalid real callback receipt")
    return deepcopy(value)


def load_frozen_scene_prefixes(path, units, assignments):
    """Freeze existing prefix files for the ordinary prepared/activated runner."""
    path = Path(path)
    artifacts, values = [], []
    if path.is_dir():
        directory = path / "results" if (path / "results").is_dir() else path
        for unit in units:
            file = directory / (unit["id"] + ".json")
            data = file.read_bytes()
            artifacts.append({"path": str(file.resolve()), "sha256": matched.sha256_bytes(data)})
            values.append(json.loads(data))
    else:
        data = path.read_bytes()
        artifacts.append({"path": str(path.resolve()), "sha256": matched.sha256_bytes(data)})
        values = json.loads(data).get("prefixes")
    if (not isinstance(values, list) or len(values) != len(units)
            or any(not isinstance(v, dict) for v in values)
            or {v.get("task_id") for v in values} != {u["id"] for u in units}):
        raise ValueError("Exactly one real frozen scene prefix per source task is required")
    indexed = {value["task_id"]: value for value in values}
    for unit in units:
        value = indexed[unit["id"]]
        validate_scene_control_prefix(value, unit, assignments, expected_sha256=fingerprint(value))
    return {"records": indexed, "artifacts": artifacts, "sha256": fingerprint(indexed)}


def bind_frozen_selection(value, unit, backend):
    """Validate an archived selector response without performing inference."""
    source = unit["source"]
    if (not isinstance(value, dict) or value.get("status") != "complete"
            or value.get("task_id") != unit["id"]
            or value.get("work_id") != source["work_id"]
            or value.get("source_sha256") != fingerprint(source)):
        raise ValueError("Frozen selection belongs to a different task or source")
    response = value.get("response", {})
    if (not isinstance(response, dict)
            or response.get("status", "complete") != "complete"
            or response.get("finish_reason") != "stop"
            or not isinstance(response.get("result"), dict)):
        raise ValueError("Frozen selection requires a complete actual selector response")
    matched.validate_backend(backend)
    if backend["transport"] == "api":
        valid_identity = (response.get("served_model") == matched.MODEL
                          and response.get("source_requested") == "wiki")
    else:
        valid_identity = (response.get("served_revision") == backend["revision"]
                          and response.get("source_requested") == "local")
    if not valid_identity:
        raise ValueError("Frozen selector identity differs from its configured backend")
    route = selection.bind_route(response["result"], source)
    archived = value.get("route", {})
    # Keep the archived prompt version as provenance, not a current-prompt claim.
    for key in ("selected_entry", "decision_basis", "decision_uncertainty", "assessments",
                "viable_alternatives", "source_sha256", "brief_sha256", "selection_origin"):
        if not isinstance(archived, dict) or archived.get(key) != route[key]:
            raise ValueError("Frozen route differs from the actual response or source brief")
    if not isinstance(archived.get("version"), str) or not archived["version"].strip():
        raise ValueError("Frozen selector prompt version must remain explicit")
    route["version"] = archived["version"]
    return route


def load_frozen_selections(path, units, backend):
    """Freeze existing selector receipts, including their exact file hashes."""
    path = Path(path)
    artifacts = []
    if path.is_dir():
        directory = path / "selection_results" if (path / "selection_results").is_dir() else path
        values = []
        for unit in units:
            file = directory / (unit["id"] + ".json")
            raw = file.read_bytes()
            artifacts.append({"path": str(file.resolve()), "sha256": matched.sha256_bytes(raw)})
            values.append(json.loads(raw))
    else:
        raw = path.read_bytes()
        artifacts.append({"path": str(path.resolve()), "sha256": matched.sha256_bytes(raw)})
        document = json.loads(raw)
        values = document.get("selections") if isinstance(document, dict) else None
        if not isinstance(values, list):
            raise ValueError("Frozen selector JSON requires a selections array of complete receipts")
    if (len(values) != len(units) or any(not isinstance(v, dict) for v in values)
            or len({v.get("task_id") for v in values}) != len(values)):
        raise ValueError("Exactly one complete frozen selection per source task is required")
    indexed = {v["task_id"]: v for v in values}
    if set(indexed) != {u["id"] for u in units}:
        raise ValueError("Frozen selections and source tasks differ")
    for unit in units:
        bind_frozen_selection(indexed[unit["id"]], unit, backend)
    return {"records": deepcopy(indexed), "artifacts": artifacts,
            "sha256": fingerprint(indexed),
            "selector_assessments_to_writer": False,
            "writer_excludes": ["selected_entry", "decision_basis", "decision_uncertainty", "viable_alternatives", "other_entry_assessments"]}


def verify_complete_entry_receipts(candidate, entry):
    """Replay recorded controller decisions without inference or creating labels."""
    calls = candidate.get("calls", [])
    if not calls or any(c.get("selector") or c.get("status") != "complete" for c in calls):
        raise ValueError("Complete entry reuse needs complete unguided native writing receipts")
    available = deepcopy(calls[0]["input"])
    initialization = candidate.get("shared_initialization")
    if not isinstance(initialization, dict) or fingerprint(initialization) != candidate.get("shared_initialization_sha256"):
        raise ValueError("Native candidate needs its actual complete shared initialization binding")
    initial_texts = validate_shared_initialization(initialization, available["source"], candidate.get("initialization_backend"))
    if any(available.get(key) != value for key, value in initial_texts.items()):
        raise ValueError("Native writing input differs from its bound common initialization")
    position = 0

    def recorded(key, phase, prompt, payload):
        nonlocal position
        if position >= len(calls):
            raise ValueError("Native entry receipt sequence is incomplete")
        row = calls[position]
        position += 1
        stage = {"develop": "plan", "explore": "trial", "realize": "realize", "final": "revise_script"}.get(phase, phase)
        system = prompt + "\n" + matched.common.THEORY + "\n" + selection.COMMON_CONTRACT
        if (row.get("phase") != phase or row.get("stage") != stage
                or row.get("request_identity") != candidate["task_id"] + "__entry_v1__" + entry + "__native__" + key
                or row.get("system") != system or row.get("input") != payload):
            raise ValueError("Native writing receipt differs from its actual controller input")
        actual_backend = row.get("backend")
        if actual_backend != candidate.get("writer_stage_backends", {}).get(phase):
            raise ValueError("Native writing receipt differs from its assigned phase backend")
        matched.validate_backend(actual_backend)
        valid_identity = (row.get("served_model") == matched.MODEL and row.get("source_requested") == "wiki"
                          if actual_backend["transport"] == "api" else
                          row.get("served_revision") == actual_backend["revision"] and row.get("source_requested") == "local")
        if not valid_identity or row.get("finish_reason") != "stop":
            raise ValueError("Native writing receipt has an invalid served identity or completion")
        projection = row.get("local_input_projection")
        if projection is not None:
            from local_adaptation_writer_interface import VERSION as projection_version, project_request
            if actual_backend["transport"] != "local":
                raise ValueError("API receipts cannot claim local input projection")
            projected_system, projected_payload = project_request(stage, system, payload)
            expected_projection = {"version": projection_version, "stage": stage,
                "original_request_sha256": fingerprint({"system": system, "input": payload}),
                "projected_request_sha256": fingerprint({"system": projected_system, "input": projected_payload}),
                "projected_input_fields": sorted(projected_payload), "creates_training_labels": False}
            if projection != expected_projection:
                raise ValueError("Local receipt differs from its actual deterministic input projection")
        return deepcopy(row.get("output"))

    executor = (algorithms.execute_entry_material_complete if candidate.get("execution_mode") == "native_complete_v4"
                else algorithms.execute_entry_complete)
    replayed = executor(entry, available, recorded, {**PROMPTS, "trial": matched.ACTUAL})
    if position != len(calls):
        raise ValueError("Unexpected writing receipts after actual final inspection")
    if any(candidate.get(key) != value for key, value in replayed.items()):
        raise ValueError("Complete entry candidate differs from its recorded final writing decisions")
    return {"controller_replayed_without_inference": True, "calls": position, "quality_assessed": False}


def reuse_selected_candidate(candidate_bank, frozen_selection, unit, selector_backend):
    """Select an already written candidate while preserving its text and context."""
    unit = matched.validate_unit(unit)
    route = bind_frozen_selection(frozen_selection, unit, selector_backend)
    if not isinstance(candidate_bank, dict) or set(candidate_bank) != set(selection.ENTRIES):
        raise ValueError("Selection reuse requires the complete four-entry candidate bank")
    common = None
    for entry, candidate in candidate_bank.items():
        if (not isinstance(candidate, dict) or candidate.get("status") != "complete"
                or candidate.get("task_id") != unit["id"]
                or candidate.get("work_id") != unit["source"]["work_id"]
                or candidate.get("author_group") != unit["author_group"]
                or candidate.get("source_sha256") != fingerprint(unit["source"])
                or candidate.get("requested_entry") != entry or candidate.get("selected_entry") != entry
                or candidate.get("selector_assessments_to_writer") is not False
                or "entry_guidance" in candidate
                or candidate.get("selector_execution") not in {"none", "frozen"}
                or not candidate.get("shared_initialization_sha256")):
            raise ValueError("Pure selection requires unguided candidates under the same shared initialization; historical guided banks are excluded")
        expected_selection = fingerprint(frozen_selection) if candidate["selector_execution"] == "frozen" else None
        if candidate.get("frozen_selection_sha256") != expected_selection:
            raise ValueError("Candidate bank belongs to a different frozen selector context")
        matched.text(candidate, "script")
        calls = candidate.get("calls", [])
        expected_stages = [selection.ENTRY_STAGE_NAMES[entry][0]] + [selection.ENTRY_STAGE_NAMES[entry][1]] * 2 + ["revise_program", "realize", "review", "final"]
        if candidate.get("execution_mode") in {"native_complete_v3", "native_complete_v4"}:
            expected_version = (algorithms.MATERIAL_ENTRY_VERSION if candidate["execution_mode"] == "native_complete_v4"
                                else algorithms.COMPLETE_ENTRY_VERSION)
            if candidate.get("writer_workflow_version") != expected_version:
                raise ValueError("Complete native workflow version differs from the current controller")
            verify_complete_entry_receipts(candidate, entry)
        else:
            if (len(calls) != 7 or [c.get("stage") for c in calls] != expected_stages
                    or any(c.get("selector") or c.get("status") != "complete" for c in calls)):
                raise ValueError("Candidate reuse requires seven complete writing calls after the shared initialization")
            if calls[-1]["output"].get("script") != candidate["script"]:
                raise ValueError("Candidate script differs from its actual writing receipts")
        initialization = {key: calls[0]["input"].get(key) for key in ("source", "initial_plan", "initial_script")}
        if not all(initialization.get(key) for key in ("initial_plan", "initial_script")):
            raise ValueError("Candidate bank must retain the complete common initial plan and script")
        for call in calls:
            if "entry_guidance" in call["input"] or "entry_guidance" in call["system"]:
                raise ValueError("Pure selection writing requests must exclude historical selector guidance")
            if any(call["input"].get(key) != value for key, value in initialization.items()):
                raise ValueError("A candidate's source or shared initialization changed during development")
            if call["input"].get("source") != unit["source"]:
                raise ValueError("Candidate writing source differs from the actual source task")
            request = {"stage": call["stage"], "system": call["system"],
                       "input": call["input"], "backend": call["backend"]}
            if fingerprint(request) != call.get("request_sha256"):
                raise ValueError("A candidate writing request changed after generation")
        conditions = {"initialization": initialization,
                      "shared_initialization_sha256": candidate["shared_initialization_sha256"],
                      "writer_stage_backends": candidate["writer_stage_backends"],
                      "execution_mode": candidate.get("execution_mode"),
                      "writer_workflow_version": candidate.get("writer_workflow_version"),
                      "selector_context": (candidate["selector_execution"], candidate.get("frozen_selection_sha256")),
                      "evaluator_validation": candidate["evaluator_validation"],
                      "development_assessment": candidate["development_assessment"]}
        if common is not None and conditions != common:
            raise ValueError("Four-entry candidates were produced under different writing conditions")
        common = conditions
    selected = candidate_bank[route["selected_entry"]]
    result = deepcopy(selected)
    result.update(requested_entry="auto", route=route,
                  selector_execution="frozen", frozen_selection_sha256=fingerprint(frozen_selection),
                  frozen_selector_version=route["version"],
                  execution_origin="reused_frozen_selected_candidate",
                  selection_reuse={"selected_candidate_sha256": fingerprint(selected),
                                   "candidate_bank_sha256": fingerprint(candidate_bank),
                                   "script_sha256": fingerprint(selected["script"]),
                                   "development_context_sha256": fingerprint(selected["development_context"]),
                                   "additional_calls": 0, "additional_billing": 0.0})
    return result


def execute_case(unit, generate, *, evaluator_validation=None, entry="auto", backend=None,
                  selector_generate=None, selector_backend=None, official=None,
                  shared_superwriter_start=True, stage_backends=None, stage_generate=None,
                  shared_initialization=None, development_assessment=None, frozen_selection=None,
                  execution_mode="native", scene_control=None, frozen_scene_prefix=None,
                  frozen_scene_prefix_sha256=None):
    """Run one entry with the same five-argument callback as the matched runner."""
    unit = matched.validate_unit(unit)
    if execution_mode not in EXECUTION_MODES:
        raise ValueError("Unknown entry execution mode")
    if scene_control is not None:
        if scene_control not in SCENE_CONTROLS or execution_mode != "native_complete_v4" or entry != "scene_first":
            raise ValueError("Scene controls require their explicit current scene-first execution mode")
        if development_assessment is not None or frozen_selection is not None:
            raise ValueError("Scene controls require the evaluator gate and exclude selector/development bypasses")
        if scene_control == "freeze_prefix":
            if frozen_scene_prefix is not None or frozen_scene_prefix_sha256 is not None:
                raise ValueError("Prefix freezing cannot consume another control prefix")
        else:
            if frozen_scene_prefix is None or not frozen_scene_prefix_sha256 or shared_superwriter_start:
                raise ValueError("Scene controls require an exact frozen prefix and reused initialization")
            if shared_initialization is None:
                shared_initialization = deepcopy(frozen_scene_prefix.get("shared_initialization"))
    elif frozen_scene_prefix is not None or frozen_scene_prefix_sha256 is not None:
        raise ValueError("Frozen scene prefixes require an explicit scene control")
    backend = deepcopy(matched.BACKEND if backend is None else backend)
    selector_backend = deepcopy(matched.BACKEND if selector_backend is None else selector_backend)
    matched.validate_backend(backend)
    assignments = bind_stage_backends(backend, stage_backends)
    if scene_control is not None and (backend["transport"] != "api"
            or any(b["transport"] != "api" for b in assignments.values()) or assignments["official"] != backend):
        raise ValueError("Matched scene controls require authorized Wiki Luna API roles and a common initialization backend")
    callbacks = check_stage_callbacks(backend, assignments, stage_generate)
    for callback in callbacks.values():
        modes = getattr(callback, "supported_execution_modes", None)
        if modes is not None and execution_mode not in modes:
            raise ValueError("Local creative adapter requires an explicitly supported complete-script execution mode")
    if entry not in ("auto", *selection.ENTRIES):
        raise ValueError("Unknown requested entry")
    frozen_route = None
    if frozen_selection is not None:
        frozen_route = bind_frozen_selection(frozen_selection, unit, selector_backend)
    if entry == "auto" and frozen_route is None:
        matched.validate_backend(selector_backend)
        if selector_generate is None and selector_backend != backend:
            raise ValueError("A separate selector backend requires its explicit callback")
    if (not shared_superwriter_start and shared_initialization is None
            and entry in {"auto", "relationships_first"}
            and not unit.get("current_script")):
        raise ValueError("The available character-reading route requires a complete generated initial script")
    if execution_mode in {"native_complete_v3", "native_complete_v4"} and not shared_superwriter_start and shared_initialization is None:
        raise ValueError("Complete entry execution requires the actual common plan and draft before routing or writing")
    if development_assessment is None:
        binding = matched.validate_evaluator(evaluator_validation or {})
        reading_binding = None
    else:
        if evaluator_validation is not None:
            raise ValueError("Development assessment and validated evaluator modes must stay separate")
        reading_binding = validate_development_assessment(development_assessment, unit)
        binding = None
    if scene_control not in {None, "freeze_prefix"}:
        validate_scene_control_prefix(frozen_scene_prefix, unit, assignments,
            expected_sha256=frozen_scene_prefix_sha256, evaluator_binding=binding)
        if shared_initialization != frozen_scene_prefix["shared_initialization"]:
            raise ValueError("Scene conditions must preserve the prefix's exact common initialization")
    functions = matched.pinned_official_functions() if official is None else official
    source = unit["source"]
    available = {"source": deepcopy(source)}
    if "current_script" in unit:
        available["current_script"] = unit["current_script"]
    if shared_initialization is not None:
        if shared_superwriter_start:
            raise ValueError("Choose reused initialization or newly generated initialization")
        initial = validate_shared_initialization(shared_initialization, source, backend)
        available.pop("current_script", None)
        available.update(initial)
        if scene_control is not None:
            from sceneweave_audience_selection import validate_common_initialization
            validate_common_initialization(shared_initialization, source, backend)
    records, trace = [], []
    result = {"task_id": unit["id"], "work_id": source["work_id"], "author_group": unit["author_group"],
              "requested_entry": entry, "status": "failed", "source_sha256": fingerprint(source),
              "quality_gain_established": False, "route_preference_validated": False,
              "route_outcome_labels": 0, "evaluator_validation": binding,
              "development_assessment": reading_binding,
              "shared_initialization_sha256": fingerprint(shared_initialization) if shared_initialization else None,
              "frozen_selection_sha256": fingerprint(frozen_selection) if frozen_selection is not None else None,
              "frozen_selector_version": frozen_route["version"] if frozen_route is not None else None,
              "selector_execution": "frozen" if frozen_route is not None else "live" if entry == "auto" else "none",
              "selector_assessments_to_writer": False,
              "writer_stage_backends": assignments, "selector_backend": selector_backend,
              "execution_mode": execution_mode,
              "writer_workflow_version": algorithms.MATERIAL_ENTRY_VERSION if execution_mode == "native_complete_v4"
                  else algorithms.COMPLETE_ENTRY_VERSION if execution_mode == "native_complete_v3"
                  else algorithms.COMPLETE_SCENE_VERSION if execution_mode == "native_complete_v2"
                  else algorithms.VERSION if execution_mode == "native" else "prompt-materials-v6"}
    if execution_mode in {"native_complete_v3", "native_complete_v4"}:
        result["shared_initialization"] = deepcopy(shared_initialization)
        result["initialization_backend"] = deepcopy(backend)
    if scene_control is not None:
        result.update(scene_control=scene_control, scene_control_version=scene_controls.VERSION,
                      split=unit["split"],
                      training_eligible=False, judgments_are_human=False,
                      frozen_scene_prefix_artifact_sha256=frozen_scene_prefix_sha256,
                      shared_prefix_billing_not_recharged=True)

    def call(key, stage, prompt, payload, *, selector=False, phase_override=None):
        phase = "selector" if selector else phase_override or stage_phase(stage)
        if not selector and phase not in WRITER_PHASES:
            raise ValueError("Unknown explicit writer task phase")
        actual_backend = selector_backend if selector else assignments[phase]
        callback = (selector_generate or generate) if selector else callbacks.get(phase, generate)
        identity = unit["id"] + "__entry_v1__" + key
        system = prompt + "\n" + (selection.COMMON_CONTRACT if selector else matched.common.THEORY + "\n" + selection.COMMON_CONTRACT)
        request_sha = fingerprint({"stage": stage, "system": system, "input": payload, "backend": actual_backend})
        record = {"request_identity": identity, "stage": stage, "phase": phase, "selector": selector,
                  "system": system, "input": deepcopy(payload), "request_sha256": request_sha,
                  "backend": deepcopy(actual_backend), "status": "failed"}
        records.append(record)
        trace.append({"request_identity": identity, "stage": stage, "selector": selector,
                      "request_sha256": request_sha})
        try:
            response = callback(identity, stage, system, deepcopy(payload), deepcopy(actual_backend))
            if scene_control is not None:
                record["response"] = deepcopy(response)
            record.update({k: deepcopy(response.get(k)) for k in
                           ("usage", "served_model", "served_revision", "source_requested", "raw_path", "finish_reason",
                            "local_input_projection")})
            if actual_backend["transport"] == "api":
                if response.get("served_model") != matched.MODEL or response.get("source_requested") != "wiki":
                    raise ValueError("Actual served API identity differs from authorized Wiki Luna")
            elif (response.get("served_revision") != actual_backend["revision"]
                  or response.get("source_requested") != "local"):
                raise ValueError("Actual local revision or local-only transport differs")
            if (response.get("status", "complete") != "complete" or not isinstance(response.get("result"), dict)
                    or response.get("finish_reason") != "stop"):
                raise ValueError("Incomplete stage; preserve original response without retry")
            record.update(status="complete", output=deepcopy(response["result"]))
            return deepcopy(response["result"])
        except Exception as exc:
            record.update(error_type=type(exc).__name__, error=str(exc)[:220] if isinstance(exc, (ValueError, RuntimeError)) else "See archived original response")
            raise

    try:
        if entry == "auto":
            if frozen_route is None:
                payload = call("selection", "select_entry", selection.ROUTE_INSTRUCTION,
                               selection.route_input(source), selector=True)
                route = selection.bind_route(payload, source)
            else:
                route = deepcopy(frozen_route)
            entry = route["selected_entry"]
        else:
            route = {"selected_entry": entry, "selection_origin": "author_fixed_entry",
                     "source_sha256": fingerprint(source), "brief_sha256": fingerprint(source["brief"]),
                     "route_outcome_labels": 0, "preference_validated": False}
        result.update(selected_entry=entry, route=route)
        if shared_superwriter_start:
            query = json.dumps({**available, "dramaturgy": matched.common.THEORY}, ensure_ascii=False)
            prompt = functions["generate_prompt"]("plan", use_query=query, previous_output=None)
            value = call("official_plan", "official", matched.OFFICIAL,
                         {**available, "official_stage": "plan", "official_prompt": prompt})
            initial_plan = matched.text(value, "answer")
            prompt = functions["generate_prompt"]("write", use_query=query,
                                                   previous_output="<answer>" + initial_plan + "</answer>")
            value = call("official_write", "official", matched.OFFICIAL,
                         {**available, "initial_plan": initial_plan, "official_stage": "write", "official_prompt": prompt})
            available.pop("current_script", None)
            available.update(initial_plan=initial_plan, initial_script=matched.text(value, "answer"))
            if scene_control is not None:
                initialization = {"origin": "shared_generated_official_plan_write", "status": "complete",
                    "backend": deepcopy(backend), "source_sha256": fingerprint(source),
                    "brief_sha256": fingerprint(source["brief"]),
                    "initial_plan": available["initial_plan"], "initial_script": available["initial_script"],
                    "receipts": [deepcopy(r["response"]) for r in records if r["phase"] == "official"]}
                for name in ("initial_plan", "initial_script"):
                    initialization[name + "_sha256"] = fingerprint(initialization[name])
                result["shared_initialization"] = initialization
                result["shared_initialization_sha256"] = fingerprint(initialization)
        if scene_control is not None:
            native_available = {**deepcopy(available), "development_entry": "scene_first"}
            control_prompts = {**PROMPTS, "trial": matched.ACTUAL}
            def control_call(key, phase, prompt, payload):
                stage = {"develop": "plan", "explore": "trial", "realize": "realize", "final": "revise_script"}.get(phase, phase)
                return call("scene_first__scene_control__" + key, stage, prompt, payload, phase_override=phase)
            if scene_control == "freeze_prefix":
                def prefix_call(key, phase, prompt, payload):
                    return control_call("freeze_prefix__" + key, phase, prompt, payload)
                bank = scene_controls.freeze_scene_exploration(native_available, control_prompts, prefix_call,
                    parameters=_scene_control_parameters(unit, result["shared_initialization"], assignments, binding))
                result.update(status="prefix_frozen", frozen_scene_prefix=bank,
                              frozen_scene_prefix_bank_sha256=bank["bank_sha256"],
                              writer_workflow_version=scene_controls.VERSION + ":freeze_prefix")
            else:
                bank = frozen_scene_prefix["frozen_scene_prefix"]
                controlled = scene_controls.run_scene_mechanism_control(bank, scene_control, control_call,
                    expected_bank_sha256=frozen_scene_prefix["frozen_scene_prefix_bank_sha256"],
                    request_namespace="matched_scene_control")
                result.update(controlled.pop("native_result"))
                result["scene_control_details"] = controlled
                result["reused_prefix_calls"] = len(bank["prefix_records"])
                result["reused_prefix_receipts_sha256"] = fingerprint(frozen_scene_prefix["calls"])
            result["algorithm_provenance"] = deepcopy(algorithms.PROVENANCE["scene_first"])
        elif execution_mode in {"native", "native_complete_v2", "native_complete_v3", "native_complete_v4"}:
            native_available = deepcopy(available)
            if execution_mode in {"native_complete_v3", "native_complete_v4"}:
                native_available["development_entry"] = entry
            if entry == "relationships_first":
                data = json.loads(DRAMATURGE_PATH.read_text())
                native_available["published_character_evaluator_role"] = data["sections"]["A1.SS2"]
            elif entry == "structure_first":
                native_available["official_prompt"] = functions["generate_prompt"](
                    "plan", use_query=json.dumps(available, ensure_ascii=False), previous_output=None)

            def native_call(key, phase, prompt, payload):
                stable_stage = {"develop": "plan", "explore": "trial", "realize": "realize",
                                "final": "revise_script"}.get(phase, phase)
                stage = stable_stage if execution_mode in {"native_complete_v2", "native_complete_v3", "native_complete_v4"} else key
                return call(entry + "__native__" + key, stage, prompt, payload, phase_override=phase)

            if execution_mode == "native_complete_v4":
                native = algorithms.execute_entry_material_complete(entry, native_available, native_call, {**PROMPTS, "trial": matched.ACTUAL})
            elif execution_mode == "native_complete_v3":
                native = algorithms.execute_entry_complete(entry, native_available, native_call, {**PROMPTS, "trial": matched.ACTUAL})
            elif execution_mode == "native_complete_v2":
                if entry != "scene_first":
                    raise ValueError("The revised complete-script workflow currently applies to scene-first only")
                native = algorithms.execute_scene_complete(native_available, native_call, {**PROMPTS, "trial": matched.ACTUAL})
            else:
                native = algorithms.execute_entry(entry, native_available, native_call, PROMPTS)
            result.update(native)
            result["algorithm_provenance"] = deepcopy(algorithms.PROVENANCE[entry])
        else:
            initial_stage, material_stage = selection.ENTRY_STAGE_NAMES[entry]
            initial_payload = deepcopy(available)
            if entry == "relationships_first":
                if not available.get("initial_script") and not available.get("current_script"):
                    raise ValueError("Published character evaluator requires a complete initial script")
                data = json.loads(DRAMATURGE_PATH.read_text())
                initial_payload["published_character_evaluator_role"] = data["sections"]["A1.SS2"]
            elif entry == "structure_first":
                initial_payload["official_prompt"] = functions["generate_prompt"](
                    "plan", use_query=json.dumps(available, ensure_ascii=False), previous_output=None)
            proposal = call(entry + "__development", initial_stage, DEVELOP[entry], initial_payload)
            program, exploration = check_development(entry, proposal)
            result["proposal"] = proposal
            materials = []
            for i, approach in enumerate(exploration["approaches"]):
                focus = {k: deepcopy(v) for k, v in exploration.items() if k != "approaches"}
                focus["approach"] = approach
                material = call(entry + "__material_" + str(i), material_stage, EXPLORE[entry],
                                {**available, "program": program, "exploration": focus, "trial_id": str(i)})
                materials.append(bind_material(entry, material, str(i), source, program, exploration))
                result["explorations"] = deepcopy(materials)
            result["material_pair_check"] = check_material_pair(materials, source, program)
            revision = call(entry + "__program_update", "revise_program", UPDATE,
                            {**available, "program": program, "exploration": exploration, "explorations": materials})
            matched.check_revision(revision, program, materials)
            result.update(revision=revision, dependency_changes=revision["dependency_changes"])
            writer_payload = {**available, "program": revision["program"], "original_program": program,
                              "explorations": materials, "selected_trial_id": revision.get("selected_trial_id"),
                              "dependency_changes": revision["dependency_changes"]}
            first = call(entry + "__realize", "realize", REALIZE, writer_payload)
            draft = matched.text(first, "script")
            result["before_whole_integration"] = draft
            history = {"program": revision["program"], "selected_trial_id": revision.get("selected_trial_id"),
                       "dependency_changes": revision["dependency_changes"],
                       "explorations": [{"trial_id": m["trial_id"], "representation": m["representation"],
                                         "text": m["text"],
                                         "text_sha256": fingerprint(m["text"]),
                                         "development_records": {k: deepcopy(v) for k, v in m.items()
                                                                 if k not in {"text", "source_sha256", "program_sha256", "representation", "trial_id"}}}
                                        for m in materials],
                       "trial_texts_available_in_this_reading": True,
                       "adaptation_decision_contract": selection.COMMON_CONTRACT,
                       "instruction": "Development decisions are fallible; assess them against this complete draft."}
            reading = call(entry + "__whole_read", "review", REVIEW,
                           {**available, "script": draft, "development_context": history})
            final = call(entry + "__whole_write", "final", FINAL,
                         {**available, "script": draft, "whole_reading": reading, "development_context": history})
            result.update(status="complete", script=matched.text(final, "script"),
                          whole_reading=reading, final_revision=final, development_context=history)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc)[:220] if isinstance(exc, (ValueError, RuntimeError)) else "See original preserved stage")
    result.update(calls=records, trace=trace, actual_unique_billing=matched.billing(records),
                  billing_by_backend=backend_billing(records),
                  selector_billing=matched.billing([r for r in records if r["selector"]]),
                  writer_billing=matched.billing([r for r in records if not r["selector"]]),
                  planned_writer_calls=(9 if shared_superwriter_start else 7) if execution_mode == "prompt_materials_v6" else None,
                  actual_writer_calls=len([r for r in records if not r["selector"]]),
                  planned_selector_calls=int(result["requested_entry"] == "auto" and frozen_route is None),
                  shared_superwriter_start=shared_superwriter_start,
                  call_counts_are_not_equal_compute=True)
    return result


def prepare_selection_development(source_path, out, *, authorized_calls):
    """Freeze a bounded source-only selector check, with no script generation."""
    source_path, out = Path(source_path), Path(out)
    value = json.loads(source_path.read_text())
    if (not isinstance(value, dict) or not isinstance(value.get("units"), list)
            or not value["units"]):
        raise ValueError("Reviewed DEV source units are required for routing development")
    units = [matched.validate_unit(u) for u in value["units"]]
    if (len({u["id"] for u in units}) != len(units)
            or len({u["source"]["work_id"] for u in units}) != len(units)):
        raise ValueError("The small selection check uses one request per DEV work")
    if (isinstance(authorized_calls, bool) or not isinstance(authorized_calls, int)
            or authorized_calls < len(units)):
        raise ValueError("The predeclared sources exceed the authorized selector calls")
    protocol = {"version": VERSION, "mode": "selection_development",
                "units": units, "backend": deepcopy(matched.BACKEND),
                "system": selection.ROUTE_INSTRUCTION + "\n" + selection.COMMON_CONTRACT,
                "source_manifest_path": str(source_path.resolve()),
                "source_manifest_sha256": matched.sha256_bytes(source_path.read_bytes()),
                "code_sha256": frozen_code(), "authorized_calls": authorized_calls,
                "planned_calls": len(units), "automatic_retry": False,
                "complete_script_generation": False, "evaluation_performed": False,
                "selector_assessments_to_writer": False,
                "reference_input": False, "test_access": False,
                "route_outcome_labels": 0, "route_preference_validated": False,
                "purpose": "Check actual source-conditioned choices and downstream process inputs; no artistic superiority or route correctness test"}
    path = out / "selection_protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Frozen routing development inputs changed; use an explicit new stage")
    if not path.exists():
        matched.save(path, protocol)
    return protocol


def run_selection_development(out, *, generate=None, workers=1):
    """Run only the frozen provisional selector; never launch a writer or judge."""
    out = Path(out)
    protocol = json.loads((out / "selection_protocol.json").read_text())
    if (protocol.get("version") != VERSION or protocol.get("mode") != "selection_development"
            or protocol.get("backend") != matched.BACKEND
            or protocol.get("selector_assessments_to_writer") is not False
            or protocol.get("code_sha256") != frozen_code()
            or protocol.get("system") != selection.ROUTE_INSTRUCTION + "\n" + selection.COMMON_CONTRACT):
        raise ValueError("The frozen selector implementation or authorized backend changed")
    units = [matched.validate_unit(u) for u in protocol["units"]]
    if (len(units) != protocol["planned_calls"]
            or len(units) > protocol["authorized_calls"]
            or not isinstance(workers, int) or isinstance(workers, bool) or workers < 1):
        raise ValueError("Invalid bounded routing development schedule")
    manifest = Path(protocol["source_manifest_path"])
    if matched.sha256_bytes(manifest.read_bytes()) != protocol["source_manifest_sha256"]:
        raise ValueError("The frozen source manifest changed")
    # A source-only process choice has no preference label and uses no evaluator.
    if generate is None:
        callback = matched.WikiGenerator(out)
        # Fail on missing transport dependencies before creating request receipts.
        callback.client()
    else:
        callback = generate
    results = []

    def run_one(unit):
        path = out / "selection_results" / (unit["id"] + ".json")
        source = unit["source"]
        payload = selection.route_input(source)
        request_sha = fingerprint({"stage": "select_entry", "system": protocol["system"],
                                   "input": payload, "backend": protocol["backend"]})
        if path.exists():
            prior = json.loads(path.read_text())
            if (prior.get("task_id") != unit["id"] or prior.get("work_id") != source["work_id"]
                    or prior.get("source_sha256") != fingerprint(source)
                    or prior.get("request_sha256") != request_sha):
                raise ValueError("The archived routing result belongs to different frozen inputs")
            if prior.get("status") == "complete":
                response = prior.get("response", {})
                if (response.get("served_model") != matched.MODEL
                        or response.get("source_requested") != "wiki"
                        or response.get("finish_reason") != "stop"):
                    raise ValueError("The cached routing receipt has an invalid actual API identity")
                route = selection.bind_route(response.get("result", {}), source)
                if (prior.get("route") != route
                        or prior.get("selector_assessments_to_writer") is not False
                        or "entry_guidance" in prior):
                    raise ValueError("The archived source-only selection record changed")
            return prior
        result = {"task_id": unit["id"], "work_id": source["work_id"],
                  "source_sha256": fingerprint(source), "request_sha256": request_sha, "status": "failed",
                  "route_outcome_labels": 0, "preference_validated": False,
                  "selector_assessments_to_writer": False,
                  "complete_script_generated": False}
        try:
            response = callback(unit["id"] + "__route_check_v2", "select_entry",
                                protocol["system"], deepcopy(payload), deepcopy(protocol["backend"]))
            result["response"] = deepcopy(response)
            if (response.get("served_model") != matched.MODEL
                    or response.get("source_requested") != "wiki"):
                raise ValueError("Actual routing model differs from authorized Wiki Luna")
            if (response.get("status", "complete") != "complete"
                    or response.get("finish_reason") != "stop"
                    or not isinstance(response.get("result"), dict)):
                raise ValueError("Preserve unavailable routing response without retry")
            route = selection.bind_route(response["result"], source)
            entry = route["selected_entry"]
            result.update(status="complete", route=route,
                          downstream_contract={"development_stage": selection.ENTRY_STAGE_NAMES[entry][0],
                                               "exploration_stage": selection.ENTRY_STAGE_NAMES[entry][1],
                                               "material_kind": selection.ENTRY_MATERIAL_KINDS[entry],
                                               "common_followup_stages": ["revise_program", "realize", "review", "final"],
                                               "executed": False})
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc)[:220])
        matched.save(path, result)
        return result

    with ThreadPoolExecutor(max_workers=min(workers, len(units))) as pool:
        results = list(pool.map(run_one, units))
    summary = {"version": VERSION, "mode": "selection_development",
               "prepared_works": len(units), "complete": sum(r["status"] == "complete" for r in results),
               "results": [{"task_id": r["task_id"], "work_id": r["work_id"], "status": r["status"],
                            "selected_entry": r.get("route", {}).get("selected_entry"),
                            "viable_alternatives": r.get("route", {}).get("viable_alternatives", [])}
                           for r in results],
               "billing": matched.billing([{**r.get("response", {}), "backend": protocol["backend"]}
                                           for r in results]),
               "route_outcome_labels": 0, "route_preference_validated": False,
               "selector_assessments_to_writer": False,
               "quality_gain_established": False, "complete_scripts_generated": 0,
               "evaluation_performed": False, "test_access": False,
               "limitation": "Contract-valid source-conditioned explanations are provisional model judgments; route distribution and self-reports do not establish useful selection"}
    matched.save(out / "selection_summary.json", summary)
    return summary


def prepare(source_path, out, *, entry="auto", backend=None, selector_backend=None,
             shared_superwriter_start=True, stage_backends=None, frozen_selector_results=None,
             execution_mode="native_complete_v3", scene_control=None, frozen_scene_prefixes=None):
    """Freeze one source-only DEV task per work without constructing API clients."""
    if entry not in ("auto", *selection.ENTRIES):
        raise ValueError("Unknown requested entry")
    if execution_mode not in EXECUTION_MODES:
        raise ValueError("Unknown entry execution mode")
    if scene_control is not None and (scene_control not in SCENE_CONTROLS or entry != "scene_first"
            or execution_mode != "native_complete_v4" or frozen_selector_results is not None):
        raise ValueError("Prepared scene controls need their explicit current fixed scene entry")
    if scene_control is None and frozen_scene_prefixes is not None:
        raise ValueError("A frozen prefix collection requires an explicit scene control")
    if scene_control == "freeze_prefix" and (frozen_scene_prefixes is not None or not shared_superwriter_start):
        raise ValueError("Prepared prefix freezing generates one real common initialization")
    if scene_control not in {None, "freeze_prefix"} and (frozen_scene_prefixes is None or shared_superwriter_start):
        raise ValueError("Prepared controls reuse their frozen prefixes without new initialization")
    if scene_control is None and execution_mode in {"native_complete_v3", "native_complete_v4"} and not shared_superwriter_start:
        raise ValueError("Prepared complete entry workflows must generate a real common plan and draft; reuse requires an explicit bound initialization")
    source_path, out = Path(source_path), Path(out)
    raw = json.loads(source_path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("units"), list) or not raw["units"]:
        raise ValueError("Reviewed source-only DEV units are required")
    units = [matched.validate_unit(u) for u in raw["units"]]
    if len({u["id"] for u in units}) != len(units) or len({u["source"]["work_id"] for u in units}) != len(units):
        raise ValueError("One predeclared task and draw per independent work is required")
    backend = deepcopy(matched.BACKEND if backend is None else backend)
    selector_backend = deepcopy(matched.BACKEND if selector_backend is None else selector_backend)
    matched.validate_backend(backend)
    assignments = bind_stage_backends(backend, stage_backends)
    if scene_control is not None and (backend["transport"] != "api"
            or any(b["transport"] != "api" for b in assignments.values()) or assignments["official"] != backend):
        raise ValueError("Prepared scene controls require the authorized Wiki Luna API roles")
    control_prefixes = (load_frozen_scene_prefixes(frozen_scene_prefixes, units, assignments)
                        if frozen_scene_prefixes is not None else None)
    frozen = (load_frozen_selections(frozen_selector_results, units, selector_backend)
              if frozen_selector_results is not None else None)
    if entry == "auto" and frozen is None:
        matched.validate_backend(selector_backend)
    if entry in {"auto", "relationships_first"} and not shared_superwriter_start and any(not u.get("current_script") for u in units):
        raise ValueError("Character evaluator entry needs a complete generated initial script")
    protocol = {"version": VERSION, "units": units, "entry": entry,
                 "backend": backend, "selector_backend": selector_backend,
                 "frozen_selector_results": frozen,
                 "selector_execution": "frozen" if frozen is not None else "live" if entry == "auto" else "none",
                 "selector_assessments_to_writer": False,
                "writer_stage_backends": assignments,
                "source_manifest_path": str(source_path.resolve()),
                "source_manifest_sha256": matched.sha256_bytes(source_path.read_bytes()),
                "shared_superwriter_start": shared_superwriter_start,
                "writer_calls_per_work": (9 if shared_superwriter_start else 7) if execution_mode == "prompt_materials_v6" else None,
                "execution_mode": execution_mode,
                "algorithm_version": algorithms.MATERIAL_ENTRY_VERSION if execution_mode == "native_complete_v4"
                    else algorithms.COMPLETE_ENTRY_VERSION if execution_mode == "native_complete_v3"
                    else algorithms.COMPLETE_SCENE_VERSION
                    if execution_mode == "native_complete_v2" else algorithms.VERSION,
                "algorithm_provenance": algorithms.PROVENANCE,
                 "selector_calls_per_work": int(entry == "auto" and frozen is None),
                "prompts": PROMPTS, "theory": matched.common.THEORY,
                "common_contract": selection.COMMON_CONTRACT, "borrowed_methods": borrowed_methods(),
                "code_sha256": frozen_code(scene_control=scene_control is not None), "route_outcome_labels": 0,
                "route_preference_validated": False, "quality_gain_established": False,
                "local_writer_template_binding": matched.local_writer_template_contract()
                if any(b["transport"] == "local" for b in assignments.values()) or selector_backend["transport"] == "local" else None,
                "reference_input": False, "test_access": False, "paid_retry": False,
                "fallback": False, "quality_selection": False,
                "evaluator_gate": "Actual applicable held-out human-validation report before any runtime call",
                "statistical_unit": "Independent source work, retaining author groups",
                "budget_claim": "Routing cost separate; equal writing-call opportunities do not establish equal compute",
                "scope": "Optional process implementation outside the main fixed-entry matched experiment"}
    if scene_control is not None:
        protocol.update(scene_control=scene_control, scene_control_version=scene_controls.VERSION,
            scene_control_module_sha256=matched.sha256_bytes(Path(scene_controls.__file__).read_bytes()),
            frozen_scene_prefixes=control_prefixes, training_eligible=False)
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Frozen protocol changed; choose an explicit new stage")
    if not path.exists():
        matched.save(path, protocol)
    return protocol


def run(out, report_path, *, workers=1, generate=None, selector_generate=None, stage_generate=None):
    """Activate the frozen stage; local inference must be injected explicitly."""
    out, report_path = Path(out), Path(report_path)
    protocol = json.loads((out / "protocol.json").read_text())
    if (protocol.get("version") != VERSION or protocol.get("code_sha256") != frozen_code(scene_control=protocol.get("scene_control") is not None)
            or protocol.get("prompts") != PROMPTS or protocol.get("theory") != matched.common.THEORY
            or protocol.get("common_contract") != selection.COMMON_CONTRACT
            or protocol.get("borrowed_methods") != borrowed_methods()):
        raise ValueError("Frozen execution code, prompt or primary artifact changed")
    if protocol.get("selector_assessments_to_writer") is not False:
        raise ValueError("Future execution must exclude selector assessments from writing")
    backend, selector_backend = protocol["backend"], protocol["selector_backend"]
    matched.validate_backend(backend)
    assignments = bind_stage_backends(backend, protocol.get("writer_stage_backends"))
    callbacks = check_stage_callbacks(backend, assignments, stage_generate)
    scene_control = protocol.get("scene_control")
    control_prefixes = protocol.get("frozen_scene_prefixes")
    if scene_control is not None:
        if (scene_control not in SCENE_CONTROLS or protocol.get("scene_control_version") != scene_controls.VERSION
                or protocol.get("scene_control_module_sha256") != matched.sha256_bytes(Path(scene_controls.__file__).read_bytes())
                or protocol.get("training_eligible") is not False):
            raise ValueError("Frozen scene-control implementation or identity changed")
        if control_prefixes is not None:
            for artifact in control_prefixes["artifacts"]:
                if matched.sha256_bytes(Path(artifact["path"]).read_bytes()) != artifact["sha256"]:
                    raise ValueError("A real frozen scene prefix changed after preparation")
            if fingerprint(control_prefixes["records"]) != control_prefixes["sha256"]:
                raise ValueError("Frozen prefix records changed")
    if generate is None and backend["transport"] == "local":
        raise ValueError("Local writer requires its explicit inference callback; no API fallback")
    frozen = protocol.get("frozen_selector_results")
    if frozen is not None:
        for artifact in frozen["artifacts"]:
            if matched.sha256_bytes(Path(artifact["path"]).read_bytes()) != artifact["sha256"]:
                raise ValueError("A frozen selector receipt changed after preparation")
        if fingerprint(frozen["records"]) != frozen["sha256"]:
            raise ValueError("Frozen selector records changed")
        for unit in protocol["units"]:
            bind_frozen_selection(frozen["records"][unit["id"]], unit, selector_backend)
    if protocol["entry"] == "auto" and frozen is None:
        matched.validate_backend(selector_backend)
        if selector_generate is None and selector_backend != backend:
            raise ValueError("A separate selector backend requires its explicit callback")
    if protocol.get("local_writer_template_binding") is not None and protocol["local_writer_template_binding"] != matched.local_writer_template_contract():
        raise ValueError("Frozen local input projection changed")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("A positive worker count is required")
    binding = matched.validate_evaluator({"report_path": str(report_path), "report_sha256": matched.sha256_bytes(report_path.read_bytes())})
    if control_prefixes is not None:
        for unit in protocol["units"]:
            prefix = control_prefixes["records"][unit["id"]]
            validate_scene_control_prefix(prefix, unit, assignments,
                expected_sha256=fingerprint(prefix), evaluator_binding=binding)
    activation = {"protocol_sha256": matched.sha256_bytes((out / "protocol.json").read_bytes()), "evaluator": binding}
    path = out / "activation.json"
    if path.exists() and json.loads(path.read_text()) != activation:
        raise ValueError("Frozen evaluator or protocol activation changed")
    if not path.exists():
        matched.save(path, activation)
    generator = matched.WikiGenerator(out) if generate is None else generate
    functions = matched.pinned_official_functions()

    def one(unit):
        path = out / "results" / (unit["id"] + ".json")
        if path.exists():
            prior = json.loads(path.read_text())
            if (prior.get("protocol_sha256") != activation["protocol_sha256"]
                    or prior.get("task_id") != unit["id"] or prior.get("source_sha256") != fingerprint(unit["source"])
                    or prior.get("evaluator_validation") != binding):
                raise ValueError("Saved candidate belongs to a different frozen stage")
            return prior
        control_settings = {}
        if scene_control is not None:
            control_settings["scene_control"] = scene_control
            if control_prefixes is not None:
                prefix = control_prefixes["records"][unit["id"]]
                control_settings.update(frozen_scene_prefix=prefix, frozen_scene_prefix_sha256=fingerprint(prefix))
        result = execute_case(unit, generator, evaluator_validation=binding, entry=protocol["entry"],
                              backend=backend, selector_generate=selector_generate, selector_backend=selector_backend,
                               official=functions, shared_superwriter_start=protocol["shared_superwriter_start"],
                               stage_backends=assignments, stage_generate=callbacks,
                               frozen_selection=frozen["records"][unit["id"]] if frozen is not None else None,
                               execution_mode=protocol["execution_mode"], **control_settings)
        result["protocol_sha256"] = activation["protocol_sha256"]
        matched.save(path, result)
        if result["status"] == "complete":
            path = out / "readable" / (unit["id"] + ".md")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result["script"])
        return result

    units = [matched.validate_unit(u) for u in protocol["units"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, units))
    summary = {"version": VERSION, "completed_at_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "planned_works": len(units), "author_groups": len({u["author_group"] for u in units}),
               "complete": sum(r["status"] == "complete" for r in results),
               "failed": sum(r["status"] != "complete" for r in results),
               "selected_entries": {e: sum(r.get("selected_entry") == e for r in results) for e in selection.ENTRIES},
               "actual_unique_billing": matched.billing([c for r in results for c in r["calls"]]),
               "selector_billing": matched.billing([c for r in results for c in r["calls"] if c["selector"]]),
               "writer_billing": matched.billing([c for r in results for c in r["calls"] if not c["selector"]]),
               "billing_by_backend": backend_billing([c for r in results for c in r["calls"]]),
                "writer_stage_backends": assignments,
                "selector_execution": protocol["selector_execution"],
                "selector_assessments_to_writer": False,
               "evaluator_validation": binding, "evaluation_performed": False, "test_accessed": False,
               "route_outcome_labels": 0, "route_preference_validated": False, "quality_gain_established": False}
    if scene_control is not None:
        summary.update(scene_control=scene_control, training_eligible=False,
            prefix_frozen=sum(r["status"] == "prefix_frozen" for r in results),
            failed=sum(r["status"] not in {"complete", "prefix_frozen"} for r in results),
            reused_prefix_calls=sum(r.get("reused_prefix_calls", 0) for r in results))
    matched.save(out / "generation_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "prepare_selection", "run_selection"))
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default="native_complete_v3")
    parser.add_argument("--scene-control", choices=SCENE_CONTROLS)
    parser.add_argument("--frozen-scene-prefixes", type=Path,
                        help="Real freeze_prefix result directory or JSON prefixes array; reused without inference")
    parser.add_argument("--entry", choices=("auto", *selection.ENTRIES), default="auto")
    parser.add_argument("--evaluator-report", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shared-superwriter-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-backend-json", type=Path)
    parser.add_argument("--selector-backend-json", type=Path)
    parser.add_argument("--stage-backends-json", type=Path)
    parser.add_argument("--frozen-selector-results", type=Path,
                        help="Existing selection_results directory or JSON selections array; no new selector call")
    parser.add_argument("--authorized-selector-calls", type=int)
    args = parser.parse_args()
    if (args.scene_control is not None or args.frozen_scene_prefixes is not None) and args.mode not in {"prepare", "run"}:
        parser.error("Scene-control flags apply only to the evaluator-gated prepare/run path")
    if args.frozen_scene_prefixes is not None and args.scene_control is None:
        parser.error("--frozen-scene-prefixes requires an explicit --scene-control")
    if args.scene_control is not None and args.out is None:
        parser.error("Scene controls require an explicit --out to preserve existing stages")
    if args.out is None:
        args.out = DEFAULT_MATERIAL_OUT if args.execution_mode == "native_complete_v4" else DEFAULT_OUT
    if args.mode == "prepare_selection":
        if args.sources is None or args.authorized_selector_calls is None:
            parser.error("--sources and --authorized-selector-calls are required")
        protocol = prepare_selection_development(args.sources, args.out,
                                                authorized_calls=args.authorized_selector_calls)
        print(json.dumps({"prepared_dev_works": len(protocol["units"]), "new_api_calls": 0,
                          "selector_calls": protocol["planned_calls"]}))
    elif args.mode == "run_selection":
        print(json.dumps(run_selection_development(args.out, workers=args.workers), ensure_ascii=False))
    elif args.mode == "prepare":
        if args.sources is None:
            parser.error("--sources is required")
        backend = json.loads(args.local_backend_json.read_text()) if args.local_backend_json else matched.BACKEND
        selector_backend = json.loads(args.selector_backend_json.read_text()) if args.selector_backend_json else matched.BACKEND
        stage_backends = json.loads(args.stage_backends_json.read_text()) if args.stage_backends_json else None
        protocol = prepare(args.sources, args.out, entry=args.entry, backend=backend,
                            selector_backend=selector_backend,
                            stage_backends=stage_backends,
                            frozen_selector_results=args.frozen_selector_results,
                           shared_superwriter_start=args.shared_superwriter_start,
                           execution_mode=args.execution_mode, scene_control=args.scene_control,
                           frozen_scene_prefixes=args.frozen_scene_prefixes)
        print(json.dumps({"prepared_dev_works": len(protocol["units"]), "new_api_calls": 0,
                          "writer_calls_per_work": protocol["writer_calls_per_work"],
                          "selector_calls_per_work": protocol["selector_calls_per_work"]}))
    else:
        if args.frozen_scene_prefixes is not None:
            parser.error("Freeze prefix files in prepare; run uses that exact frozen collection")
        if args.scene_control is not None:
            configured = json.loads((args.out / "protocol.json").read_text()).get("scene_control")
            if configured != args.scene_control:
                parser.error("--scene-control differs from the prepared protocol")
        if args.evaluator_report is None:
            parser.error("--evaluator-report is required")
        print(json.dumps(run(args.out, args.evaluator_report, workers=args.workers), ensure_ascii=False))


if __name__ == "__main__":
    main()
