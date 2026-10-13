"""Prepare and execute a source-only, opportunity-matched DEV comparison.

Preparation has no inference side effects. Execution requires an actual frozen
human-validation report and reviewed source passages. Callback tests do not
certify artistic quality. Historical experiments and TEST data are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sqlite3
import threading
from zoneinfo import ZoneInfo

from adaptation_system_contract import EDIT_AXES, fingerprint, validate_program, validate_source
import dramatic_adaptation_development as common


MODEL = "gpt-6-luna"
ARMS = ("scene_first", "detailed_planning", "whole_revision", "official_superwriter")
PRIMARY_COMPARISON_CONDITIONS = ("scene_first", "detailed_planning", "official_superwriter")
APPROACH_ABLATION = "scene_first_approach_update"
DEPENDENCIES = ("character_treatment", "preceding_preparation", "subsequent_development", "scene_organization")
VERSION = "scene-first-matched-v3"
LEGACY_VERSION = "scene-first-matched-v2"
PREFIX = "dmprobe1_scenematched3_"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "runs/comparison"
BACKEND = {"transport": "api", "model": MODEL, "source": "wiki"}

PLAN = """Read the supplied source text and adaptation brief within its stated source limits. When initial_plan and initial_script are supplied, develop from that shared official baseline rather than discarding its strengths. Develop a provisional complete short adaptation, then select ONE scene whose expression is worth exploring. Return JSON {program:{intent:string,choices:{plot:string,character_actions:string,relationships:string,ending:string,background:string,props:string,scene_organization:string,expression:string},scenes:[{scene_id:string,development:string}]},exploration:{scene_id:string,approaches:[string,string]}}. Consider all authorized edit axes; changes are optional. Propose exactly two genuinely different treatments of the SAME named scene, retaining the same situation and participants in these exploratory treatments. Keep the whole program provisional. Do not prescribe generic conflict, a moral lesson, or an automatic resolution. No hidden deliberation or quality score."""
ACTUAL = """Write one complete actual screenplay trial of exploration.scene_id using the supplied approach. Return JSON {scene_id:string,text:string}. The program provides surrounding context; explore only the named scene. Write actual speech, action, silence and chosen image, sound or interior voice. Internal cuts are allowed; do not write the surrounding scenes or the entire adaptation. Let particular interaction test the treatment. No quality claims or mandatory plot intensification."""
PLANNED = """Develop one substantial, detailed dramaturgical plan of exploration.scene_id using the supplied approach. Return JSON {scene_id:string,text:string|object|array}. The program provides surrounding context; explore only the named scene. Describe particular circumstances, interaction, psychological movement, timing, language, image, sound, preparation, later resonance and potential weaknesses in as much detail as useful. Reconsider the treatment when warranted. Keep this as a plan without fully writing the final dialogue. This control has the same whole-adaptation revision freedom as the actual-scene condition."""
UPDATE = """Read the original source, provisional program, and BOTH explored treatments of the same scene. Revise the complete adaptation program in light of what those treatments enable or lose. Return JSON {program:{intent:string,choices:{plot:string,character_actions:string,relationships:string,ending:string,background:string,props:string,scene_organization:string,expression:string},scenes:[{scene_id:string,development:string}]},selected_trial_id:string|null,dependency_changes:[{area:string,status:changed|retained|uncertain,before:string,after:string,reason:string,scene_ids:[string]}]}. selected_trial_id is '0', '1', or null; combining discoveries or rejecting both is legitimate. Record exactly one concrete entry for each area: character_treatment, preceding_preparation, subsequent_development, scene_organization. Explain actual before/after decisions or why retaining them is best. All authorized actions, cast, background, props, ending and scene arrangements may change. Preserve psychological layers, particular relationships and life meaning; do not force the provisional plan to be correct. Scene identifiers can change when their organization changes. No quality score."""
UPDATE_APPROACH = UPDATE.replace("Read the original source, provisional program, and BOTH explored treatments of the same scene.",
    "Read the original source, provisional program, and descriptions of BOTH approaches to the same scene. Actual scene trials are withheld from this update; do not invent observations from them.")
REALIZE = """Write the complete short adaptation from the source, brief, revised program and both exploratory treatments. Return JSON {script:string}. Treat exploratory material as provisional; revise its wording and expression to fit the whole. Fully develop the related character treatment, preceding preparation, subsequent development and scene organization recorded in the program. The best adaptation may retain strengths or reject exploratory ideas. Use the requested dramatic medium and finish the entire work naturally. Give only the complete script, without internal program labels or claims of improvement."""
WHOLE = """Write or revise the strongest COMPLETE short adaptation of the supplied source episode and brief. Return JSON {script:string,changes_and_losses:string}. When a current script is supplied, freely reconsider its interpretation, character treatment, relations, actions, background, props, ending, preceding preparation, subsequent development and scene organization. Retain strong passages when useful. Any part or the whole may change. Particular lived relationships, inner contradictions and rhythm matter; do not force more conflict, exposition or closure. Fully write scenes, dialogue and selected expression; an outline is insufficient. Finish naturally."""
REVIEW = """Read this complete adaptation and source as a screenwriter assessing the whole. Use supplied development_context as available earlier decisions, checking them against the actual draft; the recorded decisions are fallible and earn no quality advantage. Return JSON {strengths_to_preserve:[string],revision_opportunities:[{location:string,concern:string,proposed_change:string,possible_loss:string}],uncertainties:[string]}. Identify concrete potential improvements to particular people, inner movement, relations, life meaning, rhythm and how scenes support one another. Consider relevant earlier preparation and later development. Consolidate overlapping suggestions and identify incompatible suggestions before recommending changes; explain the effect and possible loss across the relevant scenes. Advice is fallible and may be empty. Distinguish actual incompatibilities from intended ambiguity or offscreen action. Do not award a score or require increased conflict, source fidelity or explanatory speech."""
FINAL = """Use the source, brief, complete draft, available development_context and fallible whole reading to write the strongest COMPLETE final adaptation. Check earlier decisions against the finished work; retain or explicitly reconsider them as useful. Return JSON {script:string,applied_changes:[string],rejected_suggestions:[string]}. Make useful changes and reject advice that would damage particular people, inner contradictions, relationships, life meaning or rhythm. Resolve incompatible advice and check revised character behavior, earlier preparation, later development and scene transitions against the complete finished script. Any authorized change to character treatment, earlier preparation, later development, scene organization, actions, props, background or ending is allowed. Preserve strong passages. Empty advice may justify retaining the complete draft. Finish naturally; do not truncate or add program labels or quality claims."""
OFFICIAL = """Execute the supplied pinned official writing stage for this source episode and shared adaptation brief. Return exactly JSON {answer:string}, containing the complete public plan or complete screenplay requested by that stage. The original source and brief remain available in this call. Adaptation choices may freely change within the brief. No hidden deliberation, minimum length, paid retry or quality claim."""
PROMPTS = {"plan": PLAN, "trial": ACTUAL, "scene_plan": PLANNED, "revise_program": UPDATE,
           "revise_program_approach_only": UPDATE_APPROACH,
           "realize": REALIZE, "whole": WHOLE, "review": REVIEW, "final": FINAL, "official": OFFICIAL}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_confirmation_source(unit):
    """Bind confirmation to frozen source-only reservation and actual readability QA."""
    binding = unit.get("confirmation_source_binding", {})
    artifacts = {}
    for name in ("reservation", "readability_review"):
        descriptor = binding.get(name, {})
        path = Path(descriptor.get("path", ""))
        if not path.is_file() or not descriptor.get("sha256"):
            raise ValueError("Confirmation requires its actual frozen " + name)
        body = path.read_bytes()
        if sha256_bytes(body) != descriptor["sha256"]:
            raise ValueError("Confirmation source " + name + " snapshot changed")
        artifacts[name] = json.loads(body)
    reservation, review = artifacts["reservation"], artifacts["readability_review"]
    if (reservation.get("version") != "confirmation-source-metadata-amendment-v3"
            or reservation.get("known_writer_intersections_in_selected_metadata") != 0
            or not any(row.get("work_id") == unit.get("source", {}).get("work_id")
                       for row in reservation.get("reserved_works", []))):
        raise ValueError("Confirmation source is outside its frozen known-overlap-screened reservation")
    if (review.get("version") not in {"confirmation-source-readability-review-v1", "confirmation-source-readability-review-v2"}
            or review.get("reservation_sha256") != binding["reservation"]["sha256"]
            or review.get("target_assisted_retrieval_used") is not False
            or review.get("human_preference_judgments") != 0):
        raise ValueError("Confirmation requires source-only readability review separate from preferences")
    rows = [row for row in review.get("records", []) if row.get("work_id") == unit["source"]["work_id"]]
    projection = {key: unit[key] for key in ("id", "split", "author_group", "source", "source_boundaries")}
    projection["id"] = binding.get("source_unit_id", unit["id"])
    if (len(rows) != 1 or rows[0].get("readable") is not True or rows[0].get("needs_source_repair") is not False
            or rows[0].get("reviewed_unit_sha256") != fingerprint(projection)):
        raise ValueError("This exact confirmation source unit lacks completed readability review")
    return deepcopy(binding)


def validate_unit(unit):
    """Require reviewed passages while preserving DEV and confirmation partition labels."""
    if unit.get("split") not in {"dev", "confirmation"}:
        raise ValueError("This runner permits only audited DEV or explicitly bound confirmation sources")
    confirmation = validate_confirmation_source(unit) if unit["split"] == "confirmation" else None
    if not isinstance(unit.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", unit["id"]):
        raise ValueError("A stable safe task identifier is required")
    if not unit.get("author_group"):
        raise ValueError("Author grouping is required")
    source = validate_source(unit.get("source", {}))
    if not isinstance(source["source_text"], str) or not source["source_text"].strip():
        raise ValueError("Complete source text is required")
    boundaries = unit.get("source_boundaries", {})
    if not boundaries.get("reviewer") or not boundaries.get("unit_description"):
        raise ValueError("An actual source-reading reviewer and unit description are required")
    scope = boundaries.get("scope", "complete_source_unit")
    if scope == "complete_source_unit":
        if (boundaries.get("status") != "verified" or boundaries.get("edition_verified") is not True
                or boundaries.get("complete_unit") is not True):
            raise ValueError("Actual edition and complete-source-unit boundary review is required")
    elif scope == "bounded_source_passage":
        if (boundaries.get("status") != "readable_passage_checked"
                or boundaries.get("readable_passage") is not True or boundaries.get("complete_unit") is not False
                or source["brief"].get("scope") != "bounded_source_passage"
                or not source["brief"].get("source_limits")):
            raise ValueError("Bounded passage requires actual readability review and an explicit limited adaptation brief")
    else:
        raise ValueError("Unsupported source comparison scope")
    if boundaries.get("source_sha256") != sha256_bytes(source["source_text"].encode()):
        raise ValueError("Source boundary review belongs to a different source text")
    result = {"id": unit["id"], "split": unit["split"], "author_group": unit["author_group"],
              "source": source, "source_boundaries": deepcopy(boundaries)}
    if confirmation is not None:
        result["confirmation_source_binding"] = confirmation
    if unit.get("current_script") is not None:
        if not isinstance(unit["current_script"], str) or not unit["current_script"].strip():
            raise ValueError("Optional shared current script must be complete text")
        if unit.get("current_script_origin") != "shared_generated_draft":
            raise ValueError("A real reference adaptation cannot be the shared generated draft")
        result.update(current_script=unit["current_script"], current_script_origin=unit["current_script_origin"])
    return result


def validate_evaluator(binding):
    """Validate the actual report before any generation backend is instantiated."""
    path = Path(binding.get("report_path", ""))
    if not path.is_file() or not binding.get("report_sha256"):
        raise ValueError("The actual frozen evaluator validation report is required")
    data = path.read_bytes()
    if sha256_bytes(data) != binding["report_sha256"]:
        raise ValueError("Evaluator validation report snapshot changed")
    report = json.loads(data)
    if (report.get("status") != "validated" or report.get("scope") != "screenplay_draft_preferences"
            or isinstance(report.get("human_judgments"), bool)
            or not isinstance(report.get("human_judgments"), int) or report["human_judgments"] <= 0
            or report.get("heldout_groups_verified") is not True):
        raise ValueError("Applicable held-out real-human screenplay evaluation is unvalidated")
    return {"report_path": str(path.resolve()), "report_sha256": binding["report_sha256"],
            "status": report["status"], "scope": report["scope"],
            "human_judgments": report["human_judgments"]}


def validate_backend(backend):
    if backend.get("transport") == "api":
        if backend.get("model") != MODEL or backend.get("source") != "wiki":
            raise ValueError("Only the authorized Wiki gpt-6-luna API may generate")
    elif (backend.get("transport") != "local" or not isinstance(backend.get("revision"), str)
          or not backend["revision"].strip()):
        raise ValueError("A callback-local backend requires its explicit frozen revision")
    elif backend.get("source", "local") != "local":
        raise ValueError("A local checkpoint cannot declare a remote source")


TERMINAL_CLOSURE_RULE = "eos-json-terminal-delimiters-v1"


def parse_stage_response(value, *, allow_terminal_closure=False, return_receipt=False):
    """Restore only missing terminal delimiters after EOS, preserving all values."""
    def unique(fields):
        result = {}
        for key, item in fields:
            if key in result:
                raise ValueError("Duplicate structured stage field: " + key)
            result[key] = item
        return result
    if not isinstance(value, str):
        raise TypeError("The original stage output must be text")
    raw_sha = sha256_bytes(value.encode())
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip())
    added = ""
    def strict(text):
        def invalid_constant(name):
            raise ValueError("Non-finite JSON constant: " + name)
        return json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
    try:
        result = strict(value)
    except json.JSONDecodeError:
        if not allow_terminal_closure:
            raise
        stack, quoted, escaped = [], False, False
        for character in value:
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character in "{[":
                stack.append(character)
            elif character in "}]":
                expected = "{" if character == "}" else "["
                if not stack or stack.pop() != expected:
                    raise ValueError("Mismatched JSON delimiters cannot be restored")
        if quoted or escaped or not stack:
            raise ValueError("Only missing terminal container delimiters may be restored")
        added = "".join("}" if item == "{" else "]" for item in reversed(stack))
        result = strict(value + added)
    if not isinstance(result, dict):
        raise ValueError("A writing stage must return a JSON object")
    receipt = {"rule_version": TERMINAL_CLOSURE_RULE, "original_raw_sha256": raw_sha,
               "appended_delimiters": added, "recovery_applied": bool(added),
               "parsed_text_sha256": sha256_bytes((value + added).encode()),
               "EOS_required_for_recovery": True, "key_value_or_body_changed": False}
    return (result, receipt) if return_receipt else result


def validate_creative_stage_schema(stage, value, payload):
    """Apply the existing complete creative task shape before accepting a response."""
    if isinstance(payload.get("request_json"), str):
        payload = json.loads(payload["request_json"])
    if stage == "plan" and payload.get("requested_representation") == "detailed_scene_plan":
        from scene_entry_mechanism_controls import _plan_output
        _plan_output(value, payload["scene_id"])
    elif stage == "trial":
        identifier = payload.get("scene_id", payload.get("exploration", {}).get("scene_id"))
        if (not isinstance(identifier, str) or value.get("scene_id") != identifier
                or not isinstance(value.get("text"), str) or not value["text"].strip()):
            raise ValueError("Missing assigned complete-scene identifier or text")
    elif stage in {"realize", "revise_script"}:
        if not isinstance(value.get("script"), str) or not value["script"].strip():
            raise ValueError("Missing complete-script field")


def local_writer_messages(stage, system, payload, *, serialization="legacy"):
    """Reuse the exact released adaptation_process training input projection."""
    from prepare_local_writer_data import render_request
    from local_adaptation_writer_interface import serialize_request
    return render_request("adaptation_process", {
        "stage": stage, "request_json": serialize_request(payload, mode=serialization),
        "stage_instruction": system})


def local_writer_template_contract():
    from prepare_local_writer_data import INSTRUCTIONS, INPUT_FIELDS, render_request
    return {"task": "adaptation_process", "instruction": INSTRUCTIONS["adaptation_process"],
            "fields": sorted(INPUT_FIELDS["adaptation_process"]),
            "renderer_sha256": fingerprint(inspect.getsource(render_request)),
            "scope": "Exact input projection; actual training coverage and generation competence are separate checks"}


class LocalWriterGenerator:
    """Adapt an explicit local-only inference callback, without loading weights.

    infer(messages, backend) returns complete text and actual served_revision.
    The same adapter and template apply to every condition and writing stage.
    This interface does not establish that each task has been trained or works.
    """
    def __init__(self, infer, backend, *, serialization="legacy"):
        validate_backend(backend)
        if backend["transport"] != "local":
            raise ValueError("The local writer adapter cannot invoke an API")
        from local_adaptation_writer_interface import SERIALIZATION_MODES
        if serialization not in SERIALIZATION_MODES:
            raise ValueError("Unknown local writing request serialization")
        self.infer, self.backend = infer, deepcopy(backend)
        self.serialization = serialization

    def __call__(self, key, stage, system, payload, backend):
        if backend != self.backend:
            raise ValueError("The local writer checkpoint binding changed")
        response = self.infer(local_writer_messages(stage, system, payload, serialization=self.serialization), deepcopy(backend))
        result = {k: deepcopy(response.get(k)) for k in
                  ("served_revision", "served_model", "source_requested", "finish_reason", "usage", "raw_path")}
        result["raw_text"] = deepcopy(response.get("text"))
        if (response.get("served_revision") != backend["revision"] or response.get("source_requested") != "local"
                or response.get("finish_reason") != "stop"):
            return {**result, "status": "failed", "error_type": "LocalIdentityOrTerminationFailure"}
        try:
            parsed, parsing = parse_stage_response(response["text"], allow_terminal_closure=True, return_receipt=True)
            result["parsing"] = parsing
            validate_creative_stage_schema(stage, parsed, payload)
            result.update(result=parsed, parsing=parsing, raw_text=response["text"], status="complete")
        except (ValueError, KeyError, TypeError) as exc:
            result.update(status="failed", error_type=type(exc).__name__)
        return result


def pinned_official_functions():
    from official_superwriter_adaptation import official_functions
    return official_functions()


def official_provenance():
    from official_superwriter_adaptation import REVISION, SNAPSHOT
    pinned_official_functions()
    path = SNAPSHOT / "provenance.json"
    return {"revision": REVISION, "provenance_path": str(path), "provenance_sha256": sha256_bytes(path.read_bytes()),
            "scope": "Pinned official pure inference prompts on the common API; no official trained weights or training reproduction",
            "native_stages": ["plan", "write", "refine"],
            "legacy_matched_stages": ["plan", "write", "refine", "refine", "refine"],
            "legacy_extra_refine_rounds": 2,
            "shared_start_stages": ["plan", "write"],
            "shared_start_followup_stages": ["refine"]*5,
            "shared_start_total_refine_rounds": 5}


def normalize_conditions(comparison_conditions=None):
    """Freeze the selected comparison without adding optional conditions silently."""
    if comparison_conditions is None:
        return PRIMARY_COMPARISON_CONDITIONS
    if (not isinstance(comparison_conditions, (list, tuple)) or not comparison_conditions
            or any(not isinstance(arm, str) or arm not in ARMS for arm in comparison_conditions)
            or len(set(comparison_conditions)) != len(comparison_conditions)):
        raise ValueError("Comparison conditions require a unique, nonempty supported subset")
    return tuple(arm for arm in ARMS if arm in comparison_conditions)


def opportunity_layout(shared_superwriter_start, include_approach_update_ablation, comparison_conditions=None):
    """Keep declared call opportunities identical to the actual execution graph."""
    if type(shared_superwriter_start) is not bool or type(include_approach_update_ablation) is not bool:
        raise ValueError("Experimental options require explicit Boolean values")
    conditions = normalize_conditions(comparison_conditions)
    if include_approach_update_ablation and "scene_first" not in conditions:
        raise ValueError("The approach-update diagnostic requires the actual scene-first condition")
    arms = conditions + ((APPROACH_ABLATION,) if include_approach_update_ablation else ())
    count = 9 if shared_superwriter_start else 7
    unique = (2 if shared_superwriter_start else 0)+7*len(conditions)
    if {"scene_first", "detailed_planning"} <= set(conditions):
        unique -= 1
    if include_approach_update_ablation:
        unique += 4
    return {"arms": list(arms), "draws": 1, "calls_per_arm": count,
            "comparison_conditions": list(conditions),
            "followup_calls_per_arm": 7, "shared_initial_calls_per_case": 2 if shared_superwriter_start else 0,
            "virtual_calls_per_case": len(arms)*count,
            "unique_calls_per_complete_case": unique,
            "shared_superwriter_start": shared_superwriter_start,
            "include_approach_update_ablation": include_approach_update_ablation}


def validate_opportunity_layout(protocol):
    legacy = protocol.get("version") == LEGACY_VERSION
    expected = opportunity_layout(protocol.get("shared_superwriter_start"), protocol.get("include_approach_update_ablation"),
                                  protocol.get("comparison_conditions", ARMS if legacy else None))
    if legacy and "comparison_conditions" not in protocol:
        expected.pop("comparison_conditions")
    if any(type(protocol.get(key)) is not type(value) or protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Prepared conditions or call opportunities differ from the actual execution graph")


def prepare(source_path, out, *, shared_superwriter_start=True, backend=None,
            include_approach_update_ablation=False, comparison_conditions=None):
    """Freeze audited source-only DEV inputs without calling inference."""
    source_path, out = Path(source_path), Path(out)
    raw = json.loads(source_path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("units"), list) or not raw["units"]:
        raise ValueError("Expected source-only {units:[...]} with explicit source-reading reviews")
    units = [validate_unit(row) for row in raw["units"]]
    backend = deepcopy(BACKEND if backend is None else backend)
    validate_backend(backend)
    layout = opportunity_layout(shared_superwriter_start, include_approach_update_ablation, comparison_conditions)
    if len({row["id"] for row in units}) != len(units) or len({row["source"]["work_id"] for row in units}) != len(units):
        raise ValueError("Exactly one predeclared task and one draw per independent DEV work")
    value = {"version": VERSION, "units": units, "source_manifest_path": str(source_path.resolve()),
             "source_manifest_sha256": sha256_bytes(source_path.read_bytes()), **layout,
             "primary_comparison_conditions": list(PRIMARY_COMPARISON_CONDITIONS),
             "legacy_condition_provenance": "v2 implemented four conditions including whole_revision; historical protocols, failures and results remain unchanged; v3 makes whole_revision explicitly optional",
             "approach_update_ablation": "Optional separate diagnostic; share exact 2 actual trials with scene_first; withhold texts only from global update; expose same 2 texts to final realization; add 4 unique calls" if include_approach_update_ablation else "Disabled; execute only the frozen comparison_conditions",
             "shared_initial_stages": ["plan", "write"] if shared_superwriter_start else [],
             "official_followup_stages": ["refine"] * 5 if shared_superwriter_start else ["plan", "write", "refine", "refine", "refine"],
             "shared_proposal": "One actual plan and two approaches to the same scene shared by actual and planning; reuse its exact SHA",
             "backend": backend, "effort": "low", "native_output_allowance": True,
             "local_writer_template": "prepare_local_writer_data.render_request('adaptation_process', stage/request_json/stage_instruction)" if backend["transport"] == "local" else None,
             "local_writer_template_binding": local_writer_template_contract() if backend["transport"] == "local" else None,
             "paid_retry": False, "fallback": False, "quality_selection": False,
             "reference_input": False, "test_access": False,
             "source_scope_claim": "Report complete reviewed episodes and bounded readable passages separately; bounded passages do not establish complete-book adaptation",
             "pipeline_initialization": "Shared pinned official plan/write baseline, paid once and included in every virtual pipeline" if shared_superwriter_start else "Optional identical pre-existing generated draft; its historical creation cost is separate",
             "budget_claim": "Equal call opportunities, not equal input/output/reasoning tokens or monetary cost",
             "actual_billing": "Unique request receipts; shared proposal paid once; unknown cost remains unknown",
             "virtual_billing": "Counterfactual per-pipeline receipts including the shared proposal in both pipelines",
             "dependency_areas": list(DEPENDENCIES), "prompts": PROMPTS, "theory": common.THEORY,
             "whole_revision_history": "Every arm supplies its actually available compact development decisions to the common whole reading and final writer; no extra inference",
             "common_refinement_source": {"source": "https://arxiv.org/html/2510.05188v4",
                 "appendix": ["A.2", "A.5", "B.5", "C.5"],
                 "scope": "Borrow concrete character/context diagnosis, suggestion consolidation and continuity rechecking; one common review/final pair is not a reproduction of the multi-agent Dramaturge"},
             "official": official_provenance(), "evaluator_gate": "Actual validated held-out report required at execution; no certificate created here",
             "statistical_unit": "Original work, with author clusters; no independent-count inflation from scenes or judgments",
             "failure_policy": "Retain every task, arm and incomplete attempt; never replace a failed paid call or pick a favorable script",
             "runner_sha256": sha256_bytes(Path(__file__).read_bytes())}
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("Prepared protocol changed; preserve the historical stage and choose an explicit new stage")
    if not path.exists():
        save(path, value)
    return value


def check_revision(value, program, materials):
    validate_program(value.get("program", {}))
    if value.get("selected_trial_id") not in {None, "0", "1"}:
        raise ValueError("Selected exploratory treatment does not exist")
    changes = value.get("dependency_changes", [])
    if not isinstance(changes, list) or len(changes) != len(DEPENDENCIES):
        raise ValueError("Record each character, preparation, aftermath and organization decision")
    if {row.get("area") for row in changes} != set(DEPENDENCIES):
        raise ValueError("Missing or duplicate dependency area")
    scene_ids = {s["scene_id"] for s in program["scenes"] + value["program"]["scenes"]}
    for row in changes:
        if row.get("status") not in {"changed", "retained", "uncertain"}:
            raise ValueError("Dependency decisions require an explicit status")
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("before", "after", "reason")):
            raise ValueError("Dependency decisions require concrete before, after and reasons")
        if not isinstance(row.get("scene_ids"), list) or not set(row["scene_ids"]) <= scene_ids:
            raise ValueError("Dependency refers to unknown scenes")
    if len(materials) != 2:
        raise ValueError("Both explorations must enter the identical update phase")


def text(value, key):
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Missing complete text field: " + key)
    return result


def billing(records):
    """Report known receipts and uncertainty without treating unknown as zero."""
    known, unknown, prompt, completion, reasoning, cached = 0.0, 0, 0, 0, 0, 0
    missing_prompt, missing_completion, missing_cached = 0, 0, 0
    for row in records:
        usage = row.get("usage") or {}
        cost = usage.get("cost")
        if cost is None:
            unknown += 1
        else:
            known += float(cost)
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        reasoning += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached += int(details.get("cached_tokens") or 0)
        missing_prompt += usage.get("prompt_tokens") is None
        missing_completion += usage.get("completion_tokens") is None
        missing_cached += details.get("cached_tokens") is None
    return {"requests": len(records), "known_cost_usd": known, "unknown_cost_requests": unknown,
            "total_cost_usd": known if unknown == 0 else None, "prompt_tokens": prompt,
            "completion_tokens": completion, "reported_reasoning_tokens": reasoning,
            "reported_cached_prompt_tokens": cached, "unknown_prompt_usage_requests": missing_prompt,
            "unknown_completion_usage_requests": missing_completion, "unknown_cached_prompt_usage_requests": missing_cached,
            "token_sums_scope": "Sum only fields actually reported; missing token usage is not measured zero"}


def execute_case(unit, generate, *, backend, evaluator_validation, official=None,
                 shared_superwriter_start=True, include_approach_update_ablation=False,
                 comparison_conditions=None):
    """Execute only the frozen conditions through one injectable stage callback."""
    unit = validate_unit(unit)
    validate_backend(backend)
    layout = opportunity_layout(shared_superwriter_start, include_approach_update_ablation, comparison_conditions)
    binding = validate_evaluator(evaluator_validation)
    functions = official if official is not None else pinned_official_functions()
    records, outputs = {}, {}
    source = unit["source"]
    available = {"source": deepcopy(source)}
    if "current_script" in unit:
        available["current_script"] = unit["current_script"]

    def call(key, stage, prompt, payload, trace):
        identity = unit["id"] + "__" + key
        system = prompt + "\n" + common.THEORY
        request_sha = fingerprint({"stage": stage, "system": system, "input": payload, "backend": backend})
        reused = identity in records
        if reused:
            record = records[identity]
            if record["request_sha256"] != request_sha:
                raise ValueError("Shared call changed its original inputs")
        else:
            record = {"request_identity": identity, "stage": stage, "system": system,
                      "input": deepcopy(payload), "request_sha256": request_sha,
                      "backend": deepcopy(backend), "status": "failed"}
            records[identity] = record
            try:
                response = generate(identity, stage, system, deepcopy(payload), deepcopy(backend))
                record.update({k: deepcopy(response.get(k)) for k in
                               ("usage", "served_model", "served_revision", "source_requested", "raw_path", "finish_reason")})
                if backend["transport"] == "api":
                    if response.get("served_model") != MODEL or response.get("source_requested") != "wiki":
                        raise ValueError("Actual API model or Wiki route differs from the request")
                elif response.get("served_revision") != backend["revision"] or response.get("source_requested") != "local":
                    raise ValueError("Actual local revision or local-only source differs from the request")
                if response.get("status", "complete") != "complete" or not isinstance(response.get("result"), dict):
                    raise ValueError("Unavailable stage result; preserve without another paid attempt")
                if response.get("finish_reason", "stop") != "stop":
                    raise ValueError("Incomplete response; preserve without another paid attempt")
                record.update(status="complete", output=deepcopy(response["result"]))
            except Exception as exc:
                record.update(error_type=type(exc).__name__, error=str(exc)[:200] if isinstance(exc, (ValueError, RuntimeError)) else "See original archived stage")
        trace.append({"request_identity": identity, "request_sha256": request_sha, "reused": reused,
                      "stage": stage, "status": record["status"], "usage": deepcopy(record.get("usage"))})
        if record["status"] != "complete":
            raise RuntimeError("Unavailable preserved stage: " + identity)
        return deepcopy(record["output"])

    def integration(arm, draft, trace, history):
        review = call(arm + "__whole_read", "review", REVIEW,
                      {**available, "script": draft, "development_context": history}, trace)
        final = call(arm + "__whole_write", "final", FINAL,
                     {**available, "script": draft, "whole_reading": review, "development_context": history}, trace)
        return text(final, "script"), review, final

    shared_trace, shared_initial = [], None
    shared_error = None
    if shared_superwriter_start:
        try:
            query = json.dumps({**available, "dramaturgy": common.THEORY}, ensure_ascii=False)
            prompt = functions["generate_prompt"]("plan", use_query=query, previous_output=None)
            value = call("shared_official_plan", "official", OFFICIAL,
                         {**available, "official_stage": "plan", "official_prompt": prompt}, shared_trace)
            initial_plan = text(value, "answer")
            prompt = functions["generate_prompt"]("write", use_query=query,
                                                   previous_output="<answer>" + initial_plan + "</answer>")
            value = call("shared_official_write", "official", OFFICIAL,
                         {**available, "initial_plan": initial_plan, "official_stage": "write", "official_prompt": prompt}, shared_trace)
            initial_script = text(value, "answer")
            available.pop("current_script", None)
            available.update(initial_plan=initial_plan, initial_script=initial_script)
            shared_initial = {"status": "complete", "initial_plan": initial_plan,
                              "initial_script": initial_script, "trace": deepcopy(shared_trace)}
        except Exception as exc:
            shared_error = str(exc)[:220] if isinstance(exc, (ValueError, RuntimeError)) else "See archived shared baseline stage"
            shared_initial = {"status": "failed", "error_type": type(exc).__name__,
                              "error": shared_error, "trace": deepcopy(shared_trace)}

    arms = tuple(layout["arms"])
    for arm in arms:
        trace = deepcopy(shared_trace)
        if arm != arms[0]:
            for record in trace:
                record["reused"] = True
        output = {"work_id": source["work_id"], "task_id": unit["id"], "author_group": unit["author_group"],
                  "draw": 0, "arm": arm, "status": "failed", "source_sha256": fingerprint(source),
                  "quality_gain_established": False, "trace": trace}
        try:
            if shared_error is not None:
                raise RuntimeError("Unavailable shared official baseline: " + shared_error)
            history = {}
            if arm in {"scene_first", "detailed_planning", APPROACH_ABLATION}:
                proposal = call("shared_proposal", "plan", PLAN, available, trace)
                program = validate_program(proposal.get("program", {}))
                exploration = proposal.get("exploration", {})
                approaches = exploration.get("approaches", [])
                if (exploration.get("scene_id") not in {s["scene_id"] for s in program["scenes"]}
                        or not isinstance(approaches, list) or len(approaches) != 2
                        or any(not isinstance(a, str) or not a.strip() for a in approaches)):
                    raise ValueError("Exactly two approaches to the same named program scene are required")
                materials = []
                for i, approach in enumerate(approaches):
                    payload = {**available, "program": program,
                               "exploration": {"scene_id": exploration["scene_id"], "approach": approach}, "trial_id": str(i)}
                    actual = arm != "detailed_planning"
                    trial_arm = "scene_first" if arm == APPROACH_ABLATION else arm
                    trial = call(trial_arm + "__trial_" + str(i), "trial" if actual else "scene_plan",
                                 ACTUAL if actual else PLANNED, payload, trace)
                    if trial.get("scene_id") != exploration["scene_id"]:
                        raise ValueError("Exploration changed the selected scene identifier")
                    value = trial.get("text")
                    if actual:
                        text(trial, "text")
                    elif not isinstance(value, (str, dict, list)) or not value:
                        raise ValueError("Detailed planning must contain substantive scene material")
                    materials.append({"trial_id": str(i), "scene_id": exploration["scene_id"], "text": value,
                                      "representation": "actual_screenplay" if actual else "detailed_plan"})
                content = {**available, "program": program, "exploration": exploration, "explorations": materials}
                if arm == APPROACH_ABLATION:
                    content["explorations"] = [{"trial_id": str(i), "scene_id": exploration["scene_id"], "approach": approach}
                                               for i, approach in enumerate(approaches)]
                revision = call(arm + "__program_update", "revise_program",
                                UPDATE_APPROACH if arm == APPROACH_ABLATION else UPDATE, content, trace)
                check_revision(revision, program, materials)
                first = call(arm + "__realize", "realize", REALIZE,
                             {**available, "program": revision["program"], "original_program": program,
                              "explorations": materials, "selected_trial_id": revision.get("selected_trial_id"),
                              "dependency_changes": revision["dependency_changes"]}, trace)
                output.update(proposal=proposal, proposal_sha256=fingerprint(proposal), explorations=materials,
                              revision=revision, dependency_changes=revision["dependency_changes"])
                history = {"program": revision["program"], "selected_trial_id": revision.get("selected_trial_id"),
                    "dependency_changes": revision["dependency_changes"],
                    "explorations": [{"trial_id": m["trial_id"], "scene_id": m["scene_id"], "text_sha256": fingerprint(m["text"])} for m in materials],
                    "trial_texts_available_in_this_reading": False,
                    "instruction": "Recorded decisions from actual development; trial text is not supplied here. Assess them against this complete script and preserve or revise them concretely."}
                draft = text(first, "script")
            elif arm == "whole_revision":
                payload = deepcopy(available)
                if shared_superwriter_start:
                    payload["current_script"] = available["initial_script"]
                history = {"revision_decisions": []}
                for i in range(5):
                    first = call(arm + "__rewrite_" + str(i), "whole", WHOLE, payload, trace)
                    draft = text(first, "script")
                    history["revision_decisions"].append({"iteration": i, "changes_and_losses": first.get("changes_and_losses"),
                                                           "script_sha256": fingerprint(draft)})
                    payload = {**available, "current_script": draft}
            else:
                query = json.dumps({**available, "dramaturgy": common.THEORY}, ensure_ascii=False)
                previous = "<answer>" + available["initial_script"] + "</answer>" if shared_superwriter_start else None
                stages = ("refine",) * 5 if shared_superwriter_start else ("plan", "write", "refine", "refine", "refine")
                for i, stage in enumerate(stages):
                    prompt = functions["generate_prompt"](stage, use_query=query, previous_output=previous)
                    result = call(arm + "__official_" + str(i), "official", OFFICIAL,
                                  {**available, "official_stage": stage, "official_prompt": prompt}, trace)
                    answer = text(result, "answer")
                    previous = "<answer>" + answer + "</answer>"
                draft = answer
                history = {"official_stages_completed": list(stages), "current_script_sha256": fingerprint(draft),
                           "instruction": "This script resulted from the recorded public planning/writing/refinement stages; no separate change explanation was requested by the official prompt."}
            script, reading, final = integration(arm, draft, trace, history)
            if len(trace) != (9 if shared_superwriter_start else 7):
                raise ValueError("Each complete arm must have seven followup opportunities plus its shared baseline")
            output.update(status="complete", script=script, before_whole_integration=draft,
                          whole_reading=reading, final_revision=final, development_context=history)
        except Exception as exc:
            output.update(error_type=type(exc).__name__, error=str(exc)[:220] if isinstance(exc, (ValueError, RuntimeError)) else "See archived request")
        output["virtual_pipeline_billing"] = billing([records[t["request_identity"]] for t in trace])
        output["executed_opportunities"] = len(trace)
        output["planned_opportunities"] = 9 if shared_superwriter_start else 7
        output["planned_followup_opportunities"] = 7
        outputs[arm] = output
    return {"task_id": unit["id"], "work_id": source["work_id"], "author_group": unit["author_group"],
            "outputs": outputs, "calls": list(records.values()), "actual_unique_billing": billing(records.values()),
            "shared_superwriter_start": shared_superwriter_start,
            "include_approach_update_ablation": include_approach_update_ablation,
            "comparison_conditions": layout["comparison_conditions"],
            "shared_initial": shared_initial,
            "shared_initial_billing": billing([records[t["request_identity"]] for t in shared_trace]),
            "virtual_pipeline_billing": {arm: row["virtual_pipeline_billing"] for arm, row in outputs.items()},
            "evaluator_validation": binding, "quality_gain_established": False,
            "call_counts_are_not_equal_compute": True}


class WikiGenerator:
    """Use the existing authorized transport with a task-specific receipt ledger."""
    def __init__(self, out):
        self.out = Path(out)
        self.clients = threading.local()

    def client(self):
        if not hasattr(self.clients, "value"):
            import sys
            sys.path.insert(0, str(common.shared.TOOLS))
            spec = importlib.util.spec_from_file_location("matched_api_" + str(threading.get_ident()),
                                                        common.shared.TOOLS / "experimental_inference.py")
            value = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(value)
            value.reserve_request, value.settle_request = self.reserve, self.settle
            self.clients.value = value
        return self.clients.value

    def reserve(self, name):
        with sqlite3.connect(self.out / "billing.sqlite", timeout=120) as db:
            db.execute("CREATE TABLE IF NOT EXISTS receipts (name TEXT PRIMARY KEY,status TEXT,cost_usd REAL)")
            db.execute("INSERT INTO receipts VALUES (?,?,?)", (name, "request_pending", None))

    def settle(self, name, row):
        cost = (row.get("usage") or {}).get("cost")
        with sqlite3.connect(self.out / "billing.sqlite", timeout=120) as db:
            db.execute("UPDATE receipts SET status=?,cost_usd=? WHERE name=?",
                       ("response_archived_known" if cost is not None else "response_archived_unknown", cost, name))

    def __call__(self, key, stage, system, payload, backend):
        validate_backend(backend)
        if backend != BACKEND:
            raise ValueError("The real adapter has no local or auxiliary fallback")
        request = PREFIX + sha256_bytes(str(self.out.resolve()).encode())[:10] + "_" + key
        path = self.out / "calls" / (key + ".json")
        input_sha = fingerprint({"stage": stage, "system": system, "input": payload, "backend": backend})
        if path.exists():
            prior = json.loads(path.read_text())
            if prior["input_sha256"] != input_sha:
                raise ValueError("Frozen call inputs changed")
            return prior["response"]
        response = {"status": "failed", "served_model": None, "source_requested": "wiki"}
        raw_path = common.shared.RAW / (request + ".json")
        try:
            raw = self.client().inference(request, system, json.dumps(payload, ensure_ascii=False), "low",
                                          model=MODEL, call_category="experiment")
            response.update(served_model=raw.get("model"), source_requested=raw.get("source_requested"),
                            usage=raw.get("usage"), finish_reason=raw.get("finish_reason"), raw_path=str(raw_path))
            if raw.get("finish_reason") != "stop":
                raise ValueError("Preserved non-stop response")
            parsed, parsing = parse_stage_response(raw["text"], allow_terminal_closure=True, return_receipt=True)
            response["parsing"] = parsing
            validate_creative_stage_schema(stage, parsed, payload)
            response.update(result=parsed, parsing=parsing, status="complete")
        except Exception as exc:
            if raw_path.exists():
                raw = json.loads(raw_path.read_text())
                response.update(served_model=raw.get("model"), source_requested=raw.get("source_requested"),
                                usage=raw.get("usage"), finish_reason=raw.get("finish_reason"), raw_path=str(raw_path))
            response.update(error_type=type(exc).__name__)
        save(path, {"input_sha256": input_sha, "request_identity": request, "response": response})
        return response


def run(out, report_path, *, workers=4, generate=None):
    """Start actual generation only after frozen protocol and evaluator checks."""
    out, report_path = Path(out), Path(report_path)
    protocol = json.loads((out / "protocol.json").read_text())
    if protocol.get("version") not in {VERSION, LEGACY_VERSION}:
        raise ValueError("Unexpected experimental protocol")
    validate_opportunity_layout(protocol)
    backend = protocol.get("backend", {})
    validate_backend(backend)
    if backend["transport"] == "local" and generate is None:
        raise ValueError("Local execution requires an explicitly supplied local writer callback; no API fallback")
    if protocol.get("runner_sha256") != sha256_bytes(Path(__file__).read_bytes()):
        raise ValueError("Runner changed after preparation")
    if protocol.get("prompts") != PROMPTS or protocol.get("theory") != common.THEORY:
        raise ValueError("Writing prompts changed after preparation")
    if backend["transport"] == "local" and protocol.get("local_writer_template_binding") != local_writer_template_contract():
        raise ValueError("The frozen local writer training input projection changed")
    official = official_provenance()
    if protocol.get("official") != official:
        raise ValueError("Official pinned prompt provenance changed")
    binding = validate_evaluator({"report_path": str(report_path), "report_sha256": sha256_bytes(report_path.read_bytes())})
    activation = {"evaluator": binding, "protocol_sha256": sha256_bytes((out / "protocol.json").read_bytes()),
                  "budget_authorization": "Author requested immediate parallel work, previously removed this task's budget cap; model restriction remains",
                  "paid_models": [MODEL] if backend["transport"] == "api" else [],
                  "backend": backend, "automatic_paid_retry": False}
    activation_path = out / "activation.json"
    if activation_path.exists() and json.loads(activation_path.read_text()) != activation:
        raise ValueError("The frozen evaluator activation cannot be changed during this stage")
    units = [validate_unit(row) for row in protocol["units"]]
    if (not units or len({u["id"] for u in units}) != len(units)
            or len({u["source"]["work_id"] for u in units}) != len(units)):
        raise ValueError("Frozen execution requires one predeclared draw per independent DEV work")
    if workers < 1:
        raise ValueError("Positive worker count required")
    if not activation_path.exists():
        save(activation_path, activation)
    generator = WikiGenerator(out) if generate is None else generate
    functions = pinned_official_functions()

    def run_unit(unit):
        path = out / "results" / (unit["id"] + ".json")
        if path.exists():
            prior = json.loads(path.read_text())
            if (prior.get("protocol_sha256") != activation["protocol_sha256"]
                    or prior.get("task_id") != unit["id"] or prior.get("work_id") != unit["source"]["work_id"]
                    or prior.get("evaluator_validation") != binding):
                raise ValueError("Completed result does not belong to this frozen protocol and evaluator")
            return prior
        result = execute_case(unit, generator, backend=backend, evaluator_validation=binding, official=functions,
                              shared_superwriter_start=protocol.get("shared_superwriter_start", False),
                              include_approach_update_ablation=protocol.get("include_approach_update_ablation", False),
                              comparison_conditions=protocol.get("comparison_conditions", ARMS if protocol.get("version") == LEGACY_VERSION else None))
        result["protocol_sha256"] = activation["protocol_sha256"]
        save(path, result)
        for arm, row in result["outputs"].items():
            save(out / "outputs" / (unit["id"] + "__" + arm + ".json"), row)
            if row["status"] == "complete":
                target = out / "readable" / (unit["id"] + "__" + arm + ".md")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(row["script"])
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_unit, units))
    arms = tuple(protocol["arms"])
    summary = {"version": protocol["version"], "completed_at_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "planned_works": len(units), "author_groups": len({u["author_group"] for u in units}),
               "planned_outputs": len(units) * len(arms),
               "complete_by_arm": {arm: sum(r["outputs"][arm]["status"] == "complete" for r in results) for arm in arms},
               "failure_by_arm": {arm: sum(r["outputs"][arm]["status"] != "complete" for r in results) for arm in arms},
               "actual_unique_billing": billing([call for row in results for call in row["calls"]]),
               "shared_superwriter_start": protocol.get("shared_superwriter_start", False),
               "include_approach_update_ablation": protocol.get("include_approach_update_ablation", False),
               "comparison_conditions": protocol.get("comparison_conditions", list(ARMS) if protocol.get("version") == LEGACY_VERSION else list(PRIMARY_COMPARISON_CONDITIONS)),
               "backend": backend,
               "shared_initial_billing": billing([call for row in results for call in row["calls"]
                                                  if call["request_identity"].endswith(("__shared_official_plan", "__shared_official_write"))]),
               "evaluator_validation": binding, "evaluation_performed": False,
               "quality_gain_established": False, "test_accessed": False,
               "budget_claim": "Matched opportunities only; all actual and virtual receipts remain inspectable"}
    save(out / "generation_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--evaluator-report", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shared-superwriter-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-approach-update-ablation", action="store_true")
    parser.add_argument("--comparison-conditions", nargs="+", choices=ARMS, default=list(PRIMARY_COMPARISON_CONDITIONS))
    parser.add_argument("--local-backend-json", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.sources is None:
            parser.error("--sources is required for source-only preparation")
        backend = json.loads(args.local_backend_json.read_text()) if args.local_backend_json else BACKEND
        result = prepare(args.sources, args.out, shared_superwriter_start=args.shared_superwriter_start,
                         backend=backend, include_approach_update_ablation=args.include_approach_update_ablation,
                         comparison_conditions=args.comparison_conditions)
        print(json.dumps({"prepared_dev_works": len(result["units"]), "new_api_calls": 0,
                          "run_requires_actual_validated_report": True}, ensure_ascii=False))
    else:
        if args.evaluator_report is None:
            parser.error("--evaluator-report is required before generation")
        print(json.dumps(run(args.out, args.evaluator_report, workers=args.workers), ensure_ascii=False))


if __name__ == "__main__":
    main()
