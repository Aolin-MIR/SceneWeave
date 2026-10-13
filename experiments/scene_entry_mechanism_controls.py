"""Matched scene-entry mechanism controls with injected, auditable callbacks.

This module makes no inference calls itself. Content hashes bind supplied
records; they are not evidence of a served model, human judgment or quality.
Controls are separate development/evaluation artifacts and are never TRAIN.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import adaptation_entry_algorithms as alg
from adaptation_system_contract import fingerprint, validate_source


VERSION = "scene-entry-mechanism-controls-v1"
CONDITIONS = {"actual_scene_trials", "detailed_scene_plans", "updater_pretrial_intentions"}
PLAN_FIELDS = ("situation", "interaction", "psychological_tensions", "rhythm",
               "preceding_preparation", "subsequent_development", "whole_work_meaning")
PLAN_PROMPT = (
    "Develop a detailed plan for only the assigned surviving scene of the "
    "already executed edit. Use the source, actual brief, common initial "
    "writing, candidate program and preceding scene plans in this alternative. "
    "Return JSON {scene_id:string,representation:'detailed_scene_plan',plan:{"
    "situation:string,interaction:string,psychological_tensions:string,rhythm:string,"
    "preceding_preparation:string,subsequent_development:string,whole_work_meaning:string}}. "
    "Describe concrete approaches, replies or evasions, contradictory possible "
    "motives, timing and transitions, and implications for adjoining scenes "
    "and the whole adaptation. Make this substantial enough to guide complete "
    "writing. Describe possible spoken exchanges without writing their final "
    "dialogue or a finished screenplay scene. These are intended effects, "
    "not observed achievements. No channel, conflict, restraint, source "
    "retention or change has an automatic advantage. Only the actual brief "
    "binds the adaptation; generated treatments remain revisable."
)
REPRESENTATION_GUIDANCE = (
    " Mechanism-control material labels govern how exploratory material may "
    "be interpreted: actual_screenplay is completed trial prose; "
    "detailed_scene_plan is a detailed prospective plan, not a screenplay "
    "or an observed effect; intended_treatment contains only pretrial "
    "decisions and anticipated gains/losses. Where a label conflicts with "
    "a generic mention of actual exploratory prose, follow the label. "
    "Use, transform, combine or reject any available material, including "
    "the common actual initial draft, without assuming that any "
    "representation has better quality. Return the existing stage schema."
)


class _PrefixReady(Exception):
    pass


def _raw_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _code_sha():
    return hashlib.sha256(Path(alg.__file__).read_bytes()).hexdigest()


def _available(value):
    # A fixed allowlist prevents an extra feedback/result field from bypassing
    # the deliberately restricted updater view.
    if not isinstance(value, dict) or set(value) - {
            "source", "initial_plan", "initial_script", "development_entry"}:
        raise ValueError("Controls accept source and common initialization only")
    validate_source(value.get("source", {}))
    alg.text(value["source"], "source_text")
    alg.text(value, "initial_plan")
    alg.text(value, "initial_script")
    if value.get("development_entry", "scene_first") != "scene_first":
        raise ValueError("These controls apply to the scene development entry")
    return deepcopy(value)


def _bindings(available, prompts, parameters):
    return {"source_sha256": fingerprint(available["source"]),
            "source_text_sha256": _raw_sha(available["source"]["source_text"]),
            "brief_sha256": fingerprint(available["source"]["brief"]),
            "initial_plan_sha256": fingerprint(available["initial_plan"]),
            "initial_script_sha256": _raw_sha(available["initial_script"]),
            "available_sha256": fingerprint(available), "prompts_sha256": fingerprint(prompts),
            "parameters_sha256": fingerprint(parameters), "controller_code_sha256": _code_sha(),
            "controller_version": alg.MATERIAL_ENTRY_VERSION}


def _request(key, phase, prompt, payload):
    return {"key": key, "phase": phase, "prompt": prompt, "input": deepcopy(payload)}


def _record(request, output):
    if not isinstance(output, dict):
        raise ValueError("Callbacks must return an explicit JSON object")
    return {"request": deepcopy(request), "output": deepcopy(output),
            "request_sha256": fingerprint(request), "output_sha256": fingerprint(output),
            "record_kind": "callback_content_binding", "proves_served_model": False}


def _check_record(record):
    if (record.get("record_kind") != "callback_content_binding"
            or record.get("proves_served_model") is not False
            or record.get("request_sha256") != fingerprint(record.get("request"))
            or record.get("output_sha256") != fingerprint(record.get("output"))
            or not isinstance(record.get("output"), dict)):
        raise ValueError("Frozen callback content binding changed")


def _capture_prefix(available, prompts, call):
    records, boundary = [], {}

    def capture(key, phase, prompt, payload):
        request = _request(key, phase, prompt, payload)
        if key == "program_edit":
            boundary.update(request)
            raise _PrefixReady()
        if key != "scene_edit_proposals" and not key.startswith("scene_trial_"):
            raise ValueError("Unexpected stage before the frozen updater boundary")
        value = call(key, phase, prompt, deepcopy(payload))
        if key.startswith("scene_trial_") and (not isinstance(value, dict)
                or value.get("representation", "actual_screenplay") != "actual_screenplay"
                or value.get("trial_id", payload["trial_id"]) != payload["trial_id"]):
            raise ValueError("The common trial record mislabels its representation or identifier")
        records.append(_record(request, value))
        return deepcopy(value)

    try:
        alg.execute_scene_complete(available, capture, prompts, material_development=True)
    except _PrefixReady:
        pass
    if not boundary:
        raise ValueError("The prefix did not reach the complete updater boundary")
    return records, boundary


def _target_inventory(proposal):
    program = proposal["program"]
    result = []
    for index, edit in enumerate(proposal["edits"]):
        patch = alg._patch(program, edit["operations"])
        candidate = alg.apply_program_patch(program, patch)
        ids = alg.derive_surviving_edit_targets(program, candidate, patch,
                                               proposal["scene_id"], proposal["dependencies"])
        result.append({"trial_id": str(index), "executed_patch": patch,
                       "executed_program": candidate, "surviving_target_scene_ids": ids})
    return result


def freeze_scene_exploration(available, prompts, call, *, parameters=None):
    """Capture the real common proposal/trials through the updater boundary.

    The caller owns inference authorization and authentic backend receipts.
    This function records callback content only, and stops before any updater
    or complete-writing call. Existing outputs can instead be imported with
    freeze_scene_exploration_from_records without inference.
    """
    available = _available(available)
    prompts, parameters = deepcopy(prompts), deepcopy({} if parameters is None else parameters)
    if not isinstance(parameters, dict):
        raise ValueError("Frozen inference parameters must be an explicit object")
    records, boundary = _capture_prefix(available, prompts, call)
    proposal = deepcopy(records[0]["output"])
    bank = {"version": VERSION, "controller_version": alg.MATERIAL_ENTRY_VERSION,
            "available": available, "prompts": prompts, "parameters": parameters,
            "bindings": _bindings(available, prompts, parameters), "prefix_records": records,
            "proposal": proposal, "target_inventory": _target_inventory(proposal),
            "actual_trials": deepcopy(boundary["input"]["explorations"]),
            "updater_boundary": deepcopy(boundary),
            "training_eligible": False, "judgments_are_human": False,
            "quality_gain_established": False, "backend_receipts_certified": False}
    bank["bank_sha256"] = fingerprint(bank)
    return bank


def freeze_scene_exploration_from_records(available, prompts, records, *, parameters=None):
    """Import recorded callback content after exact controller replay validation."""
    records, cursor = deepcopy(records), 0

    def replay(key, phase, prompt, payload):
        nonlocal cursor
        if cursor >= len(records):
            raise ValueError("Missing frozen proposal or surviving-target trial")
        record = records[cursor]
        _check_record(record)
        if record["request"] != _request(key, phase, prompt, payload):
            raise ValueError("Frozen prefix request disagrees with the controller")
        cursor += 1
        return deepcopy(record["output"])

    bank = freeze_scene_exploration(available, prompts, replay, parameters=parameters)
    if cursor != len(records):
        raise ValueError("Extra records are outside the common exploration prefix")
    return bank


def validate_frozen_scene_exploration(bank, *, expected_bank_sha256):
    """Bind the bank to a separately retained seal and deterministic replay."""
    if not isinstance(bank, dict) or bank.get("version") != VERSION:
        raise ValueError("Unknown mechanism-control bank")
    unsealed = {key: value for key, value in bank.items() if key != "bank_sha256"}
    if bank.get("bank_sha256") != expected_bank_sha256 or fingerprint(unsealed) != expected_bank_sha256:
        raise ValueError("The frozen bank seal changed")
    available = _available(bank["available"])
    if bank["bindings"] != _bindings(available, bank["prompts"], bank["parameters"]):
        raise ValueError("Frozen source, initialization, parameters or controller changed")
    reconstructed = freeze_scene_exploration_from_records(
        available, bank["prompts"], bank["prefix_records"], parameters=bank["parameters"])
    if reconstructed != bank:
        raise ValueError("The bank disagrees with executed edits or actual surviving targets")
    return deepcopy(bank)


def _intention_view(bank):
    """Construct from pretrial fields, never by deleting keys from actual prose."""
    proposal = bank["proposal"]
    materials = []
    for inventory, edit in zip(bank["target_inventory"], proposal["edits"]):
        candidate = inventory["executed_program"]
        identifiers = inventory["surviving_target_scene_ids"]
        scenes = [{"scene_id": row["scene_id"], "intended_development": row["development"]}
                  for row in candidate["scenes"] if row["scene_id"] in identifiers]
        materials.append({"trial_id": inventory["trial_id"], "representation": "intended_treatment",
            "executed_patch": deepcopy(inventory["executed_patch"]),
            "executed_program": deepcopy(candidate), "intended_scene_components": scenes,
            "surviving_target_scene_ids": deepcopy(identifiers),
            "intended_gain": alg.explicit_edit_annotation(edit, "intended_gain"),
            "possible_loss": alg.explicit_edit_annotation(edit, "possible_loss")})
    # The review scope here is computable before writing. It contains only
    # surviving identifiers and declared pretrial dependency hypotheses.
    scope = bank["updater_boundary"]["input"]["review_scene_ids"]
    payload = {**deepcopy(bank["available"]), "program": deepcopy(proposal["program"]),
               "explorations": materials, "declared_dependencies": deepcopy(proposal["dependencies"]),
               "review_scene_ids": deepcopy(scope),
               "retained_initial_material": alg.initial_material_reference(bank["available"]["initial_script"])}
    return alg.compile_material_roles(payload)


def _plan_output(value, identifier):
    if (not isinstance(value, dict) or set(value) != {"scene_id", "representation", "plan"}
            or value.get("scene_id") != identifier or value.get("representation") != "detailed_scene_plan"
            or not isinstance(value.get("plan"), dict) or set(value["plan"]) != set(PLAN_FIELDS)):
        raise ValueError("Detailed scene planning must be explicitly labeled and complete")
    for field in PLAN_FIELDS:
        alg.text(value["plan"], field)
    # Preserve the actual returned planning content without summarizing it.
    prose = "\n\n".join(field + ": " + value["plan"][field] for field in PLAN_FIELDS)
    return {"scene_id": identifier, "representation": "detailed_scene_plan",
            "text": prose, "scene_plan": deepcopy(value["plan"])}


def _honest_plan_labels(value):
    """Rename inherited prose-component keys in explicit planning controls only."""
    aliases = {"actual_trial_scope": "planning_trial_scope", "actual_prefix": "planning_prefix",
               "preceding_scenes": "preceding_scene_plans", "actual_scene_components": "planned_scene_components",
               "realized_scene_ids": "planned_scene_ids"}
    if isinstance(value, list):
        return [_honest_plan_labels(row) for row in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    if value.get("representation") == "common_initial_complete_script":
        # The actual shared initial draft stays an exact field reference.
        # Its empty component registry does not describe prospective plans.
        return deepcopy(value)
    result = {aliases.get(key, key): _honest_plan_labels(row) for key, row in value.items()}
    if result.get("representation") == "detailed_scene_plan":
        for row in result.get("ordered_scenes", result.get("planned_scene_components", [])):
            row["representation"] = "detailed_scene_plan"
    return result


def _label_control_payload(payload):
    """Correct generic native prose roles using explicit representation labels."""
    result = deepcopy(payload)
    provenance = result.get("material_provenance", {})
    roles = provenance.get("field_roles", {})
    containers = [("explorations", result.get("explorations"))]
    context = result.get("development_context")
    if isinstance(context, dict):
        containers.append(("development_context.explorations", context.get("explorations")))
    for path, materials in containers:
        if not isinstance(materials, list):
            continue
        if any(row.get("representation") != "actual_screenplay" for row in materials):
            roles[path] = "explicitly labeled exploratory material; plans and intentions are not realized scenes"
        for index, row in enumerate(materials):
            if row.get("representation") == "detailed_scene_plan":
                roles[path + "[" + str(index) + "].text"] = "detailed prospective scene planning; not screenplay prose or observed benefit"
            elif row.get("representation") == "intended_treatment":
                roles[path + "[" + str(index) + "]"] = "pretrial intentions only; no actual text or posttrial interpretations"
    has_plans = any("detailed_scene_plan" == row.get("representation")
                    for _, materials in containers for row in (materials or []))
    working_containers = [("scene_working_material", result.get("scene_working_material"))]
    if isinstance(context, dict):
        working_containers.append(("development_context.scene_working_material", context.get("scene_working_material")))
    for path, tasks in working_containers:
        if not isinstance(tasks, list):
            continue
        for index, task in enumerate(tasks):
            for draft_index, draft in enumerate(task.get("starting_drafts", [])):
                if draft.get("representation") == "detailed_scene_plan":
                    has_plans = True
                    roles[path + "[" + str(index) + "].starting_drafts[" + str(draft_index) + "].text"] = (
                        "detailed prospective scene planning; not screenplay prose or observed benefit")
    if has_plans:
        for path in list(roles):
            if path.endswith("scene_working_material"):
                roles[path] = "designated revisable material; each representation distinguishes plans from completed prose"
    return result


def run_scene_mechanism_control(bank, condition, call, *, expected_bank_sha256, request_namespace):
    """Run one matched condition through whole realization/read/revise/check.

    Actual and intentions conditions reuse exactly the frozen trial pool. The
    planning condition reuses the proposal, edits and surviving targets and
    requests a detailed plan for each same target. Only the updater loses
    actual-trial access in the intentions condition; subsequent stages regain
    the unchanged actual pool through the native designated-material compiler.
    """
    if condition not in CONDITIONS:
        raise ValueError("Unknown scene mechanism control")
    if not isinstance(request_namespace, str) or not request_namespace.strip():
        raise ValueError("Controls require a distinct explicit request namespace")
    bank = validate_frozen_scene_exploration(bank, expected_bank_sha256=expected_bank_sha256)
    prefix, cursor, ledger = bank["prefix_records"], 0, []

    def controlled(key, phase, prompt, payload):
        nonlocal cursor
        native_request = _request(key, phase, prompt, payload)
        identity = request_namespace + ":" + condition + ":" + key
        if key == "scene_edit_proposals" or key.startswith("scene_trial_"):
            if cursor >= len(prefix):
                raise ValueError("Control requested an unfrozen exploration target")
            record = prefix[cursor]
            cursor += 1
            if key == "scene_edit_proposals" or condition != "detailed_scene_plans":
                if native_request != record["request"]:
                    raise ValueError("Control changed the common proposal or actual-trial request")
                ledger.append({"key": key, "request_identity": identity, "kind": "frozen_content_reuse",
                    "source_request_sha256": record["request_sha256"],
                    "output_sha256": record["output_sha256"], "new_inference_claimed": False})
                return deepcopy(record["output"])
            # Preceding trial text changes representation, while the actual
            # candidate program, executed edit, target and scope stay fixed.
            for field in ("source", "initial_plan", "initial_script", "program", "scene_id",
                          "exploration", "trial_id", "executed_patch", "review_scene_ids"):
                if payload.get(field) != record["request"]["input"].get(field):
                    raise ValueError("Planning changed a frozen edit, initialization or target")
            delivered = _honest_plan_labels(payload)
            delivered["requested_representation"] = "detailed_scene_plan"
            delivered["scope_instruction"] = "Plan only this assigned surviving scene; do not write completed screenplay dialogue."
            request = _request(identity, phase, PLAN_PROMPT, delivered)
            raw = call(identity, phase, PLAN_PROMPT, deepcopy(delivered))
            output = _plan_output(raw, payload["scene_id"])
            ledger.append({"key": key, "request_identity": identity, "kind": "new_callback_content",
                           **_record(request, raw), "compiled_output_sha256": fingerprint(output)})
            return output
        if cursor != len(prefix):
            raise ValueError("Control omitted a frozen surviving-target scene")
        delivered = (_intention_view(bank) if key == "program_edit" and condition == "updater_pretrial_intentions"
                     else deepcopy(payload))
        if condition == "detailed_scene_plans":
            delivered = _honest_plan_labels(delivered)
        delivered = _label_control_payload(delivered)
        instruction = prompt + REPRESENTATION_GUIDANCE
        request = _request(identity, phase, instruction, delivered)
        output = call(identity, phase, instruction, deepcopy(delivered))
        ledger.append({"key": key, "request_identity": identity, "kind": "new_callback_content",
                       **_record(request, output)})
        return deepcopy(output)

    native = alg.execute_scene_complete(bank["available"], controlled, bank["prompts"],
        material_development=True, trial_representation=("detailed_scene_plan"
            if condition == "detailed_scene_plans" else "actual_screenplay"))
    if condition == "detailed_scene_plans":
        native = _honest_plan_labels(native)
    native["base_writer_workflow_version"] = native["writer_workflow_version"]
    native["writer_workflow_version"] = VERSION + ":" + condition
    native["control_condition"] = condition
    native["training_eligible"] = False
    realization = next(row["request"]["input"] for row in ledger if row["key"] == "realize_complete")
    designated = {"scene_working_material_sha256": fingerprint(realization["scene_working_material"]),
        "designated_explorations_sha256": fingerprint(realization["explorations"]),
        "scene_selections": [{"scene_id": row["scene_id"],
            "starting_material_ids": deepcopy(row["starting_material_ids"]),
            "draft_references": [{"trial_id": draft["trial_id"],
                "representation": draft["representation"],
                "original_scene_ids": deepcopy(draft["original_scene_ids"]),
                "source_material_sha256": draft["source_material_sha256"]}
                for draft in row["starting_drafts"]]} for row in realization["scene_working_material"]],
        "shared_bank_does_not_require_identical_designated_subsets": True}
    return {"status": "complete", "control_version": VERSION, "control_condition": condition,
            "request_namespace": request_namespace, "frozen_bank_sha256": expected_bank_sha256,
            "bindings": deepcopy(bank["bindings"]), "target_inventory": deepcopy(bank["target_inventory"]),
            "actual_trial_pool_sha256": fingerprint(bank["actual_trials"]),
            "supplied_trial_pool_sha256": fingerprint(native["explorations"]),
            "realization_designated_material": designated,
            "native_result": native, "callback_records": ledger,
            "training_eligible": False, "judgments_are_human": False, "formal_result_established": False,
            "quality_gain_established": False, "backend_receipts_certified": False}
