"""Explicit model interfaces and supervision admission for scene-first adaptation.

These adapters make no inference calls. They do not certify dramatic quality,
semantic consistency, or the completeness of source passages.
"""
import hashlib
import json
import math


EDIT_AXES = {"plot", "character_actions", "relationships", "ending",
             "background", "props", "scene_organization", "expression"}
WRITER_TASKS = {"plan", "trial", "revise_program", "realize", "revise_script"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def validate_source(source):
    for key in ("work_id", "source_text", "source_revision", "brief"):
        if not source.get(key):
            raise ValueError("Missing source field: " + key)
    if not isinstance(source["brief"], dict):
        raise ValueError("The adaptation brief must be structured")
    return {key: source[key] for key in
            ("work_id", "source_text", "source_revision", "brief")}


def validate_program(program):
    if set(program.get("choices", {})) != EDIT_AXES:
        raise ValueError("The program must cover all authorized edit axes")
    scenes = program.get("scenes", [])
    if not scenes or any(not s.get("scene_id") or not s.get("development") for s in scenes):
        raise ValueError("Scenes require stable identifiers and concrete development")
    ids = [s["scene_id"] for s in scenes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scene identifiers")
    if not program.get("intent"):
        raise ValueError("Missing adaptation intent")
    return program


def writer_input(task, source, *, program=None, trials=None, exploration=None,
                 draft=None, revision_request=None):
    """Project only inference-available fields into the writing model input."""
    if task not in WRITER_TASKS:
        raise ValueError("Unknown writing task")
    result = {"task": task, "source": validate_source(source)}
    if task != "plan":
        result["program"] = validate_program(program or {})
    if task == "trial":
        if not exploration or exploration.get("scene_id") not in {
                s["scene_id"] for s in program["scenes"]}:
            raise ValueError("Trial must address a scene in this program")
        if not exploration.get("approach"):
            raise ValueError("Missing trial approach")
        result["exploration"] = exploration
    if task in {"revise_program", "realize"}:
        result["trials"] = []
        ids = set()
        for trial in trials or []:
            if trial.get("source_sha256") != fingerprint(result["source"]):
                raise ValueError("Trial belongs to a different source or brief")
            if trial.get("program_sha256") != fingerprint(program):
                raise ValueError("Trial belongs to a different program revision")
            if trial.get("scene_id") not in {s["scene_id"] for s in program["scenes"]}:
                raise ValueError("Trial refers to a missing scene")
            if not trial.get("trial_id") or trial["trial_id"] in ids or not trial.get("text"):
                raise ValueError("Missing text or duplicate trial identifier")
            ids.add(trial["trial_id"])
            result["trials"].append(trial)
        if task == "revise_program" and len(result["trials"]) < 2:
            raise ValueError("Scene-first development needs two actual scene trials")
    if task == "revise_script":
        if not draft or not revision_request:
            raise ValueError("Script revision requires a draft and an explicit request")
        result.update(draft=draft, revision_request=revision_request)
    return result


def bind_trial(model_output, trial_id, trial_input):
    if trial_input.get("task") != "trial" or not model_output.get("text"):
        raise ValueError("Expected an actual scene trial")
    return {"trial_id": trial_id, "scene_id": trial_input["exploration"]["scene_id"],
            "text": model_output["text"],
            "source_sha256": fingerprint(trial_input["source"]),
            "program_sha256": fingerprint(trial_input["program"])}


def realization_input(source, revision, original_program, trials):
    """Attach selected trial text without claiming it belongs to the revised plan."""
    checked = writer_input("revise_program", source, program=original_program, trials=trials)
    revised = validate_program(revision["program"])
    selected = revision.get("selected_trial_id")
    if selected is not None and selected not in {t["trial_id"] for t in checked["trials"]}:
        raise ValueError("Selected trial does not exist")
    if "dependency_changes" not in revision:
        raise ValueError("Record character, preparation, aftermath and scene-organization changes")
    result = writer_input("realize", source, program=revised)
    result["selected_trial_material"] = next(
        (t for t in checked["trials"] if t["trial_id"] == selected), None)
    result["trial_material_role"] = "Earlier exploration; adapt to the revised program"
    return result


def reward_input(script):
    if not isinstance(script, str) or not script.strip():
        raise ValueError("Reward inference requires complete draft text")
    return {"text": script}


def admission_errors(row, role):
    """Reject supervision that cannot support the requested learning task."""
    errors = []
    if row.get("split") != "train":
        errors.append("Only released TRAIN records may enter training")
    for gate in ("training_ready", "family_split_verified", "version_verified"):
        if row.get(gate) is not True:
            errors.append("Pending gate: " + gate)
    if role == "scene_response_auxiliary":
        if not row.get("input", {}).get("ordered_scenes") or not row.get("target"):
            errors.append("Requires scene context and an actual response")
        if row.get("label_origin") != "human_response":
            errors.append("Response must come from an actual reader")
    elif role == "writer_revision_sft":
        target = row.get("target")
        if not row.get("input", {}).get("draft") or not isinstance(target, dict) or not target.get("revised_text"):
            errors.append("Requires an actual before/after writing pair")
        if row.get("label_origin") != "human_revision":
            errors.append("Synthetic revisions need a separate declared training stage")
    elif role == "human_preference_reward":
        if not row.get("left_text") or not row.get("right_text"):
            errors.append("Requires two actual candidate drafts")
        p = row.get("preference_left")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            errors.append("Requires an observed human preference probability")
        if row.get("label_origin") != "human_preference":
            errors.append("Comment sentiment and film stars are not draft preferences")
    else:
        errors.append("Unknown training role")
    return errors


def audit_scene_responses(rows):
    roles = ("scene_response_auxiliary", "writer_revision_sft", "human_preference_reward")
    return {"records": len(rows), "candidate_use": "Contextual human-response auxiliary learning",
            "training_admitted": {role: sum(not admission_errors(row, role) for row in rows)
                                  for role in roles},
            "no_data_modified": True,
            "limitation": "Admission guards do not establish artistic benefit or verify provenance"}
