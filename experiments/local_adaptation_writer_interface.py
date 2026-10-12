"""Project compatible creative stages onto the released local writer tasks.

This adapter does not load weights, change training, judge screenplay quality,
or supply missing planner/updater supervision. It never invokes a remote API.
"""
from copy import deepcopy
import json
import re

from adaptation_system_contract import fingerprint, validate_program
import run_scene_first_matched as matched


VERSION = "local-creative-stage-projection-v4"
SERIALIZATION_MODES = ("legacy", "compact_json")
COMPACT_SERIALIZATION_VERSION = "local-request-compact-json-v1"
STAGES = ("trial", "realize", "revise_script")
PROGRAM_BOOKKEEPING = ("original_program", "program_patch", "native_execution", "executed_patch",
                       "executed_program", "final_integrity_check")
SCENE_ROLE_FIELDS = ("actual_scene", "source", "source.brief", "initial_plan", "initial_script",
                     "actual_prefix", "program", "scenic_treatment", "expression_units",
                     "previous_expression_observations", "scene_reading", "expression_alternative",
                     "converted_scene_sequence", "actual_following_scenes")
REFERENCE_REQUEST_VERSION = "native-entries-material-development-v11"
REFERENCE_REQUEST_VERSIONS = (REFERENCE_REQUEST_VERSION, "native-entries-material-development-v11.1",
                              "native-entries-material-development-v11.2", "native-entries-material-development-v12",
                              "native-entries-material-development-v12.1", "native-entries-material-development-v12.2",
                              "native-entries-material-development-v12.3", "native-entries-material-development-v12.3.1")
TEXT_REFERENCE_KEY = "$text_ref"
TEXT_REFERENCE_MINIMUM_CHARACTERS = 256
REFERENCE_READING_INSTRUCTION = (
    "Text objects with $text_ref are lossless references to the string at their JSON pointer path; "
    "read that exact same text wherever referenced. Their SHA verifies text identity.")


def validate_reference_request(request):
    """Admit only the explicit v11 registry and check its actual original text."""
    metadata = request.get("material_provenance", {})
    version = metadata.get("request_version") if isinstance(metadata, dict) else None
    context = request.get("development_context", {})
    has_registry = ("retained_initial_material" in request or
                    isinstance(context, dict) and "retained_initial_material" in context)
    if version is None and not has_registry:
        return False
    if version not in REFERENCE_REQUEST_VERSIONS or not has_registry:
        raise ValueError("Unknown or incomplete explicit initial-material request version")
    from adaptation_entry_algorithms import validate_initial_material_references
    validate_initial_material_references(request)
    return True


def text_pointer(path):
    """Encode field locations as standard escaped JSON pointers."""
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in path)


def expand_text_references(payload):
    """Resolve local exact-text references, rejecting malformed or cyclic inputs."""
    active, cached = set(), {}

    def locate(pointer):
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise ValueError("Unknown text reference path")
        value = payload
        for raw in pointer[1:].split("/"):
            if re.search(r"~(?![01])", raw):
                raise ValueError("Unknown text reference path escape")
            token = raw.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict) and token in value:
                value = value[token]
            elif isinstance(value, list) and token.isdigit() and str(int(token)) == token and int(token) < len(value):
                value = value[int(token)]
            else:
                raise ValueError("Unknown text reference path")
        return value

    def expand(value, path):
        if isinstance(value, dict):
            if TEXT_REFERENCE_KEY in value:
                reference = value[TEXT_REFERENCE_KEY]
                if (set(value) != {TEXT_REFERENCE_KEY} or not isinstance(reference, dict)
                        or set(reference) != {"path", "sha256"}
                        or not isinstance(reference["sha256"], str)
                        or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])):
                    raise ValueError("Malformed exact-text reference")
                pointer = reference["path"]
                if not isinstance(pointer, str):
                    raise ValueError("Unknown text reference path")
                if pointer in active:
                    raise ValueError("Cyclic text reference")
                if pointer not in cached:
                    active.add(pointer)
                    cached[pointer] = expand(locate(pointer), ())
                    active.remove(pointer)
                target = cached[pointer]
                if not isinstance(target, str) or fingerprint(target) != reference["sha256"]:
                    raise ValueError("Text reference SHA differs from its actual string")
                return target
            return {key: expand(item, (*path, key)) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item, (*path, index)) for index, item in enumerate(value)]
        return value

    return expand(payload, ())


def encode_text_references(payload):
    """Keep protected material literal and deduplicate exact repeated long strings."""
    retained = {}
    current = payload.get("material_provenance", {}).get("current_complete_script_field")

    def protected(path):
        return bool(path) and ("source" in path or "initial_script" in path or path == (current,))

    def visit(value, path=()):
        if isinstance(value, dict):
            if TEXT_REFERENCE_KEY in value:
                raise ValueError("Original material contains a reserved text-reference field")
            for key, item in value.items():
                yield from visit(item, (*path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from visit(item, (*path, index))
        elif isinstance(value, str) and len(value) >= TEXT_REFERENCE_MINIMUM_CHARACTERS:
            yield path, value

    for path, value in visit(payload):
        if protected(path):
            retained.setdefault(value, text_pointer(path))

    def encode(value, path=()):
        if isinstance(value, dict):
            return {key: encode(item, (*path, key)) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item, (*path, index)) for index, item in enumerate(value)]
        if isinstance(value, str) and len(value) >= TEXT_REFERENCE_MINIMUM_CHARACTERS:
            if protected(path):
                return value
            if value in retained:
                return {TEXT_REFERENCE_KEY: {"path": retained[value], "sha256": fingerprint(value)}}
            retained[value] = text_pointer(path)
        return value

    encoded = encode(payload)
    restored = expand_text_references(encoded)
    if json.dumps(restored, ensure_ascii=False) != json.dumps(payload, ensure_ascii=False):
        raise ValueError("Exact-text reference expansion changed the original writing input")
    return encoded


def material_path_exists(request, path):
    """Resolve a compiler field path against the actual projected object."""
    if not isinstance(path, str) or not re.fullmatch(r"\w+(?:\[\d+\])?(?:\.\w+(?:\[\d+\])?)*", path):
        return False
    value = request
    for name, index in re.findall(r"(\w+)(?:\[(\d+)\])?", path):
        if not isinstance(value, dict) or name not in value:
            return False
        value = value[name]
        if index:
            if not isinstance(value, list) or int(index) >= len(value):
                return False
            value = value[int(index)]
    return True


def project_material_roles(request, renames):
    """Carry existing roles through real renames and omit removed bookkeeping."""
    if "material_provenance" not in request:
        return request
    metadata = request["material_provenance"]
    if not isinstance(metadata, dict) or not isinstance(metadata.get("field_roles"), dict):
        raise ValueError("Material provenance must contain explicit field roles")

    def renamed(path):
        for old, new in renames.items():
            if path == old or path.startswith(old + ".") or path.startswith(old + "["):
                return new + path[len(old):]
        return path

    roles = {}
    for path, role in metadata["field_roles"].items():
        actual = renamed(path)
        if material_path_exists(request, actual):
            if actual in roles and roles[actual] != role:
                raise ValueError("Conflicting projected material roles")
            roles[actual] = role
    metadata["field_roles"] = roles
    current = metadata.get("current_complete_script_field")
    if current is not None:
        current = renamed(current)
        if not isinstance(request.get(current), str) or not request[current].strip():
            raise ValueError("Projected current complete script field is absent")
        metadata["current_complete_script_field"] = current
        if "compatible_local_current_script_alias" in metadata:
            metadata["compatible_local_current_script_alias"] = current
    metadata["local_projection_version"] = VERSION
    ordered = {current: request[current]} if current is not None else {}
    ordered["material_provenance"] = metadata
    ordered.update({key: value for key, value in request.items() if key not in ordered})
    return ordered


def validate_scene_roles(request):
    """Validate the explicit current single-scene contract separately from a script."""
    metadata = request.get("scene_material_roles")
    if (not isinstance(metadata, dict) or set(metadata) != {"current_scene_field", "current_scene_scope", "field_roles"}
            or metadata.get("current_scene_field") != "actual_scene"
            or metadata.get("current_scene_scope") != "one_complete_assigned_scene"
            or not isinstance(metadata.get("field_roles"), dict)
            or "actual_scene" not in metadata["field_roles"]):
        raise ValueError("Unknown or malformed current single-scene role contract")
    for path, role in metadata["field_roles"].items():
        if path not in SCENE_ROLE_FIELDS or not isinstance(role, str) or not role.strip():
            raise ValueError("Unknown single-scene material role")
    scene = request.get("actual_scene")
    if (not isinstance(scene, dict) or not isinstance(scene.get("text"), str) or not scene["text"].strip()
            or not isinstance(scene.get("scene_id"), str) or not scene["scene_id"].strip()
            or request.get("scene_id", scene["scene_id"]) != scene["scene_id"]):
        raise ValueError("Actual current single-scene text or identifier is unavailable")
    complete_roles = request.get("material_provenance", {})
    if not isinstance(complete_roles, dict) or complete_roles.get("current_complete_script_field") is not None:
        raise ValueError("Current single-scene material cannot be labeled a current complete script")


def project_scene_roles(request):
    """Retain true short-scene roles and actual source without creating an alias."""
    if "scene_material_roles" not in request:
        return request
    validate_scene_roles(request)
    roles = request["scene_material_roles"]["field_roles"]
    request["scene_material_roles"]["field_roles"] = {
        path: role for path, role in roles.items() if material_path_exists(request, path)}
    ordered = {"actual_scene": request["actual_scene"], "source": request["source"]}
    ordered.update({key: value for key, value in request.items() if key not in ordered})
    return ordered


def serialize_request(payload, *, mode="legacy"):
    """Preserve historical sorting and current-first native material serialization."""
    if mode not in SERIALIZATION_MODES:
        raise ValueError("Unknown local writing request serialization")
    options = {"separators": (",", ":")} if mode == "compact_json" else {}
    uses_references = validate_reference_request(payload)
    if "scene_material_roles" in payload:
        validate_scene_roles(payload)
        if not isinstance(payload.get("source"), dict):
            raise ValueError("A source-conditioned current single scene is required")
        ordered = {"actual_scene": payload["actual_scene"], "source": payload["source"]}
        ordered.update({key: value for key, value in payload.items() if key not in ordered})
        return json.dumps(ordered, ensure_ascii=False, **options)
    if "material_provenance" not in payload:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, **options)
    metadata = payload["material_provenance"]
    if not isinstance(metadata, dict):
        raise ValueError("Material provenance must be an object")
    current = metadata.get("current_complete_script_field")
    if current is not None and (not isinstance(payload.get(current), str) or not payload[current].strip()):
        raise ValueError("Serialized current complete script field is absent")
    ordered = {current: payload[current]} if current is not None else {}
    ordered["material_provenance"] = metadata
    ordered.update({key: value for key, value in payload.items() if key not in ordered})
    if uses_references:
        ordered = encode_text_references(ordered)
    return json.dumps(ordered, ensure_ascii=False, **options)


def bind_scene_hybrid(infer, local_backend, *, serialization="legacy"):
    """Explicitly assign every role; unsupported roles cannot inherit local."""
    return bind_entry_hybrid("scene_first", infer, local_backend, serialization=serialization)


def bind_entry_hybrid(entry, infer, local_backend, *, serialization="legacy"):
    """Bind creative writing locally and route-state interpretation explicitly to API."""
    if entry not in ("scene_first", "relationships_first", "structure_first", "expression_first"):
        raise ValueError("Unknown adaptation entry")
    matched.validate_backend(local_backend)
    if local_backend["transport"] != "local":
        raise ValueError("An explicit local-only writer checkpoint is required")
    phases = ("official", "develop", "explore", "revise_program", "realize", "review", "final")
    creative = {"explore", "realize", "final"}
    adapter = CompatibleLocalWriterGenerator(infer, local_backend, serialization=serialization)
    assignments = {phase: deepcopy(local_backend if phase in creative else matched.BACKEND)
                   for phase in phases}
    callbacks = {phase: adapter for phase in creative}
    return assignments, callbacks


def project_request(stage, system, payload):
    """Rename existing material without inventing choices or target labels."""
    if stage not in STAGES:
        raise ValueError("The local creative writer supports only trial, realize and revise_script")
    request = deepcopy(payload)
    renames = {}
    uses_references = validate_reference_request(request)
    if not isinstance(request.get("source"), dict):
        raise ValueError("A source-conditioned writing request is required")
    request.setdefault("task", stage)
    if stage == "trial":
        validate_program(request.get("program", {}))
        if "scene_material_roles" in request and "exploration" not in request:
            validate_scene_roles(request)
            proposal = request.get("expression_alternative")
            identifier = request.get("scene_id")
            if (not isinstance(proposal, dict) or not isinstance(identifier, str) or not identifier.strip()
                    or proposal.get("scene_id") != identifier or request["actual_scene"]["scene_id"] != identifier
                    or not isinstance(proposal.get("expression_change"), str) or not proposal["expression_change"].strip()):
                raise ValueError("The actual expression alternative must name the same current scene and change")
            request["exploration"] = {"scene_id": identifier, "approach": proposal["expression_change"]}
        if not isinstance(request.get("exploration"), dict):
            raise ValueError("The actual named-scene exploration is required")
        identifier = request["exploration"].get("scene_id")
        if identifier not in {scene["scene_id"] for scene in request["program"]["scenes"]}:
            raise ValueError("The trial scene must exist in the executed program")
        if request.get("scene_id", identifier) != identifier:
            raise ValueError("Conflicting trial scene identifiers")
    elif stage == "realize":
        validate_program(request.get("program", {}))
        if "explorations" in request:
            if "trials" in request and request["trials"] != request["explorations"]:
                raise ValueError("Conflicting actual trial material")
            request["trials"] = request.pop("explorations")
            renames["explorations"] = "trials"
        if not isinstance(request.get("trials"), list):
            raise ValueError("The actual exploratory scenes are required")
        request.setdefault("selected_trial_material", None)
        request.setdefault("trial_material_role", "All supplied exploratory material is provisional; no candidate has been selected")
    else:
        if "script" in request:
            if "draft" in request and request["draft"] != request["script"]:
                raise ValueError("Conflicting complete drafts")
            request["draft"] = request.pop("script")
            renames["script"] = "draft"
        if not isinstance(request.get("draft"), str) or not request["draft"].strip():
            raise ValueError("The actual complete draft is required")
        context = request.get("development_context", {})
        if not isinstance(context, dict):
            raise ValueError("Development context must be an object")
        if "program" not in request:
            request["program"] = deepcopy(context.get("program", {}))
        validate_program(request["program"])
        reading = request.get("whole_reading", {})
        if not isinstance(reading, dict):
            raise ValueError("The complete reading must be an object")
        proposals = reading.get("revision_opportunities", request.get("modification_proposals", []))
        if not isinstance(proposals, list):
            raise ValueError("Revision opportunities must be a list")
        if "modification_proposals" in request and request["modification_proposals"] != proposals:
            raise ValueError("Conflicting actual revision suggestions")
        request["modification_proposals"] = deepcopy(proposals)
        if "revision_opportunities" in reading:
            request["whole_reading"] = {key: deepcopy(value) for key, value in reading.items()
                                       if key != "revision_opportunities"}
        request.setdefault("instruction", "Assess the supplied fallible reading against the source, brief and complete draft. Preserve useful strengths; apply, partly apply or reject suggestions as appropriate. Write the complete revised screenplay.")
        system += "\nThe complete current draft is supplied in draft and specific revision advice is in modification_proposals; whole_reading retains the remaining reading. For this local creative stage, return JSON with a nonempty script containing the COMPLETE revised screenplay; supplementary decision fields are optional. All earlier requests for applied_changes or rejected_suggestions are optional. Do not return scene patches or edit commands."
    # The API and controller retain program versions and command traces. The
    # creative model receives the current program and all actual trial prose.
    for key in PROGRAM_BOOKKEEPING:
        request.pop(key, None)
    for material_key in ("trials", "explorations"):
        if isinstance(request.get(material_key), list):
            for trial in request[material_key]:
                if isinstance(trial, dict):
                    for key in PROGRAM_BOOKKEEPING:
                        trial.pop(key, None)
    context = request.get("development_context")
    if isinstance(context, dict):
        for key in PROGRAM_BOOKKEEPING:
            context.pop(key, None)
        if context.get("program") == request.get("program"):
            context.pop("program", None)
            renames["development_context.program"] = "program"
        if isinstance(context.get("explorations"), list):
            for trial in context["explorations"]:
                if isinstance(trial, dict):
                    for key in PROGRAM_BOOKKEEPING:
                        trial.pop(key, None)
    request = project_scene_roles(project_material_roles(request, renames))
    if uses_references:
        validate_reference_request(request)
        system += "\n" + REFERENCE_READING_INSTRUCTION
    return system, request


class CompatibleLocalWriterGenerator(matched.LocalWriterGenerator):
    """Use the same deterministic projection for every comparison condition."""
    supported_execution_modes = ("native_complete_v2", "native_complete_v3", "native_complete_v4")

    def __call__(self, key, stage, system, payload, backend):
        projected_system, projected_payload = project_request(stage, system, payload)
        result = super().__call__(key, stage, projected_system, projected_payload, backend)
        result["local_input_projection"] = {
            "version": VERSION, "stage": stage,
            "original_request_sha256": fingerprint({"system": system, "input": payload}),
            "projected_request_sha256": fingerprint({"system": projected_system, "input": projected_payload}),
            "projected_input_fields": sorted(projected_payload),
            "creates_training_labels": False,
        }
        if self.serialization != "legacy":
            result["local_input_projection"]["serialization"] = {
                "mode": self.serialization, "version": COMPACT_SERIALIZATION_VERSION}
        return result
