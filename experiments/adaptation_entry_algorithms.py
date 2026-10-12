"""Execute distinct adaptation workflows with explicit state and editing rules.

The controllers use one injected writer. They do not call an API, infer human
preferences, or turn structural checks into a measure of dramatic quality.
"""
from copy import deepcopy

from adaptation_system_contract import EDIT_AXES, fingerprint, validate_program


VERSION = "native-entry-algorithms-v1"
COMPLETE_SCENE_VERSION = "native-scene-complete-v2"
COMPLETE_ENTRY_VERSION = "native-entries-complete-v3"
MATERIAL_ENTRY_VERSION = "native-entries-material-development-v12.3.1"
# Request contracts keep their actual identity independently of method releases.
MATERIAL_WIRE_VERSION = "native-entries-material-development-v12"
COMPLETE_READER_VIEW_VERSION = "native-entries-material-development-v12.1"
POST_READ_PROGRAM_WIRE_VERSION = "native-entries-material-development-v12.3"
MERGE_REPRESENTATION_COMPILER_VERSION = "explicit-singleton-merge-scene-v1"
WORKING_MATERIAL_COMPILER_VERSION = "scene-specific-working-material-v1"
MATERIAL_ROLE_COMPILER_VERSION = "source-and-current-script-roles-v1"
EDIT_VALUE_COMPILER_VERSION = "explicit-edit-value-alias-v1"
AREAS = ("character_treatment", "preceding_preparation", "subsequent_development", "scene_organization")
PROVENANCE = {
    "scene_first": {
        "borrowed": "Explicit edit transactions, graph reachability, and finite candidate search",
        "proposed": "Actual scene edits inform revisions of character treatment, preparation, continuation, and scene organization",
        "sources": ["https://arxiv.org/html/2510.05188v4#S3", "https://arxiv.org/html/2503.15655#S2.SS1"],
    },
    "relationships_first": {
        "borrowed": "Character review and sequential generation with updated textual state",
        "adapted": "Generate linked encounters while committing revisable directed relationship records after each actual encounter",
        "sources": ["https://arxiv.org/html/2510.05188v4#S3", "https://aclanthology.org/2025.naacl-long.63/"],
    },
    "structure_first": {
        "borrowed": "Hierarchical outlining, explicit presentation order, and prefix-conditioned scene generation",
        "adapted": "Expand each section against already written scenes; no mandatory act template or causal/chronological presentation",
        "sources": ["https://aclanthology.org/2025.naacl-long.63/", "https://arxiv.org/html/2503.15655#S3.SS2"],
    },
    "expression_first": {
        "borrowed": "Separate narrative development from screenplay conversion and coordinate scene/dialogue editing",
        "adapted": "Realize a scenic treatment first, assign expression to medium-appropriate units, then convert and assemble actual scenes",
        "sources": ["https://arxiv.org/html/2510.23163#S3.SS2", "https://arxiv.org/html/2510.05188v4#S3"],
    },
}
PROGRAM = "{intent:string,choices:{plot:string,character_actions:string,relationships:string,ending:string,background:string,props:string,scene_organization:string,expression:string},scenes:[{scene_id:string,development:string}]}"
OPERATIONS = "[{op:replace_scene|insert_scene|delete_scene|move_scene|merge_scenes|split_scene|set_choice|set_intent,scene_id?:string,scene_ids?:[string],scenes?:[{scene_id:string,development:string}],development?:string,anchor_id?:string|null,position?:before|after,axis?:string,value?:string}]"
DEPENDENCIES = "[{from_scene_id:string,to_scene_id:string,area:character_treatment|preceding_preparation|subsequent_development|scene_organization,reason:string}]"


def text(value, key):
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Missing nonempty text: " + key)
    return result


def intent_edit_value(operation):
    """Compile an explicit intent string supplied under either known field.

    The union-shaped historical instruction permitted development on the same
    object. Accept that string only for set_intent, without inventing text or
    resolving conflicting values. Original model output stays unchanged.
    """
    value, development = operation.get("value"), operation.get("development")
    explicit_value = isinstance(value, str) and bool(value.strip())
    explicit_development = isinstance(development, str) and bool(development.strip())
    if explicit_value and explicit_development and value != development:
        raise ValueError("Conflicting explicit intent values")
    if explicit_value:
        return value
    if "value" not in operation and explicit_development:
        return development
    raise ValueError("Missing nonempty explicit intent value")


def explicit_edit_annotation(edit, key):
    """Accept one explicitly supplied annotation without inventing missing prose."""
    direct = edit.get(key)
    if isinstance(direct, str) and direct.strip():
        return direct
    operations = edit.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("Missing nonempty explicit edit annotation: " + key)
    supplied = [operation.get(key) for operation in operations if isinstance(operation, dict)
                and isinstance(operation.get(key), str) and operation[key].strip()]
    if len(supplied) == len(operations) and len(set(supplied)) == 1:
        return supplied[0]
    raise ValueError("Missing or conflicting explicit edit annotation: " + key)


def compile_program_operations(operations):
    """Flatten one explicitly supplied merge replacement without changing prose."""
    if not isinstance(operations, list):
        raise ValueError("Program patch requires operations")
    compiled, normalizations = [], []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise ValueError("An edit operation must be an object")
        item = deepcopy(operation)
        if item.get("op") == "merge_scenes" and "scenes" in item:
            scenes = item["scenes"]
            if (not isinstance(scenes, list) or len(scenes) != 1
                    or not isinstance(scenes[0], dict)
                    or set(scenes[0]) != {"scene_id", "development"}):
                raise ValueError("A nested merge requires exactly one explicit scene replacement")
            replacement = scenes[0]
            for field in ("scene_id", "development"):
                supplied = text(replacement, field)
                if field in item and item[field] != supplied:
                    raise ValueError("Conflicting flat and nested merge fields: " + field)
                item[field] = supplied
            item.pop("scenes")
            normalizations.append({"operation_index": index,
                "representation": "singleton_merge_scene_to_flat",
                "source_field": "scenes[0]", "scene_id": replacement["scene_id"],
                "development_sha256": fingerprint(replacement["development"])})
        compiled.append(item)
    return compiled, normalizations


def apply_program_patch(program, patch):
    """Apply ordered edits atomically to an explicitly bound program version."""
    original = deepcopy(validate_program(program))
    if patch.get("base_program_sha256") != fingerprint(original):
        raise ValueError("Stale program patch")
    operations, normalizations = compile_program_operations(patch.get("operations"))
    if "operation_compilation" in patch:
        lineage = patch["operation_compilation"]
        if not isinstance(lineage, dict):
            raise ValueError("Program operation compilation requires explicit provenance")
        raw = lineage.get("raw_operations")
        recompiled, recorded = compile_program_operations(raw)
        if (lineage.get("version") != MERGE_REPRESENTATION_COMPILER_VERSION
                or not recorded or normalizations or recompiled != operations
                or lineage.get("raw_operations_sha256") != fingerprint(raw)
                or lineage.get("compiled_operations_sha256") != fingerprint(operations)
                or lineage.get("normalizations") != recorded):
            raise ValueError("Program operation compilation provenance does not match actual edits")
    result = deepcopy(original)
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("An edit operation must be an object")
        op = operation.get("op")
        ids = [s["scene_id"] for s in result["scenes"]]
        if op == "set_choice":
            if operation.get("axis") not in EDIT_AXES:
                raise ValueError("Unknown edit axis")
            result["choices"][operation["axis"]] = text(operation, "value")
        elif op == "set_intent":
            result["intent"] = intent_edit_value(operation)
        elif op == "replace_scene":
            if operation.get("scene_id") not in ids:
                raise ValueError("Cannot replace an unavailable scene")
            result["scenes"][ids.index(operation["scene_id"])]["development"] = text(operation, "development")
        elif op in {"insert_scene", "move_scene"}:
            identifier = text(operation, "scene_id")
            if op == "insert_scene":
                if identifier in ids:
                    raise ValueError("Cannot insert a duplicate scene")
                scene = {"scene_id": identifier, "development": text(operation, "development")}
            else:
                if identifier not in ids:
                    raise ValueError("Cannot move an unavailable scene")
                scene = result["scenes"].pop(ids.index(identifier))
            remaining = [s["scene_id"] for s in result["scenes"]]
            anchor = operation.get("anchor_id")
            if anchor is None:
                index = len(remaining)
            else:
                if anchor not in remaining or operation.get("position") not in {"before", "after"}:
                    raise ValueError("Invalid scene placement")
                index = remaining.index(anchor) + int(operation["position"] == "after")
            result["scenes"].insert(index, scene)
        elif op == "delete_scene":
            if operation.get("scene_id") not in ids:
                raise ValueError("Cannot delete an unavailable scene")
            result["scenes"].pop(ids.index(operation["scene_id"]))
        elif op == "merge_scenes":
            consumed = operation.get("scene_ids")
            target = text(operation, "scene_id")
            if (not isinstance(consumed, list) or len(consumed) < 2
                    or len(consumed) != len(set(consumed)) or not set(consumed) <= set(ids)
                    or target in set(ids) - set(consumed)):
                raise ValueError("A merge must consume distinct existing scenes without an ID collision")
            index = min(ids.index(identifier) for identifier in consumed)
            replacement = {"scene_id": target, "development": text(operation, "development")}
            result["scenes"] = [scene for scene in result["scenes"] if scene["scene_id"] not in consumed]
            result["scenes"].insert(index, replacement)
        elif op == "split_scene":
            parent = operation.get("scene_id")
            children = operation.get("scenes")
            if parent not in ids or not isinstance(children, list) or len(children) < 2:
                raise ValueError("A split needs an existing scene and at least two replacements")
            child_ids = [text(scene, "scene_id") for scene in children]
            if len(child_ids) != len(set(child_ids)) or set(child_ids) & (set(ids) - {parent}):
                raise ValueError("Split replacement IDs collide")
            replacements = [{"scene_id": identifier, "development": text(scene, "development")}
                            for identifier, scene in zip(child_ids, children)]
            index = ids.index(parent)
            result["scenes"][index:index + 1] = replacements
        else:
            raise ValueError("Unknown program edit operation")
    return deepcopy(validate_program(result))


def dependency_scope(before, after, edges, seeds):
    """Find a conservative review scope in declared, fallible dependencies."""
    before, after = validate_program(before), validate_program(after)
    ids = {s["scene_id"] for s in before["scenes"] + after["scenes"]}
    if not set(seeds) <= ids:
        raise ValueError("Dependency seed refers to an unavailable scene")
    links = {identifier: set() for identifier in ids}
    for edge in edges:
        left, right = edge.get("from_scene_id"), edge.get("to_scene_id")
        if left not in ids or right not in ids or edge.get("area") not in AREAS:
            raise ValueError("Invalid declared scene dependency")
        text(edge, "reason")
        # Review preparation and consequences in both directions; this is not
        # a claim that causal time or presentation order is bidirectional.
        links[left].add(right)
        links[right].add(left)
    if before["choices"] != after["choices"] or before["intent"] != after["intent"]:
        return sorted(ids)
    reached, pending = set(seeds), list(seeds)
    old = {s["scene_id"]: s for s in before["scenes"]}
    new = {s["scene_id"]: s for s in after["scenes"]}
    changed = {identifier for identifier in ids if old.get(identifier) != new.get(identifier)}
    if [s["scene_id"] for s in before["scenes"]] != [s["scene_id"] for s in after["scenes"]]:
        # A changed adjacency may affect rhythm beyond a declared causal edge.
        changed.update(ids)
    pending.extend(changed - reached)
    reached.update(changed)
    while pending:
        identifier = pending.pop()
        for neighbor in links[identifier] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    return sorted(reached)


def derive_surviving_edit_targets(before, after, patch, focus, edges=()):
    """Find actual surviving trial targets in the executed candidate arrangement."""
    before, after = validate_program(before), validate_program(after)
    original_ids = [scene["scene_id"] for scene in before["scenes"]]
    final_ids = [scene["scene_id"] for scene in after["scenes"]]
    if focus not in original_ids or fingerprint(apply_program_patch(before, patch)) != fingerprint(after):
        raise ValueError("Trial targets require the actual executed patch and original focus")
    # Validate declared dependencies without using global arrangement changes
    # to turn a deletion into a trial of every remaining scene.
    dependency_scope(before, after, edges, [focus])
    ordered, direct, deleted, boundaries, replacements = list(original_ids), set(), set(), set(), {}
    for operation in patch["operations"]:
        op = operation["op"]
        if op == "replace_scene":
            direct.add(operation["scene_id"])
        elif op in {"insert_scene", "move_scene"}:
            identifier = operation["scene_id"]
            if op == "move_scene":
                ordered.remove(identifier)
            anchor = operation.get("anchor_id")
            index = (len(ordered) if anchor is None else
                     ordered.index(anchor) + int(operation["position"] == "after"))
            ordered.insert(index, identifier)
            direct.add(identifier)
        elif op == "merge_scenes":
            consumed, target = operation["scene_ids"], operation["scene_id"]
            index = min(ordered.index(identifier) for identifier in consumed)
            for identifier in consumed:
                replacements.setdefault(identifier, set()).add(target)
            ordered = [identifier for identifier in ordered if identifier not in consumed]
            ordered.insert(index, target)
            direct.difference_update(consumed)
            direct.add(target)
        elif op == "split_scene":
            parent = operation["scene_id"]
            children = [scene["scene_id"] for scene in operation["scenes"]]
            index = ordered.index(parent)
            ordered[index:index + 1] = children
            replacements.setdefault(parent, set()).update(children)
            direct.discard(parent)
            direct.update(children)
        elif op == "delete_scene":
            identifier = operation["scene_id"]
            index = ordered.index(identifier)
            boundaries.update(ordered[max(0, index - 1):index] + ordered[index + 1:index + 2])
            deleted.add(identifier)
            ordered.pop(index)
            direct.discard(identifier)
    if ordered != final_ids:
        raise ValueError("Derived trial arrangement differs from the executed program")
    surviving = set(final_ids)
    targets = direct & surviving
    if deleted:
        links = {identifier: set() for identifier in set(original_ids + final_ids) | deleted}
        for edge in edges:
            left, right = edge["from_scene_id"], edge["to_scene_id"]
            links[left].add(right)
            links[right].add(left)
        reached, pending = set(deleted), list(deleted)
        while pending:
            identifier = pending.pop()
            following = links.get(identifier, set()) - reached
            reached.update(following)
            pending.extend(following)
        pending, reached = list(boundaries | reached), set()
        while pending:
            identifier = pending.pop()
            if identifier in reached:
                continue
            reached.add(identifier)
            if identifier in surviving:
                targets.add(identifier)
            pending.extend(replacements.get(identifier, ()))
    if not targets:
        targets.add(focus if focus in surviving else final_ids[0])
    return [identifier for identifier in final_ids if identifier in targets]


def presentation_order(scene_ids, constraints):
    """Stable topological order of explicitly requested presentation constraints."""
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Presentation requires unique scenes")
    positions = {identifier: i for i, identifier in enumerate(scene_ids)}
    links = {identifier: set() for identifier in scene_ids}
    indegree = {identifier: 0 for identifier in scene_ids}
    for edge in constraints:
        left, right = edge.get("before"), edge.get("after")
        if left not in links or right not in links or left == right:
            raise ValueError("Invalid presentation constraint")
        if edge.get("kind") != "presentation":
            raise ValueError("Event time and causal relations cannot silently become presentation constraints")
        text(edge, "reason")
        if right not in links[left]:
            links[left].add(right)
            indegree[right] += 1
    ready = [identifier for identifier in scene_ids if indegree[identifier] == 0]
    ordered = []
    while ready:
        ready.sort(key=positions.__getitem__)
        current = ready.pop(0)
        ordered.append(current)
        for following in sorted(links[current], key=positions.__getitem__):
            indegree[following] -= 1
            if indegree[following] == 0:
                ready.append(following)
    if len(ordered) != len(scene_ids):
        raise ValueError("Cyclic presentation constraints require an explicit writing decision")
    return ordered


def commit_relationship_updates(state, updates, event_id):
    """Commit version-checked textual interpretations after an actual encounter."""
    result = deepcopy(state)
    if not isinstance(updates, list):
        raise ValueError("Relationship updates must be explicit, including no changes")
    known = {row["relation_id"]: row for row in result}
    if len(known) != len(result):
        raise ValueError("Duplicate relationship identifiers")
    changed = set()
    for update in updates:
        identifier = update.get("relation_id")
        if identifier not in known or identifier in changed:
            raise ValueError("Unknown or duplicated relationship update")
        changed.add(identifier)
        relation = known[identifier]
        if update.get("before") != relation["interpretation"]:
            raise ValueError("Stale relationship interpretation")
        if update.get("status") not in {"changed", "retained", "uncertain"}:
            raise ValueError("Invalid relationship interpretation status")
        after = text(update, "after")
        if update["status"] == "retained" and after != relation["interpretation"]:
            raise ValueError("Retained relationship interpretation cannot change")
        text(update, "reason")
        relation.update(interpretation=after, uncertainty=text(update, "uncertainty"), last_event_id=event_id)
    return result


def assemble_scenes(order, parts):
    """Assemble every actual scene once without truncation or silent omission."""
    if not order or len(order) != len(set(order)):
        raise ValueError("Scene assembly requires unique scene identifiers")
    indexed = {part.get("scene_id"): part for part in parts}
    if len(indexed) != len(parts) or set(indexed) != set(order):
        raise ValueError("Missing, duplicate, or unexpected realized scene")
    return "\n\n".join(text(indexed[identifier], "text") for identifier in order)


def apply_script_patch(program, parts, patch):
    """Commit a script edit atomically; unchanged scene text stays byte-exact."""
    order = [scene["scene_id"] for scene in validate_program(program)["scenes"]]
    assemble_scenes(order, parts)
    if patch.get("base_script_sha256") != fingerprint(assemble_scenes(order, parts)):
        raise ValueError("Stale complete-script patch")
    updated = apply_program_patch(program, patch.get("program_patch", {}))
    target_order = [scene["scene_id"] for scene in updated["scenes"]]
    before = {part["scene_id"]: deepcopy(part) for part in parts}
    result = {identifier: deepcopy(before[identifier]) for identifier in target_order if identifier in before}
    updates = patch.get("scene_updates")
    if not isinstance(updates, list):
        raise ValueError("Script patch requires explicit scene updates")
    touched = set()
    for update in updates:
        identifier = update.get("scene_id")
        if identifier in touched or identifier not in target_order:
            raise ValueError("Duplicate or unavailable script edit target")
        touched.add(identifier)
        expected = fingerprint(before[identifier]["text"]) if identifier in before else None
        if update.get("base_text_sha256") != expected:
            raise ValueError("Stale scene text replacement")
        text(update, "reason")
        result[identifier] = {"scene_id": identifier, "text": text(update, "text")}
    changed_developments = {identifier for identifier in target_order if identifier in before
                            and next(s["development"] for s in updated["scenes"] if s["scene_id"] == identifier)
                            != next(s["development"] for s in program["scenes"] if s["scene_id"] == identifier)}
    if not changed_developments <= touched:
        raise ValueError("Changed scene development requires an actual scene text update")
    updated_parts = [result[identifier] for identifier in target_order if identifier in result]
    script = assemble_scenes(target_order, updated_parts)
    return {"program": updated, "scenes": updated_parts, "script": script,
            "script_patch": deepcopy(patch),
            "unchanged_scene_ids": [identifier for identifier in target_order if identifier in before and identifier not in touched]}


SCENE_SEED = (
    "Read the source, brief and common initial writing. Develop a provisional "
    "complete program and one concrete adaptation decision worth exploring in "
    "an actual scene. Return JSON {program:" + PROGRAM + ",scene_id:string,"
    "dependencies:" + DEPENDENCIES + ",edits:[{operations:" + OPERATIONS + ","
    "intended_gain:string,possible_loss:string},{operations:" + OPERATIONS + ","
    "intended_gain:string,possible_loss:string}]}. Each edit is an executable "
    "different proposal against the supplied program, not a stylistic slogan. "
    "The focal scene must belong to the provisional program and exist after each proposal. Participants, situation, "
    "actions and all eight edit axes may change within the brief. Dependencies "
    "are revisable writing hypotheses, not psychological truth. Do not invent "
    "quality scores, require more conflict, or privilege surface fidelity."
)
MATERIAL_SCENE_SEED = SCENE_SEED.replace(
    "The focal scene must belong to the provisional program and exist after each proposal.",
    "The focal scene must belong to the provisional program before editing. "
    "An edit may replace, move, merge, split or delete that scene. "
    "The controller assigns actual surviving replacement scenes or affected "
    "neighboring scenes for writing after applying the operations.")
RELATION_SEED = (
    "Read the actual source, brief and initial complete draft. Use the supplied "
    "published Character Evaluator as a task-adapted starting point. Return JSON "
    "{program:" + PROGRAM + ",people:[string],relationships:[{relation_id:string,"
    "from_person:string,to_person:string,interpretation:string,uncertainty:string}],"
    "encounter_chains:[{approach:string,encounters:[{encounter_id:string,scene_id:string,"
    "situation:string}]}]}. Propose exactly two different chains, each with at "
    "least two linked encounters. People may interpret a relation differently. "
    "Keep ambiguity and contradictory expectations; no fixed personality, "
    "numerical affection, growth, or obligatory relationship change. These are "
    "proposed adapted relationships, not unique inferred source psychology."
)
RELATION_WRITE = (
    "Write the next actual screenplay encounter using the current relationship "
    "records and all previous actual encounters in this chain. Return JSON "
    "{text:string,relationship_updates:[{relation_id:string,before:string,"
    "after:string,status:changed|retained|uncertain,reason:string,uncertainty:string}]}. "
    "before must equal the supplied current interpretation. Base the update "
    "on what this actual encounter realizes; multiple readings can coexist. "
    "The new scene must respond to earlier exchanges where relevant. A relation "
    "can remain unchanged. Write dialogue, action and selected expression; a "
    "profile or encounter synopsis alone is insufficient."
)
RELATION_SCENE = (
    "Write only the assigned complete screenplay scene in this encounter chain. "
    "Use the current directed relationship interpretations and previous actual "
    "encounters; respond to particular exchanges without resolving every motive. "
    "Return JSON {scene_id:string,text:string}. The scene_id is the assigned "
    "encounter's scene_id. Write dialogue, action and medium-appropriate expression. "
    "The interpretations are fallible, not fixed traits or psychological truth. "
    "Do not include other scenes or report state updates instead of writing."
)
RELATION_INTERPRET = (
    "Read the actual scene just written in context of previous actual encounters "
    "and the current directed relationship records. Return JSON "
    "{relationship_updates:[{relation_id:string,before:string,after:string,"
    "status:changed|retained|uncertain,reason:string,uncertainty:string}]}. "
    "before must equal the current interpretation. Update only interpretations "
    "supported by this particular exchange; unchanged records need no entry. "
    "Preserve alternative motives and contradictory expectations. Interpretations "
    "may be retained, and an empty list is legitimate. Do not write another scene "
    "or treat a character profile or the planned situation as realized action."
)
STRUCTURE_SEED = (
    "Use the supplied official planning prompt as the starting point for a "
    "complete adaptation arrangement. Return JSON {program:" + PROGRAM + ","
    "sections:[{section_id:string,purpose:string,scene_ids:[string]}],"
    "presentation_constraints:[{before:string,after:string,kind:presentation,reason:string}]}. "
    "Sections partition the program scenes; preserve no mandatory act count. "
    "Make selection, cuts, additions, merging and order real decisions. "
    "Presentation constraints concern viewer order, not chronology or all "
    "causal edges. Flashbacks and juxtaposition remain available."
)
STRUCTURE_EXPAND = (
    "Expand this section of the complete arrangement after reading the actual "
    "scenes already written. Return JSON {scene_tasks:[{scene_id:string,"
    "development:string,preparation:string,later_resonance:string}]}. Include "
    "exactly the listed section scene identifiers. Revise their concrete "
    "development when earlier writing warrants it; do not force the rough "
    "outline to be correct. No fixed beat template, escalating conflict, "
    "closed ending or mandatory emotional change."
)
STRUCTURE_WRITE = (
    "Write the next complete screenplay scene from its expanded task, the "
    "complete arrangement, and the actual previously written screenplay. "
    "Return JSON {scene_id:string,text:string}. Carry particular people, "
    "objects, relationships and rhythm across scenes; nonlinear presentation "
    "is legitimate. Write the complete scene, not a summary."
)
EXPRESSION_SEED = (
    "Develop how this source experience can live in the target dramatic medium. "
    "Return JSON {program:" + PROGRAM + ",scenic_treatment:string}. Write a "
    "substantial scenic treatment connecting particular relationships, inner "
    "contradictions, rhythm and lived meaning. Freely adapt all eight axes. "
    "This is narrative development before screenplay conversion; voice and "
    "monologue remain legitimate. Avoid default emotional gesture shorthand."
)
EXPRESSION_DISTRIBUTE = (
    "From the actual scenic treatment, allocate expression across the program "
    "scenes. Return JSON {units:[{unit_id:string,scene_id:string,medium_element:string,"
    "actual_material:string,intended_experience:string,possible_loss:string}]}. "
    "Write usable speech, action, silence, space, image, sound or temporal "
    "juxtaposition in actual_material, rather than describing a future choice. "
    "Each program scene needs material. Different elements may coexist; no "
    "fixed channel ranking or rule against narration. Consider repetition "
    "and how one element changes the reading of another."
)
EXPRESSION_CONVERT = (
    "Convert this scene's actual expression units and scenic treatment into "
    "one complete screenplay scene. Read earlier converted scenes to coordinate "
    "rhythm, language and echoes. Return JSON {scene_id:string,text:string,"
    "unit_dispositions:[{unit_id:string,status:used|revised|rejected,reason:string}]}. "
    "Every supplied unit needs a disposition. Keep effective specificity; "
    "reconsider material that duplicates or damages another element. All "
    "medium-appropriate expression remains allowed."
)
EXPRESSION_SCENE = (
    "Convert the supplied actual expression units and scenic treatment into "
    "only the assigned complete screenplay scene. Return JSON {scene_id:string,"
    "text:string}. Read previous actual scenes to coordinate rhythm, language "
    "and echoes. Expression observations are fallible advice. Use, revise or "
    "omit material where it serves the complete scene; all medium-appropriate "
    "expression, including narration and silence, remains legitimate. Do not "
    "include another scene or substitute a list of unit dispositions for writing."
)
EXPRESSION_READ = (
    "Read the actual converted scene alongside its proposed expression units, "
    "scenic treatment and previous scenes. Return JSON {unit_dispositions:"
    "[{unit_id:string,status:used|revised|rejected|uncertain,reason:string}]}. "
    "Address every supplied unit once. Distinguish the actual scene from its "
    "plan: a claimed intention alone does not establish its realization. "
    "Keep uncertainty where an image, silence or phrase allows several readings. "
    "Consider how elements interact and whether an echo is intentional. Do not "
    "score quality, require every unit's inclusion, penalize voiceover or narration, "
    "or write a replacement scene. These observations guide later writing."
)
EXPRESSION_ORGANIZATION_PROPOSAL = (
    "Read the complete actual converted scene sequence, its provisional scenic "
    "treatment and allocated expression, the source and brief, and the actual "
    "initial complete draft. Propose one different expression organization "
    "worth trying in one existing program scene. Return JSON {scene_id:string,"
    "expression_change:string,intended_gain:string,possible_loss:string}. "
    "Name an existing scene and a concrete different treatment that can be "
    "written as a complete alternative scene. Consider how expression, timing, "
    "perspective and interaction offer different experiences of the particular "
    "relationships, inner contradictions and lived circumstances. Earlier "
    "unit dispositions describe what happened, not whether it was best. A "
    "different treatment need not repair a fault. Only source.brief binds the "
    "work; generated treatment, units and readings remain revisable. People, "
    "actions and all other brief-authorized aspects can change. No channel, "
    "more dialogue, less narration, source retention, conflict or restraint "
    "has a default advantage. Give a specific intended gain and possible loss, "
    "without scores, replacement prose or an assertion that the original is wrong."
)
EXPRESSION_ALTERNATIVE_SCENE = (
    "Write the proposed different expression organization as another complete "
    "realization of the assigned scene. Return JSON {scene_id:string,text:string}. "
    "Read the actual original scene, its preceding and following actual scenes "
    "and the full converted sequence to develop a concrete alternative encounter and its "
    "rhythm. Consider who is addressed, who can hear or respond, and the "
    "consequences of this different treatment for the adjoining exchanges. "
    "Keep the assigned identifier and write only this complete scene "
    "with a natural endpoint. The original and alternative are separate "
    "possible treatments, not consecutive events. Only source.brief binds the "
    "adaptation; the proposal, scenic treatment and expression units are "
    "fallible guidance and may be transformed. Speech, private thought, "
    "audience-only voice, silence, action, image, sound and temporal "
    "juxtaposition have no default ranking. Realize the different treatment "
    "in actual writing, not in a description of its benefit. Do not include "
    "other scenes, quality claims or planning labels in the screenplay."
)
EXPRESSION_ALTERNATIVE_READ = (
    "Read both actual complete treatments of this scene in their shared source "
    "and adaptation context. Return JSON {treatments:[{trial_id:string,scene_id:string,"
    "experience:string,gains_and_losses:string,uncertainty:string}],whole_implications:string}. "
    "Describe each supplied treatment once using its actual trial and scene "
    "identifiers. Compare what the particular exchanges, psychological layers, "
    "lived circumstances and rhythm make available to a viewer. Distinguish "
    "the proposed effect from what the actual writing offers. Consider how "
    "a treatment changes who can hear, answer, evade, misunderstand or act, "
    "and its gain and loss without preferring any expression channel. Explain "
    "possible implications for character treatment, preceding preparation, "
    "subsequent development and scene organization. Keeping, combining, "
    "transforming or rejecting treatments can all serve the whole; a "
    "combination does not require their events to coexist. Only source.brief "
    "supplies binding author requirements. Do not score, rank, certify "
    "quality, demand fidelity or write a replacement scene. These are fallible "
    "readings of separate actual alternatives, not facts about the source."
)
def expression_scene_payload(payload, current_scene, *, reading=None):
    """Put the complete current single scene before separately identified context."""
    ordered = {"actual_scene": deepcopy(current_scene)}
    if "source" in payload:
        ordered["source"] = deepcopy(payload["source"])
    ordered.update({key: deepcopy(value) for key, value in payload.items()
                    if key not in {"actual_scene", "source", "scene_reading", "scene_material_roles"}})
    ordered["scene_material_roles"] = {"current_scene_field": "actual_scene",
        "current_scene_scope": "one_complete_assigned_scene",
        "field_roles": {"actual_scene": "Current complete single scene supplied for reading or rewriting",
            "source": "Supplied source material within its stated boundaries",
            "source.brief": "Binding author adaptation requirements when supplied",
            "initial_plan": "Historical provisional model plan",
            "initial_script": "Historical complete common draft, not the current single scene",
            "actual_prefix": "Actual preceding scenes, not the current single scene",
            "program": "Revisable generated arrangement",
            "scenic_treatment": "Revisable generated treatment",
            "expression_units": "Revisable generated expression allocation",
            "previous_expression_observations": "Historical readings bound to their recorded scene hashes"}}
    if reading is not None:
        ordered["scene_reading"] = deepcopy(reading)
        ordered["scene_material_roles"]["field_roles"]["scene_reading"] = (
            "Fallible reading of the supplied original current scene, not an author requirement")
    return ordered


PROGRAM_UPDATE = (
    "Read the actual exploratory material and original program. Revise the "
    "whole adaptation using executable operations. Return JSON {operations:"
    + OPERATIONS + ",dependency_changes:[{area:string,status:changed|retained|uncertain,"
    "before:string,after:string,reason:string,scene_ids:[string]}],additional_dependencies:"
    + DEPENDENCIES + "}. Include exactly one decision for each of character_treatment, "
    "preceding_preparation, subsequent_development and scene_organization. "
    "Declared dependency scopes prioritize review and do not forbid other "
    "necessary edits. Discover additional impacts and report them. Retaining "
    "or rejecting exploratory choices is legitimate. Operations change the "
    "program; no full replacement program or quality score."
)

MATERIAL_UPDATE = (
    " Extend the returned JSON with scene_working_material:[{scene_id:string,"
    "starting_material_ids:[string],starting_material_parts?:[{trial_id:string,"
    "scene_ids:[string]}],rewrite_focus:string,possible_loss:string}]. "
    "scene_id names a scene in the UPDATED program. starting_material_ids "
    "refer to supplied trial_id values or the registered initial draft ID initial. "
    "The initial draft is supplied once in initial_script; its registered field "
    "and SHA reference denotes whole_material_context with no inferred scene "
    "boundaries. Use initial alone, with trial drafts, or not at all. Never name "
    "initial in starting_material_parts because it has no recorded component scenes. "
    "Decide which actual draft material "
    "is useful to develop in which scene, and what specifically needs rewriting "
    "for the whole adaptation. Use actual exchanges and expression, not merely "
    "the intended gain. More than one alternative may inform a scene, but their "
    "events need not coexist. An empty list or no starting material is legitimate; "
    "For multi-scene material, use starting_material_parts to name the actual "
    "source scene components needed for this updated scene. Components can "
    "come from any source scene, supporting merges, splits and new placements. "
    "Omitting this optional field uses a same-ID component when available, "
    "otherwise the whole draft is explicitly marked as undivided context. "
    "all trials may be rejected. Changes to cast, actions, setting, props, plot "
    "and ending remain allowed. Do not invent replacement prose in this record, "
    "require original wording, certify quality or select by conflict intensity."
)
MATERIAL_REALIZE = (
    " Use scene_working_material as actual, revisable starting drafts for the "
    "named scenes in the revised arrangement. Develop their particular exchange "
    "and rewrite_focus; do not reduce an effective interaction to 'she asks a "
    "question' or 'they negotiate' when the exchange carries this adaptation's "
    "meaning. Starting drafts are separately identified alternatives, not a "
    "sequence of events to concatenate. Where one draft informs several scenes, "
    "redistribute its interaction rather than copying it repeatedly. Read the "
    "actual initial_script for useful passages and whole-work movement. Keep, "
    "transform, replace or discard any supplied passage when the whole warrants "
    "it. New choices may change every brief-authorized aspect of the source. "
    "Make intended psychological and relational layers available to performance "
    "through the chosen speech, action, voice, image, sound, space or timing; "
    "motives can remain unresolved. Acting and directing notes are legitimate, "
    "but a private explanatory claim alone does not create a viewer experience. "
    "Do not routinely add psychological summaries, moral lines or dramatic "
    "gestures. Write the complete script with its opening, movement and ending."
)
MATERIAL_REVIEW = (
    " Read the current script also against the actual initial_script and "
    "scene_working_material in development_context. Examine what was gained "
    "or lost in particular exchanges, ambivalence and lived circumstances. "
    "An intentional change or rejection can be successful; restoration of "
    "source events, trial wording or initial order earns no automatic advantage. "
    "Where a concern matters, put it in revision_opportunities at its actual "
    "location, with a concrete proposed_change and possible_loss. Distinguish "
    "a writer's psychological explanation from a plausible acted, audible or "
    "visible experience; acting notes, narration and uncertainty remain valid. "
    "Distinguish words addressed to another person, private thought, and speech "
    "available only to the audience. When expression changes channel, examine "
    "how the other person's opportunity to answer, evade, misunderstand or "
    "act changes, and what happens to the ensuing exchange. Compare actual "
    "initial and current passages where that change matters. Private expression "
    "can deepen a scene; losing an interpersonal response can also flatten it. "
    "Describe the particular gain and loss without preferring dialogue or "
    "requiring any earlier wording. "
    "Assess scene adjacencies and the placement of earlier preparation and "
    "later development: does their arrangement alter how the encounter is "
    "experienced, or disperse or repeat it without a useful effect? Quiet "
    "transitions, nonlinear order, repetition and open endings can be apt. "
    "Do not reward autonomy, explicitness, silence, twists or added conflict "
    "by default. Check only assertions claimed as source facts against the "
    "source; authorized changed people, actions and settings are adaptation "
    "choices. Also inspect useful initial passages outside the focal trial; "
    "an unbound scene need not be rewritten from scratch. Where useful, extend "
    "the reading with working_scene_observations:[{scene_id:string,"
    "treatment:developed|transformed|replaced|omitted|uncertain,"
    "actual_script_location:string|null,gains_and_losses:string,uncertainty:string}]. "
    "Use an actual location when identifiable and null when absent or unclear. "
    "These are fallible readings, not word-match tests or an obligation to "
    "retain material. Give no score or certified preference; advice can be empty."
)
MATERIAL_FINAL = (
    " Resolve the actual whole-reading concerns while preserving useful "
    "specificity and rhythm in the current draft. Advice about psychological "
    "expression does not require explanatory dialogue, and advice about "
    "earlier/later scenes does not require linear order or extra scenes. "
    "Reconsider source facts only where claimed as facts. Use the actual brief "
    "to determine any limits on new characters, actions, backgrounds and endings. The working materials "
    "remain revisable, including rejecting all of them. Check how revisions "
    "affect the other person's response and the surrounding scenes. An "
    "addressed question, a withheld thought and audience-only voice create "
    "different encounters. Where a revision alters who can hear and respond, "
    "develop the resulting response sequence or its deliberate absence in "
    "the actual scene. Preserve useful complexity through the chosen medium "
    "without adding explanatory speeches or automatically restoring earlier "
    "lines. Return "
    "the same complete-script JSON; do not add quality claims or planning "
    "labels to the screenplay."
)
MATERIAL_SOURCE_ROLES = (
    " Only requirements actually supplied in source.brief bind the adaptation. "
    "The initial_plan, every generated program, section plan, scenic treatment, "
    "interpretation and reading are provisional model decisions. Reconsider "
    "their self-imposed restrictions when the actual material and the artistic "
    "gain and possible loss warrant a different treatment. Judge those gains "
    "and losses by the particular exchanges and viewer experience; speech, "
    "silence, performance, narration and visual expression remain available. "
    "Restraint, ambiguity, autonomy or extent of change earns no automatic "
    "advantage, and a model preference creates no author prohibition. All eight "
    "edit axes remain available within the actual brief. The initial_script and "
    "actual exploratory drafts are separately identified historical writing; "
    "they may offer useful exchanges beyond the focal trial and remain available "
    "for transformation, reuse or rejection. Alternative drafts describe "
    "alternative treatments, whose events need not coexist."
)
MATERIAL_CHOICE_UPDATE = (
    " Check the updated scenes against the program's upper-level choices, "
    "especially ending and scene_organization. When a scene decision changes "
    "those choices, synchronize them through the existing set_choice operations "
    "and use set_intent when the intention changes. Review all eight axes as "
    "relevant; retaining a compatible choice is legitimate. Identify the actual "
    "material and location behind each judgment of gain or possible loss, including "
    "useful initial-script exchanges outside the explored material."
)
MATERIAL_CURRENT_SCRIPT = (
    " The payload identifies the actual current complete script separately "
    "from historical drafts and model advice. Locate each occurring, absent "
    "or repeated action in that current text. An action in an initial draft, "
    "trial, assembly, program or summary alone establishes no current event. "
    "Reconsider historical writing where useful while checking the resulting "
    "current scene and its surroundings. Model advice remains revisable and "
    "cannot create a new author requirement."
)
REVISION_ATTEMPT_INSTRUCTION = (
    " When revision_to_attempt is supplied, use this call to realize that "
    "specific dramatic opportunity as an alternative complete draft, including "
    "its effects on the encounter and surrounding scenes. The preceding complete "
    "draft remains a separate selection candidate. Do not discard the opportunity "
    "solely because its unwritten outcome might be too explicit, too ambiguous "
    "or insufficiently restrained. Speech, silence, image and narration have no "
    "default rank. You may reject an actual brief violation or an observation "
    "that does not occur in the current text; identify that specific mismatch. "
    "Other advice remains optional, and writing the alternative does not certify "
    "its superiority or commit it as the selected script.")


def compile_revision_attempt(reading, *, accepted_indices=None):
    """Bind the first actionable reader proposal without ranking its artistic worth."""
    opportunities = reading.get("revision_opportunities", [])
    if not isinstance(opportunities, list):
        return None
    for index, opportunity in enumerate(opportunities):
        if accepted_indices is not None and index not in accepted_indices:
            continue
        if (isinstance(opportunity, dict) and all(
                isinstance(opportunity.get(key), str) and opportunity[key].strip()
                for key in ("location", "concern", "proposed_change", "possible_loss"))):
            return {"reader_priority_index": index, "whole_reading_sha256": fingerprint(reading),
                    "proposal": deepcopy(opportunity), "proposal_quality_certified": False}
    return None


def material_scene_parts(material):
    """Read actual component scenes, never infer a scene boundary from prose."""
    for field in ("encounters", "ordered_scenes", "converted_scenes"):
        if field in material:
            parts = material[field]
            if not isinstance(parts, list) or not parts:
                raise ValueError("Actual multi-scene material needs explicit scene components")
            identifiers = [text(part, "scene_id") for part in parts]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("Actual material scene identifiers must be unique")
            for part in parts:
                text(part, "text")
            return parts
    return []


def resolve_component_ids(parts, requested, single_scene_id=None):
    """Resolve recorded scene and encounter identifiers without fuzzy matching."""
    known = {part["scene_id"]: part["scene_id"] for part in parts}
    if single_scene_id is not None:
        known[single_scene_id] = single_scene_id
    for part in parts:
        encounter = part.get("encounter_id")
        if isinstance(encounter, str) and encounter.strip():
            if encounter in known and known[encounter] != part["scene_id"]:
                raise ValueError("An actual component identifier is ambiguous")
            known[encounter] = part["scene_id"]
    if not set(requested) <= set(known):
        raise ValueError("A selected starting component is absent from actual material")
    resolved = [known[identifier] for identifier in requested]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Two references identify the same actual starting component")
    return resolved



INITIAL_MATERIAL_ID = "initial"


def initial_material_reference(initial_script):
    """Register one real complete draft by field and hash without copying its text."""
    if not isinstance(initial_script, str) or not initial_script.strip():
        raise ValueError("A real complete initial script is required for a material reference")
    return {"trial_id": INITIAL_MATERIAL_ID, "representation": "common_initial_complete_script",
            "text_reference": {"field": "initial_script", "sha256": fingerprint(initial_script)},
            "material_scope": "whole_material_context"}


def validate_initial_material_references(payload):
    """Check exact whole-draft references before or after a real input projection."""
    containers = [payload]
    if isinstance(payload.get("development_context"), dict):
        containers.append(payload["development_context"])
    present = any("retained_initial_material" in container or any(
        isinstance(draft, dict) and draft.get("trial_id") == INITIAL_MATERIAL_ID
        for row in container.get("scene_working_material", []) if isinstance(row, dict)
        for draft in row.get("starting_drafts", [])) for container in containers)
    if not present:
        return {"reference_count": 0, "initial_script_reference_present": False}
    expected = initial_material_reference(payload.get("initial_script"))
    count = 0
    for container in containers:
        registry = container.get("retained_initial_material")
        if registry is not None and registry != expected:
            raise ValueError("Initial material field, SHA, identifier or whole context changed")
        for row in container.get("scene_working_material", []):
            if not isinstance(row, dict):
                raise ValueError("Initial material tasks must be objects")
            if any(part.get("trial_id") == INITIAL_MATERIAL_ID
                   for part in row.get("declared_starting_material_parts", []) if isinstance(part, dict)):
                raise ValueError("Initial complete text has no recorded component scene boundaries")
            initial_drafts = [draft for draft in row.get("starting_drafts", [])
                              if isinstance(draft, dict) and draft.get("trial_id") == INITIAL_MATERIAL_ID]
            declared = INITIAL_MATERIAL_ID in row.get("starting_material_ids", [])
            if declared != (len(initial_drafts) == 1) or len(initial_drafts) > 1:
                raise ValueError("Declared initial material must resolve to one registered whole-draft reference")
            for draft in initial_drafts:
                if registry != expected:
                    raise ValueError("Initial material reference lacks its actual registered draft")
                allowed = set(expected) | {"original_scene_ids", "actual_scene_components",
                    "source_material_sha256", "component_selection_is_not_a_quality_claim", "also_used_for_scene_ids"}
                if (set(draft) != allowed or any(draft.get(key) != value for key, value in expected.items())
                        or draft.get("original_scene_ids") != [] or draft.get("actual_scene_components") != []
                        or draft.get("source_material_sha256") != expected["text_reference"]["sha256"]
                        or draft.get("component_selection_is_not_a_quality_claim") is not True
                        or not isinstance(draft.get("also_used_for_scene_ids"), list)):
                    raise ValueError("Initial material must be one SHA-bound field reference without text or invented components")
                count += 1
    return {"reference_count": count, "initial_script_reference_present": True,
            "initial_script_sha256": expected["text_reference"]["sha256"],
            "material_scope": "whole_material_context", "invented_scene_boundaries": False}

def compile_scene_working_material(rows, program, materials, *, retained_initial_material=None, initial_script=None):
    """Attach real draft prose to revised scene tasks without inventing quality."""
    if not isinstance(rows, list):
        raise ValueError("Working-scene decisions must be an explicit list")
    order = [scene["scene_id"] for scene in validate_program(program)["scenes"]]
    available = {}
    for material in materials:
        identifier = text(material, "trial_id")
        if identifier == INITIAL_MATERIAL_ID:
            raise ValueError("The initial material identifier is reserved for its registered field reference")
        if identifier in available:
            raise ValueError("Duplicate actual material identifier")
        text(material, "text")
        available[identifier] = material
    if retained_initial_material is not None:
        expected_initial = initial_material_reference(initial_script)
        if retained_initial_material != expected_initial or INITIAL_MATERIAL_ID in available:
            raise ValueError("Initial material field, SHA or reserved identifier changed")
        available[INITIAL_MATERIAL_ID] = deepcopy(retained_initial_material)

    def resolve_reference(reference):
        candidate = reference.removeprefix("trial_") if reference.startswith("trial_") else None
        if reference in available:
            if candidate in available and candidate != reference:
                raise ValueError("A material reference conflicts with an existing actual identifier")
            return reference
        if candidate in available:
            return candidate
        raise ValueError("Working scene references unavailable actual draft material")

    tasks = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Working-scene decisions must be objects")
        identifier = text(row, "scene_id")
        if identifier not in order or identifier in tasks:
            raise ValueError("Working scene must be unique and belong to the revised program")
        declared_references = row.get("starting_material_ids")
        if (not isinstance(declared_references, list) or any(not isinstance(v, str) for v in declared_references)
                or len(declared_references) != len(set(declared_references))):
            raise ValueError("Working scene references unavailable actual draft material")
        references = [resolve_reference(reference) for reference in declared_references]
        if len(references) != len(set(references)):
            raise ValueError("Two references identify the same actual draft material")
        aliases = [{"alias": alias, "canonical_id": canonical}
                   for alias, canonical in zip(declared_references, references) if alias != canonical]
        declared_parts = row.get("starting_material_parts")
        part_choices = {}
        if declared_parts is not None:
            if not isinstance(declared_parts, list):
                raise ValueError("Explicit starting components must be a list")
            for choice in declared_parts:
                declared_reference = text(choice, "trial_id")
                reference = resolve_reference(declared_reference)
                record = {"alias": declared_reference, "canonical_id": reference}
                if declared_reference != reference and record not in aliases:
                    aliases.append(record)
                if reference == INITIAL_MATERIAL_ID:
                    raise ValueError("Initial complete text has no recorded component scene boundaries")
                ids = choice.get("scene_ids")
                if (reference not in references or reference in part_choices
                        or not isinstance(ids, list) or not ids
                        or any(not isinstance(v, str) or not v.strip() for v in ids)
                        or len(ids) != len(set(ids))):
                    raise ValueError("Invalid explicit starting scene components")
                part_choices[reference] = ids
        task = {"scene_id": identifier, "starting_material_ids": deepcopy(references),
                "rewrite_focus": text(row, "rewrite_focus"),
                "possible_loss": text(row, "possible_loss"), "starting_drafts": []}
        if aliases:
            task["material_reference_aliases"] = aliases
            task["declared_starting_material_ids"] = deepcopy(declared_references)
        if declared_parts is not None:
            task["declared_starting_material_parts"] = deepcopy(declared_parts)
        for reference in references:
            material = available[reference]
            if reference == INITIAL_MATERIAL_ID:
                task["starting_drafts"].append({**deepcopy(material), "original_scene_ids": [],
                    "actual_scene_components": [], "source_material_sha256": material["text_reference"]["sha256"],
                    "component_selection_is_not_a_quality_claim": True})
                continue
            parts = material_scene_parts(material)
            indexed = {part["scene_id"]: part for part in parts}
            single_id = material.get("scene_id")
            scene_ids = list(indexed) if parts else ([single_id] if single_id else [])
            chosen = part_choices.get(reference)
            if chosen is not None:
                chosen = resolve_component_ids(parts, chosen, single_id)
            if parts:
                selected = chosen if chosen is not None else ([identifier] if identifier in indexed else scene_ids)
                selected_parts = [deepcopy(indexed[part_id]) for part_id in selected]
                draft_text = "\n\n".join(part["text"] for part in selected_parts)
                scope = ("explicit_components" if chosen is not None else
                         "same_scene_component" if identifier in indexed else "whole_material_context")
            else:
                selected, selected_parts, draft_text = scene_ids, [], material["text"]
                scope = "single_scene_material" if single_id else "whole_material_context"
            task["starting_drafts"].append({"trial_id": reference,
                "representation": text(material, "representation"),
                "text": draft_text, "original_scene_ids": selected,
                "actual_scene_components": selected_parts, "material_scope": scope,
                "source_material_sha256": fingerprint(material["text"]),
                "component_selection_is_not_a_quality_claim": True})
        tasks[identifier] = task
    result = [tasks[identifier] for identifier in order if identifier in tasks]
    for task in result:
        for draft in task["starting_drafts"]:
            draft["also_used_for_scene_ids"] = [other["scene_id"] for other in result
                if other is not task and draft["trial_id"] in other["starting_material_ids"]]
    return result


def project_complete_reader_input(payload):
    """Read actual complete texts and current decisions without old trace copies."""
    validate_initial_material_references(payload)
    result = {"script": text(payload, "script")}
    for field in ("source", "initial_script", "development_entry"):
        if field in payload:
            result[field] = deepcopy(payload[field])

    def component_reference(component):
        value = deepcopy(component)
        if "text" in value:
            value["actual_text_sha256"] = fingerprint(text(value, "text"))
            del value["text"]
        return value

    def reader_context(context):
        projected = {field: deepcopy(context[field]) for field in (
            "program", "dependency_changes", "unreported_dependency_areas",
            "reported_dependency_areas", "instruction", "retained_initial_material",
            "reader_proposal_decisions", "post_read_program_binding")
            if field in context}
        if "explorations" in context:
            materials = []
            for material in context["explorations"]:
                value = {field: deepcopy(material[field]) for field in (
                    "trial_id", "representation", "scene_id", "text", "presentation_order",
                    "alternative_to", "intended_gain", "possible_loss", "actual_trial_scope",
                    "review_scene_ids") if field in material}
                for field in ("trial_id", "representation", "text"):
                    text(value, field)
                for field in ("encounters", "ordered_scenes", "converted_scenes"):
                    if field not in material:
                        continue
                    for component in material[field]:
                        if text(component, "text") not in value["text"]:
                            raise ValueError("Actual exploration text does not contain its recorded component")
                    value[field] = [component_reference(part) for part in material[field]]
                organization = material.get("expression_organization_exploration")
                if isinstance(organization, dict):
                    value["expression_organization_exploration"] = {
                        field: deepcopy(organization[field]) for field in
                        ("proposal", "integration_instruction") if field in organization}
                materials.append(value)
            projected["explorations"] = materials
        if "scene_working_material" in context:
            projected["scene_working_material"] = deepcopy(context["scene_working_material"])
            available_materials = {row["trial_id"]: row for row in context.get("explorations", [])}
            for task in projected["scene_working_material"]:
                for draft in task.get("starting_drafts", []):
                    if draft.get("trial_id") == INITIAL_MATERIAL_ID:
                        continue
                    material = available_materials.get(draft.get("trial_id"))
                    if (material is None or
                            draft.get("source_material_sha256") != fingerprint(material["text"])):
                        raise ValueError("Reader working material is not bound to an actual exploration")
                    for component in draft.get("actual_scene_components", []):
                        if text(component, "text") not in material["text"]:
                            raise ValueError("Reader starting component is absent from its actual exploration")
                    draft.pop("text", None)
                    if "actual_scene_components" in draft:
                        draft["actual_scene_components"] = [component_reference(part)
                            for part in draft["actual_scene_components"]]
        return projected

    result.update(reader_context(payload))
    if "development_context" in payload:
        result["development_context"] = reader_context(payload["development_context"])
    validate_initial_material_references(result)
    return result


def project_complete_revision_input(payload):
    """Keep complete writing material and actual advice without historical copies."""
    result = project_complete_reader_input(payload)
    for field in ("initial_plan", "whole_reading", "revision_to_attempt"):
        if field in payload:
            result[field] = deepcopy(payload[field])
    return result


def project_final_inspection_input(payload):
    """Inspect the actual before/after texts with current decisions, not old trials."""
    validate_initial_material_references(payload)
    result = {"revised_script": text(payload, "revised_script"),
              "before_script": text(payload, "before_script")}
    for field in ("source", "initial_plan", "initial_script", "development_entry"):
        if field in payload:
            result[field] = deepcopy(payload[field])
    decision_fields = ("program", "dependency_changes", "unreported_dependency_areas",
                       "reported_dependency_areas", "reader_proposal_decisions", "post_read_program_binding")
    result.update({field: deepcopy(payload[field]) for field in decision_fields if field in payload})
    if "development_context" in payload:
        context = payload["development_context"]
        result["development_context"] = {field: deepcopy(context[field])
            for field in decision_fields if field in context}
        if "post_read_program_binding" in context:
            material_view = project_complete_reader_input({**payload, "script": payload["before_script"]})
            for field in ("scene_working_material", "retained_initial_material"):
                result["development_context"][field] = material_view["development_context"][field]
    validate_initial_material_references(result)
    return result


def compile_material_roles(payload, *, current_script_field=None, request_version=None):
    """Label real input fields and prioritize actual text without inventing a brief."""
    validate_initial_material_references(payload)
    result = deepcopy(payload)
    source = result.get("source")
    has_brief = isinstance(source, dict) and "brief" in source
    author_requirements = {"field": "source.brief", "present": has_brief}
    if has_brief:
        author_requirements["value_sha256"] = fingerprint(source["brief"])
    roles = {}
    definitions = {
        "source": "supplied source material; author requirements are confined to its actual brief",
        "initial_plan": "provisional model plan; reconsiderable treatment, not an author prohibition",
        "initial_script": "actual historical complete writing; reusable and revisable",
        "program": "current provisional model writing program; choices remain reconsiderable",
        "original_program": "historical model writing program; no proof of current events",
        "explorations": "actual historical exploratory alternatives; no proof of current events",
        "scene_working_material": "designated revisable actual starting drafts, separated by scene and trial",
        "retained_initial_material": "registered whole initial draft referenced by initial_script field and verified SHA; no inferred component scenes",
        "assembled_script": "historical component assembly; no proof of current events",
        "native_execution": "execution records containing fallible model interpretations",
        "dependency_changes": "proposed model impact decisions; realization is unverified",
        "whole_reading": "fallible model advice about the supplied current text",
        "before_script": "actual preceding complete version used for comparison",
        "reader_proposal_decisions": "latest updater accept/reject/uncertain handling of indexed fallible reader advice; revisable model decisions, not human labels or quality truth",
        "post_read_program_binding": "hash binding of the actual whole reading, current draft, prior and updated program/material decisions; no artistic certification",
    }
    for field, role in definitions.items():
        if field in result:
            roles[field] = role
        context = result.get("development_context")
        if isinstance(context, dict) and field in context:
            roles["development_context." + field] = role
    context = result.get("development_context")
    containers = [("explorations", result.get("explorations"))]
    if isinstance(context, dict):
        containers.append(("development_context.explorations", context.get("explorations")))
    for container, materials in containers:
        if not isinstance(materials, list):
            continue
        for index, material in enumerate(materials):
            if not isinstance(material, dict):
                continue
            prefix = container + "[" + str(index) + "]."
            if "text" in material:
                roles[prefix + "text"] = "actual historical exploratory prose; reconsiderable alternative"
            for field in ("scenic_treatment", "scene_tasks", "executed_program", "relationship_history",
                          "expression_units", "expression_observations"):
                if field in material:
                    roles[prefix + field] = "provisional model treatment or interpretation; no author prohibition or current-event proof"
    if current_script_field is not None:
        text(result, current_script_field)
        roles[current_script_field] = "actual current complete script; locate current events in this text"
    metadata = {"compiler_version": MATERIAL_ROLE_COMPILER_VERSION,
                "binding_author_requirements": author_requirements,
                "field_roles": roles, "current_complete_script_field": current_script_field,
                "model_advice_is_author_requirement": False,
                "historical_material_is_current_event_evidence": False}
    if "retained_initial_material" in result or (isinstance(context, dict) and "retained_initial_material" in context):
        containers = [payload, payload.get("development_context", {})]
        aliases_compiled = any(row.get("material_reference_aliases")
                               for container in containers if isinstance(container, dict)
                               for row in container.get("scene_working_material", []) if isinstance(row, dict))
        metadata["request_version"] = (MATERIAL_WIRE_VERSION if aliases_compiled
                                        else "native-entries-material-development-v11")
    if request_version is not None:
        metadata["request_version"] = request_version
        if "initial_plan" in result:
            roles["initial_plan"] = (
                "actual historical common model plan; reconsiderable guidance, "
                "not current-event evidence or an author prohibition")
    if current_script_field == "script":
        metadata["compatible_local_current_script_alias"] = "draft"
    ordered = {}
    if current_script_field is not None:
        ordered[current_script_field] = result.pop(current_script_field)
    ordered["material_provenance"] = metadata
    ordered.update(result)
    return ordered


def material_development_prompts(prompts):
    """Keep the same creative tasks while making scene-to-whole losses explicit."""
    realize = prompts["realize"].replace("and both exploratory treatments",
                                         "and any designated actual working materials")
    review = (
        "Read this actual complete adaptation as a screenwriter developing its "
        "particular relationships, inner contradictions, lived meaning and rhythm. "
        "Return JSON {strengths_to_preserve:[string],revision_opportunities:"
        "[{location:string,concern:string,proposed_change:string,possible_loss:string}],"
        "uncertainties:[string],supporting_observations:[string]}. Compare current "
        "exchanges with the actual initial draft and trial material. Select at most "
        "two substantial dramatic opportunities worth trying, ordered by their "
        "expected effect on the encounter and surrounding work. Describe an actual "
        "writing choice and its potential gain and loss; adaptation may change "
        "every aspect permitted by the brief. When a working-scene observation "
        "identifies a meaningful loss of voice, response, psychological layers or "
        "lived circumstances, consider a concrete revision in revision_opportunities "
        "so the final writer can act on it. Keeping the present treatment is also "
        "a creative choice, with its gain and loss. Place merely informational, "
        "physical-continuity or formatting observations in supporting_observations; "
        "a missing source detail is no reason to restore it. Only actual brief "
        "requirements bind the work. Prioritize dramatic development, without "
        "requiring more conflict, explanation, dialogue or source fidelity. "
        "Confirm every proposed change against the actual current text."
    )
    final = (
        "Use the actual complete draft, source, brief, development_context and "
        "fallible whole reading to write the strongest complete adaptation. "
        "Return JSON {script:string,applied_changes:[string],rejected_suggestions:"
        "[string]}. First consider the proposed dramatic changes and their losses. "
        "Use or transform actual language, responses, actions and scene placement "
        "to develop particular people, psychological contradictions, relationships, "
        "lived meaning and rhythm. A useful discovery may require changes to "
        "character treatment, preparation, continuation and overall organization. "
        "Retain effective passages and reject advice that harms the whole. "
        "Resolve actual brief requirements and unintended continuity problems; "
        "do not replace creative revision with restoring source details or minor "
        "clarifications. All brief-authorized changes remain available. Preserve "
        "the opening, development and ending; return the entire script without "
        "internal labels or claims of quality. Describe only actually realized "
        "changes in applied_changes."
    )
    return {**prompts, "realize": realize + MATERIAL_REALIZE + MATERIAL_SOURCE_ROLES
                + " For this realization, initial_script is the actual current complete starting draft; "
                  "the revised program and designated exploratory prose guide its development.",
            "review": review + MATERIAL_REVIEW + MATERIAL_SOURCE_ROLES + MATERIAL_CURRENT_SCRIPT,
            "final": final + MATERIAL_FINAL + MATERIAL_SOURCE_ROLES + MATERIAL_CURRENT_SCRIPT
                     + REVISION_ATTEMPT_INSTRUCTION}


def material_realization_context(context):
    """Supply designated working drafts without reinserting discarded alternatives."""
    result = deepcopy(context)
    identifiers = {identifier for row in result["scene_working_material"]
                   for identifier in row["starting_material_ids"]}
    supplied = []
    for material in result["explorations"]:
        if material["trial_id"] not in identifiers:
            continue
        selected = {identifier for row in result["scene_working_material"]
                    for draft in row["starting_drafts"]
                    if draft["trial_id"] == material["trial_id"]
                    for identifier in draft["original_scene_ids"]}
        retained = deepcopy(material)
        parts = material_scene_parts(material)
        if parts:
            selected_parts = [part for part in parts if part["scene_id"] in selected]
            retained["text"] = "\n\n".join(part["text"] for part in selected_parts)
            for field in ("encounters", "ordered_scenes", "converted_scenes"):
                if field in retained:
                    retained[field] = [part for part in retained[field] if part["scene_id"] in selected]
            for field in ("expression_units", "expression_observations", "scene_tasks"):
                if isinstance(retained.get(field), list):
                    retained[field] = [part for part in retained[field] if part.get("scene_id") in selected]
            if isinstance(retained.get("relationship_history"), list):
                events = {part.get("encounter_id") for part in selected_parts}
                retained["relationship_history"] = [part for part in retained["relationship_history"]
                    if part.get("encounter_id") in events]
            if isinstance(retained.get("presentation_order"), list):
                retained["presentation_order"] = [identifier for identifier in retained["presentation_order"]
                    if identifier in selected]
        supplied.append(retained)
    result["explorations"] = supplied
    result["trial_material_role"] = (
        "Designated revisable working drafts; unselected alternatives are retained "
        "for whole reading, not supplied as additional events to realize")
    # All designated prose is present in its actual components. Reintroducing
    # the unfiltered assembly would silently supply discarded scenes again.
    result.pop("assembled_script", None)
    result["working_material_compiler_version"] = WORKING_MATERIAL_COMPILER_VERSION
    return result


def _patch(program, operations):
    compiled, normalizations = compile_program_operations(operations)
    patch = {"base_program_sha256": fingerprint(program), "operations": compiled}
    if normalizations:
        patch["operation_compilation"] = {
            "version": MERGE_REPRESENTATION_COMPILER_VERSION,
            "raw_operations": deepcopy(operations),
            "raw_operations_sha256": fingerprint(operations),
            "compiled_operations_sha256": fingerprint(compiled),
            "normalizations": normalizations}
    return patch


def _material(trial_id, representation, value):
    """Keep controller identities authoritative while retaining raw call outputs."""
    return {**deepcopy(value), "trial_id": str(trial_id), "representation": representation}


def _check_dependency_changes(rows, program, previous_program=None, *, require_complete=True):
    if not isinstance(rows, list):
        raise ValueError("Dependency decisions must be a list")
    if require_complete and (len(rows) != len(AREAS)
            or {row.get("area") for row in rows} != set(AREAS)):
        raise ValueError("Review all four scene-to-whole implications")
    ids = {scene["scene_id"] for scene in program["scenes"]}
    if previous_program is not None:
        ids.update(scene["scene_id"] for scene in validate_program(previous_program)["scenes"])
    for row in rows:
        if row.get("area") not in AREAS:
            raise ValueError("Unknown dependency area")
        if row.get("status") not in {"changed", "retained", "uncertain"}:
            raise ValueError("Unknown dependency decision")
        for key in ("before", "after", "reason"):
            text(row, key)
        if not isinstance(row.get("scene_ids"), list) or not set(row["scene_ids"]) <= ids:
            raise ValueError("Dependency decision references an unavailable scene")


POST_READ_PROGRAM_UPDATE = (
    " Reconsider the CURRENT program and working-material assignments against the "
    "actual current complete script and the fallible whole_reading. Locate claims "
    "in the supplied actual draft, initial draft or trial text before acting. "
    "Maintain compatible choices, reject mistaken advice, or change any brief-authorized "
    "axis; no response, reconciliation, ending or expression channel is required. "
    "Synchronize affected upper-level choices through existing operations. Also return "
    "reader_proposal_decisions:[{reader_priority_index:integer,status:accept|reject|uncertain,"
    "reason:string}], exactly one decision for each actionable revision_opportunities "
    "item (all location, concern, proposed_change and possible_loss fields nonempty), "
    "using its actual zero-based index. accept means this indexed proposal is worth "
    "trying under the updated program; reject means it must not be carried forward "
    "as a required revision; uncertain remains optional advice. These are revisable "
    "writing decisions, not human labels or proof of quality. No complete script is "
    "written in this update."
)


def validate_reader_proposal_decisions(value, reading):
    """Bind model handling decisions to actual actionable reader indices."""
    opportunities = reading.get("revision_opportunities", [])
    actionable = {i for i, row in enumerate(opportunities) if isinstance(row, dict) and all(
        isinstance(row.get(key), str) and row[key].strip()
        for key in ("location", "concern", "proposed_change", "possible_loss"))} if isinstance(opportunities, list) else set()
    if not isinstance(value, list):
        raise ValueError("Post-read update requires explicit reader proposal decisions")
    decisions = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("Reader proposal decisions must be objects")
        index = row.get("reader_priority_index")
        if type(index) is not int or index not in actionable or index in decisions:
            raise ValueError("Reader proposal decision has an unknown or duplicate actual index")
        if row.get("status") not in {"accept", "reject", "uncertain"}:
            raise ValueError("Reader proposal decision has an unknown handling status")
        decisions[index] = {"reader_priority_index": index, "status": row["status"],
                            "reason": text(row, "reason")}
    if set(decisions) != actionable:
        raise ValueError("Post-read update must report handling of every actionable reader proposal")
    return [decisions[index] for index in sorted(decisions)]


def _update_program(available, program, materials, call, *, edges=(), seeds=(), require_complete=True,
                    material_development=False, post_read_state=None):
    scope = dependency_scope(program, program, edges, seeds)
    payload = {**available, "program": program, "explorations": materials,
               "declared_dependencies": list(edges), "review_scene_ids": scope}
    prompt = PROGRAM_UPDATE
    if material_development:
        retained_initial = initial_material_reference(available.get("initial_script"))
        payload["retained_initial_material"] = retained_initial
        prompt += MATERIAL_UPDATE + MATERIAL_SOURCE_ROLES + MATERIAL_CHOICE_UPDATE
        if post_read_state is None:
            payload = compile_material_roles(payload)
        else:
            payload.update(deepcopy(post_read_state))
            payload.pop("revision_to_attempt", None)
            payload = project_complete_revision_input(payload)
            payload["declared_dependencies"] = list(edges)
            payload["review_scene_ids"] = scope
            payload = compile_material_roles(payload,
                current_script_field="script", request_version=POST_READ_PROGRAM_WIRE_VERSION)
            prompt += POST_READ_PROGRAM_UPDATE
    if post_read_state is not None and not material_development:
        raise ValueError("Post-read program reconsideration requires the actual material flow")
    raw = call("post_read_program_edit" if post_read_state is not None else "program_edit",
               "revise_program", prompt, payload)
    patch = _patch(program, raw.get("operations"))
    updated = apply_program_patch(program, patch)
    _check_dependency_changes(raw.get("dependency_changes"), updated, program, require_complete=require_complete)
    additional = raw.get("additional_dependencies")
    if not isinstance(additional, list):
        raise ValueError("New dependency observations must be explicit")
    # Removed scenes remain in the before version for impact inspection.
    revised_scope = dependency_scope(program, updated, list(edges) + additional, seeds)
    result = {"program": updated, "program_patch": patch,
            "dependency_changes": deepcopy(raw["dependency_changes"]),
            "declared_dependencies": list(edges) + additional,
            "review_scene_ids_before": scope, "review_scene_ids_after": revised_scope,
            "dependency_truth_certified": False}
    if material_development:
        result["retained_initial_material"] = deepcopy(retained_initial)
        result["scene_working_material"] = compile_scene_working_material(
            raw.get("scene_working_material"), updated, materials,
            retained_initial_material=retained_initial, initial_script=available["initial_script"])
        result["scene_working_material_quality_certified"] = False
    if post_read_state is not None:
        result["reader_proposal_decisions"] = validate_reader_proposal_decisions(
            raw.get("reader_proposal_decisions"), post_read_state["whole_reading"])
    return result


def reconsider_program_after_reading(available, revision, context, draft, reading, call):
    """Execute one whole-read-informed program update before the existing final writer."""
    program = revision["program"]
    state = {"script": draft, "whole_reading": reading,
             "scene_working_material": revision["scene_working_material"],
             "dependency_changes": revision["dependency_changes"],
             "unreported_dependency_areas": revision["unreported_dependency_areas"],
             "reported_dependency_areas": revision["reported_dependency_areas"]}
    updated = _update_program(available, program, context["explorations"], call,
        seeds=[row["scene_id"] for row in program["scenes"]], material_development=True,
        post_read_state=state)
    reported = {row["area"] for row in updated["dependency_changes"]}
    updated["reported_dependency_areas"] = sorted(reported)
    updated["unreported_dependency_areas"] = sorted(set(AREAS) - reported)
    binding = {"current_script_sha256": fingerprint(draft), "whole_reading_sha256": fingerprint(reading),
               "program_before_sha256": fingerprint(program), "program_after_sha256": fingerprint(updated["program"]),
               "working_material_before_sha256": fingerprint(revision["scene_working_material"]),
               "working_material_after_sha256": fingerprint(updated["scene_working_material"]),
               "actual_trial_library_sha256": fingerprint(context["explorations"]),
               "quality_certified": False}
    new_context = deepcopy(context)
    new_context.pop("revision_to_attempt", None)
    new_context.update({key: deepcopy(updated[key]) for key in (
        "program", "scene_working_material", "retained_initial_material", "dependency_changes",
        "reported_dependency_areas", "unreported_dependency_areas", "program_patch",
        "reader_proposal_decisions")})
    new_context["post_read_program_binding"] = binding
    updated["post_read_program_binding"] = deepcopy(binding)
    accepted = [row["reader_priority_index"] for row in updated["reader_proposal_decisions"]
                if row["status"] == "accept"]
    attempt = compile_revision_attempt(reading, accepted_indices=accepted)
    if attempt is not None:
        attempt["post_read_program_sha256"] = binding["program_after_sha256"]
        attempt["post_read_working_material_sha256"] = binding["working_material_after_sha256"]
    return updated, new_context, attempt


COMPLETE_CHECK = (
    "Read the actual revised COMPLETE screenplay, the earlier complete draft, "
    "the source and brief. Check only accidental repeated passages or events "
    "and unintended continuity resets. Return JSON {decision:use_revision|keep_before|uncertain,"
    "issues:[{kind:duplicate_passage|continuity_conflict|other,passage:string,"
    "repeats_intended:boolean|null,explanation:string}],limitations:[string]}. "
    "Quote an exact substantial repeated passage when one exists. An intentional "
    "reprise, flashback, motif or uncertain interpretation is legitimate and "
    "must not be treated as an error. Do not score aesthetics or source fidelity, "
    "require new conflict, punish an open ending or write a new screenplay. "
    "The program is writing guidance, not truth about the finished text."
)


def select_complete_revision(before, revised, inspection):
    """Roll back only a corroborated accidental copied passage, not taste advice."""
    if inspection.get("decision") not in {"use_revision", "keep_before", "uncertain"}:
        raise ValueError("Unknown complete-script integrity decision")
    issues = inspection.get("issues")
    if not isinstance(issues, list):
        raise ValueError("Integrity observations must be an explicit list")
    corroborated = []
    for issue in issues:
        passage = issue.get("passage")
        if (issue.get("kind") == "duplicate_passage" and issue.get("repeats_intended") is False
                and isinstance(passage, str) and len(passage.strip()) >= 160
                and revised.count(passage) >= 2 and revised.count(passage) > before.count(passage)):
            corroborated.append(deepcopy(issue))
    rollback = inspection["decision"] == "keep_before" and bool(corroborated)
    return {"script": before if rollback else revised,
            "selection": "keep_before" if rollback else "use_revision",
            "corroborated_accidental_copies": corroborated,
            "unverified_advice_did_not_reject_revision": inspection["decision"] == "keep_before" and not rollback,
            "inspection": deepcopy(inspection), "aesthetic_evaluation": False}


def execute_scene_complete(available, call, prompts, *, material_development=False,
                           trial_representation="actual_screenplay"):
    """Execute scene edits, then write/revise complete scripts with stable outputs."""
    if trial_representation not in {"actual_screenplay", "detailed_scene_plan"}:
        raise ValueError("Unknown scene-trial representation")
    if trial_representation != "actual_screenplay" and not material_development:
        raise ValueError("Representation controls require the explicit material-development flow")
    if not isinstance(available.get("initial_script"), str) or not available["initial_script"].strip():
        raise ValueError("Scene development requires the common complete initial draft")
    available = deepcopy(available)
    proposal = call("scene_edit_proposals", "develop", (MATERIAL_SCENE_SEED if material_development else SCENE_SEED) +
        " merge_scenes consumes every listed old scene and produces the named scene; "
        "split_scene replaces its parent with the ordered scenes list. Declare all consumed scenes." +
        (" Choose an unresolved adaptation choice in the actual initial draft whose "
         "different realizations may affect particular relations, inner contradictions "
         "or whole-work rhythm. An ordinary quiet encounter can be central. Preserve "
         "useful existing writing; do not choose by conflict intensity. Distinguish "
         "a mistaken source claim from an authorized new adaptation choice."
         if material_development else ""), available)
    program = deepcopy(validate_program(proposal.get("program", {})))
    focus = text(proposal, "scene_id")
    if focus not in {s["scene_id"] for s in program["scenes"]}:
        raise ValueError("Focal scene must exist in the provisional program")
    edges, edits = proposal.get("dependencies"), proposal.get("edits")
    if not isinstance(edges, list) or not isinstance(edits, list) or len(edits) != 2:
        raise ValueError("Scene development needs two edits and explicit dependencies")
    materials, trace, candidates, revision_seeds = [], [], set(), [focus]
    for i, edit in enumerate(edits):
        patch = _patch(program, edit.get("operations"))
        candidate = apply_program_patch(program, patch)
        identity = fingerprint(candidate)
        if identity == fingerprint(program) or identity in candidates:
            raise ValueError("Actual edit proposals must differ")
        candidates.add(identity)
        targets = (derive_surviving_edit_targets(program, candidate, patch, focus, edges)
                   if material_development else [focus])
        if targets == [focus]:
            if focus not in {s["scene_id"] for s in candidate["scenes"]}:
                raise ValueError("The focal scene must survive its exploration")
            # Keep the established single-focus request and material byte-identical.
            scope = dependency_scope(program, candidate, edges, [focus])
            value = call("scene_trial_" + str(i), "explore", prompts["trial"],
                {**available, "program": candidate, "scene_id": focus,
                 "exploration": {"scene_id": focus, "approach": explicit_edit_annotation(edit, "intended_gain")},
                 "trial_id": str(i), "executed_patch": patch, "review_scene_ids": scope,
                 "scope_instruction": "Write only this named scene; omit preceding and following scenes."})
            if value.get("scene_id") != focus:
                raise ValueError("Actual trial must stay within its named scene")
            text(value, "text")
            material = _material(i, trial_representation, {**value, "executed_patch": patch,
                "executed_program": candidate, "review_scene_ids": scope,
                "intended_gain": explicit_edit_annotation(edit, "intended_gain"),
                "possible_loss": explicit_edit_annotation(edit, "possible_loss")})
            execution = {"operation": ("apply_edit_then_realize_scene" if trial_representation == "actual_screenplay"
                                       else "apply_edit_then_plan_scene"), "patch": patch,
                         "candidate_program_sha256": identity, "review_scene_ids": scope}
        else:
            scope = dependency_scope(program, candidate, edges, targets)
            actual_scenes = []
            targeted_prompt = prompts["trial"] + (
                " The named scene is a surviving target of the already executed edit. "
                "Use the candidate program and actual preceding trial scenes in this "
                "alternative. The program and intended gain propose treatment; only "
                "supplied actual text records writing already realized. The initial "
                "complete draft remains separately available for reconsideration. "
                "Write only the assigned scene and return its scene_id and text.")
            for j, identifier in enumerate(targets):
                actual_scope = {"original_focus_scene_id": focus, "target_scene_ids": targets,
                    "current_scene_index": j, "realized_scene_ids": [s["scene_id"] for s in actual_scenes]}
                key = "scene_trial_" + str(i) + ("_scene_" + str(j) if len(targets) > 1 else "")
                value = call(key, "explore", targeted_prompt,
                    {**available, "program": candidate, "scene_id": identifier,
                     "exploration": {"scene_id": identifier, "approach": explicit_edit_annotation(edit, "intended_gain")},
                     "trial_id": str(i), "executed_patch": patch, "review_scene_ids": scope,
                     "scope_instruction": "Write only this named scene; omit preceding and following scenes.",
                     "actual_trial_scope": actual_scope, "preceding_scenes": deepcopy(actual_scenes),
                     "actual_prefix": "\n\n".join(s["text"] for s in actual_scenes)})
                if value.get("scene_id") != identifier:
                    raise ValueError("Actual trial must stay within its executed edit target")
                actual_scenes.append({"scene_id": identifier, "text": text(value, "text")})
            realized = (deepcopy(actual_scenes[0]) if len(actual_scenes) == 1 else
                        {"text": "\n\n".join(s["text"] for s in actual_scenes)})
            material_scope = {"original_focus_scene_id": focus, "target_scene_ids": deepcopy(targets),
                "realized_scene_ids": [s["scene_id"] for s in actual_scenes],
                "component_role": ("Actual completed trial prose; candidate program and edit intentions remain proposals"
                    if trial_representation == "actual_screenplay" else
                    "Detailed scene plans; no realized screenplay dialogue or observed performance")}
            material = _material(i, trial_representation, {**realized, "ordered_scenes": actual_scenes,
                "executed_patch": patch, "executed_program": candidate, "review_scene_ids": scope,
                "intended_gain": explicit_edit_annotation(edit, "intended_gain"),
                "possible_loss": explicit_edit_annotation(edit, "possible_loss"),
                "actual_trial_scope": material_scope})
            execution = {"operation": ("apply_edit_then_realize_surviving_targets" if trial_representation == "actual_screenplay"
                                       else "apply_edit_then_plan_surviving_targets"), "patch": patch,
                "candidate_program_sha256": identity, "review_scene_ids": scope,
                "actual_trial_scope": material_scope}
            revision_seeds.extend(identifier for identifier in targets
                if identifier in {s["scene_id"] for s in program["scenes"]} and identifier not in revision_seeds)
        materials.append(material)
        trace.append(execution)
    revision = _update_program(available, program, materials, call, edges=edges, seeds=revision_seeds,
                               require_complete=False, material_development=material_development)
    reported = {row["area"] for row in revision["dependency_changes"]}
    revision["reported_dependency_areas"] = sorted(reported)
    revision["unreported_dependency_areas"] = sorted(set(AREAS) - reported)
    context = {"program": revision["program"], "original_program": program,
               "explorations": materials, "dependency_changes": revision["dependency_changes"],
               "unreported_dependency_areas": revision["unreported_dependency_areas"],
               "program_patch": revision["program_patch"], "native_execution": trace,
               "instruction": "Writing decisions are fallible. Missing review areas remain unreported; consider them when reading the actual whole script."}
    if material_development:
        context["scene_working_material"] = revision["scene_working_material"]
        context["retained_initial_material"] = revision["retained_initial_material"]
        prompts = material_development_prompts(prompts)
    realization_context = material_realization_context(context) if material_development else context
    realization_payload = {**available, **realization_context, "selected_trial_id": None}
    if material_development:
        realization_payload = compile_material_roles(realization_payload, current_script_field="initial_script")
    written = call("realize_complete", "realize", prompts["realize"],
        realization_payload)
    draft = text(written, "script")
    reading_payload = {**available, "script": draft, "development_context": context}
    if material_development:
        reading_payload = compile_material_roles(project_complete_reader_input(reading_payload),
            current_script_field="script", request_version=COMPLETE_READER_VIEW_VERSION)
    reading = call("read_complete", "review", prompts["review"],
        reading_payload)
    revision_before_reading, context_before_update, attempt = None, None, None
    if material_development:
        revision_before_reading, context_before_update = deepcopy(revision), deepcopy(context)
        revision, context, attempt = reconsider_program_after_reading(
            available, revision, context, draft, reading, call)
    final_payload = {**available, "script": draft, "whole_reading": reading, "development_context": context}
    if material_development:
        if attempt is not None:
            final_payload["revision_to_attempt"] = attempt
            context["revision_to_attempt"] = deepcopy(attempt)
        final_payload = compile_material_roles(project_complete_revision_input(final_payload),
            current_script_field="script", request_version=MATERIAL_ENTRY_VERSION)
    final = call("revise_complete", "final", prompts["final"],
        final_payload)
    candidate = text(final, "script")
    inspection_payload = {**available, "before_script": draft, "revised_script": candidate, "development_context": context}
    inspection_prompt = COMPLETE_CHECK
    if material_development:
        inspection_payload = compile_material_roles(project_final_inspection_input(inspection_payload),
            current_script_field="revised_script", request_version=MATERIAL_ENTRY_VERSION)
        inspection_prompt += MATERIAL_CURRENT_SCRIPT
    inspection = call("inspect_final_complete", "review", inspection_prompt, inspection_payload)
    selected = select_complete_revision(draft, candidate, inspection)
    context["final_integrity_check"] = selected
    return {"status": "complete", "proposal": proposal, "explorations": materials,
            "revision": revision, "dependency_changes": revision["dependency_changes"],
            **({"revision_before_complete_reading": revision_before_reading,
                "development_context_before_post_read_update": context_before_update,
                "post_read_program_binding": revision["post_read_program_binding"]}
               if material_development else {}),
            "before_whole_integration": draft, "before_integrity_check": candidate,
            "whole_reading": reading, "final_revision": final, "script": selected["script"],
            "integrity_selection": selected,
            **({"initial_script_identity": {"initial_script_sha256": fingerprint(available["initial_script"]),
                 "final_script_sha256": fingerprint(selected["script"]),
                 "retained_initial_script": selected["script"] == available["initial_script"],
                 "retention_is_not_a_new_quality_gain": True}} if material_development else {}), "development_context": context,
            "native_execution": trace, "writer_workflow_version": MATERIAL_ENTRY_VERSION if material_development else COMPLETE_SCENE_VERSION,
            "quality_gain_established": False,
            "final_program_is_guidance_not_verified_text_mapping": True}


def _finish(available, proposal, materials, revision, call, prompts, *, assembled=None, trace=None):
    program = revision["program"]
    payload = {**available, "program": program, "original_program": proposal["program"],
               "explorations": materials, "dependency_changes": revision["dependency_changes"],
               "program_patch": revision["program_patch"], "selected_trial_id": None}
    if assembled is not None:
        payload["assembled_script"] = assembled
    first = call("realize_complete", "realize",
                 "Write the complete adaptation as actual scenes under the committed program. "
                 "Return JSON {scenes:[{scene_id:string,text:string}]}, covering every program scene "
                 "exactly once. Use actual exploratory materials and the assembled script when "
                 "provided, revising them to fit the whole. Fully write dialogue, action and selected "
                 "expression, with a natural endpoint. No program labels or quality claims in scene text.", payload)
    parts = first.get("scenes")
    if not isinstance(parts, list):
        raise ValueError("Complete realization must return actual scene blocks")
    order = [scene["scene_id"] for scene in program["scenes"]]
    draft = assemble_scenes(order, parts)
    context = {"program": program, "explorations": materials,
               "dependency_changes": revision["dependency_changes"],
               "program_patch": revision["program_patch"],
               "native_execution": deepcopy(trace or []),
               "instruction": "Earlier decisions are fallible; read the actual complete draft."}
    reading = call("read_complete", "review", prompts["review"],
                   {**available, "script": draft, "development_context": context})
    final = call("revise_complete", "final",
                 "Improve the complete adaptation using the fallible whole reading. Return JSON "
                 "{operations:" + OPERATIONS + ",scene_updates:[{scene_id:string,base_text_sha256:string|null,"
                 "text:string,reason:string}],applied_changes:[string],rejected_suggestions:[string]}. "
                 "Only replace actual scenes that need revision; unchanged scenes are retained by "
                 "the program. Each replacement contains the full scene text and its supplied version "
                 "hash; newly inserted scenes use a null hash. Operations can add, delete, move or "
                 "change program scenes and all eight edit axes. Newly inserted scenes need actual "
                 "text. Reject advice that harms particular people, psychological layers, relationships, "
                 "lived meaning or rhythm; retaining the script is legitimate. Claims in applied_changes "
                 "must describe actual edits, without quality scores.",
                 {**available, "script": draft, "scenes": parts,
                  "scene_text_versions": {part["scene_id"]: fingerprint(part["text"]) for part in parts},
                  "whole_reading": reading, "development_context": context})
    final_patch = {"base_script_sha256": fingerprint(draft),
                   "program_patch": _patch(program, final.get("operations")),
                   "scene_updates": final.get("scene_updates")}
    committed = apply_script_patch(program, parts, final_patch)
    known_final_ids = {scene["scene_id"] for scene in program["scenes"] + committed["program"]["scenes"]}
    declared_edges = revision.get("declared_dependencies", [])
    active_edges = [edge for edge in declared_edges
                    if edge["from_scene_id"] in known_final_ids and edge["to_scene_id"] in known_final_ids]
    archived_edges = [edge for edge in declared_edges if edge not in active_edges]
    before_texts = {part["scene_id"]: part["text"] for part in parts}
    after_texts = {part["scene_id"]: part["text"] for part in committed["scenes"]}
    actual_changed_ids = sorted(identifier for identifier in known_final_ids
                                if before_texts.get(identifier) != after_texts.get(identifier))
    final_scope = dependency_scope(program, committed["program"], active_edges, actual_changed_ids)
    context["final_program"] = committed["program"]
    context["final_script_patch"] = final_patch
    context["final_dependency_review"] = {"actual_changed_scene_ids": actual_changed_ids,
                                          "review_scene_ids": final_scope,
                                          "active_declared_dependencies": active_edges,
                                          "archived_dependencies_of_earlier_removed_scenes": archived_edges,
                                          "semantic_reinspection_after_final_patch": False}
    return {"status": "complete", "proposal": proposal, "explorations": materials,
            "revision": revision, "dependency_changes": revision["dependency_changes"],
            "before_whole_integration": draft, "whole_reading": reading,
            "final_revision": final, "script": committed["script"],
            "final_program": committed["program"], "final_scenes": committed["scenes"],
            "final_script_patch": final_patch,
            "development_context": context, "native_execution": deepcopy(trace or []),
            "assembled_script": assembled, "writer_workflow_version": VERSION,
            "quality_gain_established": False}


def _finish_complete(available, proposal, materials, revision, call, prompts, *, assembled=None, trace=None):
    """Integrate actual route material through the same complete-script interface."""
    reported = {row["area"] for row in revision["dependency_changes"]}
    revision["reported_dependency_areas"] = sorted(reported)
    revision["unreported_dependency_areas"] = sorted(set(AREAS) - reported)
    context = {"program": revision["program"], "original_program": revision.get("program_before_whole_update", proposal["program"]),
               "explorations": materials, "dependency_changes": revision["dependency_changes"],
               "unreported_dependency_areas": revision["unreported_dependency_areas"],
               "program_patch": revision["program_patch"], "native_execution": deepcopy(trace or []),
               "instruction": "Actual route material and interpretations remain provisional. Integrate, adapt or reject them when writing the whole. Missing review areas remain unreported."}
    if assembled is not None:
        context["assembled_script"] = assembled
    if "scene_working_material" in revision:
        context["scene_working_material"] = revision["scene_working_material"]
        context["retained_initial_material"] = revision["retained_initial_material"]
        prompts = material_development_prompts(prompts)
    material_development = "scene_working_material" in revision
    realization_context = material_realization_context(context) if material_development else context
    realization_payload = {**available, **realization_context, "selected_trial_id": None}
    if material_development:
        realization_payload = compile_material_roles(realization_payload, current_script_field="initial_script")
    first = call("realize_complete", "realize", prompts["realize"],
                 realization_payload)
    draft = text(first, "script")
    reading_payload = {**available, "script": draft, "development_context": context}
    if material_development:
        reading_payload = compile_material_roles(project_complete_reader_input(reading_payload),
            current_script_field="script", request_version=COMPLETE_READER_VIEW_VERSION)
    reading = call("read_complete", "review", prompts["review"],
                    reading_payload)
    revision_before_reading, context_before_update, attempt = None, None, None
    if material_development:
        revision_before_reading, context_before_update = deepcopy(revision), deepcopy(context)
        revision, context, attempt = reconsider_program_after_reading(
            available, revision, context, draft, reading, call)
    final_payload = {**available, "script": draft, "whole_reading": reading, "development_context": context}
    if material_development:
        if attempt is not None:
            final_payload["revision_to_attempt"] = attempt
            context["revision_to_attempt"] = deepcopy(attempt)
        final_payload = compile_material_roles(project_complete_revision_input(final_payload),
            current_script_field="script", request_version=MATERIAL_ENTRY_VERSION)
    final = call("revise_complete", "final", prompts["final"],
                  final_payload)
    candidate = text(final, "script")
    inspection_payload = {**available, "before_script": draft, "revised_script": candidate, "development_context": context}
    inspection_prompt = COMPLETE_CHECK
    if material_development:
        inspection_payload = compile_material_roles(project_final_inspection_input(inspection_payload),
            current_script_field="revised_script", request_version=MATERIAL_ENTRY_VERSION)
        inspection_prompt += MATERIAL_CURRENT_SCRIPT
    inspection = call("inspect_final_complete", "review", inspection_prompt, inspection_payload)
    selected = select_complete_revision(draft, candidate, inspection)
    context["final_integrity_check"] = selected
    return {"status": "complete", "proposal": proposal, "explorations": materials,
            "revision": revision, "dependency_changes": revision["dependency_changes"],
            **({"revision_before_complete_reading": revision_before_reading,
                "development_context_before_post_read_update": context_before_update,
                "post_read_program_binding": revision["post_read_program_binding"]}
               if material_development else {}),
            "before_whole_integration": draft, "before_integrity_check": candidate,
            "whole_reading": reading, "final_revision": final, "script": selected["script"],
            "integrity_selection": selected,
            **({"initial_script_identity": {"initial_script_sha256": fingerprint(available["initial_script"]),
                 "final_script_sha256": fingerprint(selected["script"]),
                 "retained_initial_script": selected["script"] == available["initial_script"],
                 "retention_is_not_a_new_quality_gain": True}} if material_development else {}), "development_context": context,
            "native_execution": deepcopy(trace or []), "assembled_script": assembled,
            "writer_workflow_version": MATERIAL_ENTRY_VERSION if "scene_working_material" in revision else COMPLETE_ENTRY_VERSION,
            "quality_gain_established": False,
            "final_program_is_guidance_not_verified_text_mapping": True}


def execute_entry_complete(entry, available, call, prompts):
    """Run four distinct development methods with shared complete creative tasks."""
    if entry == "scene_first":
        result = execute_scene_complete(available, call, prompts)
        result["writer_workflow_version"] = COMPLETE_ENTRY_VERSION
        return result
    return execute_entry(entry, available, call, prompts, complete_flow=True)


def execute_entry_material_complete(entry, available, call, prompts):
    """Develop actual trial drafts into a whole script under explicit decisions."""
    if entry == "scene_first":
        return execute_scene_complete(available, call, prompts, material_development=True)
    return execute_entry(entry, available, call, prompts, complete_flow=True, material_development=True)


def execute_entry(entry, available, call, prompts, *, complete_flow=False, material_development=False):
    """Run an entry's control flow using phase-labelled, injected writing calls."""
    if entry not in PROVENANCE:
        raise ValueError("Unknown native entry")
    if material_development and not complete_flow:
        raise ValueError("Working-material development requires the complete-script workflow")
    if not isinstance(available.get("initial_script"), str) or not available["initial_script"].strip():
        raise ValueError("Native entry workflows require an explicit common initial draft")
    available = deepcopy(available)
    trace, materials = [], []
    finish = _finish_complete if complete_flow else _finish
    if entry == "scene_first":
        proposal = call("scene_edit_proposals", "develop", SCENE_SEED, available)
        program = deepcopy(validate_program(proposal.get("program", {})))
        focus = text(proposal, "scene_id")
        if focus not in {s["scene_id"] for s in program["scenes"]}:
            raise ValueError("Focal scene must belong to the provisional program before candidate edits")
        edges = proposal.get("dependencies")
        edits = proposal.get("edits")
        if not isinstance(edges, list) or not isinstance(edits, list) or len(edits) != 2:
            raise ValueError("Scene entry requires declared dependencies and two executable edits")
        fingerprints = set()
        for i, edit in enumerate(edits):
            patch = _patch(program, edit.get("operations"))
            candidate = apply_program_patch(program, patch)
            if fingerprint(candidate) == fingerprint(program) or fingerprint(candidate) in fingerprints:
                raise ValueError("Scene edit proposals must make distinct actual program changes")
            fingerprints.add(fingerprint(candidate))
            if focus not in {s["scene_id"] for s in candidate["scenes"]}:
                raise ValueError("An explored focal scene must survive its edit")
            for key in ("intended_gain", "possible_loss"):
                text(edit, key)
            scope = dependency_scope(program, candidate, edges, [focus])
            value = call("scene_trial_" + str(i), "explore",
                         "Write the actual complete focal screenplay scene under this executed edit. "
                         "Return JSON {scene_id:string,text:string}. Realize its particular people, "
                         "circumstances and expression; preserve no mandatory participants or source actions.",
                         {**available, "program": candidate, "scene_id": focus,
                          "executed_patch": patch, "review_scene_ids": scope})
            if value.get("scene_id") != focus:
                raise ValueError("Trial changed the executed focal scene")
            text(value, "text")
            materials.append(_material(i, "actual_screenplay", {**value, "executed_patch": patch,
                                      "executed_program": candidate, "review_scene_ids": scope}))
            trace.append({"operation": "apply_edit_then_realize_scene", "patch": patch,
                          "candidate_program_sha256": fingerprint(candidate), "review_scene_ids": scope,
                          "actual_scene_sha256": fingerprint(value["text"])})
        revision = _update_program(available, program, materials, call, edges=edges, seeds=[focus])
        # Re-read the impacts after committed edits, including newly found links.
        audit = call("inspect_updated_impacts", "revise_program",
                     "Read the original program, committed program edits and actual scene trials. "
                     "Inspect character treatment, preceding preparation, subsequent development and "
                     "scene organization after the edit. Return JSON {operations:" + OPERATIONS + ","
                     "dependency_changes:[{area:string,status:changed|retained|uncertain,before:string,"
                     "after:string,reason:string,scene_ids:[string]}],additional_dependencies:" + DEPENDENCIES + "}. "
                     "Report newly needed edits or explicitly retain the committed decisions. "
                     "No conflict-count or fidelity reward; declared dependencies can be wrong.",
                     {**available, "original_program": program, "committed_revision": revision,
                      "explorations": materials})
        second_patch = _patch(revision["program"], audit.get("operations"))
        audited = apply_program_patch(revision["program"], second_patch)
        _check_dependency_changes(audit.get("dependency_changes"), audited, program)
        extra = audit.get("additional_dependencies")
        if not isinstance(extra, list):
            raise ValueError("Post-edit dependencies must remain explicit")
        dependencies = revision["declared_dependencies"] + extra
        scope = dependency_scope(program, audited, dependencies, [focus])
        trace.append({"operation": "inspect_impacts_after_edit", "patch": second_patch,
                      "review_scene_ids": scope})
        revision.update(program=audited, impact_audit_patch=second_patch,
                        program_patch=_patch(program, revision["program_patch"]["operations"] + second_patch["operations"]),
                        dependency_changes=audit["dependency_changes"], declared_dependencies=dependencies,
                        review_scene_ids_after=scope)
        return finish(available, proposal, materials, revision, call, prompts, trace=trace)
    if entry == "relationships_first":
        proposal = call("relationship_state", "develop", RELATION_SEED, available)
        program = deepcopy(validate_program(proposal.get("program", {})))
        people, state, chains = proposal.get("people"), proposal.get("relationships"), proposal.get("encounter_chains")
        if (not isinstance(people, list) or not people or len(people) != len(set(people))
                or any(not isinstance(p, str) or not p.strip() for p in people)
                or not isinstance(state, list) or not state
                or not isinstance(chains, list) or len(chains) != 2):
            raise ValueError("Relationship entry requires people, directed records and two chains")
        relation_ids = set()
        for relation in state:
            identifier = text(relation, "relation_id")
            if identifier in relation_ids or relation.get("from_person") not in people or relation.get("to_person") not in people:
                raise ValueError("Invalid directed relationship")
            relation_ids.add(identifier)
            text(relation, "interpretation")
            text(relation, "uncertainty")
        scene_ids = {s["scene_id"] for s in program["scenes"]}
        for i, chain in enumerate(chains):
            encounters = chain.get("encounters")
            text(chain, "approach")
            if not isinstance(encounters, list) or len(encounters) < 2:
                raise ValueError("Relationship development needs linked actual encounters")
            current, written, updates, event_ids = deepcopy(state), [], [], set()
            written_scene_ids = set()
            for encounter in encounters:
                identifier = text(encounter, "encounter_id")
                if identifier in event_ids or encounter.get("scene_id") not in scene_ids:
                    raise ValueError("Invalid encounter reference")
                event_ids.add(identifier)
                if complete_flow and encounter["scene_id"] in written_scene_ids:
                    raise ValueError("A chain needs separate scene identifiers for separate complete encounters")
                written_scene_ids.add(encounter["scene_id"])
                text(encounter, "situation")
                before = deepcopy(current)
                payload = {**available, "program": program, "encounter": encounter,
                           "current_relationships": before, "previous_actual_encounters": written,
                           "chain_approach": chain["approach"]}
                if complete_flow:
                    payload.update(scene_id=encounter["scene_id"], exploration={
                        "scene_id": encounter["scene_id"], "approach": chain["approach"] + ": " + encounter["situation"]},
                        previous_relationship_interpretations=deepcopy(updates))
                value = call("encounter_" + str(i) + "_" + identifier, "explore",
                             RELATION_SCENE if complete_flow else RELATION_WRITE, payload)
                text(value, "text")
                if complete_flow:
                    if value.get("scene_id") != encounter["scene_id"]:
                        raise ValueError("Encounter writer changed its assigned scene")
                    interpreted = call("interpret_encounter_" + str(i) + "_" + identifier, "revise_program",
                                       RELATION_INTERPRET, {**payload, "actual_encounter": {**encounter, "text": value["text"]}})
                    changes = interpreted.get("relationship_updates")
                else:
                    changes = value.get("relationship_updates")
                current = commit_relationship_updates(current, changes, identifier)
                written.append({**encounter, "text": value["text"]})
                updates.append({"encounter_id": identifier, "before": before, "after": deepcopy(current),
                                "actual_text_sha256": fingerprint(value["text"])})
                if complete_flow:
                    updates[-1]["interpretation_updates"] = deepcopy(changes)
                    updates[-1]["psychological_truth_certified"] = False
            materials.append(_material(i, "encounter_sequence", {"text": "\n\n".join(x["text"] for x in written),
                                      "encounters": written, "relationship_history": updates}))
            trace.append({"operation": "sequential_encounter_state_updates", "trial_id": str(i),
                          "relationship_history": updates, "psychological_truth_certified": False})
        revision = _update_program(available, program, materials, call,
                                   seeds=sorted({e["scene_id"] for c in chains for e in c["encounters"]}),
                                   require_complete=not complete_flow, material_development=material_development)
        return finish(available, proposal, materials, revision, call, prompts, trace=trace)
    if entry == "structure_first":
        proposal = call("hierarchical_arrangement", "develop", STRUCTURE_SEED, available)
        program = deepcopy(validate_program(proposal.get("program", {})))
        sections = proposal.get("sections")
        constraints = proposal.get("presentation_constraints")
        if not isinstance(sections, list) or not sections or not isinstance(constraints, list):
            raise ValueError("Structure entry requires sections and explicit presentation constraints")
        original_order = [scene["scene_id"] for scene in program["scenes"]]
        flattened, section_for, section_ids = [], {}, set()
        for section in sections:
            section_id = text(section, "section_id")
            if section_id in section_ids:
                raise ValueError("Duplicate section identifier")
            section_ids.add(section_id)
            text(section, "purpose")
            ids = section.get("scene_ids")
            if not isinstance(ids, list) or not ids or any(identifier not in original_order for identifier in ids):
                raise ValueError("Invalid section scenes")
            flattened.extend(ids)
            for identifier in ids:
                section_for[identifier] = section
        if len(flattened) != len(set(flattened)) or set(flattened) != set(original_order):
            raise ValueError("Sections must partition the complete program")
        order = presentation_order(flattened, constraints)
        if complete_flow:
            indexed = {scene["scene_id"]: scene for scene in program["scenes"]}
            program["scenes"] = [indexed[identifier] for identifier in order]
            trace.append({"operation": "apply_explicit_presentation_order", "presentation_order": order,
                          "provisional_program_sha256": fingerprint(program),
                          "causal_order_certified": False})
        tasks, parts = {}, []
        for identifier in order:
            section = section_for[identifier]
            if complete_flow or identifier not in tasks:
                pending = [s for s in section["scene_ids"] if s not in {p["scene_id"] for p in parts}]
                requested = [identifier] if complete_flow else pending
                expanded_payload = {**available, "program": program, "section": {**section, "scene_ids": requested},
                                    "presentation_order": order, "actual_prefix": "\n\n".join(p["text"] for p in parts)}
                if complete_flow:
                    expanded_payload.update(remaining_section_scene_ids=pending,
                                            previously_expanded_tasks=deepcopy(list(tasks.values())))
                expanded = call("expand_" + section["section_id"] + "_" + identifier, "develop", STRUCTURE_EXPAND,
                                expanded_payload)
                values = expanded.get("scene_tasks")
                expanded_ids = [v.get("scene_id") for v in values] if isinstance(values, list) else []
                allowed = set(pending if complete_flow else requested)
                if (not isinstance(values, list) or not values
                        or len(expanded_ids) != len(set(expanded_ids))
                        or not set(requested) <= set(expanded_ids)
                        or not set(expanded_ids) <= allowed):
                    raise ValueError("Expanded tasks must cover the pending section scenes exactly")
                for task in values:
                    for field in ("development", "preparation", "later_resonance"):
                        text(task, field)
                    tasks[task["scene_id"]] = task
                task_patch = None
                if complete_flow:
                    task_patch = _patch(program, [{"op": "replace_scene", "scene_id": identifier,
                                                  "development": tasks[identifier]["development"]}])
                    program = apply_program_patch(program, task_patch)
                trace.append({"operation": "expand_section_after_actual_prefix", "section_id": section["section_id"],
                              "actual_prefix_sha256": fingerprint("\n\n".join(p["text"] for p in parts)),
                              "scene_ids": requested})
                if complete_flow:
                    trace[-1]["working_program_patch"] = task_patch
                    trace[-1]["provisional_future_scene_ids"] = [sid for sid in expanded_ids if sid not in requested]
                    trace[-1]["future_scene_tasks_refreshed_when_reached"] = True
            scene_payload = {**available, "program": program, "scene_task": tasks[identifier],
                             "presentation_order": order, "actual_prefix": "\n\n".join(p["text"] for p in parts)}
            if complete_flow:
                scene_payload.update(scene_id=identifier, exploration={"scene_id": identifier,
                                     "approach": tasks[identifier]["development"]})
            value = call("structure_scene_" + identifier, "explore", STRUCTURE_WRITE,
                         scene_payload)
            if value.get("scene_id") != identifier:
                raise ValueError("Hierarchical writer changed its assigned scene")
            text(value, "text")
            parts.append(value)
        assembled = assemble_scenes(order, parts)
        materials = [_material(0, "hierarchical_scene_sequence", {"text": assembled, "ordered_scenes": parts,
                               "presentation_order": order, "scene_tasks": list(tasks.values())})]
        revision = _update_program(available, program, materials, call, seeds=order,
                                   require_complete=not complete_flow, material_development=material_development)
        if complete_flow:
            revision["program_before_whole_update"] = deepcopy(program)
        return finish(available, proposal, materials, revision, call, prompts, assembled=assembled, trace=trace)
    proposal = call("scenic_treatment", "develop", EXPRESSION_SEED, available)
    program = deepcopy(validate_program(proposal.get("program", {})))
    treatment = text(proposal, "scenic_treatment")
    distribution = call("expression_distribution", "develop", EXPRESSION_DISTRIBUTE,
                        {**available, "program": program, "scenic_treatment": treatment})
    units = distribution.get("units")
    order = [scene["scene_id"] for scene in program["scenes"]]
    if not isinstance(units, list) or not units:
        raise ValueError("Expression conversion needs actual allocated material")
    unit_ids = set()
    grouped = {identifier: [] for identifier in order}
    for unit in units:
        identifier = text(unit, "unit_id")
        if identifier in unit_ids or unit.get("scene_id") not in grouped:
            raise ValueError("Invalid expression unit")
        unit_ids.add(identifier)
        for field in ("medium_element", "actual_material", "intended_experience", "possible_loss"):
            text(unit, field)
        grouped[unit["scene_id"]].append(unit)
    if any(not values for values in grouped.values()):
        raise ValueError("Expression distribution omitted a complete program scene")
    parts, observations = [], []
    for identifier in order:
        payload = {**available, "program": program, "scenic_treatment": treatment,
                   "scene_id": identifier, "expression_units": grouped[identifier],
                   "actual_prefix": "\n\n".join(p["text"] for p in parts)}
        if complete_flow:
            payload.update(exploration={"scene_id": identifier, "approach": "Convert this scene's actual medium-appropriate expression material"},
                           previous_expression_observations=deepcopy(observations))
        value = call("convert_expression_" + identifier, "explore",
                     EXPRESSION_SCENE if complete_flow else EXPRESSION_CONVERT, payload)
        if value.get("scene_id") != identifier:
            raise ValueError("Expression conversion changed its assigned scene")
        if complete_flow:
            text(value, "text")
            reading = call("read_expression_" + identifier, "revise_program", EXPRESSION_READ,
                           {**payload, "actual_scene": {"scene_id": identifier, "text": value["text"]}})
            dispositions = reading.get("unit_dispositions")
            observations.append({"scene_id": identifier, "actual_scene_sha256": fingerprint(value["text"]),
                                 "unit_dispositions": deepcopy(dispositions), "realization_certified": False})
        else:
            dispositions = value.get("unit_dispositions")
        if (not isinstance(dispositions, list) or len(dispositions) != len(grouped[identifier])
                or {d.get("unit_id") for d in dispositions} != {u["unit_id"] for u in grouped[identifier]}):
            raise ValueError("Every actual expression unit needs a disposition")
        for disposition in dispositions:
            permitted = {"used", "revised", "rejected", "uncertain"} if complete_flow else {"used", "revised", "rejected"}
            if disposition.get("status") not in permitted:
                raise ValueError("Unknown expression disposition")
            text(disposition, "reason")
        text(value, "text")
        parts.append(value)
        trace.append({"operation": "convert_allocated_expression", "scene_id": identifier,
                      "unit_dispositions": dispositions, "actual_scene_sha256": fingerprint(value["text"])})
    assembled = assemble_scenes(order, parts)
    materials = [_material(0, "expression_conversion", {"text": assembled, "scenic_treatment": treatment,
                           "expression_units": units, "converted_scenes": parts})]
    if complete_flow:
        materials[0]["expression_observations"] = observations
    if material_development:
        organization = call("expression_organization_proposal", "develop", EXPRESSION_ORGANIZATION_PROPOSAL,
            {"converted_scene_sequence": {"text": assembled, "scenes": deepcopy(parts), "presentation_order": order},
             **available, "program": program, "scenic_treatment": treatment,
             "expression_units": deepcopy(units), "expression_observations": deepcopy(observations)})
        target = text(organization, "scene_id")
        if target not in order:
            raise ValueError("Alternative expression proposal names an unavailable scene")
        for field in ("expression_change", "intended_gain", "possible_loss"):
            text(organization, field)
        position = order.index(target)
        alternative_payload = expression_scene_payload({**available, "program": program, "scene_id": target,
            "scenic_treatment": treatment, "expression_units": grouped[target],
            "actual_prefix": "\n\n".join(part["text"] for part in parts[:position]),
            "previous_expression_observations": deepcopy(observations[:position])}, parts[position])
        alternative_payload["expression_alternative"] = deepcopy(organization)
        alternative_payload["converted_scene_sequence"] = deepcopy(parts)
        alternative_payload["actual_following_scenes"] = deepcopy(parts[position + 1:])
        alternative_payload["scene_material_roles"]["field_roles"].update({
            "expression_alternative": "Provisional proposed different treatment, not an author requirement",
            "converted_scene_sequence": "Actual original converted sequence, separate from the alternative",
            "actual_following_scenes": "Actual original following scenes for context, not new scenes or binding requirements"})
        alternative = call("alternative_expression_scene", "explore", EXPRESSION_ALTERNATIVE_SCENE, alternative_payload)
        if alternative.get("scene_id") != target:
            raise ValueError("Alternative expression writer changed its assigned scene")
        alternative = {"scene_id": target, "text": text(alternative, "text")}
        actual_treatments = [{"trial_id": "0", "scene_id": target, "text": parts[position]["text"]},
                             {"trial_id": "1", **deepcopy(alternative)}]
        reading = call("read_expression_alternatives", "revise_program", EXPRESSION_ALTERNATIVE_READ,
            {"actual_treatments": actual_treatments, **available, "program": program,
             "converted_scene_sequence": deepcopy(parts), "actual_prefix": alternative_payload["actual_prefix"],
             "scenic_treatment": treatment, "expression_units": deepcopy(units),
             "actual_following_scenes": deepcopy(parts[position + 1:]),
             "expression_alternative": deepcopy(organization)})
        rows = reading.get("treatments")
        required_reading_fields = {"trial_id", "scene_id", "experience", "gains_and_losses"}
        if (set(reading) != {"treatments", "whole_implications"} or not isinstance(rows, list) or len(rows) != 2
                or any(not isinstance(row, dict) or not required_reading_fields <= set(row)
                       or not set(row) <= required_reading_fields | {"uncertainty"}
                       or row.get("scene_id") != target for row in rows)
                or {row.get("trial_id") for row in rows} != {"0", "1"}):
            raise ValueError("Read both actual expression alternatives once without scores or certification")
        unreported_reading_fields = []
        for row in rows:
            for field in ("experience", "gains_and_losses"):
                text(row, field)
            if "uncertainty" in row:
                text(row, "uncertainty")
            else:
                unreported_reading_fields.append({"trial_id": row["trial_id"], "scene_id": target,
                                                  "fields": ["uncertainty"]})
        text(reading, "whole_implications")
        materials[0]["expression_organization_exploration"] = {"proposal": deepcopy(organization),
            "actual_treatment_reading": deepcopy(reading), "reading_is_fallible": True,
            "unreported_reading_fields": unreported_reading_fields,
            "integration_instruction": "Treat these as alternatives of one scene. Combining their useful expression must not make mutually exclusive replies or events occur twice.",
            "reading_targets": [{"trial_id": row["trial_id"], "scene_id": target,
                                 "actual_scene_sha256": fingerprint(row["text"])} for row in actual_treatments]}
        materials.append(_material(1, "alternative_expression_scene", {"scene_id": target,
            "text": alternative["text"], "ordered_scenes": [deepcopy(alternative)], "presentation_order": [target],
            "alternative_to": {"trial_id": "0", "scene_id": target, "actual_scene_sha256": fingerprint(parts[position]["text"])}}))
        trace.append({"operation": "explore_alternative_expression_organization", "scene_id": target,
            "original_scene_sha256": fingerprint(parts[position]["text"]),
            "alternative_scene_sha256": fingerprint(alternative["text"]),
            "actual_prefix_sha256": fingerprint(alternative_payload["actual_prefix"]),
            "alternative_text_changed": alternative["text"] != parts[position]["text"],
            "original_sequence_replaced": False, "quality_gain_established": False})
    revision = _update_program(available, program, materials, call, seeds=order,
                               require_complete=not complete_flow, material_development=material_development)
    return finish(available, proposal, materials, revision, call, prompts, assembled=assembled, trace=trace)


def scene_candidate_search(initial, expand, compare, *, validation, rounds=1, frontier_size=2):
    """Search complete candidates with a qualified comparator and incumbent retention.

    compare(a, b) returns left, right, tie, or unavailable. It must compare
    complete scripts for the same writing task. Cycles are retained as
    unresolved preferences rather than converted into an invented scalar.
    """
    from run_scene_first_matched import validate_evaluator
    binding = validate_evaluator(validation)
    if (isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1
            or isinstance(frontier_size, bool) or not isinstance(frontier_size, int) or frontier_size < 1):
        raise ValueError("Search requires explicit positive integer controls")
    if not callable(expand) or not callable(compare):
        raise ValueError("Search requires explicit expansion and validated comparison callbacks")
    source = initial.get("source_sha256")
    if not source or initial.get("status") != "complete":
        raise ValueError("Search requires a complete, source-bound incumbent")
    text(initial, "script")
    frontier, history, incumbent = [deepcopy(initial)], [], deepcopy(initial)
    for round_index in range(rounds):
        candidates = [deepcopy(incumbent)] + [deepcopy(x) for x in frontier]
        failures = []
        for parent in frontier:
            children = expand(deepcopy(parent), round_index)
            if not isinstance(children, list):
                raise ValueError("Expansion must return candidate records")
            for child in children:
                if child.get("status") != "complete":
                    failures.append(deepcopy(child))
                    continue
                if child.get("source_sha256") != source:
                    raise ValueError("Search cannot compare different adaptation tasks")
                text(child, "script")
                candidates.append(deepcopy(child))
        unique = {}
        for candidate in candidates:
            unique.setdefault(fingerprint(candidate["script"]), candidate)
        candidates = list(unique.values())
        outcomes, beaten = [], set()
        wins = {i: set() for i in range(len(candidates))}
        for i, left in enumerate(candidates):
            for j in range(i + 1, len(candidates)):
                choice = compare(deepcopy(left), deepcopy(candidates[j]))
                if choice not in {"left", "right", "tie", "unavailable"}:
                    raise ValueError("Comparator returned an unsupported preference")
                outcomes.append({"left": fingerprint(left["script"]), "right": fingerprint(candidates[j]["script"]), "choice": choice})
                if choice == "left":
                    beaten.add(j)
                    wins[i].add(j)
                elif choice == "right":
                    beaten.add(i)
                    wins[j].add(i)
        survivors = [candidate for i, candidate in enumerate(candidates) if i not in beaten]
        incumbent_sha = fingerprint(incumbent["script"])
        incumbent_index = next(i for i, candidate in enumerate(candidates)
                               if fingerprint(candidate["script"]) == incumbent_sha)
        preferred_survivors = [i for i in range(len(candidates))
                               if i not in beaten and incumbent_index in wins[i]]
        decisive = [i for i in range(len(candidates)) if len(wins[i]) == len(candidates) - 1]
        if len(candidates) > 1 and len(decisive) == 1:
            incumbent = deepcopy(candidates[decisive[0]])
            decision = "candidate_preferred_to_every_alternative"
        elif not survivors:
            decision = "cyclic_preferences_keep_incumbent"
            survivors = [deepcopy(incumbent)]
        elif preferred_survivors:
            # The candidate order is fixed before judging. A missing or tied
            # comparison between improvements creates no artistic ranking.
            incumbent = deepcopy(candidates[preferred_survivors[0]])
            decision = "undefeated_candidate_directly_preferred_to_incumbent"
        else:
            decision = "unresolved_preferences_keep_incumbent"
        # Stable retention is a computational convention, not an artistic rank.
        frontier = survivors[:frontier_size]
        history.append({"round": round_index, "outcomes": outcomes, "failures": failures,
                        "decision": decision, "previous_incumbent_sha256": incumbent_sha,
                        "retained_script_sha256": fingerprint(incumbent["script"]),
                        "directly_preferred_undefeated_candidates": [fingerprint(candidates[i]["script"])
                                                                    for i in preferred_survivors],
                        "unranked_improvements_resolved_by_fixed_candidate_order":
                            decision == "undefeated_candidate_directly_preferred_to_incumbent",
                        "unresolved_survivors": [fingerprint(x["script"]) for x in survivors]})
    return {"selected": incumbent, "search_history": history, "evaluator_validation": binding,
            "quality_gain_established": False, "selection_origin": "validated_model_preference"}
