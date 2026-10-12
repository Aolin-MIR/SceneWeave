"""Synthetic controller checks without inference or aesthetic labels."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import adaptation_entry_algorithms as alg
def program():
    return {"intent": "A particular uneasy relationship", "choices": {axis: "Open treatment" for axis in alg.EDIT_AXES},
            "scenes": [{"scene_id": "s1", "development": "First encounter"},
                       {"scene_id": "s2", "development": "Later encounter"}]}

class FixtureWriter:
    """Supply deterministic stage outputs, not simulated human taste."""
    def __init__(self):
        self.calls = []
        self.program = program()

    def __call__(self, key, phase, prompt, payload):
        self.calls.append({"key": key, "phase": phase, "input": deepcopy(payload)})
        changes = [{"area": area, "status": "retained", "before": "Specific earlier decision",
                    "after": "Specific earlier decision", "reason": "Keep this particular treatment",
                    "scene_ids": ["s1", "s2"]} for area in alg.AREAS]
        if key == "scene_edit_proposals":
            return {"program": deepcopy(self.program), "scene_id": "s1", "dependencies": [
                    {"from_scene_id": "s1", "to_scene_id": "s2", "area": "subsequent_development", "reason": "Later reply"}],
                    "edits": [{"operations": [{"op": "replace_scene", "scene_id": "s1", "development": value}],
                               "intended_gain": "Explore a particular exchange", "possible_loss": "Different emphasis"}
                              for value in ("Offer help", "Withdraw the offer")]}
        if key.startswith("scene_trial_"):
            return {"scene_id": "s1", "text": "ACTUAL: " + payload["program"]["scenes"][0]["development"]}
        if key == "relationship_state":
            return {"program": deepcopy(self.program), "people": ["P", "Q"], "relationships": [
                    {"relation_id": "pq", "from_person": "P", "to_person": "Q", "interpretation": "Earlier interpretation", "uncertainty": "Several motives"}],
                    "encounter_chains": [{"approach": label, "encounters": [
                        {"encounter_id": "e1", "scene_id": "s1", "situation": "Offer"},
                        {"encounter_id": "e2", "scene_id": "s2", "situation": "Reply"}]} for label in ("One", "Two")]}
        if key.startswith("encounter_"):
            before = payload["current_relationships"][0]["interpretation"]
            after = before + " after " + payload["encounter"]["encounter_id"]
            return {"text": "ACTUAL ENCOUNTER " + after, "relationship_updates": [
                    {"relation_id": "pq", "before": before, "after": after, "status": "uncertain",
                     "reason": "A revisable reading of the actual encounter", "uncertainty": "Still ambiguous"}]}
        if key == "hierarchical_arrangement":
            return {"program": deepcopy(self.program), "sections": [
                    {"section_id": "b1", "purpose": "First section", "scene_ids": ["s1"]},
                    {"section_id": "b2", "purpose": "Second section", "scene_ids": ["s2"]}],
                    "presentation_constraints": [{"before": "s2", "after": "s1", "kind": "presentation", "reason": "Recall later"}]}
        if key.startswith("expand_"):
            return {"scene_tasks": [{"scene_id": identifier, "development": "Concrete expanded scene",
                     "preparation": "Particular setup", "later_resonance": "Particular echo"}
                    for identifier in payload["section"]["scene_ids"]]}
        if key.startswith("structure_scene_"):
            return {"scene_id": payload["scene_task"]["scene_id"], "text": "ACTUAL STRUCTURE " + payload["scene_task"]["scene_id"]}
        if key == "scenic_treatment":
            return {"program": deepcopy(self.program), "scenic_treatment": "Actual treatment of silence and offers"}
        if key == "expression_distribution":
            return {"units": [{"unit_id": "u" + str(i), "scene_id": identifier, "medium_element": "Voice",
                     "actual_material": "A particular spoken line " + identifier, "intended_experience": "Uneasy care",
                     "possible_loss": "More explicit"} for i, identifier in enumerate(("s1", "s2"))]}
        if key.startswith("convert_expression_"):
            return {"scene_id": payload["scene_id"], "text": payload["expression_units"][0]["actual_material"],
                    "unit_dispositions": [{"unit_id": unit["unit_id"], "status": "used", "reason": "Works here"}
                                          for unit in payload["expression_units"]]}
        if key in {"program_edit", "inspect_updated_impacts"}:
            return {"operations": [], "dependency_changes": changes, "additional_dependencies": []}
        if key == "realize_complete":
            return {"scenes": [{"scene_id": scene["scene_id"], "text": "COMPLETE " + scene["scene_id"]}
                               for scene in payload["program"]["scenes"]]}
        if key == "read_complete":
            return {"strengths_to_preserve": [], "revision_opportunities": [], "uncertainties": []}
        if key == "revise_complete":
            return {"operations": [], "scene_updates": [], "applied_changes": [], "rejected_suggestions": []}
        raise AssertionError(key)

class CompleteFixtureWriter(FixtureWriter):
    """Separate actual creative output from subsequent fallible interpretation."""
    def __call__(self, key, phase, prompt, payload):
        if key == "post_read_program_edit":
            self.calls.append({"key": key, "phase": phase, "input": deepcopy(payload)})
            return post_read_fixture_output(payload)
        if key.startswith("interpret_encounter_"):
            self.calls.append({"key": key, "phase": phase, "input": deepcopy(payload)})
            before = payload["current_relationships"][0]["interpretation"]
            return {"relationship_updates": [{"relation_id": "pq", "before": before,
                    "after": before + " after " + payload["actual_encounter"]["encounter_id"],
                    "status": "uncertain", "reason": "A reading of the actual line, still fallible",
                    "uncertainty": "Several readings remain possible"}]}
        if key == "expression_organization_proposal":
            self.calls.append({"key": key, "phase": phase, "input": deepcopy(payload)})
            return {"scene_id": payload["program"]["scenes"][0]["scene_id"],
                    "expression_change": "Change the timing of the particular offer",
                    "intended_gain": "Make another interpersonal possibility available",
                    "possible_loss": "The different timing may lose a useful pause"}
        if key == "alternative_expression_scene":
            self.calls.append({"key": key, "phase": phase, "input": deepcopy(payload)})
            return {"scene_id": payload["scene_id"], "text": "ALTERNATIVE ACTUAL " + payload["scene_id"]}
        if key == "read_expression_alternatives":
            self.calls.append({"key": key, "phase": phase, "input": deepcopy(payload)})
            return {"treatments": [{"trial_id": row["trial_id"], "scene_id": row["scene_id"],
                    "experience": "A particular reading of this actual treatment",
                    "gains_and_losses": "An available reply and a changed pause",
                    "uncertainty": "Other readings remain possible"} for row in payload["actual_treatments"]],
                    "whole_implications": "Reconsider adjoining exchanges without duplicating the event"}
        if key.startswith("read_expression_"):
            self.calls.append({"key": key, "phase": phase, "input": deepcopy(payload)})
            return {"unit_dispositions": [{"unit_id": unit["unit_id"], "status": "uncertain",
                    "reason": "The actual scene admits several readings"} for unit in payload["expression_units"]]}
        if key in {"realize_complete", "revise_complete", "inspect_final_complete"}:
            self.calls.append({"key": key, "phase": phase, "input": deepcopy(payload)})
            if key == "inspect_final_complete":
                return {"decision": "use_revision", "issues": []}
            return {"script": "Complete actual revision" if key == "revise_complete" else "Complete actual draft"}
        result = super().__call__(key, phase, prompt, payload)
        if key.startswith("encounter_"):
            return {"scene_id": payload["encounter"]["scene_id"], "text": result["text"]}
        if key.startswith("convert_expression_"):
            return {"scene_id": result["scene_id"], "text": result["text"]}
        return result

class CompleteEntryFlowTests(unittest.TestCase):
    def setUp(self):
        self.available = {"source": {"work_id": "fixture", "source_text": "Prose",
                          "source_revision": "fixture", "brief": {"medium": "film"}},
                          "initial_plan": "Actual common plan", "initial_script": "Actual common draft"}
        self.prompts = {"trial": "Write only the named scene", "realize": "Write complete script",
                        "review": "Read whole script", "final": "Revise complete script"}

    def run_entry(self, entry, writer=None):
        writer = CompleteFixtureWriter() if writer is None else writer
        result = alg.execute_entry_complete(entry, self.available, writer, self.prompts)
        return result, writer.calls

    def test_all_entries_finish_with_actual_complete_revision_then_inspection(self):
        for entry in alg.PROVENANCE:
            with self.subTest(entry=entry):
                result, calls = self.run_entry(entry)
                self.assertEqual([c["key"] for c in calls[-4:]],
                                 ["realize_complete", "read_complete", "revise_complete", "inspect_final_complete"])
                self.assertEqual(calls[-3]["input"]["script"], "Complete actual draft")
                self.assertEqual(calls[-1]["input"]["revised_script"], "Complete actual revision")
                self.assertEqual(result["script"], "Complete actual revision")
                self.assertEqual(result["writer_workflow_version"], alg.COMPLETE_ENTRY_VERSION)
                self.assertFalse(result["quality_gain_established"])

    def test_relationship_state_is_derived_after_actual_scene_and_reaches_successor(self):
        result, calls = self.run_entry("relationships_first")
        scenes = [c for c in calls if c["key"].startswith("encounter_")]
        interpretations = [c for c in calls if c["key"].startswith("interpret_encounter_")]
        self.assertEqual(len(scenes), 4)
        self.assertEqual(interpretations[0]["input"]["actual_encounter"]["text"],
                         scenes[1]["input"]["previous_actual_encounters"][0]["text"])
        self.assertEqual(scenes[1]["input"]["current_relationships"][0]["interpretation"], "Earlier interpretation after e1")
        self.assertEqual(scenes[2]["input"]["current_relationships"][0]["interpretation"], "Earlier interpretation")
        history = scenes[1]["input"]["previous_relationship_interpretations"][0]
        self.assertIn("actual_text_sha256", history)
        self.assertEqual(history["interpretation_updates"][0]["status"], "uncertain")
        self.assertIn("actual line", history["interpretation_updates"][0]["reason"])
        self.assertEqual(result["explorations"][0]["relationship_history"][0], history)

    def test_structure_refreshes_interleaved_section_against_actual_prefix(self):
        class Interleaved(CompleteFixtureWriter):
            def __call__(self, key, phase, prompt, payload):
                value = super().__call__(key, phase, prompt, payload)
                if key == "hierarchical_arrangement":
                    value["program"]["scenes"].append({"scene_id": "s3", "development": "Late return"})
                    value["sections"] = [{"section_id": "x", "purpose": "A recurring strand", "scene_ids": ["s1", "s3"]},
                                         {"section_id": "y", "purpose": "Interleaved strand", "scene_ids": ["s2"]}]
                    value["presentation_constraints"] = [{"before": a, "after": b, "kind": "presentation", "reason": "Juxtaposition"}
                                                          for a, b in (("s1", "s2"), ("s2", "s3"))]
                if key.startswith("expand_"):
                    value["scene_tasks"][0]["development"] = "Task responds to: " + payload["actual_prefix"]
                return value
        result, calls = self.run_entry("structure_first", Interleaved())
        expansions = [c for c in calls if c["key"].startswith("expand_")]
        self.assertEqual(len(expansions), 3)
        self.assertEqual(expansions[-1]["input"]["section"]["scene_ids"], ["s3"])
        self.assertEqual(expansions[-1]["input"]["actual_prefix"], "ACTUAL STRUCTURE s1\n\nACTUAL STRUCTURE s2")
        final_scene = next(c for c in calls if c["key"] == "structure_scene_s3")
        self.assertIn("ACTUAL STRUCTURE s2", final_scene["input"]["scene_task"]["development"])
        self.assertEqual([s["scene_id"] for s in result["revision"]["program"]["scenes"]], ["s1", "s2", "s3"])

    def test_structure_presentation_order_also_reaches_complete_writer(self):
        result, calls = self.run_entry("structure_first")
        realized = next(c for c in calls if c["key"] == "realize_complete")
        self.assertEqual([s["scene_id"] for s in realized["input"]["program"]["scenes"]], ["s2", "s1"])
        self.assertEqual(result["explorations"][0]["presentation_order"], ["s2", "s1"])

    def test_expression_reads_actual_scene_and_passes_uncertain_observations_forward(self):
        result, calls = self.run_entry("expression_first")
        readings = [c for c in calls if c["key"].startswith("read_expression_")]
        scenes = [c for c in calls if c["key"].startswith("convert_expression_")]
        self.assertEqual(readings[0]["input"]["actual_scene"]["text"], scenes[1]["input"]["actual_prefix"])
        observation = scenes[1]["input"]["previous_expression_observations"][0]
        self.assertEqual(observation["unit_dispositions"][0]["status"], "uncertain")
        self.assertFalse(observation["realization_certified"])
        self.assertEqual(result["explorations"][0]["expression_observations"][0], observation)

    def test_final_integrity_advice_without_actual_new_copy_preserves_revision(self):
        class Advisory(CompleteFixtureWriter):
            def __call__(self, key, phase, prompt, payload):
                value = super().__call__(key, phase, prompt, payload)
                if key == "inspect_final_complete":
                    value.update(decision="keep_before", issues=[{"kind": "continuity_conflict", "passage": "",
                                 "repeats_intended": None, "explanation": "Uncertain interpretation"}])
                return value
        for entry in ("relationships_first", "structure_first", "expression_first"):
            result, _ = self.run_entry(entry, Advisory())
            self.assertEqual(result["script"], "Complete actual revision")
            self.assertTrue(result["integrity_selection"]["unverified_advice_did_not_reject_revision"])
