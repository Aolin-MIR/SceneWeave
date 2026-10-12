"""Exercise actual interface mismatch and invalid supervision cases."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import adaptation_system_contract as c


class SystemContractTests(unittest.TestCase):
    def setUp(self):
        self.source = {"work_id": "fixture", "source_text": "Source passage",
                       "source_revision": "fixture-v1", "brief": {"medium": "film"},
                       "human_comment": "Must not leak", "quality_score": 9}
        self.program = {"intent": "A particular relationship",
                        "choices": {k: "retain" for k in c.EDIT_AXES},
                        "scenes": [{"scene_id": "s1", "development": "Scene development"}]}
        self.trial_input = c.writer_input("trial", self.source, program=self.program,
                                         exploration={"scene_id": "s1", "approach": "A"})
        self.trials = [c.bind_trial({"text": "Scene " + x}, x, self.trial_input)
                       for x in ("a", "b")]

    def test_full_stage_connection_and_metadata_projection(self):
        revision_input = c.writer_input("revise_program", self.source,
                                        program=self.program, trials=self.trials)
        self.assertNotIn("human_comment", revision_input["source"])
        revised = copy.deepcopy(self.program)
        revised["choices"]["ending"] = "change"
        output = {"program": revised, "selected_trial_id": "a", "dependency_changes": []}
        final_input = c.realization_input(self.source, output, self.program, self.trials)
        self.assertEqual(final_input["program"]["choices"]["ending"], "change")
        self.assertEqual(final_input["selected_trial_material"]["text"], "Scene a")
        self.assertEqual(c.reward_input("Final script"), {"text": "Final script"})

    def test_stale_trial_or_different_brief_is_rejected(self):
        changed = copy.deepcopy(self.program)
        changed["choices"]["props"] = "change"
        with self.assertRaises(ValueError):
            c.writer_input("revise_program", self.source, program=changed, trials=self.trials)
        source = copy.deepcopy(self.source)
        source["brief"]["medium"] = "stage"
        with self.assertRaises(ValueError):
            c.writer_input("revise_program", source, program=self.program, trials=self.trials)

    def test_unknown_selection_and_missing_revision_request_are_rejected(self):
        with self.assertRaises(ValueError):
            c.realization_input(self.source, {"program": self.program,
                                "selected_trial_id": "c", "dependency_changes": []},
                                self.program, self.trials)
        with self.assertRaises(ValueError):
            c.writer_input("revise_script", self.source, program=self.program, draft="Draft")

    def test_real_response_is_not_revision_or_preference(self):
        row = {"split": "train", "training_ready": True, "family_split_verified": True,
               "version_verified": True, "label_origin": "human_response",
               "input": {"ordered_scenes": [{"text": "Scene"}]},
               "target": "This ending feels devastating"}
        self.assertEqual(c.admission_errors(row, "scene_response_auxiliary"), [])
        self.assertTrue(c.admission_errors(row, "writer_revision_sft"))
        self.assertTrue(c.admission_errors(row, "human_preference_reward"))

    def test_unassigned_records_and_fabricated_preferences_are_rejected(self):
        row = {"split": "unassigned", "target": "Comment", "input": {"ordered_scenes": [1]}}
        self.assertTrue(c.admission_errors(row, "scene_response_auxiliary"))
        row.update(split="train", training_ready=True, family_split_verified=True,
                   version_verified=True, left_text="A", right_text="B",
                   preference_left=.9, label_origin="machine_sentiment")
        self.assertTrue(c.admission_errors(row, "human_preference_reward"))
        row.update(label_origin="human_preference", preference_left=float("nan"))
        self.assertTrue(c.admission_errors(row, "human_preference_reward"))
        row["preference_left"] = .5
        self.assertEqual(c.admission_errors(row, "human_preference_reward"), [])


if __name__ == "__main__":
    unittest.main()
