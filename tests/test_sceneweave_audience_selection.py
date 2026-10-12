"""Test conservative selection with deterministic fixtures, never human labels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from adaptation_system_contract import fingerprint
import sceneweave_audience_selection as audience


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "fixture_report.json"
        path.write_text(json.dumps({"status": "validated",
            "scope": "screenplay_draft_preferences", "human_judgments": 1,
            "heldout_groups_verified": True, "fixture_only": True,
            "evaluator_id": "fixture-model-sha"}))
        self.validation = {"report_path": str(path),
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.request = {"source_text": "Fixture source", "brief": {"medium": "film"}}
        self.result = {"status": "complete", "source_sha256": fingerprint(self.request),
            "before_whole_integration": "realized", "script": "revised"}

    def comparator(self, predict):
        return audience.OrderConsistentComparator(predict,
            writing_request=self.request, validation=self.validation,
            evaluator_id="fixture-model-sha")

    @staticmethod
    def response(choice):
        return {"choice": choice, "evaluator_id": "fixture-model-sha"}

    def test_unique_winner_and_anonymous_full_inputs(self):
        ranks = {"initial": 0, "realized": 1, "revised": 2}
        inputs = []
        def predict(row):
            inputs.append(deepcopy(row))
            return self.response("left" if ranks[row["a"]] > ranks[row["b"]] else "right")
        original = deepcopy(self.result)
        selected = audience.select_completed_candidates("initial", self.result,
                                                        self.comparator(predict))
        self.assertEqual(selected["selected"]["script"], "revised")
        self.assertEqual(len(inputs), 6)
        self.assertTrue(all(set(r) == {"writing_request", "a", "b"} for r in inputs))
        self.assertTrue(all(r["writing_request"] == self.request for r in inputs))
        self.assertEqual(self.result, original)
        self.assertFalse(selected["quality_gain_established"])
        self.assertFalse(selected["independent_outcome_evaluation"])

    def test_position_bias_is_unavailable_and_retains_initial(self):
        c = self.comparator(lambda row: self.response("left"))
        selected = audience.select_completed_candidates("initial", self.result, c)
        self.assertEqual(selected["selected"]["script"], "initial")
        self.assertEqual({r["choice"] for r in selected["search_history"][0]["outcomes"]},
                         {"unavailable"})

    def test_genuine_ties_remain_ties(self):
        c = self.comparator(lambda row: self.response("tie"))
        selected = audience.select_completed_candidates("initial", self.result, c)
        self.assertEqual(selected["selected"]["script"], "initial")
        self.assertEqual({r["choice"] for r in selected["search_history"][0]["outcomes"]}, {"tie"})

    def test_cycle_retains_initial(self):
        wins = {("initial", "realized"), ("realized", "revised"), ("revised", "initial")}
        c = self.comparator(lambda row: self.response("left" if (row["a"], row["b"]) in wins else "right"))
        selected = audience.select_completed_candidates("initial", self.result, c)
        self.assertEqual(selected["selected"]["script"], "initial")
        self.assertEqual(selected["search_history"][0]["decision"], "cyclic_preferences_keep_incumbent")

    def test_duplicate_drafts_are_not_independent_candidates(self):
        self.result["before_whole_integration"] = "initial"
        self.result["script"] = "initial"
        c = self.comparator(lambda row: self.fail("No duplicate comparison is allowed"))
        selected = audience.select_completed_candidates("initial", self.result, c)
        self.assertEqual(len(selected["candidate_script_sha256s"]), 1)
        self.assertEqual(c.records, [])

    def test_actual_model_mismatch_is_recorded(self):
        c = self.comparator(lambda row: {"choice": "left", "evaluator_id": "other-model"})
        selected = audience.select_completed_candidates("initial", self.result, c)
        self.assertEqual(selected["selected"]["script"], "initial")
        self.assertTrue(all(r["status"] == "failed" for r in c.records))

    def test_other_model_cannot_borrow_qualification(self):
        with self.assertRaises(ValueError):
            audience.OrderConsistentComparator(lambda row: self.fail("Must not predict"),
                writing_request=self.request, validation=self.validation,
                evaluator_id="another-model")

    def test_changed_qualification_fails_before_predict(self):
        c = self.comparator(lambda row: self.fail("Must not predict"))
        Path(c.validation["report_path"]).write_text("{}");
        with self.assertRaises(ValueError):
            audience.select_completed_candidates("initial", self.result, c)

    def test_source_mismatch_fails_before_predict(self):
        c = self.comparator(lambda row: self.fail("Must not predict"))
        self.result["source_sha256"] = "different"
        with self.assertRaises(ValueError):
            audience.select_completed_candidates("initial", self.result, c)

    def test_missing_draft_cannot_be_selected(self):
        c = self.comparator(lambda row: self.fail("Must not predict"))
        self.result.pop("before_whole_integration")
        with self.assertRaises(ValueError):
            audience.select_completed_candidates("initial", self.result, c)


if __name__ == "__main__":
    unittest.main()
