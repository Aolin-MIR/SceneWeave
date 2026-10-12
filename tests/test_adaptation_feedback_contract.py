"""Check feedback binding, abstention and score-pair symmetry."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import adaptation_feedback_contract as c
from adaptation_system_contract import EDIT_AXES


class FeedbackContractTests(unittest.TestCase):
    def setUp(self):
        self.source = {"work_id": "fixture", "source_text": "Source",
                       "source_revision": "v1", "brief": {"medium": "film"}}
        self.program = {"intent": "Particular people", "choices": {k: "retain" for k in EDIT_AXES},
                        "scenes": [{"scene_id": "s1", "development": "Development"}]}

    def test_no_change_proposal_is_allowed(self):
        inputs = c.critique_input(self.source, self.program, "Draft")
        critique = c.bind_critique(inputs, {"suggestions": []}, "fixture-critic")
        result = c.feedback_revision_input(self.source, self.program, "Draft", critique)
        self.assertEqual(result["modification_proposals"], [])

    def test_stale_draft_or_unknown_scene_is_rejected(self):
        inputs = c.critique_input(self.source, self.program, "Draft")
        critique = c.bind_critique(inputs, {"suggestions": []}, "fixture-critic")
        with self.assertRaises(ValueError):
            c.feedback_revision_input(self.source, self.program, "Another draft", critique)
        suggestion = {"scene_ids": ["absent"], "concern": "Concern", "proposed_change": "Change",
                      "expected_effect": "Effect", "preserve": [], "dependencies": []}
        with self.assertRaises(ValueError):
            c.bind_critique(inputs, {"suggestions": [suggestion]}, "fixture-critic")

    def test_score_swap_is_symmetric_without_position_prompts(self):
        a = c.preference_from_scores(3., -2., 1.5)
        b = c.preference_from_scores(-2., 3., 1.5)
        self.assertAlmostEqual(a + b, 1.)
        self.assertEqual(c.preference_from_scores(0., 100000., 1.), 0.)
        with self.assertRaises(ValueError):
            c.preference_from_scores(0., float("nan"), 1.)

    def test_revision_improvement_is_not_a_human_label(self):
        result = c.feedback_comparison_record("fixture", "Draft", "A", "B", "writer", "critic")
        self.assertIsNone(result["human_preference"])
        self.assertIsNone(result["budget_matched"])


if __name__ == "__main__":
    unittest.main()
