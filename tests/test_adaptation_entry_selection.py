"""Check selector mathematics and leakage guards with synthetic unit fixtures."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import adaptation_entry_selection as policy


class EntrySelectionTests(unittest.TestCase):
    def setUp(self):
        self.feature = policy.SourceFeatures(
            "synthetic-work-a", "train", (1., .2), "source-hash",
            "brief-hash", "frozen-encoder", True)
        self.row = policy.EntryComparison(
            self.feature.work_id, "train", "scene_first", "structure_first",
            .75, "human_votes", None, "synthetic-comparison-a", True, True, True)

    def observations(self, row=None):
        return policy.prepare_observations([self.feature], [row or self.row], "train")

    def test_gradient_matches_numerical_difference(self):
        observations = self.observations()
        parameters = np.arange(12, dtype=float).reshape(4, 3) / 20
        _, gradient = policy.loss_and_gradient(parameters, observations, .1)
        for position in np.ndindex(parameters.shape):
            plus, minus = parameters.copy(), parameters.copy()
            plus[position] += 1e-6
            minus[position] -= 1e-6
            numerical = (policy.loss_and_gradient(plus, observations, .1)[0]
                         - policy.loss_and_gradient(minus, observations, .1)[0]) / 2e-6
            self.assertAlmostEqual(gradient[position], numerical, places=7)

    def test_swapping_candidates_and_soft_preference_preserves_loss(self):
        swapped = replace(self.row, left=self.row.right, right=self.row.left,
                          preference_left=1 - self.row.preference_left)
        parameters = np.arange(12, dtype=float).reshape(4, 3) / 20
        a = policy.loss_and_gradient(parameters, self.observations())
        b = policy.loss_and_gradient(parameters, self.observations(swapped))
        self.assertAlmostEqual(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])

    def test_books_have_equal_weight_with_unequal_comparison_counts(self):
        second = replace(self.feature, work_id="synthetic-work-b")
        rows = [self.row, replace(self.row, observation_id="synthetic-comparison-a2"),
                replace(self.row, work_id=second.work_id, observation_id="synthetic-comparison-b")]
        observations = policy.prepare_observations([self.feature, second], rows, "train")
        np.testing.assert_allclose(observations["weights"], [.5, .5, 1.])
        self.assertEqual(observations["independent_works"], 2)

    def test_no_route_outcomes_cannot_be_replaced_by_reference_scripts(self):
        with self.assertRaises(ValueError):
            policy.prepare_observations([self.feature], [], "train")

    def test_test_labels_and_post_generation_features_are_rejected(self):
        for bad in [replace(self.feature, split="test"),
                    replace(self.feature, available_before_generation=False)]:
            with self.assertRaises(ValueError):
                policy.prepare_observations([bad], [self.row], "train")
        with self.assertRaises(ValueError):
            self.observations(replace(self.row, split="test"))

    def test_incomplete_judgments_are_not_ties(self):
        for bad in [replace(self.row, preference_left=float("nan")),
                    replace(self.row, comparison_complete=False),
                    replace(self.row, budget_matched=False),
                    replace(self.row, execution_valid=False)]:
            with self.assertRaises(ValueError):
                self.observations(bad)

    def test_same_work_cannot_cross_train_dev(self):
        with self.assertRaises(ValueError):
            policy.prepare_observations([self.feature, replace(self.feature, split="dev")],
                                        [self.row], "train")

    def test_training_is_blocked_without_applicable_evaluator_validation(self):
        with self.assertRaises(ValueError):
            policy.fit_selector(self.observations(), evaluation_status={}, regularization=.1)

    def test_four_workflows_share_execution_schedule(self):
        specs = [policy.workflow(entry) for entry in policy.ENTRIES]
        self.assertEqual(len({spec["stages"] for spec in specs}), 1)
        self.assertTrue(all(not spec["generation_enabled"] for spec in specs))


class SourceConditionedRouteTests(unittest.TestCase):
    def setUp(self):
        self.source = {"work_id": "synthetic-source", "source_revision": "fixture-only",
                       "source_text": "P changes how they offer Q help after an unanswered visit.",
                       "brief": {"medium": "audio drama", "focus": "Ambivalent care"}}
        self.payload = {"selected_entry": "relationships_first", "decision_basis": "Read the changed offers first",
                        "decision_uncertainty": "The unspoken thoughts could instead motivate expression development",
                        "assessments": [{"entry": e, "opportunity": "Particular feature " + e,
                                         "risk": "A treatment could flatten the ambivalence " + e,
                                         "starting_arrangement": "Develop the source under this medium " + e}
                                        for e in policy.ENTRIES],
                        "viable_alternatives": [{"entry": "expression_first", "reason": "The source contains inner reflection",
                                                 "when_to_use": "A brief centered on hearing that inner perspective"}]}

    def test_multiple_reasonable_routes_remain_visible_without_probabilities(self):
        route = policy.bind_route(self.payload, self.source)
        self.assertEqual(route["selected_entry"], "relationships_first")
        self.assertEqual(route["viable_alternatives"], self.payload["viable_alternatives"])
        self.assertEqual(route["decision_uncertainty"], self.payload["decision_uncertainty"])
        self.assertFalse(route["preference_validated"])
        self.assertEqual(route["route_outcome_labels"], 0)
        self.assertNotIn("reward", route)

    def test_alternative_selection_is_not_a_genre_mapping_or_fixed_class_quota(self):
        for entry in policy.ENTRIES:
            payload = deepcopy(self.payload)
            payload.update(selected_entry=entry, viable_alternatives=[])
            route = policy.bind_route(payload, self.source)
            self.assertEqual(route["selected_entry"], entry)
        changed_brief = {**self.source, "brief": {"medium": "film", "focus": "Practical mutual dependence"}}
        self.assertNotEqual(policy.route_input(self.source), policy.route_input(changed_brief))
        self.assertEqual(set(policy.route_input(self.source)), {"source", "available_entries"})

    def test_route_input_excludes_drafts_references_genre_labels_and_judgments(self):
        contaminated = {**self.source, "current_script": "A generated candidate", "reference_script": "A real adaptation",
                        "genre": "A genre label", "quality_judgment": "A past assessment", "human_route_vote": .9}
        payload = policy.route_input(contaminated)
        self.assertEqual(payload, policy.route_input(self.source))
        self.assertEqual(set(payload["source"]), {"work_id", "source_revision", "source_text", "brief"})

    def test_assessments_are_choice_archive_without_writer_guidance_interface(self):
        route = policy.bind_route(self.payload, self.source)
        self.assertEqual(route["assessments"], self.payload["assessments"])
        changed = {**self.source, "brief": {"medium": "film"}}
        self.assertNotEqual(route["brief_sha256"], policy.bind_route(self.payload, changed)["brief_sha256"])
        self.assertFalse(hasattr(policy, "development_guidance"))
        self.assertIn("not supplied to the writing stages", policy.ROUTE_INSTRUCTION)
        self.assertNotIn("will be passed to the first development stage", policy.ROUTE_INSTRUCTION)

    def test_malformed_or_duplicate_alternatives_are_not_silently_dropped(self):
        for alternatives in (None, [{"entry": "unknown"}],
                             [{"entry": "relationships_first", "reason": "Reason", "when_to_use": "Condition"}],
                             self.payload["viable_alternatives"] * 2,
                             [{"entry": "structure_first", "reason": "Missing condition"}]):
            payload = {**self.payload, "viable_alternatives": alternatives}
            with self.assertRaises(ValueError):
                policy.bind_route(payload, self.source)
        for field in ("decision_uncertainty", "decision_basis"):
            with self.assertRaises(ValueError):
                policy.bind_route({**self.payload, field: ""}, self.source)


if __name__ == "__main__":
    unittest.main()
