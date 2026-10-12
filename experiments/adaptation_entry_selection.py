"""Define adaptation entry workflows and source-only pairwise policy inputs.

No API client or experiment launcher is imported. These utilities implement the
offline contract; no route observations or human preferences are synthesized.
"""
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from adaptation_system_contract import fingerprint, validate_source


ENTRIES = ("scene_first", "relationships_first", "structure_first", "expression_first")
ENTRY_INSTRUCTIONS = {
    "scene_first": (
        "After reading the source and shared interpretation, identify a scene "
        "worth realizing. Keep any initial whole-work arrangement provisional. "
        "Develop two concrete realizations of that scene; use their actual "
        "strengths and limitations to reconsider character treatment, preceding "
        "preparation, subsequent development and the whole scene organization."
    ),
    "relationships_first": (
        "After reading the source and shared interpretation, develop particular "
        "people and the changing texture of their relationships across the work. "
        "Explore two concrete encounter sequences that express those changes. "
        "Select, merge, invent or revise scenes around the encounters, retaining "
        "psychological contradictions rather than reducing people to fixed goals."
    ),
    "structure_first": (
        "After reading the source and shared interpretation, develop the complete "
        "adapted story: its emphasis, beginning, development, ending and selected "
        "storylines. Explore two concrete whole-work arrangements. Reconsider "
        "cuts, expansions and order through their effects on people, relationships "
        "and lived experience, then realize the chosen arrangement as scenes."
    ),
    "expression_first": (
        "After reading the source and shared interpretation, identify what its "
        "narration, inner life, rhythm or imagery contributes. Explore two concrete "
        "ways to recreate that experience in the requested dramatic medium, using "
        "encounters, words, silence, sound, space, props or temporal arrangement "
        "as appropriate. Build scenes and the whole around effective expression. "
        "Monologue, narration and interior voice remain legitimate choices."
    ),
}
COMMON_CONTRACT = (
    "Serve the shared adaptation interpretation and the author's explicit "
    "requirements. The entry establishes a starting emphasis, not an exclusive "
    "set of later decisions. All entries can subsequently reconsider all eight "
    "edit axes. Source chapter and narration order are context, not a required "
    "screenplay scene order; consider selection, merging, reordering and "
    "earlier/later resonance where useful, without requiring them. "
    "Plot, character actions, relationships, ending, background, "
    "props, cast and expressive form may change where permitted. Changing every "
    "axis is unnecessary. Evaluate how particular relationships, psychological "
    "contradictions and lived experience acquire layered dramatic expression. "
    "Conflict count, twists, speed, ambiguity and fidelity are not universal "
    "quality objectives. Explain an editing decision's intended gain and "
    "possible loss in the supplied reason or decision fields; protect "
    "particular relationships, psychological layers and lived meaning through "
    "the resulting treatment, without a bonus for superficial retention. "
    "Revise initial decisions when realization warrants it. "
    "The final writer receives the concrete program and material, without the "
    "entry label or claims of superiority."
)

ROUTE_VERSION = "source-brief-entry-v4"
ROUTE_INSTRUCTION = """Read the actual source and the adaptation brief before choosing a starting process. Choose exactly one of the four available entries on the basis of a concrete opportunity and risk in this source under this brief. Presentation order carries no meaning; do not choose an entry because it appears first or earlier. This is a provisional writing decision, not a quality prediction validated against human outcomes. Do not choose from genre labels, length, conflict counts or a mandatory character-growth formula. Every entry retains the same authorized freedom to change source content. Do not request extra source material or look at generated candidates, reference scripts or previous quality judgments. Language is unrestricted. Return only JSON {selected_entry:scene_first|relationships_first|structure_first|expression_first,decision_basis:string,decision_uncertainty:string,assessments:[{entry:string,opportunity:string,risk:string,starting_arrangement:string}],viable_alternatives:[{entry:string,reason:string,when_to_use:string}]}. Include exactly one substantive assessment for each entry, addressing particular source circumstances and the requested dramatic medium. Explain why the selected opportunity deserves to be developed first and what could make that decision less suitable. A source can suit several entries: retain other reasonable starting processes in viable_alternatives, stating the particular reason and brief or development condition under which each could be useful. Do not force a single uniquely correct route, a numerical utility or equal numbers of works in each route. An empty alternatives list is allowed when justified by the source and brief. decision_uncertainty records the limitations of this process choice, not a probability of producing a better screenplay. Assessments document this source-only choice and are not supplied to the writing stages. Each entry develops its material independently from the source, brief and common initial plan and draft."""
ROUTE_INSTRUCTION += (
    " Identify the concrete adaptation decision that should be explored before "
    "the rest can be developed productively. The mere presence of narration or "
    "inner thought in prose does not privilege expression_first; compare whether "
    "the unresolved work concerns an actual scene, the changing relationship, "
    "the whole arrangement or a particular medium-specific effect. Likewise, "
    "the existence of a scene or several events alone does not settle the choice. "
    "Compare what concrete and distinct materials each process can develop "
    "from the available source scope under this brief. A bounded passage may "
    "support different arrangements of a short adaptation while supplying "
    "little basis for statements about an entire novel. Two entries may be "
    "almost equally reasonable: explain the practical starting choice and keep "
    "the other available, without claiming a measured advantage. "
    "Keep the attribution of source details precise: distinguish a narrator's "
    "own assertion, somebody else's quoted opinion, conjecture, imagined action "
    "and an event that actually occurs. Track who acts toward whom before "
    "describing care, injury or obligation. Preserve uncertainty in interpretation. "
    "Freely proposed additions and changes allowed by the brief are legitimate; "
    "identify them as proposed adaptation choices, rather than reporting them "
    "as supplied facts or treating their absence from the source as an automatic "
    "defect. These checks guide a fallible reading, not a fidelity reward."
)

ENTRY_MATERIAL_KINDS = {
    "scene_first": "actual_screenplay",
    "relationships_first": "encounter_sequence",
    "structure_first": "whole_work_arrangement",
    "expression_first": "expression_sample",
}

ENTRY_STAGE_NAMES = {
    "scene_first": ("provisional_scene_program", "actual_scene_trial"),
    "relationships_first": ("character_relationship_reading", "encounter_sequence"),
    "structure_first": ("official_outline_development", "whole_work_arrangement"),
    "expression_first": ("medium_expression_development", "expression_sample"),
}


def route_input(source: Mapping) -> dict:
    """Expose only the original source and brief before any writing occurs."""
    source = validate_source(dict(source))
    order = sorted(ENTRIES, key=lambda entry: fingerprint({"source": source, "entry": entry}))
    return {"source": source,
            "available_entries": [{"entry": entry, "process": ENTRY_INSTRUCTIONS[entry]}
                                  for entry in order]}


def bind_route(payload: Mapping, source: Mapping) -> dict:
    """Check the selection contract without certifying its artistic suitability."""
    source = validate_source(dict(source))
    if payload.get("selected_entry") not in ENTRIES:
        raise ValueError("Unknown selected adaptation entry")
    if not isinstance(payload.get("decision_basis"), str) or not payload["decision_basis"].strip():
        raise ValueError("A concrete source-conditioned selection reason is required")
    if (not isinstance(payload.get("decision_uncertainty"), str)
            or not payload["decision_uncertainty"].strip()):
        raise ValueError("The limitations of the provisional choice must remain explicit")
    rows = payload.get("assessments")
    if (not isinstance(rows, list) or len(rows) != len(ENTRIES)
            or any(not isinstance(row, dict) for row in rows)
            or {row.get("entry") for row in rows} != set(ENTRIES)):
        raise ValueError("Exactly one assessment for every available entry is required")
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ("opportunity", "risk", "starting_arrangement")):
            raise ValueError("Each entry requires a concrete opportunity, risk and starting arrangement")
    alternatives = payload.get("viable_alternatives")
    if not isinstance(alternatives, list):
        raise ValueError("Viable alternatives must be explicit, including when none is proposed")
    alternative_entries = []
    for row in alternatives:
        if not isinstance(row, dict) or row.get("entry") not in ENTRIES:
            raise ValueError("An alternative must refer to an available adaptation entry")
        if row["entry"] == payload["selected_entry"]:
            raise ValueError("A viable alternative cannot repeat the selected entry")
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ("reason", "when_to_use")):
            raise ValueError("An alternative requires its reason and condition of use")
        alternative_entries.append(row["entry"])
    if len(set(alternative_entries)) != len(alternative_entries):
        raise ValueError("Viable alternatives must not repeat an entry")
    return {"version": ROUTE_VERSION, "selected_entry": payload["selected_entry"],
            "decision_basis": payload["decision_basis"],
            "decision_uncertainty": payload["decision_uncertainty"],
            "assessments": [dict(row) for row in rows],
            "viable_alternatives": [dict(row) for row in alternatives],
            "source_sha256": fingerprint(source),
            "brief_sha256": fingerprint(source["brief"]),
            "displayed_entry_order": [row["entry"] for row in route_input(source)["available_entries"]],
            "selection_origin": "model_provisional_process_choice",
            "route_outcome_labels": 0, "preference_validated": False}


def workflow(entry: str) -> dict:
    """Return a fixed workflow contract without executing generation."""
    if entry not in ENTRIES:
        raise ValueError("Unknown adaptation entry")
    return {
        "entry": entry,
        "instruction": ENTRY_INSTRUCTIONS[entry] + "\n" + COMMON_CONTRACT,
        "stages": ("initial_development", "exploration_a", "exploration_b",
                   "whole_program_revision", "complete_script"),
        "execution_stages": ENTRY_STAGE_NAMES[entry],
        "material_kind": ENTRY_MATERIAL_KINDS[entry],
        "common_source_reading": True,
        "executor_shared": True,
        "equal_call_count_is_not_equal_compute": True,
        "generation_enabled": False,
    }


@dataclass(frozen=True)
class SourceFeatures:
    work_id: str
    split: str
    values: tuple[float, ...]
    source_sha256: str
    brief_sha256: str
    encoder_revision: str
    available_before_generation: bool

    def validate(self):
        if self.split not in ("train", "dev"):
            raise ValueError("Policy development cannot consume TEST inputs")
        if not self.available_before_generation:
            raise ValueError("Post-generation information cannot enter the selector")
        if not self.work_id or not self.values or not all(map(math.isfinite, self.values)):
            raise ValueError("Invalid source feature record")
        if not all((self.source_sha256, self.brief_sha256, self.encoder_revision)):
            raise ValueError("Missing feature provenance")


@dataclass(frozen=True)
class EntryComparison:
    work_id: str
    split: str
    left: str
    right: str
    preference_left: float
    label_origin: str
    evaluator_revision: str | None
    observation_id: str
    comparison_complete: bool
    execution_valid: bool
    budget_matched: bool

    def validate(self):
        if self.split not in ("train", "dev"):
            raise ValueError("Policy development cannot consume TEST labels")
        if self.left not in ENTRIES or self.right not in ENTRIES or self.left == self.right:
            raise ValueError("Invalid compared entries")
        if not math.isfinite(self.preference_left) or not 0 <= self.preference_left <= 1:
            raise ValueError("Missing comparisons cannot be converted to ties")
        if self.label_origin not in ("human_votes", "validated_model_judgment"):
            raise ValueError("Unsupported preference provenance")
        if self.label_origin == "validated_model_judgment" and not self.evaluator_revision:
            raise ValueError("Model labels require a frozen validated evaluator")
        if not all((self.observation_id, self.comparison_complete,
                    self.execution_valid, self.budget_matched)):
            raise ValueError("Incomplete or incomparable outcome")


def prepare_observations(features: Sequence[SourceFeatures],
                         comparisons: Sequence[EntryComparison], split: str):
    """Prepare book-weighted observations with strict cross-split protection."""
    if split not in ("train", "dev"):
        raise ValueError("Only TRAIN/DEV preparation is permitted")
    indexed = {}
    for feature in features:
        feature.validate()
        if feature.work_id in indexed:
            raise ValueError("Repeated work or cross-split work features")
        indexed[feature.work_id] = feature
    selected, seen = [], set()
    dimensions = {len(feature.values) for feature in features}
    revisions = {feature.encoder_revision for feature in features}
    if len(dimensions) != 1 or len(revisions) != 1:
        raise ValueError("A common frozen feature encoder is required")
    for row in comparisons:
        row.validate()
        if row.observation_id in seen:
            raise ValueError("Duplicate comparison observation")
        seen.add(row.observation_id)
        feature = indexed.get(row.work_id)
        if feature is None or feature.split != row.split:
            raise ValueError("Comparison/feature group split mismatch")
        if row.split == split:
            selected.append((feature, row))
    if not selected:
        raise ValueError("No observed entry outcomes; reference scripts are not route labels")
    counts = {}
    for feature, _ in selected:
        counts[feature.work_id] = counts.get(feature.work_id, 0) + 1
    return {
        "features": np.array([f.values for f, _ in selected], dtype=float),
        "left": np.array([ENTRIES.index(r.left) for _, r in selected]),
        "right": np.array([ENTRIES.index(r.right) for _, r in selected]),
        "preferences": np.array([r.preference_left for _, r in selected]),
        "weights": np.array([1 / counts[f.work_id] for f, _ in selected]),
        "independent_works": len(counts),
        "human_observations": sum(r.label_origin == "human_votes" for _, r in selected),
        "model_observations": sum(r.label_origin == "validated_model_judgment" for _, r in selected),
    }


def loss_and_gradient(parameters: np.ndarray, observations: Mapping,
                      regularization: float = 0.0):
    """Compute soft pairwise logistic loss for a small frozen-feature selector."""
    x = np.column_stack((observations["features"], np.ones(len(observations["features"]))))
    if parameters.shape != (len(ENTRIES), x.shape[1]):
        raise ValueError("Selector dimensions do not match the frozen features")
    if regularization < 0 or not np.isfinite(parameters).all():
        raise ValueError("Invalid selector parameters")
    left, right = observations["left"], observations["right"]
    p, weights = observations["preferences"], observations["weights"]
    weights = weights / weights.sum()
    differences = np.einsum("ij,ij->i", x, parameters[left] - parameters[right])
    loss = np.sum(weights * (np.logaddexp(0, differences) - p * differences))
    probability = np.exp(-np.logaddexp(0, -differences))
    updates = (weights * (probability - p))[:, None] * x
    gradient = np.zeros_like(parameters)
    np.add.at(gradient, left, updates)
    np.add.at(gradient, right, -updates)
    penalty = parameters.copy()
    penalty[:, -1] = 0
    return float(loss + .5 * regularization * np.sum(penalty ** 2)), gradient + regularization * penalty


def fit_selector(observations: Mapping, *, evaluation_status: Mapping,
                 regularization: float, max_iterations: int = 1000):
    """Fit a convex selector only after target-domain evaluation approval.

    This is an existing statistical tool, not an algorithmic novelty claim.
    Iteration exhaustion is reported separately from numerical convergence.
    """
    if (evaluation_status.get("status") != "validated"
            or evaluation_status.get("scope") != "adaptation_workflow_preferences"
            or not evaluation_status.get("report_sha256")):
        raise ValueError("Adaptation workflow evaluator has not passed its applicable validation")
    if regularization <= 0 or max_iterations <= 0:
        raise ValueError("A positive regularization and solver budget are required")
    from scipy.optimize import minimize
    shape = (len(ENTRIES), observations["features"].shape[1] + 1)

    def objective(flat):
        value, gradient = loss_and_gradient(flat.reshape(shape), observations, regularization)
        return value, gradient.ravel()

    result = minimize(objective, np.zeros(np.prod(shape)), jac=True,
                      method="L-BFGS-B", options={"maxiter": max_iterations})
    parameters = result.x.reshape(shape)
    parameters -= parameters.mean(axis=0, keepdims=True)
    value, gradient = loss_and_gradient(parameters, observations, regularization)
    return {
        "parameters": parameters,
        "loss": value,
        "numerically_converged": bool(result.success),
        "stop_reason": str(result.message),
        "gradient_inf_norm": float(np.abs(gradient).max()),
        "iterations": int(result.nit),
        "independent_train_works": observations["independent_works"],
        "evaluation_status": dict(evaluation_status),
        "generalization_gain_established": False,
    }


def select_entry(values: Sequence[float], parameters: np.ndarray) -> str:
    """Choose one workflow before generation using a frozen learned selector."""
    x = np.array([*values, 1.0], dtype=float)
    if parameters.shape != (len(ENTRIES), len(x)) or not np.isfinite(x).all() or not np.isfinite(parameters).all():
        raise ValueError("Invalid inference features or selector parameters")
    return ENTRIES[int(np.argmax(parameters @ x))]
