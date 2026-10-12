"""Connect critique proposals, writing requests and independently scored revisions.

No critique is certified merely by being specific or being followed. These
interfaces preserve source versions and expose the missing human validation.
"""
import math

from adaptation_system_contract import fingerprint, validate_source, validate_program


CRITIC_INSTRUCTION = """Read the actual source, adaptation task, current program and draft. Propose only important, plausible improvements to its particular human relationships, psychological contradictions, lived experience and dramatic expression. Describe the passage or scene concerned, the problem, a concrete possible change, what preceding/following material would need adjustment, and strengths to preserve. Ambiguity, silence, interior voice, comedy, lyrical or non-realist forms may be intentional and effective. Do not demand more conflict, explanation, reconciliation or a new climax by default. If no important change is warranted, return no suggestions. Your suggestions are hypotheses, not authoritative verdicts. Return JSON {suggestions:[{scene_ids,concern,proposed_change,expected_effect,preserve,dependencies}]}.
"""


def critique_input(source, program, draft):
    if not isinstance(draft, str) or not draft.strip():
        raise ValueError("Critique requires the actual complete draft")
    return {"source": validate_source(source), "program": validate_program(program), "draft": draft}


def bind_critique(inputs, output, model_revision):
    if not model_revision or not isinstance(output.get("suggestions"), list):
        raise ValueError("Missing critic identity or suggestions list")
    ids = {scene["scene_id"] for scene in inputs["program"]["scenes"]}
    fields = {"scene_ids", "concern", "proposed_change", "expected_effect", "preserve", "dependencies"}
    for suggestion in output["suggestions"]:
        if set(suggestion) != fields or not isinstance(suggestion["scene_ids"], list):
            raise ValueError("Incomplete modification proposal")
        if not set(suggestion["scene_ids"]).issubset(ids):
            raise ValueError("Proposal refers to an unknown scene")
        if any(not suggestion[key] for key in ("concern", "proposed_change", "expected_effect")):
            raise ValueError("Missing concrete proposal")
    return {"source_sha256": fingerprint(inputs["source"]),
            "program_sha256": fingerprint(inputs["program"]),
            "draft_sha256": fingerprint(inputs["draft"]),
            "critic_revision": model_revision, "suggestions": output["suggestions"],
            "status": "Unvalidated proposals; not human feedback or quality labels"}


def feedback_revision_input(source, program, draft, critique):
    actual = critique_input(source, program, draft)
    for key, value in (("source_sha256", actual["source"]),
                       ("program_sha256", actual["program"]), ("draft_sha256", draft)):
        if critique.get(key) != fingerprint(value):
            raise ValueError("Critique is bound to another source, task, program or draft")
    return {**actual, "modification_proposals": critique["suggestions"],
            "instruction": "Assess each proposal against the source and actual dramatic purpose. Apply useful changes, preserving worthwhile strengths. Revise related preparation, character treatment, subsequent development and scene organization as needed. You may reject proposals. Return the complete script and the resulting program; do not claim success from adherence alone."}


def preference_from_scores(left, right, temperature):
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
           for v in (left, right, temperature)) or temperature <= 0:
        raise ValueError("Finite scores and positive DEV calibration are required")
    z = (left - right) / temperature
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def feedback_comparison_record(work_id, draft, revision_with_feedback, revision_self_review,
                               writer_revision, critique_revision, *, split="dev"):
    if split != "dev" or not all((work_id, draft, revision_with_feedback,
                                  revision_self_review, writer_revision, critique_revision)):
        raise ValueError("Complete matched DEV material and model identities are required")
    return {"work_id": work_id, "split": split, "initial_draft": draft,
            "with_feedback": revision_with_feedback, "self_review": revision_self_review,
            "writer_revision": writer_revision, "critic_revision": critique_revision,
            "human_preference": None, "budget_matched": None,
            "use": "Critique benefit relative to strong independent self-review; no invented labels"}
