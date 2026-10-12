"""Select completed adaptations using a qualified, order-audited comparator.

This module neither loads models nor generates labels. The caller supplies the
frozen comparator and its genuine qualification report. Selection preferences
are kept separate from independent outcome evaluation.
"""
from copy import deepcopy
import json
from pathlib import Path

import adaptation_entry_algorithms as algorithms
from adaptation_system_contract import fingerprint
import run_adaptation_entry as entry_runner
import run_scene_first_matched as matched


def validate_common_initialization(value, source, backend):
    """Bind the common text to its two actual completed inference receipts."""
    initial = entry_runner.validate_shared_initialization(value, source, backend)
    receipts = value.get("receipts")
    if (value.get("status") != "complete" or not isinstance(receipts, list)
            or len(receipts) != 2 or any(not isinstance(r, dict) for r in receipts)):
        raise ValueError("The common draft requires its actual plan and write receipts")
    for field, response in zip(("initial_plan", "initial_script"), receipts):
        if response.get("status") != "complete" or response.get("finish_reason") != "stop":
            raise ValueError("Incomplete common initialization cannot be reused")
        if backend["transport"] == "api":
            authorized = (response.get("served_model") == matched.MODEL
                          and response.get("source_requested") == "wiki")
        else:
            authorized = (response.get("served_revision") == backend["revision"]
                          and response.get("source_requested") == "local")
        if not authorized:
            raise ValueError("The actual common initialization used a different backend")
        output = response.get("result")
        if not isinstance(output, dict) or output.get("answer") != initial[field]:
            raise ValueError("Common initialization text differs from its actual generated result")
    return initial


class OrderConsistentComparator:
    """Require complementary choices for both complete candidate orders."""

    def __init__(self, predict, *, writing_request, validation, evaluator_id):
        self.validation = matched.validate_evaluator(validation)
        if not callable(predict) or not evaluator_id or not writing_request:
            raise ValueError("A frozen predictor, identity and writing request are required")
        report_body = Path(self.validation["report_path"]).read_bytes()
        if matched.sha256_bytes(report_body) != self.validation["report_sha256"]:
            raise ValueError("Qualification report changed while binding the model")
        report = json.loads(report_body)
        if report.get("evaluator_id") != evaluator_id:
            raise ValueError("Qualification must bind the actual frozen evaluator identity")
        self.predict = predict
        self.writing_request = deepcopy(writing_request)
        self.evaluator_id = evaluator_id
        self.records = []

    def __call__(self, left, right):
        outcomes = []
        for a, b in ((left, right), (right, left)):
            payload = {"writing_request": deepcopy(self.writing_request),
                       "a": a["script"], "b": b["script"]}
            record = {"input": payload, "status": "failed"}
            try:
                response = self.predict(deepcopy(payload))
                record["response"] = deepcopy(response)
                if response.get("evaluator_id") != self.evaluator_id:
                    raise ValueError("Actual evaluator differs from the frozen selection model")
                choice = response.get("choice")
                if choice not in {"left", "right", "tie", "unavailable"}:
                    raise ValueError("Unsupported preference response")
                record.update(status="complete", choice=choice)
            except Exception as exc:
                record.update(error_type=type(exc).__name__,
                              error=str(exc)[:220], choice="unavailable")
            self.records.append(record)
            outcomes.append(record["choice"])
        pair = tuple(outcomes)
        if pair == ("left", "right"):
            return "left"
        if pair == ("right", "left"):
            return "right"
        if pair == ("tie", "tie"):
            return "tie"
        return "unavailable"


def select_completed_candidates(initial_script, entry_result, comparator):
    """Choose among the initial, realized and revised complete scripts."""
    if not isinstance(comparator, OrderConsistentComparator):
        raise ValueError("The explicit order-consistent qualified comparator is required")
    if entry_result.get("status") != "complete":
        raise ValueError("Incomplete adaptation cannot enter preference selection")
    if not isinstance(initial_script, str) or not initial_script.strip():
        raise ValueError("The actual complete initial draft is required")
    source_sha = entry_result.get("source_sha256")
    if not source_sha:
        raise ValueError("Completed candidates must be bound to their source and brief")
    if fingerprint(comparator.writing_request) != source_sha:
        raise ValueError("Candidate source differs from the frozen comparison request")
    initial = {"script": initial_script, "source_sha256": source_sha,
               "status": "complete", "origin": "common_initial_draft"}
    children = []
    for field in ("before_whole_integration", "script"):
        script = entry_result.get(field)
        if not isinstance(script, str) or not script.strip():
            raise ValueError("Each selection alternative must be an actual complete draft")
        children.append({"script": script, "source_sha256": source_sha,
                         "status": "complete", "origin": field})
    first_record = len(comparator.records)
    selected = algorithms.scene_candidate_search(
        initial, lambda parent, step: deepcopy(children), comparator,
        validation=comparator.validation, rounds=1, frontier_size=2)
    selected.update(evaluator_id=comparator.evaluator_id,
                    comparator_records=deepcopy(comparator.records[first_record:]),
                    candidate_script_sha256s=list(dict.fromkeys(
                        fingerprint(x["script"]) for x in [initial, *children])),
                    independent_outcome_evaluation=False)
    return selected


def execute_sceneweave_case(unit, generate, *, evaluator_validation, comparator,
                           **entry_settings):
    """Run the current four-entry writer, then select its completed drafts."""
    valid_unit = matched.validate_unit(unit)
    binding = matched.validate_evaluator(evaluator_validation)
    if not isinstance(comparator, OrderConsistentComparator):
        raise ValueError("A qualified comparator must be supplied before generation")
    if comparator.validation != binding:
        raise ValueError("Generation and selection qualification reports differ")
    if comparator.writing_request != valid_unit["source"]:
        raise ValueError("Selection must see the same source and adaptation brief")
    if entry_settings.get("execution_mode", "native_complete_v4") != "native_complete_v4":
        raise ValueError("SceneWeave selection requires the current complete-entry workflow")
    if entry_settings.get("shared_initialization") is not None:
        backend = entry_settings.get("backend") or matched.BACKEND
        validate_common_initialization(entry_settings["shared_initialization"],
                                       valid_unit["source"], backend)
    entry_settings["execution_mode"] = "native_complete_v4"
    result = entry_runner.execute_case(valid_unit, generate,
                                      evaluator_validation=evaluator_validation,
                                      **entry_settings)
    if result.get("status") != "complete":
        return result
    reused = result.get("shared_initialization")
    initial_script = reused.get("initial_script") if reused else None
    if initial_script is None:
        writes = [r for r in result["calls"] if r.get("status") == "complete"
                  and r.get("input", {}).get("official_stage") == "write"]
        if len(writes) != 1:
            raise ValueError("Cannot recover the actual common initial draft")
        initial_script = writes[0]["output"]["answer"]
    selection = select_completed_candidates(initial_script, result, comparator)
    result["before_audience_selection"] = result["script"]
    result["script"] = selection["selected"]["script"]
    result["audience_selection"] = selection
    result["quality_gain_established"] = False
    return result
