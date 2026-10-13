"""Assess anchored treatment grades through an explicitly supplied gateway adapter."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from statistics import mean

from evaluate import load, save, request, MODEL
import graded_treatment as grading


SYSTEM = "Apply the fixed anchored screenplay content criteria to the anonymous draft under its actual brief. Return JSON only."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--briefs", required=True, type=Path)
    parser.add_argument("--human-train-examples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gateway-command")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    inputs, briefs, examples = load(args.inputs), load(args.briefs), load(args.human_train_examples)
    examples = examples["examples"] if isinstance(examples, dict) else examples
    unique = {}
    for row in inputs["records"]:
        digest = hashlib.sha256(row["script"].encode()).hexdigest()
        if digest != row["script_sha256"] or row["work_id"] not in briefs:
            raise ValueError("Unverified frozen script or missing original brief")
        entry = (row["script"], briefs[row["work_id"]])
        if digest in unique and unique[digest] != entry:
            raise ValueError("Identical script assigned incompatible original briefs")
        unique[digest] = entry
    if not args.summarize_only:
        if not args.gateway_command:
            parser.error("Supply an authorized gateway adapter")
        def task(job):
            digest, dimension, reading = job
            path = args.output / "annotations" / (digest + "__" + dimension + "__" + str(reading) + ".json")
            script, brief = unique[digest]
            payload = grading.payload(script, dimension, brief, examples)
            fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if path.exists():
                if load(path)["input_sha256"] != fingerprint:
                    raise ValueError("Frozen evaluation input changed")
                return
            record = {"status": "unavailable", "input_sha256": fingerprint,
                      "dimension": dimension, "reading": reading}
            try:
                labels, receipt = request(args.gateway_command, {
                    "stage": "dramabench_label", "system": SYSTEM, "input": payload,
                    "model": MODEL, "source": "wiki", "reasoning_effort": "low",
                    "request_identity": digest + "__" + dimension + "__" + str(reading) + "__" + grading.VERSION})
                values = grading.metrics(dimension, labels, script)
                record.update(status="complete", labels=labels, metrics=values,
                              served_model=receipt["served_model"], source_requested=receipt["source_requested"])
            except Exception as exc:
                record.update(error_type=type(exc).__name__)
            save(path, record)
        jobs = [(digest, dim, reading) for digest in sorted(unique)
                for dim in grading.CRITERIA for reading in (0, 1)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(task, jobs))
    values = {}
    for digest in unique:
        values[digest] = {}
        for dim in grading.CRITERIA:
            paths = [args.output / "annotations" / (digest + "__" + dim + "__" + str(i) + ".json") for i in (0, 1)]
            if not all(p.exists() for p in paths):
                continue
            annotations = [load(p) for p in paths]
            if not all(a["status"] == "complete" for a in annotations):
                continue
            for key in annotations[0]["metrics"]:
                levels = [a["metrics"][key] for a in annotations]
                values[digest][key] = mean(levels) if all(v is not None for v in levels) else None
    by_work = {}
    for row in inputs["records"]:
        by_work.setdefault(row["work_id"], {})[row["method"]] = values[row["script_sha256"]]
    methods = sorted({r["method"] for r in inputs["records"]})
    report = {"version": grading.VERSION, "per_item": {}, "human_votes_created": 0,
              "grade_scale": [0, 3], "grades_are_ordinal": True,
              "mean_grade_is_descriptive": True, "official_DramaBench_score": False}
    for key in ("emotion_grade", "conflict_grade"):
        works = sorted(w for w, entries in by_work.items()
                       if all(m in entries and entries[m].get(key) is not None for m in methods))
        report["per_item"][key] = {"common_work_ids": works, "n_common": len(works),
            "methods": {m: {"mean": mean(by_work[w][m][key] for w in works) if works else None,
                            "n": len(works)} for m in methods}}
    save(args.output / "report.json", report)


if __name__ == "__main__":
    main()
