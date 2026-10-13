"""Run task-conditioned pairwise judgments in both presentation orders."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path

from evaluate import MODEL, load, request, save
from pairwise import DIMENSIONS, RUBRIC, SYSTEM
from retrieve import retrieve_examples


def swap(choice):
    return {"A": "B", "B": "A", "tie": "tie"}[choice]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gateway-command", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--human-train", type=Path, help="Prepared original human TRAIN records for request retrieval")
    parser.add_argument("--retrieved-examples", type=int, default=8)
    args = parser.parse_args()
    inputs = load(args.inputs)
    cases = inputs["cases"]
    examples = inputs["human_train_demonstrations"]
    if args.human_train:
        data=load(args.human_train)
        training=data["train"] if isinstance(data,dict) else data
        save(args.output/"icl_assignments.json",retrieve_examples(cases,training,args.retrieved_examples))

    def task(job):
        case, order = job
        identifier = case["id"] + "__" + order
        payload = {"rubric": RUBRIC, "human_train_demonstrations": case.get("_examples",examples),
                   "writing_request": case["request"],
                   "A": case["a"] if order == "AB" else case["b"],
                   "B": case["b"] if order == "AB" else case["a"]}
        req = {"stage": "dramabench_pairwise", "system": SYSTEM, "input": payload,
               "model": MODEL, "source": "wiki", "reasoning_effort": "low",
               "request_identity": identifier}
        digest = hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = args.output / "judgments" / (identifier + ".json")
        if path.exists():
            cached = load(path)
            if cached["input_sha256"] != digest:
                raise ValueError("Cached judgment inputs changed; use a new output directory")
            return
        record = {"id": case["id"], "order": order, "input_sha256": digest, "status": "unavailable"}
        try:
            labels, response = request(args.gateway_command, req)
            dimensions = labels["dimensions"]
            if set(dimensions) != set(DIMENSIONS) or any(v not in {"A", "B", "tie"} for v in dimensions.values()):
                raise ValueError("Invalid dimension choices")
            probabilities=labels["preference_probabilities"]
            if (set(probabilities)!={"A","B","tie"}
                    or any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in probabilities.values())
                    or abs(sum(probabilities.values())-1)>0.001):
                raise ValueError("Invalid preference probabilities")
            ordered=sorted(probabilities,key=probabilities.get,reverse=True)
            choice=ordered[0] if probabilities[ordered[0]]-probabilities[ordered[1]]>1e-8 else None
            record.update(status="complete", labels=labels,
                          aligned_choice=choice if order == "AB" or choice is None else swap(choice),
                          aligned_dimensions={k:v if order == "AB" else swap(v) for k,v in dimensions.items()},
                          served_model=response["served_model"], source_requested=response["source_requested"])
        except Exception as exc:
            record.update(error_type=type(exc).__name__)
        save(path, record)

    jobs = [(case, order) for case in cases if case["a"]!=case["b"] for order in ("AB", "BA")]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(task, job) for job in jobs]):
            future.result()
    rows = []
    for case in cases:
        if case["a"]==case["b"]:
            rows.append({"id":case["id"],"available":True,"order_stable":True,
                         "choice":"tie","dimensions":{k:"tie" for k in DIMENSIONS},
                         "identity_implies_tie":True,"raw_order_votes_created":False})
            continue
        ab = load(args.output / "judgments" / (case["id"] + "__AB.json"))
        ba = load(args.output / "judgments" / (case["id"] + "__BA.json"))
        available = ab["status"] == ba["status"] == "complete"
        stable = available and ab["aligned_choice"] is not None and ab["aligned_choice"] == ba["aligned_choice"]
        choice=ab["aligned_choice"] if stable else None
        aggregate=None
        if available:
            ap,bp=ab['labels']['preference_probabilities'],ba['labels']['preference_probabilities']
            ap={k:v/sum(ap.values()) for k,v in ap.items()}
            bp={k:v/sum(bp.values()) for k,v in bp.items()}
            aggregate={"A":(ap['A']+bp['B'])/2,"B":(ap['B']+bp['A'])/2,"tie":(ap['tie']+bp['tie'])/2}
            ordered=sorted(aggregate,key=aggregate.get,reverse=True)
            if aggregate[ordered[0]]-aggregate[ordered[1]]>1e-8:
                choice=ordered[0]
                if choice=='tie' and not (stable and ab['aligned_choice']=='tie'):
                    choice=None
        rows.append({"id":case["id"], "available":available, "order_stable":stable,
                     "choice":choice,"aggregated_probabilities":aggregate,
                     "dimensions":{k:ab["aligned_dimensions"][k] if available and ab["aligned_dimensions"][k] == ba["aligned_dimensions"][k] else None for k in DIMENSIONS}})
    save(args.output / "report.json", {"rows":rows, "annotation_model":MODEL,
         "official_DramaBench_score":False, "scope":"Task-conditioned pairwise judgments of realized dramatic treatment",
         "human_votes_created":0, "raw_order_conflicts_are_ties":False})


if __name__ == "__main__":
    main()
