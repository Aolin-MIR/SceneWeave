"""Task-separated writer request templates shared by training and inference."""
import hashlib
import json
from pathlib import Path

INSTRUCTIONS = {
    "revise_text": "Revise the complete supplied draft for its actual writing request. Preserve useful strengths and make the necessary changes. Output only the complete revised text.",
    "adapt_scene": "Adapt the supplied original prose into one complete film screenplay scene. Organize its particular human relationships, psychological contradictions and lived experience through what is shown, withheld, juxtaposed, spoken and left unspoken. Necessary changes to plot, actions, relationships, ending, background, props, scene organization and expression are allowed. Output only the scene. No reference adaptation is supplied.",
    "continue_script": "Continue the supplied screenplay prefix in its existing language and dramatic form. Develop its particular characters and relationships in complete scenes. Output only the continuation. Do not restate the prefix.",
    "revise_with_feedback": "Revise the complete supplied draft in its actual conversation context using the supplied reader feedback. Assess the feedback against the request and preserve useful strengths. Output only the complete revised text.",
    "adaptation_process": "Complete the specified adaptation stage using only the supplied request and stage instruction. Return the requested JSON object. Source context may be incomplete; do not claim invented developments appeared in the original work.",
    "write_scene_from_synopsis": "Write one complete screenplay scene for the supplied scene synopsis, in the requested language. Express the particular relationships and circumstances through staging and dialogue. Output only the scene.",
}

INPUT_FIELDS = {"revise_text": {"instruction", "draft"},
                "adapt_scene": {"source_text", "medium"},
                "continue_script": {"prefix", "language"},
                "revise_with_feedback": {"conversation_json", "draft", "feedback_json"},
                "adaptation_process": {"stage", "request_json", "stage_instruction"}}

def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()

def normalized_sha(text):
    return sha(" ".join(text.split()))

def render_request(task, inputs):
    """Use this exact projection for training and eventual inference."""
    if task not in INPUT_FIELDS or set(inputs) != INPUT_FIELDS[task]:
        raise ValueError("Unknown task or unexpected inference-visible fields")
    if any(not isinstance(v, str) or not v.strip() for v in inputs.values()):
        raise ValueError("Missing task input")
    return [{"role": "system", "content": INSTRUCTIONS[task]},
            {"role": "user", "content": json.dumps(
                {"task": task, **inputs}, ensure_ascii=False, sort_keys=True)}]

def record(identifier, task, inputs, target, split, work, provenance, scope):
    if split not in {"train", "dev", "unassigned"}:
        raise ValueError("Writer preparation cannot import TEST records")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Missing actual writing target")
    return {"id": identifier, "task": task, "split": split, "group_id": work,
            "inputs": inputs, "messages": render_request(task, inputs) +
            [{"role": "assistant", "content": target}],
            "input_sha256": sha(json.dumps(inputs, ensure_ascii=False, sort_keys=True)),
            "target_sha256": sha(target), "normalized_target_sha256": normalized_sha(target),
            "target_characters": len(target), "scope": scope, "provenance": provenance,
            "loss_mask": "Assistant target only; mask system/user/padding",
            "training_authorized": False}

def load_lines(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
