"""Run the released SuperWriter checkpoint with its native three-stage interface.

Import and preparation are CPU-only. Real inference requires an explicit GPU
admission plus an applicable evaluator report or a single-DEV capability grant.
The checkpoint, tokenizer, raw stage format and workflow are separate from the
Wiki prompt adaptation and the Qwen3.5 writer's JSON training interface.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo

from official_superwriter_adaptation import official_functions, REVISION as CODE_REVISION
from run_scene_first_matched import validate_unit, validate_evaluator

ROOT = Path(__file__).resolve().parents[1]
REPO = "mozhu/SuperWriter-Qwen2.5-7B"
REVISION = "35c24873e4af1e5a739725036ae6ecb27a886176"
MODEL_DIR = ROOT / "models/superwriter_qwen2_5_7b_official_35c2487"
VERSION = "official-superwriter-native-local-v1"
STAGES = ("plan", "write", "refine")
STOP_STRINGS = ("<|user|>", "<|endoftext|>", "#*# finish.")
EOS_IDS = (151645, 151643)
CONTEXT_LIMIT = 32768
CHAT_TEMPLATE_SHA256 = "92fb895494534bfa5194ada09be922db1ecd36b87507171d5b289f0f3040cc4c"
EXPECTED_FILES = {
    "config.json": "8e192e535d692c237487e57427fc71b50da722107eadb025deff84515fe979c4",
    "generation_config.json": "cdadfe09574e88314d6e912a94c8ac07deba412f365006fe1652a249f346e489",
    "tokenizer_config.json": "36647b81c02137a5f7e13d37156297d5596f587f7cf8139e1baa39d7e5a33783",
    "tokenizer.json": "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf",
    "added_tokens.json": "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "model.safetensors.index.json": "998a078123ffc97763690de7f2a677eb89168af5eaf8a5e12e6bc24d18e25bdb",
    "model-00001-of-00004.safetensors": "d0b23ff314afe68e49b3d620c26d4eb98ee842e62a4ed5b2518a2796c5d628c3",
    "model-00002-of-00004.safetensors": "be1d942bcd33edf4d766ee7a46f01a1f27b676d40bcb4721ba8e2443a640bd46",
    "model-00003-of-00004.safetensors": "c3e13bd91518db5b94a26b357490fe4ccc310e2a274d28d73465e2d62ca57256",
    "model-00004-of-00004.safetensors": "cbaa8a7cd8d7b4a8291993ba877104c43f87e4eb9e69680d13cc2ea24bd926d0",
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode())


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def model_contract(model_path=MODEL_DIR, *, verify_weights=True):
    """Verify published file identities without loading a tensor or using CUDA."""
    model_path = Path(model_path)
    recorded = json.loads((model_path / "source_revision.json").read_text())
    if recorded != {"repo_id": REPO, "revision": REVISION}:
        raise ValueError("The actual checkpoint is not the pinned official SuperWriter release")
    verified = {}
    for name, expected in EXPECTED_FILES.items():
        path = model_path / name
        if not path.is_file():
            raise ValueError("Missing complete official snapshot file: " + name)
        if verify_weights or not name.endswith(".safetensors"):
            actual = file_sha(path)
            if actual != expected:
                raise ValueError("Official snapshot content changed: " + name)
            verified[name] = actual
    config = json.loads((model_path / "config.json").read_text())
    token = json.loads((model_path / "tokenizer_config.json").read_text())
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    if (config.get("architectures") != ["Qwen2ForCausalLM"] or config.get("model_type") != "qwen2"
            or config.get("max_position_embeddings") != CONTEXT_LIMIT):
        raise ValueError("Unexpected published model architecture or context limit")
    return {"repo_id": REPO, "revision": REVISION, "model_path": str(model_path.resolve()),
            "architecture": "Qwen2ForCausalLM", "tensor_bytes": index["metadata"]["total_size"],
            "chat_template_sha256": sha(token["chat_template"].encode()),
            "model_context_limit": CONTEXT_LIMIT, "tokenizer_reported_context_limit": token["model_max_length"],
            "verified_files": verified, "all_weight_shas_verified": verify_weights,
            "expected_files": deepcopy(EXPECTED_FILES), "no_remote_code": True}


def native_backend(contract):
    return {"transport": "local", "source": "local", "model": REPO, "revision": REVISION,
            "workflow": VERSION, "chat_template_sha256": contract["chat_template_sha256"]}


def validate_backend(backend):
    expected = {"transport": "local", "source": "local", "model": REPO, "revision": REVISION,
                "workflow": VERSION, "chat_template_sha256": CHAT_TEMPLATE_SHA256}
    if backend != expected:
        raise ValueError("Native SuperWriter cannot use a Wiki/JSON writer or another model revision")


def native_request(key, stage, prompt):
    if stage not in STAGES or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Native inference requires a named official stage and complete raw prompt")
    return {"key": key, "workflow": VERSION, "stage": stage, "prompt": prompt,
            "messages": [{"role": "user", "content": prompt}]}


def check_request(request):
    if request != native_request(request.get("key"), request.get("stage"), request.get("prompt")):
        raise ValueError("Native SuperWriter requires the unmodified official user prompt")
    if "adaptation_process" in request["prompt"] and '"request_json"' in request["prompt"]:
        raise ValueError("A Qwen3.5 adaptation_process JSON projection cannot be a native request")


def strip_terminal(text):
    result = text.strip()
    for terminal in ("<|im_end|>",) + STOP_STRINGS:
        if result.endswith(terminal):
            result = result[:-len(terminal)].rstrip()
            break
    return result


def parse_native(stage, raw):
    """Preserve native sections; never manufacture tags, JSON, or missing text."""
    if stage not in STAGES or not isinstance(raw, str):
        raise ValueError("Unavailable native stage output")
    text = strip_terminal(raw)
    if text.startswith(("{", "[", "```json")):
        raise ValueError("JSON transport is not a native SuperWriter stage output")
    pattern = re.compile(r"<(think|answer)>(.*?)</\1>", re.DOTALL)
    sections, envelope, cursor = [], [], 0
    for match in pattern.finditer(text):
        outside = text[cursor:match.start()]
        if re.search(r"</?(?:think|answer)(?:>|\b)", outside) or re.search(r"</?(?:think|answer)>", match[2]):
            raise ValueError("Unbalanced or nested native stage content")
        if outside.strip():
            envelope.append(outside)
        if not match[2].strip():
            raise ValueError("Empty native stage section")
        sections.append({"kind": match[1], "text": match[2].strip()})
        cursor = match.end()
    if re.search(r"</?(?:think|answer)(?:>|\b)", text[cursor:]):
        raise ValueError("Unbalanced native section in the final envelope")
    if text[cursor:].strip():
        envelope.append(text[cursor:])
    if not sections or sections[0]["kind"] != "think":
        raise ValueError("Native stage output has no complete initial thinking section")
    thinks = [s["text"] for s in sections if s["kind"] == "think"]
    answers = [s["text"] for s in sections if s["kind"] == "answer"]
    if stage == "plan":
        valid = len(answers) == 2 and len(thinks) in (1, 2)
    else:
        valid = 4 <= len(answers) <= 16 and len(answers) == len(thinks)
        valid = valid and [s["kind"] for s in sections] == ["think", "answer"] * len(answers)
    if not valid:
        raise ValueError("Native official stage section-count contract failed")
    return {"answer_sections": answers, "answer": "\n".join(answers),
            "think_sections_count": len(thinks), "answer_sections_count": len(answers),
            "native_sections": sections, "native_envelope_text": envelope,
            "envelope_included_in_public_answer": False, "minimum_length_gate_applied": False}


def plan_request(unit):
    unit = validate_unit(unit)
    query = json.dumps({"source": unit["source"]}, ensure_ascii=False, sort_keys=True)
    prompt = official_functions()["generate_prompt"]("plan", use_query=query)
    return native_request(unit["id"] + "__plan", "plan", prompt)


def read_cached_plan(path, request, backend):
    """Reuse actual first-attempt output only when all generation inputs agree.

    An earlier local parsing failure may be reparsed; an unfinished model
    response or a different task/checkpoint cannot become a reusable plan.
    """
    validate_backend(backend)
    check_request(request)
    path = Path(path)
    record = json.loads(path.read_text())
    if (request["stage"] != "plan" or record.get("stage") != "plan"
            or record.get("request") != request or record.get("request_sha256") != fingerprint(request)
            or record.get("requested_backend") != backend):
        raise ValueError("Cached native plan belongs to different exact source, brief, task, or checkpoint inputs")
    response = record.get("response", {})
    if (response.get("served_revision") != REVISION or response.get("served_model") != REPO
            or response.get("source_requested") != "local" or response.get("finish_reason") != "stop"
            or response.get("chat_template_sha256") != backend["chat_template_sha256"]):
        raise ValueError("Cached native plan lacks the actual complete official checkpoint receipt")
    parse_native("plan", response["text"])
    return {**deepcopy(response), "reused_from": {"path": str(path.resolve()), "sha256": file_sha(path),
                                                 "request_sha256": record["request_sha256"]}}


class ReuseNativePlan:
    """Pass through exactly two later stages without generating the plan twice."""
    def __init__(self, generate, plan_path):
        self.generate, self.plan_path = generate, Path(plan_path)

    def __call__(self, request, backend):
        if request["stage"] == "plan":
            return read_cached_plan(self.plan_path, request, backend)
        return self.generate(request, backend)


def execute_native(unit, generate, *, backend, on_record=None):
    """Execute three native stages once; callback fixtures cannot certify taste."""
    unit = validate_unit(unit)
    validate_backend(backend)
    functions = official_functions()
    query = json.dumps({"source": unit["source"]}, ensure_ascii=False, sort_keys=True)
    trace, previous = [], None
    result = {"workflow": VERSION, "backend": deepcopy(backend), "unit": deepcopy(unit),
              "unit_sha256": fingerprint(unit), "status": "failed", "stage_records": trace,
              "quality_gain_established": False, "human_preference_labels_created": 0,
              "not_a_workflow_win_rate_comparison": True}
    for stage in STAGES:
        prompt = functions["generate_prompt"](stage, use_query=query, previous_output=previous)
        request = native_request(unit["id"] + "__" + stage, stage, prompt)
        record = {"stage": stage, "request": request, "request_sha256": fingerprint(request),
                  "requested_backend": deepcopy(backend), "status": "failed"}
        trace.append(record)
        try:
            response = generate(deepcopy(request), deepcopy(backend))
            record["response"] = deepcopy(response)
            if (response.get("served_revision") != REVISION or response.get("served_model") != REPO
                    or response.get("source_requested") != "local"
                    or response.get("chat_template_sha256") != backend["chat_template_sha256"]):
                raise ValueError("Actual served checkpoint, tokenizer, or transport differs")
            if response.get("finish_reason") != "stop":
                raise ValueError("Native response is unavailable or incomplete")
            parsed = parse_native(stage, response["text"])
            record.update(status="complete", parsed=parsed)
            previous = response["text"]
        except Exception as exc:
            record.update(error_type=type(exc).__name__, error=str(exc)[:250])
        if on_record is not None:
            on_record(deepcopy(record))
        if record["status"] != "complete":
            result.update(error_type=record["error_type"], error=record["error"])
            break
    else:
        result.update(status="complete", script=trace[-1]["parsed"]["answer"],
                      initial_plan=trace[0]["parsed"]["answer"], initial_script=trace[1]["parsed"]["answer"])
    result["executed_calls"] = len(trace)
    result["reused_stage_count"] = sum(bool(r.get("response", {}).get("reused_from")) for r in trace)
    result["new_inference_calls"] = len(trace) - result["reused_stage_count"]
    return result


class NativeSuperWriterInference:
    """Actual local callback; constructing it is an explicitly admitted GPU step."""
    def __init__(self, *, model_path=MODEL_DIR, context_limit=CONTEXT_LIMIT,
                 memory_floor_mib=32000, gpu_admitted=False):
        if gpu_admitted is not True:
            raise ValueError("GPU inference requires explicit resource admission")
        if not 2 <= context_limit <= CONTEXT_LIMIT:
            raise ValueError("Use the actual published model context without extension")
        contract = model_contract(model_path)
        free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                                           "--format=csv,noheader,nounits"], text=True).splitlines()[0])
        if free < memory_floor_mib or free <= 8192:
            raise RuntimeError("Insufficient measured GPU memory for admitted native inference")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected local environment cannot run CUDA BF16")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., (free - 8192) * 1024**2 / total), device=0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if sha(self.tokenizer.chat_template.encode()) != contract["chat_template_sha256"]:
            raise ValueError("The actual tokenizer template differs from the official release")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
        self.model.eval()
        self.backend = native_backend(contract)
        self.context_limit = context_limit
        self.allocation_policy = {"initial_free_mib": free, "safety_reserved_mib": 8192,
                                  "per_process_allowance_mib": free - 8192}

    def __call__(self, request, backend):
        validate_backend(backend)
        if backend != self.backend:
            raise ValueError("Native inference checkpoint binding changed")
        check_request(request)
        if sha(self.tokenizer.chat_template.encode()) != backend["chat_template_sha256"]:
            raise ValueError("Native chat template changed after freezing")
        import torch
        encoded = self.tokenizer.apply_chat_template(request["messages"], tokenize=True,
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        prompt_tokens = encoded["input_ids"].shape[-1]
        receipt = {"served_revision": REVISION, "served_model": REPO, "source_requested": "local",
                   "chat_template_sha256": backend["chat_template_sha256"], "allocation_policy": self.allocation_policy}
        if prompt_tokens >= self.context_limit:
            return {**receipt, "text": "", "finish_reason": "context_overflow",
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0}}
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**encoded, max_new_tokens=self.context_limit-prompt_tokens,
                do_sample=True, temperature=0.6, top_p=0.95, top_k=0, repetition_penalty=1.0,
                use_cache=True, eos_token_id=list(EOS_IDS), pad_token_id=151643,
                stop_strings=list(STOP_STRINGS), tokenizer=self.tokenizer)
        generated = output[0, prompt_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        terminated = bool(generated.numel() and int(generated[-1]) in EOS_IDS)
        terminated = terminated or any(text.rstrip().endswith(s) for s in STOP_STRINGS)
        return {**receipt, "text": text, "finish_reason": "stop" if terminated else "length",
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": generated.numel(),
                          "elapsed_seconds": time.monotonic()-start,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved()}}


def prepare(unit, out, *, model_path=MODEL_DIR, context_limit=CONTEXT_LIMIT, reuse_plan_stage=None):
    """Freeze the native protocol and tokenize the first full source prompt on CPU."""
    unit = validate_unit(unit)
    contract = model_contract(model_path)
    backend = native_backend(contract)
    validate_backend(backend)
    if not 2 <= context_limit <= CONTEXT_LIMIT:
        raise ValueError("Invalid context allowance for the actual model")
    query = json.dumps({"source": unit["source"]}, ensure_ascii=False, sort_keys=True)
    prompt = official_functions()["generate_prompt"]("plan", use_query=query)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    encoded = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                           add_generation_prompt=True, return_dict=True)
    protocol = {"version": VERSION, "backend": backend, "unit_sha256": fingerprint(unit),
                "official_code_revision": CODE_REVISION, "stages": list(STAGES), "context_limit": context_limit,
                "runner_sha256": file_sha(__file__), "source_scope": unit["source_boundaries"].get("scope"),
                "first_prompt_tokens": len(encoded["input_ids"]),
                "native_input": "Only the original user prompt and the released Human:/Assistant: chat template",
                "source_availability": "Full source and brief in plan; later native prompts consume the previous answer sections",
                "decode": {"seed": 42, "temperature": 0.6, "top_p": 0.95, "top_k": 0,
                           "repetition_penalty": 1.0, "stop_strings": list(STOP_STRINGS), "eos_ids": list(EOS_IDS)},
                "adaptations": ["No upstream minimum word count", "No format-driven retries",
                                "Use the remaining published context for generation; preserve overflow and unfinished outputs"],
                "json_transport": False, "paid_api_calls": 0, "remote_fallback": False,
                "quality_gain_established": False, "evaluation_or_capability_authorization_required": True}
    if protocol["first_prompt_tokens"] >= context_limit:
        raise ValueError("The complete source prompt does not fit; do not truncate it")
    if reuse_plan_stage is not None:
        cached = read_cached_plan(reuse_plan_stage, plan_request(unit), backend)
        protocol["cached_plan_stage"] = cached["reused_from"]
        protocol["new_inference_stages"] = ["write", "refine"]
    path = Path(out) / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Native stage protocol changed; use a new explicitly versioned output directory")
    save(path, protocol)
    save(Path(out) / "model_verification.json", contract)
    save(Path(out) / "source_unit.json", unit)
    return protocol


def validate_capability_grant(path, unit):
    """Bind a root-approved single DEV capability check without certifying quality."""
    grant = json.loads(Path(path).read_text())
    if (grant.get("scope") != "official_superwriter_single_dev_capability"
            or grant.get("task_id") != unit["id"] or grant.get("unit_sha256") != fingerprint(validate_unit(unit))
            or grant.get("gpu_admitted") is not True or grant.get("max_workflow_attempts") != 1
            or not grant.get("author_instruction") or grant.get("quality_claim_authorized") is not False):
        raise ValueError("An actual task-bound single-DEV capability authorization is required")
    return {"path": str(Path(path).resolve()), "sha256": file_sha(path), "scope": grant["scope"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-unit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--context-limit", type=int, default=CONTEXT_LIMIT)
    parser.add_argument("--memory-floor-mib", type=int, default=32000)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gpu-admitted", action="store_true")
    parser.add_argument("--evaluator-report", type=Path)
    parser.add_argument("--capability-authorization", type=Path)
    parser.add_argument("--reuse-plan-stage", type=Path)
    args = parser.parse_args()
    unit = validate_unit(json.loads(args.source_unit.read_text()))
    protocol = prepare(unit, args.out, model_path=args.model_dir, context_limit=args.context_limit,
                       reuse_plan_stage=args.reuse_plan_stage)
    if not args.run:
        print(json.dumps({"status": "prepared_cpu_only", "task_id": unit["id"],
                          "first_prompt_tokens": protocol["first_prompt_tokens"]}))
        return
    if not args.gpu_admitted:
        raise ValueError("--run requires explicit --gpu-admitted after resource coordination")
    if bool(args.evaluator_report) == bool(args.capability_authorization):
        raise ValueError("Provide exactly one applicable evaluator report or single-DEV capability grant")
    if args.capability_authorization:
        admission = validate_capability_grant(args.capability_authorization, unit)
    else:
        admission = validate_evaluator({"report_path": str(args.evaluator_report),
                                        "report_sha256": file_sha(args.evaluator_report)})
    with (args.out / "execution.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.out / "attempt_started.json").exists():
            raise ValueError("This native capability attempt has already started; preserve it without rerunning")
        save(args.out / "attempt_started.json", {"at_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
             "admission": admission, "unit_sha256": protocol["unit_sha256"]})
        try:
            infer = NativeSuperWriterInference(model_path=args.model_dir, context_limit=args.context_limit,
                                               memory_floor_mib=args.memory_floor_mib, gpu_admitted=True)
            generate = ReuseNativePlan(infer, args.reuse_plan_stage) if args.reuse_plan_stage else infer
            result = execute_native(unit, generate, backend=protocol["backend"],
                on_record=lambda row: save(args.out / "stages" / (row["stage"] + ".json"), row))
        except Exception as exc:
            result = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:250],
                      "workflow": VERSION, "quality_gain_established": False, "backend": protocol["backend"]}
        result["admission"] = admission
        save(args.out / "result.json", result)
        if result["status"] == "complete":
            (args.out / "script.md").write_text(result["script"] + "\n")
        print(json.dumps({"status": result["status"], "executed_calls": result.get("executed_calls", 0),
                          "quality_gain_established": False}))


if __name__ == "__main__":
    main()
