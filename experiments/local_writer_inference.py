"""Local Qwen writer inference using the released training request interface.

Importing this module never loads weights, initializes CUDA, or calls an API.
The caller explicitly admits the GPU allocation and freezes the actual backend.
"""
import hashlib
import json
from pathlib import Path
import time

from train_local_writer import (MODEL, REVISION, LORA_TARGETS, input_projection_contract,
                                ensure_memory, limit_writer_memory, load_writer)


def adapter_identity(directory):
    """Bind the complete saved adapter to its actual pinned writing backbone."""
    if directory is None:
        return REVISION, None
    directory = Path(directory)
    files = [directory / "adapter_config.json", directory / "adapter_model.safetensors"]
    if any(not path.is_file() for path in files):
        raise ValueError("A local writer adapter requires its actual config and weights")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return REVISION + "+writer_adapter:" + digest.hexdigest(), json.loads(files[0].read_text())


def check_messages(messages):
    """Reject a prompt that departs from the inference-available SFT projection."""
    from prepare_local_writer_data import render_request
    if not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user"]:
        raise ValueError("Local writing requires exactly the released system/user request")
    value = json.loads(messages[1]["content"])
    if not isinstance(value, dict) or "task" not in value:
        raise ValueError("The writing request requires an explicit supervised task")
    task = value.pop("task")
    if render_request(task, value) != messages:
        raise ValueError("Local inference differs from the frozen training request")
    return task


class LocalWriterInference:
    """Implement infer(messages, backend) for LocalWriterGenerator.

    Complete targets receive the available context allowance. Reaching that
    allowance without the chat EOS is a failure, never a completed script.
    """
    def __init__(self, *, model_path=MODEL, adapter=None, qlora=False,
                 context_limit=16384, memory_floor_mib=32000):
        from transformers import AutoTokenizer
        model_path = Path(model_path)
        recorded = json.loads((model_path / "source_revision.json").read_text())
        if recorded != {"repo_id": "Qwen/Qwen3.5-9B", "revision": REVISION}:
            raise ValueError("The actual local writer differs from the pinned Qwen9B")
        if context_limit <= 1:
            raise ValueError("The complete local context allowance must exceed one token")
        self.revision, adapter_config = adapter_identity(adapter)
        rank = 16 if adapter_config is None else adapter_config["r"]
        if adapter_config is not None and (
                set(adapter_config["target_modules"]) != set(LORA_TARGETS)
                or adapter_config["lora_alpha"] != 2 * rank
                or adapter_config.get("bias") != "none"):
            raise ValueError("Saved writer adapter is incompatible with the implemented text-only LoRA")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        free = ensure_memory(memory_floor_mib)
        self.allocation_policy = limit_writer_memory(free)
        self.model = load_writer(model_path, self.tokenizer, rank, qlora)
        if adapter is not None:
            from peft.utils.save_and_load import (load_peft_weights,
                set_peft_model_state_dict, get_peft_model_state_dict)
            import torch
            weights = load_peft_weights(str(adapter), device="cpu")
            result = set_peft_model_state_dict(self.model, weights, adapter_name="default")
            actual = get_peft_model_state_dict(self.model, adapter_name="default")
            if result.unexpected_keys or set(actual) != set(weights):
                raise ValueError("Saved writer adapter keys were not loaded exactly")
            if any(not torch.equal(actual[k].detach().cpu(), weights[k].to(actual[k].dtype)) for k in actual):
                raise ValueError("The actual writer adapter differs from its recorded weights")
            if adapter_identity(adapter)[0] != self.revision:
                raise ValueError("Writer adapter changed while its checkpoint was loaded")
        self.model.eval()
        self.context_limit = context_limit
        self.backend = {"transport": "local", "model": "Qwen/Qwen3.5-9B",
                        "revision": self.revision}
        self.protocol = {"base": recorded, "served_revision": self.revision,
                         "context_limit": context_limit, "enable_thinking": False,
                         "input_projection_sha256": input_projection_contract(),
                         "adapter": None if adapter is None else str(Path(adapter).resolve()),
                         "qlora": qlora, "allocation_policy": self.allocation_policy,
                         "api_fallback": False}

    def __call__(self, messages, backend):
        import torch
        if backend != self.backend:
            raise ValueError("Requested local writer identity differs from the loaded checkpoint")
        if input_projection_contract() != self.protocol["input_projection_sha256"]:
            raise ValueError("Writing request projection changed after backend freezing")
        task = check_messages(messages)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        batch = self.tokenizer(prompt, add_special_tokens=False, truncation=False,
                               return_tensors="pt")
        prompt_tokens = batch["input_ids"].shape[1]
        result = {"served_model": self.backend["model"], "served_revision": self.revision,
                  "source_requested": "local", "task": task}
        if prompt_tokens >= self.context_limit:
            return {**result, "text": "", "finish_reason": "context_overflow",
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0},
                    "error": "Complete request exceeds local context support; no truncation performed"}
        started = time.monotonic()
        device = self.model.get_input_embeddings().weight.device
        with torch.inference_mode():
            output = self.model.generate(**{k: v.to(device) for k, v in batch.items()},
                max_new_tokens=self.context_limit - prompt_tokens, do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True)
        answer = output[0, prompt_tokens:].tolist()
        stopped = bool(answer and answer[-1] == self.tokenizer.eos_token_id)
        text_ids = answer[:-1] if stopped else answer
        return {**result, "text": self.tokenizer.decode(text_ids, skip_special_tokens=False),
                "finish_reason": "stop" if stopped else "length",
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(answer),
                          "elapsed_seconds": time.monotonic() - started}}


def development_probe(dev_path, record_ids, output, *, adapter=None, qlora=False):
    """Inspect real local writing on existing DEV requests without fitting labels.

    The original reference is retained for subsequent close reading. It never
    enters the model request. The probe establishes neither human preference
    nor a controlled quality gain and does not contribute a paper result.
    """
    from train_local_writer import read_rows, save_json, timestamp
    identifiers = list(record_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("The development probe requires distinct explicit DEV IDs")
    rows = {row["id"]: row for row in read_rows(dev_path, "dev")}
    if not set(identifiers) <= set(rows):
        raise ValueError("A requested probe ID is absent from the existing DEV split")
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve an existing development probe; choose a new output")
    snapshot = hashlib.sha256(Path(dev_path).read_bytes()).hexdigest()
    report = {"created_at_beijing": timestamp(), "purpose": "Local writer development capability check",
              "data_split": "dev", "dev_sha256": snapshot,
              "reference_visible_to_model": False, "training_started": False,
              "optimizer_updates": 0, "human_preference_labels_created": 0,
              "controlled_quality_gain_established": False,
              "paper_result": False, "status": "loading", "items": []}
    save_json(output, report)
    try:
        writer = LocalWriterInference(adapter=adapter, qlora=qlora)
        report.update(protocol=writer.protocol, status="generating",
                      writer_kind="base_model" if adapter is None else "saved_writer_adapter")
        save_json(output, report)
        for identifier in identifiers:
            row = rows[identifier]
            answer = writer(row["messages"][:-1], writer.backend)
            report["items"].append({"id": identifier, "group_id": row["group_id"],
                "task": row["task"], "inputs": row["inputs"],
                "input_sha256": row["input_sha256"],
                "reference": row["messages"][-1]["content"],
                "reference_sha256": row["target_sha256"],
                "response": answer, "complete_generation": answer["finish_reason"] == "stop"})
            save_json(output, report)
            print(json.dumps({"id": identifier, "finish_reason": answer["finish_reason"],
                              "output_characters": len(answer["text"])}), flush=True)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        save_json(output, report)
        raise
    save_json(output, report)
    return report


if __name__ == "__main__":
    import argparse
    from train_local_writer import DATA
    parser = argparse.ArgumentParser(description="Probe existing DEV requests with the pinned local writer")
    parser.add_argument("--dev-data", type=Path, default=DATA / "dev.jsonl")
    parser.add_argument("--record-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()
    development_probe(args.dev_data, args.record_id, args.output,
                      adapter=args.adapter, qlora=args.qlora)
