"""Preflight and train the local text-only Qwen3.5 writer without truncation.

CPU preflight is the default. Training requires an explicit release manifest and
never invokes an API, changes a data partition, or manages another GPU process.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/restored/writer"
MODEL = ROOT / "models/qwen3_5_9b"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TASKS = {"revise_text", "adapt_scene", "continue_script", "revise_with_feedback",
         "write_scene_from_synopsis", "adaptation_process"}
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv",
                "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                "gate_proj", "up_proj", "down_proj")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def input_projection_contract():
    """Bind the exact role instructions and serializer shared with inference."""
    from prepare_local_writer_data import INSTRUCTIONS, INPUT_FIELDS, render_request
    values = {"instructions": INSTRUCTIONS,
              "fields": {task: sorted(fields) for task, fields in INPUT_FIELDS.items()},
              "renderer": inspect.getsource(render_request)}
    return sha(json.dumps(values, ensure_ascii=False, sort_keys=True).encode())


def trial_wrapper_candidate(row, stage_instruction):
    """Bind an existing scene ID without inventing or rewriting dramatic content.

    This creates a replacement candidate, not an additional independent example
    or a training release. All original targets remain untouched.
    """
    from adaptation_system_contract import validate_program
    from prepare_local_writer_data import render_request
    if (row.get("task") != "adaptation_process" or row.get("split") != "train"
            or row.get("quality_status") != "accepted_for_declared_writer_task"
            or row["inputs"].get("stage") != "trial"):
        raise ValueError("Only admitted TRAIN scene-trial records may be converted")
    payload = json.loads(row["inputs"]["request_json"])
    program = validate_program(payload["program"])
    scene_id = payload["exploration"]["scene_id"]
    if scene_id not in {scene["scene_id"] for scene in program["scenes"]}:
        raise ValueError("The original request does not bind an existing scene")
    target = json.loads(row["messages"][-1]["content"])
    if set(target) != {"text"} or not isinstance(target["text"], str) or not target["text"].strip():
        raise ValueError("Conversion requires the complete original text-only scene object")
    result = deepcopy(row)
    result["id"] = "wrapper-conversion:" + row["id"]
    result["inputs"]["stage_instruction"] = stage_instruction
    result["messages"] = render_request("adaptation_process", result["inputs"]) + [
        {"role": "assistant", "content": json.dumps({"scene_id": scene_id, "text": target["text"]},
                                                    ensure_ascii=False, sort_keys=True)}]
    complete = result["messages"][-1]["content"]
    result["input_sha256"] = sha(json.dumps(result["inputs"], ensure_ascii=False, sort_keys=True).encode())
    result["target_sha256"] = sha(complete.encode())
    result["normalized_target_sha256"] = sha(" ".join(complete.split()).encode())
    result["target_characters"] = len(complete)
    result["training_authorized"] = False
    result["quality_status"] = "candidate_for_reversible_trial_wrapper_conversion"
    result["wrapper_conversion"] = {
        "replaces_record_id": row["id"], "original_target_sha256": row["target_sha256"],
        "original_input_sha256": row["input_sha256"],
        "original_stage_instruction": row["inputs"]["stage_instruction"],
        "scene_text_sha256": sha(target["text"].encode()),
        "scene_text_changed": False, "original_request_payload_changed": False,
        "counts_as_additional_independent_example": False,
        "scope": "Existing inference-available scene identifier added to target packaging; no new aesthetic label"}
    return result


def read_rows(path, split):
    from prepare_local_writer_data import render_request
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("An empty split cannot support writer development")
    identifiers = set()
    for row in rows:
        if row["split"] != split or row["task"] not in TASKS:
            raise ValueError("Unexpected partition or unsupported writer task")
        if row["id"] in identifiers or not row.get("group_id"):
            raise ValueError("Duplicate identifiers or missing source group")
        identifiers.add(row["id"])
        messages = row["messages"]
        if [m["role"] for m in messages] != ["system", "user", "assistant"]:
            raise ValueError("Expected the frozen writer input and one complete target")
        if any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
            raise ValueError("Empty or multimodal writer messages")
        target = messages[-1]["content"]
        if sha(target.encode()) != row["target_sha256"]:
            raise ValueError("The complete writing target changed")
        normalized = " ".join(target.split())
        if sha(normalized.encode()) != row["normalized_target_sha256"]:
            raise ValueError("Target normalization or its source digest changed")
        inputs = json.dumps(row["inputs"], ensure_ascii=False, sort_keys=True)
        if sha(inputs.encode()) != row["input_sha256"]:
            raise ValueError("Inference-visible input changed")
        if messages[:-1] != render_request(row["task"], row["inputs"]):
            raise ValueError("Writer role instructions or serialization differ from inference")
        if json.loads(messages[1]["content"]) != {"task": row["task"], **row["inputs"]}:
            raise ValueError("Model-visible user message differs from the frozen task input")
        if row.get("reward_holdout_group_quarantined"):
            raise ValueError("A protected reward group entered the writer split")
        if split == "train" and row.get("quality_status") != "accepted_for_declared_writer_task":
            raise ValueError("Writer TRAIN must use admitted task-qualified records")
    return rows


def partition_audit(train, dev, protection_path=None):
    """Check existing group and exact-target protection without reading TEST."""
    groups = [{r["group_id"] for r in rows} for rows in (train, dev)]
    hashes = [{r["normalized_target_sha256"] for r in rows} for rows in (train, dev)]
    if groups[0] & groups[1] or hashes[0] & hashes[1]:
        raise ValueError("Writer TRAIN and DEV share a source group or target")
    report = {"cross_train_dev_groups": 0, "cross_train_dev_exact_targets": 0,
              "test_text_or_labels_read": False,
              "global_external_author_or_translation_clearance": False}
    if protection_path is not None:
        protection = json.loads(Path(protection_path).read_text())
        held = set(protection["known_held_groups"])
        if groups[0] & held:
            raise ValueError("A previously withheld source group was restored to TRAIN")
        report.update({"known_held_groups": len(held),
                       "accepted_train_matches_known_holds": 0,
                       "source_audit_sha256": sha(Path(protection_path).read_bytes()),
                       "local_lexical_overlap_audit": protection["local_dev_lexical_overlap"],
                       "protected_reward_audit": protection["reward_protection"]})
    return report


def encode_target(tokenizer, row):
    """Use the official non-thinking prefix and supervise the complete answer/EOS."""
    messages = row["messages"]
    target = messages[-1]["content"].strip()
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    complete = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False)
    expected = prompt + target + tokenizer.eos_token + "\n"
    if complete != expected:
        raise ValueError("The official template transformed the writing target")
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    ids = tokenizer(complete, add_special_tokens=False, truncation=False)["input_ids"]
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Token boundary differs between inference and supervised input")
    terminal = tokenizer.eos_token_id
    eos_positions = [i for i in range(len(prompt_ids), len(ids)) if ids[i] == terminal]
    if len(eos_positions) != 1:
        raise ValueError("Writing target includes a reserved chat EOS or has no EOS")
    end = eos_positions[0] + 1
    ids = ids[:end]
    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
    if not labels or labels[-1] != terminal or len(ids) != len(labels):
        raise ValueError("Complete assistant labels or termination are invalid")
    if tokenizer.decode(ids[len(prompt_ids):-1]) != target:
        raise ValueError("Decoded supervised tokens differ from the complete writing target")
    return {"input_ids": ids, "labels": labels, "prompt_tokens": len(prompt_ids),
            "target_tokens": end - len(prompt_ids), "id": row["id"], "task": row["task"],
            "group_id": row["group_id"]}


def describe_lengths(values):
    lengths = sorted(values)
    return {"minimum": lengths[0], "maximum": lengths[-1], "total": sum(lengths),
            "p50": lengths[int(.50 * (len(lengths) - 1))],
            "p90": lengths[int(.90 * (len(lengths) - 1))],
            "p95": lengths[int(.95 * (len(lengths) - 1))],
            "p99": lengths[int(.99 * (len(lengths) - 1))]}


def prepare(train_path, dev_path, model_path, max_tokens, protection_path=None):
    from transformers import AutoTokenizer
    started = time.monotonic()
    revision = json.loads((Path(model_path) / "source_revision.json").read_text())
    if revision != {"repo_id": "Qwen/Qwen3.5-9B", "revision": REVISION}:
        raise ValueError("The local writer backbone differs from the pinned Qwen9B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    train, dev = read_rows(train_path, "train"), read_rows(dev_path, "dev")
    audit = partition_audit(train, dev, protection_path)
    encoded, excluded, statistics = {}, {}, {}
    for split, rows in (("train", train), ("dev", dev)):
        all_encoded = [encode_target(tokenizer, row) for row in rows]
        excluded[split] = [{"id": r["id"], "task": r["task"],
                            "tokens": len(r["input_ids"]), "group_id": r["group_id"]}
                           for r in all_encoded if len(r["input_ids"]) > max_tokens]
        encoded[split] = [r for r in all_encoded if len(r["input_ids"]) <= max_tokens]
        by_task = {}
        for task in sorted({r["task"] for r in all_encoded}):
            selected = [r for r in all_encoded if r["task"] == task]
            by_task[task] = {"rows": len(selected),
                             "complete_input_tokens": describe_lengths([len(r["input_ids"]) for r in selected]),
                             "assistant_target_tokens": describe_lengths([r["target_tokens"] for r in selected])}
        statistics[split] = {"source_rows": len(rows), "usable_complete_rows": len(encoded[split]),
                             "groups": len({r["group_id"] for r in rows}),
                             "complete_input_tokens": describe_lengths([len(r["input_ids"]) for r in all_encoded]),
                             "assistant_target_tokens": describe_lengths([r["target_tokens"] for r in all_encoded]),
                             "by_task": by_task}
    config = json.loads((Path(model_path) / "config.json").read_text())
    report = {"created_at_beijing": timestamp(), "model": revision,
              "train_sha256": sha(Path(train_path).read_bytes()),
              "dev_sha256": sha(Path(dev_path).read_bytes()), "max_tokens": max_tokens,
              "statistics": statistics, "excluded_complete_rows": excluded,
              "partition_audit": audit, "text_only": True, "enable_thinking": False,
              "mask": "System/user/assistant-header/padding masked; complete answer and chat EOS supervised",
              "template_outer_whitespace_trim": True, "target_words_removed": 0,
              "chat_eos_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
              "bare_text_config_eos_id": config["text_config"]["eos_token_id"],
              "template_sha256": sha(tokenizer.chat_template.encode()),
              "input_projection_sha256": input_projection_contract(),
              "training_started": False, "gpu_loaded": False, "api_calls": 0,
              "seconds": time.monotonic() - started,
              "tasks_without_existing_dev": sorted({r["task"] for r in train} - {r["task"] for r in dev}),
              "same_data_baseline_requirement": "Use exactly the same admitted tasks, complete-input IDs and backbone for ordinary SFT controls",
              "scope": "Token/target/partition preflight; no aesthetic or adaptation gain established"}
    return tokenizer, encoded, report


def collate(rows, pad_id, device="cpu"):
    import torch
    width = max(len(row["input_ids"]) for row in rows)
    ids, masks, labels = [], [], []
    for row in rows:
        padding = width - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * padding)
        masks.append([1] * len(row["input_ids"]) + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
    return {"input_ids": torch.tensor(ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device)}


def chunked_causal_loss(hidden, head, labels, chunk_size=128, rematerialize=True):
    """Compute exact assistant-token CE without allocating all vocabulary logits."""
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    target = labels[:, 1:]
    valid = target != -100
    states, gold = hidden[:, :-1][valid], target[valid]
    if gold.numel() == 0:
        raise ValueError("No assistant token contributes to causal loss")
    loss = hidden.new_zeros((), dtype=torch.float32)
    def project_and_sum(x, y):
        return F.cross_entropy(head(x.to(head.weight.dtype)).float(), y, reduction="sum")
    for start in range(0, len(gold), chunk_size):
        x, y = states[start:start + chunk_size], gold[start:start + chunk_size]
        loss = loss + (checkpoint(project_and_sum, x, y, use_reentrant=False)
                       if rematerialize and torch.is_grad_enabled() else project_and_sum(x, y))
    return loss / len(gold), len(gold)


def writer_loss(model, batch, chunk_size):
    base = model.get_base_model()
    hidden = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False, return_dict=True).last_hidden_state
    return chunked_causal_loss(hidden, base.get_output_embeddings(), batch["labels"], chunk_size)


def load_writer(path, tokenizer, rank=16, qlora=False):
    import torch
    from transformers import AutoConfig, BitsAndBytesConfig, Qwen3_5ForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    config = AutoConfig.from_pretrained(path, local_files_only=True).get_text_config()
    config.eos_token_id, config.pad_token_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    config.use_cache = False
    options = {"config": config, "dtype": torch.bfloat16, "local_files_only": True,
               "attn_implementation": "sdpa"}
    if qlora:
        options.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16), device_map={"": "cuda:0"})
    model = Qwen3_5ForCausalLM.from_pretrained(path, **options)
    if qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model = model.cuda()
        # Retain the pretrained decoder and language head during key conversion.
        from script_preference_backends import check_qwen_weights
        from safetensors import safe_open
        check_qwen_weights(model.model, path)
        index = json.loads((Path(path) / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(str(Path(path) / index["lm_head.weight"]), framework="pt", device="cpu") as source:
            expected_head_rows = source.get_slice("lm_head.weight")[:2]
        if not torch.equal(model.get_output_embeddings().weight[:2].detach().cpu(), expected_head_rows):
            raise ValueError("The pretrained writing head was not preserved")
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=.05,
        target_modules=list(LORA_TARGETS), bias="none", task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if any("visual" in name or "vision" in name for name, _ in model.named_parameters()):
        raise ValueError("Text-only writer unexpectedly includes the vision encoder")
    return model


def ensure_memory(minimum_mib):
    free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    if free < minimum_mib:
        raise RuntimeError(f"Writer GPU work deferred: {free} MiB free < {minimum_mib} MiB required")
    return free


def limit_writer_memory(initial_free_mib, safety_mib=8192, ceiling_mib=None):
    """Limit this process to current free VRAM minus a reserve for other owners."""
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    allowance = max(0, (initial_free_mib - safety_mib) * 1024 ** 2)
    if ceiling_mib is not None:
        allowance = min(allowance, ceiling_mib * 1024 ** 2)
    if allowance <= 0:
        raise RuntimeError("No safe writer allocation remains after the shared-GPU reserve")
    torch.cuda.set_per_process_memory_fraction(min(1., allowance / total), device=0)
    return {"allocation_limit_bytes": allowance, "shared_gpu_reserve_mib": safety_mib,
            "initial_free_mib": initial_free_mib, "allocation_ceiling_mib": ceiling_mib}


def gpu_preflight(model, tokenizer, data, args, initial_free_mib):
    """Check real longest inputs and targets with zero optimizer updates."""
    import torch
    longest_input = max(data["train"], key=lambda row: len(row["input_ids"]))
    longest_target = max(data["train"], key=lambda row: row["target_tokens"])
    rows = {r["id"]: r for r in (longest_input, longest_target)}
    model.train(); torch.cuda.reset_peak_memory_stats(); checks = []
    for row in rows.values():
        torch.cuda.synchronize()
        started = time.monotonic()
        model.zero_grad(set_to_none=True)
        loss, count = writer_loss(model, collate([row], tokenizer.pad_token_id, "cuda"), args.loss_chunk_size)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite preflight loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise FloatingPointError("Missing or nonfinite writer gradients")
        torch.cuda.synchronize()
        checks.append({"id": row["id"], "complete_tokens": len(row["input_ids"]),
                       "assistant_tokens": count, "loss": loss.detach().item(),
                       "finite_gradients": True,
                       "forward_backward_seconds": time.monotonic() - started})
    model.zero_grad(set_to_none=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reserved = torch.cuda.max_memory_reserved()
    safety_mib = getattr(args, "safety_mib", 8192)
    required_mib = (reserved + 8 * trainable + safety_mib * 1024**2 + 1024**2 - 1) // 1024**2
    report = {"created_at_beijing": timestamp(), "real_batch_checks": checks,
              "optimizer_updates": 0, "peak_reserved_bytes": reserved,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "trainable_parameters": trainable, "estimated_adam_state_bytes": 8 * trainable,
              "initial_free_mib": initial_free_mib, "estimated_required_initial_free_mib": required_mib,
              "model": {"repo_id": "Qwen/Qwen3.5-9B", "revision": REVISION},
              "rank": args.rank, "qlora": args.qlora, "max_tokens": args.max_tokens,
              "loss_chunk_size": args.loss_chunk_size,
              "train_sha256": sha(Path(args.train_data).read_bytes()),
              "gpu_preflight_code_sha256": sha(Path(__file__).read_bytes()),
              "safe_for_training_start": initial_free_mib >= required_mib,
              "safety_margin_mib": safety_mib, "training_started": False,
              "quality_gain_established": False}
    save_json(Path(args.output) / "gpu_preflight.json", report)
    if not report["safe_for_training_start"]:
        raise RuntimeError("Measured writer demand plus optimizer states/safety margin exceeds available memory")
    return report


def length_bin(length):
    """Use power-of-two length buckets without modifying any observation."""
    return max(512, 1 << (length - 1).bit_length())


def profile_batch_capacity(model, tokenizer, rows, args):
    """Measure real forward/backward throughput including an Adam-state reserve."""
    import gc
    import torch
    bins = defaultdict(list)
    for row in rows:
        bins[length_bin(len(row["input_ids"]))].append(row)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer_reserve = torch.empty(8 * trainable, dtype=torch.uint8, device="cuda")
    model.train()
    capacities, measurements = {}, []
    for width, members in sorted(bins.items(), reverse=True):
        members.sort(key=lambda row: len(row["input_ids"]), reverse=True)
        maximum = min(args.batch_size, args.effective_batch_size, len(members))
        candidate = min(maximum, max(1, args.batch_token_hint // width))
        successful = []
        while True:
            model.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            record = {"length_bin": width, "batch_size": candidate,
                      "sample_ids": [row["id"] for row in members[:candidate]]}
            try:
                batch = collate(members[:candidate], tokenizer.pad_token_id, "cuda")
                loss, count = writer_loss(model, batch, args.loss_chunk_size)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite batch-capacity loss")
                loss.backward(); torch.cuda.synchronize()
                if not all(torch.isfinite(p.grad).all().item() for p in model.parameters()
                           if p.requires_grad and p.grad is not None):
                    raise FloatingPointError("Nonfinite batch-capacity gradient")
                elapsed = time.monotonic() - began
                record.update(status="passed", seconds=elapsed,
                              complete_tokens=sum(len(row["input_ids"]) for row in members[:candidate]),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                              supervised_tokens=count,
                              examples_per_second=candidate / elapsed)
                successful.append(record)
                del loss, batch
            except torch.OutOfMemoryError:
                record.update(status="oom_within_own_allocation_limit")
                # Clear failed graph references before reducing the candidate.
                if "loss" in locals(): del loss
                if "batch" in locals(): del batch
            finally:
                model.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache()
            measurements.append(record)
            save_json(Path(args.output) / "batch_capacity.json", {
                "created_at_beijing": timestamp(), "measurements": measurements,
                "capacities": capacities, "optimizer_updates": 0,
                "optimizer_state_reserve_bytes": optimizer_reserve.numel(),
                "selection": "Fastest measured stable real batch within each length bucket",
                "data_repetition_creates_training_samples": False})
            print(json.dumps({"batch_capacity": record}), flush=True)
            if record["status"] != "passed":
                if successful:
                    break
                if candidate == 1:
                    raise RuntimeError("A complete writer sample cannot fit safely")
                candidate = max(1, candidate // 2)
                continue
            if candidate == maximum:
                break
            candidate = min(maximum, candidate * 2)
        capacities[width] = max(successful, key=lambda item: item["examples_per_second"])["batch_size"]
    del optimizer_reserve
    model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    save_json(Path(args.output) / "batch_capacity.json", {
        "created_at_beijing": timestamp(), "measurements": measurements,
        "capacities": capacities, "optimizer_updates": 0,
        "optimizer_state_reserve_bytes": 8 * trainable,
        "selection": "Fastest measured stable real batch within each length bucket",
        "data_repetition_creates_training_samples": False})
    return capacities


def epoch_order(rows, effective_batch_size):
    """Shuffle local length-sorted pools while visiting every eligible row once."""
    order = list(range(len(rows)))
    random.shuffle(order)
    groups = []
    pool_size = effective_batch_size * 8
    for offset in range(0, len(order), pool_size):
        pool = sorted(order[offset:offset + pool_size], key=lambda i: len(rows[i]["input_ids"]))
        groups.extend(pool[start:start + effective_batch_size]
                      for start in range(0, len(pool), effective_batch_size))
    random.shuffle(groups)
    return groups


def microbatches(selected, rows, capacities):
    """Keep whole observations and apply the measured capacity of their bucket."""
    pending, width = [], None
    for index in selected:
        row_width = length_bin(len(rows[index]["input_ids"]))
        if pending and (width != row_width or len(pending) >= capacities[width]):
            yield pending
            pending = []
        width = row_width
        pending.append(index)
    if pending:
        yield pending


def measured_memory_requirement(args, report):
    """Reuse a prior real profile only for its bound data and configuration."""
    path = Path(args.output) / "gpu_preflight.json"
    if not path.is_file():
        return args.memory_floor_mib
    previous = json.loads(path.read_text())
    companion = Path(args.output) / "gpu_profile_context.json"
    context = json.loads(companion.read_text()) if companion.is_file() else {}
    if context and context.get("gpu_preflight_sha256") != sha(path.read_bytes()):
        raise ValueError("Existing writer GPU profile changed after context binding")
    values = {**context, **previous}
    expected = {"model": report["model"], "train_sha256": report["train_sha256"],
                "rank": args.rank, "qlora": args.qlora, "max_tokens": args.max_tokens,
                "loss_chunk_size": args.loss_chunk_size}
    if any(values.get(key) != value for key, value in expected.items()):
        return args.memory_floor_mib
    return max(args.memory_floor_mib, previous["estimated_required_initial_free_mib"])


def validate_release(path, report):
    if path is None:
        raise ValueError("Training requires the explicit reviewed writer release manifest")
    release = json.loads(Path(path).read_text())
    if not release.get("training_authorized") or not release.get("required_release_checks_passed"):
        raise ValueError("Writer release remains pending")
    for name in ("train_sha256", "dev_sha256", "template_sha256", "input_projection_sha256"):
        if release.get(name) != report[name]:
            raise ValueError("Writer release differs from the frozen data/template")
    if release.get("max_tokens") != report["max_tokens"]:
        raise ValueError("Writer release differs from the declared complete-input cohort")


def evaluate_writer(model, tokenizer, rows, chunk_size, capacities=None):
    import torch
    model.eval(); task_losses = defaultdict(lambda: [0., 0])
    batches = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], length_bin(len(row["input_ids"])))].append(row)
    for (task, width), members in grouped.items():
        capacity = (capacities or {}).get(width, 1)
        batches.extend((task, members[start:start + capacity])
                       for start in range(0, len(members), capacity))
    with torch.no_grad():
        for task, members in batches:
            loss, count = writer_loss(model, collate(members, tokenizer.pad_token_id, "cuda"), chunk_size)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite writer DEV loss")
            task_losses[task][0] += loss.item() * count
            task_losses[task][1] += count
    means = {task: total / count for task, (total, count) in task_losses.items()}
    return sum(means.values()) / len(means), means


def train(args, tokenizer, data, report):
    """Use complete TRAIN epochs, actual DEV selection, and resumable epoch states."""
    import torch
    validate_release(args.release, report)
    if args.minimum_epochs < 4 or args.patience < 8 or args.max_epochs < args.minimum_epochs:
        raise ValueError("Use adequate completed-epoch development and patience")
    if report["excluded_complete_rows"]["dev"]:
        raise ValueError("The predeclared DEV cohort must be completely supported")
    free = ensure_memory(measured_memory_requirement(args, report))
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    allocation_policy = limit_writer_memory(free, args.safety_mib, args.memory_ceiling_mib)
    save_json(Path(args.output) / "memory_allocation_policy.json", allocation_policy)
    model = load_writer(args.model, tokenizer, args.rank, args.qlora)
    gpu_preflight(model, tokenizer, data, args, free)
    capacities = (profile_batch_capacity(model, tokenizer, data["train"], args)
                  if args.auto_batch else {length_bin(len(row["input_ids"])): args.batch_size
                                          for row in data["train"]})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3, min_lr=1e-7)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    protocol = {**report, "training_started": True, "rank": args.rank, "qlora": args.qlora,
        "seed": args.seed, "lr": args.lr,
        "effective_batch_size_records": args.effective_batch_size,
        "maximum_microbatch_size": args.batch_size, "auto_batch": args.auto_batch,
        "length_bucket_capacities": capacities, "shared_gpu_reserve_mib": args.safety_mib,
        "lora_targets": list(LORA_TARGETS), "minimum_epochs": args.minimum_epochs,
        "patience_complete_epochs": args.patience, "maximum_epochs_budget": args.max_epochs,
        "loss_chunk_size": args.loss_chunk_size, "optimizer": "fused AdamW",
        "train_objective": "Assistant-token causal cross-entropy",
        "dev_selection": "Macro average of per-task assistant-token cross-entropy",
        "test_access": False, "training_code_sha256": sha(Path(__file__).read_bytes())}
    frozen = output / "training_protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        # Ignore creation timestamps while requiring the substantive protocol.
        previous = json.loads(frozen.read_text())
        for key in ("created_at_beijing", "seconds"):
            previous.pop(key, None); protocol.pop(key, None)
        if previous != protocol:
            raise ValueError("Existing writer run has a different protocol")
    else:
        save_json(frozen, protocol)
    history, start_epoch, best, bad, updates = [], 0, float("inf"), 0, 0
    resume = output / "latest.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if set(parameters) != set(state["adapter"]):
            raise ValueError("Resume parameter set differs")
        with torch.no_grad():
            for name, parameter in parameters.items(): parameter.copy_(state["adapter"][name].to(parameter.device))
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"]); torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        history, start_epoch, best, bad, updates = state["history"], state["epoch"], state["best_dev"], state["bad"], state["updates"]
    else:
        best, means = evaluate_writer(model, tokenizer, data["dev"], args.loss_chunk_size, capacities)
        history.append({"epoch": 0, "updates": 0, "train_rows_visited": 0,
                        "dev_loss": best, "dev_by_task": means,
                        "stage": "PRETRAINED_BASELINE_WITH_ZERO_LORA_UPDATE"})
        model.save_pretrained(output / "best_adapter")
        tokenizer.save_pretrained(output / "best_adapter")
        model.generation_config.save_pretrained(output / "best_adapter")
        save_json(output / "history.json", history)
    converged = (start_epoch >= args.minimum_epochs and bad >= args.patience
                 and optimizer.param_groups[0]["lr"] <= 1e-7 * 1.000001)
    for epoch in ([] if converged else range(start_epoch + 1, args.max_epochs + 1)):
        began = time.monotonic(); model.train()
        groups = epoch_order(data["train"], args.effective_batch_size)
        visited = set(); train_sum = 0.; target_total = 0
        for group_index, selected in enumerate(groups):
            denominator = sum(data["train"][i]["target_tokens"] for i in selected)
            saved_python_rng = random.getstate()
            saved_torch_rng = torch.get_rng_state()
            saved_cuda_rng = torch.cuda.get_rng_state_all()
            while True:
                optimizer.zero_grad(set_to_none=True)
                group_sum, group_tokens = 0., 0
                current = None
                try:
                    for current in microbatches(selected, data["train"], capacities):
                        rows = [data["train"][i] for i in current]
                        batch = collate(rows, tokenizer.pad_token_id, "cuda")
                        loss, count = writer_loss(model, batch, args.loss_chunk_size)
                        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite writer loss")
                        (loss * (count / denominator)).backward()
                        group_sum += loss.detach().item() * count; group_tokens += count
                        del loss, batch
                    break
                except torch.OutOfMemoryError:
                    if "loss" in locals(): del loss
                    if "batch" in locals(): del batch
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if current is None or len(current) == 1:
                        raise RuntimeError("A complete sample no longer fits the writer allocation")
                    width = length_bin(max(len(data["train"][i]["input_ids"]) for i in current))
                    capacities[width] = max(1, len(current) // 2)
                    random.setstate(saved_python_rng); torch.set_rng_state(saved_torch_rng)
                    torch.cuda.set_rng_state_all(saved_cuda_rng)
                    save_json(output / "runtime_batch_capacity.json", {
                        "epoch": epoch, "update": updates, "capacities": capacities,
                        "complete_group_retried_without_optimizer_update": True,
                        "other_gpu_processes_managed": False})
            if group_tokens != denominator:
                raise RuntimeError("Microbatches did not cover the complete optimizer group")
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            if not torch.isfinite(norm): raise FloatingPointError("Nonfinite writer gradient")
            optimizer.step(); updates += 1
            train_sum += group_sum; target_total += group_tokens; visited.update(selected)
            if group_index == 0 or updates % 10 == 0:
                progress = {"created_at_beijing": timestamp(), "epoch_in_progress": epoch,
                    "completed_epochs": epoch - 1, "optimizer_updates": updates,
                    "epoch_rows_visited": len(visited), "epoch_rows_eligible": len(data["train"]),
                    "seconds_in_epoch": time.monotonic() - began, "training_started": True,
                    "epoch_complete": False, "test_accessed": False}
                save_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        if len(visited) != len(data["train"]): raise RuntimeError("Incomplete TRAIN epoch cannot trigger selection")
        dev_loss, means = evaluate_writer(model, tokenizer, data["dev"], args.loss_chunk_size, capacities)
        improved = dev_loss < best - 1e-6
        best, bad = (dev_loss, 0) if improved else (best, bad + 1)
        scheduler.step(dev_loss)
        result = {"epoch": epoch, "train_rows_visited": len(visited), "train_rows_eligible": len(data["train"]),
            "updates": updates, "train_loss": train_sum / target_total, "dev_loss": dev_loss,
            "dev_by_task": means, "seconds": time.monotonic() - began,
            "learning_rate": optimizer.param_groups[0]["lr"], "best_dev": best, "bad_epochs": bad}
        history.append(result)
        state = {"epoch": epoch, "adapter": {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
            "best_dev": best, "bad": bad, "updates": updates, "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        temporary = resume.with_suffix(".tmp"); torch.save(state, temporary); temporary.replace(resume)
        if improved:
            model.save_pretrained(output / "best_adapter"); tokenizer.save_pretrained(output / "best_adapter")
            model.generation_config.save_pretrained(output / "best_adapter")
        save_json(output / "history.json", history)
        print(json.dumps(result), flush=True)
        converged = epoch >= args.minimum_epochs and bad >= args.patience
        if converged: break
    save_json(output / "completion.json", {"completed_epochs": history[-1]["epoch"],
        "reason": "DEV_PATIENCE_COMPLETE_EPOCHS" if converged else "EPOCH_BUDGET",
        "best_dev": best, "quality_gain_established": False, "test_accessed": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, default=DATA / "quality_checked_inventory_v1/accepted_writer_train.jsonl")
    parser.add_argument("--dev-data", type=Path, default=DATA / "dev.jsonl")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--protection", type=Path, default=DATA / "quality_checked_inventory_v1/hard_check_report.json")
    parser.add_argument("--output", type=Path, default=DATA / "qwen3_5_writer_preflight_v1")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--gpu-preflight", action="store_true")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--memory-floor-mib", type=int, default=32000)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--auto-batch", action="store_true")
    parser.add_argument("--batch-token-hint", type=int, default=16384)
    parser.add_argument("--safety-mib", type=int, default=8192)
    parser.add_argument("--memory-ceiling-mib", type=int)
    parser.add_argument("--loss-chunk-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--minimum-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=40)
    args = parser.parse_args()
    if args.effective_batch_size is None:
        args.effective_batch_size = args.accumulate
    if min(args.max_tokens, args.rank, args.accumulate, args.loss_chunk_size, args.memory_floor_mib,
           args.batch_size, args.effective_batch_size, args.batch_token_hint, args.safety_mib) <= 0:
        raise ValueError("Invalid writer configuration")
    tokenizer, data, report = prepare(args.train_data, args.dev_data, args.model, args.max_tokens, args.protection)
    save_json(args.output / "token_preflight.json", report)
    print(json.dumps({"preflight": str(args.output / "token_preflight.json"),
        "usable_rows": {s: len(rows) for s, rows in data.items()},
        "excluded_rows": {s: len(rows) for s, rows in report["excluded_complete_rows"].items()},
        "seconds": report["seconds"], "gpu_loaded": False}), flush=True)
    if args.train and args.gpu_preflight:
        raise ValueError("Choose either the zero-update preflight or formal training")
    if args.train:
        train(args, tokenizer, data, report)
    elif args.gpu_preflight:
        phase = "resource_admission"
        try:
            free = ensure_memory(measured_memory_requirement(args, report))
            allocation_policy = limit_writer_memory(free, args.safety_mib, args.memory_ceiling_mib)
            save_json(Path(args.output) / "memory_allocation_policy.json", allocation_policy)
            phase = "text_only_backbone_and_adapter_loading"
            model = load_writer(args.model, tokenizer, args.rank, args.qlora)
            phase = "complete_input_forward_backward"
            gpu_preflight(model, tokenizer, data, args, free)
        except Exception as exc:
            save_json(Path(args.output) / "gpu_preflight_failure.json", {
                "created_at_beijing": timestamp(), "phase": phase,
                "error_type": type(exc).__name__, "message": str(exc),
                "optimizer_updates": 0, "training_started": False,
                "quality_gain_established": False})
            raise


if __name__ == "__main__":
    main()
