#!/usr/bin/env python
"""Evaluate RTN+BC artifacts without losing dynamically-added Linear biases.

Some HF architectures (Qwen2, Mistral) construct many Linear modules with
``bias=False`` and do not expose config switches to recreate those biases on
``from_pretrained``.  A normal reload therefore ignores saved ``*.bias`` tensors
for modules such as ``o_proj`` and MLP projections.  This script loads the model
normally, then reads all saved safetensors bias tensors from the BC artifact and
attaches them back to the live ``nn.Linear`` modules before running PPL/lm-eval.
That matches the smart-flip evaluation path where +BC is evaluated in-process.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import safe_open
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from compare_slicing import AWQSlidingWindowValidator


PIQA_URLS = [
    "https://huggingface.co/datasets/lighteval/piqa/resolve/main/plain_text/validation-00000-of-00001.parquet",
    "https://huggingface.co/datasets/regisss/piqa/resolve/main/data/validation-00000-of-00001.parquet",
]


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def load_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(path: str, device: str):
    dtype = torch.float16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    device_map = None
    if device.startswith("cuda") and torch.cuda.is_available():
        device_map = device if device == "cuda" else {"": device}
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    return model


def get_submodule(root: nn.Module, module_name: str) -> nn.Module | None:
    current: Any = root
    for part in module_name.split("."):
        if part.isdigit() and isinstance(current, (nn.ModuleList, list, tuple)):
            current = current[int(part)]
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current if isinstance(current, nn.Module) else None


def safetensor_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text())
        return [model_dir / name for name in sorted(set(payload.get("weight_map", {}).values()))]

    files = sorted(model_dir.glob("*.safetensors"))
    single = model_dir / "model.safetensors"
    if single.exists() and single not in files:
        files.insert(0, single)
    return files


@torch.no_grad()
def apply_saved_linear_biases(model: nn.Module, model_dir: str) -> dict[str, int]:
    """Attach every saved Linear ``*.bias`` tensor to the live model object."""
    root = Path(model_dir)
    stats = {"seen": 0, "applied": 0, "missing_module": 0, "not_linear": 0, "shape_mismatch": 0}

    for shard_path in safetensor_files(root):
        if not shard_path.exists():
            continue
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.endswith(".bias"):
                    continue
                stats["seen"] += 1
                module_name = key[:-5]
                module = get_submodule(model, module_name)
                if module is None:
                    stats["missing_module"] += 1
                    continue
                if not isinstance(module, nn.Linear):
                    stats["not_linear"] += 1
                    continue

                bias = handle.get_tensor(key)
                if bias.numel() != module.weight.shape[0]:
                    stats["shape_mismatch"] += 1
                    continue

                bias = bias.to(device=module.weight.device, dtype=module.weight.dtype)
                if module.bias is None:
                    module.bias = nn.Parameter(bias)
                else:
                    module.bias.data.copy_(bias)
                stats["applied"] += 1

    print(
        "Applied saved Linear biases: "
        f"{stats['applied']}/{stats['seen']} "
        f"(missing={stats['missing_module']}, not_linear={stats['not_linear']}, "
        f"shape_mismatch={stats['shape_mismatch']})"
    )
    return stats


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_ppl_model(
    path: str,
    tokenizer_path: str,
    texts: list[str],
    dataset: str,
    device: str,
    cache_dir: str,
    apply_bc: bool,
):
    print(f"\n  Evaluating {'RTN+BC' if apply_bc else 'RTN'} on {dataset}...")
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_causal_lm(path, device)
    if apply_bc:
        apply_saved_linear_biases(model, path)
    validator = AWQSlidingWindowValidator(device=device, seed=42, stride=512, max_length=2048, cache_dir=cache_dir)
    result = validator.evaluate_sliding_window(model, tokenizer, texts)
    if result:
        print(f"  Perplexity: {result['perplexity']:.4f}")
    cleanup_model(model)
    return result


def run_ppl(args):
    validator = AWQSlidingWindowValidator(
        device=args.device,
        seed=42,
        stride=args.stride,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
    )
    datasets = {
        "WikiText-2": validator.load_wikitext2_test(args.ppl_samples),
        "C4": validator.load_c4_validation(args.ppl_samples),
    }

    summary = []
    for dataset_name, texts in datasets.items():
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)
        bc = evaluate_ppl_model(args.bc_path, args.tokenizer_path, texts, dataset_name, args.device, args.cache_dir, True)
        raw = evaluate_ppl_model(args.raw_path, args.tokenizer_path, texts, dataset_name, args.device, args.cache_dir, False)
        if not bc or not raw:
            continue
        bc_ppl = bc["perplexity"]
        raw_ppl = raw["perplexity"]
        delta_pct = ((bc_ppl - raw_ppl) / raw_ppl) * 100
        winner = "Heuristic" if delta_pct < -0.05 else ("Standard" if delta_pct > 0.05 else "Tie")
        summary.append((dataset_name, bc_ppl, raw_ppl, delta_pct, winner))

    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS")
    print("=" * 80)
    print(f"\n{'Dataset':<15} {'RTN+BC':<15} {'RTN':<15} {'Delta':<12} {'Winner':<10}")
    print("-" * 80)
    for dataset_name, bc_ppl, raw_ppl, delta_pct, winner in summary:
        print(f"{dataset_name:<15} {bc_ppl:<15.4f} {raw_ppl:<15.4f} {delta_pct:>+11.3f}%  {winner:<10}")


def download_piqa(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "piqa_validation.parquet"
    if target.exists() and target.stat().st_size > 0:
        return target

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN")
    last_error = None
    for url in PIQA_URLS:
        tmp = target.with_suffix(".tmp")
        try:
            request = urllib.request.Request(url)
            if token:
                request.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(request) as response, open(tmp, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            tmp.replace(target)
            return target
        except Exception as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"Could not download PIQA validation parquet: {last_error}")


@torch.no_grad()
def continuation_score(model, tokenizer, prompt: str, continuation: str, device: str):
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids[0]
    cont_ids = tokenizer(" " + continuation, add_special_tokens=False, return_tensors="pt").input_ids[0]
    input_ids = torch.cat([prompt_ids, cont_ids], dim=0).unsqueeze(0).to(device)
    cont_len = cont_ids.numel()
    if cont_len == 0:
        return float("-inf"), float("-inf")

    logits = model(input_ids).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    cont_log_probs = token_log_probs[:, -cont_len:]
    total = cont_log_probs.sum().item()
    return total, total / cont_len


def run_manual_piqa(model, tokenizer, output_json: Path, cache_dir: Path, device: str):
    import pyarrow.parquet as pq

    table = pq.read_table(download_piqa(cache_dir))
    rows = table.to_pylist()
    correct = 0
    correct_norm = 0
    total = 0
    for row in tqdm(rows, desc="Manual PIQA"):
        goal = row.get("goal")
        sol1 = row.get("sol1")
        sol2 = row.get("sol2")
        label = int(row.get("label"))
        prompt = f"Question: {goal}\nAnswer:"
        s1, s1n = continuation_score(model, tokenizer, prompt, sol1, device)
        s2, s2n = continuation_score(model, tokenizer, prompt, sol2, device)
        correct += int((0 if s1 > s2 else 1) == label)
        correct_norm += int((0 if s1n > s2n else 1) == label)
        total += 1

    payload = {
        "results": {
            "piqa": {
                "acc,none": correct / total,
                "acc_norm,none": correct_norm / total,
            }
        },
        "versions": {"piqa": "manual-parquet"},
        "n-shot": {"piqa": 0},
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["results"]["piqa"], indent=2))


def run_lm_eval(args):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    output_dir = Path(args.eval_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    tokenizer = load_tokenizer(args.tokenizer_path)
    model = load_causal_lm(args.bc_path, args.device)
    apply_saved_linear_biases(model, args.bc_path)

    eval_batch_size = 1 if args.batch_size == "auto" else args.batch_size
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size)

    for task in tasks:
        task_output = output_dir / f"{task}_results.json"
        if task_output.exists() and not args.force:
            print(f"Skip existing lm_eval result: {task_output}")
            continue
        if task_output.is_dir():
            if not args.force:
                print(f"Skip existing lm_eval result directory: {task_output}")
                continue
            shutil.rmtree(task_output)
        if task == "piqa":
            run_manual_piqa(model, tokenizer, task_output, Path(args.cache_dir) / "piqa_manual", args.device)
            continue

        print(f"\nRunning in-process lm_eval task={task}")
        payload = evaluator.simple_evaluate(
            model=hflm,
            tasks=[task],
            device=args.device,
            batch_size=eval_batch_size,
            log_samples=False,
        )
        task_output.write_text(json.dumps(json_safe(payload), indent=2))
        print(f"Wrote {task_output}")

    cleanup_model(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-path", required=True)
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tasks", default="arc_challenge,arc_easy,boolq,piqa,rte")
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--cache-dir", default="./dataset_cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--ppl-samples", type=int, default=500)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--ppl", action="store_true")
    parser.add_argument("--lm-eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.ppl and not args.lm_eval:
        raise SystemExit("Pass --ppl and/or --lm-eval")

    if args.ppl:
        run_ppl(args)
    if args.lm_eval:
        run_lm_eval(args)


if __name__ == "__main__":
    main()
