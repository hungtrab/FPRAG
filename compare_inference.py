"""
compare_inference.py

So sánh tốc độ inference giữa:
  1. FP16 baseline (model gốc)
  2. Custom AWQ FP16 dequantized (output của awq_stand_xl / awq_js_xl)
  3. vLLM AWQ INT4 (output của convert_custom_awq_to_vllm.py)

Metrics:
  - Throughput: tokens/sec
  - Latency: time-to-first-token, avg token latency
  - Memory: peak GPU VRAM (MB)

Usage:
  # So sánh tất cả
  python compare_inference.py \
      --fp16-path   ./models/Mistral-7B-v0.3 \
      --fp16dq-path ./quantized_models/model_awq_js_xl \
      --vllm-path   ./quantized_models/model_awq_js_xl_vllm

  # Chỉ so sánh FP16 vs vLLM
  python compare_inference.py \
      --fp16-path  ./models/Mistral-7B-v0.3 \
      --vllm-path  ./quantized_models/model_awq_js_xl_vllm

  # Chỉ test vLLM
  python compare_inference.py --vllm-path ./quantized_models/model_awq_js_xl_vllm
"""

import argparse
import json
import time
import gc
import torch
from pathlib import Path
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Prompts dùng cho benchmark ────────────────────────────────────────────────

SHORT_PROMPTS = [
    "The capital of France is",
    "Quantum computing uses",
    "The best way to learn programming is",
    "Machine learning is a subset of",
    "The theory of relativity states that",
]

LONG_PROMPTS = [
    "Explain the differences between supervised, unsupervised, and reinforcement learning in machine learning. Include examples for each type and discuss when you would use one approach over another in real-world applications.",
    "Describe the transformer architecture used in modern language models. Explain how attention mechanisms work, why positional encoding is needed, and how the encoder-decoder structure processes sequences.",
    "What are the main challenges in quantizing large language models? Discuss the tradeoffs between model size, inference speed, and accuracy when applying 4-bit quantization techniques.",
]


# ── HuggingFace inference benchmark ──────────────────────────────────────────

def benchmark_hf(model_path: str, label: str, prompts: list, n_tokens: int,
                 n_warmup: int, quantization=None):
    """
    Benchmark HuggingFace model (FP16 hoặc FP16-dequantized).
    Returns dict với throughput, latency, memory stats.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {label}")
    print(f"  Path: {model_path}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    load_kwargs = dict(
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()

    # Measure VRAM after load
    torch.cuda.synchronize()
    vram_load_mb = torch.cuda.memory_allocated() / 1024**2

    results = []

    # Warmup
    print(f"  Warmup ({n_warmup} runs)...")
    for i in range(n_warmup):
        prompt = prompts[i % len(prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=32, do_sample=False)
    torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmark ({len(prompts)} prompts × {n_tokens} tokens)...")
    vram_peak_mb = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
                use_cache=True,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        n_new = output.shape[1] - input_len
        peak = torch.cuda.max_memory_allocated() / 1024**2
        vram_peak_mb = max(vram_peak_mb, peak)

        results.append({
            "elapsed": elapsed,
            "n_tokens": n_new,
            "tokens_per_sec": n_new / elapsed,
            "latency_ms_per_token": elapsed / n_new * 1000,
        })

    # Summary
    tps_list = [r["tokens_per_sec"] for r in results]
    lat_list = [r["latency_ms_per_token"] for r in results]

    summary = {
        "label": label,
        "vram_load_mb": vram_load_mb,
        "vram_peak_mb": vram_peak_mb,
        "throughput_mean": np.mean(tps_list),
        "throughput_std":  np.std(tps_list),
        "throughput_min":  np.min(tps_list),
        "throughput_max":  np.max(tps_list),
        "latency_mean_ms": np.mean(lat_list),
        "latency_std_ms":  np.std(lat_list),
    }

    print(f"  VRAM (load): {vram_load_mb:.0f} MB")
    print(f"  VRAM (peak): {vram_peak_mb:.0f} MB")
    print(f"  Throughput : {summary['throughput_mean']:.1f} ± {summary['throughput_std']:.1f} tok/s")
    print(f"  Latency    : {summary['latency_mean_ms']:.2f} ± {summary['latency_std_ms']:.2f} ms/tok")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return summary


# ── vLLM inference benchmark ─────────────────────────────────────────────────

def benchmark_vllm(model_path: str, label: str, prompts: list, n_tokens: int,
                   n_warmup: int, quantization: str = "awq",
                   gpu_memory_utilization: float = 0.8):
    """
    Benchmark vLLM model (AWQ INT4).
    Requires: pip install vllm
    """
    print(f"\n{'='*60}")
    print(f"  Benchmarking (vLLM): {label}")
    print(f"  Path: {model_path}")
    print(f"  Quantization: {quantization}")
    print(f"{'='*60}")

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("  ❌ vLLM not installed. Run: pip install vllm")
        return None

    # vLLM 0.17+ validates that config.json quant_method matches quantization arg.
    # For awq_marlin, temporarily patch config.json then restore after load.
    config_path = Path(model_path) / "config.json"
    config_backup = None
    if quantization == "awq_marlin" and config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        config_backup = json.dumps(config_data, indent=2)
        if "quantization_config" in config_data:
            qc = config_data["quantization_config"]
            qc["quant_method"] = "awq_marlin"
            qc["version"] = "marlin"
            # awq_marlin schema requires "bits" and "group_size" (not "w_bit"/"q_group_size")
            if "bits" not in qc:
                qc["bits"] = qc.get("w_bit", 4)
            if "group_size" not in qc:
                qc["group_size"] = qc.get("q_group_size", 128)
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    try:
        llm = LLM(model=model_path, quantization=quantization, dtype="float16",
                  gpu_memory_utilization=gpu_memory_utilization)
    except Exception as e:
        if config_backup is not None:
            with open(config_path, "w") as f:
                f.write(config_backup)
        err = str(e)
        if "PTX" in err or "unsupported toolchain" in err or "cudaErrorUnsupportedPtxVersion" in err:
            print(f"  ⚠️  awq_marlin not supported on this CUDA driver (PTX version mismatch).")
            print(f"     Use --vllm-quant awq instead.")
        else:
            print(f"  ❌ vLLM load failed: {e}")
        return None
    finally:
        if config_backup is not None:
            with open(config_path, "w") as f:
                f.write(config_backup)
    sampling = SamplingParams(temperature=0, max_tokens=n_tokens)

    # Measure VRAM after load
    torch.cuda.synchronize()
    vram_load_mb = torch.cuda.memory_allocated() / 1024**2

    # Warmup
    print(f"  Warmup ({n_warmup} runs)...")
    for i in range(n_warmup):
        llm.generate([prompts[i % len(prompts)]], SamplingParams(temperature=0, max_tokens=32))

    # Benchmark
    print(f"  Benchmark ({len(prompts)} prompts × {n_tokens} tokens)...")
    results = []
    vram_peak_mb = 0

    for prompt in prompts:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = llm.generate([prompt], sampling)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        n_new = len(outputs[0].outputs[0].token_ids)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        vram_peak_mb = max(vram_peak_mb, peak)

        results.append({
            "elapsed": elapsed,
            "n_tokens": n_new,
            "tokens_per_sec": n_new / elapsed,
            "latency_ms_per_token": elapsed / n_new * 1000,
        })

    tps_list = [r["tokens_per_sec"] for r in results]
    lat_list = [r["latency_ms_per_token"] for r in results]

    summary = {
        "label": label,
        "vram_load_mb": vram_load_mb,
        "vram_peak_mb": vram_peak_mb,
        "throughput_mean": np.mean(tps_list),
        "throughput_std":  np.std(tps_list),
        "throughput_min":  np.min(tps_list),
        "throughput_max":  np.max(tps_list),
        "latency_mean_ms": np.mean(lat_list),
        "latency_std_ms":  np.std(lat_list),
    }

    print(f"  VRAM (load): {vram_load_mb:.0f} MB")
    print(f"  VRAM (peak): {vram_peak_mb:.0f} MB")
    print(f"  Throughput : {summary['throughput_mean']:.1f} ± {summary['throughput_std']:.1f} tok/s")
    print(f"  Latency    : {summary['latency_mean_ms']:.2f} ± {summary['latency_std_ms']:.2f} ms/tok")

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    return summary


# ── Results table ─────────────────────────────────────────────────────────────

def print_table(summaries: list, baseline_label: str = None):
    valid = [s for s in summaries if s is not None]
    if not valid:
        print("No results to display.")
        return

    print("\n" + "=" * 90)
    print("INFERENCE SPEED COMPARISON")
    print("=" * 90)
    print(f"{'Model':<30} {'VRAM Load':>10} {'VRAM Peak':>10} {'Throughput':>14} {'Latency':>14} {'Speedup':>8}")
    print(f"{'':30} {'(MB)':>10} {'(MB)':>10} {'(tok/s)':>14} {'(ms/tok)':>14} {'vs base':>8}")
    print("-" * 90)

    # Find baseline throughput
    baseline_tps = None
    if baseline_label:
        for s in valid:
            if s["label"] == baseline_label:
                baseline_tps = s["throughput_mean"]
                break
    if baseline_tps is None and valid:
        baseline_tps = valid[0]["throughput_mean"]

    for s in valid:
        speedup = s["throughput_mean"] / baseline_tps if baseline_tps else 1.0
        speedup_str = f"{speedup:.2f}x"
        tps_str = f"{s['throughput_mean']:.1f} ±{s['throughput_std']:.1f}"
        lat_str = f"{s['latency_mean_ms']:.2f} ±{s['latency_std_ms']:.2f}"
        print(f"  {s['label']:<28} {s['vram_load_mb']:>10.0f} {s['vram_peak_mb']:>10.0f} "
              f"{tps_str:>14} {lat_str:>14} {speedup_str:>8}")

    print("=" * 90)

    # VRAM savings
    if len(valid) >= 2:
        print("\nVRAM Savings (load):")
        base_vram = valid[0]["vram_load_mb"]
        for s in valid[1:]:
            saving_pct = (base_vram - s["vram_load_mb"]) / base_vram * 100
            print(f"  {s['label']} vs {valid[0]['label']}: {saving_pct:+.1f}% "
                  f"({base_vram:.0f} → {s['vram_load_mb']:.0f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare inference speed: FP16 vs AWQ-dequant vs vLLM-AWQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fp16-path",   type=str, default="", help="FP16 baseline model path")
    parser.add_argument("--fp16dq-path", type=str, default="", help="FP16 dequantized AWQ model path (awq_js_xl output)")
    parser.add_argument("--vllm-path",   type=str, default="", help="vLLM AWQ INT4 model path (convert_custom_awq_to_vllm output)")
    parser.add_argument("--vllm-quant",  type=str, default="awq", choices=["awq", "awq_marlin"],
                        help="vLLM quantization backend")
    parser.add_argument("--n-tokens",    type=int, default=128, help="Tokens to generate per prompt")
    parser.add_argument("--n-warmup",    type=int, default=2,   help="Warmup runs before timing")
    parser.add_argument("--use-long",    action="store_true",   help="Use long prompts instead of short")
    parser.add_argument("--gpu-mem-util", type=float, default=0.8, help="vLLM gpu_memory_utilization (default: 0.8)")
    args = parser.parse_args()

    if not any([args.fp16_path, args.fp16dq_path, args.vllm_path]):
        parser.error("Provide at least one of --fp16-path, --fp16dq-path, --vllm-path")

    prompts = LONG_PROMPTS if args.use_long else SHORT_PROMPTS

    print("=" * 90)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 90)
    print(f"Prompts    : {len(prompts)} ({'long' if args.use_long else 'short'})")
    print(f"New tokens : {args.n_tokens}")
    print(f"Warmup     : {args.n_warmup}")
    print("=" * 90)

    summaries = []
    baseline_label = None

    if args.fp16_path:
        s = benchmark_hf(args.fp16_path, "FP16 Baseline", prompts, args.n_tokens, args.n_warmup)
        summaries.append(s)
        baseline_label = "FP16 Baseline"

    if args.fp16dq_path:
        s = benchmark_hf(args.fp16dq_path, "AWQ FP16-dequant (HF)", prompts, args.n_tokens, args.n_warmup)
        summaries.append(s)
        if baseline_label is None:
            baseline_label = "AWQ FP16-dequant (HF)"

    if args.vllm_path:
        s = benchmark_vllm(args.vllm_path, f"vLLM AWQ INT4 ({args.vllm_quant})",
                           prompts, args.n_tokens, args.n_warmup, args.vllm_quant,
                           gpu_memory_utilization=args.gpu_mem_util)
        summaries.append(s)
        if baseline_label is None:
            baseline_label = f"vLLM AWQ INT4 ({args.vllm_quant})"

    print_table(summaries, baseline_label)


if __name__ == "__main__":
    main()
