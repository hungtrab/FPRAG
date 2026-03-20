"""
compare_inference.py

So sánh tốc độ inference giữa:
  1. FP16 baseline (model gốc, HuggingFace)
  2. Custom AWQ FP16 dequantized (output của awq_stand_xl / awq_js_xl, HuggingFace)
  3. HF AWQ INT4 (model AWQ tải từ HuggingFace, dùng vLLM)
  4. Custom AWQ INT4 (output của convert_custom_awq_to_vllm.py, dùng vLLM)

Hỗ trợ test batch size: --batch-sizes 1,16

Usage:
  # So sánh đầy đủ với 2 batch sizes
  python compare_inference.py \\
      --fp16-path    ./models/Mistral-7B-v0.3 \\
      --fp16dq-path  ./quantized_models/mistral_awq_js \\
      --hf-awq-path  ./models/Mistral-7B-v0.3-AWQ \\
      --vllm-path    ./quantized_models/mistral_awq_js_vllm \\
      --batch-sizes  1,16

  # Chỉ vLLM, batch sizes 1 và 8
  python compare_inference.py \\
      --vllm-path ./quantized_models/mistral_awq_js_vllm \\
      --batch-sizes 1,8
"""

import argparse
import json
import math
import subprocess
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_batch(prompts: list, batch_size: int) -> list:
    """Tile prompts to exactly batch_size entries."""
    return (prompts * math.ceil(batch_size / len(prompts)))[:batch_size]


def nvidia_smi_vram_mb() -> float:
    """Total GPU VRAM used across all processes via nvidia-smi (MiB)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


# ── HuggingFace inference benchmark ──────────────────────────────────────────

def benchmark_hf(model_path: str, label: str, prompts: list, n_tokens: int,
                 n_warmup: int, batch_size: int = 1):
    """
    Benchmark HuggingFace model (FP16 hoặc FP16-dequantized).
    Hỗ trợ batch_size > 1 với left-padding.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {label}  [bs={batch_size}]")
    print(f"  Path: {model_path}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding required for batched causal LM generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    torch.cuda.synchronize()
    vram_load_mb = torch.cuda.memory_allocated() / 1024**2

    # Warmup
    print(f"  Warmup ({n_warmup} runs)...")
    warmup_batch = make_batch(prompts, min(batch_size, 2))
    for _ in range(n_warmup):
        inp = tokenizer(warmup_batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()

    # Benchmark
    # bs=1: run over each prompt individually (variance across prompts)
    # bs>1: tile prompts to batch_size, run 5 rounds
    if batch_size == 1:
        bench_batches = [[p] for p in prompts]
    else:
        b = make_batch(prompts, batch_size)
        bench_batches = [b] * 5

    print(f"  Benchmark ({len(bench_batches)} runs × bs={batch_size} × {n_tokens} tokens)...")
    results = []
    vram_peak_mb = 0

    for batch in bench_batches:
        inp = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        input_len = inp["input_ids"].shape[1]

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=n_tokens,
                                 do_sample=False, use_cache=True)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        n_new_total = (out.shape[1] - input_len) * len(batch)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        vram_peak_mb = max(vram_peak_mb, peak)

        results.append({
            "elapsed": elapsed,
            "total_tokens": n_new_total,
            "throughput": n_new_total / elapsed,
            "latency_ms_per_token": elapsed / (out.shape[1] - input_len) * 1000,
        })

    tps_list = [r["throughput"] for r in results]
    lat_list = [r["latency_ms_per_token"] for r in results]

    summary = {
        "label": label,
        "batch_size": batch_size,
        "vram_load_mb": vram_load_mb,
        "vram_peak_mb": vram_peak_mb,
        "throughput_mean": np.mean(tps_list),
        "throughput_std":  np.std(tps_list),
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
                   gpu_memory_utilization: float = 0.8,
                   batch_size: int = 1):
    """
    Benchmark vLLM model.
    Hỗ trợ batch_size > 1: gửi cả batch vào llm.generate() cùng lúc.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmarking (vLLM): {label}  [bs={batch_size}]")
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
        is_ptx_error = (
            "PTX" in err or "unsupported toolchain" in err
            or "cudaErrorUnsupportedPtxVersion" in err
            or (quantization == "awq_marlin" and "Engine core initialization failed" in err)
        )
        if is_ptx_error:
            print(f"  ⚠️  awq_marlin: PTX version mismatch (CUDA driver too old).")
            print(f"     Fix: upgrade NVIDIA driver hoặc compile vLLM từ source.")
            print(f"     Xem hướng dẫn: python compare_inference.py --help-marlin")
        else:
            print(f"  ❌ vLLM load failed: {e}")
        return None
    finally:
        if config_backup is not None:
            with open(config_path, "w") as f:
                f.write(config_backup)

    sampling = SamplingParams(temperature=0, max_tokens=n_tokens)

    # vLLM engine runs in a subprocess; use nvidia-smi for real VRAM.
    vram_load_mb = nvidia_smi_vram_mb()

    # Warmup
    print(f"  Warmup ({n_warmup} runs)...")
    warmup_batch = make_batch(prompts, min(batch_size, 2))
    for _ in range(n_warmup):
        llm.generate(warmup_batch, SamplingParams(temperature=0, max_tokens=16))

    # Benchmark
    if batch_size == 1:
        bench_batches = [[p] for p in prompts]
    else:
        b = make_batch(prompts, batch_size)
        bench_batches = [b] * 5

    print(f"  Benchmark ({len(bench_batches)} runs × bs={batch_size} × {n_tokens} tokens)...")
    results = []
    vram_peak_mb = 0

    for batch in bench_batches:
        t0 = time.perf_counter()
        outputs = llm.generate(batch, sampling)
        elapsed = time.perf_counter() - t0

        n_new_total = sum(len(o.outputs[0].token_ids) for o in outputs)
        avg_per_seq = n_new_total / len(batch)
        peak = nvidia_smi_vram_mb()
        vram_peak_mb = max(vram_peak_mb, peak)

        results.append({
            "elapsed": elapsed,
            "total_tokens": n_new_total,
            "throughput": n_new_total / elapsed,
            "latency_ms_per_token": elapsed / avg_per_seq * 1000,
        })

    tps_list = [r["throughput"] for r in results]
    lat_list = [r["latency_ms_per_token"] for r in results]

    summary = {
        "label": label,
        "batch_size": batch_size,
        "vram_load_mb": vram_load_mb,
        "vram_peak_mb": vram_peak_mb,
        "throughput_mean": np.mean(tps_list),
        "throughput_std":  np.std(tps_list),
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

    # Group by batch_size for display
    batch_sizes = sorted(set(s["batch_size"] for s in valid))

    for bs in batch_sizes:
        group = [s for s in valid if s["batch_size"] == bs]

        # Find baseline tps for this group
        baseline_tps = None
        if baseline_label:
            for s in group:
                if s["label"] == baseline_label:
                    baseline_tps = s["throughput_mean"]
                    break
        if baseline_tps is None and group:
            baseline_tps = group[0]["throughput_mean"]

        W = 100
        print(f"\n{'='*W}")
        print(f"  INFERENCE SPEED COMPARISON  [Batch Size = {bs}]")
        print(f"{'='*W}")
        print(f"  {'Model':<32} {'VRAM Load':>10} {'VRAM Peak':>10} {'Throughput':>16} {'Latency':>16} {'Speedup':>8}")
        print(f"  {'':32} {'(MB)':>10} {'(MB)':>10} {'(tok/s)':>16} {'(ms/tok)':>16} {'vs base':>8}")
        print(f"  {'-'*96}")

        for s in group:
            speedup = s["throughput_mean"] / baseline_tps if baseline_tps else 1.0
            tps_str = f"{s['throughput_mean']:.1f} ±{s['throughput_std']:.1f}"
            lat_str = f"{s['latency_mean_ms']:.2f} ±{s['latency_std_ms']:.2f}"
            print(f"  {s['label']:<32} {s['vram_load_mb']:>10.0f} {s['vram_peak_mb']:>10.0f}"
                  f" {tps_str:>16} {lat_str:>16} {speedup:>7.2f}x")

        print(f"{'='*W}")

    # VRAM savings across all (use first batch_size group)
    first_bs = batch_sizes[0]
    group0 = [s for s in valid if s["batch_size"] == first_bs]
    if len(group0) >= 2:
        print(f"\nVRAM Savings (load, bs={first_bs}):")
        base_vram = group0[0]["vram_load_mb"]
        for s in group0[1:]:
            if base_vram > 0:
                pct = (base_vram - s["vram_load_mb"]) / base_vram * 100
                print(f"  {s['label']} vs {group0[0]['label']}: {pct:+.1f}%"
                      f"  ({base_vram:.0f} → {s['vram_load_mb']:.0f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def print_marlin_help():
    print("""
awq_marlin PTX Fix Guide
========================

Lỗi: CUDA error: the provided PTX was compiled with an unsupported toolchain
Nguyên nhân: vLLM 0.17 compile Marlin kernels cho CUDA 12.4+,
             nhưng NVIDIA driver trên server quá cũ.

Cách kiểm tra:
  nvidia-smi                          # xem "CUDA Version: X.Y" góc phải
  python -c "import torch; print(torch.version.cuda)"  # CUDA của PyTorch

Cách fix (chọn 1):

1. Nâng cấp NVIDIA driver (cần sudo/root):
   # Ubuntu
   sudo apt install nvidia-driver-535   # hoặc mới hơn
   sudo reboot

2. Compile vLLM từ source cho đúng CUDA:
   pip uninstall vllm -y
   git clone https://github.com/vllm-project/vllm
   cd vllm
   # Đảm bảo CUDA toolkit khớp driver
   pip install -e . --no-build-isolation

3. Dùng vLLM nightly wheel cho CUDA cũ hơn:
   pip install vllm --pre --index-url https://wheels.vllm.ai/nightly/

Nếu không fix được, dùng --vllm-quant awq (không dùng Marlin).
AWQ thường vẫn nhanh hơn FP16 đáng kể (~2-3x với batching).
""")


def main():
    parser = argparse.ArgumentParser(
        description="Compare inference speed: FP16 vs AWQ-dequant vs HF-AWQ vs vLLM-AWQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fp16-path",    type=str, default="",
                        help="FP16 baseline model path (HuggingFace format)")
    parser.add_argument("--fp16dq-path",  type=str, default="",
                        help="FP16 dequantized AWQ model path (awq_js_xl output)")
    parser.add_argument("--hf-awq-path",  type=str, default="",
                        help="AWQ INT4 model từ HuggingFace (vd: ./models/Mistral-7B-AWQ), load bằng vLLM")
    parser.add_argument("--vllm-path",    type=str, default="",
                        help="Custom vLLM AWQ checkpoint (convert_custom_awq_to_vllm output)")
    parser.add_argument("--vllm-quant",   type=str, default="awq",
                        choices=["awq", "awq_marlin"],
                        help="vLLM quantization backend cho --vllm-path")
    parser.add_argument("--n-tokens",     type=int, default=128,
                        help="Tokens to generate per sequence")
    parser.add_argument("--n-warmup",     type=int, default=2,
                        help="Warmup runs before timing")
    parser.add_argument("--use-long",     action="store_true",
                        help="Use long prompts instead of short")
    parser.add_argument("--batch-sizes",  type=str, default="1",
                        help="Batch sizes to test, comma-separated (vd: '1,16')")
    parser.add_argument("--gpu-mem-util", type=float, default=0.8,
                        help="vLLM gpu_memory_utilization")
    parser.add_argument("--help-marlin",  action="store_true",
                        help="Show guide to fix awq_marlin PTX error")
    args = parser.parse_args()

    if args.help_marlin:
        print_marlin_help()
        return

    if not any([args.fp16_path, args.fp16dq_path, args.hf_awq_path, args.vllm_path]):
        parser.error("Provide at least one of --fp16-path, --fp16dq-path, --hf-awq-path, --vllm-path")

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    prompts = LONG_PROMPTS if args.use_long else SHORT_PROMPTS

    print("=" * 90)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 90)
    print(f"Prompts     : {len(prompts)} ({'long' if args.use_long else 'short'})")
    print(f"New tokens  : {args.n_tokens}")
    print(f"Warmup      : {args.n_warmup}")
    print(f"Batch sizes : {batch_sizes}")
    print("=" * 90)

    summaries = []
    baseline_label = None

    for bs in batch_sizes:
        print(f"\n{'#'*60}")
        print(f"#  Batch Size = {bs}")
        print(f"{'#'*60}")

        if args.fp16_path:
            s = benchmark_hf(args.fp16_path, "FP16 Baseline", prompts,
                             args.n_tokens, args.n_warmup, batch_size=bs)
            summaries.append(s)
            if baseline_label is None:
                baseline_label = "FP16 Baseline"

        if args.fp16dq_path:
            s = benchmark_hf(args.fp16dq_path, "AWQ FP16-dequant (HF)", prompts,
                             args.n_tokens, args.n_warmup, batch_size=bs)
            summaries.append(s)
            if baseline_label is None:
                baseline_label = "AWQ FP16-dequant (HF)"

        if args.hf_awq_path:
            # HuggingFace AWQ model loaded via vLLM (quant already set in config.json)
            s = benchmark_vllm(args.hf_awq_path, "HF AWQ INT4 (vLLM)", prompts,
                               args.n_tokens, args.n_warmup,
                               quantization="awq",
                               gpu_memory_utilization=args.gpu_mem_util,
                               batch_size=bs)
            summaries.append(s)
            if baseline_label is None:
                baseline_label = "HF AWQ INT4 (vLLM)"

        if args.vllm_path:
            label = f"Custom AWQ INT4 ({args.vllm_quant})"
            s = benchmark_vllm(args.vllm_path, label, prompts,
                               args.n_tokens, args.n_warmup,
                               quantization=args.vllm_quant,
                               gpu_memory_utilization=args.gpu_mem_util,
                               batch_size=bs)
            summaries.append(s)
            if baseline_label is None:
                baseline_label = label

    print_table(summaries, baseline_label)


if __name__ == "__main__":
    main()
