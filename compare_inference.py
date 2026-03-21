"""
compare_inference.py

So sánh tốc độ inference giữa:
  1. FP16 baseline (model gốc, HuggingFace)
  2. Custom AWQ FP16 dequantized (output của awq_stand_xl / awq_js_xl, HuggingFace)
  3. HF AWQ INT4 (model AWQ tải từ HuggingFace, dùng vLLM)
  4. Custom AWQ INT4 (output của convert_custom_awq_to_vllm.py, dùng vLLM)

Mỗi benchmark chạy trong subprocess riêng → CUDA context sạch hoàn toàn giữa các lần,
tránh OOM do PyTorch giữ CUDA cache sau khi unload model.

Usage:
  python compare_inference.py \\
      --fp16-path    ./models/Mistral-7B-v0.3 \\
      --hf-awq-path  ./models/Mistral-7B-v0.3-AWQ \\
      --vllm-path    ./quantized_models/mistral_awq_js_vllm \\
      --batch-sizes  1,16
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import gc
import torch
from pathlib import Path
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Prompts ───────────────────────────────────────────────────────────────────

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
    return (prompts * math.ceil(batch_size / len(prompts)))[:batch_size]


def nvidia_smi_vram_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


# ── HuggingFace benchmark (runs inside worker subprocess) ─────────────────────

def _bench_hf(model_path: str, label: str, prompts: list, n_tokens: int,
              n_warmup: int, batch_size: int) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    torch.cuda.synchronize()
    vram_load_mb = nvidia_smi_vram_mb()  # total GPU usage, same metric as vLLM

    # Warmup
    warmup_batch = make_batch(prompts, min(batch_size, 2))
    for _ in range(n_warmup):
        inp = tokenizer(warmup_batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()

    # Benchmark
    bench_batches = [[p] for p in prompts] if batch_size == 1 else [make_batch(prompts, batch_size)] * 5

    results = []
    vram_peak_mb = 0
    for batch in bench_batches:
        inp = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        input_len = inp["input_ids"].shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=n_tokens, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_new_total = (out.shape[1] - input_len) * len(batch)
        peak = nvidia_smi_vram_mb()  # total GPU usage, same metric as vLLM
        vram_peak_mb = max(vram_peak_mb, peak)
        results.append({
            "throughput": n_new_total / elapsed,
            "latency_ms_per_token": elapsed / (out.shape[1] - input_len) * 1000,
        })

    tps = [r["throughput"] for r in results]
    lat = [r["latency_ms_per_token"] for r in results]
    return {
        "label": label, "batch_size": batch_size,
        "vram_load_mb": vram_load_mb, "vram_peak_mb": vram_peak_mb,
        "throughput_mean": float(np.mean(tps)), "throughput_std": float(np.std(tps)),
        "latency_mean_ms": float(np.mean(lat)), "latency_std_ms": float(np.std(lat)),
    }


# ── vLLM benchmark (runs inside worker subprocess) ────────────────────────────

def _bench_vllm(model_path: str, label: str, prompts: list, n_tokens: int,
                n_warmup: int, quantization: str, gpu_memory_utilization: float,
                batch_size: int, max_model_len: int) -> dict:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        return {"error": "vLLM not installed"}

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
                  gpu_memory_utilization=gpu_memory_utilization,
                  max_model_len=max_model_len)
    except Exception as e:
        if config_backup is not None:
            with open(config_path, "w") as f:
                f.write(config_backup)
        err = str(e)
        is_ptx = ("PTX" in err or "unsupported toolchain" in err
                  or "cudaErrorUnsupportedPtxVersion" in err
                  or (quantization == "awq_marlin" and "Engine core initialization failed" in err))
        return {"error": f"PTX_MISMATCH: {err}" if is_ptx else f"LOAD_FAILED: {err}"}
    finally:
        if config_backup is not None:
            with open(config_path, "w") as f:
                f.write(config_backup)

    sampling = SamplingParams(temperature=0, max_tokens=n_tokens)
    vram_load_mb = nvidia_smi_vram_mb()

    warmup_batch = make_batch(prompts, min(batch_size, 2))
    for _ in range(n_warmup):
        llm.generate(warmup_batch, SamplingParams(temperature=0, max_tokens=16))

    bench_batches = [[p] for p in prompts] if batch_size == 1 else [make_batch(prompts, batch_size)] * 5

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
            "throughput": n_new_total / elapsed,
            "latency_ms_per_token": elapsed / avg_per_seq * 1000,
        })

    tps = [r["throughput"] for r in results]
    lat = [r["latency_ms_per_token"] for r in results]
    return {
        "label": label, "batch_size": batch_size,
        "vram_load_mb": vram_load_mb, "vram_peak_mb": vram_peak_mb,
        "throughput_mean": float(np.mean(tps)), "throughput_std": float(np.std(tps)),
        "latency_mean_ms": float(np.mean(lat)), "latency_std_ms": float(np.std(lat)),
    }


# ── Subprocess worker entry point ─────────────────────────────────────────────
# Called internally: python compare_inference.py --_worker <config.json> <result.json>

def _worker_main():
    config_file = sys.argv[2]
    result_file = sys.argv[3]
    with open(config_file) as f:
        cfg = json.load(f)

    func = cfg["func"]
    kwargs = cfg["kwargs"]

    if func == "bench_hf":
        result = _bench_hf(**kwargs)
    elif func == "bench_vllm":
        result = _bench_vllm(**kwargs)
    else:
        result = {"error": f"Unknown func: {func}"}

    with open(result_file, "w") as f:
        json.dump(result, f)


# ── Subprocess launcher ───────────────────────────────────────────────────────

def run_subprocess_benchmark(func: str, kwargs: dict, label: str) -> dict | None:
    """
    Spawn a fresh Python subprocess to run one benchmark.
    Each subprocess gets a clean CUDA context → no residual memory from prior runs.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as cf:
        json.dump({"func": func, "kwargs": kwargs}, cf)
        config_file = cf.name
    result_file = config_file + ".result"

    print(f"\n  → Spawning subprocess for: {label}")
    proc = subprocess.run(
        [sys.executable, __file__, "--_worker", config_file, result_file],
        text=True,
    )

    result = None
    if os.path.exists(result_file):
        with open(result_file) as f:
            result = json.load(f)
        os.unlink(result_file)
    os.unlink(config_file)

    if result is None or "error" in result:
        err = result.get("error", "subprocess failed") if result else "no result"
        if "PTX_MISMATCH" in err:
            print(f"  ⚠️  awq_marlin PTX mismatch (CUDA driver too old). Run --help-marlin.")
        else:
            print(f"  ❌ {label}: {err}")
        return None

    return result


# ── Results table ─────────────────────────────────────────────────────────────

def print_table(summaries: list, baseline_label: str = None):
    valid = [s for s in summaries if s is not None]
    if not valid:
        print("No results to display.")
        return

    batch_sizes = sorted(set(s["batch_size"] for s in valid))
    for bs in batch_sizes:
        group = [s for s in valid if s["batch_size"] == bs]
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

    first_bs = batch_sizes[0]
    group0 = [s for s in valid if s["batch_size"] == first_bs]
    if len(group0) >= 2:
        print(f"\nVRAM (load weights only, bs={first_bs}):")
        print(f"  Note: vLLM models show higher total VRAM due to KV cache pre-allocation.")
        print(f"  Compare 'VRAM Load' across HF vs vLLM models carefully.")
        base_vram = group0[0]["vram_load_mb"]
        for s in group0[1:]:
            if base_vram > 0:
                delta_mb = s["vram_load_mb"] - base_vram
                pct = delta_mb / base_vram * 100
                direction = "more" if delta_mb > 0 else "less"
                print(f"  {s['label']} vs {group0[0]['label']}: {abs(pct):.1f}% {direction} VRAM"
                      f"  ({base_vram:.0f} → {s['vram_load_mb']:.0f} MB)")


# ── Help text ──────────────────────────────────────────────────────────────────

def print_marlin_help():
    print("""
awq_marlin PTX Fix Guide
========================
Lỗi: CUDA error: the provided PTX was compiled with an unsupported toolchain
Nguyên nhân: vLLM 0.17 compile Marlin kernels cho CUDA 12.5+,
             nhưng NVIDIA driver trên server quá cũ.

Kiểm tra:
  nvidia-smi                                           # CUDA Version góc phải
  python -c "import torch; print(torch.version.cuda)" # CUDA của PyTorch

Fix (chọn 1):
  1. Nâng NVIDIA driver (cần sudo): sudo apt install nvidia-driver-565 && sudo reboot
  2. Cài lại torch + vLLM đúng cu124:
       pip install torch --index-url https://download.pytorch.org/whl/cu124
       pip install vllm==0.17.1
  3. vLLM nightly cu124: pip install vllm --pre --index-url https://wheels.vllm.ai/nightly/cu124/
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Internal worker mode — called by subprocess
    if len(sys.argv) >= 2 and sys.argv[1] == "--_worker":
        _worker_main()
        return

    parser = argparse.ArgumentParser(
        description="Compare inference speed. Each model runs in an isolated subprocess.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fp16-path",    type=str, default="",
                        help="FP16 baseline model path")
    parser.add_argument("--fp16dq-path",  type=str, default="",
                        help="FP16 dequantized AWQ model path (awq_js_xl output)")
    parser.add_argument("--hf-awq-path",  type=str, default="",
                        help="AWQ INT4 model từ HuggingFace, load bằng vLLM")
    parser.add_argument("--vllm-path",    type=str, default="",
                        help="Custom vLLM AWQ checkpoint (convert_custom_awq_to_vllm output)")
    parser.add_argument("--vllm-quant",   type=str, default="awq",
                        choices=["awq", "awq_marlin"],
                        help="vLLM quantization backend cho --vllm-path")
    parser.add_argument("--n-tokens",     type=int, default=128,
                        help="Tokens to generate per sequence")
    parser.add_argument("--n-warmup",     type=int, default=2,
                        help="Warmup runs")
    parser.add_argument("--use-long",     action="store_true",
                        help="Use long prompts")
    parser.add_argument("--batch-sizes",  type=str, default="1",
                        help="Batch sizes, comma-separated (vd: '1,16')")
    parser.add_argument("--gpu-mem-util", type=float, default=0.8,
                        help="vLLM gpu_memory_utilization")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="vLLM max_model_len (giảm KV cache, default 4096)")
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
    print("(each model runs in isolated subprocess — clean CUDA context)")
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

        base_kwargs = dict(prompts=prompts, n_tokens=args.n_tokens,
                           n_warmup=args.n_warmup, batch_size=bs)

        if args.fp16_path:
            s = run_subprocess_benchmark(
                "bench_hf",
                dict(model_path=args.fp16_path, label="FP16 Baseline", **base_kwargs),
                f"FP16 Baseline bs={bs}",
            )
            summaries.append(s)
            if baseline_label is None:
                baseline_label = "FP16 Baseline"

        if args.fp16dq_path:
            s = run_subprocess_benchmark(
                "bench_hf",
                dict(model_path=args.fp16dq_path, label="AWQ FP16-dequant (HF)", **base_kwargs),
                f"AWQ FP16-dequant bs={bs}",
            )
            summaries.append(s)
            if baseline_label is None:
                baseline_label = "AWQ FP16-dequant (HF)"

        if args.hf_awq_path:
            s = run_subprocess_benchmark(
                "bench_vllm",
                dict(model_path=args.hf_awq_path, label="HF AWQ INT4 (vLLM)",
                     quantization="awq",
                     gpu_memory_utilization=args.gpu_mem_util,
                     max_model_len=args.max_model_len, **base_kwargs),
                f"HF AWQ INT4 bs={bs}",
            )
            summaries.append(s)
            if baseline_label is None:
                baseline_label = "HF AWQ INT4 (vLLM)"

        if args.vllm_path:
            label = f"Custom AWQ INT4 ({args.vllm_quant})"
            s = run_subprocess_benchmark(
                "bench_vllm",
                dict(model_path=args.vllm_path, label=label,
                     quantization=args.vllm_quant,
                     gpu_memory_utilization=args.gpu_mem_util,
                     max_model_len=args.max_model_len, **base_kwargs),
                f"Custom AWQ INT4 bs={bs}",
            )
            summaries.append(s)
            if baseline_label is None:
                baseline_label = label

    print_table(summaries, baseline_label)


if __name__ == "__main__":
    main()
