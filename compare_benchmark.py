"""
compare_benchmark.py

Kiểm tra tính đúng đắn của benchmark (perplexity evaluation):

1. SANITY CHECK — FP16 model có PPL hợp lý không?
   - WikiText-2: Mistral-7B ~5.0, Llama-3-8B ~6.0 (từ paper)
   - Nếu PPL quá cao (~15) → BOS bug / tokenizer bug

2. SLIDING WINDOW CHECK — So sánh 3 cách tính PPL:
   a. Simple (no stride) — sai chuẩn, chỉ dùng để so sánh
   b. Sliding window stride=512 — chuẩn lm-eval
   c. Full context (stride=max_len) — cận trên

3. CONVERT ROUNDTRIP CHECK — Sau khi convert_custom_awq_to_vllm:
   - PPL vLLM AWQ có sát với FP16-dequant không?
   - Max dequant error có nhỏ không? (< 0.01)

4. BOS TOKEN CHECK — Phát hiện "Double BOS" bug
   - So sánh PPL khi có / không có BOS

Usage:
  # Chỉ sanity check FP16 baseline
  python compare_benchmark.py --model-path ./models/Mistral-7B-v0.3

  # Full check: FP16 vs vLLM
  python compare_benchmark.py \
      --model-path  ./models/Mistral-7B-v0.3 \
      --vllm-path   ./quantized_models/model_awq_js_xl_vllm \
      --fp16dq-path ./quantized_models/model_awq_js_xl

  # Chỉ check sliding window trên FP16
  python compare_benchmark.py --model-path ./models/Mistral-7B-v0.3 --check sliding
"""

import argparse
import time
import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ── Tham chiếu PPL từ paper / lm-eval-harness ─────────────────────────────────
KNOWN_PPL = {
    # model_name_substr : (dataset, expected_ppl, tolerance)
    "Mistral-7B":    ("WikiText-2", 5.25,  0.5),
    "Llama-3-8B":    ("WikiText-2", 6.14,  0.5),
    "Llama-2-7B":    ("WikiText-2", 5.47,  0.5),
    "Llama-3.1-8B":  ("WikiText-2", 6.24,  0.5),
}

# ── Load WikiText-2 test ───────────────────────────────────────────────────────

def load_wikitext2(max_chars=None):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n".join(x for x in dataset["text"] if x)
    if max_chars:
        full_text = full_text[:max_chars]
    return full_text


# ── Simple PPL (no sliding window) ───────────────────────────────────────────

@torch.no_grad()
def ppl_simple(model, tokenizer, text, max_length=2048, device="cuda"):
    """Tính PPL đơn giản: truncate từng chunk, không overlap."""
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encodings.input_ids.to(device)

    # Inject BOS once
    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
        input_ids = torch.cat([bos, input_ids], dim=1)

    total_nll = 0.0
    total_tokens = 0

    for i in range(0, input_ids.shape[1] - 1, max_length):
        chunk = input_ids[:, i:i + max_length]
        if chunk.shape[1] < 2:
            break
        labels = chunk.clone()
        outputs = model(chunk, labels=labels)
        n = labels.shape[1]
        total_nll += outputs.loss.item() * n
        total_tokens += n

    return float(np.exp(total_nll / total_tokens)) if total_tokens > 0 else float("inf")


# ── Sliding window PPL (chuẩn lm-eval) ───────────────────────────────────────

@torch.no_grad()
def ppl_sliding_window(model, tokenizer, text, max_length=2048, stride=512, device="cuda"):
    """PPL chuẩn với sliding window (stride)."""
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encodings.input_ids.to(device)

    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
        input_ids = torch.cat([bos, input_ids], dim=1)

    seq_len = input_ids.shape[1]
    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end

        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        if begin > 0:
            labels[:, :-trg_len] = -100

        outputs = model(chunk, labels=labels)
        nlls.append(outputs.loss * trg_len)
        prev_end = end
        if end == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    return float(torch.exp(total_nll / seq_len).item())


# ── BOS check ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def check_bos_bug(model, tokenizer, sample_text, max_length=512, device="cuda"):
    """
    So sánh PPL với và không có BOS.
    Double BOS thường gây PPL tăng ~2-3x.
    """
    encodings = tokenizer(sample_text[:2000], return_tensors="pt", add_special_tokens=True)
    ids_with_bos = encodings.input_ids.to(device)[:, :max_length]

    encodings2 = tokenizer(sample_text[:2000], return_tensors="pt", add_special_tokens=False)
    ids_no_bos = encodings2.input_ids.to(device)[:, :max_length]

    def quick_ppl(ids):
        out = model(ids, labels=ids.clone())
        return float(torch.exp(out.loss).item())

    ppl_with = quick_ppl(ids_with_bos)
    ppl_without = quick_ppl(ids_no_bos)

    ratio = ppl_with / ppl_without if ppl_without > 0 else float("inf")
    double_bos = ids_with_bos[0, 0] == ids_with_bos[0, 1] == tokenizer.bos_token_id

    return {
        "ppl_with_bos":    ppl_with,
        "ppl_without_bos": ppl_without,
        "ratio":           ratio,
        "double_bos_detected": bool(double_bos),
    }


# ── Convert roundtrip check ───────────────────────────────────────────────────

def check_convert_roundtrip(fp16dq_path: str, vllm_path: str, group_size: int = 128,
                             n_layers: int = 5):
    """
    So sánh dequantized weights giữa FP16-dequant và re-quantized vLLM format.
    Dùng safetensors trực tiếp, không cần load model đầy đủ.
    """
    print("\n[Convert Roundtrip Check]")

    try:
        from safetensors.torch import load_file
    except ImportError:
        print("  ❌ safetensors not installed")
        return None

    fp16_files = sorted(Path(fp16dq_path).glob("*.safetensors"))
    vllm_files = sorted(Path(vllm_path).glob("*.safetensors"))

    if not fp16_files:
        fp16_files = [Path(fp16dq_path) / "pytorch_model.bin"]
    if not vllm_files:
        print("  ❌ No safetensors in vllm-path")
        return None

    fp16_sd = {}
    for f in fp16_files:
        fp16_sd.update(load_file(f))

    vllm_sd = {}
    for f in vllm_files:
        vllm_sd.update(load_file(f))

    # Find quantized layers (have .qweight)
    qweight_keys = [k for k in vllm_sd if k.endswith(".qweight")][:n_layers]
    if not qweight_keys:
        print("  ❌ No .qweight keys found in vllm checkpoint")
        return None

    errors = []
    for qkey in qweight_keys:
        prefix = qkey[:-len(".qweight")]
        w_key = f"{prefix}.weight"

        if w_key not in fp16_sd:
            continue

        W_fp16 = fp16_sd[w_key].float()   # [N, K]
        qweight = vllm_sd[qkey]           # [K, N//8]
        scales  = vllm_sd.get(f"{prefix}.scales")
        qzeros  = vllm_sd.get(f"{prefix}.qzeros")

        if scales is None or qzeros is None:
            continue

        # Dequant manually từ INT4 packed format
        N = W_fp16.shape[0]
        K = W_fp16.shape[1]
        n_groups = K // group_size

        # Unpack qweight [K, N//8] → [K, N] int32
        W_int = torch.zeros(K, N, dtype=torch.int32)
        for i in range(8):
            W_int[:, i::8] = (qweight >> (i * 4)) & 0xF

        # Unpack qzeros [n_groups, N//8] → [n_groups, N] int32
        pack_order = [0, 2, 4, 6, 1, 3, 5, 7]
        zeros_int = torch.zeros(n_groups, N, dtype=torch.int32)
        for bit_pos, col_off in enumerate(pack_order):
            zeros_int[:, col_off::8] = (qzeros >> (bit_pos * 4)) & 0xF

        # scales: [n_groups, N] fp16 → expand to [K, N]
        s = scales.float()   # [n_groups, N]
        z = zeros_int.float()

        s_exp = s.unsqueeze(1).expand(n_groups, group_size, N).reshape(K, N)
        z_exp = z.unsqueeze(1).expand(n_groups, group_size, N).reshape(K, N)

        # Dequant: (W_int - zero) * scale → [K, N] → transpose → [N, K]
        W_deq = ((W_int.float() - z_exp) * s_exp).T   # [N, K]

        err = (W_deq - W_fp16).abs()
        errors.append({
            "layer": prefix,
            "max_err": err.max().item(),
            "mean_err": err.mean().item(),
        })
        status = "OK" if err.mean().item() < 0.01 else "WARN"
        print(f"  [{status}] {prefix}: max={err.max().item():.5f}  mean={err.mean().item():.6f}")

    if errors:
        avg_mean = np.mean([e["mean_err"] for e in errors])
        avg_max  = np.max([e["max_err"]  for e in errors])
        print(f"\n  Summary: avg_mean_err={avg_mean:.6f}  max_err={avg_max:.5f}")
        if avg_mean < 0.01:
            print("  ✅ Conversion roundtrip looks correct")
        else:
            print("  ⚠️  High dequantization error — check group_size or pack order")

    return errors


# ── Load model helper ─────────────────────────────────────────────────────────

def load_model_hf(path, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify benchmark correctness for AWQ quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path",  type=str, default="", help="FP16 baseline model")
    parser.add_argument("--fp16dq-path", type=str, default="", help="FP16 dequantized AWQ (awq_js_xl output)")
    parser.add_argument("--vllm-path",   type=str, default="", help="vLLM AWQ INT4 checkpoint")
    parser.add_argument("--group-size",  type=int, default=128, help="Must match converter group_size")
    parser.add_argument("--max-length",  type=int, default=2048)
    parser.add_argument("--stride",      type=int, default=512)
    parser.add_argument("--max-chars",   type=int, default=500_000,
                        help="Limit WikiText-2 chars for speed (0=full)")
    parser.add_argument("--check", type=str, default="all",
                        choices=["all", "sanity", "sliding", "bos", "roundtrip"],
                        help="Which checks to run")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    print("=" * 70)
    print("BENCHMARK CORRECTNESS CHECKER")
    print("=" * 70)

    max_chars = args.max_chars if args.max_chars > 0 else None

    # ── 1. Sliding window check trên FP16 ─────────────────────────────────────
    if args.model_path and args.check in ("all", "sanity", "sliding", "bos"):
        print("\n" + "="*70)
        print(f"Model: {args.model_path}")
        print("="*70)

        model, tokenizer = load_model_hf(args.model_path, device)
        text = load_wikitext2(max_chars)

        # BOS check
        if args.check in ("all", "bos"):
            print("\n[BOS Token Check]")
            bos_res = check_bos_bug(model, tokenizer, text[:10000], device=device)
            print(f"  PPL with BOS    : {bos_res['ppl_with_bos']:.4f}")
            print(f"  PPL without BOS : {bos_res['ppl_without_bos']:.4f}")
            print(f"  Ratio           : {bos_res['ratio']:.3f}")
            if bos_res["double_bos_detected"]:
                print("  ⚠️  DOUBLE BOS DETECTED! Use add_special_tokens=False + manual BOS")
            else:
                print("  ✅ No double BOS")
            results["bos_check"] = bos_res

        # Sliding window check
        if args.check in ("all", "sliding"):
            print("\n[Sliding Window Comparison]")
            print(f"  Text length: {len(text):,} chars")

            print("  Computing Simple PPL (no stride)...")
            ppl_s = ppl_simple(model, tokenizer, text, args.max_length, device)
            print(f"  Simple PPL    : {ppl_s:.4f}")

            print(f"  Computing Sliding Window PPL (stride={args.stride})...")
            ppl_sw = ppl_sliding_window(model, tokenizer, text, args.max_length, args.stride, device)
            print(f"  Sliding PPL   : {ppl_sw:.4f}")

            diff_pct = (ppl_sw - ppl_s) / ppl_s * 100
            print(f"  Difference    : {diff_pct:+.2f}%")
            if abs(diff_pct) > 20:
                print("  ⚠️  Large difference — check stride/masking logic")
            else:
                print("  ✅ Consistent")

            results["sliding_check"] = {
                "ppl_simple": ppl_s,
                "ppl_sliding": ppl_sw,
                "diff_pct": diff_pct,
            }

        # Sanity check vs known PPL
        if args.check in ("all", "sanity"):
            print("\n[Sanity Check vs Known PPL]")
            if "ppl_sliding" not in results.get("sliding_check", {}):
                print("  Computing Sliding Window PPL...")
                ppl_sw = ppl_sliding_window(model, tokenizer, text, args.max_length, args.stride, device)
            else:
                ppl_sw = results["sliding_check"]["ppl_sliding"]

            model_name = Path(args.model_path).name
            matched = False
            for key, (dataset, expected, tol) in KNOWN_PPL.items():
                if key in model_name or key in args.model_path:
                    diff = abs(ppl_sw - expected)
                    ok = diff <= tol
                    print(f"  Model: {key}  Expected {dataset} PPL ≈ {expected:.2f}  Got: {ppl_sw:.4f}")
                    if ok:
                        print(f"  ✅ Within tolerance (±{tol})")
                    else:
                        print(f"  ⚠️  Outside tolerance! diff={diff:.4f} > {tol}")
                    matched = True
                    break

            if not matched:
                print(f"  FP16 Baseline PPL (WikiText-2): {ppl_sw:.4f}")
                print(f"  (No known reference for this model — verify manually)")

            results["sanity_check"] = {"ppl_wikitext2": ppl_sw}

        del model
        torch.cuda.empty_cache()

    # ── 2. FP16-dequant check ─────────────────────────────────────────────────
    if args.fp16dq_path and args.check in ("all", "sanity", "sliding"):
        print("\n" + "="*70)
        print(f"[FP16-dequant Model] {args.fp16dq_path}")
        print("="*70)

        model, tokenizer = load_model_hf(args.fp16dq_path, device)
        text = load_wikitext2(max_chars)

        print("  Computing Sliding Window PPL...")
        ppl_dq = ppl_sliding_window(model, tokenizer, text, args.max_length, args.stride, device)
        print(f"  FP16-dequant PPL (WikiText-2): {ppl_dq:.4f}")

        if "sanity_check" in results:
            base_ppl = results["sanity_check"]["ppl_wikitext2"]
            degradation = (ppl_dq - base_ppl) / base_ppl * 100
            print(f"  Degradation vs FP16: {degradation:+.2f}%")
            if degradation > 5:
                print("  ⚠️  High degradation! Check quantization quality")
            else:
                print("  ✅ Acceptable degradation")

        results["fp16dq_check"] = {"ppl_wikitext2": ppl_dq}
        del model
        torch.cuda.empty_cache()

    # ── 3. Roundtrip check ────────────────────────────────────────────────────
    if args.fp16dq_path and args.vllm_path and args.check in ("all", "roundtrip"):
        errors = check_convert_roundtrip(
            args.fp16dq_path, args.vllm_path, args.group_size, n_layers=5
        )
        results["roundtrip_check"] = errors

    # ── 4. vLLM PPL check ─────────────────────────────────────────────────────
    # Dùng vLLM generate() để tính PPL thay vì load HF (vLLM format không tương thích HF)
    if args.vllm_path and args.check in ("all", "sanity"):
        print("\n" + "="*70)
        print(f"[vLLM AWQ INT4] {args.vllm_path}")
        print("="*70)

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            print("  ❌ vLLM not installed, skipping vLLM check")
        else:
            try:
                llm = LLM(model=args.vllm_path, quantization="awq", dtype="float16")
                tokenizer_vllm = AutoTokenizer.from_pretrained(
                    args.vllm_path, trust_remote_code=True, use_fast=True
                )

                text = load_wikitext2(max_chars)
                # Tokenize và chia thành chunks để tính PPL qua vLLM logprobs
                encodings = tokenizer_vllm(
                    text[:50000], return_tensors="pt", add_special_tokens=False
                )
                input_ids = encodings.input_ids[0].tolist()

                chunk_size = args.max_length - 1
                stride = args.stride
                nlls = []
                total_tokens = 0

                print(f"  Computing PPL via vLLM logprobs (stride={stride})...")
                prev_end = 0
                for begin in range(0, len(input_ids), stride):
                    end = min(begin + chunk_size, len(input_ids))
                    trg_len = end - prev_end
                    chunk = input_ids[begin:end]
                    if len(chunk) < 2:
                        break

                    prompt_ids = chunk[:-1]
                    target_ids = chunk[1:]

                    sp = SamplingParams(
                        temperature=0, max_tokens=1,
                        prompt_logprobs=len(prompt_ids),
                    )
                    # vLLM >= 0.4: pass token IDs via dict, not kwarg
                    out = llm.generate(
                        [{"prompt_token_ids": prompt_ids}], sampling_params=sp
                    )
                    logprobs_list = out[0].prompt_logprobs  # list of dicts

                    # logprobs_list[i] = {token_id: Logprob} or None for first token
                    nll = 0.0
                    scored = 0
                    start_score = max(0, len(chunk) - 1 - trg_len)
                    for i in range(start_score, len(target_ids)):
                        lp_dict = logprobs_list[i + 1] if (i + 1) < len(logprobs_list) else None
                        if lp_dict and target_ids[i] in lp_dict:
                            nll -= lp_dict[target_ids[i]].logprob
                            scored += 1

                    if scored > 0:
                        nlls.append(nll)
                        total_tokens += scored

                    prev_end = end
                    if end == len(input_ids):
                        break

                if total_tokens > 0:
                    import math
                    ppl_vllm = math.exp(sum(nlls) / total_tokens)
                    print(f"  vLLM AWQ PPL (WikiText-2 subset): {ppl_vllm:.4f}")

                    if "fp16dq_check" in results:
                        base = results["fp16dq_check"]["ppl_wikitext2"]
                        diff = (ppl_vllm - base) / base * 100
                        print(f"  vs FP16-dequant: {diff:+.2f}%")
                        if abs(diff) < 1.0:
                            print("  ✅ PPL consistent — conversion correct")
                        else:
                            print("  ⚠️  PPL differs — check group_size or pack order")

                    results["vllm_check"] = {"ppl_wikitext2": ppl_vllm}
                else:
                    print("  ⚠️  Could not compute vLLM PPL (no logprobs)")

                del llm
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ❌ vLLM PPL check failed: {e}")
                import traceback
                traceback.print_exc()

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if "bos_check" in results:
        r = results["bos_check"]
        status = "❌ Double BOS" if r["double_bos_detected"] else "✅ BOS OK"
        print(f"  BOS check         : {status}")

    if "sliding_check" in results:
        r = results["sliding_check"]
        ok = abs(r["diff_pct"]) <= 20
        print(f"  Sliding window    : {'✅' if ok else '⚠️ '} simple={r['ppl_simple']:.4f}  sliding={r['ppl_sliding']:.4f}  diff={r['diff_pct']:+.2f}%")

    if "sanity_check" in results:
        print(f"  FP16 PPL          : {results['sanity_check']['ppl_wikitext2']:.4f}")

    if "fp16dq_check" in results:
        print(f"  FP16-dequant PPL  : {results['fp16dq_check']['ppl_wikitext2']:.4f}")

    if "vllm_check" in results:
        print(f"  vLLM AWQ PPL      : {results['vllm_check']['ppl_wikitext2']:.4f}")

    if "roundtrip_check" in results and results["roundtrip_check"]:
        avg = np.mean([e["mean_err"] for e in results["roundtrip_check"]])
        status = "✅ OK" if avg < 0.01 else "⚠️  High error"
        print(f"  Roundtrip error   : {status} (avg_mean={avg:.6f})")

    print("="*70)


if __name__ == "__main__":
    main()
