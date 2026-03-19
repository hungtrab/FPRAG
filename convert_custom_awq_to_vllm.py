"""
convert_custom_awq_to_vllm.py

Chuyển đổi checkpoint từ awq_stand_xl.py / awq_js_xl.py (FP16 dequantized)
sang AWQ format mà vLLM có thể load bằng quantization="awq" hoặc "awq_marlin".

Vấn đề:
  Teacher scripts lưu model bằng model.save_pretrained() sau khi fake-quantize:
    W_final = (quantize(W * col_scale) / col_scale)  ← FP16, không phải INT4
  vLLM cần: qweight [K, N//8] int32, scales [K//g, N] fp16, qzeros [K//g, N//8] int32

Giải pháp:
  W_final đã nằm trên INT4 grid (đã có quantization noise baked-in).
  Re-quantize lại với group-wise asymmetric sẽ recover gần đúng các INT4 values gốc.
  Double-quantization error ≈ 0 vì W_final = dequant(INT4) là exact.

Ví dụ dùng:
  python convert_custom_awq_to_vllm.py \\
      --source /path/to/awq_js_output \\
      --output /tmp/converted_awq \\
      --group-size 128

  python convert_custom_awq_to_vllm.py \\
      --source /path/to/awq_js_output \\
      --output /tmp/converted_awq \\
      --group-size 128 --validate
"""

import os
import json
import shutil
import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file, load_file
from transformers import AutoConfig

from vllm.model_executor.layers.quantization.utils.quant_utils import awq_pack

PACK_FACTOR = 8   # 8 INT4 per int32

SKIP_IF_NAME_CONTAINS = ["lm_head", "embed", "norm", "layernorm"]

TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "vocab.json", "merges.txt", "special_tokens_map.json",
    "generation_config.json",
]


# ── Core quantization ─────────────────────────────────────────────────────────

def quantize_to_awq(W_fp16: torch.Tensor, group_size: int):
    """
    W_fp16 : [N, K]  (out_features × in_features, PyTorch convention)
    Returns : qweight [K, N//8] int32
              scales  [K//g, N] fp16
              qzeros  [K//g, N//8] int32

    Dequant formula: W[k,n] = (q[k,n] - zero[k//g, n]) * scale[k//g, n]
    """
    N, K = W_fp16.shape
    n_groups = K // group_size

    W = W_fp16.float().T.contiguous()               # [K, N] fp32
    W_g = W.reshape(n_groups, group_size, N)         # [n_groups, gs, N]

    w_min = W_g.min(dim=1).values                    # [n_groups, N]
    w_max = W_g.max(dim=1).values
    scales = ((w_max - w_min) / 15.0).clamp(min=1e-8)   # [n_groups, N]
    zeros  = (-w_min / scales).round().clamp(0, 15)      # [n_groups, N]

    s_exp = scales.unsqueeze(1)                      # [n_groups, 1, N]
    z_exp = zeros.unsqueeze(1)
    W_int = (W_g / s_exp + z_exp).round().clamp(0, 15).to(torch.int32)
    W_int = W_int.reshape(K, N)                      # [K, N] int32

    # Pack qweight [K, N] → [K, N//8]
    qweight = awq_pack(W_int.cpu(), 4, K, N)

    # Pack qzeros  [n_groups, N] → [n_groups, N//8]
    pack_order = [0, 2, 4, 6, 1, 3, 5, 7]
    zeros_i = zeros.to(torch.int32).cpu()
    qzeros = torch.zeros(n_groups, N // PACK_FACTOR, dtype=torch.int32)
    for bit_pos, col_off in enumerate(pack_order):
        qzeros |= (zeros_i[:, col_off::PACK_FACTOR] & 0xF) << (bit_pos * 4)

    return qweight, scales.half(), qzeros


def should_skip(name: str, shape: tuple, group_size: int) -> tuple[bool, str]:
    if len(shape) != 2:
        return True, "not 2D"
    if any(s in name for s in SKIP_IF_NAME_CONTAINS):
        return True, f"skip pattern"
    N, K = shape
    if K < group_size or K % group_size != 0:
        return True, f"K={K} not divisible by group_size={group_size}"
    if N % PACK_FACTOR != 0:
        return True, f"N={N} not divisible by {PACK_FACTOR}"
    return False, ""


# ── Validation ────────────────────────────────────────────────────────────────

def validate_layer(W_fp16, qweight, scales, qzeros, name):
    try:
        import vllm._custom_ops as ops
        dev = "cuda"
        W_deq = ops.awq_dequantize(
            qweight.to(dev), scales.to(dev), qzeros.to(dev), 0, 0, 0
        )  # [K, N] fp16
        W_ref = W_fp16.T.half().to(dev)  # [K, N]
        err = (W_deq - W_ref).abs()
        max_e = err.max().item()
        mean_e = err.mean().item()
        status = "OK" if mean_e < 0.01 else "WARN"
        print(f"    [{status}] max_err={max_e:.5f}  mean_err={mean_e:.6f}")
    except Exception as e:
        print(f"    [skip validation] {e}")


# ── Main converter ────────────────────────────────────────────────────────────

def convert(source_dir: str, output_dir: str, group_size: int, validate_n: int):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load state dict ──
    print(f"\nLoading weights from {source_dir} ...")
    sf_files = sorted(source_dir.glob("*.safetensors"))
    if sf_files:
        state_dict = {}
        for f in sf_files:
            state_dict.update(load_file(f))
    else:
        state_dict = torch.load(source_dir / "pytorch_model.bin", map_location="cpu")
    print(f"  {len(state_dict)} tensors loaded")

    # ── Identify linear weight keys ──
    weight_keys = [
        k for k, v in state_dict.items()
        if k.endswith(".weight") and v.dim() == 2
    ]

    new_sd = {}
    converted, skipped_names = 0, []
    validated = 0

    print(f"\nConverting layers (group_size={group_size}) ...")
    for key in weight_keys:
        W = state_dict[key]
        prefix = key[:-len(".weight")]
        skip, reason = should_skip(prefix, tuple(W.shape), group_size)

        if skip:
            new_sd[key] = W
            skipped_names.append(prefix)
            continue

        N, K = W.shape
        qweight, scales, qzeros = quantize_to_awq(W.cuda(), group_size)

        if validate_n > 0 and validated < validate_n:
            print(f"  Validate {prefix} [{N},{K}]:")
            validate_layer(W, qweight, scales, qzeros, prefix)
            validated += 1

        new_sd[f"{prefix}.qweight"] = qweight.cpu()
        new_sd[f"{prefix}.scales"]  = scales.cpu()
        new_sd[f"{prefix}.qzeros"]  = qzeros.cpu()

        # copy bias if present
        bias_key = f"{prefix}.bias"
        if bias_key in state_dict:
            new_sd[bias_key] = state_dict[bias_key]

        converted += 1
        if converted % 20 == 0:
            print(f"  ... {converted} layers done")

    # ── Pass through all non-weight tensors ──
    quantized_prefixes = {
        k[:-len(".weight")] for k in weight_keys
        if not should_skip(k[:-len(".weight")], tuple(state_dict[k].shape), group_size)[0]
    }
    for key, val in state_dict.items():
        if key in new_sd:
            continue
        # skip .weight of quantized layers (already replaced by qweight/scales/qzeros)
        prefix = key.rsplit(".", 1)[0] if "." in key else ""
        if key.endswith(".weight") and prefix in quantized_prefixes:
            continue
        new_sd[key] = val

    print(f"\n  Quantized : {converted} layers")
    print(f"  Kept FP16 : {len(skipped_names)} layers")
    if skipped_names:
        print(f"  Skipped   : {skipped_names}")

    # ── Save state dict ──
    print(f"\nSaving to {output_dir} ...")
    save_file(new_sd, output_dir / "model.safetensors")

    # ── Write config.json with quantization_config ──
    config = AutoConfig.from_pretrained(source_dir)
    config_dict = json.loads(config.to_json_string())
    config_dict["quantization_config"] = {
        "quant_method": "awq",
        "version": "gemm",
        "w_bit": 4,
        "q_group_size": group_size,
        "zero_point": True,
        "modules_to_not_convert": skipped_names,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # ── Copy tokenizer files ──
    for fname in TOKENIZER_FILES:
        src = source_dir / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)

    print(f"\nDone! AWQ checkpoint saved to {output_dir}")
    print(f"\nUsage:")
    print(f"  from vllm import LLM")
    print(f"  llm = LLM(model='{output_dir}', quantization='awq')")
    print(f"  # or with Marlin kernel (faster):")
    print(f"  llm = LLM(model='{output_dir}', quantization='awq_marlin')")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert fake-quantized AWQ checkpoint to vLLM AWQ format"
    )
    parser.add_argument("--source",     required=True, help="Teacher's saved model dir (FP16)")
    parser.add_argument("--output",     required=True, help="Output AWQ checkpoint dir")
    parser.add_argument("--group-size", type=int, default=128, help="Must match teacher's group_size")
    parser.add_argument("--validate",   action="store_true", help="Validate first N layers with awq_dequantize")
    parser.add_argument("--validate-n", type=int, default=3, help="Number of layers to validate")
    args = parser.parse_args()

    convert(
        source_dir=args.source,
        output_dir=args.output,
        group_size=args.group_size,
        validate_n=args.validate_n if args.validate else 0,
    )
