"""
convert_to_flute_int3.py

Convert FP16 model to FLUTE LUT-quantized format (INT3 or INT2/INT4).

FLUTE (Fast LUT-based Quantization) dùng lookup-table thay vì uniform grid,
cho phép quantize arbitrary bit widths như INT3 với custom kernels tốc độ cao.

Quy trình:
  1. Load FP16 model
  2. apply flute.integrations.base.prepare_model_flute() → thay các Linear layers
     bằng FluteLinear (lưu qweight + scales + lookup tables)
  3. (Optional) learn_scales() để fine-tune scales trên calibration data
  4. save_pretrained() → lưu model có thể load lại với FLUTE kernels

Usage:
  # INT3, group_size=128 (recommended)
  python convert_to_flute_int3.py \\
      --model-path ./models/Mistral-7B-v0.3 \\
      --output-dir ./quantized_models/mistral_flute_int3 \\
      --num-bits 3 --group-size 128 --n-calib 128

  # INT2 (aggressive compression)
  python convert_to_flute_int3.py \\
      --model-path ./models/Mistral-7B-v0.3 \\
      --output-dir ./quantized_models/mistral_flute_int2 \\
      --num-bits 2 --group-size 64

Installation:
  pip install flute-kernel jaxtyping          # CUDA 12.1
  pip install flute-kernel jaxtyping -i https://flute-ai.github.io/whl/cu124  # CUDA 12.4

Supported GPUs: A100, A6000, RTX 4090, L40S (compute capability 80+)
"""

import argparse
import gc
import json
import sys
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Check FLUTE ────────────────────────────────────────────────────────────────

try:
    import flute
    import flute.integrations.base
    HAS_FLUTE = True
except ImportError:
    HAS_FLUTE = False

try:
    import flute.integrations.learnable
    HAS_FLUTE_LEARNABLE = True
except ImportError:
    HAS_FLUTE_LEARNABLE = False

# ── Helpers ───────────────────────────────────────────────────────────────────

LAYER_PATHS = [
    # (attribute chain, display name)
    ("model.model.layers",      "model.model.layers"),   # LLaMA, Mistral, Qwen, MiniCPM
    ("model.transformer.h",     "model.transformer.h"),  # GPT-2, Falcon
    ("model.decoder.layers",    "model.decoder.layers"), # OPT
    ("model.layers",            "model.layers"),         # some custom models
]


def find_layers_module(model):
    """Find the transformer block list (model-agnostic)."""
    for attr_chain, name in LAYER_PATHS:
        try:
            obj = model
            for attr in attr_chain.split("."):
                obj = getattr(obj, attr)
            return obj, name
        except AttributeError:
            continue
    raise RuntimeError(
        "Cannot detect layer path. Supported: LLaMA/Mistral/Qwen/MiniCPM/GPT-2/OPT. "
        "Please open an issue or pass --layers-path manually."
    )


def count_linear_layers(module):
    return sum(1 for m in module.modules() if isinstance(m, torch.nn.Linear))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert FP16 model to FLUTE LUT-quantized format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path",  required=True,
                        help="Path to FP16 source model (local or HuggingFace repo)")
    parser.add_argument("--output-dir",  required=True,
                        help="Output directory for FLUTE model")
    parser.add_argument("--num-bits",    type=int, default=3, choices=[2, 3, 4],
                        help="Quantization bit width")
    parser.add_argument("--group-size",  type=int, default=128, choices=[32, 64, 128, 256],
                        help="Channels per quantization group")
    parser.add_argument("--learn-scales", action="store_true", default=False,
                        help="Fine-tune scales on calibration data after quantization "
                             "(better quality, requires more time)")
    parser.add_argument("--n-calib",     type=int, default=128,
                        help="Calibration samples for learn-scales")
    parser.add_argument("--calib-dataset", default="wikitext2-simple",
                        choices=["c4", "wikitext2", "wikitext2-simple"],
                        help="Calibration dataset (used only with --learn-scales)")
    parser.add_argument("--layers-path", type=str, default="",
                        help="Override layer path (e.g. 'model.model.layers'). "
                             "Auto-detected if empty.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Check requirements ─────────────────────────────────────────────────────
    if not HAS_FLUTE:
        print("ERROR: flute-kernel not installed.")
        print("Install with one of:")
        print("  pip install flute-kernel jaxtyping")
        print("  pip install flute-kernel jaxtyping -i https://flute-ai.github.io/whl/cu124")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: FLUTE requires a CUDA GPU (compute capability 80+).")
        sys.exit(1)

    cc_major, _ = torch.cuda.get_device_capability()
    if cc_major < 8:
        print(f"WARNING: FLUTE is designed for compute capability 80+ (A100/RTX3090+).")
        print(f"  Your GPU: cc{cc_major}x. May fail.")

    # ── Banner ─────────────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  FLUTE INT{args.num_bits} Quantization")
    print("=" * 65)
    print(f"  Source model : {args.model_path}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Bits         : {args.num_bits}  (range: [0, {2**args.num_bits - 1}])")
    print(f"  Group size   : {args.group_size}")
    print(f"  Learn scales : {args.learn_scales}")
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    # ── Load model ─────────────────────────────────────────────────────────────
    print("\n[1/4] Loading FP16 model...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # ── Find layers ────────────────────────────────────────────────────────────
    if args.layers_path:
        module = model
        for attr in args.layers_path.split("."):
            module = getattr(module, attr)
        layers_name = args.layers_path
    else:
        module, layers_name = find_layers_module(model)

    n_linear = count_linear_layers(module)
    print(f"  Layers path  : {layers_name}  ({n_linear} linear layers to quantize)")

    # ── Apply FLUTE quantization ───────────────────────────────────────────────
    print(f"\n[2/4] Applying FLUTE INT{args.num_bits} quantization...")
    print(f"  Replacing Linear → FluteLinear (qweight + scales + lookup tables)...")
    t0 = time.perf_counter()

    flute.integrations.base.prepare_model_flute(
        name=layers_name,
        module=module,
        num_bits=args.num_bits,
        group_size=args.group_size,
        fake=False,
        handle_hooks=True,
    )

    torch.cuda.synchronize()
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ── Optional: learn scales ─────────────────────────────────────────────────
    if args.learn_scales:
        if not HAS_FLUTE_LEARNABLE:
            print("\n[3/4] Skipping learn-scales (flute.integrations.learnable not available)")
        else:
            print(f"\n[3/4] Learning scales on {args.n_calib} calibration samples "
                  f"({args.calib_dataset})...")
            try:
                from calibration_utils import load_calibration_data
                calib_tokens = load_calibration_data(
                    dataset_name=args.calib_dataset,
                    tokenizer=tokenizer,
                    n_samples=args.n_calib,
                    seqlen=512,
                    seed=args.seed,
                )
                # FLUTE learnable API: takes model + tokenizer, uses internal data pipeline
                # Pass custom_data if API supports it, otherwise fall back
                try:
                    flute.integrations.learnable.learn_scales(
                        model=model,
                        tokenizer=tokenizer,
                        num_bits=args.num_bits,
                        group_size=args.group_size,
                    )
                    print("  Scale learning complete.")
                except TypeError:
                    # Some versions have different signature
                    flute.integrations.learnable.learn_scales(model, tokenizer)
                    print("  Scale learning complete.")
            except Exception as e:
                print(f"  WARNING: Scale learning failed: {e}")
                print("  Continuing with default scales.")
    else:
        print("\n[3/4] Skipping learn-scales (use --learn-scales to enable)")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving to {args.output_dir}...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Patch config.json to mark quantization type (for compare_inference.py)
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["quantization_config"] = {
        "quant_type":   "flute",
        "num_bits":     args.num_bits,
        "group_size":   args.group_size,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved.")
    print("\n" + "=" * 65)
    print(f"  FLUTE INT{args.num_bits} model ready at: {args.output_dir}")
    print("=" * 65)
    print("\nNext steps:")
    print(f"  # Benchmark with compare_inference.py:")
    print(f"  python compare_inference.py \\")
    print(f"      --fp16-path  ./models/<base-model> \\")
    print(f"      --flute-path {args.output_dir} \\")
    print(f"      --flute-num-bits {args.num_bits} \\")
    print(f"      --batch-sizes 1,16")
    print(f"\n  # Or load directly with FLUTE:")
    print(f"  from flute.integrations.huggingface import from_pretrained")
    print(f"  model = from_pretrained('{args.output_dir}', device_map='auto')")


if __name__ == "__main__":
    main()
