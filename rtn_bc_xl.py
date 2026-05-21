"""
RTN + Bias Correction (XL)

Group-wise asymmetric RTN baseline plus post-quantization bias correction.

For each Linear layer:
    1. collect E[X] from calibration inputs,
    2. apply the same group-wise asymmetric RTN used by rtn.py,
    3. add bias delta:

           bias += (W_orig - W_rtn) @ E[X]

This is intended to be run directly from the FPRAG repo root.
"""

import argparse
import gc
import hashlib
import os
import random

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    def get_c4_calibration_data(*_args, **_kwargs):
        raise NotImplementedError("calibration_utils.py missing")

    def get_wikitext2_calibration_data(*_args, **_kwargs):
        raise NotImplementedError("calibration_utils.py missing")


def make_bias_checkpoint_reloadable(model):
    """Set config flags needed for HF models to recreate Linear bias modules."""
    config = getattr(model, "config", None)
    if config is None:
        return [], []

    linear_with_bias = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.bias is not None
    ]
    changed = []
    warnings = []

    has_attention_bias = any(".self_attn." in name for name in linear_with_bias)
    has_mlp_bias = any(".mlp." in name for name in linear_with_bias)

    if has_attention_bias:
        if hasattr(config, "attention_bias"):
            if not bool(getattr(config, "attention_bias")):
                setattr(config, "attention_bias", True)
                changed.append("attention_bias=True")
        else:
            warnings.append("config has no attention_bias flag; attention BC biases may be ignored on reload")

    if has_mlp_bias:
        if hasattr(config, "mlp_bias"):
            if not bool(getattr(config, "mlp_bias")):
                setattr(config, "mlp_bias", True)
                changed.append("mlp_bias=True")
        else:
            warnings.append("config has no mlp_bias flag; MLP BC biases may be ignored on reload")

    return changed, warnings


def validate_bias_reload_config(output_dir, expected_attention_bias, expected_mlp_bias):
    config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    errors = []
    if expected_attention_bias and hasattr(config, "attention_bias") and not bool(config.attention_bias):
        errors.append("attention_bias is not true in saved config")
    if expected_mlp_bias and hasattr(config, "mlp_bias") and not bool(config.mlp_bias):
        errors.append("mlp_bias is not true in saved config")
    if errors:
        raise RuntimeError("; ".join(errors))


class RTNBiasCorrectionXL:
    def __init__(
        self,
        model,
        tokenizer,
        device="cuda",
        bits=4,
        group_size=128,
        skip_lm_head=True,
        layer_batch_size=16,
        max_tokens_per_sample=2048,
        bias_chunk_size=4096,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.group_size = group_size
        self.skip_lm_head = skip_lm_head
        self.layer_batch_size = layer_batch_size
        self.max_tokens_per_sample = max_tokens_per_sample
        self.bias_chunk_size = bias_chunk_size

        self.activation_sums = {}
        self.activation_counts = {}
        self.activation_means = {}
        self.bc_norms = {}

        print("\n[RTN + Bias Correction XL Initialized]")
        print(f"  Target bits:       {bits}")
        print(f"  Group size:        {group_size}")
        print(f"  Skip lm_head:      {skip_lm_head}")
        print(f"  Layer batch size:  {layer_batch_size}")
        print(f"  Tokens per sample: {max_tokens_per_sample}")
        print(f"  Bias chunk size:   {bias_chunk_size}")
        print(f"  Quantization:      GROUP-WISE ASYMMETRIC [0, {2 ** bits - 1}]")

    def get_hook(self, name):
        def hook(_module, inputs, _output):
            inp = inputs[0] if isinstance(inputs, tuple) else inputs
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                idx = torch.randperm(inp.shape[1], device=inp.device)[: self.max_tokens_per_sample]
                inp = inp[:, idx.sort()[0], :]
            flat = inp.detach().reshape(-1, inp.shape[-1]).float()
            cur_sum = flat.sum(dim=0).cpu()
            cur_count = flat.shape[0]
            if name not in self.activation_sums:
                self.activation_sums[name] = cur_sum
                self.activation_counts[name] = cur_count
            else:
                self.activation_sums[name] += cur_sum
                self.activation_counts[name] += cur_count
        return hook

    def get_activation_mean(self, name, in_features):
        if name in self.activation_means:
            mean = self.activation_means[name]
            if mean.numel() == in_features:
                return mean
        if name not in self.activation_sums:
            return None
        count = self.activation_counts.get(name, 0)
        if count <= 0:
            return None
        mean = self.activation_sums[name] / count
        if mean.numel() != in_features:
            return None
        return mean.cpu().float()

    @torch.no_grad()
    def calibrate_layer_batch(self, layer_batch, calibration_texts, n_samples):
        self.activation_sums = {}
        self.activation_counts = {}
        self.activation_means = {}
        handles = [module.register_forward_hook(self.get_hook(name)) for name, module in layer_batch]

        successful = 0
        with torch.no_grad():
            for i, text in enumerate(tqdm(calibration_texts[:n_samples], desc="  Calibration", leave=False)):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1
                    if (i + 1) % 32 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    continue

        for handle in handles:
            handle.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if successful == 0:
            print("  Warning: no successful calibration passes in this batch.")

    def load_calibration_batch_cache(self, cache_path, expected_names):
        if not cache_path or not os.path.isfile(cache_path):
            return False
        try:
            payload = torch.load(cache_path, map_location="cpu")
            means = payload.get("activation_means", payload)
            if not isinstance(means, dict):
                return False
            missing = [name for name in expected_names if name not in means]
            if missing:
                print(f"  Calibration cache incomplete ({len(missing)} missing); recalibrating.")
                return False
            self.activation_sums = {}
            self.activation_counts = {}
            self.activation_means = {name: means[name].cpu().float() for name in expected_names}
            print(f"  Calibration cache hit: {cache_path}")
            return True
        except Exception as exc:
            print(f"  Calibration cache unreadable ({exc}); recalibrating.")
            return False

    def save_calibration_batch_cache(self, cache_path, batch):
        if not cache_path:
            return
        means = {}
        for name, module in batch:
            mean = self.get_activation_mean(name, module.weight.data.shape[1])
            if mean is not None:
                means[name] = mean.cpu().float()
        if not means:
            return
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = cache_path + ".tmp"
        torch.save({"activation_means": means}, tmp_path)
        os.replace(tmp_path, cache_path)
        self.activation_means = means
        print(f"  Saved calibration cache: {cache_path}")

    @torch.no_grad()
    def quantize_weight_groupwise_asymmetric(self, weight):
        out_features, in_features = weight.shape
        dtype = weight.dtype
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in = n_groups * self.group_size

        if padded_in > in_features:
            weight_padded = torch.zeros(out_features, padded_in, device=weight.device, dtype=dtype)
            weight_padded[:, :in_features] = weight
        else:
            weight_padded = weight

        weight_groups = weight_padded.reshape(out_features, n_groups, self.group_size)
        w_min = weight_groups.min(dim=2, keepdim=True)[0]
        w_max = weight_groups.max(dim=2, keepdim=True)[0]

        max_int = 2 ** self.bits - 1
        scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
        zero_point = torch.round(-w_min / scale).clamp(0, max_int)

        weight_int = torch.round(weight_groups / scale + zero_point).clamp(0, max_int)
        weight_dq = (weight_int - zero_point) * scale
        weight_dq = weight_dq.reshape(out_features, padded_in)
        if padded_in > in_features:
            weight_dq = weight_dq[:, :in_features]
        return weight_dq.to(dtype)

    @torch.no_grad()
    def compute_bias_delta(self, weight_orig, weight_quant, activation_mean):
        deltas = []
        x_mean = activation_mean.to(weight_orig.device).float()
        for start in range(0, weight_orig.shape[0], self.bias_chunk_size):
            end = min(start + self.bias_chunk_size, weight_orig.shape[0])
            diff = weight_orig[start:end].float() - weight_quant[start:end].float()
            deltas.append(torch.matmul(diff, x_mean))
            del diff
        return torch.cat(deltas, dim=0)

    @torch.no_grad()
    def quantize_layer(self, name, module):
        weight_orig = module.weight.data.clone()
        weight_quant = self.quantize_weight_groupwise_asymmetric(module.weight.data)

        activation_mean = self.get_activation_mean(name, module.weight.data.shape[1])
        if activation_mean is None:
            bias_delta = torch.zeros(module.weight.data.shape[0], device=module.weight.data.device, dtype=torch.float32)
        else:
            bias_delta = self.compute_bias_delta(weight_orig, weight_quant, activation_mean)

        module.weight.data = weight_quant
        if module.bias is None:
            module.bias = nn.Parameter(bias_delta.to(device=module.weight.device, dtype=module.weight.dtype))
        else:
            module.bias.data = module.bias.data + bias_delta.to(device=module.bias.device, dtype=module.bias.dtype)

        self.bc_norms[name] = float(bias_delta.norm().item())
        del weight_orig, weight_quant, bias_delta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def quantize_model(self, calibration_texts, n_samples=128, calib_cache_dir=None, calib_cache_key=None):
        print("\n" + "=" * 80)
        print("Batched Sequential RTN + Bias Correction")
        print("=" * 80)

        all_layers = [(name, module) for name, module in self.model.named_modules() if isinstance(module, nn.Linear)]
        layers_to_q = []
        for name, module in all_layers:
            is_lmhead = ("lm_head" in name.lower()) or name.endswith("lm_head")
            if is_lmhead and self.skip_lm_head:
                continue
            layers_to_q.append((name, module))

        n_total = len(layers_to_q)
        n_batches = (n_total + self.layer_batch_size - 1) // self.layer_batch_size
        print(f"  Total Linear: {len(all_layers)} | to quantize: {n_total} | skipped: {len(all_layers) - n_total}")
        print(f"  Batches: {n_batches} (batch size = {self.layer_batch_size})")

        quantized = 0
        for batch_idx in range(n_batches):
            start = batch_idx * self.layer_batch_size
            end = min(start + self.layer_batch_size, n_total)
            batch = layers_to_q[start:end]
            batch_names = [name for name, _module in batch]
            cache_path = None
            if calib_cache_dir and calib_cache_key:
                cache_path = os.path.join(
                    calib_cache_dir,
                    f"{calib_cache_key}_batch{batch_idx:03d}_{start:04d}_{end - 1:04d}.pt",
                )

            print(f"\n[Batch {batch_idx + 1}/{n_batches}] Layers {start}-{end - 1}")
            if not self.load_calibration_batch_cache(cache_path, batch_names):
                self.calibrate_layer_batch(batch, calibration_texts, n_samples)
                self.save_calibration_batch_cache(cache_path, batch)

            for name, module in tqdm(batch, desc="  Quantize+BC", leave=False):
                try:
                    self.quantize_layer(name, module)
                    quantized += 1
                except Exception as exc:
                    print(f"\n  Warning: error on {name}: {exc}")

            self.activation_sums = {}
            self.activation_counts = {}
            self.activation_means = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if HAS_PSUTIL:
                print(f"  RAM after batch {batch_idx + 1}: {psutil.virtual_memory().percent:.1f}%")

        print("\n" + "=" * 80)
        print(f"Done: quantized+BC {quantized}/{n_total} layers")
        if self.bc_norms:
            norms = list(self.bc_norms.values())
            print(f"Bias delta norm: mean={np.mean(norms):.4f}, median={np.median(norms):.4f}, max={np.max(norms):.4f}")
        print("=" * 80)


def load_wikitext2_simple(n_samples=128):
    print("Loading WikiText-2 (simple)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 100]
    return texts[:n_samples]


def make_calibration_cache_key(args):
    model_id = os.path.abspath(args.model_path) if os.path.exists(args.model_path) else args.model_path
    model_hash = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:12]
    model_name = os.path.basename(str(args.model_path).rstrip("/")).replace(".", "p").replace("-", "_")
    return (
        f"{model_name}_{model_hash}"
        f"_dataset{args.calib_dataset}"
        f"_n{args.n_calib}"
        f"_seq{args.max_tokens_per_sample}"
        f"_seed{args.seed}"
        f"_skiplm{int(args.skip_lm_head)}"
        f"_lbs{args.layer_batch_size}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="RTN + bias correction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--calib-dataset", type=str, default="c4", choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache")
    parser.add_argument("--no-calib-cache", action="store_true")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048)
    parser.add_argument("--layer-batch-size", type=int, default=16)
    parser.add_argument("--bias-chunk-size", type=int, default=4096)
    parser.add_argument("--skip-lm-head", action="store_true", default=True)
    parser.add_argument("--quantize-lm-head", dest="skip_lm_head", action="store_false")
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/model_rtn_bc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float32
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("=" * 80)
    print("RTN + Bias Correction")
    print(f"Target Model: {args.model_path}")
    print(f"Model dtype: {model_dtype}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calibration_texts = get_c4_calibration_data(
            tokenizer,
            n_samples=args.n_calib,
            seqlen=2048,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )
    elif args.calib_dataset == "wikitext2-simple":
        calibration_texts = load_wikitext2_simple(n_samples=args.n_calib)
    else:
        calibration_texts = get_wikitext2_calibration_data(
            tokenizer,
            n_samples=args.n_calib,
            seqlen=2048,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )

    quantizer = RTNBiasCorrectionXL(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        group_size=args.group_size,
        skip_lm_head=args.skip_lm_head,
        layer_batch_size=args.layer_batch_size,
        max_tokens_per_sample=args.max_tokens_per_sample,
        bias_chunk_size=args.bias_chunk_size,
    )

    calib_cache_dir = None
    calib_cache_key = None
    if not args.no_calib_cache:
        calib_cache_dir = os.path.join(args.cache_dir, "rtn_bc_activation_means")
        calib_cache_key = make_calibration_cache_key(args)
        print(f"Calibration mean cache: {calib_cache_dir}/{calib_cache_key}_batch*.pt")

    quantizer.quantize_model(
        calibration_texts,
        n_samples=args.n_calib,
        calib_cache_dir=calib_cache_dir,
        calib_cache_key=calib_cache_key,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    expected_attention_bias = any(
        isinstance(module, nn.Linear) and module.bias is not None and ".self_attn." in name
        for name, module in model.named_modules()
    )
    expected_mlp_bias = any(
        isinstance(module, nn.Linear) and module.bias is not None and ".mlp." in name
        for name, module in model.named_modules()
    )
    config_changes, config_warnings = make_bias_checkpoint_reloadable(model)
    if config_changes:
        print("Enabled reloadable BC bias config: " + ", ".join(config_changes))
    for warning in config_warnings:
        print(f"Warning: {warning}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    validate_bias_reload_config(args.output_dir, expected_attention_bias, expected_mlp_bias)
    print(f"\nSaved RTN+BC model to {args.output_dir}")


if __name__ == "__main__":
    main()
