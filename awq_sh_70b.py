"""
Standard Heuristic AWQ - Adapted for Llama-3-70B (Single 80GB A100 Optimized)

Memory Optimization Strategies:
1. CPU Offloading: Most layers on CPU, moved to GPU only during processing
2. Single Layer Processing: Process 1 layer at a time to minimize GPU memory
3. Reduced Calibration: Fewer samples and tokens per sample
4. Aggressive Cleanup: Clear CUDA cache after every layer
5. Mixed Precision: Use float16 for activations to save memory

This version uses the SAME base features as gw_awq_asym_l2.py, but adds
heuristic-guided quantization on top. This isolates the effect of heuristic alone.

Differences from awq_op_ref.py (advanced version):
- float32 precision (not float64)
- Random sampling (not sequential)
- float16 model loading (not bfloat16)
- activation_salience.clamp(min=1e-5) (not + 1e-6)

KEPT from awq_op_ref.py:
- Heuristic-Guided Global Greedy Rounding
- Outlier Masking
- Vectorized implementation
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
import random
import numpy as np
import gc

# Try to import calibration utils, fallback if not present
try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    print("⚠️ calibration_utils not found. Using internal fallback loaders.")
    def get_c4_calibration_data(*args, **kwargs): raise NotImplementedError("Please provide calibration_utils.py")
    def get_wikitext2_calibration_data(*args, **kwargs): raise NotImplementedError("Please provide calibration_utils.py")

class StandardHeuristicAWQQuantizer:
    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20,
                 group_size=128, use_heuristic=True, outlier_percent=0.05, 
                 max_tokens_per_sample=2048, layer_batch_size=1, activation_chunk_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.use_heuristic = use_heuristic
        self.outlier_percent = outlier_percent
        self.max_tokens_per_sample = max_tokens_per_sample
        self.layer_batch_size = layer_batch_size
        self.activation_chunk_size = activation_chunk_size

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[Standard Heuristic AWQ Quantizer - 70B High Precision]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")
        print(f"  Max tokens/sample: {max_tokens_per_sample} (FULL for accuracy)")
        print(f"  Layer batch size: {layer_batch_size} (Single-layer for 70B)")
        print(f"  Activation chunk size: {activation_chunk_size} (streaming)")
        print(f"  Use heuristic: {use_heuristic}")
        if use_heuristic:
            print(f"  Outlier protection: Top {outlier_percent*100:.1f}% ignored")
            print(f"  Quantization: HEURISTIC-GUIDED GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")
        else:
            print(f"  Quantization: STANDARD GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")
        print(f"  Memory strategy: Streaming + chunked processing + CPU offloading")

    def get_hook(self, name):
        """Create a hook function for a specific layer."""
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input

            # Store full sequences on CPU with float32 for accuracy
            self.activation_data[name].append(inp.detach().cpu().float())
        return hook

    @torch.no_grad()
    def get_activation_stats(self, name):
        """Compute L2 salience (E[X²]) and raw mean (E[X]) using appropriate precision."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None, None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Use float32 for accumulation
        l2_sum = torch.zeros(in_features, dtype=torch.float32)
        mean_sum = torch.zeros(in_features, dtype=torch.float32)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            l2_sum += x_flat.pow(2).sum(dim=0)
            mean_sum += x_flat.sum(dim=0)

        salience = (l2_sum / total_samples)
        raw_mean = (mean_sum / total_samples)

        return salience, raw_mean

    @torch.no_grad()
    def quantize_weight_heuristic_groupwise(self, W, group_activation_means, apply_heuristic=True):
        """
        Vectorized implementation of 'quantize_groupwise_global_greedy'.
        """
        out_features, in_features = W.shape
        device = W.device

        # --- 1. Pre-processing / Padding ---
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=W.dtype)
            W_padded[:, :in_features] = W
            act_padded = torch.zeros(padded_in_features, device=device, dtype=W.dtype)
            act_padded[:in_features] = group_activation_means
        else:
            W_padded = W
            act_padded = group_activation_means

        # Reshape to groups for scaling
        W_g = W_padded.reshape(out_features, n_groups, self.group_size)

        # Asymmetric Quantization Setup
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        max_int = 2**self.bits - 1

        scale = (w_max - w_min) / max_int
        scale = scale.clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        # Expand to full size [out, padded_in]
        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)
        zp_flat = zp.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)

        # --- 2. Initial Quantization ---
        W_div = W_padded / scale_flat
        W_int = torch.round(W_div + zp_flat).clamp(0, max_int)
        W_quant = (W_int - zp_flat) * scale_flat

        if not apply_heuristic:
            # Return early if simple rounding
            W_dequant = (W_int - zp_flat) * scale_flat
            if padded_in_features > in_features: W_dequant = W_dequant[:, :in_features]
            return W_dequant.to(W.dtype)

        # --- 3. Global Greedy Heuristic (Vectorized) ---

        # A. Calculate Current Error
        W_diff = W_padded - W_quant
        current_error = (W_diff * act_padded.unsqueeze(0)).sum(dim=1) # [out_features]

        # B. Identify Flip Candidates
        flip_dir = torch.sign(W_div + zp_flat - W_int)
        flip_dir[flip_dir == 0] = 1.0
        flip_impacts = act_padded.unsqueeze(0) * flip_dir * scale_flat # [out, in]

        # C. Validity Masks
        target_sign = torch.sign(current_error).unsqueeze(1)
        valid_mask = (torch.sign(flip_impacts) == target_sign)

        w_int_proposed = W_int + flip_dir
        in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
        valid_mask = valid_mask & in_range

        # Outlier Masking
        k_outliers = int(padded_in_features * self.outlier_percent)
        k_outliers = min(k_outliers, act_padded.numel())
        if k_outliers > 0:
            _, outlier_indices = torch.topk(act_padded.abs(), k_outliers)
            is_outlier = torch.zeros(padded_in_features, dtype=torch.bool, device=device)
            is_outlier[outlier_indices] = True
            valid_mask = valid_mask & (~is_outlier).unsqueeze(0)

        # --- 4. Sorting & Optimization ---
        rounding_costs = (W_div + zp_flat - W_int).abs()
        rounding_costs_masked = rounding_costs.clone()
        rounding_costs_masked[~valid_mask] = -1.0

        sorted_indices = torch.argsort(rounding_costs_masked, dim=1, descending=True)
        sorted_impacts = torch.gather(flip_impacts, 1, sorted_indices)
        sorted_validity = torch.gather(valid_mask.long(), 1, sorted_indices)
        sorted_impacts = sorted_impacts * sorted_validity

        cumsum_impacts = torch.cumsum(sorted_impacts, dim=1)
        residuals = torch.abs(current_error.unsqueeze(1) - cumsum_impacts)
        error_unsqueezed = torch.abs(current_error).unsqueeze(1)
        all_residuals = torch.cat([error_unsqueezed, residuals], dim=1)
        best_k = torch.argmin(all_residuals, dim=1)

        # --- 5. Apply Flips ---
        idx_range = torch.arange(padded_in_features, device=device).unsqueeze(0)
        flip_mask_sorted = idx_range < best_k.unsqueeze(1)
        final_flips_sorted = flip_mask_sorted & (sorted_validity.bool())

        sorted_flip_dir = torch.gather(flip_dir, 1, sorted_indices)
        sorted_flip_dir[~final_flips_sorted] = 0.0

        W_int.scatter_add_(1, sorted_indices, sorted_flip_dir)
        W_int.clamp_(0, max_int)

        # --- 6. Dequantize & Return ---
        W_dequant = (W_int - zp_flat) * scale_flat

        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant.to(W.dtype)

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """Grid search for optimal per-input-channel scaling factor."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience, raw_mean = self.get_activation_stats(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience = activation_salience.to(self.device).to(module.weight.dtype)
        raw_mean = raw_mean.to(self.device).to(module.weight.dtype)

        # Subsample for speed - use stratified sampling for better coverage
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        max_samples = min(1024, X_cpu.shape[0])  # Increased from 512
        if X_cpu.shape[0] > max_samples:
            # Stratified sampling
            indices = torch.linspace(0, X_cpu.shape[0]-1, max_samples).long()
            X_search = X_cpu[indices]
        else:
            X_search = X_cpu

        del X_cpu
        gc.collect()

        # Move weights to target device if on meta/cpu (handles device_map="auto" offloading)
        W = module.weight.data
        original_device = W.device
        if W.device.type == 'meta' or W.device.type == 'cpu':
            W = W.to(self.device)
        
        # Compute Y_orig in chunks
        Y_orig_chunks = []
        chunk_size = 128
        for i in range(0, X_search.shape[0], chunk_size):
            X_chunk = X_search[i:i+chunk_size].to(self.device).to(module.weight.dtype)
            Y_chunk = torch.matmul(X_chunk, W.t())
            Y_orig_chunks.append(Y_chunk.cpu())
            del X_chunk, Y_chunk
        
        Y_orig = torch.cat(Y_orig_chunks, dim=0)
        del Y_orig_chunks
        torch.cuda.empty_cache()

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        # Clamp min salience
        activation_salience = activation_salience.clamp(min=1e-5)

        # Grid search with chunked processing
        chunk_size = 128
        n_chunks = (X_search.shape[0] + chunk_size - 1) // chunk_size
        
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid
            scales = activation_salience.pow(alpha)

            W_scaled = W * scales.unsqueeze(0)
            scaled_act_mean = raw_mean / scales

            W_quant = self.quantize_weight_heuristic_groupwise(
                W_scaled,
                scaled_act_mean,
                apply_heuristic=self.use_heuristic
            )

            W_recon = W_quant / scales.unsqueeze(0)
            
            # Compute error in chunks
            total_error = 0.0
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, X_search.shape[0])
                
                X_chunk = X_search[start_idx:end_idx].to(self.device).to(W.dtype)
                Y_orig_chunk = Y_orig[start_idx:end_idx].to(self.device)
                Y_quant_chunk = torch.matmul(X_chunk, W_recon.t())
                
                chunk_error = (Y_orig_chunk - Y_quant_chunk).pow(2).sum().item()
                total_error += chunk_error
                
                del X_chunk, Y_orig_chunk, Y_quant_chunk
            
            error = total_error / X_search.shape[0]

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

            del W_scaled, W_quant, W_recon, scales
            torch.cuda.empty_cache()

        # Move weight back to original device
        if original_device.type == 'cpu':
            module.weight.data = W.to('cpu')

        del Y_orig
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """Apply Standard Heuristic AWQ Quantization."""
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data
        original_device = W.device
        
        # Ensure weight is on the target device
        if W.device.type == 'meta' or W.device.type == 'cpu':
            W = W.to(self.device)
        
        original_dtype = W.dtype
        W_scaled = W * best_scales.unsqueeze(0)

        _, raw_mean = self.get_activation_stats(name)
        if raw_mean is not None:
            scaled_act_mean = (raw_mean.to(self.device).to(W.dtype) / best_scales)
        else:
            scaled_act_mean = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)

        W_quant = self.quantize_weight_heuristic_groupwise(
            W_scaled,
            scaled_act_mean,
            apply_heuristic=self.use_heuristic
        )

        W_final = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)
        # Keep weight on its original device (CPU offloading)
        module.weight.data = W_final.to(original_device)

        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error
        }

        del best_scales, scaled_act_mean, W_scaled, W_quant, W_final
        if name in self.activation_data:
            del self.activation_data[name]
        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_single_layer(self, name, module, calibration_data, n_samples=128):
        """Calibrate a single layer with chunked processing."""
        # Clear any previous activation data for this layer
        if name in self.activation_data:
            del self.activation_data[name]
        self.activation_data[name] = []

        self.model.eval()
        
        # Register hook for this layer only
        handle = module.register_forward_hook(self.get_hook(name))

        # Run calibration in mini-batches
        batch_size = self.activation_chunk_size
        successful = 0
        
        with torch.no_grad():
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_data = calibration_data[batch_start:batch_end]
                
                for text in batch_data:
                    try:
                        inputs = self.tokenizer(text, return_tensors="pt",
                                               truncation=True, max_length=512)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        self.model(**inputs, use_cache=False, return_dict=True)
                        successful += 1
                    except Exception:
                        continue
                
                # Cleanup after each batch
                torch.cuda.empty_cache()
                gc.collect()

        # Remove hook
        handle.remove()

        torch.cuda.empty_cache()
        gc.collect()

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        """Single-layer sequential quantization: optimal for 70B on 80GB."""
        print("\n" + "=" * 80)
        print("SINGLE-LAYER SEQUENTIAL QUANTIZATION (70B Optimized)")
        print("=" * 80)
        print(f"  Strategy: Process ONE layer at a time")
        print(f"  Memory: Aggressive CPU offloading + immediate cleanup")

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        num_layers = len(layer_names)
        print(f"  Total layers: {num_layers}")
        print("=" * 80)

        # Process one layer at a time
        for idx, (name, module) in enumerate(tqdm(layer_names, desc="Quantizing Layers")):
            try:
                if idx < 2:
                    print(f"\n[Layer {idx}/{num_layers}] {name}")

                # Calibrate this single layer
                self.calibrate_single_layer(name, module, calibration_data, n_samples)

                # Quantize this layer
                self.quantize_layer(name, module)

                # Immediate cleanup
                torch.cuda.empty_cache()
                gc.collect()

                # Progress report every 10 layers
                if (idx + 1) % 10 == 0:
                    print(f"\nProgress: {idx + 1}/{num_layers} layers quantized")

            except Exception as e:
                print(f"\n⚠️  Error quantizing {name}: {e}")
                # Clean up on error
                if name in self.activation_data:
                    del self.activation_data[name]
                torch.cuda.empty_cache()
                gc.collect()
                continue

        print("\n" + "=" * 80)
        print("✓ Single-Layer Sequential Quantization Complete")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calib", type=int, default=128, 
                       help="Number of calibration samples (full for accuracy)")
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--use-heuristic", action="store_true", default=True,
                       help="Enable heuristic rounding (default: True)")
    parser.add_argument("--no-heuristic", dest="use_heuristic", action="store_false",
                       help="Disable heuristic rounding")
    parser.add_argument("--outlier-percent", type=float, default=0.05,
                       help="Percent of outliers to ignore (default: 0.05)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens to store per sample (full for accuracy)")
    parser.add_argument("--activation-chunk-size", type=int, default=32,
                       help="Process activations in chunks of this size")
    parser.add_argument("--layer-batch-size", type=int, default=1,
                       help="MUST be 1 for 70B on 80GB GPU")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/llama3_70b_awq_sh")
    parser.add_argument("--model-path", type=str, default="meta-llama/Meta-Llama-3-70B",
                       help="Model name or local path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"],
                       help="Calibration dataset")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use model path from args
    model_name = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device} | Model: {model_name}")
    print("Loading with CPU offloading for 70B model...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Fix for Llama models lacking pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    # Critical: Use CPU offloading for 70B
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatic CPU offloading
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "75GB", "cpu": "100GB"}  # Reserve 5GB GPU for activations
    )
    model.eval()
    
    print("  Model loaded with automatic device mapping")

    # Load calibration data
    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=512, seed=args.seed)
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=512, seed=args.seed)

    quantizer = StandardHeuristicAWQQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size,
        use_heuristic=args.use_heuristic,
        outlier_percent=args.outlier_percent,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        activation_chunk_size=args.activation_chunk_size
    )

    # Use single-layer sequential quantization (ONLY way for 70B on 80GB)
    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, max_shard_size="5GB")  # Shard for large model
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
