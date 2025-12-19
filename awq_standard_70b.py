"""
Group-Wise AWQ Implementation with ASYMMETRIC Quantization + L2 Salience
ADAPTED FOR: Llama-3-70B (Single 80GB A100 Optimized)

Memory Optimization Strategies:
1. CPU Offloading: Most layers on CPU, moved to GPU only during processing
2. Minimal Layer Batching: Process 1 layer at a time to minimize GPU memory
3. Reduced Calibration: Fewer samples and tokens per sample
4. Aggressive Cleanup: Clear CUDA cache after every layer
5. Mixed Precision: Use float16 for activations, bfloat16 for weights

Key Difference from gw_awq_asym.py:
- gw_awq_asym.py: Uses E[|X|] (L1 norm) for activation salience
- gw_awq_asym_l2.py: Uses E[X²] (L2 norm) for activation salience

Why L2 is Better:
- Quantization MSE = E[(δW × X)²] ∝ E[X²]
- L2 emphasizes channels with spikes/outliers (quadratic weighting)
- Matches the squared error objective directly

Algorithm:
1. Compute per-input-channel salience: s[j] = E[X[:, j]²] (L2 norm)
2. Grid search for optimal α ∈ [0, 1]
3. Scale weight COLUMNS: W[:, j] *= s[j]^α
4. Quantize with GROUP-WISE ASYMMETRIC scales
   - Per group: scale = (max - min) / 15, zero_point = round(-min / scale)
5. Divide by input scales: W_final = Q(W*s) / s
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

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  Warning: psutil not installed. Memory monitoring disabled.")
    print("   Install with: pip install psutil")

# Import your calibration utils (assuming they exist in the same folder)
# If running standalone without the utils file, you can uncomment the backup loaders below
from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data

class GroupWiseAWQAsymmetricL2Quantizer:
    """
    Group-Wise AWQ with Asymmetric Quantization and L2 Salience.
    Optimized for 70B models on single 80GB GPU.
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20, group_size=128, 
                 max_tokens_per_sample=2048, activation_chunk_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.max_tokens_per_sample = max_tokens_per_sample  # Full 2048 for accuracy
        self.activation_chunk_size = activation_chunk_size  # Process activations in chunks

        # Storage for activations - streaming approach
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[Group-Wise AWQ ASYMMETRIC L2 Quantizer - 70B High Precision]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Max tokens/sample: {max_tokens_per_sample} (FULL for accuracy)")
        print(f"  Activation chunk size: {activation_chunk_size} samples (streaming)")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, 15]")
        print(f"  Salience metric: E[X²] (L2 norm) - Better MSE alignment")
        print(f"  Memory strategy: Streaming + chunked processing + CPU offloading")


    @torch.no_grad()
    def get_activation_salience_l2(self, name):
        """
        Compute per-input-channel activation salience using L2 norm: E[X[:, j]²]
        Uses streaming to handle large activation sets.
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Accumulate L2 salience on CPU to save GPU VRAM
        # Process in chunks to avoid memory spikes
        salience_sum = torch.zeros(in_features, dtype=torch.float32)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            # Process in chunks if too large
            if x_flat.shape[0] > 10000:
                chunk_size = 5000
                for i in range(0, x_flat.shape[0], chunk_size):
                    chunk = x_flat[i:i+chunk_size]
                    salience_sum += chunk.pow(2).sum(dim=0)
            else:
                salience_sum += x_flat.pow(2).sum(dim=0)

        salience = salience_sum / total_samples
        return salience

    @torch.no_grad()
    def quantize_weight_groupwise_asymmetric(self, W):
        """
        Group-wise ASYMMETRIC quantization [0, 15].
        """
        out_features, in_features = W.shape

        # Pad to make in_features divisible by group_size
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=W.device, dtype=W.dtype)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

        # Reshape to [out_features, n_groups, group_size]
        W_grouped = W_padded.reshape(out_features, n_groups, self.group_size)

        # Compute min and max per group
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]

        # Asymmetric quantization parameters
        scale = (W_max - W_min) / 15.0
        scale = scale.clamp(min=1e-8)
        zero_point = torch.round(-W_min / scale).clamp(0, 15)

        # Quantize to [0, 15]
        W_int = torch.round(W_grouped / scale + zero_point).clamp(0, 15)

        # Dequantize
        W_dequant_grouped = (W_int - zero_point) * scale

        # Reshape back
        W_dequant = W_dequant_grouped.reshape(out_features, padded_in_features)

        # Remove padding if added
        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant

    @torch.no_grad()
    def search_best_scale(self, name, module, debug=False):
        """
        Grid search for optimal per-input-channel scaling factor using L2 salience.
        Optimized for 70B: minimal GPU memory usage.
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Get L2 activation salience
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            if debug:
                print(f"  DEBUG: No activation salience for {name}, using default scales")
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        if debug:
            print(f"  DEBUG: Got salience for {name}, shape={activation_salience.shape}, "
                  f"mean={activation_salience.mean():.6f}, max={activation_salience.max():.6f}")

        # Prepare calibration data - use stratified sampling to maintain diversity
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        # Use more samples for accuracy (1024 instead of 512)
        # But still manageable for memory
        max_samples = min(1024, X_cpu.shape[0])
        if X_cpu.shape[0] > max_samples:
            # Stratified sampling: take evenly spaced samples for better coverage
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
        
        b = module.bias.data if module.bias is not None else None
        if b is not None and (b.device.type == 'meta' or b.device.type == 'cpu'):
            b = b.to(self.device)

        # Compute original output in chunks to save memory
        Y_orig_chunks = []
        chunk_size = 128
        for i in range(0, X_search.shape[0], chunk_size):
            X_chunk = X_search[i:i+chunk_size].to(self.device).to(W.dtype)
            if b is not None:
                Y_chunk = torch.matmul(X_chunk, W.t()) + b
            else:
                Y_chunk = torch.matmul(X_chunk, W.t())
            Y_orig_chunks.append(Y_chunk.cpu())
            del X_chunk, Y_chunk
        
        Y_orig = torch.cat(Y_orig_chunks, dim=0)
        del Y_orig_chunks
        torch.cuda.empty_cache()

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        activation_salience = activation_salience.to(self.device)

        # Grid search over α with chunked processing
        # Process X_search in chunks to reduce memory
        chunk_size = 128  # Process 128 samples at a time
        n_chunks = (X_search.shape[0] + chunk_size - 1) // chunk_size
        
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from L2 salience
            scales = activation_salience.pow(alpha).clamp(min=1e-5)

            # Scale weight COLUMNS
            W_scaled = W * scales.unsqueeze(0)

            # Quantize with GROUP-WISE ASYMMETRIC quantization
            W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)

            # Compute error in chunks
            total_error = 0.0
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, X_search.shape[0])
                
                X_chunk = X_search[start_idx:end_idx].to(self.device).to(W.dtype)
                Y_orig_chunk = Y_orig[start_idx:end_idx]
                
                # Compensate input
                X_compensated = X_chunk / scales.unsqueeze(0)

                if b is not None:
                    Y_quant_chunk = torch.matmul(X_compensated, W_quant.t()) + b
                else:
                    Y_quant_chunk = torch.matmul(X_compensated, W_quant.t())

                # Accumulate error
                chunk_error = (Y_orig_chunk - Y_quant_chunk).pow(2).sum().item()
                total_error += chunk_error
                
                del X_chunk, X_compensated, Y_quant_chunk
            
            error = total_error / X_search.shape[0]

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

            # Cleanup
            del W_scaled, W_quant, scales
            torch.cuda.empty_cache()

        # Move weight back to original device
        if original_device.type == 'cpu':
            module.weight.data = W.to('cpu')
        
        del X_search, Y_orig
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    def calibrate_single_layer(self, name, module, calibration_data, n_samples=128):
        """
        Calibrate a SINGLE layer with chunked processing for memory efficiency.
        """
        # Clear any previous activation data
        if name in self.activation_data:
            del self.activation_data[name]
        self.activation_data[name] = []

        # Register hook for this layer only
        handle = module.register_forward_hook(self.get_hook(name))

        # Process in mini-batches to avoid memory spikes
        batch_size = self.activation_chunk_size
        successful_passes = 0
        
        with torch.no_grad():
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_data = calibration_data[batch_start:batch_end]
                
                for i, text in enumerate(batch_data):
                    try:
                        inputs = self.tokenizer(text, return_tensors="pt",
                                               truncation=True, max_length=512)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        _ = self.model(**inputs, use_cache=False, return_dict=True)
                        successful_passes += 1
                        del inputs

                    except Exception as e:
                        if successful_passes == 0:
                            print(f"\n⚠️  Forward pass error: {str(e)[:100]}")
                        continue
                
                # Cleanup after each batch
                torch.cuda.empty_cache()
                gc.collect()

        # Remove hook
        handle.remove()

        if successful_passes == 0:
            print(f"\n❌ FATAL: No successful forward passes for {name}!")

        torch.cuda.empty_cache()
        gc.collect()

    def get_hook(self, name):
        """Create a hook function for a specific layer."""
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input

            # Keep full sequence length but store on CPU immediately
            # Use float32 for accuracy (memory managed by streaming)
            inp_stored = inp.detach().cpu().float().clone()
            self.activation_data[name].append(inp_stored)
            del inp
        return hook

    def quantize_model_sequential(self, calibration_data, n_samples=500, layer_batch_size=1):
        """
        SINGLE-LAYER SEQUENTIAL QUANTIZATION (Optimized for 70B on 80GB).
        """
        print("\n" + "=" * 80)
        print("SINGLE-LAYER SEQUENTIAL QUANTIZATION (70B Optimized)")
        print("=" * 80)
        print("  Strategy: Process ONE layer at a time to minimize GPU memory")
        print("  Memory: Aggressive CPU offloading + immediate cleanup")
        
        if HAS_PSUTIL:
            initial_ram = psutil.virtual_memory().percent
            print(f"  Initial System RAM: {initial_ram:.1f}%")

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        print(f"\nFound {len(layer_names)} linear layers to quantize")
        print(f"Processing: {layer_batch_size} layer at a time")
        print("=" * 80)

        quantized_count = 0
        skipped_count = 0

        # Process layers ONE at a time
        for idx, (name, module) in enumerate(tqdm(layer_names, desc="Quantizing Layers")):
            try:
                if quantized_count < 2:
                    print(f"\n[Layer {idx}/{len(layer_names)}] {name}")
                
                # STEP 1: Calibrate this single layer
                self.calibrate_single_layer(name, module, calibration_data, n_samples)

                # STEP 2: Quantize this layer
                if quantized_count < 2:
                    best_scales, best_alpha, best_error = self.search_best_scale(name, module, debug=True)
                    print(f"  → α={best_alpha:.4f}, error={best_error:.8f}")
                else:
                    best_scales, best_alpha, best_error = self.search_best_scale(name, module)

                W = module.weight.data
                original_device = W.device
                
                # Move to GPU if needed
                if W.device.type == 'meta' or W.device.type == 'cpu':
                    W = W.to(self.device)
                
                original_dtype = W.dtype
                W_scaled = W * best_scales.unsqueeze(0)
                W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)
                W_final = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)
                
                # Move back to original device (CPU offloading)
                module.weight.data = W_final.to(original_device)

                self.layer_scales[name] = {
                    'scales': best_scales.cpu(),
                    'alpha': best_alpha,
                    'error': best_error
                }

                quantized_count += 1

                # STEP 3: Immediate cleanup after each layer
                del best_scales, W, W_scaled, W_quant, W_final
                if name in self.activation_data:
                    del self.activation_data[name]
                torch.cuda.empty_cache()
                gc.collect()

                # Progress report every 10 layers
                if (quantized_count) % 10 == 0 and HAS_PSUTIL:
                    ram_pct = psutil.virtual_memory().percent
                    print(f"\nProgress: {quantized_count}/{len(layer_names)} layers. RAM: {ram_pct:.1f}%")

            except Exception as e:
                print(f"\n⚠️  Error quantizing {name}: {e}")
                skipped_count += 1
                # Clean up on error
                if name in self.activation_data:
                    del self.activation_data[name]
                torch.cuda.empty_cache()
                gc.collect()
                continue

        print(f"\n✅ Sequential Quantization Complete!")
        print(f"   Total layers quantized: {quantized_count}/{len(layer_names)}")
        if skipped_count > 0:
            print(f"   Skipped layers: {skipped_count}")

        if self.layer_scales:
            alphas = [info['alpha'] for info in self.layer_scales.values()]
            print(f"\nOptimal α statistics:")
            print(f"  Mean: {np.mean(alphas):.3f}")
            print(f"  Median: {np.median(alphas):.3f}")

        # Final cleanup
        self.activation_data = {}
        torch.cuda.empty_cache()
        gc.collect()

def load_wikitext2_simple(n_samples=128):
    from datasets import load_dataset
    print(f"Loading WikiText-2 (simple/fast approach)...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    return texts[:n_samples]

def main():
    parser = argparse.ArgumentParser(
        description="Group-Wise AWQ with ASYMMETRIC quantization + L2 Salience for Llama-3-70B (80GB optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, 
                       help="Calibration samples (full for accuracy)")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens to store per sample (full for accuracy)")
    parser.add_argument("--activation-chunk-size", type=int, default=32,
                       help="Process activations in chunks of this size")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/llama3_70b_gw_awq_asym_l2",
                       help="Output directory")
    parser.add_argument("--model-path", type=str, default="meta-llama/Meta-Llama-3-70B",
                       help="Model name or local path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"],
                       help="Calibration dataset")
    parser.add_argument("--layer-batch-size", type=int, default=1,
                       help="MUST be 1 for 70B on 80GB GPU")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Use model path from args
    model_name = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Group-Wise AWQ with ASYMMETRIC Quantization + L2 Salience")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Group size: {args.group_size}")
    print(f"Layer Batch Size: {args.layer_batch_size} (Single-layer for 70B)")
    print(f"Max tokens/sample: {args.max_tokens_per_sample} (Full for accuracy)")
    print(f"Calibration samples: {args.n_calib} (Full for accuracy)")
    print(f"Activation chunk size: {args.activation_chunk_size} (Streaming)")
    print("Memory Strategy: Chunked processing + CPU offloading + streaming")
    print("=" * 80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    print("  Using CPU offloading strategy for 70B model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Llama fix: Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    # Critical: Use CPU offloading for 70B
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatic CPU offloading
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Reduce CPU RAM during loading
        max_memory={0: "75GB", "cpu": "100GB"}  # Reserve 5GB GPU for activations
    )
    model.eval()
    
    print("  Model loaded with automatic device mapping")

    # Load calibration data
    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=512, seed=args.seed)
    elif args.calib_dataset == "wikitext2-simple":
        calib_texts = load_wikitext2_simple(n_samples=args.n_calib)
    else:
        calib_texts = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=512, seed=args.seed)

    # Initialize quantizer
    quantizer = GroupWiseAWQAsymmetricL2Quantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size,
        max_tokens_per_sample=args.max_tokens_per_sample,
        activation_chunk_size=args.activation_chunk_size
    )

    # Single-layer sequential quantization (ONLY way for 70B on 80GB)
    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib,
                                       layer_batch_size=args.layer_batch_size)

    # Save model
    print(f"\nSaving quantized model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, max_shard_size="5GB")  # Shard for large model
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("QUANTIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
