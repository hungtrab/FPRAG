"""
Export James-Stein estimation for E[X[:,j]] and Q/K/V weights for one GQA group.

This script:
1. Captures activations X from calibration data
2. Computes naive channel means: E[X[:,j]] for each input channel j
3. Applies James-Stein shrinkage estimation for improved mean estimates
4. Exports Q, K, V weights for a specific GQA group (default: group 0)

James-Stein Estimator:
- Shrinks individual channel means towards the grand mean
- Provably dominates naive sample mean for p ≥ 3 dimensions
- Formula: θ̂_JS[j] = μ̄ + (1 - λ) * (x̄[j] - μ̄)
  where:
    λ = (p - 2) * σ² / ||x̄ - μ̄||²
    μ̄ = grand mean (average of all channel means)
    x̄[j] = naive sample mean for channel j
    p = number of channels
    σ² = estimated variance

Usage:
    python xspot.py --layer-id 0 --group-id 0
    python xspot.py --layer-id 5 --group-id 2 --n-samples 256
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json

# Import calibration utilities
from calibration_utils import get_c4_calibration_data


class JamesSteinGQAExporter:
    """Export James-Stein estimates and GQA weights for one group."""

    def __init__(self, model, layer_id, group_id, device):
        self.model = model
        self.layer_id = layer_id
        self.group_id = group_id
        self.device = device
        self.activations = []

        # Get attention module
        self.attn_module = self._get_attention_module()

        # Get Q, K, V projections
        self.q_proj = getattr(self.attn_module, 'q_proj')
        self.k_proj = getattr(self.attn_module, 'k_proj')
        self.v_proj = getattr(self.attn_module, 'v_proj')

        # Get GQA configuration
        self.config = model.config
        # Try different config attribute names used by different models
        self.num_heads = getattr(self.config, 'num_attention_heads', 
                                 getattr(self.config, 'num_query_heads', None))
        self.num_key_value_heads = getattr(self.config, 'num_key_value_heads',
                                           getattr(self.config, 'num_kv_heads', self.num_heads))
        self.hidden_size = self.config.hidden_size
        
        # Compute head_dim from actual weight shapes (more reliable for models like Gemma-2)
        # where head_dim may not equal hidden_size // num_heads
        q_proj_out_features = self.q_proj.weight.shape[0]
        self.head_dim = q_proj_out_features // self.num_heads
        
        self.num_groups = self.num_heads // self.num_key_value_heads if self.num_key_value_heads else 1

        print(f"\n=== GQA Configuration ===")
        print(f"Layer: {self.layer_id}")
        print(f"Group: {self.group_id} (out of {self.num_key_value_heads} groups)")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Number of query heads: {self.num_heads}")
        print(f"Number of key/value heads: {self.num_key_value_heads}")
        print(f"Head dimension: {self.head_dim}")
        print(f"Queries per KV group: {self.num_groups}")
        print(f"========================\n")

        # Validate group_id
        if self.group_id >= self.num_key_value_heads:
            raise ValueError(f"group_id {self.group_id} out of range [0, {self.num_key_value_heads-1}]")

    def _get_attention_module(self):
        """Get the attention module for the specified layer."""
        if hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model

        if hasattr(base_model, 'layers'):
            # Llama-style architecture (Llama, Gemma, Mistral, etc.)
            layer = base_model.layers[self.layer_id]
            # Check for self_attn (most common)
            if hasattr(layer, 'self_attn'):
                return layer.self_attn
            # Some models use 'attention' instead
            elif hasattr(layer, 'attention'):
                return layer.attention
            else:
                raise ValueError(f"Layer {self.layer_id} has no self_attn or attention module")
        elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
            return base_model.transformer.h[self.layer_id].attn
        else:
            raise ValueError(f"Unknown model architecture. Cannot find layer {self.layer_id}")

    def register_hooks(self):
        """Register forward hooks to capture input activations."""
        def hook_fn(_module, input, _output):
            X = input[0].detach().cpu()
            self.activations.append(X)

        self.hook_handle = self.q_proj.register_forward_hook(hook_fn)
        print(f"Registered hook on layer {self.layer_id} q_proj")

    def remove_hooks(self):
        """Remove the registered hooks."""
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()
            print("Removed hooks")

    def get_activations(self):
        """Get concatenated activations."""
        if not self.activations:
            raise ValueError("No activations captured. Run calibration first.")

        # Reshape to handle variable sequence lengths
        reshaped = [x.reshape(-1, x.shape[-1]) for x in self.activations]
        all_activations = torch.cat(reshaped, dim=0)

        # Convert to float32 for numpy compatibility
        return all_activations.float()

    def compute_james_stein_estimates(self, X):
        """
        Compute James-Stein shrinkage estimates for channel means.

        Uses the same implementation as awq_js_xl.py:
        - Variance estimate: mean absolute deviation squared (not activation variance)
        - Shrinkage formula: c = (p - 2) * σ² / Σ(X̄[j] - μ̄)²

        Args:
            X: Activations tensor of shape [num_tokens, hidden_size]

        Returns:
            dict with:
                - naive_means: E[X[:,j]] for each channel j
                - js_means: James-Stein estimates
                - grand_mean: μ̄ (mean of all channel means)
                - shrinkage_factor: λ
        """
        print(f"\n=== Computing James-Stein Estimates ===")
        print(f"Activation shape: {X.shape}")

        # Compute naive sample means per channel
        # x̄[j] = E[X[:,j]]
        naive_means = X.mean(dim=0)  # shape: [hidden_size]
        print(f"Naive means shape: {naive_means.shape}")

        # Number of channels
        p = self.hidden_size
        print(f"Number of channels (p): {p}")

        # Need at least 3 dimensions for James-Stein to be beneficial
        if p < 3:
            print("WARNING: p < 3, James-Stein not applicable, returning naive means")
            return {
                'naive_means': naive_means,
                'js_means': naive_means,
                'grand_mean': naive_means.mean(),
                'shrinkage_factor': torch.tensor(0.0),
                'variance': torch.tensor(0.0),
                'squared_distance': torch.tensor(0.0)
            }

        # Compute grand mean (mean of all channel means)
        # μ̄ = mean(x̄)
        grand_mean = naive_means.mean()
        print(f"Grand mean: {grand_mean.item():.6f}")

        # Compute deviations from grand mean
        deviations = naive_means - grand_mean

        # Compute sum of squared deviations
        sum_sq_dev = (deviations ** 2).sum()
        print(f"Sum of squared deviations: {sum_sq_dev.item():.6f}")

        # Prevent division by zero
        if sum_sq_dev < 1e-10:
            print("WARNING: All means are the same, no shrinkage needed")
            return {
                'naive_means': naive_means,
                'js_means': naive_means,
                'grand_mean': grand_mean,
                'shrinkage_factor': torch.tensor(0.0),
                'variance': torch.tensor(0.0),
                'squared_distance': sum_sq_dev
            }

        # CRITICAL FIX: Estimate variance using mean absolute deviation squared
        # This is the variance of the MEANS, not the variance of the data
        # Use a conservative estimate: mean absolute deviation squared
        variance_estimate = ((naive_means - grand_mean).abs().mean()) ** 2
        # Add small constant for numerical stability
        variance_estimate = variance_estimate.clamp(min=1e-8)
        print(f"Variance estimate (MAD²): {variance_estimate.item():.6f}")

        # James-Stein shrinkage factor
        # c = (p - 2) * σ² / Σ(X̄[j] - μ̄)²
        shrinkage_factor = ((p - 2) * variance_estimate) / sum_sq_dev

        # Clip shrinkage factor to [0, 1] for stability
        shrinkage_factor = shrinkage_factor.clamp(0.0, 1.0)
        print(f"Shrinkage factor c: {shrinkage_factor.item():.6f}")

        # Apply James-Stein shrinkage
        # θ̂_JS[j] = μ̄ + (1 - c) * (x̄[j] - μ̄)
        js_means = grand_mean + (1 - shrinkage_factor) * deviations
        print(f"James-Stein means shape: {js_means.shape}")

        # Compute statistics
        mean_diff = (js_means - naive_means).abs().mean()
        max_diff = (js_means - naive_means).abs().max()
        print(f"\nShrinkage statistics:")
        print(f"  Mean absolute difference: {mean_diff.item():.6f}")
        print(f"  Max absolute difference: {max_diff.item():.6f}")
        print(f"  Shrinkage direction: towards grand mean ({grand_mean.item():.6f})")

        return {
            'naive_means': naive_means,
            'js_means': js_means,
            'grand_mean': grand_mean,
            'shrinkage_factor': shrinkage_factor,
            'variance': variance_estimate,
            'squared_distance': sum_sq_dev
        }

    def extract_group_weights(self):
        """
        Extract Q, K, V weights for the specified GQA group.

        Returns:
            dict with:
                - Wq_group: Query weights for this group [queries_per_group, head_dim, hidden_size]
                - Wk_group: Key weights for this group [head_dim, hidden_size]
                - Wv_group: Value weights for this group [head_dim, hidden_size]
        """
        print(f"\n=== Extracting Group {self.group_id} Weights ===")

        # Get weights (move to CPU and convert to float32)
        Wq = self.q_proj.weight.data.cpu().float()  # [hidden_size, hidden_size]
        Wk = self.k_proj.weight.data.cpu().float()  # [num_kv_heads * head_dim, hidden_size]
        Wv = self.v_proj.weight.data.cpu().float()  # [num_kv_heads * head_dim, hidden_size]

        print(f"Original weight shapes:")
        print(f"  Wq: {Wq.shape}")
        print(f"  Wk: {Wk.shape}")
        print(f"  Wv: {Wv.shape}")

        # Reshape to expose head structure
        Wq_reshaped = Wq.reshape(self.num_heads, self.head_dim, self.hidden_size)
        Wk_reshaped = Wk.reshape(self.num_key_value_heads, self.head_dim, self.hidden_size)
        Wv_reshaped = Wv.reshape(self.num_key_value_heads, self.head_dim, self.hidden_size)

        # Reshape Wq to show GROUP structure
        # [num_kv_heads, queries_per_group, head_dim, hidden_size]
        Wq_grouped = Wq_reshaped.reshape(
            self.num_key_value_heads,
            self.num_groups,
            self.head_dim,
            self.hidden_size
        )

        # Extract weights for the specified group
        Wq_group = Wq_grouped[self.group_id]  # [queries_per_group, head_dim, hidden_size]
        Wk_group = Wk_reshaped[self.group_id]  # [head_dim, hidden_size]
        Wv_group = Wv_reshaped[self.group_id]  # [head_dim, hidden_size]

        print(f"\nGroup {self.group_id} weight shapes:")
        print(f"  Wq_group: {Wq_group.shape} [queries_per_group, head_dim, hidden_size]")
        print(f"  Wk_group: {Wk_group.shape} [head_dim, hidden_size]")
        print(f"  Wv_group: {Wv_group.shape} [head_dim, hidden_size]")

        # Calculate which query head indices belong to this group
        start_head = self.group_id * self.num_groups
        end_head = start_head + self.num_groups
        print(f"\nThis group corresponds to query heads {start_head}-{end_head-1}")

        return {
            'Wq_group': Wq_group,
            'Wk_group': Wk_group,
            'Wv_group': Wv_group,
            'query_head_range': (start_head, end_head)
        }

    def save_data(self, output_dir, X, js_results, group_weights):
        """Save all exported data."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save activations
        X_np = X.numpy()
        np.save(output_dir / 'X_activations.npy', X_np)
        print(f"\nSaved X_activations.npy: {X_np.shape}")

        # Save James-Stein results
        np.save(output_dir / 'naive_means.npy', js_results['naive_means'].numpy())
        np.save(output_dir / 'js_means.npy', js_results['js_means'].numpy())
        np.save(output_dir / 'grand_mean.npy', js_results['grand_mean'].numpy())
        np.save(output_dir / 'shrinkage_factor.npy', js_results['shrinkage_factor'].numpy())
        print(f"Saved James-Stein estimation results")

        # Save group weights
        np.save(output_dir / f'Wq_group{self.group_id}.npy', group_weights['Wq_group'].numpy())
        np.save(output_dir / f'Wk_group{self.group_id}.npy', group_weights['Wk_group'].numpy())
        np.save(output_dir / f'Wv_group{self.group_id}.npy', group_weights['Wv_group'].numpy())
        print(f"Saved group {self.group_id} Q/K/V weights")

        # Create metadata
        metadata = {
            'layer_id': self.layer_id,
            'group_id': self.group_id,
            'model_config': {
                'hidden_size': self.hidden_size,
                'num_attention_heads': self.num_heads,
                'num_key_value_heads': self.num_key_value_heads,
                'head_dim': self.head_dim,
                'queries_per_kv_group': self.num_groups
            },
            'james_stein': {
                'grand_mean': float(js_results['grand_mean']),
                'shrinkage_factor': float(js_results['shrinkage_factor']),
                'variance': float(js_results['variance']),
                'squared_distance': float(js_results['squared_distance']),
                'mean_abs_diff': float((js_results['js_means'] - js_results['naive_means']).abs().mean()),
                'max_abs_diff': float((js_results['js_means'] - js_results['naive_means']).abs().max())
            },
            'group_info': {
                'group_id': self.group_id,
                'query_head_start': group_weights['query_head_range'][0],
                'query_head_end': group_weights['query_head_range'][1] - 1,
                'num_query_heads_in_group': self.num_groups
            },
            'shapes': {
                'X_activations': list(X_np.shape),
                'naive_means': list(js_results['naive_means'].shape),
                'js_means': list(js_results['js_means'].shape),
                'Wq_group': list(group_weights['Wq_group'].shape),
                'Wk_group': list(group_weights['Wk_group'].shape),
                'Wv_group': list(group_weights['Wv_group'].shape)
            }
        }

        # Save JSON metadata
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata.json")

        # Save human-readable README
        with open(output_dir / 'README.txt', 'w') as f:
            f.write(f"=== James-Stein Estimation + GQA Group Weights Export ===\n\n")
            f.write(f"Layer: {self.layer_id}\n")
            f.write(f"GQA Group: {self.group_id}\n\n")

            f.write(f"--- James-Stein Estimation ---\n")
            f.write(f"Goal: Estimate E[X[:,j]] for each input channel j with shrinkage.\n\n")
            f.write(f"Naive means (sample mean per channel):\n")
            f.write(f"  File: naive_means.npy\n")
            f.write(f"  Shape: {js_results['naive_means'].shape}\n")
            f.write(f"  Formula: x̄[j] = (1/N) Σ X[i,j]\n\n")

            f.write(f"James-Stein estimates (shrunk towards grand mean):\n")
            f.write(f"  File: js_means.npy\n")
            f.write(f"  Shape: {js_results['js_means'].shape}\n")
            f.write(f"  Formula: θ̂_JS[j] = μ̄ + (1-λ)(x̄[j] - μ̄)\n")
            f.write(f"  Grand mean μ̄: {js_results['grand_mean']:.6f}\n")
            f.write(f"  Shrinkage λ: {js_results['shrinkage_factor']:.6f}\n")
            f.write(f"  Mean absolute difference: {metadata['james_stein']['mean_abs_diff']:.6f}\n")
            f.write(f"  Max absolute difference: {metadata['james_stein']['max_abs_diff']:.6f}\n\n")

            f.write(f"Why James-Stein?\n")
            f.write(f"  - Provably dominates naive sample mean for p ≥ 3 dimensions\n")
            f.write(f"  - Reduces mean squared error by shrinking towards grand mean\n")
            f.write(f"  - Particularly effective when channel means are similar\n")
            f.write(f"  - Optimal for channels with similar importance (not outliers)\n\n")

            f.write(f"--- GQA Group {self.group_id} Weights ---\n")
            f.write(f"Query heads in this group: {group_weights['query_head_range'][0]}-{group_weights['query_head_range'][1]-1}\n")
            f.write(f"(Total {self.num_groups} query heads share 1 KV head)\n\n")

            f.write(f"Wq_group{self.group_id}.npy:\n")
            f.write(f"  Shape: {group_weights['Wq_group'].shape}\n")
            f.write(f"  [queries_per_group, head_dim, hidden_size]\n")
            f.write(f"  Contains {self.num_groups} query weight matrices\n\n")

            f.write(f"Wk_group{self.group_id}.npy:\n")
            f.write(f"  Shape: {group_weights['Wk_group'].shape}\n")
            f.write(f"  [head_dim, hidden_size]\n")
            f.write(f"  Shared by all {self.num_groups} query heads in group {self.group_id}\n\n")

            f.write(f"Wv_group{self.group_id}.npy:\n")
            f.write(f"  Shape: {group_weights['Wv_group'].shape}\n")
            f.write(f"  [head_dim, hidden_size]\n")
            f.write(f"  Shared by all {self.num_groups} query heads in group {self.group_id}\n\n")

            f.write(f"--- Usage Example ---\n")
            f.write(f"import numpy as np\n\n")
            f.write(f"# Load James-Stein estimates\n")
            f.write(f"naive = np.load('naive_means.npy')  # E[X[:,j]]\n")
            f.write(f"js = np.load('js_means.npy')        # James-Stein shrinkage\n\n")
            f.write(f"# Load group {self.group_id} weights\n")
            f.write(f"Wq = np.load('Wq_group{self.group_id}.npy')  # [{self.num_groups}, {self.head_dim}, {self.hidden_size}]\n")
            f.write(f"Wk = np.load('Wk_group{self.group_id}.npy')  # [{self.head_dim}, {self.hidden_size}]\n")
            f.write(f"Wv = np.load('Wv_group{self.group_id}.npy')  # [{self.head_dim}, {self.hidden_size}]\n\n")
            f.write(f"# All {self.num_groups} query heads use the same Wk and Wv\n")

        print(f"Saved README.txt")
        print(f"\nAll data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Export James-Stein estimates and GQA group weights')
    parser.add_argument('--model-path', type=str, default='./models/Llama-3-8B',
                        help='Model name or path (default: ./models/Llama-3-8B)')
    parser.add_argument('--layer-id', type=int, default=0,
                        help='Layer index (default: 0)')
    parser.add_argument('--group-id', type=int, default=0,
                        help='GQA group index (default: 0)')
    parser.add_argument('--n-samples', type=int, default=128,
                        help='Number of calibration samples (default: 128)')
    parser.add_argument('--seqlen', type=int, default=512,
                        help='Sequence length (default: 512)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./xspot_layer{layer_id}_group{group_id})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--cache-dir', type=str, default='./calibration_cache',
                        help='Cache directory for calibration data')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set output directory
    if args.output_dir is None:
        args.output_dir = f'./xspot_layer{args.layer_id}_group{args.group_id}'

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model: {args.model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model with bfloat16, trying float16: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded successfully")

    # Create exporter
    exporter = JamesSteinGQAExporter(model, args.layer_id, args.group_id, device)

    # Register hooks
    exporter.register_hooks()

    # Get calibration data
    print(f"\nLoading calibration data: C4")
    calib_texts = get_c4_calibration_data(
        tokenizer,
        n_samples=args.n_samples,
        seqlen=args.seqlen,
        seed=args.seed,
        cache_dir=args.cache_dir
    )
    print(f"Loaded {len(calib_texts)} calibration samples")

    # Run calibration
    print(f"\nRunning calibration to capture activations...")
    with torch.no_grad():
        for i, text in enumerate(calib_texts):
            if i % 20 == 0:
                print(f"Processing sample {i+1}/{len(calib_texts)}")

            inputs = tokenizer(
                text,
                return_tensors='pt',
                max_length=args.seqlen,
                truncation=True,
                padding=False
            ).to(device)

            _ = model(**inputs, use_cache=False)

    print(f"Captured activations from {len(exporter.activations)} samples")

    # Remove hooks
    exporter.remove_hooks()

    # Get activations
    print(f"\nConcatenating activations...")
    X = exporter.get_activations()
    print(f"Total activation shape: {X.shape}")

    # Compute James-Stein estimates
    js_results = exporter.compute_james_stein_estimates(X)

    # Extract group weights
    group_weights = exporter.extract_group_weights()

    # Save data
    print(f"\nSaving data to {args.output_dir}...")
    exporter.save_data(args.output_dir, X, js_results, group_weights)

    print(f"\n{'='*60}")
    print(f"✓ Export complete!")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Layer: {args.layer_id}")
    print(f"  GQA Group: {args.group_id}")
    print(f"  Calibration: C4 dataset, {args.n_samples} samples")
    print(f"  Output: {args.output_dir}")
    print(f"\nFiles saved:")
    print(f"  - X_activations.npy [{X.shape}]")
    print(f"  - naive_means.npy [{js_results['naive_means'].shape}]")
    print(f"  - js_means.npy [{js_results['js_means'].shape}]")
    print(f"  - Wq_group{args.group_id}.npy [{group_weights['Wq_group'].shape}]")
    print(f"  - Wk_group{args.group_id}.npy [{group_weights['Wk_group'].shape}]")
    print(f"  - Wv_group{args.group_id}.npy [{group_weights['Wv_group'].shape}]")
    print(f"  - metadata.json")
    print(f"  - README.txt")
    print(f"\nJames-Stein Statistics:")
    print(f"  Grand mean: {js_results['grand_mean']:.6f}")
    print(f"  Shrinkage factor λ: {js_results['shrinkage_factor']:.6f}")
    print(f"  Mean absolute correction: {(js_results['js_means'] - js_results['naive_means']).abs().mean():.6f}")
    print(f"\nLoad in Python:")
    print(f"  import numpy as np")
    print(f"  naive = np.load('{args.output_dir}/naive_means.npy')")
    print(f"  js = np.load('{args.output_dir}/js_means.npy')")
    print(f"  Wq = np.load('{args.output_dir}/Wq_group{args.group_id}.npy')")
    print(f"  Wk = np.load('{args.output_dir}/Wk_group{args.group_id}.npy')")
    print(f"  Wv = np.load('{args.output_dir}/Wv_group{args.group_id}.npy')")


if __name__ == '__main__':
    main()
