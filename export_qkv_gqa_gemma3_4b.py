"""
Export Q, K, V weights and input activations from Group Query Attention (GQA) module.
Adapted for Google Gemma 3 4B (and other Gemma 3 variants).

This script extracts:
- X: Input activations to the attention module
- Wq: Query projection weights (reshaped to show heads)
- Wk: Key projection weights (reshaped to show groups)
- Wv: Value projection weights (reshaped to show groups)

Supports Gemma 3 architecture (GQA).

Usage:
    python export_gemma3_qkv.py --model-path google/gemma-3-4b-it --layer-id 0
    python export_gemma3_qkv.py --model-path ./local_gemma3_model --layer-id 5 --n-samples 64
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json

# Import calibration utilities from the project
try:
    from calibration_utils import get_c4_calibration_data
except ImportError:
    print("Warning: calibration_utils not found. Using fallback calibration data loading.")
    def get_c4_calibration_data(tokenizer, n_samples=128, seqlen=512, seed=42, cache_dir='./calibration_cache'):
        """Fallback C4 data loader."""
        from datasets import load_dataset
        dataset = load_dataset('c4', 'en', split='train', streaming=True)

        samples = []
        for sample in dataset:
            text = sample.get('text', '')
            if len(text.strip()) > 0:
                samples.append(text)
                if len(samples) >= n_samples:
                    break
        return samples


class GQAExtractor:
    """Extract data from Group Query Attention module."""

    def __init__(self, model, layer_id, device):
        self.model = model
        self.layer_id = layer_id
        self.device = device
        self.activations = []

        # Find the attention module for the specified layer
        self.attn_module = self._get_attention_module()

        # Get Q, K, V projection modules
        self.q_proj = self._get_projection('q_proj')
        self.k_proj = self._get_projection('k_proj')
        self.v_proj = self._get_projection('v_proj')

        # Get GQA configuration
        self.config = model.config
        # Try different config attribute names used by different models
        self.num_heads = getattr(self.config, 'num_attention_heads', 
                                 getattr(self.config, 'num_query_heads', None))
        self.num_key_value_heads = getattr(self.config, 'num_key_value_heads', 
                                           getattr(self.config, 'num_kv_heads', self.num_heads))
        self.hidden_size = self.config.hidden_size
        self.head_dim = getattr(self.config, 'head_dim', self.hidden_size // self.num_heads)

        # Calculate number of groups
        self.num_groups = self.num_heads // self.num_key_value_heads if self.num_key_value_heads else 1

        print(f"\n=== Gemma 3 GQA Configuration ===")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Number of query heads: {self.num_heads}")
        print(f"Number of key/value heads: {self.num_key_value_heads}")
        print(f"Head dimension: {self.head_dim}")
        print(f"Number of groups: {self.num_groups}")
        print(f"===================================\n")

    def _get_attention_module(self):
        """Get the attention module for the specified layer."""
        # Try different model architectures
        if hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model

        # Debug: print available attributes
        print(f"\nDEBUG: Model structure detection (Gemma 3)")
        print(f"  has model.model: {hasattr(self.model, 'model')}")
        if hasattr(self.model, 'model'):
            print(f"  base_model type: {type(base_model).__name__}")
            print(f"  base_model attributes: {[attr for attr in dir(base_model) if not attr.startswith('_')][:20]}")
        
        if hasattr(base_model, 'layers'):
            # Llama-style / Gemma-style architecture
            print(f"  ✓ Found base_model.layers (Llama/Gemma-style)")
            layer = base_model.layers[self.layer_id]
            # Check for self_attn (most common for Llama/Gemma)
            if hasattr(layer, 'self_attn'):
                return layer.self_attn
            # Some models use 'attention' instead
            elif hasattr(layer, 'attention'):
                return layer.attention
            else:
                raise ValueError(f"Layer {self.layer_id} has no self_attn or attention module")
        elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
            # GPT-style architecture
            print(f"  ✓ Found base_model.transformer.h (GPT-style)")
            return base_model.transformer.h[self.layer_id].attn
        else:
            # Enhanced error message with debugging info
            print(f"\n  ✗ Unknown architecture!")
            print(f"  base_model type: {type(base_model).__name__}")
            attrs = [attr for attr in dir(base_model) if not attr.startswith('_')]
            print(f"  Available attributes: {attrs[:30]}")
            
            raise ValueError(
                f"Unknown model architecture. Cannot find layer {self.layer_id}.\n"
                f"Model type: {type(base_model).__name__}\n"
                f"Available attributes: {attrs[:30]}\n"
                f"Please report this model structure for support."
            )

    def _get_projection(self, proj_name):
        """Get Q, K, or V projection module."""
        if hasattr(self.attn_module, proj_name):
            return getattr(self.attn_module, proj_name)
        else:
            raise ValueError(f"Projection {proj_name} not found in attention module")

    def register_hooks(self):
        """Register forward hooks to capture input activations."""
        def hook_fn(_module, input, _output):
            # Capture input activation X
            X = input[0].detach().cpu()
            self.activations.append(X)

        # Register hook on q_proj to capture input to attention
        self.hook_handle = self.q_proj.register_forward_hook(hook_fn)
        print(f"Registered hook on layer {self.layer_id} q_proj")

    def remove_hooks(self):
        """Remove the registered hooks."""
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()
            print("Removed hooks")

    def get_activations(self):
        """Get concatenated activations, handling variable sequence lengths."""
        if not self.activations:
            raise ValueError("No activations captured. Run calibration first.")

        # Reshape to handle variable sequence lengths
        reshaped = [x.reshape(-1, x.shape[-1]) for x in self.activations]
        all_activations = torch.cat(reshaped, dim=0)

        # Convert to float32 for numpy compatibility
        return all_activations.float()

    def extract_weights(self):
        """Extract and reshape Q, K, V weight matrices."""
        # Get weights (move to CPU and convert to float32)
        Wq = self.q_proj.weight.data.cpu().float()  # [hidden_size, hidden_size]
        Wk = self.k_proj.weight.data.cpu().float()  # [num_kv_heads * head_dim, hidden_size]
        Wv = self.v_proj.weight.data.cpu().float()  # [num_kv_heads * head_dim, hidden_size]

        print(f"\nOriginal weight shapes:")
        print(f"Wq: {Wq.shape}")
        print(f"Wk: {Wk.shape}")
        print(f"Wv: {Wv.shape}")

        # Reshape to expose head structure
        # Wq: [num_heads, head_dim, hidden_size]
        Wq_reshaped = Wq.reshape(self.num_heads, self.head_dim, self.hidden_size)

        # Wk, Wv: [num_kv_heads, head_dim, hidden_size]
        Wk_reshaped = Wk.reshape(self.num_key_value_heads, self.head_dim, self.hidden_size)
        Wv_reshaped = Wv.reshape(self.num_key_value_heads, self.head_dim, self.hidden_size)

        # CRITICAL: Reshape Wq to show GROUP structure
        # Wq_grouped: [num_kv_heads, queries_per_group, head_dim, hidden_size]
        # This shows which query heads are GROUPED to share the same KV head
        Wq_grouped = Wq_reshaped.reshape(
            self.num_key_value_heads,
            self.num_groups,
            self.head_dim,
            self.hidden_size
        )

        print(f"\nReshaped weight shapes (showing head structure):")
        print(f"Wq: {Wq_reshaped.shape} [num_heads, head_dim, hidden_size]")
        print(f"Wk: {Wk_reshaped.shape} [num_kv_heads, head_dim, hidden_size]")
        print(f"Wv: {Wv_reshaped.shape} [num_kv_heads, head_dim, hidden_size]")

        print(f"\nGrouped weight shapes (showing GROUP structure for GQA):")
        print(f"Wq_grouped: {Wq_grouped.shape} [num_kv_heads, queries_per_group, head_dim, hidden_size]")
        print(f"  → Each of {self.num_key_value_heads} KV heads is shared by {self.num_groups} query heads")

        return {
            'Wq_original': Wq,
            'Wk_original': Wk,
            'Wv_original': Wv,
            'Wq_reshaped': Wq_reshaped,
            'Wk_reshaped': Wk_reshaped,
            'Wv_reshaped': Wv_reshaped,
            'Wq_grouped': Wq_grouped
        }

    def save_data(self, output_dir, X, weights):
        """Save extracted data and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save activations
        X_np = X.numpy()
        np.save(output_dir / 'X_activations.npy', X_np)
        print(f"\nSaved X_activations.npy: {X_np.shape}")

        # Save original weights
        np.save(output_dir / 'Wq_original.npy', weights['Wq_original'].numpy())
        np.save(output_dir / 'Wk_original.npy', weights['Wk_original'].numpy())
        np.save(output_dir / 'Wv_original.npy', weights['Wv_original'].numpy())
        print(f"Saved original weights (2D matrices)")

        # Save reshaped weights
        np.save(output_dir / 'Wq_reshaped.npy', weights['Wq_reshaped'].numpy())
        np.save(output_dir / 'Wk_reshaped.npy', weights['Wk_reshaped'].numpy())
        np.save(output_dir / 'Wv_reshaped.npy', weights['Wv_reshaped'].numpy())
        print(f"Saved reshaped weights (3D tensors with head structure)")

        # Save grouped weights (shows GQA group structure)
        np.save(output_dir / 'Wq_grouped.npy', weights['Wq_grouped'].numpy())
        print(f"Saved Wq_grouped.npy (4D tensor showing which Q heads share KV heads)")

        # Create metadata
        metadata = {
            'model': 'Gemma 3 4B',
            'layer_id': self.layer_id,
            'model_config': {
                'hidden_size': self.hidden_size,
                'num_attention_heads': self.num_heads,
                'num_key_value_heads': self.num_key_value_heads,
                'head_dim': self.head_dim,
                'num_groups': self.num_groups
            },
            'shapes': {
                'X_activations': list(X_np.shape),
                'Wq_original': list(weights['Wq_original'].shape),
                'Wk_original': list(weights['Wk_original'].shape),
                'Wv_original': list(weights['Wv_original'].shape),
                'Wq_reshaped': list(weights['Wq_reshaped'].shape),
                'Wk_reshaped': list(weights['Wk_reshaped'].shape),
                'Wv_reshaped': list(weights['Wv_reshaped'].shape),
                'Wq_grouped': list(weights['Wq_grouped'].shape)
            },
            'description': {
                'X': f'Input activations, shape: [num_tokens, hidden_size] = {X_np.shape}',
                'Wq_original': f'Query weights (original), shape: [hidden_size, hidden_size] = {weights["Wq_original"].shape}',
                'Wk_original': f'Key weights (original), shape: [num_kv_heads * head_dim, hidden_size] = {weights["Wk_original"].shape}',
                'Wv_original': f'Value weights (original), shape: [num_kv_heads * head_dim, hidden_size] = {weights["Wv_original"].shape}',
                'Wq_reshaped': f'Query weights (reshaped), shape: [num_heads, head_dim, hidden_size] = {weights["Wq_reshaped"].shape}',
                'Wk_reshaped': f'Key weights (reshaped), shape: [num_kv_heads, head_dim, hidden_size] = {weights["Wk_reshaped"].shape}',
                'Wv_reshaped': f'Value weights (reshaped), shape: [num_kv_heads, head_dim, hidden_size] = {weights["Wv_reshaped"].shape}',
                'Wq_grouped': f'Query weights (grouped), shape: [num_kv_heads, queries_per_group, head_dim, hidden_size] = {weights["Wq_grouped"].shape}'
            },
            'gqa_info': {
                'num_query_heads': self.num_heads,
                'num_kv_heads': self.num_key_value_heads,
                'queries_per_kv_group': self.num_groups,
                'explanation': f'This is Group Query Attention with {self.num_groups} query heads per key/value head'
            }
        }

        # Save JSON metadata
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata.json")

        # Save human-readable text description
        with open(output_dir / 'README.txt', 'w') as f:
            f.write(f"=== Group Query Attention (GQA) Data Export ===\n")
            f.write(f"Model: Gemma 3 4B\n\n")
            f.write(f"Layer: {self.layer_id}\n\n")

            f.write(f"--- GQA Configuration ---\n")
            f.write(f"Hidden size: {self.hidden_size}\n")
            f.write(f"Number of query heads: {self.num_heads}\n")
            f.write(f"Number of key/value heads: {self.num_key_value_heads}\n")
            f.write(f"Head dimension: {self.head_dim}\n")
            f.write(f"Number of groups (queries per KV head): {self.num_groups}\n\n")

            f.write(f"--- Data Shapes ---\n")
            f.write(f"X_activations: {X_np.shape}\n")
            f.write(f"  - [num_tokens, hidden_size]\n")
            f.write(f"  - Input activations to the attention module\n\n")

            f.write(f"Wq_original: {weights['Wq_original'].shape}\n")
            f.write(f"  - [hidden_size, hidden_size]\n")
            f.write(f"  - Query projection weights (original 2D format)\n\n")

            f.write(f"Wk_original: {weights['Wk_original'].shape}\n")
            f.write(f"  - [num_kv_heads * head_dim, hidden_size]\n")
            f.write(f"  - Key projection weights (original 2D format)\n\n")

            f.write(f"Wv_original: {weights['Wv_original'].shape}\n")
            f.write(f"  - [num_kv_heads * head_dim, hidden_size]\n")
            f.write(f"  - Value projection weights (original 2D format)\n\n")

            f.write(f"Wq_reshaped: {weights['Wq_reshaped'].shape}\n")
            f.write(f"  - [num_heads, head_dim, hidden_size]\n")
            f.write(f"  - Query weights reshaped to show {self.num_heads} heads\n")
            f.write(f"  - Each head has dimension {self.head_dim}\n\n")

            f.write(f"Wk_reshaped: {weights['Wk_reshaped'].shape}\n")
            f.write(f"  - [num_kv_heads, head_dim, hidden_size]\n")
            f.write(f"  - Key weights reshaped to show {self.num_key_value_heads} KV heads\n")
            f.write(f"  - Each KV head is shared by {self.num_groups} query heads\n\n")

            f.write(f"Wv_reshaped: {weights['Wv_reshaped'].shape}\n")
            f.write(f"  - [num_kv_heads, head_dim, hidden_size]\n")
            f.write(f"  - Value weights reshaped to show {self.num_key_value_heads} KV heads\n")
            f.write(f"  - Each KV head is shared by {self.num_groups} query heads\n\n")

            f.write(f"Wq_grouped: {weights['Wq_grouped'].shape}\n")
            f.write(f"  - [num_kv_heads, queries_per_group, head_dim, hidden_size]\n")
            f.write(f"  - Query weights reshaped to show GROUP structure\n")
            f.write(f"  - Dimension 0: Which KV head this group uses ({self.num_key_value_heads} groups)\n")
            f.write(f"  - Dimension 1: Which query within the group ({self.num_groups} queries per group)\n")
            f.write(f"  - This makes it explicit which Q heads share which KV heads!\n\n")

            f.write(f"--- Group Query Attention Explanation (Gemma 3) ---\n")
            f.write(f"In GQA, queries have {self.num_heads} heads, but keys and values\n")
            f.write(f"have only {self.num_key_value_heads} heads. This means {self.num_groups} query heads\n")
            f.write(f"share the same key and value heads (grouped).\n\n")

            f.write(f"GROUP STRUCTURE (see Wq_grouped.npy):\n")
            f.write(f"  - KV group 0: shared by query heads 0-{self.num_groups-1}\n")
            f.write(f"  - KV group 1: shared by query heads {self.num_groups}-{2*self.num_groups-1}\n")
            f.write(f"  - ...\n")
            f.write(f"  - KV group {self.num_key_value_heads-1}: shared by query heads {(self.num_key_value_heads-1)*self.num_groups}-{self.num_heads-1}\n\n")

            f.write(f"Example: Wq_grouped[0, :, :, :] contains all {self.num_groups} query heads\n")
            f.write(f"that share KV head 0 (i.e., Wk_reshaped[0] and Wv_reshaped[0]).\n\n")

            f.write(f"This reduces memory and computation while maintaining quality.\n")

        print(f"Saved README.txt")
        print(f"\nAll data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Export GQA Q/K/V weights and activations for Gemma 3 4B')
    parser.add_argument('--model-path', type=str, default='google/gemma-3-4b-it',
                        help='Model name or path (default: google/gemma-3-4b-it). '
                             'Supports Gemma 3 and other GQA models.')
    parser.add_argument('--layer-id', type=int, default=0,
                        help='Layer index to extract from (default: 0)')
    parser.add_argument('--n-samples', type=int, default=128,
                        help='Number of calibration samples (default: 128)')
    parser.add_argument('--seqlen', type=int, default=512,
                        help='Sequence length (default: 512)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./gemma3_export_layer{layer_id})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--cache-dir', type=str, default='./calibration_cache',
                        help='Directory to cache calibration data (default: ./calibration_cache)')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set output directory
    if args.output_dir is None:
        args.output_dir = f'./gemma3_export_layer{args.layer_id}'

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
    # Gemma 3 usually uses model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        print(f"Total layers: {len(model.model.layers)}")
    elif hasattr(model, 'layers'):
        print(f"Total layers: {len(model.layers)}")
    else:
        print(f"Total layers: Unknown (check architecture)")

    # Create extractor
    extractor = GQAExtractor(model, args.layer_id, device)

    # Register hooks
    extractor.register_hooks()

    # Get calibration data (default: C4, matching awq_stand_xl.py)
    print(f"\nLoading calibration data: C4 (default)")
    calib_texts = get_c4_calibration_data(
        tokenizer,
        n_samples=args.n_samples,
        seqlen=args.seqlen,
        seed=args.seed,
        cache_dir=args.cache_dir
    )
    print(f"Loaded {len(calib_texts)} calibration samples")

    # Run calibration to capture activations
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

            # Forward pass (use_cache=False to avoid cache issues)
            _ = model(**inputs, use_cache=False)

    print(f"Captured activations from {len(extractor.activations)} samples")

    # Remove hooks
    extractor.remove_hooks()

    # Get activations
    print(f"\nConcatenating activations...")
    X = extractor.get_activations()
    print(f"Total activation shape: {X.shape}")

    # Extract weights
    print(f"\nExtracting weights...")
    weights = extractor.extract_weights()

    # Save data
    print(f"\nSaving data to {args.output_dir}...")
    extractor.save_data(args.output_dir, X, weights)

    print(f"\n{'='*60}")
    print(f"✓ Export complete!")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Layer: {args.layer_id}")
    print(f"  Calibration: C4 dataset, {args.n_samples} samples")
    print(f"  Output: {args.output_dir}")
    print(f"\nFiles saved:")
    print(f"  - X_activations.npy")
    print(f"  - Wq_original.npy, Wk_original.npy, Wv_original.npy (2D)")
    print(f"  - Wq_reshaped.npy, Wk_reshaped.npy, Wv_reshaped.npy (3D with heads)")
    print(f"  - Wq_grouped.npy (4D showing GQA group structure)")
    print(f"  - metadata.json")
    print(f"  - README.txt")
    print(f"\nLoad data in Python:")
    print(f"  import numpy as np")
    print(f"  X = np.load('{args.output_dir}/X_activations.npy')")
    print(f"  Wq = np.load('{args.output_dir}/Wq_reshaped.npy')  # [num_heads, head_dim, hidden]")
    print(f"  Wk = np.load('{args.output_dir}/Wk_reshaped.npy')  # [num_kv_heads, head_dim, hidden]")
    print(f"  Wv = np.load('{args.output_dir}/Wv_reshaped.npy')  # [num_kv_heads, head_dim, hidden]")
    print(f"  Wq_grouped = np.load('{args.output_dir}/Wq_grouped.npy')  # [num_kv_heads, queries_per_group, head_dim, hidden]")


if __name__ == '__main__':
    main()