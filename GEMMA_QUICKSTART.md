# Quick Start: Using with Gemma Models

## Important Note

**`google/gemma-3-4b-it` does not exist!**

Available Gemma models (as of December 2025):
- `google/gemma-2-2b-it` (2B parameters)
- `google/gemma-2-9b-it` (9B parameters)  
- `google/gemma-2-27b-it` (27B parameters)

## Quick Verification

Check if your model is compatible:

```bash
# Quick check (config only, very fast)
python verify_model_compatibility.py google/gemma-2-2b-it

# Full check (loads model, slower but thorough)
python test_gemma_arch.py google/gemma-2-2b-it
```

## Usage with Gemma

### Step 1: Export Q/K/V data from Gemma

```bash
# Use gemma-2-2b-it (smallest, fastest)
python export_qkv_gqa.py \
    --model-path google/gemma-2-2b-it \
    --layer-id 0 \
    --n-samples 128 \
    --output-dir ./gemma_export_layer0

# Or use gemma-2-9b-it (medium)
python export_qkv_gqa.py \
    --model-path google/gemma-2-9b-it \
    --layer-id 0 \
    --n-samples 128 \
    --output-dir ./gemma9b_export_layer0
```

### Step 2: Run quantization

The `quantize_qkv.py` script expects data in `./xspot_layer0_group0/`. Either:

**Option A: Move the exported data**
```bash
mv gemma_export_layer0 xspot_layer0_group0
```

**Option B: Update paths in quantize_qkv.py**

Edit lines 602-605 in `quantize_qkv.py`:
```python
js_means = np.load('./gemma_export_layer0/X_activations.npy')[0]  # Use first sample
Wq = np.load('./gemma_export_layer0/Wq_grouped.npy')[:, 0]  # First group
Wk = np.load('./gemma_export_layer0/Wk_reshaped.npy')[0]
Wv = np.load('./gemma_export_layer0/Wv_reshaped.npy')[0]
```

Then run:
```bash
python quantize_qkv.py --critical-dim-pct 0.15
```

## What Was Changed

### export_qkv_gqa.py

1. **Attention module detection**: Now checks for both `self_attn` and `attention`
2. **Config attributes**: Supports `num_query_heads`, `num_kv_heads`, explicit `head_dim`
3. **Dtype fallback**: Automatically tries float16 if bfloat16 fails
4. **Better error messages**: Warns about gemma-3 models

### New Files

- `verify_model_compatibility.py`: Quick config-based check (no model loading)
- `test_gemma_arch.py`: Full architecture test (loads model)
- `GEMMA_SUPPORT.md`: Detailed documentation
- `GEMMA_QUICKSTART.md`: This file

## Troubleshooting

### "Model not found"
- Check spelling: `gemma-2-2b-it` not `gemma-3-4b-it`
- Verify on HuggingFace: https://huggingface.co/google/gemma-2-2b-it

### "Out of memory"
- Use smaller model: gemma-2-2b-it instead of gemma-2-9b-it
- Reduce samples: `--n-samples 64` or `--n-samples 32`

### "bfloat16 not supported"
- This is normal! Code automatically falls back to float16
- No action needed

## Files Overview

| File | Purpose | Loads Model? |
|------|---------|--------------|
| `verify_model_compatibility.py` | Quick compatibility check | No (fast) |
| `test_gemma_arch.py` | Full architecture test | Yes (slow) |
| `export_qkv_gqa.py` | Export Q/K/V data | Yes |
| `quantize_qkv.py` | Quantize exported data | No |

## Example: Complete Workflow

```bash
# 1. Verify model exists and is compatible
python verify_model_compatibility.py google/gemma-2-2b-it

# 2. (Optional) Full architecture test
python test_gemma_arch.py google/gemma-2-2b-it

# 3. Export data from Gemma model
python export_qkv_gqa.py \
    --model-path google/gemma-2-2b-it \
    --layer-id 0 \
    --n-samples 128 \
    --output-dir ./gemma_layer0

# 4. Prepare for quantization
mv gemma_layer0 xspot_layer0_group0

# 5. Run quantization
python quantize_qkv.py \
    --critical-dim-pct 0.15 \
    --group-size 128

# 6. View results
# Check: attention_quantization_analysis.png
# Check: sorted_error_comparison.png
# Check: quantization_results.npz
```

## Need Help?

See `GEMMA_SUPPORT.md` for detailed documentation.
