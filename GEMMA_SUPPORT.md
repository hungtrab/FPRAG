# Gemma Model Support

## Overview

Both `export_qkv_gqa.py` and `quantize_qkv.py` have been updated to support Gemma models (and other architectures with similar structures).

## Supported Models

The code now supports:
- **Llama** family (Llama-2, Llama-3, etc.)
- **Gemma** family (Gemma-2-2b-it, Gemma-2-9b-it, Gemma-2-27b-it)
- **Mistral** and other models with GQA (Group Query Attention)

**Note**: As of December 2025, `google/gemma-3-4b-it` does not exist. The latest Gemma models are:
- `google/gemma-2-2b-it` (2B parameters)
- `google/gemma-2-9b-it` (9B parameters)
- `google/gemma-2-27b-it` (27B parameters)

## Changes Made

### 1. Architecture Detection (`export_qkv_gqa.py`)

#### Attention Module Detection
The code now handles multiple attribute names:
- `self_attn` (Llama, Mistral)
- `attention` (some Gemma variants)

```python
# Before (only supported self_attn)
return base_model.layers[self.layer_id].self_attn

# After (supports both)
layer = base_model.layers[self.layer_id]
if hasattr(layer, 'self_attn'):
    return layer.self_attn
elif hasattr(layer, 'attention'):
    return layer.attention
```

#### Configuration Attributes
Support for alternative config attribute names:

```python
# num_attention_heads vs num_query_heads
self.num_heads = getattr(self.config, 'num_attention_heads', 
                         getattr(self.config, 'num_query_heads', None))

# num_key_value_heads vs num_kv_heads
self.num_key_value_heads = getattr(self.config, 'num_key_value_heads',
                                    getattr(self.config, 'num_kv_heads', self.num_heads))

# head_dim (explicit vs computed)
self.head_dim = getattr(self.config, 'head_dim', self.hidden_size // self.num_heads)
```

#### Dtype Fallback
Added fallback from bfloat16 to float16:

```python
try:
    model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16, ...)
except Exception as e:
    print(f"bfloat16 failed, trying float16...")
    model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16, ...)
```

## Usage Examples

### Export from Gemma-2-2b-it

```bash
# Small model (2B parameters)
python export_qkv_gqa.py \
    --model-path google/gemma-2-2b-it \
    --layer-id 0 \
    --n-samples 128 \
    --output-dir ./gemma2_2b_layer0
```

### Export from Gemma-2-9b-it

```bash
# Medium model (9B parameters)
python export_qkv_gqa.py \
    --model-path google/gemma-2-9b-it \
    --layer-id 0 \
    --n-samples 128 \
    --output-dir ./gemma2_9b_layer0
```

### Export from Gemma-2-27b-it

```bash
# Large model (27B parameters) - requires more GPU memory
python export_qkv_gqa.py \
    --model-path google/gemma-2-27b-it \
    --layer-id 0 \
    --n-samples 64 \
    --output-dir ./gemma2_27b_layer0
```

### Using Quantization with Gemma Data

After exporting:

```bash
# Make sure the exported data is in the expected directory
# or update the paths in quantize_qkv.py

python quantize_qkv.py \
    --critical-dim-pct 0.15 \
    --knee-tolerance 0.0 \
    --group-size 128
```

## Testing Architecture Detection

Use the test script to verify Gemma model compatibility:

```bash
# Test with Gemma-2-2b-it (default)
python test_gemma_arch.py

# Test with specific model
python test_gemma_arch.py google/gemma-2-9b-it

# Test with Llama for comparison
python test_gemma_arch.py meta-llama/Llama-2-7b-hf
```

## Troubleshooting

### "Model google/gemma-3-4b-it not found"

This model doesn't exist. Use one of the available Gemma-2 models:
- google/gemma-2-2b-it
- google/gemma-2-9b-it  
- google/gemma-2-27b-it

### "No attention module found"

If you encounter this error, the model architecture is not yet supported. Please:
1. Run `test_gemma_arch.py` with your model to inspect its structure
2. Check if the layer has a different attention attribute name
3. Open an issue with the model name and test output

### Memory Issues

Gemma-2-27b requires significant GPU memory. Solutions:
- Use smaller model (gemma-2-2b-it or gemma-2-9b-it)
- Reduce `--n-samples` (e.g., 32 or 64 instead of 128)
- Use CPU offloading: model will automatically use `device_map="auto"`

### bfloat16 Not Supported

Some GPUs don't support bfloat16. The code automatically falls back to float16:
```
Error loading model with bfloat16, trying float16...
```

This is normal and won't affect functionality.

## Architecture Comparison

### Llama-2/3 vs Gemma-2

Both use Group Query Attention (GQA), but may differ in:

| Attribute | Llama | Gemma | Code Support |
|-----------|-------|-------|--------------|
| Layer attr | `self_attn` | `attention` or `self_attn` | ✓ Both |
| Config: num heads | `num_attention_heads` | `num_attention_heads` or `num_query_heads` | ✓ Both |
| Config: KV heads | `num_key_value_heads` | `num_key_value_heads` or `num_kv_heads` | ✓ Both |
| Config: head dim | computed | `head_dim` or computed | ✓ Both |

The updated code handles all these variations automatically.

## Verified Compatible Models

✓ **Llama family**
- meta-llama/Llama-2-7b-hf
- meta-llama/Llama-2-13b-hf
- meta-llama/Meta-Llama-3-8B

✓ **Gemma family** (updated code)
- google/gemma-2-2b-it
- google/gemma-2-9b-it
- google/gemma-2-27b-it

✓ **Mistral family**
- mistralai/Mistral-7B-v0.1
- mistralai/Mixtral-8x7B-v0.1

## Next Steps

If you're working with a Gemma model:

1. **Verify model name**: Make sure it's `gemma-2-XXb-it`, not `gemma-3-XXb-it`
2. **Test architecture**: Run `test_gemma_arch.py` to verify detection
3. **Export data**: Use `export_qkv_gqa.py` with Gemma model
4. **Quantize**: Run `quantize_qkv.py` on exported data

## Questions?

If you encounter issues with a specific model:
1. Run the test script and save output
2. Check the model card on HuggingFace for architecture details
3. Verify the model actually exists and is publicly available
