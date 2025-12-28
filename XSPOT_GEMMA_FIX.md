# Fix for Gemma-2 xspot.py Error

## Problem

When running `xspot.py` with Gemma-2-2b-it, the script crashed with:

```
RuntimeError: shape '[8, 288, 2304]' is invalid for input of size 4718592
```

## Root Cause

**Gemma-2 models use a different head dimension than `hidden_size // num_heads`.**

For Gemma-2-2b-it:
- `hidden_size` = 2304
- `num_attention_heads` = 8
- Naive calculation: `head_dim` = 2304 / 8 = **288** ❌
- Actual from weights: `Wq.shape[0]` / `num_heads` = 2048 / 8 = **256** ✓

The script was using the naive calculation, causing reshape errors.

## Solution

Compute `head_dim` from the **actual weight shapes** instead of the config:

```python
# OLD (incorrect for Gemma-2)
self.head_dim = self.hidden_size // self.num_heads  # 2304 / 8 = 288

# NEW (correct for all models)
q_proj_out_features = self.q_proj.weight.shape[0]  # 2048 for Gemma-2-2b
self.head_dim = q_proj_out_features // self.num_heads  # 2048 / 8 = 256
```

## Changes Made to xspot.py

### 1. Fixed head_dim Calculation (lines 55-68)

```python
# Compute head_dim from actual weight shapes (more reliable for models like Gemma-2)
# where head_dim may not equal hidden_size // num_heads
q_proj_out_features = self.q_proj.weight.shape[0]
self.head_dim = q_proj_out_features // self.num_heads
```

### 2. Added Attention Module Detection (lines 82-102)

Now supports both `self_attn` and `attention` attributes:

```python
layer = base_model.layers[self.layer_id]
if hasattr(layer, 'self_attn'):
    return layer.self_attn
elif hasattr(layer, 'attention'):
    return layer.attention
```

### 3. Added Config Attribute Fallbacks (lines 55-61)

Supports alternative config names:

```python
self.num_heads = getattr(self.config, 'num_attention_heads', 
                         getattr(self.config, 'num_query_heads', None))
self.num_key_value_heads = getattr(self.config, 'num_key_value_heads',
                                   getattr(self.config, 'num_kv_heads', self.num_heads))
```

### 4. Added dtype Fallback (lines 456-470)

Automatically tries float16 if bfloat16 fails:

```python
try:
    model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16, ...)
except Exception as e:
    model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16, ...)
```

## Verification

For Gemma-2-2b-it, the fix ensures:

```
Original weight shapes:
  Wq: torch.Size([2048, 2304])  # num_heads * head_dim = 8 * 256 = 2048 ✓
  Wk: torch.Size([1024, 2304])  # num_kv_heads * head_dim = 4 * 256 = 1024 ✓
  Wv: torch.Size([1024, 2304])  # num_kv_heads * head_dim = 4 * 256 = 1024 ✓

Reshape to:
  Wq: [8, 256, 2304]  # Works! 8 * 256 * 2304 = 4,718,592 ✓
  Wk: [4, 256, 2304]  # Works! 4 * 256 * 2304 = 2,359,296 ✓
  Wv: [4, 256, 2304]  # Works! 4 * 256 * 2304 = 2,359,296 ✓
```

## Now Works With

✓ **Llama-2/3** (head_dim = hidden_size / num_heads)
✓ **Gemma-2** (head_dim ≠ hidden_size / num_heads)
✓ **Mistral** and other GQA models

## Usage

```bash
# Now works with Gemma-2-2b-it
python xspot.py --model-path google/gemma-2-2b-it --layer-id 0 --group-id 0

# Also works with Gemma-2-9b-it
python xspot.py --model-path google/gemma-2-9b-it --layer-id 0 --group-id 0

# And still works with Llama
python xspot.py --model-path meta-llama/Llama-2-7b-hf --layer-id 0 --group-id 0
```

## Why This Matters

Some models (like Gemma-2) don't follow the standard pattern where:
```
head_dim = hidden_size / num_attention_heads
```

Instead, they may use:
- Smaller head dimensions to reduce parameters
- Different projection sizes for efficiency
- Non-standard architecture choices

The fix makes the code **robust to any model** by reading the actual weight shapes instead of assuming a formula.
