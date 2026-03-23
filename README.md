# LLM Weight Quantization: Dynamic Heuristic AWQ

Research project comparing INT4 weight quantization methods for Large Language Models, focusing on activation-aware scaling and heuristic rounding correction.

## Overview

Standard AWQ rounds each weight independently to the nearest grid point. This project investigates two improvements:

1. **Better outlier detection** — Kneedle algorithm adapts the outlier threshold per layer instead of using a fixed percentage.
2. **Heuristic rounding correction** — Global greedy rounding minimizes the total output error `x·w` instead of per-weight error.

### Methods

| Method | Script | Salience | Rounding | Outlier Detection |
|--------|--------|----------|----------|-------------------|
| Standard AWQ | `awq_stand_xl.py` | L2 (E[X²]) | Nearest | — |
| Standard Heuristic AWQ | `awq_sh_xl.py` | L2 | Heuristic greedy | Fixed 5% |
| **Dynamic Heuristic AWQ** | `awq_dh_xl.py` | L2 | Heuristic greedy | Kneedle (adaptive) |
| James-Stein AWQ | `awq_js_xl.py` | L2 | Heuristic greedy | Kneedle + JS shrinkage |

All methods use **group-wise asymmetric INT4** quantization with the `[0, 15]` range and batched sequential processing for memory efficiency.

---

## Results

Perplexity (↓ lower is better):

### Mistral-7B-v0.3

| Dataset    | FP16   | Standard AWQ | Dynamic AWQ |
|------------|--------|--------------|-------------|
| WikiText-2 | 4.8454 | 4.9778       | **4.9689**  |
| C4         | 7.6040 | 7.7892       | **7.7830**  |

### Llama-3-8B

| Dataset    | FP16   | Standard AWQ | Dynamic AWQ |
|------------|--------|--------------|-------------|
| WikiText-2 | 5.4425 | 6.7386       | **6.6863**  |
| C4         | 8.6383 | 10.4595      | **10.3014** |

### Llama-2-7B

| Dataset    | FP16   | Standard AWQ | Dynamic AWQ |
|------------|--------|--------------|-------------|
| WikiText-2 | 4.9712 | 5.1280       | **5.1270**  |
| C4         | 6.5748 | 6.7986       | **6.7983**  |

### Qwen2.5-7B

| Dataset    | FP16    | Standard AWQ | Dynamic AWQ |
|------------|---------|--------------|-------------|
| WikiText-2 | 23.1382 | 24.0180      | **23.3029** |
| C4         | 36.1769 | 37.5713      | **36.4447** |

**Summary:** Dynamic AWQ consistently outperforms Standard AWQ, with the largest gains on Qwen2.5-7B (+7–11%) and Llama-3-8B (+5.2% C4).

---

## Inference Speed (Mistral-7B-v0.3, RTX 4090)

After converting to vLLM AWQ format with Marlin kernel:

| Model | VRAM | Throughput (bs=1) | Throughput (bs=16) | Speedup |
|-------|------|-------------------|--------------------|---------|
| FP16 (HuggingFace) | 14 GB | 18 tok/s | 212 tok/s | 1.0× |
| AWQ INT4 (vLLM GEMM) | 20 GB* | 21 tok/s | 308 tok/s | 1.4× |
| AWQ INT4 (vLLM Marlin) | 20 GB* | 175 tok/s | 2553 tok/s | **12×** |

\* Higher total VRAM because vLLM pre-allocates KV cache with `gpu_memory_utilization=0.8`.

---

## Pipeline

```
1. Quantize  →  2. Evaluate PPL  →  3. Convert to vLLM  →  4. Benchmark speed
```

### Step 1 — Quantize

```bash
# Standard AWQ
python awq_stand_xl.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --output-dir ./quantized_models/mistral_awq_standard \
    --n-calib 128 --layer-batch-size 16

# Dynamic Heuristic AWQ (recommended)
python awq_dh_xl.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --output-dir ./quantized_models/mistral_awq_dh \
    --n-calib 128 --layer-batch-size 16

# James-Stein AWQ
python awq_js_xl.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --output-dir ./quantized_models/mistral_awq_js \
    --n-calib 128 --layer-batch-size 16
```

### Step 2 — Evaluate perplexity

```bash
python compare_awq_slicing.py \
    --standard-path ./quantized_models/mistral_awq_standard \
    --heuristic-path ./quantized_models/mistral_awq_dh \
    --n-samples 2000
```

### Step 3 — Convert to vLLM format

```bash
python convert_custom_awq_to_vllm.py \
    --input-dir  ./quantized_models/mistral_awq_js \
    --output-dir ./quantized_models/mistral_awq_js_vllm \
    --group-size 128
```

### Step 4 — Benchmark inference speed

```bash
# Compare FP16, HF AWQ (vLLM), and Custom AWQ (vLLM) with GEMM kernel
python compare_inference.py \
    --fp16-path    ./models/Mistral-7B-v0.3 \
    --hf-awq-path  ./models/Mistral-7B-v0.3-AWQ \
    --vllm-path    ./quantized_models/mistral_awq_js_vllm \
    --vllm-quant   awq \
    --n-tokens 128 --batch-sizes 1,16 --max-model-len 4096

# With Marlin kernel (requires CUDA 12.5+, sm_80+)
python compare_inference.py \
    --fp16-path    ./models/Mistral-7B-v0.3 \
    --hf-awq-path  ./models/Mistral-7B-v0.3-AWQ \
    --vllm-path    ./quantized_models/mistral_awq_js_vllm \
    --vllm-quant   awq_marlin \
    --n-tokens 128 --batch-sizes 1,16 --max-model-len 4096
```

---

## Key Parameters

### Quantization (`awq_dh_xl.py`, `awq_js_xl.py`, `awq_stand_xl.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-calib` | 128 | Calibration samples |
| `--n-grid` | 20 | Grid search points for α (AWQ scaling exponent) |
| `--group-size` | 128 | Channels per quantization group |
| `--layer-batch-size` | 16 | Layers processed per batch |
| `--lmhead-chunks` | 4 | Chunks to split lm_head (avoids OOM) |
| `--calib-dataset` | `c4` | Calibration data (`c4`, `wikitext2`, `wikitext2-simple`) |
| `--knee-tolerance` | 0.0 | Kneedle outlier threshold (Dynamic AWQ only) |
| `--max-flip-percent` | 0.01 | Max weight flips per output channel (Heuristic only) |

### Inference benchmark (`compare_inference.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vllm-quant` | `awq` | vLLM quantization kernel (`awq`, `awq_marlin`) |
| `--batch-sizes` | `1,16` | Comma-separated batch sizes to benchmark |
| `--n-tokens` | 128 | Output tokens per request |
| `--max-model-len` | 4096 | Max context length (affects KV cache allocation) |
| `--gpu-mem-util` | 0.85 | vLLM GPU memory utilization |

---

## Memory Guide

Memory per quantization batch ≈ `layer_batch_size × 280 MB`:

| `--layer-batch-size` | Peak VRAM |
|---------------------|-----------|
| 8 | ~8 GB |
| 16 | ~12 GB |
| 32 | ~20 GB |
| 64 | ~28 GB |

For limited VRAM (8–12 GB):
```bash
python awq_dh_xl.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --layer-batch-size 8 \
    --lmhead-chunks 8 \
    --n-calib 64 \
    --calib-dataset wikitext2-simple
```

---

## How It Works

### AWQ Scaling (all methods)

For each linear layer, find scaling exponent α ∈ [0, 1] that minimizes quantization error:

```
s[j] = E[X[:,j]²]^α          # per input-channel scale
W_scaled[:,j] = W[:,j] * s[j]
W_quant = round(W_scaled / scale + zero_point) * scale - zero_point
W_final[:,j] = W_quant[:,j] / s[j]
```

Using `E[X²]` (L2) rather than `E[|X|]` (L1) aligns with the MSE objective of quantization.

### Heuristic Rounding Correction

After standard rounding, greedily flip individual weights to reduce `‖x · (w - w_quant)‖`:

```
error = x · (w_nearest - w_true)
for each candidate flip (sorted by cost):
    if flip reduces |error|: accept
```

### Dynamic Outlier Detection (Kneedle)

Instead of a fixed 5% outlier threshold, fit the sorted activation distribution with the Kneedle algorithm to adaptively find the "knee point" — the natural boundary between normal and outlier channels. Prevents over-masking on layers with few outliers and under-masking on layers with many.

---

## File Structure

```
├── awq_stand_xl.py          # Standard AWQ (L2 salience, nearest rounding)
├── awq_sh_xl.py             # Standard Heuristic AWQ (fixed 5% outliers)
├── awq_dh_xl.py             # Dynamic Heuristic AWQ (Kneedle outlier detection)
├── awq_js_xl.py             # James-Stein AWQ (JS shrinkage + Kneedle)
├── calibration_utils.py     # C4 / WikiText-2 data loading
├── convert_custom_awq_to_vllm.py  # Convert FP16-dequant → vLLM AWQ format
├── compare_awq_slicing.py   # Perplexity evaluation
├── compare_inference.py     # Inference speed benchmark (subprocess-isolated)
└── explain.md               # Line-by-line code explanation
```

---

## Requirements

```bash
pip install torch transformers datasets tqdm
pip install vllm              # for inference benchmark
pip install safetensors scipy matplotlib
```

- Python 3.8+, PyTorch 2.0+, Transformers 4.35+
- CUDA 11.8+ (GEMM kernel), CUDA 12.5+ (Marlin kernel)
- GPU: 16 GB+ VRAM recommended

---

## License

MIT
