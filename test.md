# Test Pipeline: AWQ / FLUTE Quantization → vLLM → Benchmark

Toàn bộ pipeline từ download model đến benchmark PPL và inference speed.

---

## 0. Setup môi trường

```bash
cd ~/Work/Compression/FPRAG

# Core dependencies
pip install torch transformers datasets safetensors tqdm psutil scipy jaxtyping

# vLLM (cần cho convert AWQ INT4 + inference benchmark)
pip install vllm

# FLUTE (cần cho INT3 inference — chọn đúng CUDA version)
pip install flute-kernel -i https://flute-ai.github.io/whl/cu124   # CUDA 12.4
pip install flute-kernel                                            # CUDA 12.1 (default)

# HuggingFace Hub (cần cho download)
pip install huggingface_hub
```

---

## 1. Download Models

### 1a. Mistral-7B-v0.3 (FP16)

```bash
mkdir -p ./models

# Dùng Python (nếu huggingface-cli không có)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mistralai/Mistral-7B-v0.3',
    local_dir='./models/Mistral-7B-v0.3',
    local_dir_use_symlinks=False,
)
"
```

### 1b. Llama-3-8B-Instruct (FP16)

> **Lưu ý:** Cần accept license trên HuggingFace trước khi download.
> Vào https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct → Accept

```bash
# Login với HF token
python -c "from huggingface_hub import login; login()"

# Download
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    local_dir='./models/Llama-3-8B-Instruct',
    local_dir_use_symlinks=False,
)
"
```

### 1c. Mistral-7B-AWQ từ HuggingFace (optional, để compare)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='solidrust/Mistral-7B-v0.3-AWQ',
    local_dir='./models/Mistral-7B-v0.3-AWQ',
    local_dir_use_symlinks=False,
)
"
```

### Kiểm tra download

```bash
ls ./models/Mistral-7B-v0.3/
ls ./models/Llama-3-8B-Instruct/
# Phải thấy: config.json, tokenizer.json, *.safetensors
```

---

## 2. Quantize — INT4

### 2a. Standard AWQ (không có heuristic)

```bash
# Mistral-7B → INT4
python awq_stand_xl.py \
    --model-path  ./models/Mistral-7B-v0.3 \
    --output-dir  ./quantized_models/mistral_awq_stand \
    --calib-dataset c4 \
    --n-calib 128 --n-grid 20 --group-size 128 \
    --layer-batch-size 16 --lmhead-chunks 4

# Llama-3-8B → INT4
python awq_stand_xl.py \
    --model-path  ./models/Llama-3-8B-Instruct \
    --output-dir  ./quantized_models/llama3_awq_stand \
    --calib-dataset c4 \
    --n-calib 128 --n-grid 20 --group-size 128 \
    --layer-batch-size 16 --lmhead-chunks 4
```

### 2b. James-Stein Heuristic AWQ (INT4, recommended)

```bash
# Mistral-7B → INT4
python awq_js_xl.py \
    --model-path  ./models/Mistral-7B-v0.3 \
    --output-dir  ./quantized_models/mistral_awq_js \
    --calib-dataset c4 \
    --n-calib 128 --n-grid 20 --group-size 128 \
    --layer-batch-size 16 --lmhead-chunks 4 \
    --use-heuristic --use-james-stein --max-flip-percent 0.05

# Llama-3-8B → INT4
python awq_js_xl.py \
    --model-path  ./models/Llama-3-8B-Instruct \
    --output-dir  ./quantized_models/llama3_awq_js \
    --calib-dataset c4 \
    --n-calib 128 --n-grid 20 --group-size 128 \
    --layer-batch-size 16 --lmhead-chunks 4 \
    --use-heuristic --use-james-stein --max-flip-percent 0.05
```

> **OOM?** Giảm `--layer-batch-size 8` hoặc `--n-calib 64`

### 2c. James-Stein Heuristic AWQ — INT3

> INT3 lưu trong FP16 (fake-quantize, dùng cho PPL eval).
> Để inference nhanh phải convert sang FLUTE (bước 3b).

```bash
# Mistral-7B → INT3
python awq_js_xl.py \
    --model-path  ./models/Mistral-7B-v0.3 \
    --output-dir  ./quantized_models/mistral_awq_js_int3 \
    --calib-dataset c4 \
    --n-calib 128 --n-grid 20 --group-size 128 \
    --bits 3 \
    --layer-batch-size 16 --lmhead-chunks 4 \
    --use-heuristic --use-james-stein --max-flip-percent 0.05

# Llama-3-8B → INT3
python awq_js_xl.py \
    --model-path  ./models/Llama-3-8B-Instruct \
    --output-dir  ./quantized_models/llama3_awq_js_int3 \
    --calib-dataset c4 \
    --n-calib 128 --n-grid 20 --group-size 128 \
    --bits 3 \
    --layer-batch-size 16 --lmhead-chunks 4 \
    --use-heuristic --use-james-stein --max-flip-percent 0.05
```

---

## 3. Convert sang vLLM / FLUTE format

### 3a. Convert → vLLM AWQ INT4 format

```bash
# Mistral — Standard AWQ INT4
python convert_custom_awq_to_vllm.py \
    --source ./quantized_models/mistral_awq_stand \
    --output ./quantized_models/mistral_awq_stand_vllm \
    --group-size 128 --validate

# Mistral — JS Heuristic AWQ INT4
python convert_custom_awq_to_vllm.py \
    --source ./quantized_models/mistral_awq_js \
    --output ./quantized_models/mistral_awq_js_vllm \
    --group-size 128 --validate

# Llama-3 — Standard AWQ INT4
python convert_custom_awq_to_vllm.py \
    --source ./quantized_models/llama3_awq_stand \
    --output ./quantized_models/llama3_awq_stand_vllm \
    --group-size 128 --validate

# Llama-3 — JS Heuristic AWQ INT4
python convert_custom_awq_to_vllm.py \
    --source ./quantized_models/llama3_awq_js \
    --output ./quantized_models/llama3_awq_js_vllm \
    --group-size 128 --validate
```

### 3b. Convert → FLUTE INT3 format

> FLUTE dùng lookup-table thay vì uniform grid, hỗ trợ INT2/3/4.
> Kernel tối ưu cho RTX 4090 / A100 (compute capability 80+).

```bash
# Mistral-7B → FLUTE INT3 (từ FP16 gốc, FLUTE tự quantize)
python convert_to_flute_int3.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --output-dir ./quantized_models/mistral_flute_int3 \
    --num-bits 3 --group-size 128 --n-calib 128

# Llama-3-8B → FLUTE INT3
python convert_to_flute_int3.py \
    --model-path ./models/Llama-3-8B-Instruct \
    --output-dir ./quantized_models/llama3_flute_int3 \
    --num-bits 3 --group-size 128 --n-calib 128

# Tuỳ chọn: thêm --learn-scales để fine-tune scales trên calib data (+quality, +time)
python convert_to_flute_int3.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --output-dir ./quantized_models/mistral_flute_int3 \
    --num-bits 3 --group-size 128 --n-calib 128 \
    --learn-scales
```

---

## 4. Kiểm tra roundtrip (compare_benchmark.py)

```bash
# Mistral: kiểm tra convert AWQ INT4 không bị lỗi
python compare_benchmark.py \
    --model-path  ./models/Mistral-7B-v0.3 \
    --fp16dq-path ./quantized_models/mistral_awq_js \
    --vllm-path   ./quantized_models/mistral_awq_js_vllm \
    --group-size 128 --check all

# Llama-3: tương tự
python compare_benchmark.py \
    --model-path  ./models/Llama-3-8B-Instruct \
    --fp16dq-path ./quantized_models/llama3_awq_js \
    --vllm-path   ./quantized_models/llama3_awq_js_vllm \
    --group-size 128 --check all

# Chỉ check roundtrip (nhẹ hơn, không cần load model gốc)
python compare_benchmark.py \
    --fp16dq-path ./quantized_models/mistral_awq_js \
    --vllm-path   ./quantized_models/mistral_awq_js_vllm \
    --group-size 128 --check roundtrip
```

---

## 5. Benchmark PPL — Standard vs Heuristic (compare_awq_slicing.py)

```bash
# Mistral: Standard vs JS Heuristic (INT4)
python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/mistral_awq_js \
    --standard-path  ./quantized_models/mistral_awq_stand \
    --cache-dir ./dataset_cache

# Llama-3: Standard vs JS Heuristic (INT4)
python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/llama3_awq_js \
    --standard-path  ./quantized_models/llama3_awq_stand \
    --cache-dir ./dataset_cache
```

---

## 6. So sánh inference speed (compare_inference.py)

> Mỗi model chạy trong **subprocess riêng** → CUDA context sạch, tránh OOM.
> Dùng `2>&1 | tee out.txt` để lưu log.

### 6a. Mistral-7B: FP16 vs AWQ INT4 — short prompts, AWQ GEMM kernel

```bash
python compare_inference.py \
    --fp16-path    ./models/Mistral-7B-v0.3 \
    --fp16dq-path  ./quantized_models/mistral_awq_js \
    --hf-awq-path  ./models/Mistral-7B-v0.3-AWQ \
    --vllm-path    ./quantized_models/mistral_awq_js_vllm \
    --vllm-quant   awq \
    --n-tokens 128 --n-warmup 2 \
    --batch-sizes  1,16 \
    --max-model-len 4096
```

### 6b. Mistral-7B: AWQ Marlin kernel (nhanh hơn GEMM ~10x)

```bash
python compare_inference.py \
    --fp16-path    ./models/Mistral-7B-v0.3 \
    --hf-awq-path  ./models/Mistral-7B-v0.3-AWQ \
    --vllm-path    ./quantized_models/mistral_awq_js_vllm \
    --vllm-quant   awq_marlin \
    --n-tokens 128 --batch-sizes 1,16 \
    --max-model-len 4096
```

> Nếu lỗi PTX version mismatch (CUDA < 12.5):
> ```bash
> python compare_inference.py --help-marlin
> ```

### 6c. Llama-3-8B: short prompts, AWQ GEMM

```bash
python compare_inference.py \
    --fp16-path    ./models/Llama-3-8B-Instruct \
    --fp16dq-path  ./quantized_models/llama3_awq_js \
    --vllm-path    ./quantized_models/llama3_awq_js_vllm \
    --vllm-quant   awq \
    --n-tokens 128 --batch-sizes 1,16 \
    --max-model-len 4096
```

### 6d. Llama-3-8B: AWQ Marlin kernel

```bash
python compare_inference.py \
    --fp16-path    ./models/Llama-3-8B-Instruct \
    --vllm-path    ./quantized_models/llama3_awq_js_vllm \
    --vllm-quant   awq_marlin \
    --n-tokens 128 --batch-sizes 1,16 \
    --max-model-len 4096
```

### 6e. Mistral-7B: long prompts — AWQ GEMM vs Marlin

```bash
# GEMM kernel
python compare_inference.py \
    --fp16-path    ./models/Mistral-7B-v0.3 \
    --vllm-path    ./quantized_models/mistral_awq_js_vllm \
    --vllm-quant   awq \
    --use-long \
    --n-tokens 256 --batch-sizes 1,16 \
    --max-model-len 4096

# Marlin kernel
python compare_inference.py \
    --fp16-path    ./models/Mistral-7B-v0.3 \
    --vllm-path    ./quantized_models/mistral_awq_js_vllm \
    --vllm-quant   awq_marlin \
    --use-long \
    --n-tokens 256 --batch-sizes 1,16 \
    --max-model-len 4096
```

### 6f. Llama-3-8B: long prompts — AWQ GEMM vs Marlin

```bash
# GEMM kernel
python compare_inference.py \
    --fp16-path    ./models/Llama-3-8B-Instruct \
    --vllm-path    ./quantized_models/llama3_awq_js_vllm \
    --vllm-quant   awq \
    --use-long \
    --n-tokens 256 --batch-sizes 1,16 \
    --max-model-len 4096

# Marlin kernel
python compare_inference.py \
    --fp16-path    ./models/Llama-3-8B-Instruct \
    --vllm-path    ./quantized_models/llama3_awq_js_vllm \
    --vllm-quant   awq_marlin \
    --use-long \
    --n-tokens 256 --batch-sizes 1,16 \
    --max-model-len 4096
```

### 6g. Mistral-7B: FLUTE INT3 (short prompts)

```bash
python compare_inference.py \
    --fp16-path    ./models/Mistral-7B-v0.3 \
    --vllm-path    ./quantized_models/mistral_awq_js_vllm \
    --vllm-quant   awq_marlin \
    --flute-path   ./quantized_models/mistral_flute_int3 \
    --flute-num-bits 3 \
    --n-tokens 128 --batch-sizes 1,16 \
    --max-model-len 4096
```

### 6h. Mistral-7B: FLUTE INT3 (long prompts)

```bash
python compare_inference.py \
    --fp16-path    ./models/Mistral-7B-v0.3 \
    --vllm-path    ./quantized_models/mistral_awq_js_vllm \
    --vllm-quant   awq_marlin \
    --flute-path   ./quantized_models/mistral_flute_int3 \
    --flute-num-bits 3 \
    --use-long \
    --n-tokens 256 --batch-sizes 1,16 \
    --max-model-len 4096
```

### 6i. Llama-3-8B: FLUTE INT3 (short prompts)

```bash
python compare_inference.py \
    --fp16-path    ./models/Llama-3-8B-Instruct \
    --vllm-path    ./quantized_models/llama3_awq_js_vllm \
    --vllm-quant   awq_marlin \
    --flute-path   ./quantized_models/llama3_flute_int3 \
    --flute-num-bits 3 \
    --n-tokens 128 --batch-sizes 1,16 \
    --max-model-len 4096
```

### 6j. Llama-3-8B: FLUTE INT3 (long prompts)

```bash
python compare_inference.py \
    --fp16-path    ./models/Llama-3-8B-Instruct \
    --vllm-path    ./quantized_models/llama3_awq_js_vllm \
    --vllm-quant   awq_marlin \
    --flute-path   ./quantized_models/llama3_flute_int3 \
    --flute-num-bits 3 \
    --use-long \
    --n-tokens 256 --batch-sizes 1,16 \
    --max-model-len 4096
```

> Để lưu log đầy đủ:
> ```bash
> python compare_inference.py ... 2>&1 | tee results_mistral_flute.txt
> ```

---

## 7. Kết quả kỳ vọng

### Perplexity (WikiText-2, ↓ better)

| Model | FP16 | AWQ INT4 (Standard) | AWQ INT4 (JS) | AWQ INT3 (JS) |
|-------|------|---------------------|---------------|---------------|
| Mistral-7B | ~4.85 | ~4.98 | ~4.97 | ~5.3–5.6 |
| Llama-3-8B | ~5.44 | ~6.74 | ~6.69 | ~7.2–7.8 |

### Inference Speed (RTX 4090, bs=16, ~310 tok/s baseline)

| Method | Kernel | Throughput (bs=16) | Speedup |
|--------|--------|--------------------|---------|
| FP16 (HF) | — | ~215 tok/s | 1.0× |
| AWQ INT4 | GEMM | ~308 tok/s | 1.4× |
| AWQ INT4 | Marlin | ~2500 tok/s | 12× |
| FLUTE INT3 | FLUTE | ~tbd | tbd |

> - FLUTE INT3 dùng ít VRAM hơn AWQ INT4 (~25% nhờ 3-bit vs 4-bit)
> - Nếu PPL vLLM cao hơn FP16-dequant > 0.1 ppt → lỗi convert (chạy `--check roundtrip`)

---

## 8. Cấu trúc thư mục sau khi hoàn thành

```
./models/
├── Mistral-7B-v0.3/              # FP16 gốc (~14GB)
├── Mistral-7B-v0.3-AWQ/          # AWQ INT4 từ HuggingFace (~4.5GB)
└── Llama-3-8B-Instruct/          # FP16 gốc (~16GB)

./quantized_models/
├── mistral_awq_stand/            # FP16-dequant standard AWQ INT4 (~14GB)
├── mistral_awq_js/               # FP16-dequant JS heuristic AWQ INT4 (~14GB)
├── mistral_awq_js_int3/          # FP16-dequant JS heuristic AWQ INT3 (~14GB)
├── mistral_awq_stand_vllm/       # vLLM INT4 GEMM format (~4.5GB)
├── mistral_awq_js_vllm/          # vLLM INT4 GEMM format (~4.5GB)
├── mistral_flute_int3/           # FLUTE INT3 format (~3.5GB)
├── llama3_awq_stand/             # FP16-dequant standard AWQ INT4 (~16GB)
├── llama3_awq_js/                # FP16-dequant JS heuristic AWQ INT4 (~16GB)
├── llama3_awq_js_int3/           # FP16-dequant JS heuristic AWQ INT3 (~16GB)
├── llama3_awq_stand_vllm/        # vLLM INT4 GEMM format (~5GB)
├── llama3_awq_js_vllm/           # vLLM INT4 GEMM format (~5GB)
└── llama3_flute_int3/            # FLUTE INT3 format (~4GB)

./dataset_cache/                  # WikiText-2, C4 cache (tự động tạo)
```

---

## Troubleshooting

### OOM trong khi quantize

```bash
python awq_js_xl.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --output-dir ./quantized_models/mistral_awq_js \
    --layer-batch-size 8 \    # giảm từ 16
    --n-calib 64 \            # giảm từ 128
    --lmhead-chunks 8         # tăng chunks cho lm_head
```

### FLUTE kernel không tìm thấy (compute capability < 80)

```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
# Cần (8, 0) trở lên — A100, RTX 30xx/40xx, A6000
# RTX 20xx (sm_75) không được hỗ trợ
```

### vLLM AWQ Marlin PTX mismatch (CUDA < 12.5)

```bash
python compare_inference.py --help-marlin
```

### PPL bất thường cao (> 15)

```bash
python compare_benchmark.py --model-path <path> --check bos
# double_bos_detected = True → bug tokenizer
```

### convert_custom_awq_to_vllm lỗi import awq_pack

```bash
pip show vllm | grep Version   # cần >= 0.4.0
pip install vllm --upgrade
```
