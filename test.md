# Test Pipeline: AWQ Quantization → vLLM Conversion → Benchmark

Toàn bộ pipeline từ download model đến so sánh benchmark và inference speed.

---

## 0. Setup môi trường

```bash
# Tạo thư mục làm việc
cd ~/Work/Compression/FPRAG

# Cài dependencies (nếu chưa có)
pip install torch transformers datasets safetensors tqdm psutil
pip install vllm                          # cần cho convert + inference benchmark
pip install huggingface_hub               # cần cho download model
```

---

## 1. Download Models

### 1a. Mistral-7B-v0.3

```bash
mkdir -p ./models

# Cách 1: dùng huggingface-cli (khuyên dùng)
huggingface-cli download mistralai/Mistral-7B-v0.3 \
    --local-dir ./models/Mistral-7B-v0.3 \
    --local-dir-use-symlinks False

# Cách 2: dùng Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mistralai/Mistral-7B-v0.3',
    local_dir='./models/Mistral-7B-v0.3',
    local_dir_use_symlinks=False,
)
"
```

### 1b. Llama-3-8B (Meta-Llama-3-8B-Instruct)

> **Lưu ý:** Meta yêu cầu accept license trên HuggingFace trước khi download.
> Vào: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct → Accept license
> Sau đó login:

```bash
huggingface-cli login    # nhập HF token

huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir ./models/Llama-3-8B-Instruct \
    --local-dir-use-symlinks False
```

### Kiểm tra download thành công

```bash
ls ./models/Mistral-7B-v0.3/
ls ./models/Llama-3-8B-Instruct/
# Phải thấy: config.json, tokenizer.json, *.safetensors
```

---

## 2. Quantize Models

### 2a. Standard AWQ (awq_stand_xl) — không có heuristic flip

```bash
# Mistral-7B
python awq_stand_xl.py \
    --model-path  ./models/Mistral-7B-v0.3 \
    --output-dir  ./quantized_models/mistral_awq_stand \
    --calib-dataset c4 \
    --n-calib 128 \
    --n-grid  20 \
    --group-size 128 \
    --layer-batch-size 16 \
    --lmhead-chunks 4

# Llama-3-8B
python awq_stand_xl.py \
    --model-path  ./models/Llama-3-8B-Instruct \
    --output-dir  ./quantized_models/llama3_awq_stand \
    --calib-dataset c4 \
    --n-calib 128 \
    --n-grid  20 \
    --group-size 128 \
    --layer-batch-size 16 \
    --lmhead-chunks 4
```

### 2b. James-Stein Heuristic AWQ (awq_js_xl) — có flip + JS estimator

```bash
# Mistral-7B
python awq_js_xl.py \
    --model-path  ./models/Mistral-7B-v0.3 \
    --output-dir  ./quantized_models/mistral_awq_js \
    --calib-dataset c4 \
    --n-calib 128 \
    --n-grid  20 \
    --group-size 128 \
    --layer-batch-size 16 \
    --lmhead-chunks 4 \
    --use-heuristic \
    --use-james-stein \
    --max-flip-percent 0.05

# Llama-3-8B
python awq_js_xl.py \
    --model-path  ./models/Llama-3-8B-Instruct \
    --output-dir  ./quantized_models/llama3_awq_js \
    --calib-dataset c4 \
    --n-calib 128 \
    --n-grid  20 \
    --group-size 128 \
    --layer-batch-size 16 \
    --lmhead-chunks 4 \
    --use-heuristic \
    --use-james-stein \
    --max-flip-percent 0.05
```

> **Nếu OOM:** giảm `--layer-batch-size 8` hoặc `--n-calib 64`

---

## 3. Convert sang vLLM AWQ INT4 format

```bash
# Mistral — Standard AWQ
python convert_custom_awq_to_vllm.py \
    --source    ./quantized_models/mistral_awq_stand \
    --output    ./quantized_models/mistral_awq_stand_vllm \
    --group-size 128 \
    --validate

# Mistral — JS Heuristic AWQ
python convert_custom_awq_to_vllm.py \
    --source    ./quantized_models/mistral_awq_js \
    --output    ./quantized_models/mistral_awq_js_vllm \
    --group-size 128 \
    --validate

# Llama-3 — Standard AWQ
python convert_custom_awq_to_vllm.py \
    --source    ./quantized_models/llama3_awq_stand \
    --output    ./quantized_models/llama3_awq_stand_vllm \
    --group-size 128 \
    --validate

# Llama-3 — JS Heuristic AWQ
python convert_custom_awq_to_vllm.py \
    --source    ./quantized_models/llama3_awq_js \
    --output    ./quantized_models/llama3_awq_js_vllm \
    --group-size 128 \
    --validate
```

---

## 4. Kiểm tra benchmark đúng không (compare_benchmark.py)

### 4a. Kiểm tra BOS bug + Sliding Window trên FP16 baseline

```bash
# Mistral
python compare_benchmark.py \
    --model-path  ./models/Mistral-7B-v0.3 \
    --fp16dq-path ./quantized_models/mistral_awq_js \
    --vllm-path   ./quantized_models/mistral_awq_js_vllm \
    --group-size 128 \
    --check all

# Llama-3
python compare_benchmark.py \
    --model-path  ./models/Llama-3-8B-Instruct \
    --fp16dq-path ./quantized_models/llama3_awq_js \
    --vllm-path   ./quantized_models/llama3_awq_js_vllm \
    --group-size 128 \
    --check all
```

### 4b. Chỉ check BOS (nếu nghi PPL cao bất thường)

```bash
python compare_benchmark.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --check bos
```

### 4c. Chỉ check roundtrip convert (không cần load model lớn)

```bash
python compare_benchmark.py \
    --fp16dq-path ./quantized_models/mistral_awq_js \
    --vllm-path   ./quantized_models/mistral_awq_js_vllm \
    --group-size 128 \
    --check roundtrip
```

---

## 5. Benchmark PPL — Standard vs Heuristic (compare_awq_slicing.py)

```bash
# Mistral: Standard vs JS Heuristic
python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/mistral_awq_js \
    --standard-path  ./quantized_models/mistral_awq_stand \
    --cache-dir ./dataset_cache

# Llama-3: Standard vs JS Heuristic
python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/llama3_awq_js \
    --standard-path  ./quantized_models/llama3_awq_stand \
    --cache-dir ./dataset_cache
```

---

## 6. So sánh inference speed (compare_inference.py)

### 6a. Mistral-7B: FP16 vs FP16-dequant vs vLLM INT4, batch size 1 và 16

```bash
python compare_inference.py \
    --fp16-path   ./models/Mistral-7B-v0.3 \
    --fp16dq-path ./quantized_models/mistral_awq_js \
    --vllm-path   ./quantized_models/mistral_awq_js_vllm \
    --vllm-quant  awq \
    --n-tokens    128 \
    --n-warmup    2 \
    --batch-sizes 1,16
```

> Nếu có model AWQ tải từ HuggingFace (vd: TheBloke/Mistral-7B-v0.3-AWQ):
> ```bash
> python compare_inference.py \
>     --fp16-path   ./models/Mistral-7B-v0.3 \
>     --hf-awq-path ./models/Mistral-7B-v0.3-AWQ \
>     --vllm-path   ./quantized_models/mistral_awq_js_vllm \
>     --vllm-quant  awq \
>     --n-tokens    128 \
>     --batch-sizes 1,16
> ```

### 6b. Thử vLLM với Marlin kernel (nhanh hơn AWQ thường)

```bash
python compare_inference.py \
    --fp16-path   ./models/Mistral-7B-v0.3 \
    --vllm-path   ./quantized_models/mistral_awq_js_vllm \
    --vllm-quant  awq_marlin \
    --n-tokens    256 \
    --batch-sizes 1,16
```

> Nếu gặp lỗi PTX version mismatch, xem hướng dẫn fix:
> ```bash
> python compare_inference.py --help-marlin
> ```

### 6c. Llama-3-8B inference

```bash
python compare_inference.py \
    --fp16-path   ./models/Llama-3-8B-Instruct \
    --fp16dq-path ./quantized_models/llama3_awq_js \
    --vllm-path   ./quantized_models/llama3_awq_js_vllm \
    --vllm-quant  awq \
    --n-tokens 128
```

### 6d. Dùng long prompts (test với input dài), batch size 1 và 16

```bash
python compare_inference.py \
    --fp16-path   ./models/Mistral-7B-v0.3 \
    --vllm-path   ./quantized_models/mistral_awq_js_vllm \
    --use-long \
    --n-tokens    256 \
    --batch-sizes 1,16
```

---

## 7. Kết quả kỳ vọng

| Model | Metric | FP16 | AWQ FP16-dq | vLLM AWQ INT4 |
|-------|--------|------|-------------|---------------|
| Mistral-7B | PPL (Wiki) | ~5.25 | ~5.4–5.6 | ~5.4–5.6 |
| Mistral-7B | VRAM | ~14GB | ~14GB | ~4.5GB |
| Mistral-7B | Throughput | 1x | ~1x | ~2.5–4x |
| Llama-3-8B | PPL (Wiki) | ~6.14 | ~6.3–6.5 | ~6.3–6.5 |
| Llama-3-8B | VRAM | ~16GB | ~16GB | ~5GB |

> - PPL vLLM AWQ phải **sát với FP16-dequant** (< 0.1 ppt khác biệt)
> - Nếu PPL vLLM cao hơn nhiều → lỗi convert (check roundtrip)
> - Nếu PPL FP16 quá cao (>10) → BOS bug (chạy `--check bos`)

---

## 8. Cấu trúc thư mục sau khi hoàn thành

```
./models/
├── Mistral-7B-v0.3/          # FP16 gốc (~14GB)
└── Llama-3-8B-Instruct/      # FP16 gốc (~16GB)

./quantized_models/
├── mistral_awq_stand/         # FP16-dequant standard AWQ (~14GB)
├── mistral_awq_js/            # FP16-dequant JS heuristic AWQ (~14GB)
├── mistral_awq_stand_vllm/    # vLLM INT4 format (~4.5GB)
├── mistral_awq_js_vllm/       # vLLM INT4 format (~4.5GB)
├── llama3_awq_stand/          # FP16-dequant standard AWQ (~16GB)
├── llama3_awq_js/             # FP16-dequant JS heuristic AWQ (~16GB)
├── llama3_awq_stand_vllm/     # vLLM INT4 format (~5GB)
└── llama3_awq_js_vllm/        # vLLM INT4 format (~5GB)

./dataset_cache/               # WikiText-2, C4 cache (tự động tạo)
```

---

## Troubleshooting

### OOM trong khi quantize
```bash
# Giảm batch size và số calib samples
python awq_js_xl.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --output-dir ./quantized_models/mistral_awq_js \
    --layer-batch-size 8 \     # giảm từ 16
    --n-calib 64 \             # giảm từ 128
    --lmhead-chunks 8          # tăng chunks cho lm_head
```

### vLLM import error
```bash
pip install vllm --upgrade
# Hoặc cài từ nightly nếu cần CUDA version mới hơn
pip install vllm --pre
```

### PPL bất thường cao (>15)
```bash
# Chạy BOS check
python compare_benchmark.py --model-path <path> --check bos
# Nếu double_bos_detected = True → bug trong tokenizer setup
```

### convert_custom_awq_to_vllm lỗi import vllm.model_executor
```bash
# awq_pack cần vLLM >= 0.4.0
pip show vllm | grep Version
# Nếu version thấp → nâng cấp hoặc implement awq_pack thủ công
```
