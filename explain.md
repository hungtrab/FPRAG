# Giải thích code: convert + compare_inference

---

## 1. `convert_custom_awq_to_vllm.py`

### Bối cảnh: tại sao cần convert?

`awq_js_xl.py` lưu model dưới dạng **FP16 dequantized** — tức là weight đã được quantize sang INT4 rồi dequantize lại thành FP16:

```
W_original (FP16)
  → scale theo activation salience
  → quantize → INT4 values [0..15]
  → dequantize → W_final (FP16, nhưng nằm đúng trên INT4 grid)
  → lưu bằng model.save_pretrained()
```

vLLM **không thể load FP16**. vLLM cần weight ở dạng packed INT4:
```
qweight : [K, N//8]  int32  — 8 giá trị INT4 nhét vào 1 int32
scales  : [K//g, N]  fp16   — scale per group
qzeros  : [K//g, N//8] int32 — zero point per group, cũng packed
```

`convert_custom_awq_to_vllm.py` làm nhiệm vụ: đọc FP16 dequantized → re-quantize → pack sang format vLLM cần.

**Tại sao re-quantize không làm mất thêm precision?**
Vì `W_final` đã nằm *đúng trên INT4 grid* (là dequant(INT4)), nên re-quantize sẽ recover lại đúng INT4 values gốc. Double-quantization error ≈ 0.

---

### Hằng số và cấu hình

```python
PACK_FACTOR = 8   # 8 INT4 values (4-bit) packed vào 1 int32 (32-bit)

SKIP_IF_NAME_CONTAINS = ["lm_head", "embed", "norm", "layernorm"]
# Không quantize các layer này:
# - lm_head: output projection, quantize sẽ làm lệch logits nghiêm trọng
# - embed: embedding table, không phải linear projection
# - norm/layernorm: chứa gamma/beta, cực nhỏ, không cần quantize

TOKENIZER_FILES = [...]
# Danh sách file tokenizer cần copy sang output dir
# (tokenizer không thay đổi khi quantize)
```

---

### Hàm `quantize_to_awq(W_fp16, group_size)`

Đây là hàm cốt lõi. Input: weight FP16 shape `[N, K]`, output: 3 tensor INT4 packed.

```python
N, K = W_fp16.shape
# N = out_features (số neuron output)
# K = in_features  (số neuron input)
# PyTorch lưu weight linear layer theo convention [out, in]

n_groups = K // group_size
# Chia chiều K (in_features) thành các group 128 channels
# Mỗi group có scale và zero riêng → asymmetric quantization
```

```python
W = W_fp16.float().T.contiguous()   # [K, N] fp32
# .float(): chuyển sang fp32 để tính toán chính xác hơn
# .T: transpose từ [N,K] sang [K,N] — vLLM convention
# .contiguous(): đảm bảo memory layout liên tục sau transpose

W_g = W.reshape(n_groups, group_size, N)  # [n_groups, 128, N]
# Reshape để xử lý từng group một
# Mỗi group gồm 128 channels liên tiếp theo chiều K
```

```python
w_min = W_g.min(dim=1).values   # [n_groups, N] — min per group per output neuron
w_max = W_g.max(dim=1).values   # [n_groups, N] — max per group per output neuron
scales = ((w_max - w_min) / 15.0).clamp(min=1e-8)
# scale = range / 15 vì INT4 asymmetric có 16 levels: 0..15
# .clamp(min=1e-8): tránh chia cho 0 với constant weight

zeros = (-w_min / scales).round().clamp(0, 15)
# zero_point: số bù để shift range về [0,15]
# Công thức: W[i] = (q[i] - zero) * scale
# → zero = -w_min / scale (để q=0 tương ứng với w_min)
# .round(): làm tròn về integer
# .clamp(0,15): đảm bảo nằm trong range INT4
```

```python
s_exp = scales.unsqueeze(1)   # [n_groups, 1, N] — broadcast cho group dimension
z_exp = zeros.unsqueeze(1)    # [n_groups, 1, N]

W_int = (W_g / s_exp + z_exp).round().clamp(0, 15).to(torch.int32)
# Quantize: q = round(W/scale + zero)
# .clamp(0,15): clip về [0,15]
# .to(int32): cần int32 để pack sau

W_int = W_int.reshape(K, N)   # [K, N] — flatten groups lại
```

```python
qweight = awq_pack(W_int.cpu(), 4, K, N)
# awq_pack từ vLLM: pack 8 INT4 values vào 1 int32
# Input:  [K, N] int32 với values 0..15
# Output: [K, N//8] int32
# Layout: col0[3:0] | col1[3:0] | ... | col7[3:0] trong 1 int32
```

```python
pack_order = [0, 2, 4, 6, 1, 3, 5, 7]
# AWQ kernel dùng interleaved packing order (không phải 0,1,2,...,7)
# Lý do: tối ưu memory access pattern của CUDA kernel khi dequantize
# Thứ tự này match với cách AWQ CUDA kernel đọc qzeros

zeros_i = zeros.to(torch.int32).cpu()
qzeros = torch.zeros(n_groups, N // PACK_FACTOR, dtype=torch.int32)
for bit_pos, col_off in enumerate(pack_order):
    qzeros |= (zeros_i[:, col_off::PACK_FACTOR] & 0xF) << (bit_pos * 4)
# Pack zeros theo interleaved order:
# - col_off::PACK_FACTOR: lấy mỗi 8th column bắt đầu từ col_off
# - & 0xF: chỉ lấy 4 bit thấp
# - << (bit_pos * 4): shift vào đúng vị trí trong int32
# - |=: OR vào qzeros (ghép 8 values vào 1 int32)

return qweight, scales.half(), qzeros
# scales.half(): convert về fp16 để tiết kiệm memory
```

---

### Hàm `should_skip(name, shape, group_size)`

```python
if len(shape) != 2:
    return True, "not 2D"
# Chỉ quantize 2D tensor (weight của Linear layer)
# Bias, embedding, v.v. không phải 2D linear weight

if any(s in name for s in SKIP_IF_NAME_CONTAINS):
    return True, "skip pattern"
# Bỏ qua lm_head, embed, norm, layernorm (xem giải thích ở trên)

N, K = shape
if K < group_size or K % group_size != 0:
    return True, f"K={K} not divisible by group_size"
# K phải chia hết cho group_size=128
# Nếu K < 128 (layer quá nhỏ) → skip

if N % PACK_FACTOR != 0:
    return True, f"N={N} not divisible by 8"
# N phải chia hết cho 8 để pack N//8 int32
```

---

### Hàm `validate_layer`

```python
W_deq = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
# Gọi CUDA kernel của vLLM để dequantize ngược lại
# → W_deq phải ≈ W_fp16.T

err = (W_deq - W_ref).abs()
status = "OK" if mean_e < 0.01   # mean error < 0.01 là chấp nhận được
# Nếu double-quantization đúng, mean_err thường < 0.005
```

---

### Hàm `convert()` — main pipeline

```python
sf_files = sorted(source_dir.glob("*.safetensors"))
if sf_files:
    state_dict = {}
    for f in sf_files:
        state_dict.update(load_file(f))
# Hỗ trợ cả multi-shard (nhiều file .safetensors)
# load_file từ safetensors library: nhanh, memory-safe hơn torch.load
else:
    state_dict = torch.load(source_dir / "pytorch_model.bin", map_location="cpu")
# Fallback về .bin cũ nếu không có safetensors
```

```python
weight_keys = [
    k for k, v in state_dict.items()
    if k.endswith(".weight") and v.dim() == 2
]
# Lọc ra chỉ các key là weight của Linear layer
# Ví dụ: "model.layers.0.self_attn.q_proj.weight"
```

```python
qweight, scales, qzeros = quantize_to_awq(W.cuda(), group_size)
# .cuda(): chuyển lên GPU để tính toán nhanh hơn
# Kết quả được trả về trên CPU (trong hàm)

new_sd[f"{prefix}.qweight"] = qweight.cpu()
new_sd[f"{prefix}.scales"]  = scales.cpu()
new_sd[f"{prefix}.qzeros"]  = qzeros.cpu()
# Lưu 3 tensor thay cho 1 tensor .weight gốc
# Ví dụ: "model.layers.0.self_attn.q_proj.weight" (14GB FP16)
# → "...q_proj.qweight" + "...q_proj.scales" + "...q_proj.qzeros" (~1.75GB INT4)
```

```python
quantized_prefixes = {
    k[:-len(".weight")] for k in weight_keys
    if not should_skip(...)[0]
}
for key, val in state_dict.items():
    if key in new_sd:
        continue
    prefix = key.rsplit(".", 1)[0] if "." in key else ""
    if key.endswith(".weight") and prefix in quantized_prefixes:
        continue   # bỏ qua .weight gốc của các layer đã quantize
    new_sd[key] = val
# Copy toàn bộ tensor còn lại (bias, embedding, norm weights, v.v.)
# mà không bị duplicate hay bỏ sót
```

```python
config_dict["quantization_config"] = {
    "quant_method": "awq",   # vLLM nhận diện đây là AWQ model
    "version": "gemm",       # dùng GEMM kernel (thay vì marlin)
    "w_bit": 4,              # 4-bit quantization
    "q_group_size": group_size,   # 128
    "zero_point": True,      # asymmetric (có zero point)
    "modules_to_not_convert": skipped_names,  # layer nào giữ FP16
}
# vLLM đọc config này để biết cách load và dequantize weight
```

---

## 2. `compare_inference.py`

### Thiết kế tổng thể: subprocess isolation

```
Main process (không chạm CUDA)
  ├── Spawn subprocess 1 → bench FP16    → kết thúc → CUDA freed
  ├── Spawn subprocess 2 → bench AWQ HF → kết thúc → CUDA freed
  └── Spawn subprocess 3 → bench vLLM   → kết thúc → CUDA freed
  → Thu thập JSON → in bảng kết quả
```

Lý do cần subprocess: sau khi load model FP16 (~14GB) và `del model + empty_cache()`, PyTorch vẫn **giữ CUDA memory pool** trong process hiện tại. Subprocess vLLM thấy GPU chỉ còn ~9GB free dù model đã bị xóa. Subprocess riêng = CUDA context sạch hoàn toàn.

---

### `make_batch(prompts, batch_size)`

```python
return (prompts * math.ceil(batch_size / len(prompts)))[:batch_size]
# Tile prompts để đủ batch_size phần tử
# Ví dụ: prompts=["A","B","C"], batch_size=8
# → ["A","B","C","A","B","C","A","B"]
# Đảm bảo benchmark luôn dùng đúng batch_size, không phụ thuộc số prompts
```

---

### `nvidia_smi_vram_mb()`

```python
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
return float(out.strip().split("\n")[0])
# nvidia-smi đo tổng VRAM đang dùng trên toàn GPU (mọi process)
# → Chính xác hơn torch.cuda.memory_allocated() chỉ đo process hiện tại
# split("\n")[0]: lấy GPU đầu tiên nếu có nhiều GPU
```

---

### `_bench_hf(model_path, label, prompts, n_tokens, n_warmup, batch_size)`

```python
tokenizer.padding_side = "left"
# Causal LM generation cần left-padding!
# Nếu right-padding: các sequence ngắn hơn sẽ generate từ padding token
# → output sai. Left-padding đặt padding ở đầu, model generate từ cuối.
```

```python
bench_batches = [[p] for p in prompts] if batch_size == 1 else [make_batch(prompts, batch_size)] * 5
# bs=1: chạy từng prompt riêng → đo variance qua các prompt khác nhau
# bs>1: chạy 5 lần cùng batch → đo variance qua các lần chạy
```

```python
inp = tokenizer(batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512).to(device)
input_len = inp["input_ids"].shape[1]
# padding=True: pad tất cả sequence về cùng độ dài (cần cho batch)
# max_length=512: truncate prompt dài
# input_len: cần để tính n_new = output_len - input_len

out = model.generate(**inp, max_new_tokens=n_tokens, do_sample=False, use_cache=True)
# do_sample=False: greedy decoding → deterministic → benchmark ổn định
# use_cache=True: dùng KV cache (default, cần cho performance)

n_new_total = (out.shape[1] - input_len) * len(batch)
# Tổng số token generated = tokens_per_seq × batch_size
# Dùng để tính throughput = total_tokens / elapsed

peak = nvidia_smi_vram_mb()  # đo sau generate để bắt KV cache peak
```

---

### `_bench_vllm(model_path, ...)`

```python
llm = LLM(model=model_path, quantization=quantization, dtype="float16",
          gpu_memory_utilization=gpu_memory_utilization,
          max_model_len=max_model_len)
# gpu_memory_utilization=0.8: vLLM dùng 80% VRAM cho model + KV cache
# max_model_len=4096: giới hạn context length → giảm KV cache pre-allocation
#   Không giới hạn: 32768 tokens → cần ~4GB KV cache → OOM trên 24GB GPU
#   4096 tokens: KV cache ~0.5GB → đủ cho benchmark ngắn

sampling = SamplingParams(temperature=0, max_tokens=n_tokens)
# temperature=0: greedy (deterministic)

outputs = llm.generate(batch, sampling)
# batch: list of strings → vLLM xử lý tất cả cùng lúc với PagedAttention
# Khác HF: không cần padding, mỗi sequence độc lập về memory

n_new_total = sum(len(o.outputs[0].token_ids) for o in outputs)
# o.outputs[0].token_ids: list các token ID được generate
# sum: tổng tất cả sequence trong batch
```

---

### `run_subprocess_benchmark(func, kwargs, label)`

```python
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as cf:
    json.dump({"func": func, "kwargs": kwargs}, cf)
    config_file = cf.name
result_file = config_file + ".result"
# Giao tiếp giữa main process và subprocess qua file JSON tạm
# kwargs chứa: model_path, prompts, n_tokens, batch_size, v.v.
```

```python
proc = subprocess.run(
    [sys.executable, __file__, "--_worker", config_file, result_file],
    text=True,
)
# sys.executable: dùng đúng Python interpreter hiện tại (kể cả virtualenv)
# __file__: gọi lại chính file này với flag --_worker
# → Subprocess load config JSON, chạy benchmark, ghi result JSON, exit
```

```python
# Trong _worker_main():
if sys.argv[1] == "--_worker":
    _worker_main()   # chạy benchmark + ghi JSON
    return           # exit → CUDA context bị hủy hoàn toàn
```

---

### `print_table(summaries, baseline_label)`

```python
batch_sizes = sorted(set(s["batch_size"] for s in valid))
for bs in batch_sizes:
    group = [s for s in valid if s["batch_size"] == bs]
    # In bảng riêng cho từng batch size
    # → dễ so sánh ảnh hưởng của batching lên từng model

speedup = s["throughput_mean"] / baseline_tps
# Speedup so với FP16 baseline cùng batch size
# baseline_tps là throughput của FP16 trong cùng group batch_size
```

---

## 3. Tại sao FP16 (HuggingFace) chậm hơn Custom AWQ INT4 (vLLM)?

### 3.1 Bottleneck của LLM inference: memory bandwidth

LLM inference **không bị bottleneck bởi FLOPS** — GPU có quá nhiều FLOPS so với nhu cầu. Bottleneck thực sự là **tốc độ đọc weight từ VRAM vào compute units** (memory bandwidth).

Mỗi token generate = 1 lần đọc toàn bộ weight của model:
```
FP16 Mistral-7B:  14 GB đọc mỗi token  → chậm
INT4 Mistral-7B:  3.5 GB đọc mỗi token → nhanh hơn ~4x về bandwidth
```

RTX 4090: băng thông ~1 TB/s
```
FP16: 14 GB / 1 TB/s = 14 ms/token lý thuyết
INT4: 3.5 GB / 1 TB/s = 3.5 ms/token lý thuyết
```
Thực tế còn có compute overhead, nhưng INT4 vẫn nhanh hơn đáng kể.

### 3.2 HuggingFace bị padding waste khi batch > 1

```
Prompt lengths: [5, 12, 8, 3, 15, 7, 10, 4] tokens (bs=8)
HF: pad tất cả về max=15 → xử lý 8×15=120 tokens (dù chỉ cần 64)
vLLM PagedAttention: xử lý đúng 64 tokens thực, không có padding
```

Ở bs=16, HF lãng phí trung bình ~30-50% compute cho padding.

### 3.3 vLLM continuous batching + PagedAttention

```
HF generate():
  Step 1: forward toàn bộ batch → generate token 1 cho mọi sequence
  Step 2: forward lại → token 2
  ...
  Tất cả sequence phải kết thúc cùng lúc (hoặc pad đến max_length)

vLLM:
  Sequence nào xong thì xong, slot đó nhận request mới ngay
  → GPU luôn bận với useful work, không bao giờ chờ sequence dài nhất
```

### 3.4 KV cache pre-allocation của vLLM

HF: cấp phát KV cache động → overhead GC/allocation mỗi step
vLLM: pre-allocate toàn bộ KV cache budget ngay khi load → zero allocation overhead khi generate

### 3.5 Tóm tắt

| Yếu tố | FP16 HF | AWQ INT4 vLLM |
|--------|---------|---------------|
| Lượng data đọc/token | 14 GB | 3.5 GB (-75%) |
| Padding waste (bs=16) | ~40% compute | 0% |
| Batching strategy | static padding | continuous batching |
| KV cache | dynamic alloc | pre-allocated |
| **Kết quả thực tế** | **215 tok/s** | **308 tok/s (+43%)** |

Speedup 1.43x ở bs=16 là **hợp lý và kỳ vọng**. Ở bs lớn hơn (32, 64), vLLM sẽ càng vượt trội hơn nữa vì continuous batching tận dụng tốt hơn.

### 3.6 Tại sao AWQ FP16-dequant (HF) không nhanh hơn FP16 baseline?

```
AWQ FP16-dequant: weight đã được "fake-quantize" → vẫn lưu dưới dạng FP16
→ kích thước file vẫn ~14 GB
→ vẫn phải đọc 14 GB/token từ VRAM
→ throughput ≈ FP16 baseline (216 vs 215 tok/s)
```

AWQ FP16-dequant là để **đánh giá PPL** (đo chất lượng quantization), không phải để tăng tốc inference. Phải convert sang INT4 thật sự (vLLM format) mới tăng tốc được.
