"""
RTN + James-Stein + Heuristic-Guided Rounding (XL Version)

This is the RTN-analog of `awq_js_xl.py`: same heuristic-guided global greedy
rounding (Kneedle outlier detection + per-output max-flip cap), but *without*
AWQ's per-channel scaling search. The only activation statistic consumed is
the James-Stein-shrunk per-channel mean E[X]_JS.

Why this is the right RTN counterpart
-------------------------------------
For Y = X W^T (+ b), the per-output expected error from quantization is

        E[ΔY_i] = sum_j  E[X_j] * (W_ij - W_q_ij)              (1)

Plain RTN minimizes |W - W_q| element-wise, which is NOT the same as
minimizing |E[ΔY]|. Heuristic-guided rounding starts from RTN and greedily
"flips" rounding decisions (up <-> down by 1 quant step) to drive |E[ΔY_i]|
toward 0 per row, ordered by rounding regret. James-Stein gives a
low-variance estimate of E[X] under small calibration.

Knobs (same semantics as awq_js_xl.py)
--------------------------------------
  --knee-tolerance      Offset added to the Kneedle knee index (descending-
                        sorted |E[X]|). Larger → MORE channels masked as
                        outliers (more conservative).
  --max-flip-percent    Per-output-row cap on the fraction of in-channels
                        that may be flipped. 0.05 = 5% of in_features.
  --skip-lm-head        Default True. Leave lm_head in full precision.
                        If False, lm_head is processed in chunks along the
                        output dimension.

Pipeline per layer
------------------
  1. Calibrate to collect activations (CPU float32, subsampled along seq).
  2. Compute E[X]_JS.
  3. Group-wise asymmetric RTN → W_int, scale, zp.
  4. Compute current per-row expected error using E[X]_JS.
  5. Build flip candidates; mask outlier channels via Kneedle on |E[X]_JS|.
  6. Greedily pick flips per row, capped at max_flip_percent of in_features.
  7. Dequantize. Weight replaced; bias untouched.

XL handling
-----------
Batched sequential calibration; lm_head split into lmhead_chunks along the
output dim so peak memory is bounded.
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
import hashlib
import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil not installed. RAM monitoring disabled.")

try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    print("⚠️  calibration_utils not found. Use --calib-dataset wikitext2-simple as fallback.")
    def get_c4_calibration_data(*a, **k):
        raise NotImplementedError("calibration_utils.py missing")
    def get_wikitext2_calibration_data(*a, **k):
        raise NotImplementedError("calibration_utils.py missing")


# ---------------------------------------------------------------------------
# Kneedle knee-point finder (matches awq_js_xl.py)
# ---------------------------------------------------------------------------

def find_knee_point(values, tolerance_offset=0.0):
    """
    Max-distance-from-chord knee. Returns an index in [0, n-1]; tolerance_offset
    (as a fraction of n) shifts the knee toward later indices (more conservative).
    """
    n = len(values)
    if n < 3:
        return n // 2

    if torch.is_tensor(values):
        y = values.cpu().float().numpy()
    else:
        y = np.asarray(values)

    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-10:
        return n // 2

    y_norm = (y - y_min) / (y_max - y_min)
    x_norm = np.linspace(0, 1, n)
    y_line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    distances = np.abs(y_norm - y_line)
    knee_idx = int(np.argmax(distances))

    if knee_idx < n - 1:
        offset_indices = int(tolerance_offset * n)
        knee_idx = max(0, min(knee_idx + offset_indices, n - 1))
    return knee_idx


# ---------------------------------------------------------------------------
# James-Stein shrinkage for per-channel means
# ---------------------------------------------------------------------------

def compute_james_stein_mean(raw_means, variance_estimate=None):
    """
    μ̂_JS[j] = μ̄ + (1 - c) · (X̄[j] - μ̄),   c = (p - 2) σ² / Σ(X̄[j] - μ̄)²

    Falls through unchanged when p < 3 or when deviations are degenerate.
    """
    p = raw_means.numel()
    if p < 3:
        return raw_means

    grand_mean = raw_means.mean()
    deviations = raw_means - grand_mean
    sum_sq_dev = (deviations ** 2).sum()
    if sum_sq_dev < 1e-10:
        return raw_means

    if variance_estimate is None:
        variance_estimate = ((raw_means - grand_mean).abs().mean()) ** 2
        variance_estimate = variance_estimate.clamp(min=1e-8)

    c = ((p - 2) * variance_estimate) / sum_sq_dev
    c = c.clamp(0.0, 1.0)
    return grand_mean + (1.0 - c) * deviations


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class RTN_JS_Heuristic_XL_Quantizer:
    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 use_heuristic=True, use_james_stein=True,
                 knee_tolerance=0.0, max_flip_percent=0.05,
                 skip_lm_head=True,
                 max_tokens_per_sample=2048, layer_batch_size=16,
                 lmhead_chunks=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.group_size = group_size
        self.use_heuristic = use_heuristic
        self.use_james_stein = use_james_stein
        self.knee_tolerance = knee_tolerance
        self.max_flip_percent = max_flip_percent
        self.skip_lm_head = skip_lm_head
        self.max_tokens_per_sample = max_tokens_per_sample
        self.layer_batch_size = layer_batch_size
        self.lmhead_chunks = lmhead_chunks

        self.activation_data = {}     # name -> list[CPU float32 tensor]
        self.activation_means = {}    # name -> CPU float32 tensor, reusable across k/f
        self.layer_stats = {}         # name -> dict

        max_int = 2 ** bits - 1
        print(f"\n[RTN + James-Stein + Heuristic Rounding (XL) Initialized]")
        print(f"  Target bits:          {bits}")
        print(f"  Group size:           {group_size}")
        print(f"  Skip lm_head:         {skip_lm_head}")
        print(f"  Use heuristic:        {use_heuristic}")
        print(f"  Use James-Stein E[X]: {use_james_stein}")
        if use_heuristic:
            print(f"  Outlier detection:    Kneedle on sorted |E[X]|")
            print(f"  Knee tolerance:       {knee_tolerance:.6f}")
            print(f"  Max flip percent:     {max_flip_percent*100:.2f}% per output row")
        print(f"  Layer batch size:     {layer_batch_size}")
        print(f"  lm_head chunks:       {lmhead_chunks}")
        print(f"  Tokens per sample:    {max_tokens_per_sample}")
        print(f"  Quantization:         GROUP-WISE ASYMMETRIC [0, {max_int}]")

    @staticmethod
    def fmt_seconds(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        return f"{seconds / 60:.1f}m"

    # ----- activation hook ---------------------------------------------------

    def get_hook(self, name):
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            inp = input[0] if isinstance(input, tuple) else input
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                idx = torch.randperm(seq_len, device=inp.device)[:self.max_tokens_per_sample]
                idx = idx.sort()[0]
                inp = inp[:, idx, :]
            self.activation_data[name].append(inp.detach().cpu().float())
        return hook

    @torch.no_grad()
    def get_activation_mean(self, name, in_features):
        """Return [in_features] CPU float32 tensor of E[X], JS-shrunk if enabled."""
        if name in self.activation_means:
            cached = self.activation_means[name]
            if cached.numel() == in_features:
                return cached
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None
        mean_sum = torch.zeros(in_features, dtype=torch.float32)
        total = 0
        for x in self.activation_data[name]:
            x_flat = x.reshape(-1, x.shape[-1])
            mean_sum += x_flat.sum(dim=0)
            total += x_flat.shape[0]
        if total == 0:
            return None
        raw_mean = mean_sum / total
        return compute_james_stein_mean(raw_mean) if self.use_james_stein else raw_mean

    # ----- Kneedle outlier threshold on |E[X]| ------------------------------

    @torch.no_grad()
    def compute_dynamic_outlier_threshold(self, activation_means, debug=False):
        """
        Sort |E[X]| descending; apply Kneedle to the first half to find the
        outlier→normal transition. Channels with |E[X]| > threshold are
        MASKED OUT of the flip pool (we don't try to fix their error via flips).
        Returns (threshold_value, outlier_percent).
        """
        sorted_desc, _ = torch.sort(activation_means.abs(), descending=True)
        n = len(sorted_desc)
        first_half = sorted_desc[: n // 2]

        if len(first_half) < 3:
            threshold_idx = max(0, int(0.05 * n))
            threshold = sorted_desc[threshold_idx].item()
            outlier_pct = 0.05
            if debug:
                print(f"    DEBUG: too few channels for Kneedle, using top 5%")
            return threshold, outlier_pct

        knee_idx = find_knee_point(first_half, tolerance_offset=self.knee_tolerance)
        threshold = sorted_desc[knee_idx].item()
        num_outliers = (activation_means.abs() >= threshold).sum().item()
        outlier_pct = num_outliers / n

        if debug:
            print(f"    DEBUG: sorted |E[X]| desc: [{sorted_desc[0]:.4e} .. {sorted_desc[-1]:.4e}]")
            print(f"    DEBUG: knee idx = {knee_idx}/{n} ({knee_idx/n*100:.1f}%)")
            print(f"    DEBUG: threshold = {threshold:.4e}, outliers = "
                  f"{num_outliers}/{n} ({outlier_pct*100:.2f}%)")
        return threshold, outlier_pct

    # ----- core: heuristic-guided group-wise asymmetric quantization --------

    @torch.no_grad()
    def quantize_weight_heuristic_groupwise(self, W, ex_mean, debug=False):
        """
        RTN group-wise asym → optional heuristic flips guided by E[X].

        Args:
            W: [out, in] weight, on device
            ex_mean: [in] per-input-channel activation mean (E[X]_JS), on device
            debug: verbose Kneedle output for first call

        Returns:
            W_dequant: [out, in] fake-quantized weight (same dtype as W)
            outlier_percent: float or None
            flip_stats: dict
        """
        out_features, in_features = W.shape
        device = W.device
        dtype = W.dtype

        # ---- 1. pad + group-wise asym scale/zp ----
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in = n_groups * self.group_size

        if padded_in > in_features:
            W_padded = torch.zeros(out_features, padded_in, device=device, dtype=dtype)
            W_padded[:, :in_features] = W
            act_padded = torch.zeros(padded_in, device=device, dtype=dtype)
            act_padded[:in_features] = ex_mean
        else:
            W_padded = W
            act_padded = ex_mean

        W_g = W_padded.reshape(out_features, n_groups, self.group_size)
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]

        max_int = 2 ** self.bits - 1
        scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_features, padded_in)
        zp_flat = zp.repeat(1, 1, self.group_size).reshape(out_features, padded_in)

        # ---- 2. RTN nearest-rounding ----
        W_div = W_padded / scale_flat
        W_int = torch.round(W_div + zp_flat).clamp(0, max_int)

        # Path A: pure RTN, no flips
        if not self.use_heuristic:
            W_dq = (W_int - zp_flat) * scale_flat
            if padded_in > in_features:
                W_dq = W_dq[:, :in_features]
            empty = {k: 0 for k in (
                'total', 'per_row_mean', 'per_row_max', 'per_row_cap',
                'per_channel_mean', 'per_channel_median', 'per_channel_std',
                'per_channel_p95', 'per_channel_p99')}
            empty['per_channel_zero_pct'] = 100.0
            return W_dq.to(dtype), None, empty

        # ---- 3. current per-row expected output error ----
        W_quant = (W_int - zp_flat) * scale_flat
        W_diff = W_padded - W_quant
        current_error = (W_diff * act_padded.unsqueeze(0)).sum(dim=1)  # [out]

        # ---- 4. flip candidates ----
        # Sign of (W_div + zp_flat - W_int) gives the direction of residual.
        # flip_dir == +1 → round() rounded DOWN, flipping would INCREMENT int
        # flip_dir == -1 → round() rounded UP, flipping would DECREMENT int
        flip_dir = torch.sign(W_div + zp_flat - W_int)
        flip_dir = torch.where(flip_dir == 0, torch.ones_like(flip_dir), flip_dir)

        # Δ(row_error_i) from flipping (i,j):
        #   = ex_mean[j] * flip_dir[i,j] * scale_flat[i,j]
        flip_impacts = act_padded.unsqueeze(0) * flip_dir * scale_flat   # [out, padded_in]

        # ---- 5. validity masks ----
        # Only flip if doing so REDUCES |current_error| → sign(impact) == sign(error)
        target_sign = torch.sign(current_error).unsqueeze(1)
        valid_mask = (torch.sign(flip_impacts) == target_sign)

        # Proposed int must remain in [0, max_int]
        w_int_proposed = W_int + flip_dir
        in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
        valid_mask = valid_mask & in_range

        # Outlier masking: never flip in outlier channels
        outlier_threshold, outlier_pct = self.compute_dynamic_outlier_threshold(
            act_padded, debug=debug)
        is_outlier = act_padded.abs() > outlier_threshold
        valid_mask = valid_mask & (~is_outlier).unsqueeze(0)

        # ---- 6. sort flips by rounding regret (closeness to 0.5) ----
        rounding_costs = (W_div + zp_flat - W_int).abs()         # in [0, 0.5]
        rc_masked = rounding_costs.clone()
        rc_masked[~valid_mask] = -1.0                            # invalid → end

        sorted_indices = torch.argsort(rc_masked, dim=1, descending=True)
        sorted_impacts = torch.gather(flip_impacts, 1, sorted_indices)
        sorted_validity = torch.gather(valid_mask.long(), 1, sorted_indices)
        sorted_impacts = sorted_impacts * sorted_validity

        # ---- 7. choose best-k per row ----
        cumsum_impacts = torch.cumsum(sorted_impacts, dim=1)
        residuals = torch.abs(current_error.unsqueeze(1) - cumsum_impacts)
        zero_k = torch.abs(current_error).unsqueeze(1)
        all_residuals = torch.cat([zero_k, residuals], dim=1)
        best_k = torch.argmin(all_residuals, dim=1)              # in [0, padded_in]

        idx_range = torch.arange(padded_in, device=device).unsqueeze(0)
        flip_mask_sorted = idx_range < best_k.unsqueeze(1)
        final_flips_sorted = flip_mask_sorted & sorted_validity.bool()

        # ---- 8. enforce max_flip_percent per output row ----
        # cap measured against in_features (real, not padded)
        max_flips_per_row = max(1, int(self.max_flip_percent * in_features))
        cumsum_flips = final_flips_sorted.long().cumsum(dim=1)
        within_cap = cumsum_flips <= max_flips_per_row
        final_flips_sorted = final_flips_sorted & within_cap

        # ---- 9. apply flips back to W_int ----
        sorted_flip_dir = torch.gather(flip_dir, 1, sorted_indices)
        sorted_flip_dir = torch.where(final_flips_sorted, sorted_flip_dir,
                                      torch.zeros_like(sorted_flip_dir))
        W_int.scatter_add_(1, sorted_indices, sorted_flip_dir)
        W_int.clamp_(0, max_int)

        # ---- 10. flip statistics ----
        flips_per_row = final_flips_sorted.sum(dim=1).float()        # [out]
        flips_per_channel = final_flips_sorted.sum(dim=0).float()    # [padded_in]
        if padded_in > in_features:
            flips_per_channel = flips_per_channel[:in_features]

        flip_stats = {
            'total': int(final_flips_sorted.sum().item()),
            'per_row_mean':         flips_per_row.mean().item(),
            'per_row_max':          flips_per_row.max().item(),
            'per_row_cap':          max_flips_per_row,
            'per_channel_mean':     flips_per_channel.mean().item(),
            'per_channel_median':   flips_per_channel.median().item(),
            'per_channel_std':      flips_per_channel.std().item(),
            'per_channel_p95':      torch.quantile(flips_per_channel, 0.95).item(),
            'per_channel_p99':      torch.quantile(flips_per_channel, 0.99).item(),
            'per_channel_zero_pct': (flips_per_channel == 0).float().mean().item() * 100,
        }

        # ---- 11. dequantize ----
        W_dq = (W_int - zp_flat) * scale_flat
        if padded_in > in_features:
            W_dq = W_dq[:, :in_features]
        return W_dq.to(dtype), outlier_pct, flip_stats

    # ----- per-layer driver --------------------------------------------------

    @torch.no_grad()
    def quantize_layer(self, name, module, debug=False):
        """One non-lm_head Linear layer."""
        W = module.weight.data
        in_features = W.shape[1]
        dtype = W.dtype

        ex_mean_cpu = self.get_activation_mean(name, in_features)
        if ex_mean_cpu is None:
            ex_mean = torch.zeros(in_features, device=W.device, dtype=dtype)
            had_calib = False
        else:
            ex_mean = ex_mean_cpu.to(W.device).to(dtype)
            had_calib = True

        W_dq, outlier_pct, flip_stats = self.quantize_weight_heuristic_groupwise(
            W, ex_mean, debug=debug
        )
        module.weight.data = W_dq

        self.layer_stats[name] = {
            'had_calib': had_calib,
            'outlier_percent': outlier_pct if outlier_pct is not None else 0.0,
            'flip_stats': flip_stats,
        }

        if name in self.activation_data:
            del self.activation_data[name]
        if name in self.activation_means:
            del self.activation_means[name]
        del ex_mean, W_dq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def quantize_lmhead_chunked(self, name, module, debug=False):
        """
        Split lm_head along output dim. Each chunk runs the full heuristic-guided
        pipeline independently. Outlier detection is identical across chunks
        (driven by E[X], which is input-side).
        """
        print(f"\n  🔧 Special handling for {name} (split into {self.lmhead_chunks} chunks)")

        W = module.weight.data
        out_features, in_features = W.shape
        dtype = W.dtype
        print(f"     Shape: {tuple(W.shape)} ({W.numel()/1e6:.1f}M params)")

        ex_mean_cpu = self.get_activation_mean(name, in_features)
        if ex_mean_cpu is None:
            ex_mean = torch.zeros(in_features, device=W.device, dtype=dtype)
            had_calib = False
        else:
            ex_mean = ex_mean_cpu.to(W.device).to(dtype)
            had_calib = True

        n_chunks = self.lmhead_chunks
        chunk_size = (out_features + n_chunks - 1) // n_chunks

        W_chunks = []
        chunk_stats = []
        for ci in range(n_chunks):
            s = ci * chunk_size
            e = min(s + chunk_size, out_features)
            if s >= e:
                break
            print(f"     Chunk {ci+1}/{n_chunks}: rows {s}-{e-1}")
            W_chunk = W[s:e, :].contiguous()
            W_dq, outlier_pct, flip_stats = self.quantize_weight_heuristic_groupwise(
                W_chunk, ex_mean, debug=(debug and ci == 0)
            )
            W_chunks.append(W_dq)
            chunk_stats.append({
                'outlier_percent': outlier_pct if outlier_pct is not None else 0.0,
                'flip_stats': flip_stats,
            })
            del W_chunk, W_dq
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        W_final = torch.cat(W_chunks, dim=0)
        module.weight.data = W_final

        total_flips = sum(c['flip_stats']['total'] for c in chunk_stats)
        avg_outlier = float(np.mean([c['outlier_percent'] for c in chunk_stats]))
        agg = {
            'total': total_flips,
            'per_row_mean':         float(np.mean([c['flip_stats']['per_row_mean'] for c in chunk_stats])),
            'per_row_max':          float(np.max([c['flip_stats']['per_row_max'] for c in chunk_stats])),
            'per_row_cap':          chunk_stats[0]['flip_stats']['per_row_cap'],
            'per_channel_mean':     float(np.mean([c['flip_stats']['per_channel_mean'] for c in chunk_stats])),
            'per_channel_median':   float(np.mean([c['flip_stats']['per_channel_median'] for c in chunk_stats])),
            'per_channel_std':      float(np.mean([c['flip_stats']['per_channel_std'] for c in chunk_stats])),
            'per_channel_p95':      float(np.mean([c['flip_stats']['per_channel_p95'] for c in chunk_stats])),
            'per_channel_p99':      float(np.mean([c['flip_stats']['per_channel_p99'] for c in chunk_stats])),
            'per_channel_zero_pct': float(np.mean([c['flip_stats']['per_channel_zero_pct'] for c in chunk_stats])),
        }
        self.layer_stats[name] = {
            'had_calib': had_calib,
            'outlier_percent': avg_outlier,
            'flip_stats': agg,
        }

        if name in self.activation_data:
            del self.activation_data[name]
        if name in self.activation_means:
            del self.activation_means[name]
        del ex_mean, W_chunks, W_final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"     ✓ total flips={total_flips:,}, "
              f"outlier%={avg_outlier*100:.2f}%, "
              f"per_row mean={agg['per_row_mean']:.1f} "
              f"(cap={agg['per_row_cap']})")

    # ----- calibration -------------------------------------------------------

    def calibrate_layer_batch(self, layer_batch, calibration_texts, n_samples):
        self.activation_data = {}
        self.activation_means = {}
        handles = [m.register_forward_hook(self.get_hook(n)) for n, m in layer_batch]

        successful = 0
        with torch.no_grad():
            for i, text in enumerate(tqdm(calibration_texts[:n_samples],
                                          desc="  Calibration", leave=False)):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                            truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1
                    if (i + 1) % 32 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    continue

        for h in handles:
            h.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if successful == 0:
            print("⚠️  No successful calibration passes in this batch.")

    def load_calibration_batch_cache(self, cache_path, expected_names):
        if not cache_path or not os.path.isfile(cache_path):
            return False
        try:
            payload = torch.load(cache_path, map_location="cpu")
            means = payload.get("activation_means", payload)
            if not isinstance(means, dict):
                return False
            missing = [name for name in expected_names if name not in means]
            if missing:
                print(f"  Calibration cache incomplete ({len(missing)} missing); recalibrating.")
                return False
            self.activation_data = {}
            self.activation_means = {name: means[name].cpu().float() for name in expected_names}
            print(f"  Calibration cache hit: {cache_path}")
            return True
        except Exception as exc:
            print(f"  Calibration cache unreadable ({exc}); recalibrating.")
            return False

    def save_calibration_batch_cache(self, cache_path, batch):
        if not cache_path:
            return
        means = {}
        for name, module, _is_lmhead in batch:
            in_features = module.weight.data.shape[1]
            mean = self.get_activation_mean(name, in_features)
            if mean is not None:
                means[name] = mean.cpu().float()
        if not means:
            return
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = cache_path + ".tmp"
        torch.save({"activation_means": means}, tmp_path)
        os.replace(tmp_path, cache_path)
        self.activation_means = means
        print(f"  Saved calibration cache: {cache_path}")

    # ----- top-level driver --------------------------------------------------

    def quantize_model(self, calibration_texts, n_samples=128,
                       calib_cache_dir=None, calib_cache_key=None):
        print("\n" + "=" * 80)
        print("Batched Sequential RTN + JS + Heuristic Rounding")
        print("=" * 80)

        all_layers = [(name, module) for name, module in self.model.named_modules()
                      if isinstance(module, nn.Linear)]

        layers_to_q = []
        for name, module in all_layers:
            is_lmhead = ('lm_head' in name.lower()) or name.endswith('lm_head')
            if is_lmhead and self.skip_lm_head:
                continue
            layers_to_q.append((name, module, is_lmhead))

        n_total = len(layers_to_q)
        print(f"  Total Linear: {len(all_layers)}  |  to quantize: {n_total}  "
              f"|  skipped (lm_head): {len(all_layers) - n_total}")

        n_batches = (n_total + self.layer_batch_size - 1) // self.layer_batch_size
        print(f"  Batches: {n_batches} (batch size = {self.layer_batch_size})")

        quantized = 0
        for b in range(n_batches):
            s = b * self.layer_batch_size
            e = min(s + self.layer_batch_size, n_total)
            batch = layers_to_q[s:e]
            batch_for_hooks = [(n, m) for n, m, _ in batch]
            batch_names = [n for n, _m, _is_lmhead in batch]
            cache_path = None
            if calib_cache_dir and calib_cache_key:
                cache_path = os.path.join(
                    calib_cache_dir,
                    f"{calib_cache_key}_batch{b:03d}_{s:04d}_{e-1:04d}.pt",
                )

            print(f"\n[Batch {b+1}/{n_batches}] Layers {s}-{e-1}")
            batch_start = time.time()
            if not self.load_calibration_batch_cache(cache_path, batch_names):
                calib_start = time.time()
                self.calibrate_layer_batch(batch_for_hooks, calibration_texts, n_samples)
                print(f"  Calibration time: {self.fmt_seconds(time.time() - calib_start)}")
                self.save_calibration_batch_cache(cache_path, batch)
            else:
                print("  Calibration time: cache hit")

            quant_start = time.time()
            for name, module, is_lmhead in tqdm(batch, desc="  Quantize", leave=False):
                try:
                    if is_lmhead:
                        self.quantize_lmhead_chunked(name, module, debug=(quantized < 2))
                    else:
                        self.quantize_layer(name, module, debug=(quantized < 2))
                    quantized += 1
                except Exception as exc:
                    print(f"\n⚠️  Error on {name}: {exc}")
                    continue
            print(f"  Quantize time: {self.fmt_seconds(time.time() - quant_start)}")

            self.activation_data = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if HAS_PSUTIL:
                print(f"  RAM after batch {b+1}: {psutil.virtual_memory().percent:.1f}%")
            print(f"  Batch time: {self.fmt_seconds(time.time() - batch_start)}")

        # ----- summary -------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"✓ Done: quantized {quantized}/{n_total} layers")
        print("=" * 80)

        if self.layer_stats:
            outliers = [s['outlier_percent'] for s in self.layer_stats.values()]
            print(f"\nOutlier %:  mean={np.mean(outliers)*100:.2f}%  "
                  f"median={np.median(outliers)*100:.2f}%  "
                  f"min={np.min(outliers)*100:.2f}%  "
                  f"max={np.max(outliers)*100:.2f}%")

            if self.use_heuristic:
                totals = [s['flip_stats']['total'] for s in self.layer_stats.values()]
                row_means = [s['flip_stats']['per_row_mean'] for s in self.layer_stats.values()]
                row_maxes = [s['flip_stats']['per_row_max'] for s in self.layer_stats.values()]
                zero_pcts = [s['flip_stats']['per_channel_zero_pct'] for s in self.layer_stats.values()]
                print(f"\nFlip statistics:")
                print(f"  total flips: {int(np.sum(totals)):,}")
                print(f"  per-layer total: mean={np.mean(totals):,.0f}, "
                      f"median={np.median(totals):,.0f}, max={int(np.max(totals)):,}")
                print(f"  per-row count: mean={np.mean(row_means):.2f}, "
                      f"max={np.max(row_maxes):.0f}  "
                      f"(cap = {self.max_flip_percent*100:.2f}% of in_features)")
                print(f"  channels with 0 flips (avg over layers): {np.mean(zero_pcts):.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_wikitext2_simple(n_samples=128):
    print("Loading WikiText-2 (simple)...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in ds if len(item['text'].strip()) > 100]
    return texts[:n_samples]


def make_calibration_cache_key(args):
    model_id = os.path.abspath(args.model_path) if os.path.exists(args.model_path) else args.model_path
    model_hash = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:12]
    model_name = os.path.basename(str(args.model_path).rstrip("/")).replace(".", "p").replace("-", "_")
    return (
        f"{model_name}_{model_hash}"
        f"_dataset{args.calib_dataset}"
        f"_n{args.n_calib}"
        f"_seq{args.max_tokens_per_sample}"
        f"_seed{args.seed}"
        f"_js{int(args.use_james_stein)}"
        f"_skiplm{int(args.skip_lm_head)}"
        f"_lbs{args.layer_batch_size}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="RTN + James-Stein + heuristic-guided rounding (XL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048)
    parser.add_argument("--layer-batch-size", type=int, default=16)
    parser.add_argument("--lmhead-chunks", type=int, default=4)

    # heuristic knobs
    parser.add_argument("--use-heuristic", action="store_true", default=True)
    parser.add_argument("--no-heuristic", dest="use_heuristic", action="store_false",
                        help="Disable heuristic flips → pure RTN")
    parser.add_argument("--knee-tolerance", type=float, default=0.0,
                        help="Offset added to Kneedle knee index (fraction of n). "
                             "Larger → more channels masked as outliers.")
    parser.add_argument("--max-flip-percent", type=float, default=0.05,
                        help="Max fraction of in_features that may be flipped per output row.")

    # JS
    parser.add_argument("--use-james-stein", action="store_true", default=True)
    parser.add_argument("--no-james-stein", dest="use_james_stein", action="store_false",
                        help="Use raw sample mean instead of JS shrinkage.")

    # lm_head
    parser.add_argument("--skip-lm-head", action="store_true", default=True,
                        help="Leave lm_head in full precision (default: True).")
    parser.add_argument("--quantize-lm-head", dest="skip_lm_head", action="store_false",
                        help="Quantize lm_head (chunked processing).")

    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/model_rtn_js_xl")
    parser.add_argument("--calib-dataset", type=str, default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache")
    parser.add_argument("--no-calib-cache", action="store_true",
                        help="Disable reusable activation-mean calibration cache.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("RTN + James-Stein + Heuristic-Guided Rounding (XL)")
    print(f"Target Model: {args.model_path}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.float32
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Model dtype: {model_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        calib_texts = load_wikitext2_simple(n_samples=args.n_calib)
    else:
        calib_texts = get_wikitext2_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir)

    quantizer = RTN_JS_Heuristic_XL_Quantizer(
        model=model, tokenizer=tokenizer, device=device,
        bits=args.bits, group_size=args.group_size,
        use_heuristic=args.use_heuristic,
        use_james_stein=args.use_james_stein,
        knee_tolerance=args.knee_tolerance,
        max_flip_percent=args.max_flip_percent,
        skip_lm_head=args.skip_lm_head,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks,
    )
    calib_cache_dir = None
    calib_cache_key = None
    if not args.no_calib_cache:
        calib_cache_dir = os.path.join(args.cache_dir, "rtn_js_xl_activation_means")
        calib_cache_key = make_calibration_cache_key(args)
        print(f"Calibration mean cache: {calib_cache_dir}/{calib_cache_key}_batch*.pt")

    quantizer.quantize_model(
        calib_texts,
        n_samples=args.n_calib,
        calib_cache_dir=calib_cache_dir,
        calib_cache_key=calib_cache_key,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved to {args.output_dir}") 


if __name__ == "__main__":
    main()

