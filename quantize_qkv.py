"""
INT4 Weight-Only Quantization on GQA Attention with ReFlip Strategy

This script compares three quantization strategies:
1. Nearest Rounding (baseline)
2. Heuristic Flip Correction (global greedy)
3. ReFlip (new): Targeted error correction on critical head dimensions

ReFlip Strategy:
1. Apply initial heuristic quantization
2. Use Kneedle algorithm to identify critical head dimensions (based on |Q_orig|)
3. Select top ~5% critical dimensions per head (configurable)
4. Compute target error correction = -current_error for critical dimensions
5. Redistribute correction proportionally to input magnitudes
6. Apply second heuristic flip to reduce critical dimension errors

Usage:
    python quantize_qkv.py [--critical-dim-pct 0.05] [--knee-tolerance 0.0]
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns.set_style("whitegrid")


def find_knee_point(values, tolerance_offset=0.0):
    """
    Find knee point in sorted values using Kneedle algorithm.

    Args:
        values: 1D array of sorted values (ascending order)
        tolerance_offset: Additional offset to add to knee point (default: 0.0)

    Returns:
        index of knee point

    Algorithm:
    1. Normalize values to [0, 1]
    2. Create reference line from start to end
    3. Find point with maximum distance from reference line
    4. Apply tolerance offset
    """
    n = len(values)
    if n < 3:
        return n // 2

    # Normalize to [0, 1]
    y_min, y_max = values.min(), values.max()
    if y_max - y_min < 1e-10:
        # All values are the same, no knee
        return n // 2

    y_norm = (values - y_min) / (y_max - y_min)
    x_norm = np.linspace(0, 1, n)

    # Compute distances from the line connecting first and last point
    # Line equation: y = m*x + b
    # For normalized: line goes from (0, y_norm[0]) to (1, y_norm[-1])
    y_line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm

    # Perpendicular distance from each point to the line
    distances = np.abs(y_norm - y_line)

    # Find point with maximum distance (the knee)
    knee_idx = np.argmax(distances)

    # Apply tolerance offset
    if knee_idx < n - 1:
        # Calculate how many indices to shift based on offset
        offset_indices = int(tolerance_offset * n)
        knee_idx = min(knee_idx + offset_indices, n - 1)
        knee_idx = max(knee_idx, 0)

    return knee_idx


def compute_dynamic_outlier_threshold(activation_means, knee_tolerance=0.0, debug=False):
    """
    Compute dynamic outlier threshold using Kneedle algorithm.

    Strategy:
    1. Sort activation means in DESCENDING order [high ... medium ... low]
    2. Apply Kneedle to FIRST HALF [high ... medium] to find outlier→normal transition
    3. Tolerance offset allows tuning: positive = more conservative (keep more outliers)

    Args:
        activation_means: Array of per-channel activation means (E[X])
        knee_tolerance: Tolerance offset for knee point (default: 0.0)
        debug: Print debug information

    Returns:
        tuple: (threshold value, outlier percentage)
    """
    # Sort activation means in DESCENDING order [high → low]
    sorted_means = np.sort(np.abs(activation_means))[::-1]  # Descending
    n = len(sorted_means)

    # Apply Kneedle to FIRST HALF [high ... medium] to find outlier transition
    first_half = sorted_means[:n // 2]

    if len(first_half) < 3:
        # Not enough data, use a conservative default (top 5%)
        threshold_idx = int(0.05 * n)
        threshold = sorted_means[threshold_idx]
        outlier_percent = 0.05
        if debug:
            print(f"    DEBUG: Not enough data for Kneedle, using top 5% as default")
        return threshold, outlier_percent

    # Find knee point in first half (where outliers end, normal begins)
    knee_idx_in_half = find_knee_point(first_half, tolerance_offset=knee_tolerance)

    # This is already the index in full array (descending sorted)
    knee_idx = knee_idx_in_half

    # The threshold is the value at the knee point
    threshold = sorted_means[knee_idx]

    # Count how many channels are outliers (above or equal to threshold)
    num_outliers = (np.abs(activation_means) >= threshold).sum()
    outlier_percent = num_outliers / n

    if debug:
        print(f"    DEBUG: Sorted means (descending): [{sorted_means[0]:.6f} ... {sorted_means[-1]:.6f}]")
        print(f"    DEBUG: First half range: [{first_half[0]:.6f} ... {first_half[-1]:.6f}]")
        print(f"    DEBUG: Knee point index in first half: {knee_idx_in_half}/{len(first_half)}")
        print(f"    DEBUG: Knee point index in full array: {knee_idx}/{n} ({knee_idx/n*100:.1f}%)")
        print(f"    DEBUG: Knee threshold value: {threshold:.6f}")
        print(f"    DEBUG: Outliers (>= threshold): {num_outliers}/{n} ({outlier_percent*100:.2f}%)")
        print(f"    DEBUG: vs Default 5.00%: {outlier_percent*100 - 5.0:+.2f}% difference")

    return threshold, outlier_percent


def quantize_weight_groupwise_int4(W, group_size=128, method='nearest'):
    """
    Quantize weights to INT4 using group-wise asymmetric quantization [0, 15].

    Args:
        W: Weight matrix of shape [..., in_features]
        group_size: Size of each quantization group (default: 128)
        method: 'nearest' for nearest rounding (default)

    Returns:
        W_quant: Dequantized weights (same shape as W)
        scales: Per-group scales, shape [..., n_groups]
        zp: Per-group zero points, shape [..., n_groups]
        W_int: Integer weights, shape [..., in_features]
    """
    original_shape = W.shape

    # Flatten to 2D if needed
    if W.ndim > 2:
        # Reshape keeping last dimension
        W_flat = W.reshape(-1, W.shape[-1])
    else:
        W_flat = W.copy()

    out_features, in_features = W_flat.shape
    n_groups = (in_features + group_size - 1) // group_size
    padded_in = n_groups * group_size

    # Pad if needed
    if padded_in > in_features:
        W_padded = np.zeros((out_features, padded_in), dtype=W.dtype)
        W_padded[:, :in_features] = W_flat
    else:
        W_padded = W_flat

    # Reshape to groups: [out_features, n_groups, group_size]
    W_grouped = W_padded.reshape(out_features, n_groups, group_size)

    # Asymmetric quantization [0, 15] (INT4)
    w_min = W_grouped.min(axis=2, keepdims=True)
    w_max = W_grouped.max(axis=2, keepdims=True)
    max_int = 15

    # Compute scale and zero point
    scale = (w_max - w_min) / max_int
    scale = np.maximum(scale, 1e-8)  # Avoid division by zero
    zp = np.round(-w_min / scale).clip(0, max_int)

    # Quantize with NEAREST rounding
    W_div = W_grouped / scale
    W_int = np.round(W_div + zp).clip(0, max_int)

    # Dequantize
    W_quant_grouped = (W_int - zp) * scale

    # Reshape back to flat
    W_quant_flat = W_quant_grouped.reshape(out_features, padded_in)
    W_int_flat = W_int.reshape(out_features, padded_in)

    # Remove padding
    if padded_in > in_features:
        W_quant_flat = W_quant_flat[:, :in_features]
        W_int_flat = W_int_flat[:, :in_features]

    # Reshape back to original shape
    W_quant = W_quant_flat.reshape(original_shape)
    W_int_final = W_int_flat.reshape(original_shape)

    # Flatten scales and zp for output
    scales_flat = scale.reshape(out_features, n_groups)
    zp_flat = zp.reshape(out_features, n_groups)

    # If original was 3D, reshape scales/zp to match original structure
    if len(original_shape) == 3:
        # Original: [num_heads, head_dim, hidden_size] e.g. [4, 128, 4096]
        # Flattened to: [num_heads * head_dim, hidden_size] e.g. [512, 4096]
        # Scales: [num_heads * head_dim, n_groups] e.g. [512, 32]
        # Reshape to: [num_heads, head_dim, n_groups] e.g. [4, 128, 32]
        scales_out = scales_flat.reshape(original_shape[0], original_shape[1], n_groups)
        zp_out = zp_flat.reshape(original_shape[0], original_shape[1], n_groups)
    else:
        scales_out = scales_flat
        zp_out = zp_flat

    return W_quant, scales_out, zp_out, W_int_final


def quantize_weight_groupwise_int4_with_flip(W, activation_means, group_size=128,
                                              knee_tolerance=0.0, max_flip_pct=0.01, debug=False):
    """
    Quantize weights to INT4 with heuristic flip correction (from awq_js_xl.py).

    This implements the global greedy rounding correction that flips quantization
    directions to minimize the overall error in X @ W computation.

    Uses DYNAMIC outlier detection via Kneedle algorithm instead of fixed percentile.

    Args:
        W: Weight matrix of shape [..., in_features]
        activation_means: Channel-wise activation means, shape [in_features]
        group_size: Size of each quantization group (default: 128)
        knee_tolerance: Tolerance offset for Kneedle algorithm (default: 0.0)
        max_flip_pct: Max percentage of weights that can be flipped per row (default: 0.01 = 1%)
        debug: Print debug information (default: False)

    Returns:
        W_quant: Dequantized weights (same shape as W)
        scales: Per-group scales
        zp: Per-group zero points
        W_int: Integer weights
        flip_stats: Statistics about flips (includes 'outlier_percent' from dynamic detection)
    """
    original_shape = W.shape

    # Flatten to 2D if needed
    if W.ndim > 2:
        W_flat = W.reshape(-1, W.shape[-1])
    else:
        W_flat = W.copy()

    out_features, in_features = W_flat.shape
    n_groups = (in_features + group_size - 1) // group_size
    padded_in = n_groups * group_size

    # Pad weights if needed
    if padded_in > in_features:
        W_padded = np.zeros((out_features, padded_in), dtype=W.dtype)
        W_padded[:, :in_features] = W_flat
        act_padded = np.zeros(padded_in, dtype=activation_means.dtype)
        act_padded[:in_features] = activation_means
    else:
        W_padded = W_flat
        act_padded = activation_means

    # Reshape to groups
    W_grouped = W_padded.reshape(out_features, n_groups, group_size)

    # Asymmetric quantization [0, 15]
    w_min = W_grouped.min(axis=2, keepdims=True)
    w_max = W_grouped.max(axis=2, keepdims=True)
    max_int = 15

    scale = (w_max - w_min) / max_int
    scale = np.maximum(scale, 1e-8)
    zp = np.round(-w_min / scale).clip(0, max_int)

    # Expand to full size
    scale_flat = np.repeat(scale, group_size, axis=2).reshape(out_features, padded_in)
    zp_flat = np.repeat(zp, group_size, axis=2).reshape(out_features, padded_in)

    # Initial quantization (nearest rounding)
    W_div = W_padded / scale_flat
    W_int = np.round(W_div + zp_flat).clip(0, max_int)
    W_quant = (W_int - zp_flat) * scale_flat

    # --- HEURISTIC FLIP CORRECTION ---

    # A. Calculate current error
    W_diff = W_padded - W_quant
    current_error = (W_diff * act_padded[np.newaxis, :]).sum(axis=1)  # [out_features]

    # B. Identify flip directions
    flip_dir = np.sign(W_div + zp_flat - W_int)
    flip_dir[flip_dir == 0] = 1.0

    # C. Calculate flip impacts
    flip_impacts = act_padded[np.newaxis, :] * flip_dir * scale_flat  # [out, in]

    # D. Validity masks
    target_sign = np.sign(current_error)[:, np.newaxis]
    valid_mask = (np.sign(flip_impacts) == target_sign)

    # Check if flips are in range
    w_int_proposed = W_int + flip_dir
    in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
    valid_mask = valid_mask & in_range

    # DYNAMIC outlier masking using Kneedle algorithm
    outlier_threshold, outlier_percent = compute_dynamic_outlier_threshold(
        act_padded, knee_tolerance=knee_tolerance, debug=debug
    )
    is_outlier = np.abs(act_padded) > outlier_threshold
    valid_mask = valid_mask & (~is_outlier)[np.newaxis, :]

    if debug:
        print(f"    DEBUG: Dynamic outlier detection found {outlier_percent*100:.2f}% outliers")
        print(f"    DEBUG: Outlier threshold: {outlier_threshold:.6f}")

    # E. Sorting & Optimization
    rounding_costs = np.abs(W_div + zp_flat - W_int)
    rounding_costs_masked = rounding_costs.copy()
    rounding_costs_masked[~valid_mask] = -1.0

    # Sort by rounding cost (descending)
    sorted_indices = np.argsort(-rounding_costs_masked, axis=1)  # Descending
    sorted_impacts = np.take_along_axis(flip_impacts, sorted_indices, axis=1)
    sorted_validity = np.take_along_axis(valid_mask.astype(float), sorted_indices, axis=1)
    sorted_impacts = sorted_impacts * sorted_validity

    # Cumulative sum of impacts
    cumsum_impacts = np.cumsum(sorted_impacts, axis=1)
    residuals = np.abs(current_error[:, np.newaxis] - cumsum_impacts)
    error_unsqueezed = np.abs(current_error)[:, np.newaxis]
    all_residuals = np.concatenate([error_unsqueezed, residuals], axis=1)
    best_k = np.argmin(all_residuals, axis=1)

    # F. Apply flips with max flip constraint
    idx_range = np.arange(padded_in)[np.newaxis, :]
    flip_mask_sorted = idx_range < best_k[:, np.newaxis]
    final_flips_sorted = flip_mask_sorted & (sorted_validity > 0)

    # Constraint: limit flips per row
    max_flips_per_row = int(max_flip_pct * in_features)
    cumsum_flips = np.cumsum(final_flips_sorted.astype(int), axis=1)
    within_limit = cumsum_flips <= max_flips_per_row

    # Get flip directions
    sorted_flip_dir = np.take_along_axis(flip_dir, sorted_indices, axis=1)
    sorted_flip_dir[~(final_flips_sorted & within_limit)] = 0.0

    # Apply flips
    W_int_flipped = W_int.copy()
    np.put_along_axis(W_int_flipped, sorted_indices,
                      np.take_along_axis(W_int, sorted_indices, axis=1) + sorted_flip_dir, axis=1)
    W_int_flipped = W_int_flipped.clip(0, max_int)

    # Dequantize
    W_quant_flipped = (W_int_flipped - zp_flat) * scale_flat

    # Remove padding
    if padded_in > in_features:
        W_quant_flipped = W_quant_flipped[:, :in_features]
        W_int_flipped = W_int_flipped[:, :in_features]

    # Reshape back
    W_quant_final = W_quant_flipped.reshape(original_shape)
    W_int_final = W_int_flipped.reshape(original_shape)

    # Reshape scales and zp
    scales_flat = scale.reshape(out_features, n_groups)
    zp_flat_out = zp.reshape(out_features, n_groups)

    if len(original_shape) == 3:
        scales_out = scales_flat.reshape(original_shape[0], original_shape[1], n_groups)
        zp_out = zp_flat_out.reshape(original_shape[0], original_shape[1], n_groups)
    else:
        scales_out = scales_flat
        zp_out = zp_flat_out

    # Flip statistics
    total_flips = (final_flips_sorted & within_limit).sum()
    flips_per_row = (final_flips_sorted & within_limit).sum(axis=1)

    flip_stats = {
        'total_flips': int(total_flips),
        'flips_per_row_mean': float(flips_per_row.mean()),
        'flips_per_row_max': int(flips_per_row.max()),
        'flips_per_row_min': int(flips_per_row.min()),
        'flip_rate_pct': float(total_flips / (out_features * in_features) * 100),
        'outlier_percent': float(outlier_percent)  # From dynamic Kneedle detection
    }

    return W_quant_final, scales_out, zp_out, W_int_final, flip_stats


def quantize_qkv_reflip(Wq, Wk, X, Q_orig_all, Q_heuristic_all,
                         Wq_heuristic, Wk_heuristic,
                         critical_dim_pct=0.15, knee_tolerance=0.0,
                         group_size=128, max_flip_pct=0.05,
                         correction_scale=10.0, debug=False):
    """
    ReFlip: Targeted error correction on critical head dimensions.

    Strategy:
    1. Start with heuristic quantization results (already quantized)
    2. For each head, identify critical dimensions using Kneedle on |Q_orig|
    3. Select top ~15% critical dimensions (configurable, increased from 5%)
    4. Compute target error correction for critical dimensions
    5. Apply aggressive weighted heuristic flip (5% max flip, 10x correction scale)

    Args:
        Wq: Query weights [num_heads, head_dim, hidden_dim] = [4, 128, 4096]
        Wk: Key weights [head_dim, hidden_dim] = [128, 4096]
        X: Input activation vector [hidden_dim] = [4096]
        Q_orig_all: Original Q vectors for all heads [num_heads, head_dim]
        Q_heuristic_all: Heuristic quantized Q vectors [num_heads, head_dim]
        Wq_heuristic: Heuristic quantized Wq weights (starting point)
        Wk_heuristic: Heuristic quantized Wk weights (starting point)
        critical_dim_pct: Percentage of head dims to protect (default: 0.15 = 15%)
        knee_tolerance: Tolerance for Kneedle algorithm (default: 0.0)
        group_size: Quantization group size (default: 128)
        max_flip_pct: Max flip percentage for heuristic (default: 0.05 = 5%)
        correction_scale: Scaling factor for error correction weighting (default: 10.0)
        debug: Print debug information

    Returns:
        Wq_quant_reflip: ReFlip quantized Wq weights
        Wk_quant_reflip: ReFlip quantized Wk weights (same as heuristic)
        flip_stats_reflip: Statistics about the ReFlip correction
    """
    num_heads = Wq.shape[0]
    head_dim = Wq.shape[1]
    hidden_dim = Wq.shape[2]

    # CRITICAL FIX: Start with heuristic-quantized weights, not original weights
    Wq_reflip = Wq_heuristic.copy()

    all_critical_dims = []
    all_corrections = []

    # Step 1: Identify critical dimensions for each head
    for head_idx in range(num_heads):
        Q_orig = Q_orig_all[head_idx]
        Q_heuristic = Q_heuristic_all[head_idx]

        # Compute error after heuristic quantization
        error = Q_heuristic - Q_orig  # [head_dim]

        # Use Kneedle to find critical dimensions based on |Q_orig|
        sorted_indices_desc = np.argsort(np.abs(Q_orig))[::-1]  # Descending by magnitude
        sorted_magnitudes = np.abs(Q_orig[sorted_indices_desc])

        # Apply Kneedle to first half to find threshold
        first_half = sorted_magnitudes[:head_dim // 2]
        knee_idx = find_knee_point(first_half[::-1], tolerance_offset=knee_tolerance)  # Reverse for ascending
        knee_idx = len(first_half) - knee_idx - 1  # Convert back to descending index

        # Select critical dimensions: those above knee + top critical_dim_pct
        num_critical = max(int(critical_dim_pct * head_dim), 1)
        num_critical = min(num_critical, knee_idx + 1)  # Don't exceed knee threshold

        critical_indices = sorted_indices_desc[:num_critical]
        all_critical_dims.append(critical_indices)

        # Compute target error correction for critical dimensions
        # Goal: reduce error to 0, so correction = -error
        target_corrections = -error[critical_indices]  # [num_critical]

        all_corrections.append(target_corrections)

        if debug:
            print(f"\nHead {head_idx}:")
            print(f"  Critical dimensions: {num_critical}/{head_dim} ({num_critical/head_dim*100:.1f}%)")
            print(f"  Knee index: {knee_idx}, Magnitude threshold: {sorted_magnitudes[knee_idx]:.4f}")
            print(f"  Critical dims: {critical_indices[:5]}...")  # Show first 5
            print(f"  Target corrections (first 5): {target_corrections[:5]}")

    # Step 2: Apply weighted heuristic flip for critical dimensions
    # For each critical dimension, we want to adjust Wq[head, dim, :]
    # The adjustment should reduce the error in Q[dim] = X @ Wq[dim, :]

    # Create weighted activation means for second flip
    # Weight by: |X[j]| * sum_over_heads(target_correction[dim] for critical dims)
    activation_weights = np.abs(X)  # Base weight from input magnitude

    # For each head, apply correction to critical dimensions only
    for head_idx in range(num_heads):
        critical_indices = all_critical_dims[head_idx]
        target_corrections = all_corrections[head_idx]

        if len(critical_indices) == 0:
            continue

        # Extract weight rows for critical dimensions
        W_critical = Wq_reflip[head_idx, critical_indices, :]  # [num_critical, 4096]

        # For these rows, apply aggressive heuristic flip with weighted activations
        # Weight each row's activation by the target correction magnitude * scaling factor
        for i, (dim_idx, correction) in enumerate(zip(critical_indices, target_corrections)):
            # AGGRESSIVE WEIGHTING: Scale up by correction_scale (default 10x)
            # This makes the flips more responsive to error correction needs
            weighted_act = activation_weights * np.abs(correction) * correction_scale

            # Apply single-row heuristic flip with aggressive settings
            # Quantize this single row with weighted activations
            W_row = Wq_reflip[head_idx, dim_idx:dim_idx+1, :]  # [1, 4096]

            try:
                W_row_quant, _, _, _, row_stats = quantize_weight_groupwise_int4_with_flip(
                    W_row, weighted_act, group_size=group_size,
                    knee_tolerance=knee_tolerance, max_flip_pct=max_flip_pct, debug=False
                )
                Wq_reflip[head_idx, dim_idx, :] = W_row_quant[0]

                if debug:
                    print(f"    Row {dim_idx}: correction={correction:.6f}, flips={row_stats['total_flips']}")
            except Exception as e:
                if debug:
                    print(f"  Warning: Failed to flip row {dim_idx} in head {head_idx}: {e}")
                continue

    # CRITICAL FIX: Don't re-quantize! Wq_reflip is already refined from heuristic quantization
    # Just return the refined weights directly
    Wq_quant_reflip = Wq_reflip

    # For compatibility, create dummy scales/zp (not used since weights are already quantized)
    Wq_scales = np.ones((num_heads, head_dim, hidden_dim // group_size))
    Wq_zp = np.zeros((num_heads, head_dim, hidden_dim // group_size))
    Wq_int = Wq_reflip  # Already in FP form from heuristic

    # Wk remains the same (use heuristic version)
    Wk_quant_reflip = Wk_heuristic
    Wk_scales = np.ones((head_dim, hidden_dim // group_size))
    Wk_zp = np.zeros((head_dim, hidden_dim // group_size))
    Wk_int = Wk_heuristic

    reflip_stats = {
        'critical_dims_per_head': [len(dims) for dims in all_critical_dims],
        'total_critical_dims': sum(len(dims) for dims in all_critical_dims),
        'critical_dim_pct': critical_dim_pct,
        'knee_tolerance': knee_tolerance
    }

    return (Wq_quant_reflip, Wq_scales, Wq_zp, Wq_int,
            Wk_quant_reflip, Wk_scales, Wk_zp, Wk_int,
            reflip_stats)


def compute_quantization_error(W_orig, W_quant):
    """Compute quantization error metrics."""
    diff = W_quant - W_orig
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_error = np.max(np.abs(diff))
    rel_error = mae / (np.mean(np.abs(W_orig)) + 1e-10) * 100

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'rel_error_pct': rel_error
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='INT4 Quantization with ReFlip Strategy')
    parser.add_argument('--critical-dim-pct', type=float, default=0.15,
                        help='Percentage of head dimensions to protect in ReFlip (default: 0.15 = 15%%)')
    parser.add_argument('--knee-tolerance', type=float, default=0.0,
                        help='Tolerance offset for Kneedle algorithm (default: 0.0)')
    parser.add_argument('--group-size', type=int, default=128,
                        help='Quantization group size (default: 128)')
    parser.add_argument('--max-flip-pct', type=float, default=0.05,
                        help='Max flip percentage for ReFlip (default: 0.05 = 5%%)')
    parser.add_argument('--correction-scale', type=float, default=10.0,
                        help='Error correction scaling factor for ReFlip (default: 10.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    args = parser.parse_args()

    print("="*70)
    print("INT4 Weight Quantization for GQA Attention (Group 0)")
    print("Comparing: Nearest | Heuristic | ReFlip")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Critical dim %%: {args.critical_dim_pct*100:.1f}%% (increased from 5%% for more aggressive correction)")
    print(f"  Knee tolerance: {args.knee_tolerance}")
    print(f"  Group size: {args.group_size}")
    print(f"  Max flip %% (ReFlip): {args.max_flip_pct*100:.1f}%% (5x more than heuristic)")
    print(f"  Correction scale: {args.correction_scale}x (amplifies error correction weighting)")

    # Load data
    print("\n[1] Loading data...")
    try:
        js_means = np.load('./xspot_layer0_group0/js_means.npy')  # [4096]
        Wq = np.load('./xspot_layer0_group0/Wq_group0.npy')  # [4, 128, 4096]
        Wk = np.load('./xspot_layer0_group0/Wk_group0.npy')  # [128, 4096]
        Wv = np.load('./xspot_layer0_group0/Wv_group0.npy')  # [128, 4096]
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        print("Please ensure you've run xspot.py and have the following files:")
        print("  - js_means.npy")
        print("  - Wq_group0.npy")
        print("  - Wk_group0.npy")
        print("  - Wv_group0.npy")
        return

    print(f"  JS means: {js_means.shape}")
    print(f"  Wq (group 0): {Wq.shape} [num_heads, head_dim, hidden_size]")
    print(f"  Wk (group 0): {Wk.shape} [head_dim, hidden_size]")
    print(f"  Wv (group 0): {Wv.shape} [head_dim, hidden_size]")

    # Use JS means as input X (single activation vector)
    X = js_means  # [4096]
    print(f"\n  Using JS means as input X: {X.shape}")
    print(f"  X statistics: min={X.min():.6f}, max={X.max():.6f}, "
          f"mean={X.mean():.6f}, std={X.std():.6f}")

    # Quantize weights with BOTH strategies
    print("\n[2] Quantizing weights to INT4 (group_size=128)...")
    print("  Strategy 1: Nearest rounding")
    print("  Strategy 2: Heuristic flip correction (from awq_js_xl.py)")

    # Get actual number of heads from data shape
    num_heads = Wq.shape[0]
    print(f"\n  Detected {num_heads} query heads in this group")

    # Strategy 1: Nearest rounding
    print("\n  [2a] Quantizing with NEAREST rounding...")
    print(f"    Quantizing Wq ({num_heads} heads)...")
    Wq_quant_nearest, Wq_scales_nearest, Wq_zp_nearest, Wq_int_nearest = \
        quantize_weight_groupwise_int4(Wq, group_size=128)

    print("    Quantizing Wk (1 head)...")
    Wk_quant_nearest, Wk_scales_nearest, Wk_zp_nearest, Wk_int_nearest = \
        quantize_weight_groupwise_int4(Wk, group_size=128)

    # Strategy 2: Heuristic flip correction
    print("\n  [2b] Quantizing with HEURISTIC FLIP correction...")
    print(f"    Quantizing Wq ({num_heads} heads)...")
    Wq_quant_flip, Wq_scales_flip, Wq_zp_flip, Wq_int_flip, Wq_flip_stats = \
        quantize_weight_groupwise_int4_with_flip(Wq, X, group_size=128)

    print("    Quantizing Wk (1 head)...")
    Wk_quant_flip, Wk_scales_flip, Wk_zp_flip, Wk_int_flip, Wk_flip_stats = \
        quantize_weight_groupwise_int4_with_flip(Wk, X, group_size=128)

    print(f"\n  Wq scales: {Wq_scales_nearest.shape} [num_heads, head_dim, n_groups]")
    print(f"  Wk scales: {Wk_scales_nearest.shape} [head_dim, n_groups]")

    print(f"\n  Flip statistics:")
    print(f"    Wq: {Wq_flip_stats['total_flips']} flips "
          f"({Wq_flip_stats['flip_rate_pct']:.4f}% of weights)")
    print(f"        Dynamic outlier detection: {Wq_flip_stats['outlier_percent']*100:.2f}% outliers")
    print(f"    Wk: {Wk_flip_stats['total_flips']} flips "
          f"({Wk_flip_stats['flip_rate_pct']:.4f}% of weights)")
    print(f"        Dynamic outlier detection: {Wk_flip_stats['outlier_percent']*100:.2f}% outliers")

    # Strategy 3: ReFlip (targeted error correction on critical dimensions)
    print("\n  [2c] Applying REFLIP correction (targeted critical dimensions)...")

    # Compute Q_orig and Q_heuristic for all heads (needed for ReFlip)
    Q_orig_all = np.zeros((num_heads, Wq.shape[1]))  # [num_heads, head_dim]
    Q_heuristic_all = np.zeros((num_heads, Wq.shape[1]))  # [num_heads, head_dim]

    for head_idx in range(num_heads):
        Q_orig_all[head_idx] = X @ Wq[head_idx].T  # [4096] @ [128, 4096]^T = [128]
        Q_heuristic_all[head_idx] = X @ Wq_quant_flip[head_idx].T

    # Apply ReFlip (build on top of heuristic quantization)
    (Wq_quant_reflip, Wq_scales_reflip, Wq_zp_reflip, Wq_int_reflip,
     Wk_quant_reflip, Wk_scales_reflip, Wk_zp_reflip, Wk_int_reflip,
     reflip_stats) = quantize_qkv_reflip(
        Wq, Wk, X, Q_orig_all, Q_heuristic_all,
        Wq_quant_flip, Wk_quant_flip,  # Pass heuristic-quantized weights
        critical_dim_pct=args.critical_dim_pct,
        knee_tolerance=args.knee_tolerance,
        group_size=args.group_size,
        max_flip_pct=args.max_flip_pct,
        correction_scale=args.correction_scale,
        debug=args.debug
    )

    print(f"\n  ReFlip statistics:")
    print(f"    Total critical dims: {reflip_stats['total_critical_dims']} "
          f"across {num_heads} heads")
    print(f"    Critical dims per head: {reflip_stats['critical_dims_per_head']}")
    print(f"    Target percentage: {reflip_stats['critical_dim_pct']*100:.1f}%")

    # Compute weight quantization errors
    print("\n[3] Weight quantization errors:")
    print("\n  Strategy 1: NEAREST rounding")
    for head_idx in range(num_heads):
        err = compute_quantization_error(Wq[head_idx], Wq_quant_nearest[head_idx])
        print(f"    Wq head {head_idx}: MAE={err['mae']:.6f}, "
              f"Max={err['max_error']:.6f}, Rel={err['rel_error_pct']:.4f}%")

    err_k_nearest = compute_quantization_error(Wk, Wk_quant_nearest)
    print(f"    Wk: MAE={err_k_nearest['mae']:.6f}, "
          f"Max={err_k_nearest['max_error']:.6f}, Rel={err_k_nearest['rel_error_pct']:.4f}%")

    print("\n  Strategy 2: HEURISTIC FLIP correction")
    for head_idx in range(num_heads):
        err = compute_quantization_error(Wq[head_idx], Wq_quant_flip[head_idx])
        print(f"    Wq head {head_idx}: MAE={err['mae']:.6f}, "
              f"Max={err['max_error']:.6f}, Rel={err['rel_error_pct']:.4f}%")

    err_k_flip = compute_quantization_error(Wk, Wk_quant_flip)
    print(f"    Wk: MAE={err_k_flip['mae']:.6f}, "
          f"Max={err_k_flip['max_error']:.6f}, Rel={err_k_flip['rel_error_pct']:.4f}%")

    print("\n  Strategy 3: REFLIP correction")
    for head_idx in range(num_heads):
        err = compute_quantization_error(Wq[head_idx], Wq_quant_reflip[head_idx])
        print(f"    Wq head {head_idx}: MAE={err['mae']:.6f}, "
              f"Max={err['max_error']:.6f}, Rel={err['rel_error_pct']:.4f}%")

    err_k_reflip = compute_quantization_error(Wk, Wk_quant_reflip)
    print(f"    Wk: MAE={err_k_reflip['mae']:.6f}, "
          f"Max={err_k_reflip['max_error']:.6f}, Rel={err_k_reflip['rel_error_pct']:.4f}%")

    # Compute attention scores
    print("\n[4] Computing attention scores: (X @ Wq^T) · (X @ Wk^T)")
    print("="*70)

    # num_heads already defined above from Wq.shape[0]
    results = []

    for head_idx in range(num_heads):
        print(f"\n--- Query Head {head_idx} ---")

        # Original computation
        # Q = X @ Wq^T: [4096] @ [4096, 128]^T = [128]
        Q_orig = X @ Wq[head_idx].T  # [128]

        # K = X @ Wk^T: [4096] @ [4096, 128]^T = [128]
        K_orig = X @ Wk.T  # [128] (shared across all heads)

        # Attention score = Q · K (dot product)
        score_orig = Q_orig @ K_orig  # scalar

        # Strategy 1: Nearest rounding
        Q_quant_nearest = X @ Wq_quant_nearest[head_idx].T
        K_quant_nearest = X @ Wk_quant_nearest.T
        score_quant_nearest = Q_quant_nearest @ K_quant_nearest

        # Strategy 2: Heuristic flip
        Q_quant_flip = X @ Wq_quant_flip[head_idx].T
        K_quant_flip = X @ Wk_quant_flip.T
        score_quant_flip = Q_quant_flip @ K_quant_flip

        # Strategy 3: ReFlip
        Q_quant_reflip = X @ Wq_quant_reflip[head_idx].T
        K_quant_reflip = X @ Wk_quant_reflip.T
        score_quant_reflip = Q_quant_reflip @ K_quant_reflip

        # Errors
        error_nearest = score_quant_nearest - score_orig
        rel_error_nearest = error_nearest / (np.abs(score_orig) + 1e-10) * 100

        error_flip = score_quant_flip - score_orig
        rel_error_flip = error_flip / (np.abs(score_orig) + 1e-10) * 100

        error_reflip = score_quant_reflip - score_orig
        rel_error_reflip = error_reflip / (np.abs(score_orig) + 1e-10) * 100

        improvement_h = error_nearest - error_flip
        improvement_h_pct = (abs(error_nearest) - abs(error_flip)) / (abs(error_nearest) + 1e-10) * 100

        improvement_r = error_nearest - error_reflip
        improvement_r_pct = (abs(error_nearest) - abs(error_reflip)) / (abs(error_nearest) + 1e-10) * 100

        improvement_hr = error_flip - error_reflip
        improvement_hr_pct = (abs(error_flip) - abs(error_reflip)) / (abs(error_flip) + 1e-10) * 100

        print(f"Original score:           {score_orig:15.6f}")
        print(f"\nStrategy 1 (Nearest):")
        print(f"  Score:                  {score_quant_nearest:15.6f}")
        print(f"  Absolute error:         {error_nearest:15.6f}")
        print(f"  Relative error:         {rel_error_nearest:15.4f}%")
        print(f"\nStrategy 2 (Heuristic):")
        print(f"  Score:                  {score_quant_flip:15.6f}")
        print(f"  Absolute error:         {error_flip:15.6f}")
        print(f"  Relative error:         {rel_error_flip:15.4f}%")
        print(f"\nStrategy 3 (ReFlip):")
        print(f"  Score:                  {score_quant_reflip:15.6f}")
        print(f"  Absolute error:         {error_reflip:15.6f}")
        print(f"  Relative error:         {rel_error_reflip:15.4f}%")
        print(f"\nImprovements:")
        print(f"  Nearest → Heuristic:    {improvement_h_pct:15.2f}%")
        print(f"  Nearest → ReFlip:       {improvement_r_pct:15.2f}%")
        print(f"  Heuristic → ReFlip:     {improvement_hr_pct:15.2f}%")

        # Scale statistics for this head
        head_scales_nearest = Wq_scales_nearest[head_idx].flatten()
        head_scales_flip = Wq_scales_flip[head_idx].flatten()
        head_scales_reflip = Wq_scales_reflip[head_idx].flatten()

        results.append({
            'head': head_idx,
            'score_orig': score_orig,
            'score_quant_nearest': score_quant_nearest,
            'score_quant_flip': score_quant_flip,
            'score_quant_reflip': score_quant_reflip,
            'error_nearest': error_nearest,
            'error_flip': error_flip,
            'error_reflip': error_reflip,
            'rel_error_nearest': rel_error_nearest,
            'rel_error_flip': rel_error_flip,
            'rel_error_reflip': rel_error_reflip,
            'improvement_h_pct': improvement_h_pct,
            'improvement_r_pct': improvement_r_pct,
            'improvement_hr_pct': improvement_hr_pct,
            'wq_scales_nearest': head_scales_nearest,
            'wq_scales_flip': head_scales_flip,
            'wq_scales_reflip': head_scales_reflip,
            'Q_orig': Q_orig,
            'Q_quant_nearest': Q_quant_nearest,
            'Q_quant_flip': Q_quant_flip,
            'Q_quant_reflip': Q_quant_reflip,
            'K_orig': K_orig,
            'K_quant_nearest': K_quant_nearest,
            'K_quant_flip': K_quant_flip,
            'K_quant_reflip': K_quant_reflip
        })

    # Summary table
    print("\n" + "="*100)
    print("SUMMARY: Attention Score Quantization Comparison (3 Strategies)")
    print("="*100)
    print(f"{'Head':<6} {'Original':<13} {'Nearest':<13} {'Heuristic':<13} {'ReFlip':<13} "
          f"{'Err(N)':<10} {'Err(H)':<10} {'Err(R)':<10}")
    print("-"*100)
    for r in results:
        print(f"{r['head']:<6} {r['score_orig']:<13.6f} "
              f"{r['score_quant_nearest']:<13.6f} {r['score_quant_flip']:<13.6f} {r['score_quant_reflip']:<13.6f} "
              f"{r['error_nearest']:<10.4f} {r['error_flip']:<10.4f} {r['error_reflip']:<10.4f}")

    errors_nearest = [r['error_nearest'] for r in results]
    errors_flip = [r['error_flip'] for r in results]
    errors_reflip = [r['error_reflip'] for r in results]
    rel_errors_nearest = [r['rel_error_nearest'] for r in results]
    rel_errors_flip = [r['rel_error_flip'] for r in results]
    rel_errors_reflip = [r['rel_error_reflip'] for r in results]
    improvements_h = [r['improvement_h_pct'] for r in results]
    improvements_r = [r['improvement_r_pct'] for r in results]
    improvements_hr = [r['improvement_hr_pct'] for r in results]

    print("-"*100)
    print("\nStrategy 1 (Nearest):")
    print(f"  Mean absolute error:    {np.mean(np.abs(errors_nearest)):.6f}")
    print(f"  Mean relative error:    {np.mean(np.abs(rel_errors_nearest)):.4f}%")
    print(f"  Max absolute error:     {np.max(np.abs(errors_nearest)):.6f}")

    print("\nStrategy 2 (Heuristic):")
    print(f"  Mean absolute error:    {np.mean(np.abs(errors_flip)):.6f}")
    print(f"  Mean relative error:    {np.mean(np.abs(rel_errors_flip)):.4f}%")
    print(f"  Max absolute error:     {np.max(np.abs(errors_flip)):.6f}")

    print("\nStrategy 3 (ReFlip):")
    print(f"  Mean absolute error:    {np.mean(np.abs(errors_reflip)):.6f}")
    print(f"  Mean relative error:    {np.mean(np.abs(rel_errors_reflip)):.4f}%")
    print(f"  Max absolute error:     {np.max(np.abs(errors_reflip)):.6f}")

    print("\nImprovements:")
    print(f"  Nearest → Heuristic:    {np.mean(improvements_h):.2f}% (mean)")
    print(f"  Nearest → ReFlip:       {np.mean(improvements_r):.2f}% (mean)")
    print(f"  Heuristic → ReFlip:     {np.mean(improvements_hr):.2f}% (mean)")
    print(f"  Best ReFlip reduction:  {np.max(improvements_r):.2f}% (head {np.argmax(improvements_r)})")
    print(f"  Worst ReFlip reduction: {np.min(improvements_r):.2f}% (head {np.argmin(improvements_r)})")

    print("\nFlip Statistics:")
    print(f"  Wq total flips:         {Wq_flip_stats['total_flips']:,} "
          f"({Wq_flip_stats['flip_rate_pct']:.4f}% of weights)")
    print(f"  Wk total flips:         {Wk_flip_stats['total_flips']:,} "
          f"({Wk_flip_stats['flip_rate_pct']:.4f}% of weights)")
    print("="*70)

    # Visualization
    print("\n[5] Generating visualizations...")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    heads = [r['head'] for r in results]
    scores_orig = [r['score_orig'] for r in results]
    scores_quant_nearest = [r['score_quant_nearest'] for r in results]
    scores_quant_flip = [r['score_quant_flip'] for r in results]
    scores_quant_reflip = [r['score_quant_reflip'] for r in results]

    # 1. Attention scores comparison (4-way)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(heads))
    width = 0.2
    ax1.bar(x - 1.5*width, scores_orig, width, label='Original', alpha=0.8, color='blue')
    ax1.bar(x - 0.5*width, scores_quant_nearest, width, label='Nearest', alpha=0.8, color='orange')
    ax1.bar(x + 0.5*width, scores_quant_flip, width, label='Heuristic', alpha=0.8, color='green')
    ax1.bar(x + 1.5*width, scores_quant_reflip, width, label='ReFlip', alpha=0.8, color='purple')
    ax1.set_xlabel('Query Head')
    ax1.set_ylabel('Attention Score (Q·K)')
    ax1.set_title('Original vs Quantized Attention Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'H{i}' for i in heads])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Error comparison (3-way)
    ax2 = fig.add_subplot(gs[0, 1])
    width2 = 0.25
    ax2.bar(x - width2, errors_nearest, width2, label='Nearest', alpha=0.7, color='orange')
    ax2.bar(x, errors_flip, width2, label='Heuristic', alpha=0.7, color='green')
    ax2.bar(x + width2, errors_reflip, width2, label='ReFlip', alpha=0.7, color='purple')
    ax2.set_xlabel('Query Head')
    ax2.set_ylabel('Error (Quantized - Original)')
    ax2.set_title('Absolute Error Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'H{i}' for i in heads])
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Error reduction % (comparing all strategies)
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(heads))
    width = 0.25
    ax3.bar(x - width, improvements_h, width, alpha=0.7, color='orange', label='N→H')
    ax3.bar(x, improvements_r, width, alpha=0.7, color='green', label='N→R')
    ax3.bar(x + width, improvements_hr, width, alpha=0.7, color='purple', label='H→R')
    ax3.set_xlabel('Query Head')
    ax3.set_ylabel('Error Reduction (%)')
    ax3.set_title('Error Reduction Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'H{i}' for i in heads])
    ax3.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Relative errors comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(x - width/2, rel_errors_nearest, width, label='Nearest', alpha=0.7, color='orange')
    ax4.bar(x + width/2, rel_errors_flip, width, label='Heuristic', alpha=0.7, color='green')
    ax4.set_xlabel('Query Head')
    ax4.set_ylabel('Relative Error (%)')
    ax4.set_title('Relative Error Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'H{i}' for i in heads])
    ax4.axhline(0, color='black', linestyle='--', linewidth=1)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Scale distributions comparison (nearest vs flip)
    ax5 = fig.add_subplot(gs[1, 1])
    scale_data_nearest = [results[i]['wq_scales_nearest'] for i in range(num_heads)]
    scale_data_flip = [results[i]['wq_scales_flip'] for i in range(num_heads)]
    # Show first head as example
    ax5.hist(scale_data_nearest[0], bins=30, alpha=0.5, label='Nearest (H0)', color='orange')
    ax5.hist(scale_data_flip[0], bins=30, alpha=0.5, label='Heuristic (H0)', color='green')
    ax5.set_xlabel('Scale Value')
    ax5.set_ylabel('Count')
    ax5.set_title('Wq Scale Distribution (Head 0)')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Wk scale comparison
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(Wk_scales_nearest.flatten(), bins=30, alpha=0.5, label='Nearest',
             color='orange', edgecolor='black')
    ax6.hist(Wk_scales_flip.flatten(), bins=30, alpha=0.5, label='Heuristic',
             color='green', edgecolor='black')
    ax6.set_xlabel('Scale Value')
    ax6.set_ylabel('Count')
    ax6.set_title(f'Wk Scale Distribution Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Q vector comparison (head 0) - 3-way
    ax7 = fig.add_subplot(gs[2, 0])
    Q0_orig = results[0]['Q_orig']
    Q0_nearest = results[0]['Q_quant_nearest']
    Q0_flip = results[0]['Q_quant_flip']
    ax7.plot(Q0_orig, label='Original', alpha=0.7, linewidth=1.5, color='blue')
    ax7.plot(Q0_nearest, label='Nearest', alpha=0.7, linewidth=1.5, color='orange')
    ax7.plot(Q0_flip, label='Heuristic', alpha=0.7, linewidth=1.5, color='green')
    ax7.set_xlabel('Dimension')
    ax7.set_ylabel('Value')
    ax7.set_title('Query Vector Q (Head 0)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. K vector comparison (shared) - 3-way
    ax8 = fig.add_subplot(gs[2, 1])
    K_orig = results[0]['K_orig']
    K_nearest = results[0]['K_quant_nearest']
    K_flip = results[0]['K_quant_flip']
    ax8.plot(K_orig, label='Original', alpha=0.7, linewidth=1.5, color='blue')
    ax8.plot(K_nearest, label='Nearest', alpha=0.7, linewidth=1.5, color='orange')
    ax8.plot(K_flip, label='Heuristic', alpha=0.7, linewidth=1.5, color='green')
    ax8.set_xlabel('Dimension')
    ax8.set_ylabel('Value')
    ax8.set_title('Key Vector K (Shared)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Q-K error comparison (head 0)
    ax9 = fig.add_subplot(gs[2, 2])
    Q_error_nearest = Q0_nearest - Q0_orig
    Q_error_flip = Q0_flip - Q0_orig
    ax9.scatter(Q_error_nearest, Q_error_flip, alpha=0.5, s=10)
    ax9.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax9.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax9.plot([Q_error_nearest.min(), Q_error_nearest.max()],
             [Q_error_nearest.min(), Q_error_nearest.max()],
             'k--', alpha=0.5, label='y=x (no improvement)')
    ax9.set_xlabel('Q Error (Nearest)')
    ax9.set_ylabel('Q Error (Heuristic)')
    ax9.set_title('Q Error: Nearest vs Heuristic (Head 0)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # 10. Flip statistics per head
    ax10 = fig.add_subplot(gs[3, 0])
    # This would require per-head flip stats, skip for now
    ax10.text(0.5, 0.5, f"Total Wq flips: {Wq_flip_stats['total_flips']:,}\n"
                         f"Flip rate: {Wq_flip_stats['flip_rate_pct']:.4f}%\n\n"
                         f"Total Wk flips: {Wk_flip_stats['total_flips']:,}\n"
                         f"Flip rate: {Wk_flip_stats['flip_rate_pct']:.4f}%",
              ha='center', va='center', fontsize=12,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax10.set_xlim(0, 1)
    ax10.set_ylim(0, 1)
    ax10.axis('off')
    ax10.set_title('Flip Statistics')

    # 11. Error reduction summary
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.text(0.5, 0.5, f"Error Reduction (mean):\n"
                         f"N→H: {np.mean(improvements_h):.2f}%\n"
                         f"N→R: {np.mean(improvements_r):.2f}%\n"
                         f"H→R: {np.mean(improvements_hr):.2f}%\n\n"
                         f"Strategy MAE:\n"
                         f"Nearest: {np.mean(np.abs(errors_nearest)):.6f}\n"
                         f"Heuristic: {np.mean(np.abs(errors_flip)):.6f}\n"
                         f"ReFlip: {np.mean(np.abs(errors_reflip)):.6f}",
              ha='center', va='center', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax11.set_xlim(0, 1)
    ax11.set_ylim(0, 1)
    ax11.axis('off')
    ax11.set_title('Summary Statistics')

    # 12. Reserved for future use
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.axis('off')

    plt.savefig('attention_quantization_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: attention_quantization_analysis.png")

    # New figure: Sorted errors over 128 head dimensions for 4 heads
    # Split into two plots per head: errors (top) and magnitude (bottom)
    print("\n[5b] Generating sorted error visualization...")
    fig2 = plt.figure(figsize=(16, 16))
    gs2 = fig2.add_gridspec(4, 4, hspace=0.4, wspace=0.3, height_ratios=[1, 0.6, 1, 0.6])

    for head_idx in range(num_heads):
        # Calculate grid position
        col = head_idx % 2  # 0 or 1
        row_offset = (head_idx // 2) * 2  # 0 or 2

        # Top subplot: Errors
        ax_error = fig2.add_subplot(gs2[row_offset, col*2:(col+1)*2])

        # Bottom subplot: Magnitude
        ax_mag = fig2.add_subplot(gs2[row_offset + 1, col*2:(col+1)*2], sharex=ax_error)

        # Get Q vectors for this head
        Q_orig = results[head_idx]['Q_orig']
        Q_nearest = results[head_idx]['Q_quant_nearest']
        Q_flip = results[head_idx]['Q_quant_flip']

        # Compute errors (keeping sign)
        Q_error_nearest = Q_nearest - Q_orig
        Q_error_flip = Q_flip - Q_orig

        # Get sort indices from nearest error
        sort_indices = np.argsort(Q_error_nearest)

        # Sort errors and Q_orig magnitude using the same indices
        sorted_error_nearest = Q_error_nearest[sort_indices]
        sorted_error_flip = Q_error_flip[sort_indices]
        sorted_Q_magnitude = np.abs(Q_orig[sort_indices])

        x_dims = np.arange(len(sorted_error_nearest))

        # Top plot: Errors
        ax_error.plot(x_dims, sorted_error_nearest, label='Error (Nearest)', alpha=0.8,
                      linewidth=1.5, color='orange', marker='o', markersize=2)
        ax_error.plot(x_dims, sorted_error_flip, label='Error (Heuristic)', alpha=0.8,
                      linewidth=1.5, color='green', marker='s', markersize=2)
        ax_error.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Statistics
        mean_nearest = Q_error_nearest.mean()
        mean_flip = Q_error_flip.mean()
        std_nearest = Q_error_nearest.std()
        std_flip = Q_error_flip.std()

        ax_error.set_ylabel('Error', fontsize=10)
        ax_error.set_title(f'Head {head_idx}: Sorted Q Errors (128 dims)\n'
                          f'Nearest: μ={mean_nearest:.4f}, σ={std_nearest:.4f} | '
                          f'Heuristic: μ={mean_flip:.4f}, σ={std_flip:.4f}',
                          fontsize=9)
        ax_error.legend(loc='best', fontsize=9)
        ax_error.grid(True, alpha=0.3)
        ax_error.tick_params(labelbottom=False)

        # Bottom plot: Magnitude
        ax_mag.plot(x_dims, sorted_Q_magnitude, label='|Q_orig|', alpha=0.7,
                    linewidth=1.5, color='blue', marker='s', markersize=3)
        ax_mag.set_xlabel('Sorted Dimension Index', fontsize=10)
        ax_mag.set_ylabel('|Q_orig| Magnitude', fontsize=10)
        ax_mag.legend(loc='best', fontsize=9)
        ax_mag.grid(True, alpha=0.3)

    plt.savefig('sorted_error_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: sorted_error_comparison.png")

    # Save detailed results
    print("\n[6] Saving detailed results...")
    np.savez('quantization_results.npz',
             # Input
             X=X,
             js_means=js_means,
             # Original weights
             Wq_orig=Wq,
             Wk_orig=Wk,
             # Quantized weights - Nearest
             Wq_quant_nearest=Wq_quant_nearest,
             Wk_quant_nearest=Wk_quant_nearest,
             Wq_int_nearest=Wq_int_nearest,
             Wk_int_nearest=Wk_int_nearest,
             # Quantized weights - Heuristic
             Wq_quant_flip=Wq_quant_flip,
             Wk_quant_flip=Wk_quant_flip,
             Wq_int_flip=Wq_int_flip,
             Wk_int_flip=Wk_int_flip,
             # Scales and zero points - Nearest
             Wq_scales_nearest=Wq_scales_nearest,
             Wk_scales_nearest=Wk_scales_nearest,
             Wq_zp_nearest=Wq_zp_nearest,
             Wk_zp_nearest=Wk_zp_nearest,
             # Scales and zero points - Heuristic
             Wq_scales_flip=Wq_scales_flip,
             Wk_scales_flip=Wk_scales_flip,
             Wq_zp_flip=Wq_zp_flip,
             Wk_zp_flip=Wk_zp_flip,
             # Attention scores
             attention_scores_orig=np.array(scores_orig),
             attention_scores_nearest=np.array(scores_quant_nearest),
             attention_scores_flip=np.array(scores_quant_flip),
             attention_scores_reflip=np.array(scores_quant_reflip),
             # Errors
             errors_nearest=np.array(errors_nearest),
             errors_flip=np.array(errors_flip),
             errors_reflip=np.array(errors_reflip),
             rel_errors_nearest=np.array(rel_errors_nearest),
             rel_errors_flip=np.array(rel_errors_flip),
             rel_errors_reflip=np.array(rel_errors_reflip),
             improvements_h=np.array(improvements_h),
             improvements_r=np.array(improvements_r),
             improvements_hr=np.array(improvements_hr),
             # Query and Key vectors
             Q_orig=[r['Q_orig'] for r in results],
             Q_quant_nearest=[r['Q_quant_nearest'] for r in results],
             Q_quant_flip=[r['Q_quant_flip'] for r in results],
             Q_quant_reflip=[r['Q_quant_reflip'] for r in results],
             K_orig=K_orig,
             K_quant_nearest=K_nearest,
             K_quant_flip=K_flip,
             K_quant_reflip=K_quant_reflip)
    print(f"  Saved: quantization_results.npz")

    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70)
    print("\nKey findings:")
    print(f"  1. Nearest quantization:")
    print(f"     - Mean attention score error: {np.mean(np.abs(errors_nearest)):.6f} "
          f"({np.mean(np.abs(rel_errors_nearest)):.4f}%)")
    print(f"  2. Heuristic quantization:")
    print(f"     - Mean attention score error: {np.mean(np.abs(errors_flip)):.6f} "
          f"({np.mean(np.abs(rel_errors_flip)):.4f}%)")
    print(f"     - Flipped {Wq_flip_stats['total_flips'] + Wk_flip_stats['total_flips']:,} weights "
          f"({((Wq_flip_stats['total_flips'] + Wk_flip_stats['total_flips']) / (Wq.size + Wk.size) * 100):.4f}%)")
    print(f"  3. ReFlip quantization:")
    print(f"     - Mean attention score error: {np.mean(np.abs(errors_reflip)):.6f} "
          f"({np.mean(np.abs(rel_errors_reflip)):.4f}%)")
    print(f"     - Protected {reflip_stats['total_critical_dims']} critical dimensions "
          f"({reflip_stats['critical_dim_pct']*100:.1f}% target)")
    print(f"\n  Improvements:")
    print(f"     - Nearest → Heuristic: {np.mean(improvements_h):.2f}% average error reduction")
    print(f"     - Nearest → ReFlip:    {np.mean(improvements_r):.2f}% average error reduction")
    print(f"     - Heuristic → ReFlip:  {np.mean(improvements_hr):.2f}% average error reduction")
    print(f"     - Best ReFlip improvement: {np.max(improvements_r):.2f}% (head {np.argmax(improvements_r)})")


if __name__ == '__main__':
    main()
