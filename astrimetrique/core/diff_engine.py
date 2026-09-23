"""
Photometric Diffing and Transient Detection Engine.
Performs background normalization, gain matching, noise-thresholded image subtraction,
and automated detection of moving asteroid candidates.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import label, center_of_mass


@dataclass
class MovingCandidate:
    """Detected moving object / transient candidate in difference map."""
    candidate_id: str
    x: float  # Centroid X (pixels)
    y: float  # Centroid Y (pixels)
    snr: float  # Significance in units of background sigma
    peak_diff: float  # Peak difference intensity (ADU)
    total_flux_diff: float  # Integrated difference flux
    pixel_area: int  # Number of pixels above threshold
    polarity: str = "Positive (Epoch 1)"  # 'Positive (Epoch 1)' or 'Negative (Epoch 2)'


@dataclass
class DiffResult:
    """Complete results from image subtraction and candidate detection."""
    diff_signed: np.ndarray  # I_A - s * I_B_warped
    diff_abs: np.ndarray  # |I_A - s * I_B_warped|
    diff_filtered: np.ndarray  # Thresholded significance map
    bg_a: float  # Background level of Image A
    bg_b: float  # Background level of Image B
    gain_scale_b: float  # Photometric scaling factor s
    noise_sigma: float  # Robust background noise (MAD)
    threshold_sigma: float  # Detection threshold in units of sigma
    candidates: List[MovingCandidate] = field(default_factory=list)


def estimate_robust_background_and_noise(data: np.ndarray) -> Tuple[float, float]:
    """
    Estimate background median and robust noise standard deviation (via MAD).
    """
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return 0.0, 1.0

    # Median as robust background estimate
    median_bg = float(np.median(valid))
    
    # Median Absolute Deviation (MAD) -> sigma = 1.4826 * MAD
    abs_deviations = np.abs(valid - median_bg)
    mad = float(np.median(abs_deviations))
    sigma = max(1.4826 * mad, 1e-4)

    return median_bg, sigma


def match_background_and_gain(
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Calculate background offsets (bg_A, bg_B) and photometric gain ratio s:
    Normalized Image B = s * (Image B - bg_B) + bg_A
    Uses robust linear regression on source signal pixels.
    """
    valid_mask = (
        np.isfinite(image_a)
        & np.isfinite(image_b)
        & (image_a > 0.0)
        & (image_b > 0.0)
    )

    if not np.any(valid_mask):
        return 0.0, 0.0, 1.0

    sub_a = image_a[valid_mask]
    sub_b = image_b[valid_mask]

    # Robust background medians
    bg_a, sigma_a = estimate_robust_background_and_noise(sub_a)
    bg_b, sigma_b = estimate_robust_background_and_noise(sub_b)

    # Identify significant signal pixels (stars) present in both images
    sig_thresh_a = bg_a + 3.0 * sigma_a
    sig_thresh_b = bg_b + 3.0 * sigma_b
    star_mask = (image_a > sig_thresh_a) & (image_b > sig_thresh_b) & valid_mask

    val_a = image_a[star_mask] - bg_a
    val_b = image_b[star_mask] - bg_b

    if len(val_a) >= 15:
        # Robust linear slope: s = sum(val_a * val_b) / sum(val_b^2)
        gain_scale = float(np.sum(val_a * val_b) / (np.sum(val_b ** 2) + 1e-9))
    else:
        # Fallback to 1.0 or percentile ratio
        gain_scale = 1.0

    # Constrain to realistic range [0.2, 5.0]
    gain_scale = float(np.clip(gain_scale, 0.2, 5.0))

    return bg_a, bg_b, gain_scale


def compute_difference_map(
    image_a: np.ndarray,
    image_b_warped: np.ndarray,
    threshold_sigma: float = 3.5,
    min_area: int = 3,
) -> DiffResult:
    """
    Perform gain-matched, background-subtracted image difference:
    Diff = (Image A - bg_A) - s * (Image B_warped - bg_B)
    """
    img_a = np.nan_to_num(image_a, nan=0.0).astype(np.float32)
    img_b = np.nan_to_num(image_b_warped, nan=0.0).astype(np.float32)

    bg_a, bg_b, gain_b = match_background_and_gain(img_a, img_b)

    # Subtract background and scale
    norm_a = img_a - bg_a
    norm_b = gain_b * (img_b - bg_b)

    # Only compute difference where both images have valid coverage
    valid_overlap = (img_a > 0.0) & (img_b > 0.0)

    diff_signed = np.zeros_like(img_a, dtype=np.float32)
    diff_signed[valid_overlap] = norm_a[valid_overlap] - norm_b[valid_overlap]

    diff_abs = np.abs(diff_signed)

    # Estimate residual noise sigma on difference array
    overlap_diff = diff_signed[valid_overlap]
    if overlap_diff.size > 0:
        _, noise_sigma = estimate_robust_background_and_noise(overlap_diff)
    else:
        noise_sigma = 1.0

    # Noise thresholding
    sig_threshold = threshold_sigma * noise_sigma
    diff_filtered = np.where(diff_abs >= sig_threshold, diff_abs, 0.0).astype(np.float32)

    # Candidate extraction via connected component analysis
    candidates: List[MovingCandidate] = []
    
    # 1. Positive transients (present in Image A, absent or shifted in Image B)
    pos_mask = (diff_signed >= sig_threshold) & valid_overlap
    labeled_pos, num_pos = label(pos_mask)
    
    cand_idx = 1
    for label_id in range(1, num_pos + 1):
        cluster_mask = (labeled_pos == label_id)
        area = int(np.sum(cluster_mask))
        if area >= min_area:
            # Weighted center of mass
            weights = diff_signed * cluster_mask
            total_flux = float(np.sum(weights))
            if total_flux > 0:
                cy, cx = center_of_mass(weights)
                peak = float(np.max(diff_signed[cluster_mask]))
                snr = float(peak / noise_sigma)
                candidates.append(MovingCandidate(
                    candidate_id=f"CAND-{cand_idx:02d} (Epoch A)",
                    x=float(cx),
                    y=float(cy),
                    snr=snr,
                    peak_diff=peak,
                    total_flux_diff=total_flux,
                    pixel_area=area,
                    polarity="Epoch 1 (Positive)",
                ))
                cand_idx += 1

    # 2. Negative transients (present in Image B / Epoch 2, absent in Image A)
    neg_mask = (diff_signed <= -sig_threshold) & valid_overlap
    labeled_neg, num_neg = label(neg_mask)
    for label_id in range(1, num_neg + 1):
        cluster_mask = (labeled_neg == label_id)
        area = int(np.sum(cluster_mask))
        if area >= min_area:
            weights = np.abs(diff_signed) * cluster_mask
            total_flux = float(np.sum(weights))
            if total_flux > 0:
                cy, cx = center_of_mass(weights)
                peak = float(np.max(weights[cluster_mask]))
                snr = float(peak / noise_sigma)
                candidates.append(MovingCandidate(
                    candidate_id=f"CAND-{cand_idx:02d} (Epoch B)",
                    x=float(cx),
                    y=float(cy),
                    snr=snr,
                    peak_diff=peak,
                    total_flux_diff=total_flux,
                    pixel_area=area,
                    polarity="Epoch 2 (Negative)",
                ))
                cand_idx += 1

    # Sort candidates by SNR descending
    candidates.sort(key=lambda c: c.snr, reverse=True)

    return DiffResult(
        diff_signed=diff_signed,
        diff_abs=diff_abs,
        diff_filtered=diff_filtered,
        bg_a=bg_a,
        bg_b=bg_b,
        gain_scale_b=gain_b,
        noise_sigma=noise_sigma,
        threshold_sigma=threshold_sigma,
        candidates=candidates,
    )
