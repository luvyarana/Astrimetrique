"""
Photometric Diffing and Topological Transient Detection Engine.
Features background normalization, robust gain matching, PSF-matched Gaussian convolution,
adaptive saturated core blooming masking, and topological dipole symmetry filtering.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import label, center_of_mass, gaussian_filter


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
    polarity: str = "Epoch 1 (Positive)"  # 'Epoch 1 (Positive)' or 'Epoch 2 (Negative)'
    is_dipole: bool = False
    is_saturated: bool = False


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
    saturation_mask: Optional[np.ndarray] = None  # Saturated core mask
    candidates: List[MovingCandidate] = field(default_factory=list)
    rejected_dipoles_count: int = 0
    rejected_saturated_count: int = 0


def estimate_robust_background_and_noise(data: np.ndarray) -> Tuple[float, float]:
    """
    Estimate background median and robust noise standard deviation (via MAD).
    """
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return 0.0, 1.0

    median_bg = float(np.median(valid))
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
        gain_scale = 1.0

    gain_scale = float(np.clip(gain_scale, 0.2, 5.0))
    return bg_a, bg_b, gain_scale


def build_adaptive_saturation_mask(
    image_a: np.ndarray,
    image_b: np.ndarray,
    saturation_limit: float = 55000.0,
    base_radius: int = 5,
) -> np.ndarray:
    """
    Create an adaptive binary mask covering saturated cores and their blooming radius
    proportional to brightness: R = base_radius + k * log(Intensity / saturation_limit).
    """
    h, w = image_a.shape
    sat_mask = np.zeros((h, w), dtype=bool)

    # Find saturated coordinates in either image
    sat_a_coords = np.argwhere(image_a >= saturation_limit)
    sat_b_coords = np.argwhere(image_b >= saturation_limit)
    all_sat = np.vstack([sat_a_coords, sat_b_coords]) if len(sat_b_coords) > 0 else sat_a_coords

    if len(all_sat) == 0:
        return sat_mask

    y_grid, x_grid = np.indices((h, w))

    # Cluster or iterate saturated points
    for (sy, sx) in all_sat:
        val_a = float(image_a[sy, sx]) if (0 <= sy < h and 0 <= sx < w) else 0.0
        val_b = float(image_b[sy, sx]) if (0 <= sy < h and 0 <= sx < w) else 0.0
        peak = max(val_a, val_b)

        # Adaptive blooming radius: R = R_0 + 3.0 * ln(Peak / Sat_limit + 1.0)
        r = base_radius + 3.0 * np.log(max(1.0, peak / saturation_limit))
        r_sq = r ** 2

        # Bounding box for efficiency
        xmin = max(0, int(sx - r - 1))
        xmax = min(w, int(sx + r + 2))
        ymin = max(0, int(sy - r - 1))
        ymax = min(h, int(sy + r + 2))

        dist_sq = (x_grid[ymin:ymax, xmin:xmax] - sx) ** 2 + (y_grid[ymin:ymax, xmin:xmax] - sy) ** 2
        sat_mask[ymin:ymax, xmin:xmax] |= (dist_sq <= r_sq)

    return sat_mask


def is_dipole_artifact(
    diff_signed: np.ndarray,
    cx: float,
    cy: float,
    window_radius: int = 3,
    min_dipole_ratio: float = 0.20,
) -> bool:
    """
    Topological Signature Analysis:
    Identifies if a candidate is a symmetric dipole artifact (positive peak adjacent to negative trough)
    caused by sub-pixel misalignment of a stationary star.
    
    A dipole artifact satisfies:
    1. Both strong positive and negative flux exist in immediate neighborhood.
    2. |P + N| < |P - N| (opposite polarity with comparable magnitude).
    3. min(|P|, |N|) / max(|P|, |N|) >= min_dipole_ratio.
    """
    h, w = diff_signed.shape
    ix = int(round(cx))
    iy = int(round(cy))

    x0 = max(0, ix - window_radius)
    x1 = min(w, ix + window_radius + 1)
    y0 = max(0, iy - window_radius)
    y1 = min(h, iy + window_radius + 1)

    patch = diff_signed[y0:y1, x0:x1]
    if patch.size < 4:
        return False

    pos_max = float(np.max(patch))
    neg_min = float(np.min(patch))

    # Must contain both positive and negative values
    if pos_max > 0.0 and neg_min < 0.0:
        # Check dipole cancellation condition |P + N| < |P - N|
        if abs(pos_max + neg_min) < abs(pos_max - neg_min):
            ratio = min(pos_max, abs(neg_min)) / max(pos_max, abs(neg_min))
            if ratio >= min_dipole_ratio:
                return True

    return False


def compute_difference_map(
    image_a: np.ndarray,
    image_b_warped: np.ndarray,
    threshold_sigma: float = 3.5,
    min_area: int = 3,
    psf_match_sigma: float = 0.8,
    saturation_limit: float = 55000.0,
    enable_dipole_filter: bool = True,
    enable_saturation_mask: bool = True,
) -> DiffResult:
    """
    Perform advanced photometric image subtraction with:
    1. Background & Gain Matching
    2. Adaptive Saturated Core Masking
    3. PSF-Matched Gaussian Convolution
    4. Topological Dipole Signature Analysis
    """
    img_a = np.nan_to_num(image_a, nan=0.0).astype(np.float32)
    img_b = np.nan_to_num(image_b_warped, nan=0.0).astype(np.float32)

    # 1. Background offsets and Gain matching
    bg_a, bg_b, gain_b = match_background_and_gain(img_a, img_b)

    # Background-subtracted & scaled arrays
    norm_a = img_a - bg_a
    norm_b = gain_b * (img_b - bg_b)

    # Valid overlap mask
    valid_overlap = (img_a > 0.0) & (img_b > 0.0)

    # 2. Adaptive Saturated Core Mask
    if enable_saturation_mask:
        sat_mask = build_adaptive_saturation_mask(img_a, img_b, saturation_limit=saturation_limit)
        valid_overlap &= ~sat_mask
    else:
        sat_mask = np.zeros(img_a.shape, dtype=bool)

    # 3. PSF-Matching via Gaussian Convolution
    # Convolve with a small Gaussian kernel to match PSFs and suppress high-frequency edge ringing
    if psf_match_sigma > 0.0:
        norm_a_conv = gaussian_filter(norm_a, sigma=psf_match_sigma)
        norm_b_conv = gaussian_filter(norm_b, sigma=psf_match_sigma)
    else:
        norm_a_conv = norm_a
        norm_b_conv = norm_b

    # Compute Difference Map
    diff_signed = np.zeros_like(img_a, dtype=np.float32)
    diff_signed[valid_overlap] = norm_a_conv[valid_overlap] - norm_b_conv[valid_overlap]
    diff_abs = np.abs(diff_signed)

    # Recalculate background noise sigma on difference map via MAD
    overlap_diff = diff_signed[valid_overlap]
    if overlap_diff.size > 0:
        _, noise_sigma = estimate_robust_background_and_noise(overlap_diff)
    else:
        noise_sigma = 1.0

    # 4. Significance thresholding
    sig_threshold = threshold_sigma * noise_sigma
    diff_filtered = np.where(diff_abs >= sig_threshold, diff_abs, 0.0).astype(np.float32)

    candidates: List[MovingCandidate] = []
    rejected_dipoles_count = 0
    rejected_saturated_count = 0

    # Candidate extraction via connected component analysis
    # A) Positive transients (Epoch 1)
    pos_mask = (diff_signed >= sig_threshold) & valid_overlap
    labeled_pos, num_pos = label(pos_mask)

    cand_idx = 1
    for label_id in range(1, num_pos + 1):
        cluster_mask = (labeled_pos == label_id)
        area = int(np.sum(cluster_mask))
        if area >= min_area:
            weights = diff_signed * cluster_mask
            total_flux = float(np.sum(weights))
            if total_flux > 0:
                cy, cx = center_of_mass(weights)
                
                # Check saturation mask
                if sat_mask[int(round(cy)), int(round(cx))]:
                    rejected_saturated_count += 1
                    continue

                # Check Dipole filter
                if enable_dipole_filter and is_dipole_artifact(diff_signed, cx, cy, window_radius=3):
                    rejected_dipoles_count += 1
                    continue

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

    # B) Negative transients (Epoch 2)
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

                # Check saturation mask
                if sat_mask[int(round(cy)), int(round(cx))]:
                    rejected_saturated_count += 1
                    continue

                # Check Dipole filter
                if enable_dipole_filter and is_dipole_artifact(diff_signed, cx, cy, window_radius=3):
                    rejected_dipoles_count += 1
                    continue

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
        saturation_mask=sat_mask,
        candidates=candidates,
        rejected_dipoles_count=rejected_dipoles_count,
        rejected_saturated_count=rejected_saturated_count,
    )
