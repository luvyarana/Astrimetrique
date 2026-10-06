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
    aspect_ratio: float = 1.0
    compactness: float = 1.0
    is_psf_wing_artifact: bool = False
    local_gradient: float = 0.0
    scale: str = "High (20σ)"  # 'High (20σ)', 'Mid (8σ)', 'Low (3σ)'
    confidence_score: float = 1.0  # 0.0 to 1.0 (0% to 100%)
    is_priority: bool = False


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
    rejected_morphology_count: int = 0
    rejected_proximity_count: int = 0
    rejected_gradient_count: int = 0
    rejected_low_confidence_count: int = 0


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
    noise_sigma: float = 1.0,
    window_radius: int = 2,
    min_dipole_ratio: float = 0.30,
) -> bool:
    """
    Gate A: Topological Signature Analysis (Dipole Filter)
    Identifies if a candidate is a symmetric dipole artifact (positive peak adjacent to negative trough)
    caused by sub-pixel misalignment of a stationary star.
    
    A dipole artifact satisfies in immediate neighborhood:
    1. Both strong positive and negative flux exist immediately adjacent to the peak.
    2. |P + N| < |P - N| (opposite polarity with comparable magnitude).
    3. min(|P|, |N|) / max(|P|, |N|) >= min_dipole_ratio.
    4. Opposite trough is significant: |N| >= max(2.0 * noise_sigma, 0.30 * P).
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
            if ratio >= min_dipole_ratio and abs(neg_min) >= max(2.0 * noise_sigma, 0.30 * pos_max):
                return True

    return False


def evaluate_blob_morphology(
    cluster_mask: np.ndarray,
    max_aspect_ratio: float = 2.0,
    min_compactness: float = 0.6,
) -> Tuple[bool, float, float]:
    """
    Gate B: Geometric Morphology (The Shape Filter)
    For every detected blob, calculate its spatial properties using its pixel mask.
    
    Aspect Ratio:
        BoundingBox_max / BoundingBox_min
        Discard if Ratio > 2.0 (kills linear streaks/satellite trails/cosmic rays).
        
    Circularity / Compactness:
        Area A = total pixels in blob
        MinEnclosingCircleArea = pi * (max(W, H) / 2)^2
        Compactness = Area / MinEnclosingCircleArea
        Discard if Compactness < 0.6 (kills irregular blobs and PSF wing fragments).
        
    Returns:
        (is_valid, aspect_ratio, compactness)
    """
    ys, xs = np.where(cluster_mask)
    if len(ys) == 0:
        return False, 1.0, 0.0

    min_y, max_y = int(np.min(ys)), int(np.max(ys))
    min_x, max_x = int(np.min(xs)), int(np.max(xs))

    w = max_x - min_x + 1
    h = max_y - min_y + 1
    area = len(ys)

    ratio = max(w, h) / max(1, min(w, h))

    radius = max(w, h) / 2.0
    if radius <= 0.5:
        compactness = 1.0
    else:
        circle_area = np.pi * (radius ** 2)
        compactness = float(area / circle_area)

    is_valid = (ratio <= max_aspect_ratio) and (compactness >= min_compactness)
    return is_valid, float(ratio), float(compactness)


def check_proximity_tether(
    image_ref: np.ndarray,
    cx: float,
    cy: float,
    cand_peak: float,
    bg_level: float = 0.0,
    search_radius: int = 15,
    max_intensity_ratio: float = 5.0,
) -> Tuple[bool, float]:
    """
    Gate C: Spatial Context (The 'Tether' Filter)
    Prevents PSF wing leaks from bright stars from being flagged as independent asteroids.
    
    Searches image_ref (original frame) within circular radius R=15 px.
    Compares net peak intensity above background:
    If I_star_net > 5.0 * I_candidate, candidate is flagged as a PSF Wing Residual.
    
    Returns:
        (is_artifact, max_nearby_net_intensity)
    """
    h, w = image_ref.shape
    ix = int(round(cx))
    iy = int(round(cy))

    x0 = max(0, ix - search_radius)
    x1 = min(w, ix + search_radius + 1)
    y0 = max(0, iy - search_radius)
    y1 = min(h, iy + search_radius + 1)

    y_sub, x_sub = np.ogrid[y0:y1, x0:x1]
    dist_sq = (x_sub - cx) ** 2 + (y_sub - cy) ** 2
    circle_mask = dist_sq <= (search_radius ** 2)

    sub_img = image_ref[y0:y1, x0:x1]
    valid_pixels = sub_img[circle_mask]

    if valid_pixels.size == 0:
        return False, 0.0

    i_max = float(np.max(valid_pixels))
    i_net_max = max(0.0, i_max - bg_level)

    # If I_star_net > 5.0 * cand_peak, candidate is a wing residual of the nearby star
    is_artifact = i_net_max > (max_intensity_ratio * cand_peak)
    return is_artifact, i_net_max


def compute_local_gradient(
    image_ref: np.ndarray,
    cx: float,
    cy: float,
) -> float:
    """
    Gate C: Spatial Context (The 'Slope' Filter)
    Computes central difference approximation for the image gradient ||grad I||
    in the original reference frame at the candidate's centroid:
    
    grad_I = sqrt( ((I_{x+1, y} - I_{x-1, y}) / 2)^2 + ((I_{x, y+1} - I_{x, y-1}) / 2)^2 )
    """
    h, w = image_ref.shape
    ix = int(round(cx))
    iy = int(round(cy))

    # Clamp indices to ensure valid 1-pixel boundary for central differences
    ix = max(1, min(w - 2, ix))
    iy = max(1, min(h - 2, iy))

    gx = float(image_ref[iy, ix + 1] - image_ref[iy, ix - 1]) / 2.0
    gy = float(image_ref[iy + 1, ix] - image_ref[iy - 1, ix]) / 2.0

    grad_mag = float(np.sqrt(gx ** 2 + gy ** 2))
    return grad_mag


def calculate_candidate_confidence(
    scale_name: str,
    snr: float,
    area: int,
    aspect_ratio: float,
    compactness: float,
    cand_peak: float,
    max_nearby: float,
    grad_mag: float,
    effective_grad_max: float,
) -> float:
    """
    Calculate confidence score (0.0 to 1.0) based on scale tier and quality gates.
    Arbitration Logic:
    - High-Pass detections: Automatically 1.0 (100% confidence).
    - Mid-Pass detections: High base confidence (0.85-1.00).
    - Low-Pass detections: Evaluated under strict Confidence Sieve requiring
      sufficient SNR, spatial area, round morphology, isolation, and flat background.
    """
    if "High" in scale_name:
        return 1.0

    if "Mid" in scale_name:
        score = 0.85
        if aspect_ratio <= 1.3 and compactness >= 0.75:
            score += 0.05
        if max_nearby <= 1.5 * cand_peak:
            score += 0.05
        if grad_mag <= 0.30 * effective_grad_max:
            score += 0.05
        return float(np.clip(score, 0.0, 1.0))

    # Low-Pass: Strict quality gate evaluation
    score = 0.0

    # 1. Spatial Extension Gate (real astronomical PSF footprint)
    if area >= 8:
        score += 0.30
    elif area >= 6:
        score += 0.15

    # 2. Significance Gate
    if snr >= 5.0:
        score += 0.25
    elif snr >= 4.0:
        score += 0.15
    elif snr >= 3.5:
        score += 0.05

    # 3. Geometric Symmetry Gate
    if aspect_ratio <= 1.3 and compactness >= 0.75:
        score += 0.15
    elif aspect_ratio <= 1.6 and compactness >= 0.65:
        score += 0.10

    # 4. Proximity Isolation Gate
    if max_nearby <= 1.5 * cand_peak:
        score += 0.15
    elif max_nearby <= 3.0 * cand_peak:
        score += 0.05

    # 5. Background Slope Gate
    if grad_mag <= 0.30 * effective_grad_max:
        score += 0.15
    elif grad_mag <= 0.60 * effective_grad_max:
        score += 0.05

    return float(np.clip(score, 0.0, 1.0))


def compute_difference_map(
    image_a: np.ndarray,
    image_b_warped: np.ndarray,
    threshold_sigma: float = 3.0,
    high_pass_sigma: float = 20.0,
    mid_pass_sigma: float = 8.0,
    low_pass_sigma: Optional[float] = None,
    min_area: int = 3,
    psf_match_sigma: float = 0.8,
    saturation_limit: float = 55000.0,
    enable_dipole_filter: bool = True,
    enable_saturation_mask: bool = True,
    enable_morphology_filter: bool = True,
    enable_proximity_filter: bool = True,
    enable_gradient_filter: bool = True,
    max_aspect_ratio: float = 2.0,
    min_compactness: float = 0.6,
    proximity_radius: int = 15,
    proximity_intensity_ratio: float = 5.0,
    max_gradient: Optional[float] = None,
    min_low_pass_confidence: float = 0.75,
) -> DiffResult:
    """
    Multi-Scale Discovery Engine & Morphological Sieve:
    1. Background & Gain Matching
    2. Adaptive Saturated Core Masking
    3. PSF-Matched Gaussian Convolution
    4. Three-Tier Multi-Pass Sieve (High-Pass 20σ, Mid-Pass 8σ, Low-Pass 3σ)
    5. Gate A: Topological Signature (Dipole Filter)
    6. Gate B: Geometric Morphology (Aspect Ratio <= 2.0, Circularity >= 0.6)
    7. Gate C: Spatial Context (Tether R=15 px & Local Slope Gradient)
    8. Confidence Scoring & High-Pass Priority Promotion
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

    # Effective threshold for low-pass
    effective_low_sigma = low_pass_sigma if low_pass_sigma is not None else threshold_sigma

    # Visual significance map based on low pass
    sig_threshold = effective_low_sigma * noise_sigma
    diff_filtered = np.where(diff_abs >= sig_threshold, diff_abs, 0.0).astype(np.float32)

    candidates: List[MovingCandidate] = []
    rejected_dipoles_count = 0
    rejected_saturated_count = 0
    rejected_morphology_count = 0
    rejected_proximity_count = 0
    rejected_gradient_count = 0
    rejected_low_confidence_count = 0

    # 4. Multi-Pass Detection Loop (The Sieve)
    tiers = [
        ("High (20σ)", high_pass_sigma, True),
        ("Mid (8σ)", mid_pass_sigma, False),
        ("Low (3σ)", effective_low_sigma, False),
    ]

    cand_idx = 1

    # Loop over both polarities: Positive (Epoch 1) and Negative (Epoch 2)
    polarities = [
        ("Epoch 1 (Positive)", img_a, bg_a, True),
        ("Epoch 2 (Negative)", img_b, bg_b, False),
    ]

    for pol_name, ref_img, bg_ref, is_pos in polarities:
        accepted_coords: List[Tuple[float, float]] = []

        for scale_name, tier_sigma, is_priority in tiers:
            tier_thresh = tier_sigma * noise_sigma
            if is_pos:
                tier_mask = (diff_signed >= tier_thresh) & valid_overlap
            else:
                tier_mask = (diff_signed <= -tier_thresh) & valid_overlap

            labeled, num_features = label(tier_mask)
            if num_features == 0:
                continue

            for label_id in range(1, num_features + 1):
                cluster_mask = (labeled == label_id)
                area = int(np.sum(cluster_mask))
                if area < min_area:
                    continue

                if is_pos:
                    weights = diff_signed * cluster_mask
                else:
                    weights = np.abs(diff_signed) * cluster_mask

                total_flux = float(np.sum(weights))
                if total_flux <= 0:
                    continue

                cy, cx = center_of_mass(weights)
                
                # Check deduplication against higher passes (within 4.0 pixels)
                if any(np.hypot(cx - ax, cy - ay) <= 4.0 for (ax, ay) in accepted_coords):
                    continue

                # Check saturation mask
                if sat_mask[int(round(cy)), int(round(cx))]:
                    rejected_saturated_count += 1
                    continue

                # Gate A: Topological Signature (Dipole Filter)
                py, px = np.unravel_index(np.argmax(weights), weights.shape)
                is_dipole = is_dipole_artifact(diff_signed, float(px), float(py), noise_sigma=noise_sigma, window_radius=2) or is_dipole_artifact(diff_signed, cx, cy, noise_sigma=noise_sigma, window_radius=2)
                if enable_dipole_filter and is_dipole:
                    rejected_dipoles_count += 1
                    continue

                # Gate B: Geometric Morphology (Shape Filter evaluated at current tier threshold)
                is_valid_shape, aspect_ratio, compactness = evaluate_blob_morphology(
                    cluster_mask,
                    max_aspect_ratio=max_aspect_ratio,
                    min_compactness=min_compactness,
                )
                if enable_morphology_filter and not is_valid_shape:
                    rejected_morphology_count += 1
                    continue

                peak = float(np.max(weights[cluster_mask]))
                snr = float(peak / noise_sigma)

                # Gate C: Spatial Context (Tether Filter in reference image)
                is_wing_artifact, max_nearby = check_proximity_tether(
                    ref_img,
                    cx,
                    cy,
                    cand_peak=peak,
                    bg_level=bg_ref,
                    search_radius=proximity_radius,
                    max_intensity_ratio=proximity_intensity_ratio,
                )
                if enable_proximity_filter and is_wing_artifact:
                    rejected_proximity_count += 1
                    continue

                # Gate C: Spatial Context (Slope Filter in reference image)
                grad_mag = compute_local_gradient(ref_img, cx, cy)
                effective_grad_max = max_gradient if max_gradient is not None else max(50.0, 0.5 * peak)
                if enable_gradient_filter and (grad_mag > effective_grad_max):
                    rejected_gradient_count += 1
                    continue

                # Confidence Scoring & Arbitration
                confidence_score = calculate_candidate_confidence(
                    scale_name=scale_name,
                    snr=snr,
                    area=area,
                    aspect_ratio=aspect_ratio,
                    compactness=compactness,
                    cand_peak=peak,
                    max_nearby=max_nearby,
                    grad_mag=grad_mag,
                    effective_grad_max=effective_grad_max,
                )

                if (scale_name == "Low (3σ)") and (confidence_score < min_low_pass_confidence):
                    rejected_low_confidence_count += 1
                    continue

                # Priority Target status for High-Pass detections
                priority_flag = is_priority or (snr >= high_pass_sigma)
                tag = " (Priority)" if priority_flag else ""

                candidate = MovingCandidate(
                    candidate_id=f"CAND-{cand_idx:02d}{tag}",
                    x=float(cx),
                    y=float(cy),
                    snr=snr,
                    peak_diff=peak,
                    total_flux_diff=total_flux,
                    pixel_area=area,
                    polarity=pol_name,
                    aspect_ratio=aspect_ratio,
                    compactness=compactness,
                    is_psf_wing_artifact=False,
                    local_gradient=grad_mag,
                    scale=scale_name,
                    confidence_score=confidence_score,
                    is_priority=priority_flag,
                )
                candidates.append(candidate)
                accepted_coords.append((cx, cy))
                cand_idx += 1

    # Sort candidates: Priority / High confidence first, then SNR descending
    candidates.sort(key=lambda c: (c.is_priority, c.confidence_score, c.snr), reverse=True)

    return DiffResult(
        diff_signed=diff_signed,
        diff_abs=diff_abs,
        diff_filtered=diff_filtered,
        bg_a=bg_a,
        bg_b=bg_b,
        gain_scale_b=gain_b,
        noise_sigma=noise_sigma,
        threshold_sigma=effective_low_sigma,
        saturation_mask=sat_mask,
        candidates=candidates,
        rejected_dipoles_count=rejected_dipoles_count,
        rejected_saturated_count=rejected_saturated_count,
        rejected_morphology_count=rejected_morphology_count,
        rejected_proximity_count=rejected_proximity_count,
        rejected_gradient_count=rejected_gradient_count,
        rejected_low_confidence_count=rejected_low_confidence_count,
    )


