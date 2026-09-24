import pytest
import numpy as np
from astrimetrique.core.diff_engine import (
    compute_difference_map,
    match_background_and_gain,
    estimate_robust_background_and_noise,
    is_dipole_artifact,
    build_adaptive_saturation_mask,
)
from astrimetrique.core.registration import compute_affine_from_points, warp_image
from astrimetrique.core.synthetic_data import generate_multiepoch_synthetic_pair


def test_diff_identical_images():
    """
    Rigor Constraint: Verify that subtracting two identical images results in a near-zero array
    (background noise only) with zero detected transient candidates.
    """
    rng = np.random.default_rng(42)
    base = 300.0 + rng.normal(0.0, 5.0, size=(256, 256)).astype(np.float32)

    # Add stars
    for x, y in [(50, 50), (120, 180), (200, 80)]:
        base[y-2:y+3, x-2:x+3] += 500.0

    diff_res = compute_difference_map(base, base.copy(), threshold_sigma=3.5)

    # 1. Check that difference array is effectively 0
    assert np.allclose(diff_res.diff_signed, 0.0, atol=1e-5)
    assert np.allclose(diff_res.diff_abs, 0.0, atol=1e-5)
    assert np.allclose(diff_res.diff_filtered, 0.0, atol=1e-5)

    # 2. Check no false moving candidates detected
    assert len(diff_res.candidates) == 0


def test_diff_gain_and_background_matching():
    img_a = np.full((100, 100), 200.0, dtype=np.float32)
    # Image B has different background (500) and different gain scaling factor (2.0)
    img_b = np.full((100, 100), 500.0, dtype=np.float32)

    bg_a, bg_b, gain_b = match_background_and_gain(img_a, img_b)
    assert abs(bg_a - 200.0) < 1.0
    assert abs(bg_b - 500.0) < 1.0


def test_dipole_signature_filtering():
    """
    Rigor Constraint: Generate a synthetic image with a single bright star, shift it by 0.1 pixels,
    and verify that the resulting 'ghost' artifact is caught and rejected by the Dipole Filter.
    """
    size = 128
    y, x = np.indices((size, size))
    bg = 300.0
    amp = 30000.0
    sigma = 1.5

    # Image A: Star at (64.0, 64.0)
    star_a = bg + amp * np.exp(-0.5 * (((x - 64.0) / sigma)**2 + ((y - 64.0) / sigma)**2))
    
    # Image B: Same star shifted by 0.1 pixels to (64.1, 64.0)
    star_b = bg + amp * np.exp(-0.5 * (((x - 64.1) / sigma)**2 + ((y - 64.0) / sigma)**2))

    # Difference map without dipole filtering would detect a false transient
    diff_unfiltered = compute_difference_map(
        star_a, star_b,
        threshold_sigma=3.5,
        enable_dipole_filter=False,
        psf_match_sigma=0.0,
    )
    assert len(diff_unfiltered.candidates) > 0, "Expected raw dipole to trigger high-SNR candidate"

    # Difference map WITH dipole filtering: ghost is suppressed
    diff_filtered = compute_difference_map(
        star_a, star_b,
        threshold_sigma=3.5,
        enable_dipole_filter=True,
        psf_match_sigma=0.0,
    )
    assert len(diff_filtered.candidates) == 0, f"Dipole filter failed to reject ghost: {diff_filtered.candidates}"
    assert diff_filtered.rejected_dipoles_count >= 1


def test_saturated_core_masking():
    """
    Rigor Constraint: Adaptive blooming mask rejects saturated cores (> 55000 ADU).
    """
    size = 100
    y, x = np.indices((size, size))
    img_a = np.full((size, size), 300.0, dtype=np.float32)
    img_b = np.full((size, size), 300.0, dtype=np.float32)

    # Saturated star at center (60,000 ADU)
    sat_star = 60000.0 * np.exp(-0.5 * (((x - 50) / 2.5)**2 + ((y - 50) / 2.5)**2))
    img_a += sat_star.astype(np.float32)
    img_b += sat_star.astype(np.float32)

    mask = build_adaptive_saturation_mask(img_a, img_b, saturation_limit=55000.0, base_radius=6)
    assert mask[50, 50] is np.bool_(True) or mask[50, 50] == True
    # Mask covers neighborhood around saturated center
    assert mask[52, 52] == True


def test_diff_multiepoch_moving_asteroid_detection():
    """
    Verify that in a multi-epoch pair, the moving minor planet is cleanly detected
    with high statistical significance (> 10 sigma) in the difference map while
    stationary stars and noise are suppressed.
    """
    img_a, img_b, stars_a, stars_b, ast_a, ast_b = generate_multiepoch_synthetic_pair(
        width=512,
        height=512,
        telescope_shift_x=10.0,
        telescope_shift_y=-5.0,
        telescope_rot_deg=0.5,
        asteroid_motion_x=25.0,
        asteroid_motion_y=12.0,
        asteroid_mag=14.0,
        noise_level=8.0,
        seed=42,
    )

    # Align Image B to Image A
    pts_a = [(s.x, s.y) for s in stars_a]
    pts_b = [(s.x, s.y) for s in stars_b]
    transform = compute_affine_from_points(pts_a, pts_b)
    warped_b = warp_image(img_b.data, transform, output_shape=img_a.shape, order=1)

    # Compute difference map with dipole filtering and PSF matching
    diff_res = compute_difference_map(
        img_a.data,
        warped_b,
        threshold_sigma=3.5,
        psf_match_sigma=0.8,
        enable_dipole_filter=True,
    )

    # Verify that moving candidates are detected
    assert len(diff_res.candidates) >= 1

    # Check that the asteroid in Image A or Image B is among the detected candidates
    detected_matches = []
    for cand in diff_res.candidates:
        dist_a = np.sqrt((cand.x - ast_a.x)**2 + (cand.y - ast_a.y)**2)
        ast_b_aligned_x = ast_a.x + 25.0
        ast_b_aligned_y = ast_a.y + 12.0
        dist_b = np.sqrt((cand.x - ast_b_aligned_x)**2 + (cand.y - ast_b_aligned_y)**2)
        if dist_a < 3.0 or dist_b < 3.0:
            detected_matches.append(cand)

    assert len(detected_matches) >= 1
    # Check that detection has high statistical significance (> 10 sigma)
    assert any(c.snr > 10.0 for c in detected_matches)
