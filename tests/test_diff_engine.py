import pytest
import numpy as np
from astrimetrique.core.diff_engine import (
    compute_difference_map,
    match_background_and_gain,
    estimate_robust_background_and_noise,
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


def test_diff_multiepoch_moving_asteroid_detection():
    """
    Verify that in a multi-epoch pair, the moving minor planet is cleanly detected
    with high statistical significance (> 10 sigma) in the difference map.
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

    # Compute difference map
    diff_res = compute_difference_map(img_a.data, warped_b, threshold_sigma=3.5)

    # Verify that moving candidates are detected
    assert len(diff_res.candidates) >= 1

    # Check that the asteroid in Image A or Image B is among the detected candidates
    detected_matches = []
    for cand in diff_res.candidates:
        dist_a = np.sqrt((cand.x - ast_a.x)**2 + (cand.y - ast_a.y)**2)
        # In aligned frame B, the asteroid was shifted by +25 px in X, +12 px in Y relative to ast_a
        ast_b_aligned_x = ast_a.x + 25.0
        ast_b_aligned_y = ast_a.y + 12.0
        dist_b = np.sqrt((cand.x - ast_b_aligned_x)**2 + (cand.y - ast_b_aligned_y)**2)
        if dist_a < 3.0 or dist_b < 3.0:
            detected_matches.append(cand)

    assert len(detected_matches) >= 1
    # Check that detection has high statistical significance (> 10 sigma)
    assert any(c.snr > 10.0 for c in detected_matches)
