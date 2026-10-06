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

    # Difference map without dipole filtering detects false transient
    diff_unfiltered = compute_difference_map(
        star_a, star_b,
        threshold_sigma=3.5,
        enable_dipole_filter=False,
        enable_proximity_filter=False,
        enable_gradient_filter=False,
        psf_match_sigma=0.0,
    )
    assert len(diff_unfiltered.candidates) > 0, "Expected raw dipole to trigger candidate"

    # Difference map WITH dipole filtering: ghost is suppressed
    diff_filtered = compute_difference_map(
        star_a, star_b,
        threshold_sigma=3.5,
        enable_dipole_filter=True,
        enable_proximity_filter=False,
        enable_gradient_filter=False,
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


def test_vesta_high_snr_point_source_preserved():
    """
    The Vesta Test: Verify that a high-S/N, round, isolated point source (like Asteroid 4 Vesta)
    is detected in the High-Pass (20σ), promoted to Priority status, and PRESERVED without
    being rejected by morphology, proximity, or gradient filters.
    """
    size = 256
    y, x = np.indices((size, size))
    bg = 300.0
    rng = np.random.default_rng(123)
    noise = rng.normal(0.0, 4.0, size=(size, size)).astype(np.float32)

    # Frame A: Background + Noise + Bright round Asteroid (Vesta, amplitude 35,000 ADU, sigma 1.8)
    vesta_x, vesta_y = 128.0, 128.0
    vesta_amp = 35000.0
    vesta_profile = vesta_amp * np.exp(-0.5 * (((x - vesta_x) / 1.8)**2 + ((y - vesta_y) / 1.8)**2))
    img_a = (bg + noise + vesta_profile).astype(np.float32)

    # Frame B: Background + Noise (Vesta moved elsewhere, isolated location)
    noise_b = rng.normal(0.0, 4.0, size=(size, size)).astype(np.float32)
    img_b = (bg + noise_b).astype(np.float32)

    diff_res = compute_difference_map(
        img_a,
        img_b,
        threshold_sigma=3.0,
        high_pass_sigma=20.0,
        mid_pass_sigma=8.0,
        enable_morphology_filter=True,
        enable_proximity_filter=True,
        enable_gradient_filter=True,
    )

    # Vesta must be preserved
    assert len(diff_res.candidates) == 1, f"Expected 1 candidate (Vesta), got {len(diff_res.candidates)}"
    vesta_cand = diff_res.candidates[0]

    # Verify high SNR, High-Pass scale tier, priority status, round shape, and correct coordinates
    assert vesta_cand.snr > 500.0
    assert "High" in vesta_cand.scale
    assert vesta_cand.is_priority == True
    assert vesta_cand.confidence_score == 1.0
    assert abs(vesta_cand.x - vesta_x) < 0.5
    assert abs(vesta_cand.y - vesta_y) < 0.5
    assert vesta_cand.aspect_ratio <= 2.0
    assert vesta_cand.compactness >= 0.6


def test_satellite_linear_streak_discarded():
    """
    The Trail Test: Verify that a high-S/N, linear streak (satellite trail / cosmic ray)
    is DISCARDED via the Morphological Aspect Ratio check.
    """
    size = 256
    y, x = np.indices((size, size))
    bg = 300.0
    rng = np.random.default_rng(456)
    noise_a = rng.normal(0.0, 4.0, size=(size, size)).astype(np.float32)
    noise_b = rng.normal(0.0, 4.0, size=(size, size)).astype(np.float32)

    img_a = (bg + noise_a).astype(np.float32)
    img_b = (bg + noise_b).astype(np.float32)

    # Add a linear streak in Frame A from x=100 to x=135 at y=120 (length 35 px, height 2 px, amp 15,000 ADU)
    img_a[119:122, 100:136] += 15000.0

    diff_res = compute_difference_map(
        img_a,
        img_b,
        threshold_sigma=3.5,
        enable_morphology_filter=True,
        max_aspect_ratio=2.0,
    )

    # Streak must be discarded
    assert len(diff_res.candidates) == 0, f"Expected satellite streak to be discarded, got {diff_res.candidates}"
    assert diff_res.rejected_morphology_count >= 1


def test_ghost_psf_wing_residual_discarded():
    """
    The Ghost Test: Verify that a high-S/N point source artifact located 5 pixels away
    from a brighter star is DISCARDED via Proximity / Gradient filtering.
    """
    size = 256
    y, x = np.indices((size, size))
    bg = 300.0
    rng = np.random.default_rng(789)
    noise_a = rng.normal(0.0, 3.0, size=(size, size)).astype(np.float32)
    noise_b = rng.normal(0.0, 3.0, size=(size, size)).astype(np.float32)

    # Bright star at (100, 100) with peak 45,000 ADU
    star_x, star_y = 100.0, 100.0
    star_amp = 45000.0
    star_sigma = 2.0
    star_profile = star_amp * np.exp(-0.5 * (((x - star_x) / star_sigma)**2 + ((y - star_y) / star_sigma)**2))

    # Image A has the bright star
    img_a = (bg + noise_a + star_profile).astype(np.float32)

    # Image B has the star
    img_b = (bg + noise_b + star_profile).astype(np.float32)

    # Inject a ghost residual artifact in Image A at (105, 100) on the star's PSF wing (5 px away, peak 1,500 ADU)
    ghost_x, ghost_y = 105.0, 100.0
    ghost_amp = 1500.0
    ghost_profile = ghost_amp * np.exp(-0.5 * (((x - ghost_x) / 1.2)**2 + ((y - ghost_y) / 1.2)**2))
    img_a += ghost_profile.astype(np.float32)

    # Without proximity/gradient filtering, the ghost triggers a candidate
    diff_unfiltered = compute_difference_map(
        img_a,
        img_b,
        threshold_sigma=3.0,
        enable_proximity_filter=False,
        enable_gradient_filter=False,
    )
    assert len(diff_unfiltered.candidates) >= 1, "Expected raw ghost to trigger candidate without filters"

    # With proximity and gradient filtering, ghost is suppressed
    diff_filtered = compute_difference_map(
        img_a,
        img_b,
        threshold_sigma=3.0,
        enable_proximity_filter=True,
        enable_gradient_filter=True,
        proximity_radius=15,
        proximity_intensity_ratio=5.0,
    )

    assert len(diff_filtered.candidates) == 0, f"Ghost artifact was not discarded: {diff_filtered.candidates}"
    assert (diff_filtered.rejected_proximity_count >= 1 or diff_filtered.rejected_gradient_count >= 1)


def test_multi_scale_tier_assignment():
    """
    Verify that candidates of varying brightness are properly partitioned into
    High (20σ), Mid (8σ), and Low (3σ) scale tiers with appropriate confidence scores.
    """
    size = 256
    y, x = np.indices((size, size))
    bg = 300.0
    rng = np.random.default_rng(999)
    noise_a = rng.normal(0.0, 2.0, size=(size, size)).astype(np.float32)
    noise_b = rng.normal(0.0, 2.0, size=(size, size)).astype(np.float32)

    img_a = (bg + noise_a).astype(np.float32)
    img_b = (bg + noise_b).astype(np.float32)

    # Candidate 1: Very bright asteroid at (50, 50), amp = 500 ADU (SNR ~ 250σ -> High tier)
    img_a += (500.0 * np.exp(-0.5 * (((x - 50.0) / 1.5)**2 + ((y - 50.0) / 1.5)**2))).astype(np.float32)

    # Candidate 2: Mid-brightness asteroid at (150, 150), amp = 30 ADU (SNR ~ 15σ -> Mid tier)
    img_a += (30.0 * np.exp(-0.5 * (((x - 150.0) / 1.5)**2 + ((y - 150.0) / 1.5)**2))).astype(np.float32)

    # Candidate 3: Faint asteroid at (200, 80), amp = 10 ADU (SNR ~ 5σ -> Low tier)
    img_a += (10.0 * np.exp(-0.5 * (((x - 200.0) / 1.5)**2 + ((y - 80.0) / 1.5)**2))).astype(np.float32)

    diff_res = compute_difference_map(
        img_a,
        img_b,
        threshold_sigma=3.0,
        high_pass_sigma=20.0,
        mid_pass_sigma=8.0,
    )

    assert len(diff_res.candidates) == 3
    cand_high = next(c for c in diff_res.candidates if abs(c.x - 50.0) < 2.0)
    cand_mid = next(c for c in diff_res.candidates if abs(c.x - 150.0) < 2.0)
    cand_low = next(c for c in diff_res.candidates if abs(c.x - 200.0) < 2.0)

    assert "High" in cand_high.scale
    assert cand_high.is_priority == True
    assert cand_high.confidence_score == 1.0

    assert "Mid" in cand_mid.scale
    assert cand_mid.confidence_score >= 0.85

    assert "Low" in cand_low.scale
    assert cand_low.confidence_score >= 0.70


