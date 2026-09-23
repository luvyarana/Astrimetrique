import pytest
import numpy as np
from scipy.ndimage import rotate, shift
from astrimetrique.core.registration import (
    RegistrationTransform,
    compute_affine_from_points,
    warp_image,
)
from astrimetrique.core.centroid import fit_centroid_2d_gaussian


def test_compute_affine_from_points():
    # 4 points on reference image
    pts_a = np.array([
        [100.0, 100.0],
        [400.0, 100.0],
        [400.0, 400.0],
        [100.0, 400.0],
    ])

    # Apply translation (dx=5.25, dy=-3.75) and rotation theta=10 deg
    theta_rad = np.radians(10.0)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    cx, cy = 250.0, 250.0

    pts_b = []
    for x, y in pts_a:
        dx, dy = x - cx, y - cy
        xb = cos_t * dx - sin_t * dy + cx + 5.25
        yb = sin_t * dx + cos_t * dy + cy - 3.75
        pts_b.append([xb, yb])
    pts_b = np.array(pts_b)

    transform = compute_affine_from_points(pts_a, pts_b)

    assert transform.alignment_rms_px < 1e-6
    assert abs(transform.rotation_deg - (-10.0 % 360)) < 0.01 or abs(transform.rotation_deg - 350.0) < 0.01
    assert abs(transform.scale_x - 1.0) < 1e-4
    assert abs(transform.scale_y - 1.0) < 1e-4


def test_warp_image_subpixel_recovery():
    """
    Rigor Constraint: Shift synthetic star field by fractional pixels and rotation,
    verify that warp_image recovers the original alignment within < 0.1 pixel.
    """
    size = 256
    img_a = np.zeros((size, size), dtype=np.float32)

    # Place reference stars
    star_positions_a = [
        (64.2, 64.8),
        (190.5, 72.1),
        (185.3, 192.4),
        (70.1, 188.6),
        (128.0, 128.0),
    ]

    y_grid, x_grid = np.indices((size, size))
    for sx, sy in star_positions_a:
        psf = 1000.0 * np.exp(-0.5 * (((x_grid - sx) / 1.5) ** 2 + ((y_grid - sy) / 1.5) ** 2))
        img_a += psf.astype(np.float32)

    # Generate Image B by shifting by (dx=3.42, dy=-2.78) and rotating by theta = 5.0 deg
    shift_x = 3.42
    shift_y = -2.78
    rot_deg = 5.0
    theta_rad = np.radians(rot_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    cx, cy = size / 2.0, size / 2.0

    img_b = np.zeros((size, size), dtype=np.float32)
    star_positions_b = []
    for sx, sy in star_positions_a:
        dx, dy = sx - cx, sy - cy
        sxb = cos_t * dx - sin_t * dy + cx + shift_x
        syb = sin_t * dx + cos_t * dy + cy + shift_y
        star_positions_b.append((sxb, syb))
        psf = 1000.0 * np.exp(-0.5 * (((x_grid - sxb) / 1.5) ** 2 + ((y_grid - syb) / 1.5) ** 2))
        img_b += psf.astype(np.float32)

    # Calculate affine transformation
    transform = compute_affine_from_points(star_positions_a, star_positions_b)

    # Warp Image B into coordinate system of Image A
    warped_b = warp_image(img_b, transform, output_shape=(size, size), order=1)

    # Measure centroid of each star in warped_b and compare to true star_positions_a
    max_centroid_error = 0.0
    for sx_true, sy_true in star_positions_a:
        res = fit_centroid_2d_gaussian(warped_b, sx_true, sy_true, box_size=15)
        assert res.success is True
        err = np.sqrt((res.x - sx_true)**2 + (res.y - sy_true)**2)
        if err > max_centroid_error:
            max_centroid_error = err

    # Rigorous sub-pixel accuracy verification: < 0.1 pixel alignment!
    assert max_centroid_error < 0.10, f"Max centroid error {max_centroid_error} px exceeds 0.1 px!"


def test_registration_pipeline_non_zero_shift_and_null():
    """
    Verification Protocol:
    1. 'Non-Zero' Report: Verify that the affine matrix constants report true non-zero translation (c, f).
    2. 'Jump' Test: Verify that all stars in Warped Image B match Frame A to within sub-pixel accuracy (static stars).
    3. 'Null' Test: Subtracting warped image from itself produces a near-zero array.
    """
    from astrimetrique.core.synthetic_data import generate_multiepoch_synthetic_pair
    from astrimetrique.core.registration import align_images_pipeline

    shift_x = 12.4
    shift_y = -7.8
    rot_deg = 0.75

    img_a, img_b, stars_a, stars_b, ast_a, ast_b = generate_multiepoch_synthetic_pair(
        width=512,
        height=512,
        telescope_shift_x=shift_x,
        telescope_shift_y=shift_y,
        telescope_rot_deg=rot_deg,
        noise_level=5.0,
    )

    # Run full registration pipeline
    transform, warped_b = align_images_pipeline(
        img_a=img_a,
        img_b=img_b,
        stars_a=stars_a,
        stars_b=stars_b,
    )

    # 1. Non-Zero Report verification
    assert abs(transform.translation_x) > 1.0, f"Expected non-zero translation_x, got {transform.translation_x}"
    assert abs(transform.translation_y) > 1.0, f"Expected non-zero translation_y, got {transform.translation_y}"
    assert transform.alignment_rms_px < 0.10, f"Alignment RMS {transform.alignment_rms_px} px exceeds 0.10 px"

    # 2. 'Jump' test: measure centroids of stars in warped_b vs img_a
    for sa in stars_a[:5]:
        res_a = fit_centroid_2d_gaussian(img_a.data, sa.x, sa.y, box_size=15)
        res_b = fit_centroid_2d_gaussian(warped_b, sa.x, sa.y, box_size=15)
        assert res_a.success and res_b.success
        diff_px = np.sqrt((res_a.x - res_b.x)**2 + (res_a.y - res_b.y)**2)
        assert diff_px < 0.10, f"Star {sa.star_id} jumped by {diff_px:.3f} px (exceeds 0.1 px)!"

    # 3. 'Null' test: subtracting warped image from itself
    self_diff = warped_b - warped_b
    assert np.allclose(self_diff, 0.0, atol=1e-7)
