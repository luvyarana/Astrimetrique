import numpy as np
from astrimetrique.core.centroid import fit_centroid_2d_gaussian, calculate_center_of_mass


def test_centroid_2d_gaussian_accuracy():
    # Synthetic Gaussian star at known sub-pixel position
    true_x = 34.68
    true_y = 42.31
    amplitude = 1500.0
    sigma = 1.45
    background = 250.0

    width, height = 80, 80
    y_grid, x_grid = np.indices((height, width))
    star = background + amplitude * np.exp(
        -0.5 * (((x_grid - true_x) / sigma) ** 2 + ((y_grid - true_y) / sigma) ** 2)
    )

    # Add realistic Gaussian noise
    rng = np.random.default_rng(42)
    noisy_star = (star + rng.normal(0.0, 5.0, size=(height, width))).astype(np.float32)

    # Click slightly off-center (e.g. at pixel (36, 40))
    res = fit_centroid_2d_gaussian(noisy_star, click_x=36.0, click_y=40.0, box_size=15)

    assert res.success is True
    # Verify sub-pixel accuracy within 0.05 pixels
    assert abs(res.x - true_x) < 0.05
    assert abs(res.y - true_y) < 0.05
    assert abs(res.background - background) < 15.0
    assert res.fwhm > 2.5
    assert res.snr > 20.0


def test_centroid_center_of_mass_fallback():
    # Very small cutout where Gaussian curve fit might struggle
    cutout = np.array([
        [10, 10, 10, 10, 10],
        [10, 50, 80, 50, 10],
        [10, 80, 200, 80, 10],
        [10, 50, 80, 50, 10],
        [10, 10, 10, 10, 10],
    ], dtype=np.float32)

    res = calculate_center_of_mass(cutout, xmin=100, ymin=200)
    assert res.success is True
    assert abs(res.x - 102.0) < 0.2
    assert abs(res.y - 202.0) < 0.2
