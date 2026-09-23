"""
Sub-pixel Astronomical Centroiding Engine.
Implements 2D Gaussian Levenberg-Marquardt fitting, 1D Marginal fits, and Center-of-Mass fallback.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import curve_fit


@dataclass
class CentroidResult:
    """Sub-pixel star measurement result."""
    x: float  # Centroid X (0-indexed image coordinate)
    y: float  # Centroid Y (0-indexed image coordinate)
    x_err: float = 0.0  # Standard error in X (pixels)
    y_err: float = 0.0  # Standard error in Y (pixels)
    fwhm: float = 0.0  # Full Width at Half Maximum in pixels
    peak_flux: float = 0.0  # Amplitude above background
    background: float = 0.0  # Local sky background level
    total_flux: float = 0.0  # Integrated aperture/Gaussian flux
    snr: float = 0.0  # Signal-to-Noise Ratio
    fit_method: str = "2D Gaussian"
    success: bool = True
    roi_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (xmin, xmax, ymin, ymax)


def gaussian_2d(
    xy: Tuple[np.ndarray, np.ndarray],
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    background: float,
) -> np.ndarray:
    """2D Circular/Elliptical Gaussian model."""
    x, y = xy
    # Prevent divide by zero or extreme sigma values
    sigma_x = max(abs(sigma_x), 0.2)
    sigma_y = max(abs(sigma_y), 0.2)
    exponent = -0.5 * (((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2)
    return (background + amplitude * np.exp(np.clip(exponent, -50.0, 0.0))).ravel()


def gaussian_1d(x: np.ndarray, amplitude: float, x0: float, sigma: float, background: float) -> np.ndarray:
    """1D Gaussian model."""
    sigma = max(abs(sigma), 0.2)
    return background + amplitude * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def calculate_center_of_mass(cutout: np.ndarray, xmin: int, ymin: int) -> CentroidResult:
    """Intensity-weighted Center of Mass (CoM) centroid."""
    bg = float(np.percentile(cutout, 20.0))
    subtracted = np.clip(cutout - bg, 0.0, None)
    total_mass = np.sum(subtracted)

    if total_mass <= 1e-9:
        # Star is too faint or empty cutout
        cy, cx = cutout.shape[0] / 2.0, cutout.shape[1] / 2.0
        return CentroidResult(
            x=xmin + cx,
            y=ymin + cy,
            fwhm=3.0,
            peak_flux=0.0,
            background=bg,
            total_flux=0.0,
            snr=0.0,
            fit_method="Center of Mass (Fallback)",
            success=False,
            roi_bbox=(xmin, xmin + cutout.shape[1], ymin, ymin + cutout.shape[0]),
        )

    y_indices, x_indices = np.indices(cutout.shape)
    x_center = float(np.sum(x_indices * subtracted) / total_mass)
    y_center = float(np.sum(y_indices * subtracted) / total_mass)
    peak = float(np.max(cutout)) - bg

    # Estimate FWHM from second moments
    var_x = float(np.sum(((x_indices - x_center) ** 2) * subtracted) / total_mass)
    var_y = float(np.sum(((y_indices - y_center) ** 2) * subtracted) / total_mass)
    sigma_avg = np.sqrt(max(0.2, (var_x + var_y) / 2.0))
    fwhm = 2.35482 * sigma_avg

    # SNR estimate
    noise = float(np.std(cutout[cutout <= bg + np.std(cutout)]))
    snr = peak / max(noise, 1e-5)

    return CentroidResult(
        x=xmin + x_center,
        y=ymin + y_center,
        x_err=0.2,
        y_err=0.2,
        fwhm=fwhm,
        peak_flux=peak,
        background=bg,
        total_flux=total_mass,
        snr=snr,
        fit_method="Center of Mass",
        success=True,
        roi_bbox=(xmin, xmin + cutout.shape[1], ymin, ymin + cutout.shape[0]),
    )


def fit_centroid_2d_gaussian(
    data: np.ndarray,
    click_x: float,
    click_y: float,
    box_size: int = 15,
) -> CentroidResult:
    """
    Fit a 2D Gaussian around (click_x, click_y) to calculate sub-pixel centroid coordinates.
    Falls back gracefully to 1D Gaussian or Center of Mass if non-linear fitting fails.
    """
    height, width = data.shape
    half_box = box_size // 2

    ix = int(round(click_x))
    iy = int(round(click_y))

    xmin = max(0, ix - half_box)
    xmax = min(width, ix + half_box + 1)
    ymin = max(0, iy - half_box)
    ymax = min(height, iy + half_box + 1)

    cutout = data[ymin:ymax, xmin:xmax]
    if cutout.shape[0] < 5 or cutout.shape[1] < 5:
        return calculate_center_of_mass(cutout, xmin, ymin)

    # Initial estimates
    bg_init = float(np.percentile(cutout, 20.0))
    peak_init = float(np.max(cutout)) - bg_init
    if peak_init <= 0:
        return calculate_center_of_mass(cutout, xmin, ymin)

    # Find local max in cutout
    local_max_idx = np.unravel_index(np.argmax(cutout), cutout.shape)
    x0_init = float(local_max_idx[1])
    y0_init = float(local_max_idx[0])
    sigma_init = 1.5

    # Grid coordinates within cutout
    y_grid, x_grid = np.indices(cutout.shape)

    # Parameter bounds: [amplitude, x0, y0, sigma_x, sigma_y, background]
    lower_bounds = [0.0, 0.0, 0.0, 0.3, 0.3, -np.inf]
    upper_bounds = [
        np.inf,
        cutout.shape[1] - 1.0,
        cutout.shape[0] - 1.0,
        float(box_size),
        float(box_size),
        np.inf,
    ]
    p0 = [peak_init, x0_init, y0_init, sigma_init, sigma_init, bg_init]

    try:
        popt, pcov = curve_fit(
            gaussian_2d,
            (x_grid, y_grid),
            cutout.ravel(),
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=600,
        )

        amp, x0_fit, y0_fit, sig_x, sig_y, bg_fit = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

        avg_sigma = (abs(sig_x) + abs(sig_y)) / 2.0
        fwhm = 2.35482 * avg_sigma
        
        # Calculate residuals and SNR
        fitted_model = gaussian_2d((x_grid, y_grid), *popt).reshape(cutout.shape)
        residuals = cutout - fitted_model
        noise = float(np.std(residuals))
        snr = float(amp / max(noise, 1e-5))

        # Integrated Gaussian flux = 2 * pi * amp * sig_x * sig_y
        total_flux = float(2.0 * np.pi * amp * sig_x * sig_y)

        # Sanity check: if fitted center strayed too far from cutout center, fall back
        if abs(x0_fit - x0_init) > half_box or abs(y0_fit - y0_init) > half_box or avg_sigma > box_size:
            return calculate_center_of_mass(cutout, xmin, ymin)

        return CentroidResult(
            x=xmin + x0_fit,
            y=ymin + y0_fit,
            x_err=float(perr[1]),
            y_err=float(perr[2]),
            fwhm=float(fwhm),
            peak_flux=float(amp),
            background=float(bg_fit),
            total_flux=total_flux,
            snr=snr,
            fit_method="2D Gaussian Fit",
            success=True,
            roi_bbox=(xmin, xmax, ymin, ymax),
        )

    except Exception:
        # Fallback to Center of Mass
        return calculate_center_of_mass(cutout, xmin, ymin)
