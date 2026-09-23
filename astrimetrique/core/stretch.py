"""
Astronomical Image Stretch and Dynamic Range Compression.
Implements ZScale (Astropy), Linear, Log, Sqrt, Min-Max, and Histogram Equalization.
"""

from enum import Enum
from typing import Tuple
import numpy as np
from astropy.visualization import ZScaleInterval


class StretchMode(str, Enum):
    ZSCALE = "ZScale"
    LINEAR = "Linear"
    LOG = "Logarithmic"
    SQRT = "Square Root"
    MINMAX = "Min-Max"
    HISTEQ = "Hist Equalize"


def calculate_zscale_limits(data: np.ndarray, contrast: float = 0.25, num_samples: int = 1000) -> Tuple[float, float]:
    """
    Calculate astronomical ZScale vmin and vmax limits.
    """
    valid_data = data[np.isfinite(data)]
    if valid_data.size == 0:
        return 0.0, 1.0
    
    interval = ZScaleInterval(contrast=contrast, n_samples=num_samples)
    try:
        vmin, vmax = interval.get_limits(valid_data)
    except Exception:
        # Fallback to robust percentiles if ZScale fitting encounters flat data
        vmin, vmax = float(np.percentile(valid_data, 1.0)), float(np.percentile(valid_data, 99.0))

    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def apply_stretch(
    data: np.ndarray,
    mode: StretchMode = StretchMode.ZSCALE,
    vmin: float | None = None,
    vmax: float | None = None,
    contrast: float = 0.25,
    invert: bool = False,
) -> np.ndarray:
    """
    Stretch high dynamic range astronomical data to uint8 (0-255) for display.
    """
    clean_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Determine limits
    if vmin is None or vmax is None:
        if mode == StretchMode.ZSCALE:
            z_min, z_max = calculate_zscale_limits(clean_data, contrast=contrast)
            vmin = z_min if vmin is None else vmin
            vmax = z_max if vmax is None else vmax
        elif mode == StretchMode.MINMAX:
            vmin = float(np.min(clean_data)) if vmin is None else vmin
            vmax = float(np.max(clean_data)) if vmax is None else vmax
        else:
            # Default robust percentile limits for Log, Sqrt, Linear
            vmin = float(np.percentile(clean_data, 0.5)) if vmin is None else vmin
            vmax = float(np.percentile(clean_data, 99.5)) if vmax is None else vmax

    if vmax <= vmin:
        vmax = vmin + 1.0

    # Clip to [vmin, vmax] and normalize to [0, 1]
    clipped = np.clip(clean_data, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin)

    if mode == StretchMode.LOG:
        # log(1 + a * x) / log(1 + a)
        a = 1000.0
        stretched = np.log1p(a * norm) / np.log1p(a)
    elif mode == StretchMode.SQRT:
        stretched = np.sqrt(norm)
    elif mode == StretchMode.HISTEQ:
        # Histogram Equalization
        flat = norm.flatten()
        hist, bins = np.histogram(flat, bins=256, range=(0.0, 1.0))
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-9)
        stretched = np.interp(flat, bins[:-1], cdf_normalized).reshape(norm.shape)
    else:
        # Linear / ZScale / MinMax
        stretched = norm

    stretched = np.clip(stretched, 0.0, 1.0)

    if invert:
        stretched = 1.0 - stretched

    # Convert to 8-bit uint8 for GPU / Qt rendering
    return (stretched * 255.0).astype(np.uint8)
