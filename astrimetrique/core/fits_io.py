"""
FITS I/O Module for Astrimetrique.
Handles loading, metadata extraction, and multi-extension parsing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


@dataclass
class FITSMetadata:
    """Metadata extracted from FITS headers."""
    filename: str = ""
    object_name: str = ""
    telescope: str = ""
    instrument: str = ""
    filter_name: str = ""
    exposure_time: float = 0.0
    date_obs: Optional[datetime] = None
    date_obs_raw: str = ""
    site_lat: Optional[float] = None
    site_lon: Optional[float] = None
    site_elev: Optional[float] = None
    gain: Optional[float] = None
    pixel_scale_approx: Optional[float] = None  # arcsec / pixel if WCS present
    ra_approx: Optional[float] = None  # degrees
    dec_approx: Optional[float] = None  # degrees
    image_shape: Tuple[int, int] = (0, 0)
    bitpix: int = 16
    headers: Dict[str, Any] = field(default_factory=dict)
    has_wcs: bool = False


class FITSImage:
    """Represents a loaded astronomical FITS image and its metadata."""

    def __init__(self, data: np.ndarray, metadata: FITSMetadata, wcs: Optional[WCS] = None):
        self.data: np.ndarray = data.astype(np.float32)
        self.metadata: FITSMetadata = metadata
        self.wcs: Optional[WCS] = wcs

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    def get_pixel_value(self, x: float, y: float) -> float:
        """Get pixel intensity at (x, y) coordinates (0-indexed)."""
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < self.height and 0 <= ix < self.width:
            return float(self.data[iy, ix])
        return 0.0


def parse_date_obs(date_str: str) -> Optional[datetime]:
    """Parse various FITS DATE-OBS formats."""
    if not date_str:
        return None
    date_str = str(date_str).strip()
    
    # Common formats: '2024-05-18T22:15:30.123', '2024-05-18 22:15:30', '18/05/24'
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%y",
        "%d/%m/%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def load_fits(file_path: str | Path) -> FITSImage:
    """
    Robustly load a FITS file and parse image data + metadata.
    Handles PrimaryHDU and ImageHDU extensions.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"FITS file not found: {path}")

    with fits.open(path, memmap=True) as hdul:
        # Find the first HDU with 2D image data
        image_hdu = None
        for hdu in hdul:
            if hdu.data is not None and isinstance(hdu.data, np.ndarray) and hdu.data.ndim >= 2:
                image_hdu = hdu
                break

        if image_hdu is None:
            raise ValueError(f"No 2D image data found in FITS file: {path}")

        raw_data = image_hdu.data
        # If 3D (e.g. RGB or datacube), extract first slice or 2D plane
        if raw_data.ndim == 3:
            raw_data = raw_data[0]
        elif raw_data.ndim > 3:
            raw_data = np.squeeze(raw_data)
            if raw_data.ndim > 2:
                raw_data = raw_data[0]

        # Ensure 2D float array with finite values
        data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Parse header
        hdr = image_hdu.header
        primary_hdr = hdul[0].header if len(hdul) > 0 else hdr

        # Merge primary and image headers for comprehensive metadata
        all_headers: Dict[str, Any] = {}
        for key in primary_hdr.keys():
            if key and key not in all_headers:
                all_headers[key] = primary_hdr[key]
        for key in hdr.keys():
            if key:
                all_headers[key] = hdr[key]

        # Extract metadata fields
        date_obs_raw = str(all_headers.get("DATE-OBS", all_headers.get("DATE", "")))
        time_obs_raw = str(all_headers.get("TIME-OBS", all_headers.get("UT", "")))
        
        # Combine DATE-OBS and TIME-OBS if separate
        if date_obs_raw and time_obs_raw and "T" not in date_obs_raw and " " not in date_obs_raw:
            combined_date_str = f"{date_obs_raw}T{time_obs_raw}"
        else:
            combined_date_str = date_obs_raw

        date_obs = parse_date_obs(combined_date_str)

        # Approximate celestial center and scale from WCS if available
        parsed_wcs = None
        ra_approx = None
        dec_approx = None
        pixel_scale_approx = None
        has_wcs = False

        try:
            parsed_wcs = WCS(hdr)
            if parsed_wcs.has_celestial:
                has_wcs = True
                cy, cx = data.shape[0] / 2.0, data.shape[1] / 2.0
                sky_center = parsed_wcs.pixel_to_world(cx, cy)
                ra_approx = float(sky_center.ra.deg)
                dec_approx = float(sky_center.dec.deg)
                # Compute pixel scale in arcseconds/pixel
                scale_deg = np.mean(np.abs(parsed_wcs.pixel_scale_matrix.diagonal()))
                pixel_scale_approx = float(scale_deg * 3600.0)
        except Exception:
            parsed_wcs = None
            has_wcs = False

        # Fallback to direct header coordinates if WCS parser didn't resolve
        if ra_approx is None:
            if "CRVAL1" in all_headers:
                ra_approx = float(all_headers["CRVAL1"])
            elif "RA" in all_headers:
                try:
                    ra_approx = float(all_headers["RA"])
                except ValueError:
                    pass
        if dec_approx is None:
            if "CRVAL2" in all_headers:
                dec_approx = float(all_headers["CRVAL2"])
            elif "DEC" in all_headers:
                try:
                    dec_approx = float(all_headers["DEC"])
                except ValueError:
                    pass

        metadata = FITSMetadata(
            filename=path.name,
            object_name=str(all_headers.get("OBJECT", "")),
            telescope=str(all_headers.get("TELESCOP", "")),
            instrument=str(all_headers.get("INSTRUME", "")),
            filter_name=str(all_headers.get("FILTER", "")),
            exposure_time=float(all_headers.get("EXPTIME", all_headers.get("EXPOSURE", 0.0))),
            date_obs=date_obs,
            date_obs_raw=combined_date_str,
            site_lat=float(all_headers["SITELAT"]) if "SITELAT" in all_headers else None,
            site_lon=float(all_headers["SITELONG"]) if "SITELONG" in all_headers else None,
            site_elev=float(all_headers["SITEELEV"]) if "SITEELEV" in all_headers else None,
            gain=float(all_headers["GAIN"]) if "GAIN" in all_headers else None,
            pixel_scale_approx=pixel_scale_approx,
            ra_approx=ra_approx,
            dec_approx=dec_approx,
            image_shape=data.shape,
            bitpix=int(all_headers.get("BITPIX", 16)),
            headers=all_headers,
            has_wcs=has_wcs,
        )

        return FITSImage(data=data, metadata=metadata, wcs=parsed_wcs)
