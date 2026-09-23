"""
Synthetic Astronomical Data Generator.
Creates realistic CCD star fields, optical noise, and moving minor planet targets
for rigorous testing, validation, and offline demonstrations.
"""

from datetime import datetime, timezone
from typing import List, Tuple
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astrimetrique.core.fits_io import FITSImage, FITSMetadata
from astrimetrique.core.plate_solver import (
    ReferenceStar,
    radec_to_standard_coords,
    standard_coords_to_radec,
)


def generate_synthetic_fits_field(
    width: int = 1024,
    height: int = 1024,
    center_ra_deg: float = 180.0,
    center_dec_deg: float = 25.0,
    pixel_scale_arcsec: float = 1.25,
    rotation_deg: float = 15.0,
    num_stars: int = 16,
    include_asteroid: bool = True,
    asteroid_mag: float = 15.5,
    obs_time: datetime | None = None,
    noise_level: float = 15.0,
    sky_background: float = 300.0,
    seed: int = 42,
) -> Tuple[FITSImage, List[ReferenceStar], ReferenceStar | None]:
    """
    Generate a realistic synthetic astronomical CCD image with known ground-truth
    astrometry, Gaussian stars, noise, and minor planet target.
    """
    rng = np.random.default_rng(seed)
    obs_time = obs_time or datetime(2026, 9, 22, 21, 30, 0, tzinfo=timezone.utc)

    # 1. Initialize background sky with gradient and Poisson / Gaussian noise
    y_coords, x_coords = np.indices((height, width))
    gradient = 0.02 * (x_coords - width / 2.0) - 0.015 * (y_coords - height / 2.0)
    base_image = sky_background + gradient + rng.normal(0.0, noise_level, size=(height, width))
    base_image = np.clip(base_image, 0.0, None).astype(np.float32)

    # 2. Transformation matrix from (x, y) relative to center to (xi, eta) in radians
    # 1 rad = 206264.806247 arcsec
    rad_per_pixel = (pixel_scale_arcsec / 3600.0) * (np.pi / 180.0)
    theta_rad = np.radians(rotation_deg)

    # Transformation matrix M: [xi, eta]^T = M * [x - cx, y - cy]^T
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    
    # [xi, eta]^T = rad_per_pixel * [[cos(t), -sin(t)], [sin(t), cos(t)]] * [dx, dy]^T
    # This represents standard astrometric orientation
    a_mat = rad_per_pixel * cos_t
    b_mat = -rad_per_pixel * sin_t
    d_mat = rad_per_pixel * sin_t
    e_mat = rad_per_pixel * cos_t

    cx = width / 2.0
    cy = height / 2.0

    # 3. Place reference stars with Gaussian PSFs
    margin = 80
    x_positions = rng.uniform(margin, width - margin, size=num_stars)
    y_positions = rng.uniform(margin, height - margin, size=num_stars)
    star_mags = rng.uniform(11.0, 16.0, size=num_stars)

    ref_stars: List[ReferenceStar] = []

    for i in range(num_stars):
        x0 = float(x_positions[i])
        y0 = float(y_positions[i])
        mag = float(star_mags[i])

        # Compute ground-truth (xi, eta)
        dx = x0 - cx
        dy = y0 - cy
        xi = a_mat * dx + b_mat * dy
        eta = d_mat * dx + e_mat * dy

        # Unproject to celestial (RA, Dec)
        ra_deg, dec_deg = standard_coords_to_radec(xi, eta, center_ra_deg, center_dec_deg)

        ref_stars.append(ReferenceStar(
            star_id=f"REF-{i+1:02d}",
            x=x0,
            y=y0,
            ra_deg=float(ra_deg),
            dec_deg=float(dec_deg),
            mag=mag,
            enabled=True,
        ))

        # Render 2D Gaussian PSF
        # Amplitude scaled inversely with magnitude
        amp = float(10.0 ** ((20.0 - mag) / 2.5) * 8.0)
        sigma = rng.uniform(1.2, 1.6)  # FWHM ~ 3.0 px

        # Render onto image within bounding box
        box_r = int(np.ceil(sigma * 4.5))
        xmin = max(0, int(round(x0)) - box_r)
        xmax = min(width, int(round(x0)) + box_r + 1)
        ymin = max(0, int(round(y0)) - box_r)
        ymax = min(height, int(round(y0)) + box_r + 1)

        yg, xg = np.indices((ymax - ymin, xmax - xmin))
        xg_abs = xmin + xg
        yg_abs = ymin + yg

        psf = amp * np.exp(-0.5 * (((xg_abs - x0) / sigma) ** 2 + ((yg_abs - y0) / sigma) ** 2))
        base_image[ymin:ymax, xmin:xmax] += psf.astype(np.float32)

    # 4. Optional Minor Planet / Asteroid target
    asteroid_star = None
    if include_asteroid:
        # Place asteroid near center-quarter
        ast_x = cx + rng.uniform(-150, 150)
        ast_y = cy + rng.uniform(-150, 150)
        
        dx = ast_x - cx
        dy = ast_y - cy
        xi = a_mat * dx + b_mat * dy
        eta = d_mat * dx + e_mat * dy
        ast_ra, ast_dec = standard_coords_to_radec(xi, eta, center_ra_deg, center_dec_deg)

        asteroid_star = ReferenceStar(
            star_id="2024 AB",
            x=float(ast_x),
            y=float(ast_y),
            ra_deg=float(ast_ra),
            dec_deg=float(ast_dec),
            mag=asteroid_mag,
            enabled=True,
        )

        amp = float(10.0 ** ((20.0 - asteroid_mag) / 2.5) * 8.0)
        sigma = 1.35
        box_r = int(np.ceil(sigma * 4.5))
        xmin = max(0, int(round(ast_x)) - box_r)
        xmax = min(width, int(round(ast_x)) + box_r + 1)
        ymin = max(0, int(round(ast_y)) - box_r)
        ymax = min(height, int(round(ast_y)) + box_r + 1)

        yg, xg = np.indices((ymax - ymin, xmax - xmin))
        xg_abs = xmin + xg
        yg_abs = ymin + yg
        psf = amp * np.exp(-0.5 * (((xg_abs - ast_x) / sigma) ** 2 + ((yg_abs - ast_y) / sigma) ** 2))
        base_image[ymin:ymax, xmin:xmax] += psf.astype(np.float32)

    # 5. Create Astropy WCS and FITS Header
    w = WCS(naxis=2)
    w.wcs.crpix = [cx + 1.0, cy + 1.0]
    w.wcs.cdelt = [-pixel_scale_arcsec / 3600.0, pixel_scale_arcsec / 3600.0]
    w.wcs.crval = [center_ra_deg, center_dec_deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crota = [0.0, rotation_deg]

    header_dict = {
        "SIMPLE": True,
        "BITPIX": -32,
        "NAXIS": 2,
        "NAXIS1": width,
        "NAXIS2": height,
        "OBJECT": "Synthetic Asteroid Field 2024 AB",
        "TELESCOP": "0.4m f/8 Astrograph",
        "INSTRUME": "Astrimetrique Scientific CCD",
        "FILTER": "R",
        "EXPTIME": 120.0,
        "DATE-OBS": obs_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "CRVAL1": center_ra_deg,
        "CRVAL2": center_dec_deg,
        "CRPIX1": cx + 1.0,
        "CRPIX2": cy + 1.0,
        "CDELT1": -pixel_scale_arcsec / 3600.0,
        "CDELT2": pixel_scale_arcsec / 3600.0,
        "CROTA2": rotation_deg,
        "GAIN": 1.4,
    }

    metadata = FITSMetadata(
        filename="synthetic_field_2024_AB.fits",
        object_name="Synthetic Asteroid Field 2024 AB",
        telescope="0.4m f/8 Astrograph",
        instrument="Astrimetrique Scientific CCD",
        filter_name="R",
        exposure_time=120.0,
        date_obs=obs_time,
        date_obs_raw=obs_time.strftime("%Y-%m-%dT%H:%M:%S"),
        site_lat=31.95,
        site_lon=-111.60,
        gain=1.4,
        pixel_scale_approx=pixel_scale_arcsec,
        ra_approx=center_ra_deg,
        dec_approx=center_dec_deg,
        image_shape=(height, width),
        bitpix=-32,
        headers=header_dict,
        has_wcs=True,
    )

    fits_image = FITSImage(data=base_image, metadata=metadata, wcs=w)
    return fits_image, ref_stars, asteroid_star


def generate_multiepoch_synthetic_pair(
    width: int = 1024,
    height: int = 1024,
    center_ra_deg: float = 180.0,
    center_dec_deg: float = 25.0,
    pixel_scale_arcsec: float = 1.25,
    num_stars: int = 16,
    telescope_shift_x: float = 14.5,
    telescope_shift_y: float = -8.2,
    telescope_rot_deg: float = 0.8,
    asteroid_motion_x: float = 28.0,
    asteroid_motion_y: float = 15.0,
    asteroid_mag: float = 15.0,
    noise_level: float = 12.0,
    seed: int = 42,
) -> Tuple[FITSImage, FITSImage, List[ReferenceStar], List[ReferenceStar], ReferenceStar, ReferenceStar]:
    """
    Generate a pair of multi-epoch observations (Epoch 1 Reference Image A, Epoch 2 Target Image B)
    with telescope pointing drift, rotation, and a moving minor planet.
    """
    from datetime import timedelta
    t1 = datetime(2026, 9, 22, 21, 0, 0, tzinfo=timezone.utc)
    t2 = t1 + timedelta(minutes=45)

    # Epoch 1: Reference frame
    img_a, stars_a, ast_a = generate_synthetic_fits_field(
        width=width,
        height=height,
        center_ra_deg=center_ra_deg,
        center_dec_deg=center_dec_deg,
        pixel_scale_arcsec=pixel_scale_arcsec,
        rotation_deg=0.0,
        num_stars=num_stars,
        include_asteroid=True,
        asteroid_mag=asteroid_mag,
        obs_time=t1,
        noise_level=noise_level,
        seed=seed,
    )
    img_a.metadata.filename = "synthetic_epoch1_ref.fits"
    img_a.metadata.object_name = "Synthetic Field Epoch 1"

    # Epoch 2: Telescope shifted & rotated, asteroid displaced by motion
    rng = np.random.default_rng(seed + 10)
    theta_rad = np.radians(telescope_rot_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    cx, cy = width / 2.0, height / 2.0

    # Base image with noise
    y_coords, x_coords = np.indices((height, width))
    base_b = 310.0 + rng.normal(0.0, noise_level, size=(height, width)).astype(np.float32)

    stars_b: List[ReferenceStar] = []
    # Map stars from Epoch 1 to Epoch 2 coordinates
    for star in stars_a:
        dx = star.x - cx
        dy = star.y - cy
        # Rotate around center and shift
        x_rot = cos_t * dx - sin_t * dy + cx + telescope_shift_x
        y_rot = sin_t * dx + cos_t * dy + cy + telescope_shift_y

        stars_b.append(ReferenceStar(
            star_id=star.star_id,
            x=float(x_rot),
            y=float(y_rot),
            ra_deg=star.ra_deg,
            dec_deg=star.dec_deg,
            mag=star.mag,
            enabled=True,
        ))

        # Render star PSF
        amp = float(10.0 ** ((20.0 - (star.mag or 13.0)) / 2.5) * 8.0)
        sigma = 1.4
        box_r = int(np.ceil(sigma * 4.5))
        xmin = max(0, int(round(x_rot)) - box_r)
        xmax = min(width, int(round(x_rot)) + box_r + 1)
        ymin = max(0, int(round(y_rot)) - box_r)
        ymax = min(height, int(round(y_rot)) + box_r + 1)

        yg, xg = np.indices((ymax - ymin, xmax - xmin))
        xg_abs = xmin + xg
        yg_abs = ymin + yg
        psf = amp * np.exp(-0.5 * (((xg_abs - x_rot) / sigma) ** 2 + ((yg_abs - y_rot) / sigma) ** 2))
        base_b[ymin:ymax, xmin:xmax] += psf.astype(np.float32)

    # Asteroid in Epoch 2: shifted by telescope motion + physical orbital displacement
    ast_dx = (ast_a.x - cx)
    ast_dy = (ast_a.y - cy)
    ast_x2 = cos_t * ast_dx - sin_t * ast_dy + cx + telescope_shift_x + asteroid_motion_x
    ast_y2 = sin_t * ast_dx + cos_t * ast_dy + cy + telescope_shift_y + asteroid_motion_y

    ast_b = ReferenceStar(
        star_id="2024 AB",
        x=float(ast_x2),
        y=float(ast_y2),
        ra_deg=ast_a.ra_deg + (asteroid_motion_x * pixel_scale_arcsec / 3600.0) / np.cos(np.radians(center_dec_deg)),
        dec_deg=ast_a.dec_deg + (asteroid_motion_y * pixel_scale_arcsec / 3600.0),
        mag=asteroid_mag,
        enabled=True,
    )

    amp = float(10.0 ** ((20.0 - asteroid_mag) / 2.5) * 8.0)
    sigma = 1.35
    box_r = int(np.ceil(sigma * 4.5))
    xmin = max(0, int(round(ast_x2)) - box_r)
    xmax = min(width, int(round(ast_x2)) + box_r + 1)
    ymin = max(0, int(round(ast_y2)) - box_r)
    ymax = min(height, int(round(ast_y2)) + box_r + 1)

    yg, xg = np.indices((ymax - ymin, xmax - xmin))
    xg_abs = xmin + xg
    yg_abs = ymin + yg
    psf = amp * np.exp(-0.5 * (((xg_abs - ast_x2) / sigma) ** 2 + ((yg_abs - ast_y2) / sigma) ** 2))
    base_b[ymin:ymax, xmin:xmax] += psf.astype(np.float32)

    meta_b = FITSMetadata(
        filename="synthetic_epoch2_target.fits",
        object_name="Synthetic Field Epoch 2",
        telescope="0.4m f/8 Astrograph",
        instrument="Astrimetrique Scientific CCD",
        filter_name="R",
        exposure_time=120.0,
        date_obs=t2,
        date_obs_raw=t2.strftime("%Y-%m-%dT%H:%M:%S"),
        site_lat=31.95,
        site_lon=-111.60,
        gain=1.4,
        pixel_scale_approx=pixel_scale_arcsec,
        ra_approx=center_ra_deg,
        dec_approx=center_dec_deg,
        image_shape=(height, width),
        bitpix=-32,
        headers=img_a.metadata.headers.copy(),
        has_wcs=True,
    )
    img_b = FITSImage(data=base_b, metadata=meta_b, wcs=img_a.wcs)

    return img_a, img_b, stars_a, stars_b, ast_a, ast_b
