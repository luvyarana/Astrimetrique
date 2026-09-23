import pytest
import numpy as np
from astrimetrique.core.plate_solver import (
    LeastSquaresPlateSolver,
    ReferenceStar,
    radec_to_standard_coords,
    standard_coords_to_radec,
)
from astrimetrique.core.synthetic_data import generate_synthetic_fits_field
from astrimetrique.utils.coordinates import angular_distance_arcsec


def test_standard_coords_invertibility():
    ra_0, dec_0 = 150.0, 30.0
    test_ra, test_dec = 150.25, 30.18

    xi, eta = radec_to_standard_coords(test_ra, test_dec, ra_0, dec_0)
    rec_ra, rec_dec = standard_coords_to_radec(xi, eta, ra_0, dec_0)

    assert abs(rec_ra - test_ra) < 1e-9
    assert abs(rec_dec - test_dec) < 1e-9


def test_plate_solver_synthetic_field():
    # Generate synthetic field with 12 reference stars and known asteroid
    center_ra, center_dec = 200.0, 15.0
    scale_arcsec = 1.35
    rotation_deg = 20.0

    fits_img, ref_stars, target_star = generate_synthetic_fits_field(
        width=1024,
        height=1024,
        center_ra_deg=center_ra,
        center_dec_deg=center_dec,
        pixel_scale_arcsec=scale_arcsec,
        rotation_deg=rotation_deg,
        num_stars=12,
        include_asteroid=True,
        noise_level=5.0,
        seed=100,
    )

    solver = LeastSquaresPlateSolver()
    solution = solver.solve(ref_stars, center_ra_hint=center_ra, center_dec_hint=center_dec)

    # 1. Verify Plate Scale matches within 0.01 arcsec/px
    assert abs(solution.pixel_scale_avg_arcsec - scale_arcsec) < 0.02

    # 2. Verify Rotation angle matches
    angle_diff = (solution.rotation_deg - rotation_deg + 180.0) % 360.0 - 180.0
    assert abs(angle_diff) < 0.5

    # 3. Verify Sub-arcsecond Total RMS error (< 0.05 arcsec)
    assert solution.rms_total_arcsec < 0.05

    # 4. Verify coordinate transformation on the unknown target asteroid
    target_ra_solved, target_dec_solved = solver.pixel_to_sky(target_star.x, target_star.y)
    sep_arcsec = angular_distance_arcsec(target_ra_solved, target_dec_solved, target_star.ra_deg, target_star.dec_deg)
    assert sep_arcsec < 0.05  # Sub-arcsecond accuracy!

    # 5. Verify inverse pixel lookup
    inv_x, inv_y = solver.sky_to_pixel(target_ra_solved, target_dec_solved)
    assert abs(inv_x - target_star.x) < 0.05
    assert abs(inv_y - target_star.y) < 0.05
