"""
Least-Squares Plate-Constants (LSPC) Plate Solver Engine.
Implements standard tangent-plane (gnomonic) projection and 6-constant affine astrometric reduction.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from astrimetrique.utils.coordinates import angular_distance_arcsec, deg_to_dec_dms, deg_to_ra_hms


@dataclass
class ReferenceStar:
    """A reference star with CCD pixel coordinates and catalog RA/Dec."""
    star_id: str
    x: float  # CCD X (pixels)
    y: float  # CCD Y (pixels)
    ra_deg: float  # Catalog RA in J2000 decimal degrees
    dec_deg: float  # Catalog Dec in J2000 decimal degrees
    mag: Optional[float] = None
    enabled: bool = True
    # Solved / Residual values populated after fit:
    solved_ra_deg: Optional[float] = None
    solved_dec_deg: Optional[float] = None
    res_ra_arcsec: float = 0.0  # Delta RA * cos(Dec) in arcsec
    res_dec_arcsec: float = 0.0  # Delta Dec in arcsec
    res_total_arcsec: float = 0.0  # Total angular separation in arcsec


@dataclass
class PlateSolution:
    """Astrometric Plate Solution results and diagnostics."""
    # Tangent point center (J2000)
    ra_0_deg: float
    dec_0_deg: float
    
    # 6 Plate constants: xi = ax + by + c, eta = dx + ey + f (in radians)
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    
    # Inverted constants: x = a_inv * xi + b_inv * eta + c_inv ...
    a_inv: float = 0.0
    b_inv: float = 0.0
    c_inv: float = 0.0
    d_inv: float = 0.0
    e_inv: float = 0.0
    f_inv: float = 0.0
    
    # Derived physical properties
    pixel_scale_x_arcsec: float = 0.0  # arcsec / pixel
    pixel_scale_y_arcsec: float = 0.0  # arcsec / pixel
    pixel_scale_avg_arcsec: float = 0.0  # arcsec / pixel
    rotation_deg: float = 0.0  # Position angle of X-axis in degrees
    
    # Solution quality diagnostics
    num_stars_used: int = 0
    num_stars_total: int = 0
    rms_ra_arcsec: float = 0.0
    rms_dec_arcsec: float = 0.0
    rms_total_arcsec: float = 0.0
    max_residual_arcsec: float = 0.0
    condition_number: float = 1.0
    stars: List[ReferenceStar] = field(default_factory=list)


def radec_to_standard_coords(
    ra_deg: float | np.ndarray,
    dec_deg: float | np.ndarray,
    ra_0_deg: float,
    dec_0_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert celestial coordinates (RA, Dec) to standard coordinates (xi, eta)
    on the tangent plane (gnomonic projection) around tangent center (ra_0, dec_0).
    Returns (xi, eta) in radians.
    """
    alpha = np.radians(ra_deg)
    delta = np.radians(dec_deg)
    alpha_0 = np.radians(ra_0_deg)
    delta_0 = np.radians(dec_0_deg)

    d_alpha = alpha - alpha_0
    
    # Denominator for gnomonic projection
    denom = np.sin(delta) * np.sin(delta_0) + np.cos(delta) * np.cos(delta_0) * np.cos(d_alpha)
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)

    xi = (np.cos(delta) * np.sin(d_alpha)) / denom
    eta = (np.sin(delta) * np.cos(delta_0) - np.cos(delta) * np.sin(delta_0) * np.cos(d_alpha)) / denom

    return xi, eta


def standard_coords_to_radec(
    xi: float | np.ndarray,
    eta: float | np.ndarray,
    ra_0_deg: float,
    dec_0_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert standard tangent-plane coordinates (xi, eta) in radians
    back into celestial coordinates (RA, Dec) in decimal degrees.
    """
    alpha_0 = np.radians(ra_0_deg)
    delta_0 = np.radians(dec_0_deg)

    # Inverse gnomonic projection equations
    denom = np.cos(delta_0) - eta * np.sin(delta_0)
    
    # RA calculation
    d_alpha = np.arctan2(xi, denom)
    alpha = alpha_0 + d_alpha
    ra_deg = np.degrees(alpha) % 360.0

    # Dec calculation
    delta = np.arctan2(
        (np.sin(delta_0) + eta * np.cos(delta_0)) * np.cos(d_alpha),
        denom,
    )
    dec_deg = np.degrees(delta)

    return ra_deg, dec_deg


class LeastSquaresPlateSolver:
    """
    High-precision Least-Squares Plate-Constants (LSPC) astrometric solver.
    """

    def __init__(self):
        self.solution: Optional[PlateSolution] = None

    def solve(
        self,
        reference_stars: List[ReferenceStar],
        center_ra_hint: Optional[float] = None,
        center_dec_hint: Optional[float] = None,
    ) -> PlateSolution:
        """
        Compute the 6-constant astrometric plate solution using active reference stars.
        Requires at least 3 non-collinear reference stars (recommended 4+).
        """
        active_stars = [s for s in reference_stars if s.enabled]
        if len(active_stars) < 3:
            raise ValueError(
                f"Astrometric reduction requires at least 3 active reference stars (got {len(active_stars)})."
            )

        # 1. Determine tangent point (ra_0, dec_0)
        if center_ra_hint is not None and center_dec_hint is not None:
            ra_0 = center_ra_hint
            dec_0 = center_dec_hint
        else:
            # Mean celestial coordinate of reference stars
            ra_0 = float(np.mean([s.ra_deg for s in active_stars]))
            dec_0 = float(np.mean([s.dec_deg for s in active_stars]))

        # 2. Extract pixel coordinates and compute standard coordinates (xi, eta)
        x_vals = np.array([s.x for s in active_stars], dtype=np.float64)
        y_vals = np.array([s.y for s in active_stars], dtype=np.float64)
        ra_vals = np.array([s.ra_deg for s in active_stars], dtype=np.float64)
        dec_vals = np.array([s.dec_deg for s in active_stars], dtype=np.float64)

        xi_vals, eta_vals = radec_to_standard_coords(ra_vals, dec_vals, ra_0, dec_0)

        # 3. Build Design Matrix A = [x, y, 1]
        n_stars = len(active_stars)
        A = np.column_stack([x_vals, y_vals, np.ones(n_stars, dtype=np.float64)])

        # Calculate condition number to ensure matrix is not singular / degenerate
        cond_num = float(np.linalg.cond(A))
        if np.isinf(cond_num) or cond_num > 1e12:
            raise ValueError(
                "Reference stars are collinear or degenerate; design matrix is singular."
            )

        # 4. Solve Normal Equations via SVD Least Squares:
        # A * [a, b, c]^T = xi
        # A * [d, e, f]^T = eta
        coeffs_xi, _, _, _ = np.linalg.lstsq(A, xi_vals, rcond=None)
        coeffs_eta, _, _, _ = np.linalg.lstsq(A, eta_vals, rcond=None)

        a, b, c = coeffs_xi
        d, e, f = coeffs_eta

        # 5. Compute Inverse Matrix: [x, y]^T = M_inv * [xi - c, eta - f]^T
        M = np.array([[a, b], [d, e]], dtype=np.float64)
        det_M = np.linalg.det(M)
        if abs(det_M) < 1e-18:
            raise ValueError("Degenerate linear transformation matrix (determinant ~ 0).")
        
        M_inv = np.linalg.inv(M)
        a_inv, b_inv = M_inv[0, 0], M_inv[0, 1]
        d_inv, e_inv = M_inv[1, 0], M_inv[1, 1]
        # Offsets
        c_inv = -(a_inv * c + b_inv * f)
        f_inv = -(d_inv * c + e_inv * f)

        # 6. Physical properties
        # Plate scale in radians/pixel -> convert to arcsec/pixel (1 rad = 206264.806247 arcsec)
        rad_to_arcsec = 206264.806247
        scale_x_rad = np.sqrt(a**2 + d**2)
        scale_y_rad = np.sqrt(b**2 + e**2)
        pixel_scale_x = float(scale_x_rad * rad_to_arcsec)
        pixel_scale_y = float(scale_y_rad * rad_to_arcsec)
        pixel_scale_avg = float((pixel_scale_x + pixel_scale_y) / 2.0)

        # Field rotation angle (Position angle theta)
        rotation_rad = np.arctan2(d, a)
        rotation_deg = float(np.degrees(rotation_rad) % 360.0)

        # 7. Evaluate residuals on all reference stars
        res_ra_list = []
        res_dec_list = []
        res_tot_list = []

        for star in reference_stars:
            # Predict RA, Dec for this star using the solution
            pred_xi = a * star.x + b * star.y + c
            pred_eta = d * star.x + e * star.y + f
            pred_ra, pred_dec = standard_coords_to_radec(pred_xi, pred_eta, ra_0, dec_0)
            star.solved_ra_deg = float(pred_ra)
            star.solved_dec_deg = float(pred_dec)

            # Residuals in arcseconds:
            # Delta RA * cos(Dec)
            delta_ra = (star.solved_ra_deg - star.ra_deg)
            # Handle 360 degree wrap-around
            if delta_ra > 180.0:
                delta_ra -= 360.0
            elif delta_ra < -180.0:
                delta_ra += 360.0
                
            delta_ra_arcsec = float(delta_ra * 3600.0 * np.cos(np.radians(star.dec_deg)))
            delta_dec_arcsec = float((star.solved_dec_deg - star.dec_deg) * 3600.0)
            tot_arcsec = angular_distance_arcsec(star.solved_ra_deg, star.solved_dec_deg, star.ra_deg, star.dec_deg)

            star.res_ra_arcsec = delta_ra_arcsec
            star.res_dec_arcsec = delta_dec_arcsec
            star.res_total_arcsec = tot_arcsec

            if star.enabled:
                res_ra_list.append(delta_ra_arcsec)
                res_dec_list.append(delta_dec_arcsec)
                res_tot_list.append(tot_arcsec)

        # RMS Error calculations
        if len(res_ra_list) > 0:
            rms_ra = float(np.sqrt(np.mean(np.array(res_ra_list) ** 2)))
            rms_dec = float(np.sqrt(np.mean(np.array(res_dec_list) ** 2)))
            rms_tot = float(np.sqrt(np.mean(np.array(res_tot_list) ** 2)))
            max_res = float(np.max(res_tot_list))
        else:
            rms_ra, rms_dec, rms_tot, max_res = 0.0, 0.0, 0.0, 0.0

        self.solution = PlateSolution(
            ra_0_deg=ra_0,
            dec_0_deg=dec_0,
            a=float(a),
            b=float(b),
            c=float(c),
            d=float(d),
            e=float(e),
            f=float(f),
            a_inv=float(a_inv),
            b_inv=float(b_inv),
            c_inv=float(c_inv),
            d_inv=float(d_inv),
            e_inv=float(e_inv),
            f_inv=float(f_inv),
            pixel_scale_x_arcsec=pixel_scale_x,
            pixel_scale_y_arcsec=pixel_scale_y,
            pixel_scale_avg_arcsec=pixel_scale_avg,
            rotation_deg=rotation_deg,
            num_stars_used=len(active_stars),
            num_stars_total=len(reference_stars),
            rms_ra_arcsec=rms_ra,
            rms_dec_arcsec=rms_dec,
            rms_total_arcsec=rms_tot,
            max_residual_arcsec=max_res,
            condition_number=cond_num,
            stars=reference_stars,
        )
        return self.solution

    def pixel_to_sky(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert CCD pixel coordinate (x, y) to celestial (RA, Dec) in J2000 decimal degrees.
        """
        if self.solution is None:
            raise RuntimeError("Plate solution has not been computed yet.")

        sol = self.solution
        xi = sol.a * x + sol.b * y + sol.c
        eta = sol.d * x + sol.e * y + sol.f
        ra_deg, dec_deg = standard_coords_to_radec(xi, eta, sol.ra_0_deg, sol.dec_0_deg)
        return float(ra_deg), float(dec_deg)

    def sky_to_pixel(self, ra_deg: float, dec_deg: float) -> Tuple[float, float]:
        """
        Convert celestial (RA, Dec) to CCD pixel coordinate (x, y).
        """
        if self.solution is None:
            raise RuntimeError("Plate solution has not been computed yet.")

        sol = self.solution
        xi, eta = radec_to_standard_coords(ra_deg, dec_deg, sol.ra_0_deg, sol.dec_0_deg)
        x = sol.a_inv * xi + sol.b_inv * eta + sol.c_inv
        y = sol.d_inv * xi + sol.e_inv * eta + sol.f_inv
        return float(x), float(y)
