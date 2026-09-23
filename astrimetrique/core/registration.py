"""
Astronomical Image Registration Engine.
Computes 6-constant affine transformations between reference and target frames,
and performs high-precision sub-pixel image warping via bilinear/bicubic interpolation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import map_coordinates

from astrimetrique.core.centroid import fit_centroid_2d_gaussian
from astrimetrique.core.fits_io import FITSImage
from astrimetrique.core.plate_solver import (
    LeastSquaresPlateSolver,
    PlateSolution,
    ReferenceStar,
)


@dataclass
class RegistrationTransform:
    """Affine coordinate transformation parameters from Target (Image B) to Reference (Image A)."""
    # Forward Affine matrix 2x3: [x_A, y_A]^T = M * [x_B, y_B, 1]^T
    # [x_A] = [a b c] [x_B]
    # [y_A]   [d e f] [y_B]
    #                 [ 1 ]
    matrix_b_to_a: np.ndarray
    # Inverse Affine matrix 2x3: [x_B, y_B]^T = M_inv * [x_A, y_A, 1]^T
    matrix_a_to_b: np.ndarray
    
    # 6 Plate/Affine Constants: x_A = ax_B + by_B + c, y_A = dx_B + ey_B + f
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0  # Translation X (Delta X in pixels)
    d: float = 0.0
    e: float = 1.0
    f: float = 0.0  # Translation Y (Delta Y in pixels)
    
    # Physical decomposition
    translation_x: float = 0.0  # pixels
    translation_y: float = 0.0  # pixels
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation_deg: float = 0.0  # degrees
    shear: float = 0.0
    
    # Registration quality
    num_matched_points: int = 0
    alignment_rms_px: float = 0.0


def compute_affine_from_points(
    points_ref: np.ndarray | List[Tuple[float, float]],
    points_target: np.ndarray | List[Tuple[float, float]],
) -> RegistrationTransform:
    """
    Compute 6-constant affine transformation mapping Target points (Image B) to Reference points (Image A):
    x_A = a * x_B + b * y_B + c
    y_A = d * x_B + e * y_B + f
    Requires at least 3 non-collinear matched points.
    """
    pts_a = np.asarray(points_ref, dtype=np.float64)
    pts_b = np.asarray(points_target, dtype=np.float64)

    if len(pts_a) < 3 or len(pts_b) < 3:
        raise ValueError(f"Registration requires at least 3 matched points (got {len(pts_a)}).")

    n = len(pts_a)
    # Design matrix: A_mat = [[x_B, y_B, 1]]
    A_mat = np.column_stack([pts_b[:, 0], pts_b[:, 1], np.ones(n, dtype=np.float64)])

    # Check condition number to avoid singular/collinear degeneracies
    cond = np.linalg.cond(A_mat)
    if np.isinf(cond) or cond > 1e12:
        raise ValueError("Matched reference stars are collinear or degenerate.")

    # Solve least-squares:
    # A_mat * [a, b, c]^T = x_a
    # A_mat * [d, e, f]^T = y_a
    params_x, _, _, _ = np.linalg.lstsq(A_mat, pts_a[:, 0], rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(A_mat, pts_a[:, 1], rcond=None)

    a, b, c = params_x
    d, e, f = params_y

    # 3x3 Homogeneous transformation matrix B -> A
    M_3x3 = np.array([
        [a, b, c],
        [d, e, f],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Invert 3x3 matrix to obtain A -> B mapping
    M_inv_3x3 = np.linalg.inv(M_3x3)

    # Compute alignment residuals
    predicted_a = (M_3x3 @ np.column_stack([pts_b, np.ones(n)]).T).T[:, :2]
    residuals = np.linalg.norm(predicted_a - pts_a, axis=1)
    rms_px = float(np.sqrt(np.mean(residuals ** 2)))

    # Physical decomposition
    scale_x = float(np.sqrt(a**2 + d**2))
    scale_y = float(np.sqrt(b**2 + e**2))
    rotation_rad = np.arctan2(d, a)
    rotation_deg = float(np.degrees(rotation_rad) % 360.0)

    return RegistrationTransform(
        matrix_b_to_a=M_3x3[:2, :],
        matrix_a_to_b=M_inv_3x3[:2, :],
        a=float(a),
        b=float(b),
        c=float(c),
        d=float(d),
        e=float(e),
        f=float(f),
        translation_x=float(c),
        translation_y=float(f),
        scale_x=scale_x,
        scale_y=scale_y,
        rotation_deg=rotation_deg,
        num_matched_points=n,
        alignment_rms_px=rms_px,
    )


def compute_registration_from_solutions(
    sol_ref: PlateSolution,
    sol_target: PlateSolution,
    shape_ref: Tuple[int, int],
) -> RegistrationTransform:
    """
    Compute image registration transformation directly from the solved LSPC astrometric solutions
    of Image A (Reference) and Image B (Target).
    """
    solver_ref = LeastSquaresPlateSolver()
    solver_ref.solution = sol_ref
    solver_target = LeastSquaresPlateSolver()
    solver_target.solution = sol_target

    h, w = shape_ref
    grid_x = np.linspace(w * 0.15, w * 0.85, 6)
    grid_y = np.linspace(h * 0.15, h * 0.85, 6)
    gx, gy = np.meshgrid(grid_x, grid_y)
    ref_pts = np.column_stack([gx.ravel(), gy.ravel()])

    target_pts = []
    for x_a, y_a in ref_pts:
        ra, dec = solver_ref.pixel_to_sky(x_a, y_a)
        x_b, y_b = solver_target.sky_to_pixel(ra, dec)
        target_pts.append((x_b, y_b))

    target_pts_arr = np.array(target_pts, dtype=np.float64)
    return compute_affine_from_points(ref_pts, target_pts_arr)


def compute_registration_from_wcs(
    wcs_ref,
    wcs_target,
    shape_ref: Tuple[int, int],
) -> RegistrationTransform:
    """
    Compute affine registration transformation using FITS WCS headers.
    """
    h, w = shape_ref
    grid_x = np.linspace(w * 0.15, w * 0.85, 6)
    grid_y = np.linspace(h * 0.15, h * 0.85, 6)
    gx, gy = np.meshgrid(grid_x, grid_y)
    ref_pts = np.column_stack([gx.ravel(), gy.ravel()])

    target_pts = []
    for x_a, y_a in ref_pts:
        sky = wcs_ref.pixel_to_world(x_a, y_a)
        target_pix = wcs_target.world_to_pixel(sky)
        target_pts.append((float(target_pix[0]), float(target_pix[1])))

    target_pts_arr = np.array(target_pts, dtype=np.float64)
    return compute_affine_from_points(ref_pts, target_pts_arr)


def align_images_pipeline(
    img_a: FITSImage,
    img_b: FITSImage,
    stars_a: List[ReferenceStar],
    stars_b: Optional[List[ReferenceStar]] = None,
    sol_a: Optional[PlateSolution] = None,
    sol_b: Optional[PlateSolution] = None,
) -> Tuple[RegistrationTransform, np.ndarray]:
    """
    Comprehensive robust image alignment pipeline.
    Uses matched stars, plate solutions, WCS, or cross-centroiding.
    """
    active_a = [s for s in stars_a if s.enabled]
    active_b = [s for s in stars_b if s.enabled] if stars_b else []

    transform = None

    # Strategy 1: Both star lists available with matched IDs or distinct positions
    if len(active_a) >= 3 and len(active_b) >= 3:
        dict_a = {s.star_id: (s.x, s.y) for s in active_a}
        dict_b = {s.star_id: (s.x, s.y) for s in active_b}
        common_ids = [sid for sid in dict_a if sid in dict_b]

        if len(common_ids) >= 3:
            pts_a = [dict_a[sid] for sid in common_ids]
            pts_b = [dict_b[sid] for sid in common_ids]
            transform = compute_affine_from_points(pts_a, pts_b)
        else:
            # Match by celestial RA/Dec
            pts_a = []
            pts_b = []
            for sa in active_a:
                for sb in active_b:
                    if abs(sa.ra_deg - sb.ra_deg) < 0.005 and abs(sa.dec_deg - sb.dec_deg) < 0.005:
                        pts_a.append((sa.x, sa.y))
                        pts_b.append((sb.x, sb.y))
                        break
            if len(pts_a) >= 3:
                transform = compute_affine_from_points(pts_a, pts_b)

    # Strategy 2: LSPC Plate Solutions
    if transform is None and sol_a is not None and sol_b is not None:
        try:
            transform = compute_registration_from_solutions(sol_a, sol_b, img_a.shape)
        except Exception:
            transform = None

    # Strategy 3: FITS WCS Headers
    if transform is None and img_a.wcs is not None and img_b.wcs is not None and img_a.wcs.has_celestial and img_b.wcs.has_celestial:
        try:
            transform = compute_registration_from_wcs(img_a.wcs, img_b.wcs, img_a.shape)
        except Exception:
            transform = None

    # Strategy 4: Project stars from A to B and centroid on Image B
    if transform is None and len(active_a) >= 3:
        pts_a = []
        pts_b = []
        for s in active_a:
            # Estimate position on Image B: if WCS present use WCS, else use (s.x, s.y)
            if img_a.wcs and img_b.wcs and img_a.wcs.has_celestial and img_b.wcs.has_celestial:
                sky = img_a.wcs.pixel_to_world(s.x, s.y)
                guess_x, guess_y = img_b.wcs.world_to_pixel(sky)
            else:
                guess_x, guess_y = s.x, s.y

            if 15 <= guess_x < img_b.width - 15 and 15 <= guess_y < img_b.height - 15:
                # Refine centroid on Image B
                res_b = fit_centroid_2d_gaussian(img_b.data, guess_x, guess_y, box_size=19)
                if res_b.success and res_b.snr > 3.0:
                    pts_a.append((s.x, s.y))
                    pts_b.append((res_b.x, res_b.y))

        if len(pts_a) >= 3:
            transform = compute_affine_from_points(pts_a, pts_b)

    if transform is None:
        raise ValueError(
            "Could not determine alignment transformation between Image A and Image B.\n"
            "Please ensure at least 3 reference stars are defined or both images have plate solutions/WCS."
        )

    # Warp Image B into coordinate system of Image A
    warped_b = warp_image(img_b.data, transform, output_shape=img_a.shape, order=1)
    return transform, warped_b


def warp_image(
    image: np.ndarray,
    transform: RegistrationTransform | np.ndarray,
    output_shape: Optional[Tuple[int, int]] = None,
    order: int = 1,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Warp Image B into the coordinate system of Image A using 6-constant affine matrix
    and sub-pixel interpolation.
    
    Parameters:
    -----------
    image : np.ndarray
        Source Image B to be warped.
    transform : RegistrationTransform or 2x3/3x3 matrix
        Affine transformation matrix M mapping Image B to Image A.
    output_shape : Tuple[int, int], optional
        (height, width) of output image. Defaults to input image shape.
    order : int
        Interpolation order (1 = Bilinear, 3 = Bicubic Spline).
    fill_value : float
        Fill value for boundary pixels.
    """
    if output_shape is None:
        out_h, out_w = image.shape
    else:
        out_h, out_w = output_shape

    if isinstance(transform, RegistrationTransform):
        M_inv = transform.matrix_a_to_b
    elif isinstance(transform, np.ndarray):
        if transform.shape == (2, 3):
            M_3x3 = np.vstack([transform, [0.0, 0.0, 1.0]])
            M_inv = np.linalg.inv(M_3x3)[:2, :]
        elif transform.shape == (3, 3):
            M_inv = np.linalg.inv(transform)[:2, :]
        else:
            raise ValueError(f"Invalid transform matrix shape: {transform.shape}")
    else:
        raise TypeError("Transform must be RegistrationTransform or numpy array.")

    # 1. Coordinate grid of output shape (Image A coordinate system)
    y_indices, x_indices = np.indices((out_h, out_w), dtype=np.float64)
    ones = np.ones_like(x_indices)

    # 2. Map coordinates back to source Image B: [x_B, y_B]^T = M_inv * [x_A, y_A, 1]^T
    x_b = M_inv[0, 0] * x_indices + M_inv[0, 1] * y_indices + M_inv[0, 2] * ones
    y_b = M_inv[1, 0] * x_indices + M_inv[1, 1] * y_indices + M_inv[1, 2] * ones

    # 3. Bilinear / Bicubic interpolation
    coordinates = [y_b, x_b]
    clean_input = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    warped = map_coordinates(
        clean_input,
        coordinates,
        order=order,
        mode="constant",
        cval=fill_value,
        prefilter=(order > 1),
    )

    return warped.astype(np.float32)
