from .fits_io import FITSImage, FITSMetadata, load_fits
from .stretch import StretchMode, apply_stretch, calculate_zscale_limits
from .centroid import CentroidResult, fit_centroid_2d_gaussian
from .plate_solver import LeastSquaresPlateSolver, PlateSolution, ReferenceStar
from .gaia_catalog import GaiaStar, query_gaia_cone, find_nearest_gaia_star
from .mpc_exporter import MPCObservation, MPCObservatoryHeader, format_mpc_80_col, generate_mpc_report
from .synthetic_data import generate_synthetic_fits_field, generate_multiepoch_synthetic_pair
from .registration import (
    RegistrationTransform,
    compute_affine_from_points,
    compute_registration_from_solutions,
    compute_registration_from_wcs,
    align_images_pipeline,
    warp_image,
)
from .diff_engine import (
    MovingCandidate,
    DiffResult,
    match_background_and_gain,
    compute_difference_map,
)

__all__ = [
    "FITSImage",
    "FITSMetadata",
    "load_fits",
    "StretchMode",
    "apply_stretch",
    "calculate_zscale_limits",
    "CentroidResult",
    "fit_centroid_2d_gaussian",
    "LeastSquaresPlateSolver",
    "PlateSolution",
    "ReferenceStar",
    "GaiaStar",
    "query_gaia_cone",
    "find_nearest_gaia_star",
    "MPCObservation",
    "MPCObservatoryHeader",
    "format_mpc_80_col",
    "generate_mpc_report",
    "generate_synthetic_fits_field",
    "generate_multiepoch_synthetic_pair",
    "RegistrationTransform",
    "compute_affine_from_points",
    "compute_registration_from_solutions",
    "warp_image",
    "MovingCandidate",
    "DiffResult",
    "match_background_and_gain",
    "compute_difference_map",
]
