"""
Coordinate and time conversion utilities for astrometry.
"""

from datetime import datetime, timezone
from typing import Tuple
import numpy as np


def deg_to_ra_hms(ra_deg: float) -> Tuple[int, int, float]:
    """Convert RA in decimal degrees to (hours, minutes, seconds)."""
    ra_deg = ra_deg % 360.0
    hours_total = ra_deg / 15.0
    hours = int(hours_total)
    minutes_total = (hours_total - hours) * 60.0
    minutes = int(minutes_total)
    seconds = (minutes_total - minutes) * 60.0
    return hours, minutes, seconds


def deg_to_dec_dms(dec_deg: float) -> Tuple[str, int, int, float]:
    """Convert Dec in decimal degrees to (sign, degrees, arcminutes, arcseconds)."""
    sign = "+" if dec_deg >= 0 else "-"
    abs_dec = abs(dec_deg)
    degrees = int(abs_dec)
    minutes_total = (abs_dec - degrees) * 60.0
    minutes = int(minutes_total)
    seconds = (minutes_total - minutes) * 60.0
    return sign, degrees, minutes, seconds


def format_ra(ra_deg: float, precision: int = 2) -> str:
    """Format RA in degrees as 'HH MM SS.ss' string."""
    h, m, s = deg_to_ra_hms(ra_deg)
    return f"{h:02d} {m:02d} {s:0{3+precision}.{precision}f}"


def format_dec(dec_deg: float, precision: int = 1) -> str:
    """Format Dec in degrees as '+DD MM SS.s' string."""
    sign, d, m, s = deg_to_dec_dms(dec_deg)
    return f"{sign}{d:02d} {m:02d} {s:0{3+precision}.{precision}f}"


def parse_ra(ra_str: str) -> float:
    """
    Parse RA string in various formats:
    - 'HH MM SS.ss' or 'HH:MM:SS.ss' or 'HHhMMmSSs'
    - Decimal degrees as float string '123.456'
    Returns RA in decimal degrees [0, 360).
    """
    ra_str = ra_str.strip()
    # Normalize separators
    for char in [":", "h", "m", "s", "H", "M", "S"]:
        ra_str = ra_str.replace(char, " ")
    tokens = ra_str.split()

    if len(tokens) == 1:
        # Decimal degrees
        return float(tokens[0]) % 360.0
    elif len(tokens) == 3:
        h = float(tokens[0])
        m = float(tokens[1])
        s = float(tokens[2])
        return ((h + m / 60.0 + s / 3600.0) * 15.0) % 360.0
    elif len(tokens) == 2:
        h = float(tokens[0])
        m = float(tokens[1])
        return ((h + m / 60.0) * 15.0) % 360.0
    else:
        raise ValueError(f"Cannot parse RA format: {ra_str}")


def parse_dec(dec_str: str) -> float:
    """
    Parse Dec string in various formats:
    - '+DD MM SS.s' or '-DD:MM:SS.s' or '+DDdMMmSSs'
    - Decimal degrees as float string '-12.345'
    Returns Dec in decimal degrees [-90, +90].
    """
    dec_str = dec_str.strip()
    is_negative = dec_str.startswith("-")

    # Normalize separators
    clean_str = dec_str
    for char in [":", "d", "m", "s", "D", "M", "S", "+", "-"]:
        clean_str = clean_str.replace(char, " ")
    tokens = clean_str.split()

    if len(tokens) == 1:
        val = float(tokens[0])
        return -val if is_negative else val
    elif len(tokens) == 3:
        d = float(tokens[0])
        m = float(tokens[1])
        s = float(tokens[2])
        val = d + m / 60.0 + s / 3600.0
        return -val if is_negative else val
    elif len(tokens) == 2:
        d = float(tokens[0])
        m = float(tokens[1])
        val = d + m / 60.0
        return -val if is_negative else val
    else:
        raise ValueError(f"Cannot parse Dec format: {dec_str}")


def datetime_to_fractional_day(dt: datetime) -> str:
    """
    Convert a datetime to MPC fractional day format: 'YYYY MM DD.ddddd' (UTC).
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    
    seconds_in_day = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    fraction = seconds_in_day / 86400.0
    
    # Format: YYYY MM DD.ddddd
    day_fraction_str = f"{fraction:.5f}"[1:]  # e.g. ".12345"
    return f"{dt.year:04d} {dt.month:02d} {dt.day:02d}{day_fraction_str}"


def angular_distance_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """
    Calculate angular separation between two (RA, Dec) coordinates in decimal degrees.
    Returns separation in arcseconds using Haversine formula.
    """
    r1, d1 = np.radians(ra1), np.radians(dec1)
    r2, d2 = np.radians(ra2), np.radians(dec2)
    
    dlon = r2 - r1
    dlat = d2 - d1
    
    a = np.sin(dlat / 2.0)**2 + np.cos(d1) * np.cos(d2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return np.degrees(c) * 3600.0
