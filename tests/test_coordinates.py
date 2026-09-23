import pytest
from datetime import datetime, timezone
import numpy as np
from astrimetrique.utils.coordinates import (
    deg_to_ra_hms,
    deg_to_dec_dms,
    format_ra,
    format_dec,
    parse_ra,
    parse_dec,
    datetime_to_fractional_day,
    angular_distance_arcsec,
)


def test_deg_to_ra_hms():
    # 0 deg = 00h 00m 00s
    h, m, s = deg_to_ra_hms(0.0)
    assert h == 0 and m == 0 and abs(s - 0.0) < 1e-6

    # 180 deg = 12h 00m 00s
    h, m, s = deg_to_ra_hms(180.0)
    assert h == 12 and m == 0 and abs(s - 0.0) < 1e-6

    # 15.5 deg = 01h 02m 00s
    h, m, s = deg_to_ra_hms(15.5)
    assert h == 1 and m == 2 and abs(s - 0.0) < 1e-6


def test_deg_to_dec_dms():
    # +25.5 deg = +25° 30' 00"
    sign, d, m, s = deg_to_dec_dms(25.5)
    assert sign == "+" and d == 25 and m == 30 and abs(s - 0.0) < 1e-6

    # -10.25 deg = -10° 15' 00"
    sign, d, m, s = deg_to_dec_dms(-10.25)
    assert sign == "-" and d == 10 and m == 15 and abs(s - 0.0) < 1e-6


def test_format_and_parse_ra():
    ra_deg = 185.3421
    formatted = format_ra(ra_deg)
    parsed = parse_ra(formatted)
    assert abs(parsed - ra_deg) < 0.01

    # Decimal format parsing
    assert abs(parse_ra("185.3421") - 185.3421) < 1e-6

    # Colon format parsing
    parsed_colon = parse_ra("12:21:22.10")
    assert abs(parsed_colon - (12 + 21/60 + 22.10/3600)*15) < 1e-5


def test_format_and_parse_dec():
    dec_deg = -32.654
    formatted = format_dec(dec_deg)
    parsed = parse_dec(formatted)
    assert abs(parsed - dec_deg) < 0.01

    # Decimal format
    assert abs(parse_dec("-32.654") - (-32.654)) < 1e-6


def test_datetime_to_fractional_day():
    dt = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)
    res = datetime_to_fractional_day(dt)
    assert res == "2026 09 22.50000"

    dt2 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    res2 = datetime_to_fractional_day(dt2)
    assert res2 == "2024 01 01.00000"


def test_angular_distance_arcsec():
    # Same point
    assert angular_distance_arcsec(100.0, 20.0, 100.0, 20.0) < 1e-6
    # 1 degree apart along declination = 3600 arcsec
    sep = angular_distance_arcsec(0.0, 0.0, 0.0, 1.0)
    assert abs(sep - 3600.0) < 1e-3
