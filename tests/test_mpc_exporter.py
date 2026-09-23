from datetime import datetime, timezone
from astrimetrique.core.mpc_exporter import (
    MPCObservation,
    MPCObservatoryHeader,
    format_mpc_80_col,
    generate_mpc_report,
)


def test_format_mpc_80_col_length_and_structure():
    obs = MPCObservation(
        designation="2024 AB",
        utc_time=datetime(2026, 9, 22, 21, 30, 0, tzinfo=timezone.utc),
        ra_deg=180.25417,  # 12 01 01.00
        dec_deg=25.50833,   # +25 30 30.0
        mag=15.2,
        band="R",
        obs_code="G96",
        is_discovery=True,
    )

    line = format_mpc_80_col(obs)

    # 1. Exact 80 characters
    assert len(line) == 80, f"Line length is {len(line)}, expected 80: '{line}'"

    # 2. Check discovery flag at index 12 (1-indexed col 13)
    assert line[12] == "*"

    # 3. Check observatory code in cols 78-80
    assert line[77:80] == "G96"

    # 4. Check magnitude and band in cols 66-71
    assert "15.2 R" in line[65:71]

    # 5. Check date format in line
    assert "2026 09 22." in line


def test_generate_mpc_report():
    obs1 = MPCObservation(
        designation="99942",
        utc_time=datetime(2026, 9, 22, 20, 0, 0, tzinfo=timezone.utc),
        ra_deg=100.0,
        dec_deg=-15.0,
        mag=14.1,
        band="V",
        obs_code="500",
    )
    header = MPCObservatoryHeader(cod="500", con="Dr. Jane Doe")
    report = generate_mpc_report([obs1], header)

    assert "COD 500" in report
    assert "CON Dr. Jane Doe" in report
    assert "--- 80-column observations ---" in report
    lines = [l for l in report.split("\n") if l and not l.startswith("COD") and not l.startswith("CON") and not l.startswith("OBS") and not l.startswith("MEA") and not l.startswith("TEL") and not l.startswith("NET") and not l.startswith("ACK") and not l.startswith("---")]
    assert len(lines) == 1
    assert len(lines[0]) == 80
