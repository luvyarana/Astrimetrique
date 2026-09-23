from astrimetrique.core.gaia_catalog import GaiaStar, find_nearest_gaia_star


def test_find_nearest_gaia_star():
    stars = [
        GaiaStar(source_id="Gaia-1", ra_deg=180.0, dec_deg=20.0, phot_g_mean_mag=14.0),
        GaiaStar(source_id="Gaia-2", ra_deg=180.001, dec_deg=20.001, phot_g_mean_mag=15.5),
        GaiaStar(source_id="Gaia-3", ra_deg=181.0, dec_deg=21.0, phot_g_mean_mag=12.0),
    ]

    # Target close to Gaia-2
    nearest = find_nearest_gaia_star(180.00105, 20.00105, stars, max_separation_arcsec=10.0)
    assert nearest is not None
    assert nearest.source_id == "Gaia-2"
    assert nearest.separation_arcsec < 5.0

    # Target far away
    far_match = find_nearest_gaia_star(190.0, 30.0, stars, max_separation_arcsec=10.0)
    assert far_match is None
