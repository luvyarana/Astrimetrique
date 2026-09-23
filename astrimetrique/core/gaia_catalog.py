"""
Gaia DR3 Catalog Query Engine via astroquery.
Enables automated coordinate fetching and reference star matching from Gaia DR3.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import astropy.units as u
from astropy.coordinates import SkyCoord
from astrimetrique.utils.coordinates import angular_distance_arcsec


@dataclass
class GaiaStar:
    """Catalog star record from Gaia DR3."""
    source_id: str
    ra_deg: float
    dec_deg: float
    phot_g_mean_mag: Optional[float] = None
    pmra: Optional[float] = None
    pmdec: Optional[float] = None
    separation_arcsec: Optional[float] = None


def query_gaia_cone(
    ra_center_deg: float,
    dec_center_deg: float,
    radius_arcmin: float = 15.0,
    mag_limit: float = 17.5,
    row_limit: int = 150,
) -> List[GaiaStar]:
    """
    Query Gaia DR3 stars around a given celestial coordinate.
    Uses astroquery.gaia with fallback to astroquery.vizier.
    """
    coord = SkyCoord(ra=ra_center_deg * u.deg, dec=dec_center_deg * u.deg, frame="icrs")
    stars: List[GaiaStar] = []

    # Attempt 1: astroquery.gaia (ESA Gaia Archive TAP)
    try:
        from astroquery.gaia import Gaia
        
        # Limit noise from TAP queries
        Gaia.ROW_LIMIT = row_limit
        query_str = f"""
        SELECT TOP {row_limit} source_id, ra, dec, phot_g_mean_mag, pmra, pmdec
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_center_deg:.6f}, {dec_center_deg:.6f}, {radius_arcmin / 60.0:.6f})
        )
        AND phot_g_mean_mag <= {mag_limit:.2f}
        ORDER BY phot_g_mean_mag ASC
        """
        job = Gaia.launch_job_async(query_str, verbose=False)
        table = job.get_results()

        for row in table:
            source_id = str(row["source_id"])
            ra = float(row["ra"])
            dec = float(row["dec"])
            g_mag = float(row["phot_g_mean_mag"]) if "phot_g_mean_mag" in row.colnames and not row.mask["phot_g_mean_mag"] else None
            pmra = float(row["pmra"]) if "pmra" in row.colnames and not row.mask["pmra"] else None
            pmdec = float(row["pmdec"]) if "pmdec" in row.colnames and not row.mask["pmdec"] else None
            sep = angular_distance_arcsec(ra_center_deg, dec_center_deg, ra, dec)

            stars.append(GaiaStar(
                source_id=f"Gaia DR3 {source_id}",
                ra_deg=ra,
                dec_deg=dec,
                phot_g_mean_mag=g_mag,
                pmra=pmra,
                pmdec=pmdec,
                separation_arcsec=sep,
            ))
        if len(stars) > 0:
            return stars

    except Exception:
        pass

    # Attempt 2: VizieR Cone Search fallback for Gaia DR3 (I/355/gaiadr3)
    try:
        from astroquery.vizier import Vizier
        v = Vizier(
            columns=["Source", "RA_ICRS", "DE_ICRS", "Gmag", "pmRA", "pmDE"],
            column_filters={"Gmag": f"<{mag_limit}"},
            row_limit=row_limit,
        )
        results = v.query_region(coord, radius=radius_arcmin * u.arcmin, catalog="I/355/gaiadr3")
        if results and len(results) > 0:
            tbl = results[0]
            for row in tbl:
                source_id = str(row.get("Source", ""))
                ra = float(row["RA_ICRS"])
                dec = float(row["DE_ICRS"])
                g_mag = float(row["Gmag"]) if "Gmag" in row.colnames else None
                pmra = float(row["pmRA"]) if "pmRA" in row.colnames else None
                pmdec = float(row["pmDE"]) if "pmDE" in row.colnames else None
                sep = angular_distance_arcsec(ra_center_deg, dec_center_deg, ra, dec)

                stars.append(GaiaStar(
                    source_id=f"Gaia DR3 {source_id}" if source_id else "Gaia Star",
                    ra_deg=ra,
                    dec_deg=dec,
                    phot_g_mean_mag=g_mag,
                    pmra=pmra,
                    pmdec=pmdec,
                    separation_arcsec=sep,
                ))
            return stars
    except Exception:
        pass

    return stars


def find_nearest_gaia_star(
    target_ra_deg: float,
    target_dec_deg: float,
    catalog_stars: List[GaiaStar],
    max_separation_arcsec: float = 10.0,
) -> Optional[GaiaStar]:
    """
    Find the closest Gaia star from a catalog list within max_separation_arcsec.
    """
    if not catalog_stars:
        return None

    best_star = None
    min_sep = float("inf")

    for star in catalog_stars:
        sep = angular_distance_arcsec(target_ra_deg, target_dec_deg, star.ra_deg, star.dec_deg)
        if sep < min_sep:
            min_sep = sep
            best_star = star

    if best_star and min_sep <= max_separation_arcsec:
        # Return a copy with exact separation relative to target
        return GaiaStar(
            source_id=best_star.source_id,
            ra_deg=best_star.ra_deg,
            dec_deg=best_star.dec_deg,
            phot_g_mean_mag=best_star.phot_g_mean_mag,
            pmra=best_star.pmra,
            pmdec=best_star.pmdec,
            separation_arcsec=min_sep,
        )
    return None
