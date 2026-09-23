import numpy as np
from pathlib import Path
from astropy.io import fits
from astrimetrique.core.fits_io import load_fits, FITSImage, FITSMetadata
from astrimetrique.core.synthetic_data import generate_synthetic_fits_field


def test_fits_synthetic_generation(tmp_path):
    fits_img, ref_stars, target_star = generate_synthetic_fits_field(
        width=512,
        height=512,
        num_stars=8,
        include_asteroid=True,
    )
    assert isinstance(fits_img, FITSImage)
    assert fits_img.shape == (512, 512)
    assert len(ref_stars) == 8
    assert target_star is not None

    # Write to temp FITS file and reload
    fits_path = tmp_path / "test_synth.fits"
    hdu = fits.PrimaryHDU(data=fits_img.data)
    for k, v in fits_img.metadata.headers.items():
        if k not in ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND"]:
            try:
                hdu.header[k] = v
            except Exception:
                pass
    hdu.writeto(fits_path, overwrite=True)

    loaded_img = load_fits(fits_path)
    assert loaded_img.shape == (512, 512)
    assert loaded_img.metadata.object_name == fits_img.metadata.object_name
    assert loaded_img.get_pixel_value(10, 10) > 0.0
