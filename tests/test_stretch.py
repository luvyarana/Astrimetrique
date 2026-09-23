import numpy as np
from astrimetrique.core.stretch import StretchMode, apply_stretch, calculate_zscale_limits


def test_calculate_zscale_limits():
    # Normal distributed data with a few bright stars
    rng = np.random.default_rng(123)
    data = rng.normal(300.0, 10.0, size=(100, 100)).astype(np.float32)
    # Add star
    data[50, 50] = 5000.0

    zmin, zmax = calculate_zscale_limits(data)
    assert zmin < zmax
    assert 250.0 < zmin < 350.0
    assert zmax < 1000.0  # ZScale clips extreme outliers effectively


def test_apply_stretch_modes():
    data = np.linspace(0.0, 1000.0, 10000, dtype=np.float32).reshape((100, 100))

    for mode in StretchMode:
        res = apply_stretch(data, mode=mode)
        assert res.shape == (100, 100)
        assert res.dtype == np.uint8
        assert res.min() >= 0
        assert res.max() <= 255


def test_apply_stretch_invert():
    data = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape((10, 10))
    normal = apply_stretch(data, mode=StretchMode.LINEAR, vmin=0.0, vmax=1.0, invert=False)
    inverted = apply_stretch(data, mode=StretchMode.LINEAR, vmin=0.0, vmax=1.0, invert=True)

    assert normal[0, 0] == 0
    assert inverted[0, 0] == 255
    assert normal[-1, -1] == 255
    assert inverted[-1, -1] == 0
