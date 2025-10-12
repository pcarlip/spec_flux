import sys

import cupy as cp
import fluidsf
import numpy as np

sys.path.append("..")
from numerics.sf_ln import sf_ln


def test_minimal() -> None:
    """The structure function of a uniform array must be 0"""
    u = np.ones(shape=(3, 3, 3))
    x = np.array([0, 1, 2])
    out = sf_ln(u, u, u, x, x, x)
    assert np.all(out[1] == 0)


def test_rectangle() -> None:
    """Test that we can calculate the structure function of a non-cubic array"""
    u = np.ones(shape=(5, 4, 3))
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2, 3])
    z = np.array([0, 1, 2, 3, 4])
    out = sf_ln(u, u, u, x, y, z)
    assert np.all(out[1] == 0)


def test_minimal_cp() -> None:
    """Repeat test_minimal with a cupy array"""
    u = cp.ones(shape=(3, 3, 3))
    x = cp.array([0, 1, 2])
    out = sf_ln(u, u, u, x, x, x)
    assert np.all(out[1] == 0)


def test_rectangle_cp() -> None:
    """Repeat test_rectangle with a cupy array"""
    u = cp.ones(shape=(5, 4, 3))
    x = cp.array([0, 1, 2])
    y = cp.array([0, 1, 2, 3])
    z = cp.array([0, 1, 2, 3, 4])
    out = sf_ln(u, u, u, x, y, z)
    assert np.all(out[1] == 0)


def test_fluidsf_comp() -> None:
    """Structure function results along only one axis should match fluidsf"""
    rng = np.random.default_rng(31415)
    u = rng.normal(size=(10, 11, 12))
    v = rng.normal(size=(10, 11, 12))
    w = rng.normal(size=(10, 11, 12))
    x = np.arange(12)
    y = np.arange(11)
    z = np.arange(10)

    full_sf = sf_ln(u, v, w, x, y, z)
    fsf = fluidsf.generate_structure_functions_3d(u, v, w, x, y, z, ["LLL"])
    assert np.isclose(full_sf[1][0, 0, :], fsf["SF_LLL_x"]).all()
    assert np.isclose(full_sf[1][0, :, 0], fsf["SF_LLL_y"]).all()
    assert np.isclose(full_sf[1][:, 0, 0], fsf["SF_LLL_z"]).all()


def test_cupy_comp() -> None:
    """Cupy and numpy calculations should agree"""
    rng = np.random.default_rng(31415)
    u = rng.normal(size=(10, 11, 12))
    v = rng.normal(size=(10, 11, 12))
    w = rng.normal(size=(10, 11, 12))
    x = np.arange(12)
    y = np.arange(11)
    z = np.arange(10)

    cu = cp.array(u)
    cv = cp.array(v)
    cw = cp.array(w)
    cx = cp.array(x)
    cy = cp.array(y)
    cz = cp.array(z)

    np_sf = sf_ln(u, v, w, x, y, z)
    cp_sf = sf_ln(cu, cv, cw, cx, cy, cz)

    assert np.isclose(np_sf[1], cp.asnumpy(cp_sf[1])).all()
