import sys

import cupy as cp
import cupy_xarray
import fluidsf
import numpy as np
import xarray as xr

sys.path.append("..")
from numerics.sf_ln import sf_ln, sf_ln_dir, sf_ln_dir_xr, sf_ln_nd, sf_ln_xr
from numerics.utils import Axis


def test_minimal() -> None:
    """The structure function of a uniform array must be 0"""
    u = np.ones(shape=(3, 3, 3))
    x = np.array([0, 1, 2])
    out = sf_ln(u, u, u, x, x, x, 2)
    assert np.all(out[1] == 0)


def test_rectangle() -> None:
    """Test that we can calculate the structure function of a non-cubic array"""
    u = np.ones(shape=(5, 4, 3))
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2, 3])
    z = np.array([0, 1, 2, 3, 4])
    out = sf_ln(u, u, u, x, y, z, 2)
    assert np.all(out[1] == 0)


def test_minimal_cp() -> None:
    """Repeat test_minimal with a cupy array"""
    u = cp.ones(shape=(3, 3, 3))
    x = cp.array([0, 1, 2])
    out = sf_ln(u, u, u, x, x, x, 2)
    assert np.all(out[1] == 0)


def test_rectangle_cp() -> None:
    """Repeat test_rectangle with a cupy array"""
    u = cp.ones(shape=(5, 4, 3))
    x = cp.array([0, 1, 2])
    y = cp.array([0, 1, 2, 3])
    z = cp.array([0, 1, 2, 3, 4])
    out = sf_ln(u, u, u, x, y, z, 2)
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

    full_sf = sf_ln(u, v, w, x, y, z, 2)
    fsf = fluidsf.generate_structure_functions_3d(u, v, w, x, y, z, ["LL"])
    assert np.isclose(full_sf[1][0, 0, :], fsf["SF_LL_x"]).all()
    assert np.isclose(full_sf[1][0, :, 0], fsf["SF_LL_y"]).all()
    assert np.isclose(full_sf[1][:, 0, 0], fsf["SF_LL_z"]).all()


def test_fluidsf_comp_dir() -> None:
    """Structure function results along only one axis should match fluidsf"""
    rng = np.random.default_rng(31415)
    u = rng.normal(size=(10, 11, 12))
    v = rng.normal(size=(10, 11, 12))
    w = rng.normal(size=(10, 11, 12))
    x = np.arange(12)
    y = np.arange(11)
    z = np.arange(10)

    full_sf = (
        sf_ln_dir(u, v, w, x, y, z, Axis.x, 2),
        sf_ln_dir(u, v, w, x, y, z, Axis.y, 2),
        sf_ln_dir(u, v, w, x, y, z, Axis.z, 2),
    )
    fsf = fluidsf.generate_structure_functions_3d(u, v, w, x, y, z, ["LL"])
    assert np.isclose(full_sf[0][1], fsf["SF_LL_x"]).all()
    assert np.isclose(full_sf[1][1], fsf["SF_LL_y"]).all()
    assert np.isclose(full_sf[2][1], fsf["SF_LL_z"]).all()


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

    np_sf = sf_ln(u, v, w, x, y, z, 2)
    cp_sf = sf_ln(cu, cv, cw, cx, cy, cz, 2)

    assert np.isclose(np_sf[1], cp.asnumpy(cp_sf[1])).all()
    assert np.isclose(np_sf[0][0], cp.asnumpy(cp_sf[0][0])).all()
    assert np.isclose(np_sf[0][1], cp.asnumpy(cp_sf[0][1])).all()
    assert np.isclose(np_sf[0][2], cp.asnumpy(cp_sf[0][2])).all()


def test_cupy_comp_dir() -> None:
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

    np_sf = (
        sf_ln_dir(u, v, w, x, y, z, Axis.x, 2),
        sf_ln_dir(u, v, w, x, y, z, Axis.y, 2),
        sf_ln_dir(u, v, w, x, y, z, Axis.z, 2),
    )
    cp_sf = (
        sf_ln_dir(cu, cv, cw, cx, cy, cz, Axis.x, 2),
        sf_ln_dir(cu, cv, cw, cx, cy, cz, Axis.y, 2),
        sf_ln_dir(cu, cv, cw, cx, cy, cz, Axis.z, 2),
    )

    for i in range(3):
        for j in range(2):
            assert np.isclose(np_sf[i][j], cp.asnumpy(cp_sf[i][j])).all()


def test_fluidsf_xr_dir() -> None:
    """Structure function results along only one axis should match fluidsf"""
    rng = np.random.default_rng(31415)
    u = rng.normal(size=(10, 11, 12))
    v = rng.normal(size=(10, 11, 12))
    w = rng.normal(size=(10, 11, 12))
    x = np.arange(12)
    y = np.arange(11)
    z = np.arange(10)

    dims = ("z_aac", "y_aca", "x_caa")

    ds = xr.Dataset(
        {"u": (dims, u), "v": (dims, v), "w": (dims, w)},
        coords={"z_aac": z, "y_aca": y, "x_caa": x},
    )

    fsf = fluidsf.generate_structure_functions_3d(u, v, w, x, y, z, ["LL"])
    np.testing.assert_allclose(sf_ln_dir_xr(ds, Axis.x, 2).data, fsf["SF_LL_x"])
    np.testing.assert_allclose(sf_ln_dir_xr(ds, Axis.y, 2).data, fsf["SF_LL_y"])
    np.testing.assert_allclose(sf_ln_dir_xr(ds, Axis.z, 2).data, fsf["SF_LL_z"])


def test_fluidsf_xr_dir_cp() -> None:
    """Structure function results along only one axis should match fluidsf"""
    rng = np.random.default_rng(31415)
    u = rng.normal(size=(10, 11, 12))
    v = rng.normal(size=(10, 11, 12))
    w = rng.normal(size=(10, 11, 12))
    x = np.arange(12)
    y = np.arange(11)
    z = np.arange(10)

    dims = ("z_aac", "y_aca", "x_caa")

    ds = xr.Dataset(
        {"u": (dims, u), "v": (dims, v), "w": (dims, w)},
        coords={"z_aac": z, "y_aca": y, "x_caa": x},
    ).as_cupy()

    fsf = fluidsf.generate_structure_functions_3d(u, v, w, x, y, z, ["LL"])
    np.testing.assert_allclose(sf_ln_dir_xr(ds, Axis.x, 2).data.get(), fsf["SF_LL_x"])
    np.testing.assert_allclose(sf_ln_dir_xr(ds, Axis.y, 2).data.get(), fsf["SF_LL_y"])
    np.testing.assert_allclose(sf_ln_dir_xr(ds, Axis.z, 2).data.get(), fsf["SF_LL_z"])


def test_xr_comp() -> None:
    rng = np.random.default_rng(31415)
    u = rng.normal(size=(10, 11, 12))
    v = rng.normal(size=(10, 11, 12))
    w = rng.normal(size=(10, 11, 12))
    x = np.arange(12)
    y = np.arange(11)
    z = np.arange(10)

    dims = ("z_aac", "y_aca", "x_caa")

    ds = xr.Dataset(
        {"u": (dims, u), "v": (dims, v), "w": (dims, w)},
        coords={"z_aac": z, "y_aca": y, "x_caa": x},
    )

    np.testing.assert_allclose(sf_ln_xr(ds, 2).data, sf_ln(u, v, w, x, y, z, 2)[1])


def test_nd_comp() -> None:
    rng = np.random.default_rng(31415)
    u = rng.normal(size=(10, 11, 12))
    v = rng.normal(size=(10, 11, 12))
    w = rng.normal(size=(10, 11, 12))
    x = np.arange(12)
    y = np.arange(11)
    z = np.arange(10)

    dims = ("z_aac", "y_aca", "x_caa")

    ds = xr.Dataset(
        {"u": (dims, u), "v": (dims, v), "w": (dims, w)},
        coords={"z_aac": z, "y_aca": y, "x_caa": x},
    )

    np.testing.assert_allclose(sf_ln_nd(ds, 2).data, sf_ln_xr(ds, 2).data)
