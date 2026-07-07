import sys

import cupy as cp
import cupy_xarray
import fluidsf
import numpy as np
import xarray as xr

sys.path.append("..")
from numerics.sf_au import Axis, sf_au, sf_au_dir, sf_au_dir_xr, sf_au_nd, sf_au_xr


def test_minimal() -> None:
    """The structure function of a uniform array must be 0"""
    u = np.ones(shape=(3, 3, 3))
    x = np.array([0, 1, 2])
    out = sf_au(u, u, u, x, x, x)
    assert np.all(out[1] == 0)


def test_rectangle() -> None:
    """Test that we can calculate the structure function of a non-cubic array"""
    u = np.ones(shape=(5, 4, 3))
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2, 3])
    z = np.array([0, 1, 2, 3, 4])
    out = sf_au(u, u, u, x, y, z)
    assert np.all(out[1] == 0)


def test_minimal_cp() -> None:
    """Repeat test_minimal with a cupy array"""
    u = cp.ones(shape=(3, 3, 3))
    x = cp.array([0, 1, 2])
    out = sf_au(u, u, u, x, x, x)
    assert np.all(out[1] == 0)


def test_rectangle_cp() -> None:
    """Repeat test_rectangle with a cupy array"""
    u = cp.ones(shape=(5, 4, 3))
    x = cp.array([0, 1, 2])
    y = cp.array([0, 1, 2, 3])
    z = cp.array([0, 1, 2, 3, 4])
    out = sf_au(u, u, u, x, y, z)
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

    full_sf = sf_au(u, v, w, x, y, z)
    fsf = fluidsf.generate_structure_functions_3d(u, v, w, x, y, z, ["ASF_V"])
    np.testing.assert_allclose(full_sf[1][0, 0, :], fsf["SF_advection_velocity_x"])
    np.testing.assert_allclose(full_sf[1][0, :, 0], fsf["SF_advection_velocity_y"])
    np.testing.assert_allclose(full_sf[1][:, 0, 0], fsf["SF_advection_velocity_z"])


def test_fluidsf_comp_dir() -> None:
    """Structure function results along only one axis should match fluidsf"""
    rng = np.random.default_rng(31415)
    u = rng.normal(size=(10, 11, 12))
    v = rng.normal(size=(10, 11, 12))
    w = rng.normal(size=(10, 11, 12))
    x = np.arange(12)
    y = np.arange(11)
    z = np.arange(10)

    fsf = fluidsf.generate_structure_functions_3d(u, v, w, x, y, z, ["ASF_V"])
    np.testing.assert_allclose(
        sf_au_dir(u, v, w, x, y, z, Axis.x)[1], fsf["SF_advection_velocity_x"]
    )
    np.testing.assert_allclose(
        sf_au_dir(u, v, w, x, y, z, Axis.y)[1], fsf["SF_advection_velocity_y"]
    )
    np.testing.assert_allclose(
        sf_au_dir(u, v, w, x, y, z, Axis.z)[1], fsf["SF_advection_velocity_z"]
    )


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

    fsf = fluidsf.generate_structure_functions_3d(u, v, w, x, y, z, ["ASF_V"])
    np.testing.assert_allclose(
        sf_au_dir_xr(ds, Axis.x).data, fsf["SF_advection_velocity_x"]
    )
    np.testing.assert_allclose(
        sf_au_dir_xr(ds, Axis.y).data, fsf["SF_advection_velocity_y"]
    )
    np.testing.assert_allclose(
        sf_au_dir_xr(ds, Axis.z).data, fsf["SF_advection_velocity_z"]
    )


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

    fsf = fluidsf.generate_structure_functions_3d(u, v, w, x, y, z, ["ASF_V"])
    np.testing.assert_allclose(
        sf_au_dir_xr(ds, Axis.x).data.get(), fsf["SF_advection_velocity_x"]
    )
    np.testing.assert_allclose(
        sf_au_dir_xr(ds, Axis.y).data.get(), fsf["SF_advection_velocity_y"]
    )
    np.testing.assert_allclose(
        sf_au_dir_xr(ds, Axis.z).data.get(), fsf["SF_advection_velocity_z"]
    )


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

    np_sf = sf_au(u, v, w, x, y, z)
    cp_sf = sf_au(cu, cv, cw, cx, cy, cz)

    np.testing.assert_allclose(np_sf[1], cp.asnumpy(cp_sf[1]))


def test_cupy_comp_dir() -> None:
    """Cupy and numpy calculations along one axis should agree"""
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

    np_sf_x = sf_au_dir(u, v, w, x, y, z, Axis.x)
    cp_sf_x = sf_au_dir(cu, cv, cw, cx, cy, cz, Axis.x)

    np.testing.assert_allclose(np_sf_x[1], cp.asnumpy(cp_sf_x[1]))

    np_sf_y = sf_au_dir(u, v, w, x, y, z, Axis.y)
    cp_sf_y = sf_au_dir(cu, cv, cw, cx, cy, cz, Axis.y)

    np.testing.assert_allclose(np_sf_y[1], cp.asnumpy(cp_sf_y[1]))

    np_sf_z = sf_au_dir(u, v, w, x, y, z, Axis.z)
    cp_sf_z = sf_au_dir(cu, cv, cw, cx, cy, cz, Axis.z)

    np.testing.assert_allclose(np_sf_z[1], cp.asnumpy(cp_sf_z[1]))


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

    np.testing.assert_allclose(sf_au_xr(ds).data[0], sf_au(u, v, w, x, y, z)[1])


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

    np.testing.assert_allclose(sf_au_nd(ds).data, sf_au_xr(ds).data)
