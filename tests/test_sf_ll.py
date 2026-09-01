import cupy as cp
import cupy_xarray
import fluidsf
import numpy as np
import xarray as xr

from spec_flux.sf_ln import sf_ln_dir_xr, sf_ln_nd, sf_ln_xr
from spec_flux.utils import Axis


def test_minimal() -> None:
    """The structure function of a uniform array must be 0"""
    u = np.ones(shape=(3, 3, 3))
    x = np.array([0, 1, 2])
    dims = ("z_aac", "y_aca", "x_caa")
    vels = xr.Dataset(
        {"u": (dims, u), "v": (dims, u), "w": (dims, u)}, dict.fromkeys(dims, x)
    )
    out = sf_ln_xr(vels, 2)
    assert np.all(out.data == 0)


def test_rectangle() -> None:
    """Test that we can calculate the structure function of a non-cubic array"""
    u = np.ones(shape=(5, 4, 3))
    x = np.arange(3)
    y = np.arange(4)
    z = np.arange(5)
    dims = ("z_aac", "y_aca", "x_caa")
    vels = xr.Dataset(
        {"u": (dims, u), "v": (dims, u), "w": (dims, u)},
        coords={"z_aac": z, "y_aca": y, "x_caa": x},
    )
    out = sf_ln_xr(vels, 2)
    assert np.all(out.data == 0)


def test_minimal_cp() -> None:
    """Repeat test_minimal with a cupy array"""
    u = np.ones(shape=(3, 3, 3))
    x = np.array([0, 1, 2])
    dims = ("z_aac", "y_aca", "x_caa")
    vels = xr.Dataset(
        {"u": (dims, u), "v": (dims, u), "w": (dims, u)}, dict.fromkeys(dims, x)
    ).cupy.as_cupy()
    out = sf_ln_xr(vels, 2)
    assert np.all(out.data == 0)


def test_rectangle_cp() -> None:
    u = np.ones(shape=(5, 4, 3))
    x = np.arange(3)
    y = np.arange(4)
    z = np.arange(5)
    dims = ("z_aac", "y_aca", "x_caa")
    vels = xr.Dataset(
        {"u": (dims, u), "v": (dims, u), "w": (dims, u)},
        coords={"z_aac": z, "y_aca": y, "x_caa": x},
    ).cupy.as_cupy()
    out = sf_ln_xr(vels, 2)
    assert np.all(out.data == 0)


def random_data() -> xr.Dataset:
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
    return ds


def test_fluidsf_comp() -> None:
    """Structure function results along only one axis should match fluidsf"""
    ds = random_data()

    fsf = fluidsf.generate_structure_functions_3d(
        ds.u.data,
        ds.v.data,
        ds.w.data,
        ds.x_caa.data,
        ds.y_aca.data,
        ds.z_aac.data,
        ["LL"],
    )
    full_sf = sf_ln_xr(ds, 2)
    np.testing.assert_allclose(full_sf.data[0, 0, :], fsf["SF_LL_x"])
    np.testing.assert_allclose(full_sf.data[0, :, 0], fsf["SF_LL_y"])
    np.testing.assert_allclose(full_sf.data[:, 0, 0], fsf["SF_LL_z"])


def test_fluidsf_xr_dir() -> None:
    """Structure function results along only one axis should match fluidsf"""
    ds = random_data()

    fsf = fluidsf.generate_structure_functions_3d(
        ds.u.data,
        ds.v.data,
        ds.w.data,
        ds.x_caa.data,
        ds.y_aca.data,
        ds.z_aac.data,
        ["LL"],
    )
    np.testing.assert_allclose(sf_ln_dir_xr(ds, Axis.x, 2).data, fsf["SF_LL_x"])
    np.testing.assert_allclose(sf_ln_dir_xr(ds, Axis.y, 2).data, fsf["SF_LL_y"])
    np.testing.assert_allclose(sf_ln_dir_xr(ds, Axis.z, 2).data, fsf["SF_LL_z"])


def test_cupy_comp() -> None:
    """Cupy and numpy calculations should agree"""
    ds = random_data()

    np_sf = sf_ln_xr(ds, 2)
    cp_sf = sf_ln_xr(ds.cupy.as_cupy(), 2)

    np.testing.assert_allclose(np_sf.data, cp_sf.as_numpy().data)


def test_cupy_comp_dir() -> None:
    """Cupy and numpy calculations along one axis should agree"""
    ds = random_data()

    np_sf_x = sf_ln_dir_xr(ds, Axis.x, 2)
    cp_sf_x = sf_ln_dir_xr(ds.as_cupy(), Axis.x, 2)

    np.testing.assert_allclose(np_sf_x.data, cp_sf_x.as_numpy().data)

    np_sf_y = sf_ln_dir_xr(ds, Axis.y, 2)
    cp_sf_y = sf_ln_dir_xr(ds.as_cupy(), Axis.y, 2)

    np.testing.assert_allclose(np_sf_y.data, cp_sf_y.as_numpy().data)

    np_sf_z = sf_ln_dir_xr(ds, Axis.z, 2)
    cp_sf_z = sf_ln_dir_xr(ds.as_cupy(), Axis.z, 2)

    np.testing.assert_allclose(np_sf_z.data, cp_sf_z.as_numpy().data)


def test_nd_comp() -> None:
    ds = random_data()

    np.testing.assert_allclose(sf_ln_nd(ds, 2).data, sf_ln_xr(ds, 2).data)
