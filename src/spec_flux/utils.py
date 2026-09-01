from enum import Enum, StrEnum
from types import ModuleType

import cupy as cp
import cupyx
import numpy as np
import pandas as pd
import xarray as xr
from xarray_extras.interpolate import splev, splrep

type ndarray = np.ndarray | cp.ndarray  # noqa: PYI042


class GradMethod(Enum):
    precalc = 1
    spectral = 2
    numpy = 3


class Axis(Enum):
    z = 0
    y = 1
    x = 2


def axis_name(ax: Axis) -> str:
    if ax == Axis.x:
        return "x_caa"
    elif ax == Axis.y:
        return "y_aca"
    else:
        return "z_aac"


class StrAxis(StrEnum):
    x = "x"
    y = "y"
    z = "z"


class SFType(StrEnum):
    Au = "Au"
    LLL = "LLL"


def xp_fft(array: ndarray) -> tuple[ModuleType, ModuleType]:
    """Choose the correct array and fft modules for a given array

    Parameters
    ----------
    array : ndarray
        Input array to type

    Returns
    -------
    ModuleType
        numpy or cupy
    ModuleType
        scipy.fft or cupyx.scipy.fft
    """
    xp = cp.get_array_module(array)
    genfft = cupyx.scipy.get_array_module(array).fft
    return (xp, genfft)


def krange_fft(
    data: xr.Dataset, axes: tuple[str, str, str] = ("z_aac", "y_aca", "x_caa")
) -> tuple[ndarray, ndarray, ndarray]:
    """Get the fourier-space ranges of k-values from the range of z, y, and x values

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing grid axes
    axes : tuple[str, str, str], optional
        Axis names in dataset; by default ("z_aac", "y_aca", "x_caa")

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        arrays of m, l, and k values of the fourier-transformed grids
    """
    _, genfft = xp_fft(data.u)
    ranges = []
    for ax in axes:
        ds_ax = data[ax]
        Ni = len(ds_ax)
        spacing = float(ds_ax[1] - ds_ax[0])
        Li = Ni * spacing
        dk = 2 * np.pi / Li
        k_range = genfft.fftshift(genfft.fftfreq(Ni) * Ni * dk)
        if data.cupy.is_cupy:
            # not sure why I need this check given genfft, but it breaks without it
            k_range = cp.array(k_range)
        ranges.append(k_range)
    ranges = tuple(ranges)
    return ranges


def krange_int(model: xr.Dataset, n: int = 1000, log: bool = False) -> np.ndarray:
    """Generate a reasonable range of k values from a netcdf model made in Oceananigans

    Parameters
    ----------
    model : xr.DataArray
        the .nc file, read into xarray
    n : int, optional
        number of k values to include, by default 1000
    log : bool, optional
        return a logspace rather than a linspace, by default False

    Returns
    -------
    np.ndarray
        Array of k values
    """
    L = float(model["x_caa"][-1]) - float(model["x_caa"][0])
    size = len(model["x_caa"].values)
    kmin = np.pi / L
    kmax = kmin * size * 2
    if log:
        return np.logspace(np.log10(kmin), np.log10(kmax), n)
    else:
        return np.linspace(kmin, kmax, n)


def ocean_interp_per(
    oc_input: xr.Dataset,
    time: int = -1,
    periodic_axes: tuple[Axis, ...] = (Axis.x, Axis.y, Axis.z),
    advection: bool = False,
) -> xr.Dataset:
    """Interpolate Oceananigans output velocities to use the same axes,
    uses periodic interpolation

    Parameters
    ----------
    oc_input : xr.Dataset
        Oceananigans output NetCDF. Must be stored on CPU, interpolation does not work
        with cupy-xarray.
    time : int, optional
        Index of desired timestep, by default -1
    periodic_axes : tuple[Axis, ...], optional
        Tuple of axes to interpolate as periodic (rather than extrapolate),
        by default (Axis.x, Axis.y, Axis.z)
    advection : bool, optional
        Whether dataset has arrays of advection as well as velocity

    Returns
    -------
    xr.Dataset
        Dataset with u,v,w on the same set of axes (cell centers)
    """
    u_interp = splrep(oc_input["u"].isel(time=time), "x_faa")
    v_interp = splrep(oc_input["v"].isel(time=time), "y_afa")
    w_interp = splrep(oc_input["w"].isel(time=time), "z_aaf")

    x_extrap = "periodic" if Axis.x in periodic_axes else True
    y_extrap = "periodic" if Axis.y in periodic_axes else True
    z_extrap = "periodic" if Axis.z in periodic_axes else True

    uvar = (
        splev(oc_input["x_caa"], u_interp, x_extrap)
        .rename("u")
        .transpose("z_aac", "y_aca", "x_caa", transpose_coords=True)
    )
    vvar = (
        splev(oc_input["y_aca"], v_interp, y_extrap)
        .rename("v")
        .transpose("z_aac", "y_aca", "x_caa", transpose_coords=True)
    )
    wvar = splev(oc_input["z_aac"], w_interp, z_extrap).rename("w")
    if advection:
        uadv_interp = splrep(oc_input["uadv"].isel(time=time), "x_faa")
        vadv_interp = splrep(oc_input["vadv"].isel(time=time), "x_faa")
        wadv_interp = splrep(oc_input["wadv"].isel(time=time), "x_faa")
        uadvvar = (
            splev(oc_input["x_caa"], uadv_interp, x_extrap)
            .rename("uadv")
            .transpose("z_aac", "y_aca", "x_caa", transpose_coords=True)
        )
        vadvvar = (
            splev(oc_input["x_caa"], vadv_interp, x_extrap)
            .rename("vadv")
            .transpose("z_aac", "y_aca", "x_caa", transpose_coords=True)
        )
        wadvvar = (
            splev(oc_input["x_caa"], wadv_interp, x_extrap)
            .rename("wadv")
            .transpose("z_aac", "y_aca", "x_caa", transpose_coords=True)
        )
        return xr.merge(
            [uvar, vvar, wvar, uadvvar, vadvvar, wadvvar], compat="no_conflicts"
        )
    else:
        return xr.merge([uvar, vvar, wvar], compat="no_conflicts")


def sf_au_prop_xr(sf_tab: xr.Dataset) -> xr.Dataset:
    pi_au_x = (-sf_tab.SF_Au_x / 2).rename("x")
    pi_au_y = (-sf_tab.SF_Au_y / 2).rename("y")
    pi_au_z = (-sf_tab.SF_Au_z / 2).rename("z")
    return xr.merge([pi_au_x, pi_au_y, pi_au_z], compat="no_conflicts")  # type: ignore


def sf_lll_prop_xr(sf_tab: xr.Dataset) -> xr.Dataset:
    pi_lll_x = (-sf_tab.SF_LLL_x * 5 / (sf_tab.SF_LLL_x.dr * 4)).rename("x")
    pi_lll_y = (-sf_tab.SF_LLL_y * 5 / (sf_tab.SF_LLL_y.dr * 4)).rename("y")
    pi_lll_z = (-sf_tab.SF_LLL_z * 5 / (sf_tab.SF_LLL_z.dr * 4)).rename("z")
    return xr.merge([pi_lll_x, pi_lll_y, pi_lll_z], compat="no_conflicts")  # type: ignore


def sf_prop_pd(sf_tab: xr.Dataset, sf_type: SFType) -> pd.DataFrame:
    """Turn a structure function dataset into a dataframe suitable for plotting with sns

    Parameters
    ----------
    sf_tab : xr.Dataset
        Dataset of SF values
    sf_type : SFType
        Type of SF (Au or LLL, used for proportionality relations to get spectral flux)

    Returns
    -------
    pd.DataFrame
        Long df with estimated spectral flux vs separation
    """
    pi = sf_au_prop_xr(sf_tab) if sf_type == SFType.Au else sf_lll_prop_xr(sf_tab)
    tab_short = pi.to_dataframe().reset_index()
    return tab_short.melt(id_vars=["time", "dr", "k"], value_name="ε", var_name="axis")


def roll_da(ds: xr.DataArray, args: dict) -> xr.DataArray:
    """Turns xarray roll into a separate function"""
    return ds.roll(args)


def shift_da(ds: xr.DataArray, args: dict) -> xr.DataArray:
    """Turns xarray shift into a separate function"""
    return ds.shift(args)
