from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, StrEnum
from types import ModuleType
from typing import Self

import cupy as cp
import cupyx
import numpy as np
import xarray as xr
from xarray_extras.interpolate import splev, splrep

type ndarray = np.ndarray | cp.ndarray  # noqa: PYI042


@dataclass
class SimData:
    """A wrapper to store the relevant data from the NetCDF at a particular time step,
    including pre-calculated advection components"""

    u: ndarray
    v: ndarray
    w: ndarray
    uadv: ndarray
    vadv: ndarray
    wadv: ndarray
    x: ndarray
    y: ndarray
    z: ndarray

    @classmethod
    def from_xr(cls, ds: xr.Dataset):  # noqa: ANN206
        return cls(
            ds["u"].data,
            ds["v"].data,
            ds["w"].data,
            ds["uadv"].data,
            ds["vadv"].data,
            ds["wadv"].data,
            ds["x_caa"].data,
            ds["y_aca"].data,
            ds["z_aac"].data,
        )


@dataclass
class SimDataLite:
    """A wrapper to store the relevant data from the NetCDF at a particular time step,
    not including advection components"""

    u: ndarray
    v: ndarray
    w: ndarray
    x: ndarray
    y: ndarray
    z: ndarray

    @classmethod
    def from_xr(cls, ds: xr.Dataset):  # noqa: ANN206
        return cls(
            ds["u"].data,
            ds["v"].data,
            ds["w"].data,
            ds["x_caa"].data,
            ds["y_aca"].data,
            ds["z_aac"].data,
        )

    def to_cp(self) -> Self:
        self.u = cp.array(self.u)
        self.v = cp.array(self.v)
        self.w = cp.array(self.w)
        self.x = cp.array(self.x)
        self.y = cp.array(self.y)
        self.z = cp.array(self.z)
        return self

    def to_np(self) -> Self:
        self.u = cp.asnumpy(self.u)
        self.v = cp.asnumpy(self.v)
        self.w = cp.asnumpy(self.w)
        self.x = cp.asnumpy(self.x)
        self.y = cp.asnumpy(self.y)
        self.z = cp.asnumpy(self.z)
        return self


class GradMethod(Enum):
    oceananigans = 1
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


class IntMethod(Enum):
    simpson = 0
    addition = 1


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


def spacings_krange(
    data: SimData | SimDataLite,
) -> tuple[tuple[float, float, float], tuple[ndarray, ndarray, ndarray]]:
    """Get the realspace spacings and fourier-space ranges of k-values from the range of
    x, y, and z values

    Parameters
    ----------
    data : SimData | SimDataLite
        Dataclass containing grid axes

    Returns
    -------
    tuple[float, float, float]
        grid spacings in z, y, x (assumed to be constant, but not necessarily equal)
    tuple[ndarray, ndarray, ndarray]
        arrays of m, l, and k values of the fourier-transformed grids
    """
    xp, genfft = xp_fft(data.u)
    spacings = (data.z[1] - data.z[0], data.y[1] - data.y[0], data.x[1] - data.x[0])
    ranges = []
    for i in range(3):
        Ni = data.u.shape[i]
        Li = Ni * spacings[i]
        dk = 2 * xp.pi / Li
        k_range = genfft.fftshift(genfft.fftfreq(Ni) * Ni * dk)
        ranges.append(k_range)
    ranges = tuple(ranges)
    return (spacings, ranges)


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

    return xr.merge([uvar, vvar, wvar], compat="no_conflicts")


def ocean_interp_adv_per(
    oc_input: xr.Dataset,
    time: int = -1,
    periodic_axes: tuple[Axis, ...] = (Axis.x, Axis.y, Axis.z),
) -> xr.Dataset:
    """Interpolate Oceananigans output velocities and advection to use the same axes,
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

    Returns
    -------
    xr.Dataset
        Dataset with u,v,w,uadv,vadv,wadv on the same set of axes (cell centers)
    """
    u_interp = splrep(oc_input["u"].isel(time=time), "x_faa")
    v_interp = splrep(oc_input["v"].isel(time=time), "y_afa")
    w_interp = splrep(oc_input["w"].isel(time=time), "z_aaf")
    uadv_interp = splrep(oc_input["uadv"].isel(time=time), "x_faa")
    vadv_interp = splrep(oc_input["vadv"].isel(time=time), "x_faa")
    wadv_interp = splrep(oc_input["wadv"].isel(time=time), "x_faa")

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

    return xr.merge([uvar, vvar, wvar, uadvvar, vadvvar, wadvvar], compat="no_conflicts")
