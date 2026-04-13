from dataclasses import dataclass
from enum import Enum, StrEnum
from types import ModuleType

import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np
import xarray as xr
from scipy import fft

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


class GradMethod(Enum):
    oceananigans = 1
    spectral = 2
    numpy = 3


class Axis(Enum):
    z = 0
    y = 1
    x = 2


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
    genfft = cufft if xp.__name__ == "cupy" else fft
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
    size = len(model["x_faa"].values)
    kmin = np.pi / L
    kmax = kmin * size * 2
    if log:
        return np.logspace(np.log10(kmin), np.log10(kmax), n)
    else:
        return np.linspace(kmin, kmax, n)
