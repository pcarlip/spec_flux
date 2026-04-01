from dataclasses import dataclass
from enum import Enum, StrEnum
from types import ModuleType

import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np
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
    xp, genfft = xp_fft(data.u)
    spacings = (data.x[1] - data.x[0], data.y[1] - data.y[0], data.z[1] - data.z[0])
    ranges = []
    for i in range(3):
        Ni = data.u.shape[i]
        Li = Ni * spacings[i]
        dk = 2 * xp.pi / Li
        k_range = genfft.fftshift(genfft.fftfreq(Ni) * Ni * dk)
        ranges.append(k_range)
    ranges = tuple(ranges)
    return (spacings, ranges)
