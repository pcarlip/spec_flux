from dataclasses import dataclass
from enum import Enum, StrEnum

import cupy as cp
import numpy as np

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


@dataclass
class SimDataLite:
    """A wrapper to store the relevant data from the NetCDF at a particular time step,
    not including advection components"""

    u: ndarray
    v: ndarray
    w: ndarray
    x: ndarray


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


def meshgrid_sel(grid: tuple, axis: Axis) -> ndarray:
    """Get the appropriate element from a 3d meshgrid to use in a spectral gradient
    (I don't know why it works like that, it just does)

    Parameters
    ----------
    grid : tuple[ndarray, ndarray, ndarray]
        meshgrid constructed as (y,z,x)
        (necessary to give the right shape when indexed as (z,y,x))
    axis : Axis
        Axis along which you're taking the spectral gradient

    Returns
    -------
    ndarray
        one element from the meshgrid,
        appropriate for spectral derivatives along the given axis
    """
    if axis == Axis.x:
        return grid[2]
    elif axis == Axis.y:
        return grid[0]
    else:
        return grid[1]
