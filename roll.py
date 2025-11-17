import cupy as cp
import numpy as np
from numba import njit, prange

type ndarray = np.ndarray | cp.ndarray  # noqa: PYI042


@njit
def roll_numba(
    arr: np.ndarray,
    dz: int,
    dy: int,
    dx: int,
    tmp1: np.ndarray,
    tmp2: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    shifted_z = tmp1
    if dz != 0:
        shifted_z[:-dz, :, :] = arr[dz:, :, :]
        shifted_z[-dz:, :, :] = arr[:dz, :, :]
    else:
        shifted_z = arr
    shifted_y = tmp2
    if dy != 0:
        shifted_y[:, :-dy, :] = shifted_z[:, dy:, :]
        shifted_y[:, -dy:, :] = shifted_z[:, :dy, :]
    else:
        shifted_y = shifted_z
    if dx != 0:
        out[:, :, :-dx] = shifted_y[:, :, dx:]
        out[:, :, -dx:] = shifted_y[:, :, :dx]
    else:
        out = shifted_y
    return out


@njit(parallel=True)
def roll_par(
    arr: np.ndarray,
    dz: int,
    dy: int,
    dx: int,
    tmp1: np.ndarray,
    tmp2: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    shifted_z = tmp1
    if dz != 0:
        shifted_z[:-dz, :, :] = arr[dz:, :, :]
        shifted_z[-dz:, :, :] = arr[:dz, :, :]
    else:
        shifted_z = arr
    shifted_y = tmp2
    if dy != 0:
        shifted_y[:, :-dy, :] = shifted_z[:, dy:, :]
        shifted_y[:, -dy:, :] = shifted_z[:, :dy, :]
    else:
        shifted_y = shifted_z
    if dx != 0:
        out[:, :, :-dx] = shifted_y[:, :, dx:]
        out[:, :, -dx:] = shifted_y[:, :, :dx]
    else:
        out = shifted_y
    return out


def roll_old(
    arr: ndarray,
    dz: int,
    dy: int,
    dx: int,
    tmp1: ndarray,
    tmp2: ndarray,
    out: ndarray,
) -> ndarray:
    shifted_z = tmp1
    if dz != 0:
        shifted_z[:-dz, :, :] = arr[dz:, :, :]
        shifted_z[-dz:, :, :] = arr[:dz, :, :]
    else:
        shifted_z = arr
    shifted_y = tmp2
    if dy != 0:
        shifted_y[:, :-dy, :] = shifted_z[:, dy:, :]
        shifted_y[:, -dy:, :] = shifted_z[:, :dy, :]
    else:
        shifted_y = shifted_z
    if dx != 0:
        out[:, :, :-dx] = shifted_y[:, :, dx:]
        out[:, :, -dx:] = shifted_y[:, :, :dx]
    else:
        out = shifted_y
    return out


def roll(
    arr: ndarray,
    dz: int,
    dy: int,
    dx: int,
    tmp1: ndarray,
    tmp2: ndarray,
    out: ndarray,
) -> ndarray:
    return np.roll(arr, shift=(-dz, -dy, -dx), axis=(0, 1, 2))
