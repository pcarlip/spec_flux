import time

import cupy as cp
import numpy as np
from numba import njit, prange

from .roll import roll_numba, roll_par
from .utils import Axis, ndarray

# steps:
# set of motion amounts: can just use motion of i,j,k for range N//2
#   (more precisely L, M, N; might not want to assume cubic array)
# make array of Δx, Δy, Δz: meshgrid?
# for each index: roll array (or similar), take differences,
#   apply operation (e.g. power), take mean
#   note: L^n is longitudinal,
#   I need to get the angle and appropriately weight the components:
#   proj_b(a) = a•b^, which is e.g. (u*i+v*j+w*k)/r
# should be easy to parallelize the outer loop (e.g. multiprocessing, maybe try numba)
# but I don't know if that's compatible w/ cupy for the objects/inner loop


@cp.fuse
def sf_kernel(
    du: ndarray,
    dv: ndarray,
    dw: ndarray,
    i: int,
    j: int,
    k: int,
    r: float,
    order: int,
) -> ndarray:
    """Kernel for taking the longitudinal component of the velocity structure function"""
    du_l = (du * k + dv * j + dw * i) / r
    return du_l**order


def sf_ln(
    u: ndarray,
    v: ndarray,
    w: ndarray,
    x: ndarray,
    y: ndarray,
    z: ndarray,
    order: int = 3,
) -> tuple[tuple[ndarray, ndarray, ndarray], ndarray]:
    """Calculate a longitudinal structure function of an arbitrary order on 3d velocities
    Use all sets of separations, assume periodic data, accepts numpy or cupy arrays

    Parameters
    ----------
    u : ndarray
        3d array of x velocities
    v : ndarray
        3d array of y velocities
    w : ndarray
        3d array of z velocities
    x : ndarray
        1d array of x positions
    y : ndarray
        1d array of y positions
    z : ndarray
        1d array of z positions
    order : int
        power of the velocity differences (e.g. use SF_LL or LLL)
        Default value is 3 (LLL)

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        1d arrays of spacing values in z, y, and x
    ndarray
        structure function value at each set of spacings
    """
    xp = cp.get_array_module(u)

    L = len(z) // 2
    M = len(y) // 2
    N = len(x) // 2

    sf = np.zeros((L, M, N), like=u)

    dz = z[:L] - z[0]
    dy = y[:M] - y[0]
    dx = x[:N] - x[0]

    diffs = (dz, dy, dx)

    for i in range(L):
        if i % 25 == 0:
            print(i, flush=True)
            print(time.ctime(), flush=True)
        for j in range(M):
            for k in range(N):
                if not (i == 0 and j == 0 and k == 0):
                    r = xp.sqrt(i**2 + j**2 + k**2)
                    du = xp.roll(u, (-i, -j, -k), axis=(0, 1, 2)) - u
                    dv = xp.roll(v, (-i, -j, -k), axis=(0, 1, 2)) - v
                    dw = xp.roll(w, (-i, -j, -k), axis=(0, 1, 2)) - w
                    sf[i, j, k] = xp.mean(sf_kernel(du, dv, dw, i, j, k, r, order))

    return (diffs, sf)


def sf_ln_dir(
    u: ndarray,
    v: ndarray,
    w: ndarray,
    x: ndarray,
    y: ndarray,
    z: ndarray,
    axis: Axis,
    order: int = 3,
) -> tuple[ndarray, ndarray]:
    """Calculate a longitudinal structure function of an arbitrary order on 3d velocities
    Use separations along one axis, assume periodic data, accepts numpy or cupy arrays
    note: convention is that z is the first axis of the velocity arrays, x is the third

    Parameters
    ----------
    u : ndarray
        3d array of x velocities
    v : ndarray
        3d array of y velocities
    w : ndarray
        3d array of z velocities
    x : ndarray
        1d array of x positions
    y : ndarray
        1d array of y positions
    z : ndarray
        1d array of z positions
    axis: Axis
        Axis along which separations are used
    order : int
        power of the velocity differences (e.g. use SF_LL or LLL)
        Default value is 3 (LLL)

    Returns
    -------
    ndarray
        1d arrays of spacing values along specified axis
    ndarray
        structure function value at each spacing
    """
    xp = cp.get_array_module(u)

    L = len(z) // 2
    M = len(y) // 2
    N = len(x) // 2
    count = (L, M, N)[axis.value]

    sf = xp.zeros(count)

    dz = z[:L] - z[0]
    dy = y[:M] - y[0]
    dx = x[:N] - x[0]

    diffs = (dz, dy, dx)[axis.value]

    vel = (w, v, u)[axis.value]

    for i in range(1, count):
        sf[i] = xp.mean((xp.roll(vel, -i, axis.value) - vel) ** order)

    return (diffs, sf)


def sf_ln_numba(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    order: int = 3,
    parallel: bool = False,
) -> tuple[tuple[ndarray, ndarray, ndarray], ndarray]:
    """Calculate a longitudinal structure function of an arbitrary order on 3d velocities
    Use all sets of separations, assume periodic data, accelerated by numba

    Parameters
    ----------
    u : np.ndarray
        3d array of x velocities
    v : np.ndarray
        3d array of y velocities
    w : np.ndarray
        3d array of z velocities
    x : np.ndarray
        1d array of x positions
    y : np.ndarray
        1d array of y positions
    z : np.ndarray
        1d array of z positions
    order : int
        power of the velocity differences (e.g. use SF_LL or LLL).
        Default value is 3 (LLL)

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        1d arrays of spacing values in z, y, and x
    np.ndarray
        structure function value at each set of spacings
    """
    L = len(z) // 2
    M = len(y) // 2
    N = len(x) // 2

    dz = z[:L] - z[0]
    dy = y[:M] - y[0]
    dx = x[:N] - x[0]

    diffs = (dz, dy, dx)

    if parallel:
        sf = sf_ln_numba_calc_par(u, v, w, x, y, z, order)
    else:
        sf = sf_ln_numba_calc(u, v, w, x, y, z, order)

    return (diffs, sf)


@njit
def sf_ln_numba_calc(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    order: int = 3,
) -> np.ndarray:
    """Internal function for numba-accelerated structure function calculation"""
    L = len(z) // 2
    M = len(y) // 2
    N = len(x) // 2

    sf = np.zeros((L, M, N), np.float64)

    tmp1 = np.zeros_like(u)
    tmp2 = np.zeros_like(u)
    tmp3 = np.zeros_like(u)

    for i in range(L):
        for j in range(M):
            for k in range(N):
                if not (i == 0 and j == 0 and k == 0):
                    du = roll_numba(u, i, j, k, tmp1, tmp2, tmp3) - u
                    dv = roll_numba(v, i, j, k, tmp1, tmp2, tmp3) - v
                    dw = roll_numba(w, i, j, k, tmp1, tmp2, tmp3) - w
                    du_l = (du * k + dv * j + dw * i) / np.sqrt(i**2 + j**2 + k**2)
                    sf[i, j, k] = np.mean(du_l**order)

    return sf


@njit(parallel=True)
def sf_ln_numba_calc_par(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    order: int = 3,
) -> np.ndarray:
    """Internal function for numba-accelerated, parallel structure function calculation"""
    L = len(z) // 2
    M = len(y) // 2
    N = len(x) // 2

    sf = np.zeros((L, M, N))

    tmp1 = np.zeros_like(u)
    tmp2 = np.zeros_like(u)
    tmp3 = np.zeros_like(u)

    for i in prange(L):
        for j in prange(M):
            for k in prange(N):
                if not (i == 0 and j == 0 and k == 0):
                    du = roll_par(u, i, j, k, tmp1, tmp2, tmp3) - u
                    dv = roll_par(v, i, j, k, tmp1, tmp2, tmp3) - v
                    dw = roll_par(w, i, j, k, tmp1, tmp2, tmp3) - w
                    du_l = (du * k + dv * j + dw * i) / np.sqrt(i**2 + j**2 + k**2)
                    sf[i, j, k] = np.mean(du_l**order)

    return sf
