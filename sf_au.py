import cupy as cp
import numpy as np
from numba import njit, prange

from .roll import roll_numba, roll_par
from .utils import Axis, ndarray


@cp.fuse
def sf_au_kernel(
    du: ndarray,
    dv: ndarray,
    dw: ndarray,
    dau: ndarray,
    dav: ndarray,
    daw: ndarray,
) -> ndarray:
    """kernel for dotting advection and velocity differences"""
    return du * dau + dv * dav + dw * daw


def sf_au(
    u: ndarray,
    v: ndarray,
    w: ndarray,
    x: ndarray,
    y: ndarray,
    z: ndarray,
) -> tuple[tuple[ndarray, ndarray, ndarray], ndarray]:
    """Calculate the advective structure function on 3d velocity data
    Use all sets of separations, assume periodic data, accepts numpy or cupy arrays
    note: x is the third index in the velocity and output arrays, z is the first

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
    spectral : bool
        Whether to use spectral calculation of gradients
        (default false, true is currently broken)

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        1d arrays of spacing values in z, y, and x
    ndarray
        structure function value at each set of spacings
    """
    L = len(z) // 2
    M = len(y) // 2
    N = len(x) // 2

    sf = np.zeros((L, M, N), np.float64, like=u)

    dz = z[:L] - z[0]
    dy = y[:M] - y[0]
    dx = x[:N] - x[0]

    diffs = (dx, dy, dz)
    xp = cp.get_array_module(u)

    dudz, dudy, dudx = xp.gradient(u, z, y, x, axis=(0, 1, 2))
    dvdz, dvdy, dvdx = xp.gradient(v, z, y, x, axis=(0, 1, 2))
    dwdz, dwdy, dwdx = xp.gradient(w, z, y, x, axis=(0, 1, 2))

    uadv = u * dudx + v * dudy + w * dudz
    vadv = u * dvdx + v * dvdy + w * dvdz
    wadv = u * dwdx + v * dwdy + w * dwdz

    for i in range(L):
        for j in range(M):
            for k in range(N):
                if not (i == 0 and j == 0 and k == 0):
                    du = np.roll(u, shift=(-i, -j, -k), axis=(0, 1, 2)) - u
                    dv = np.roll(v, shift=(-i, -j, -k), axis=(0, 1, 2)) - v
                    dw = np.roll(w, shift=(-i, -j, -k), axis=(0, 1, 2)) - w
                    dau = np.roll(uadv, shift=(-i, -j, -k), axis=(0, 1, 2)) - uadv
                    dav = np.roll(vadv, shift=(-i, -j, -k), axis=(0, 1, 2)) - vadv
                    daw = np.roll(wadv, shift=(-i, -j, -k), axis=(0, 1, 2)) - wadv
                    sf[i, j, k] = xp.mean(sf_au_kernel(du, dv, dw, dau, dav, daw))

    return (diffs, sf)


def sf_au_dir(
    u: ndarray,
    v: ndarray,
    w: ndarray,
    x: ndarray,
    y: ndarray,
    z: ndarray,
    axis: Axis,
) -> tuple[ndarray, ndarray]:
    """Calculate the advective structure function on 3d velocity data
    Use separations in a specified direction, assume periodic data,
    accepts numpy or cupy arrays
    note: x is the third index in the velocity and output arrays, z is the first

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
    axis : Axis
        Axis along which separations are used
    spectral : bool
        Whether to use spectral calculation of gradients
        (default false, true is currently broken)

    Returns
    -------
    ndarray
        1d arrays of spacing values in the specified axis
    ndarray
        structure function value at each spacing
    """
    L = len(z) // 2
    M = len(y) // 2
    N = len(x) // 2

    count = (L, M, N)[axis.value]

    sf = np.zeros(count, np.float64, like=u)

    dz = z[:L] - z[0]
    dy = y[:M] - y[0]
    dx = x[:N] - x[0]

    diffs = (dx, dy, dz)[axis.value]

    dudz, dudy, dudx = np.gradient(u, z, y, x, axis=(0, 1, 2))
    dvdz, dvdy, dvdx = np.gradient(v, z, y, x, axis=(0, 1, 2))
    dwdz, dwdy, dwdx = np.gradient(w, z, y, x, axis=(0, 1, 2))

    uadv = u * dudx + v * dudy + w * dudz
    vadv = u * dvdx + v * dvdy + w * dvdz
    wadv = u * dwdx + v * dwdy + w * dwdz

    for i in range(count):
        du = np.roll(u, -i, axis=axis.value) - u
        dv = np.roll(v, -i, axis=axis.value) - v
        dw = np.roll(w, -i, axis=axis.value) - w
        dau = np.roll(uadv, -i, axis=axis.value) - uadv
        dav = np.roll(vadv, -i, axis=axis.value) - vadv
        daw = np.roll(wadv, -i, axis=axis.value) - wadv
        sf[i] = np.mean(sf_au_kernel(du, dv, dw, dau, dav, daw))

    return (diffs, sf)


def sf_au_numba(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    parallel: bool = False,
) -> tuple[tuple[ndarray, ndarray, ndarray], ndarray]:
    """Calculate the advective structure function on 3d velocity data
    Use all sets of separations, assume periodic data, accelerated by numba
    note: x is the third index in the velocity and output arrays, z is the first

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
    parallel : bool, default false
        whether to use the parallel numba parameter

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

    grads = (
        np.gradient(u, z, y, x, axis=(0, 1, 2)),
        np.gradient(v, z, y, x, axis=(0, 1, 2)),
        np.gradient(w, z, y, x, axis=(0, 1, 2)),
    )

    dz = z[:L] - z[0]
    dy = y[:M] - y[0]
    dx = x[:N] - x[0]

    diffs = (dz, dy, dx)

    if parallel:
        sf = sf_au_numba_calc_par(u, v, w, x, y, z, grads)
    else:
        sf = sf_au_numba_calc(u, v, w, x, y, z, grads)

    return (diffs, sf)


@njit
def sf_au_numba_calc(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grads: tuple,
) -> np.ndarray:
    """Internal function for numba-accelerated structure function calculation"""
    L = len(z) // 2
    M = len(y) // 2
    N = len(x) // 2

    sf = np.zeros((L, M, N), np.float64)

    uadv = w * grads[0][0] + v * grads[0][1] + u * grads[0][2]
    vadv = w * grads[1][0] + v * grads[1][1] + u * grads[1][2]
    wadv = w * grads[2][0] + v * grads[2][1] + u * grads[2][2]

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
                    dau = roll_numba(uadv, i, j, k, tmp1, tmp2, tmp3) - uadv
                    dav = roll_numba(vadv, i, j, k, tmp1, tmp2, tmp3) - vadv
                    daw = roll_numba(wadv, i, j, k, tmp1, tmp2, tmp3) - wadv
                    sf[i, j, k] = np.mean(du * dau + dv * dav + dw * daw)

    return sf


@njit(parallel=True)
def sf_au_numba_calc_par(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    grads: tuple,
) -> np.ndarray:
    """Internal function for numba-accelerated, parallel structure function calculation"""
    L = len(z) // 2
    M = len(y) // 2
    N = len(x) // 2

    sf = np.zeros((L, M, N), np.float64)

    uadv = w * grads[0][0] + v * grads[0][1] + u * grads[0][2]
    vadv = w * grads[1][0] + v * grads[1][1] + u * grads[1][2]
    wadv = w * grads[2][0] + v * grads[2][1] + u * grads[2][2]

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
                    dau = roll_par(uadv, i, j, k, tmp1, tmp2, tmp3) - uadv
                    dav = roll_par(vadv, i, j, k, tmp1, tmp2, tmp3) - vadv
                    daw = roll_par(wadv, i, j, k, tmp1, tmp2, tmp3) - wadv
                    sf[i, j, k] = np.mean(du * dau + dv * dav + dw * daw)

    return sf
