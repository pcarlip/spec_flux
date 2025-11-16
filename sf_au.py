import cupy as cp
import numpy as np
from numba import njit, prange

from .roll import roll, roll_numba, roll_par

type ndarray = np.ndarray | cp.ndarray  # noqa: PYI042


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
    spectral: bool = False,
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
        Whether to use spectral calculation of gradients (default false)

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

    if spectral:
        if cp.get_array_module(u) == cp:
            from cupyx.scipy import fft
        else:
            from scipy import fft
        dk = 2 * np.pi / (len(x) * dx[1])
        dl = 2 * np.pi / (len(y) * dy[1])
        dm = 2 * np.pi / (len(z) * dz[1])
        k_range = fft.fftfreq(len(x)) * len(x) * dk
        l_range = fft.fftfreq(len(y)) * len(y) * dl
        m_range = fft.fftfreq(len(z)) * len(z) * dm
        u_hat = fft.fftn(u)
        v_hat = fft.fftn(v)
        w_hat = fft.fftn(w)
        k_mesh = np.meshgrid(l_range, m_range, k_range)
        dudx = np.real(fft.ifftn(1j * k_mesh[2] * u_hat))
        dudy = np.real(fft.ifftn(1j * k_mesh[0] * u_hat))
        dudz = np.real(fft.ifftn(1j * k_mesh[1] * u_hat))
        dvdx = np.real(fft.ifftn(1j * k_mesh[2] * v_hat))
        dvdy = np.real(fft.ifftn(1j * k_mesh[0] * v_hat))
        dvdz = np.real(fft.ifftn(1j * k_mesh[1] * v_hat))
        dwdx = np.real(fft.ifftn(1j * k_mesh[2] * w_hat))
        dwdy = np.real(fft.ifftn(1j * k_mesh[0] * w_hat))
        dwdz = np.real(fft.ifftn(1j * k_mesh[1] * w_hat))
        grads = ((dudz, dudy, dudx), (dvdz, dvdy, dvdx), (dwdz, dwdy, dwdx))
    else:
        grads = (
            np.gradient(u, z, y, x, axis=(0, 1, 2)),
            np.gradient(v, z, y, x, axis=(0, 1, 2)),
            np.gradient(w, z, y, x, axis=(0, 1, 2)),
        )

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
                    du = roll(u, i, j, k, tmp1, tmp2, tmp3) - u
                    dv = roll(v, i, j, k, tmp1, tmp2, tmp3) - v
                    dw = roll(w, i, j, k, tmp1, tmp2, tmp3) - w
                    dau = roll(uadv, i, j, k, tmp1, tmp2, tmp3) - uadv
                    dav = roll(vadv, i, j, k, tmp1, tmp2, tmp3) - vadv
                    daw = roll(wadv, i, j, k, tmp1, tmp2, tmp3) - wadv
                    sf[i, j, k] = np.mean(sf_au_kernel(du, dv, dw, dau, dav, daw))

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
