import time
from collections.abc import Collection, Hashable, Mapping
from itertools import product

import cupy as cp
import cupy_xarray
import numpy as np
import xarray as xr
from numba import njit, prange

from .advection import advection, advection_xr
from .roll import roll_numba, roll_par
from .utils import (
    Axis,
    GradMethod,
    SimDataLite,
    axis_name,
    ndarray,
    spacings_krange,
    xp_fft,
)


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
    debug_print: bool = False,
    spacing_lst: Collection[int] | None = None,
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
    spectral : bool, optional
        Whether to use spectral calculation of gradients, by default False
        Note: the spectral case is not directly tested, but differs only in the
        calculation of gradients, which is tested
    debug_print : bool, optional
        Whether to print messages every 25th iteration of the outer loop to monitor
        progress, by default False

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        1d arrays of spacing values in z, y, and x
    ndarray
        structure function value at each set of spacings
    """
    xp, _ = xp_fft(u)

    if spacing_lst is not None:
        xind = spacing_lst
        yind = spacing_lst
        zind = spacing_lst
    else:
        Nx = len(x) // 2
        Ny = len(y) // 2
        Nz = len(z) // 2
        xind = range(Nx)
        yind = range(Ny)
        zind = range(Nz)

    dx = [x[i] - x[0] for i in xind]
    dy = [y[i] - y[0] for i in yind]
    dz = [z[i] - z[0] for i in zind]

    sf = xp.zeros((len(zind), len(yind), len(xind)))

    diffs = (dx, dy, dz)

    grad_method = GradMethod.spectral if spectral else GradMethod.numpy
    data = SimDataLite(u, v, w, x, y, z)
    spacings, ranges = spacings_krange(data)

    uadv = advection(data, Axis.x, spacings, ranges, grad_method)
    vadv = advection(data, Axis.y, spacings, ranges, grad_method)
    wadv = advection(data, Axis.z, spacings, ranges, grad_method)

    for ni, i in enumerate(zind):
        if i % 5 == 0 and debug_print:
            print(i, flush=True)
        for nj, j in enumerate(yind):
            for nk, k in enumerate(xind):
                if not (i == 0 and j == 0 and k == 0):
                    du = xp.roll(u, shift=(-i, -j, -k), axis=(0, 1, 2)) - u
                    dv = xp.roll(v, shift=(-i, -j, -k), axis=(0, 1, 2)) - v
                    dw = xp.roll(w, shift=(-i, -j, -k), axis=(0, 1, 2)) - w
                    dau = xp.roll(uadv, shift=(-i, -j, -k), axis=(0, 1, 2)) - uadv
                    dav = xp.roll(vadv, shift=(-i, -j, -k), axis=(0, 1, 2)) - vadv
                    daw = xp.roll(wadv, shift=(-i, -j, -k), axis=(0, 1, 2)) - wadv
                    sf[ni, nj, nk] = xp.mean(sf_au_kernel(du, dv, dw, dau, dav, daw))

    return (diffs, sf)


def sf_au_dir(
    u: ndarray,
    v: ndarray,
    w: ndarray,
    x: ndarray,
    y: ndarray,
    z: ndarray,
    axis: Axis,
    spectral: bool = False,
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
    spectral : bool, optional
        Whether to use spectral calculation of gradients, by default False
        Note: the spectral case is not directly tested, but differs only in the
        calculation of gradients, which is tested

    Returns
    -------
    ndarray
        1d arrays of spacing values in the specified axis
    ndarray
        structure function value at each spacing
    """
    xp, _ = xp_fft(u)

    L = len(z) // 2
    M = len(y) // 2
    N = len(x) // 2

    count = (L, M, N)[axis.value]

    sf = np.zeros(count, like=u)

    dz = z[:L] - z[0]
    dy = y[:M] - y[0]
    dx = x[:N] - x[0]

    diffs = (dx, dy, dz)[axis.value]

    grad_method = GradMethod.spectral if spectral else GradMethod.numpy
    data = SimDataLite(u, v, w, x, y, z)

    spacings, ranges = spacings_krange(data)

    uadv = advection(data, Axis.x, spacings, ranges, grad_method)
    vadv = advection(data, Axis.y, spacings, ranges, grad_method)
    wadv = advection(data, Axis.z, spacings, ranges, grad_method)

    for i in range(count):
        du = xp.roll(u, -i, axis=axis.value) - u
        dv = xp.roll(v, -i, axis=axis.value) - v
        dw = xp.roll(w, -i, axis=axis.value) - w
        dau = xp.roll(uadv, -i, axis=axis.value) - uadv
        dav = xp.roll(vadv, -i, axis=axis.value) - vadv
        daw = xp.roll(wadv, -i, axis=axis.value) - wadv
        sf[i] = xp.mean(sf_au_kernel(du, dv, dw, dau, dav, daw))

    return (diffs, sf)


def sf_au_nd(
    data: xr.Dataset,
    spacing: int = 1,
    offset: int = 0,
    spacing_lst: Collection[int] | None = None,
    spacing_dict: dict[str, Collection[int]] | None = None,
    dims: tuple[str, ...] = ("z_aac", "y_aca", "x_caa"),
    vels: tuple[str, ...] = ("w", "v", "u"),
) -> xr.DataArray:
    """Calculate the advective structure function for an xarray dataset, with all
    combinations of spacings in some range

    Parameters
    ----------
    data : xr.Dataset
        Dataset with velocities and axes
    spacing : int, optional
        Use every nth spacing along each axis, if spacing_lst or spacing_dict not given,
        by default 1
    offset : int, optional
        Offset the first spacing by m before starting, if spacing_lst or spacing_dict
        not given, by default 0
    spacing_lst : Collection[int] | None, optional
        List of (integer) spacings to use along all axes, by default None
    spacing_dict : dict[str, Collection[int]] | None, optional
        Dict of (integer) spacings to use along each distinct axis, keys must match dims
        by default None
    dims : tuple[str, ...]
        List of (spatial) dimension names in the data along which to take SFs,
        by default ("z_aac","y_aca","x_caa")
    vels : tuple[str, ...]
        List of velocity names in the data corresponding to the above dimensions,
        by default ("w", "v", "u")

    Returns
    -------
    xr.DataArray
        3D data of the structure function at each spacing combination
    """
    ndim = len(dims)
    if spacing_lst is not None:
        spacing_lsts = [spacing_lst] * ndim
    elif spacing_dict is not None:
        spacing_lsts = [(spacing_dict[i]) for i in dims]
    else:
        dim_len = [len(data[dim]) for dim in dims]
        spacing_lsts = [range(offset, N // 2, spacing) for N in dim_len]
    spacing_comb = product(*spacing_lsts)
    spacing_enum = product(*[range(len(i)) for i in spacing_lsts])
    # I'm not sure what the *[] does here, but it's apparently necessary

    diffs = [data[dims[i]][spacing_lsts[i]] - data[dims[i]][0] for i in range(ndim)]

    adv_lst: list[xr.DataArray] = []
    for i in range(ndim):
        adv = xr.zeros_like(data[vels[i]])
        for j in range(ndim):
            adv += data[vels[j]] * data[vels[i]].differentiate(dims[j])
        adv_lst.append(adv)

    vel_data = [data[vel].transpose(*dims).data for vel in vels]
    adv_data = [adv_lst[i].transpose(*dims).data for i in range(ndim)]

    xp = cp.get_array_module(vel_data[0])
    out = xp.zeros([len(i) for i in spacing_lsts])

    for inds, ninds in zip(spacing_comb, spacing_enum, strict=True):
        sf_arr = xp.zeros_like(vel_data[0])
        for i in range(ndim):
            du = vel_data[i] - xp.roll(vel_data[i], inds, range(ndim))
            dau = adv_data[i] - xp.roll(adv_data[i], inds, range(ndim))
            sf_arr += du * dau
        out[ninds] = xp.mean(sf_arr)

    return xr.DataArray(
        out,
        [("d" + dims[i], diffs[i].data) for i in range(ndim)],
        name="SF_Au",
    )


def sf_au_xr(
    data: xr.Dataset,
    spectral: bool = False,
    spacing: int = 1,
    offset: int = 0,
    spacing_lst: Collection[int] | None = None,
    spacing_dict: dict[str, Collection[int]] | None = None,
    debug_print: bool = False,
) -> xr.DataArray:
    """Calculate the advective structure function for an xarray dataset, with all
    combinations of spacings in some range

    Parameters
    ----------
    data : xr.Dataset
        Dataset with velocities u, v, w; dimensions x_caa, y_aca, z_aac
    spectral : bool, optional
        Use spectral derivatives to calculate advection, by default False
    spacing : int, optional
        Use every nth spacing along each axis, if spacing_lst or spacing_dict not given,
        by default 1
    offset : int, optional
        Offset the first spacing by m before starting, if spacing_lst or spacing_dict
        not given, by default 0
    spacing_lst : Collection[int] | None, optional
        List of (integer) spacings to use along all axes, by default None
    spacing_dict : dict[str, Collection[int]] | None, optional
        Dict of (integer) spacings to use along each distinct axis, with keys x,y,z,
        by default None

    Returns
    -------
    xr.DataArray
        3D data of the structure function at each spacing combination
    """
    z = data["z_aac"]
    y = data["y_aca"]
    x = data["x_caa"]

    if spacing_lst is not None:
        xind = spacing_lst
        yind = spacing_lst
        zind = spacing_lst
    elif spacing_dict is not None:
        xind = spacing_dict["x"]
        yind = spacing_dict["y"]
        zind = spacing_dict["z"]
    else:
        Nx = len(x) // 2
        Ny = len(y) // 2
        Nz = len(z) // 2
        xind = range(offset, Nx, spacing)
        yind = range(offset, Ny, spacing)
        zind = range(offset, Nz, spacing)

    dx = [x[i] - x[0] for i in xind]
    dy = [y[i] - y[0] for i in yind]
    dz = [z[i] - z[0] for i in zind]

    grad_method = GradMethod.spectral if spectral else GradMethod.numpy
    uadv = advection_xr(data, Axis.x, grad_method).data
    vadv = advection_xr(data, Axis.y, grad_method).data
    wadv = advection_xr(data, Axis.z, grad_method).data

    xp = cp.get_array_module(data["u"].data)
    out = xp.zeros((len(zind), len(yind), len(xind)))

    u = data["u"].data
    v = data["v"].data
    w = data["w"].data

    for ni, i in enumerate(zind):
        if i % 5 == 0 and debug_print:
            print(i)
        for nj, j in enumerate(yind):
            for nk, k in enumerate(xind):
                du = xp.roll(u, (-i, -j, -k), axis=(0, 1, 2)) - u
                dv = xp.roll(v, (-i, -j, -k), axis=(0, 1, 2)) - v
                dw = xp.roll(w, (-i, -j, -k), axis=(0, 1, 2)) - w
                dau = xp.roll(uadv, (-i, -j, -k), axis=(0, 1, 2)) - uadv
                dav = xp.roll(vadv, (-i, -j, -k), axis=(0, 1, 2)) - vadv
                daw = xp.roll(wadv, (-i, -j, -k), axis=(0, 1, 2)) - wadv
                out[ni, nj, nk] = xp.mean(sf_au_kernel(du, dv, dw, dau, dav, daw))

    return xr.DataArray(
        out,
        [("dz_aac", dz), ("dy_aca", dy), ("dx_caa", dx)],
        name="SF_Au",
    )


def sf_au_dir_xr(data: xr.Dataset, axis: Axis, spectral: bool = False) -> xr.DataArray:
    """Get the advective structure function of an xarray dataset with all spacings along
    a specified axis

    Parameters
    ----------
    data : xr.Dataset
        Dataset with velocities u, v, w; dimensions x_caa, y_aca, z_aac
    axis : Axis
        Axis along which to shift the velocities and advections
    spectral : bool, optional
        Use spectral derivatives to calculate advection, by default False

    Returns
    -------
    xr.DataArray
        1D DataArray with structure function values and spacings
    """
    z = data["z_aac"]
    y = data["y_aca"]
    x = data["x_caa"]

    axis_xr = (z, y, x)[axis.value]

    count = len(axis_xr) // 2
    diffs = axis_xr[:count] - axis_xr[0]

    grad_method = GradMethod.spectral if spectral else GradMethod.numpy
    uadv = advection_xr(data, Axis.x, grad_method)
    vadv = advection_xr(data, Axis.y, grad_method)
    wadv = advection_xr(data, Axis.z, grad_method)

    ax_name = axis_name(axis)

    au_lst = []

    for i in range(count):
        roll: Mapping[Hashable, int] = {ax_name: -i}
        du = data["u"].roll(roll) - data["u"]
        dv = data["v"].roll(roll) - data["v"]
        dw = data["w"].roll(roll) - data["w"]
        dau = uadv.roll(roll) - uadv
        dav = vadv.roll(roll) - vadv
        daw = wadv.roll(roll) - wadv
        au = (
            (du * dau + dv * dav + dw * daw)
            .mean(["z_aac", "y_aca", "x_caa"])
            .expand_dims({"dr": [diffs[i]]})
            .rename(f"SF_Au_{ax_name[0]}")
        )
        au_lst.append(au)

    out = (
        xr.concat(au_lst, dim="dr")
        .assign_coords({"dr": diffs.data})
        .assign_attrs(axis=ax_name[0])
    )
    out = out.assign_coords({"k": 2 * np.pi / out.dr})

    return out


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
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
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
