import time
from collections.abc import Hashable, Iterable, Mapping
from itertools import product

import cupy as cp
import cupy_xarray
import numpy as np
import xarray as xr
from numba import njit, prange

from .roll import roll_numba, roll_par
from .utils import Axis, axis_name, ndarray

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
    debug_print: bool = False,
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
    order : int, optional
        power of the velocity differences (e.g. use SF_LL or LLL)
        Default value is 3 (LLL)
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
        if i % 25 == 0 and debug_print:
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


def sf_ln_nd(
    data: xr.Dataset,
    order: int = 3,
    spacing: int = 1,
    offset: int = 0,
    spacing_lst: Iterable[int] | None = None,
    spacing_dict: dict[str, Iterable[int]] | None = None,
    dims: tuple[str, ...] = ("z_aac", "y_aca", "x_caa"),
    vels: tuple[str, ...] = ("w", "v", "u"),
) -> xr.DataArray:
    """Calculate the 3rd order longitudinal structure function for an xarray dataset,
    with all combinations of spacings in some range

    Parameters
    ----------
    data : xr.Dataset
        Dataset with velocities and axes
    order : int, optional
        power of the velocity differences (e.g. use SF_LL or LLL)
        Default value is 3 (LLL)
    spacing : int, optional
        Use every nth spacing along each axis, if spacing_lst or spacing_dict not given,
        by default 1
    offset : int, optional
        Offset the first spacing by m before starting, if spacing_lst or spacing_dict
        not given, by default 0
    spacing_lst : Iterable[int] | None, optional
        List of (integer) spacings to use along all axes, by default None
    spacing_dict : dict[str, Iterable[int]] | None, optional
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
        spacing_comb = product(spacing_lst, repeat=ndim)
    elif spacing_dict is not None:
        spacing_comb = product(*[(spacing_dict[i]) for i in dims])
    else:
        dim_len = [len(data[dim]) // 2 for dim in dims]
        spacing_comb = product(*[range(offset, N, spacing) for N in dim_len])
    # I'm not convinced the *[] does anything here, but it makes vscode properly
    # understand the typing

    diffs = [data[dim][1] - data[dim][0] for dim in dims]

    sf_vals: list[xr.DataArray] = []
    for inds in spacing_comb:
        roll: Mapping[Hashable, int] = {dims[i]: -inds[i] for i in range(ndim)}
        # annotating the type here is also silly, but again necessary for vscode
        sf_arr = xr.zeros_like(data[vels[0]])
        r = np.sqrt(sum([ind**2 for ind in inds]))
        for i in range(ndim):
            du = data[vels[i]].roll(roll) - data[vels[i]]
            if r != 0:
                sf_arr += du * inds[i] / r
        sf_vals.append(
            (sf_arr**order)
            .mean(dim=dims)
            .expand_dims({"d" + dims[i]: [diffs[i] * inds[i]] for i in range(ndim)})
            .rename("SF_Au")
        )

    out = xr.combine_by_coords(sf_vals)
    return out if type(out) is xr.DataArray else out.to_dataarray()


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
    order : int, optional
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


def sf_ln_dir_xr(data: xr.Dataset, axis: Axis, order: int = 3) -> xr.DataArray:
    """Get the nth order structure function of an xarray dataset with all spacings along
    a specified axis

    Parameters
    ----------
    data : xr.Dataset
        Dataset with velocities u, v, w; dimensions x_caa, y_aca, z_aac
    axis : Axis
        Axis along which to shift the velocities and advections
    order : int, optional
        power of the velocity differences (e.g. use SF_LL or LLL)
        Default value is 3 (LLL)

    Returns
    -------
    xr.DataArray
        1D DataArray with structure function values and spacings
    """
    ax_name = axis_name(axis)
    axis_xr = data[ax_name]
    vel = (data["w"], data["v"], data["u"])[axis.value]

    count = len(axis_xr) // 2
    diffs = axis_xr[:count] - axis_xr[0]

    ln_lst = [
        (
            ((vel.roll({ax_name: -i}) - vel) ** order)
            .mean(["z_aac", "y_aca", "x_caa"])
            .expand_dims({"dr": [diffs[i]]})
            .rename(f"SF_{'L' * order}_{ax_name[0]}")
        )
        for i in range(count)
    ]

    out = (
        xr.concat(ln_lst, dim="dr")
        .assign_coords({"dr": diffs.data})
        .assign_attrs(axis=ax_name[0])
    )
    out = out.assign_coords({"k": 2 * np.pi / out.dr})
    return out


def sf_ln_xr(
    data: xr.Dataset,
    order: int = 3,
    spacing: int = 1,
    offset: int = 0,
    spacing_lst: Iterable[int] | None = None,
    spacing_dict: dict[str, Iterable[int]] | None = None,
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
    spacing_lst : Iterable[int] | None, optional
        List of (integer) spacings to use along all axes, by default None
    spacing_dict : dict[str, Iterable[int]] | None, optional
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

    ln_lst = []

    for i, dzi in zip(zind, dz, strict=True):
        if i % 5 == 0 and debug_print:
            print(i)
        for j, dyj in zip(yind, dy, strict=True):
            for k, dxk in zip(xind, dx, strict=True):
                if not (i == 0 and j == 0 and k == 0):
                    roll: Mapping[Hashable, int] = {"z_aac": -i, "y_aca": -j, "x_caa": -k}
                    du = data["u"].roll(roll) - data["u"]
                    dv = data["v"].roll(roll) - data["v"]
                    dw = data["w"].roll(roll) - data["w"]
                    r = np.sqrt(i**2 + j**2 + k**2)
                    du_l = (((du * k + dv * j + dw * i) / r) ** order).mean(
                        ["z_aac", "y_aca", "x_caa"]
                    )
                else:
                    du_l = xr.DataArray(0)
                    if data.cupy.is_cupy:
                        du_l = du_l.as_cupy()
                ln_lst.append(
                    du_l.expand_dims(dz=[dzi], dy=[dyj], dx=[dxk]).rename(
                        f"SF_{'L' * order}"
                    )
                )
    out = xr.combine_by_coords(ln_lst)
    return out if type(out) is xr.DataArray else out.to_dataarray()


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
