import time
from collections.abc import Collection
from itertools import product

import cupy as cp
import cupy_xarray
import numpy as np
import xarray as xr

from .utils import Axis, axis_name, ndarray, roll_da, shift_da

# steps:
# set of motion amounts: can just use motion of i,j,k for range N//2, for periodic
#   (more precisely L, M, N; might not want to assume cubic array)
# make array of Δx, Δy, Δz
# for each index: roll array (or similar), take differences,
#   apply operation (e.g. power), take mean
#   note: L^n is longitudinal,
#   I need to get the angle and appropriately weight the components:
#   proj_b(a) = a•b^, which is e.g. (u*i+v*j+w*k)/r


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


def sf_ln_nd(
    data: xr.Dataset,
    order: int = 3,
    spacing: int = 1,
    offset: int = 0,
    spacing_lst: Collection[int] | None = None,
    spacing_dict: dict[str, Collection[int]] | None = None,
    dims: tuple[str, ...] = ("z_aac", "y_aca", "x_caa"),
    vels: tuple[str, ...] = ("w", "v", "u"),
    per_dims: tuple[bool, ...] = (True, True, True),
    debug_print: bool = False,
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
    per_dims : tuple[bool, ...], optional
        Axes along which the grid is periodic, which changes the roll operation,
        by default (True, True, True), non-periodic calcs not currently implemented
    debug_print : bool, optional
        Whether to print a regular message with the current iteration of the outer loop
        and timestamp, by default False, not currently implemented

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

    vel_data = [data[vel].transpose(*dims).data for vel in vels]

    xp = cp.get_array_module(vel_data[0])
    out = xp.zeros([len(i) for i in spacing_lsts])

    for inds, ninds in zip(spacing_comb, spacing_enum, strict=True):
        sf_arr = xp.zeros_like(vel_data[0])
        r = np.sqrt(sum([ind**2 for ind in inds]))
        if r != 0:
            for i in range(ndim):
                du = vel_data[i] - xp.roll(vel_data[i], inds, range(ndim))
                sf_arr += du * inds[i] / r
            out[ninds] = xp.mean(sf_arr**order)  # type: ignore

    return xr.DataArray(
        out,
        [("d" + dims[i], diffs[i].data) for i in range(ndim)],
        name=f"SF_{'L' * order}",
    )


def sf_ln_dir_xr(
    data: xr.Dataset, axis: Axis, order: int = 3, periodic: bool = True
) -> xr.DataArray:
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
    periodic : bool, optional
        Whether grid is periodic along the given axis, by default True

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

    roll_func = roll_da if periodic else shift_da

    ln_lst = [
        (
            ((roll_func(vel, {ax_name: -i}) - vel) ** order)
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
    spacing_lst: Collection[int] | None = None,
    spacing_dict: dict[str, Collection[int]] | None = None,
    per_dims: tuple[bool, bool, bool] = (True, True, True),
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
    per_dims : tuple[bool, bool, bool], optional
        Axes along which the grid is periodic, which changes the roll operation,
        by default (True, True, True), non-periodic calcs not currently implemented
    debug_print : bool, optional
        Whether to print a regular message with the current iteration of the outer loop
        and timestamp, by default False

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

    # ln_lst = []
    xp = cp.get_array_module(data["u"].data)
    out = xp.zeros((len(zind), len(yind), len(xind)))

    u = data["u"].data
    v = data["v"].data
    w = data["w"].data

    for ni, i in enumerate(zind):
        if i % 5 == 0 and debug_print:
            print(i, flush=True)
            print(time.ctime(), flush=True)
        for nj, j in enumerate(yind):
            for nk, k in enumerate(xind):
                if not (i == 0 and j == 0 and k == 0):
                    du = xp.roll(u, (-i, -j, -k), axis=(0, 1, 2)) - u
                    dv = xp.roll(v, (-i, -j, -k), axis=(0, 1, 2)) - v
                    dw = xp.roll(w, (-i, -j, -k), axis=(0, 1, 2)) - w
                    r = np.sqrt(i**2 + j**2 + k**2)
                    out[ni, nj, nk] = xp.mean(sf_kernel(du, dv, dw, i, j, k, r, order))

    return xr.DataArray(
        out,
        [("dz_aac", dz), ("dy_aca", dy), ("dx_caa", dx)],
        name=f"SF_{'L' * order}",
    )
