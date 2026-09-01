import time
from collections.abc import Collection, Hashable, Mapping
from itertools import product

import cupy as cp
import cupy_xarray
import numpy as np
import xarray as xr

from .advection import advection_xr
from .utils import Axis, GradMethod, axis_name, ndarray, roll_da, shift_da


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


def sf_au_nd(
    data: xr.Dataset,
    spacing: int = 1,
    offset: int = 0,
    spacing_lst: Collection[int] | None = None,
    spacing_dict: dict[str, Collection[int]] | None = None,
    dims: tuple[str, ...] = ("z_aac", "y_aca", "x_caa"),
    vels: tuple[str, ...] = ("w", "v", "u"),
    per_dims: tuple[bool, ...] = (True, True, True),
    debug_print: bool = False,
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
        if debug_print and ninds[0] % 5 == 0:
            print(ninds[0], flush=True)
            print(time.ctime(), flush=True)
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
            print(i, flush=True)
            print(time.ctime(), flush=True)
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


def sf_au_dir_xr(
    data: xr.Dataset,
    axis: Axis,
    grad_method: GradMethod = GradMethod.numpy,
    periodic: bool = True,
) -> xr.DataArray:
    """Get the advective structure function of an xarray dataset with all spacings along
    a specified axis

    Parameters
    ----------
    data : xr.Dataset
        Dataset with velocities u, v, w; dimensions x_caa, y_aca, z_aac
    axis : Axis
        Axis along which to shift the velocities and advections
    grad_method : GradMethod, optional
        How to calculate derivatives for advection, by default GradMethod.numpy
    periodic : bool, optional
        Whether grid is periodic along the given axis, by default True

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

    uadv = advection_xr(data, Axis.x, grad_method)
    vadv = advection_xr(data, Axis.y, grad_method)
    wadv = advection_xr(data, Axis.z, grad_method)

    ax_name = axis_name(axis)

    au_lst = []

    roll_func = roll_da if periodic else shift_da

    for i in range(count):
        roll: Mapping[Hashable, int] = {ax_name: -i}
        du = roll_func(data["u"], roll) - data["u"]
        dv = roll_func(data["v"], roll) - data["v"]
        dw = roll_func(data["w"], roll) - data["w"]
        dau = roll_func(uadv, roll) - uadv
        dav = roll_func(vadv, roll) - vadv
        daw = roll_func(wadv, roll) - wadv
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
