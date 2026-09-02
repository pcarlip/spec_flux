from collections.abc import Iterable

import cupy as cp
import cupy_xarray
import xarray as xr
from cupyx.scipy.ndimage import gaussian_filter


def pi_cg_gauss_nd(
    data: xr.Dataset,
    k: float,
    skip_dims: tuple[str, ...] = ("time",),
    vel_names: tuple[str, str, str] = ("u", "v", "w"),
    axes: tuple[str, str, str] = ("x_caa", "y_aca", "z_aac"),
    periodic: tuple[bool, ...] | bool = True,
) -> xr.DataArray:
    """Estimate spectral flux via coarse-graining

    Parameters
    ----------
    data : xr.Dataset
        Dataset with 3 velocity components
    k : float
        Wavenumber at which to calculate spectral flux
    skip_dims : tuple[str, ...], optional
        Any dimensions of the velocity arrays not so smooth along, by default ("time",)
    vel_names : tuple[str, str, str], optional
        Names of velocity components, in any order, by default ("u", "v", "w")
    axes : tuple[str, str, str], optional
        Spatial axes of velocity components, in any order,
        by default ("x_caa", "y_aca", "z_aac")
    periodic : tuple[bool, ...] | bool, optional
        Whether each smoothed axis is periodic. If a tuple, must be the same length as
        set of smoothed axes and match the order of the axes as dimensions of the
        velocity DataArrays.
        By default True

    Returns
    -------
    xr.DataArray
        Coarse-graining estimate of spectral flux at the specified wavenumber
    """

    # <f(s)> = ∫dr G(r)f(s+r), for which I use "gaussian_filter"
    # τ_ij = <u_i u_j> - <u_i> <u_j>
    # Π = -(∂_i <u_j>) τ_ij

    smooth_dims = [i for i in axes if i not in skip_dims]
    smooth_axes = [data[vel_names[0]].dims.index(i) for i in smooth_dims]

    vels = tuple(data[i] for i in vel_names)
    dx = [float(data[axes[i]][1] - data[axes[i]][0]) for i in smooth_axes]
    size = tuple((1 / k) / dxi for dxi in dx)

    if isinstance(periodic, tuple):
        periodic_dims = periodic
    else:
        periodic_dims = tuple(periodic for i in range(len(smooth_dims)))
    modes = ["wrap" if i else "nearest" for i in periodic_dims]
    gauss_kwargs = {"sigma": size, "mode": modes, "axes": smooth_axes}

    smoothed_vels = [
        xr.apply_ufunc(gaussian_filter, vels[i], kwargs=gauss_kwargs) for i in range(3)
    ]
    running_sum = xr.DataArray(0.0, {"time": data.time, "k": k})

    for i in range(3):
        for j in range(3):
            tau_1 = xr.apply_ufunc(
                gaussian_filter, vels[i] * vels[j], kwargs=gauss_kwargs
            )
            tau_2 = smoothed_vels[i] * smoothed_vels[j]
            tau = tau_1 - tau_2
            grad = vels[i].differentiate(axes[j], 2)
            running_sum -= cp.mean(tau * grad).data.get()

    return running_sum


def pi_cg_gauss_xr(
    data: xr.Dataset,
    k: float,
    vel_names: tuple[str, str, str] = ("u", "v", "w"),
    axes: tuple[str, str, str] = ("x_caa", "y_aca", "z_aac"),
    periodic: tuple[bool, bool, bool] | bool = True,
) -> xr.DataArray:
    """A special case of `pi_cg_gauss_nd`, in which no spatial dimensions are skipped"""
    return pi_cg_gauss_nd(data, k, vel_names=vel_names, axes=axes, periodic=periodic)


def pi_cg_lst_xr(
    data: xr.Dataset,
    k_cg: Iterable[float],
    vel_names: tuple[str, str, str] = ("u", "v", "w"),
    axes: tuple[str, str, str] = ("x_caa", "y_aca", "z_aac"),
    periodic: tuple[bool, bool, bool] | bool = True,
) -> xr.DataArray:
    """Run `pi_cg_gauss_xr` on a collection of several wavenumbers"""
    return xr.concat(
        [pi_cg_gauss_xr(data, k, vel_names, axes, periodic) for k in k_cg], "k"
    )


def pi_cg_lst_nd(
    data: xr.Dataset,
    k_cg: Iterable[float],
    skip_dims: tuple[str, ...] = ("time",),
    vel_names: tuple[str, str, str] = ("u", "v", "w"),
    axes: tuple[str, str, str] = ("x_caa", "y_aca", "z_aac"),
    periodic: tuple[bool, ...] | bool = True,
) -> xr.DataArray:
    """Run `pi_cg_gauss_nd` on a collection of several wavenumbers"""
    return xr.concat(
        [pi_cg_gauss_nd(data, k, skip_dims, vel_names, axes, periodic) for k in k_cg], "k"
    )
