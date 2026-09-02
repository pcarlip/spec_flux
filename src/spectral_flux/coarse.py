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
    axes: tuple[str, str, str] = ("z_aac", "y_aca", "x_caa"),
    periodic: tuple[bool, ...] = (True, True, True),
) -> xr.DataArray:

    # <f(s)> = ∫dr G(r)f(s+r), for which I use "gaussian_filter"
    # τ_ij = <u_i u_j> - <u_i> <u_j>
    # Π = -(∂_i <u_j>) τ_ij

    smooth_dims = [i for i in axes if i not in skip_dims]
    smooth_axes = [data[vel_names[0]].dims.index(i) for i in smooth_dims]

    vels = tuple(data[i] for i in vel_names)
    dx = [float(data[axes[i]][1] - data[axes[i]][0]) for i in smooth_axes]
    size = tuple((1 / k) / dxi for dxi in dx)

    modes = ["wrap" if i else "nearest" for i in periodic]
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
    axes: tuple[str, str, str] = ("z_aac", "y_aca", "x_caa"),
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> xr.DataArray:
    return pi_cg_gauss_nd(data, k, vel_names=vel_names, axes=axes, periodic=periodic)


def pi_cg_lst_xr(data: xr.Dataset, k_cg: Iterable[float]) -> xr.DataArray:
    return xr.concat([pi_cg_gauss_xr(data, k) for k in k_cg], "k")


def pi_cg_lst_nd(
    data: xr.Dataset, k_cg: Iterable[float], skip_dims: tuple[str, ...] = ("time",)
) -> xr.DataArray:
    return xr.concat([pi_cg_gauss_nd(data, k, skip_dims) for k in k_cg], "k")
