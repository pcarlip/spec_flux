from collections.abc import Collection, Iterable
from typing import Literal

import cupy_xarray
import numpy as np
import xarray as xr
import xrft
from xrscipy.integrate import cumulative_simpson

from .advection import advection_xr
from .utils import Axis, GradMethod, axis_name


def pi_int_dir_xr(
    data: xr.Dataset,
    axis: Axis,
    method: GradMethod,
    edge_order: Literal[1, 2] = 2,
) -> xr.DataArray:
    """Generate a component of  array to integrate: Re[FT(u)* • FT((u•∇)u)],
    with optional cupy acceleration

    Parameters
    ----------
    data : xr.Dataset
        Dataset with realspace velocity components (on the same grid)
    axis : Axis
        Axis along which to generate the component
    method : GradMethod
        Method for calculating gradients
    edge_order : Literal[1, 2], optional
        Order of finite difference gradients at domain edges, by default 2

    Returns
    -------
    xr.DataArray
        Directional component of Re[FT(u)* • FT((u•∇)u)]
    """
    vel = (data.w, data.v, data.u)[axis.value]
    adv_realspace = advection_xr(data, axis, method, edge_order=edge_order)
    vel_hat = np.conj(xrft.fft(vel, dim=["x_caa", "y_aca", "z_aac"]))
    adv_spec = xrft.fft(adv_realspace, dim=["x_caa", "y_aca", "z_aac"])
    return np.conj(adv_spec * vel_hat)  # type: ignore


def fourier_prep_xr(
    data: xr.Dataset,
    grad_method: GradMethod = GradMethod.numpy,
    edge_order: Literal[1, 2] = 2,
) -> xr.Dataset:
    """Calculate integrand for spectral flux via fourier methods, with optional cupy
    acceleration

    Parameters
    ----------
    data : xr.Dataset
        Dataset with realspace velocity components (on the same grid)
    grad_method : GradMethod, optional
        Method for calculating gradients, by default GradMethod.numpy
    edge_order : Literal[1, 2], optional
        Order of finite difference gradients at domain edges, by default 2

    Returns
    -------
    xr.Dataset
        Dataset with integrand values, k magnitudes
    """
    pi_int = (
        pi_int_dir_xr(data, Axis.x, grad_method, edge_order)
        + pi_int_dir_xr(data, Axis.y, grad_method, edge_order)
        + pi_int_dir_xr(data, Axis.z, grad_method, edge_order)
    ).rename("pi_int")
    k = (pi_int.freq_x_caa**2 + pi_int.freq_y_aca**2 + pi_int.freq_z_aac**2).rename("k")
    out = xr.merge([pi_int, k]).assign_attrs({"L": data.x_caa[-1] - data.x_caa[0]})
    if pi_int.cupy.is_cupy:
        return out.as_cupy()
    else:
        return out


def fourier_int_xr(data: xr.Dataset, klim: float) -> xr.DataArray:
    """Calculate spectral flux of energy dissipation through a fourier transform
    at a specific wavenumber

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing integrand, k magnitudes
    klim : float
        K value to integrate up to

    Returns
    -------
    xr.DataArray
        Spectral flux at klim
    """
    masked = data["pi_int"].where(data["k"] <= (klim / (2 * np.pi)) ** 2, 0.0)
    val = masked.integrate(["freq_x_caa", "freq_y_aca", "freq_z_aac"])
    num = np.real(val.item()) / data.L**3
    return xr.DataArray(num, {"time": data.time, "k": klim})


def fourier_int_xr_lst(data: xr.Dataset, k_lst: Iterable[float]) -> xr.DataArray:
    """Calculate spectral flux of energy dissipation through a fourier transform
    over a list of wavenumbers

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing integrand, k magnitudes
    k_lst : Iterable[float]
        List of k values to integrate to

    Returns
    -------
    xr.DataArray
        Spectral flux as a function of k, at given k values
    """
    return xr.concat([fourier_int_xr(data, k) for k in k_lst], "k")


def van_atta_prep(
    data: xr.Dataset, cor_ax: Axis, vel_ax: Axis, periodic: bool = True
) -> xr.DataArray:
    """Triple product calculation of transfer function using Van Atta and Chen (1969)
    velocities are assumed to be: u along x_caa, v along y_aca, w along z_aac

    Parameters
    ----------
    data : xr.Dataset
        n-dimensional velocity data in all directions, must include the cor_ax
        but need not include other dimensions
    cor_ax : Axis
        axis along which to take the correlations
    vel_ax : Axis
        second axis from which to use velocities (may be the same as cor_ax)
    periodic : bool
        whether to rotate (if periodic) or shift (if not) the velocities along cor_ax

    Returns
    -------
    xr.DataArray
        transfer function as a function of k
    """
    cor_ax_name = axis_name(cor_ax)
    cor_ax_xr = data[cor_ax_name]
    cor_vel = data[("w", "v", "u")[cor_ax.value]]
    shift_vel = data[("w", "v", "u")[vel_ax.value]]
    n_shifts = len(cor_ax_xr) // 2 if periodic else len(cor_ax_xr)
    shift_range = range(-n_shifts, n_shifts + 1)
    dx = cor_ax_xr[1] - cor_ax_xr[0]
    if periodic:
        si1i = xr.concat(
            [
                (shift_vel * cor_vel * shift_vel.roll({cor_ax_name: i}))
                .mean(cor_ax_name)
                .expand_dims({"r": [i * dx]})
                for i in shift_range
            ],
            "r",
        )
    else:
        si1i = xr.concat(
            [
                (shift_vel * cor_vel * shift_vel.shift({cor_ax_name: i}))
                .mean(cor_ax_name, skipna=True)
                .expand_dims({"r": [i * dx]})
                for i in shift_range
            ],
            "r",
        )
    si1i = 0.5 * (si1i - si1i.isel(r=slice(None, None, -1)).data)
    l1 = xrft.fft(si1i, dim="r", real_dim="r").as_numpy()
    l1 = l1.assign_coords(k=("freq_r", l1["freq_r"].data * 2 * np.pi))
    l1 = l1 * 1j * l1["k"] / (2 * np.pi)
    return 4 * l1 - 2 * l1["k"] * l1.differentiate("k")


def van_atta_int(data: xr.DataArray, mean_axes: Collection[str]) -> xr.DataArray:
    out: xr.DataArray = cumulative_simpson(data, coord="k")  # type: ignore
    mean = out.isel(freq_r=slice(None, -1)).mean(mean_axes).real
    norm = mean - mean.isel(freq_r=-1)
    return norm.assign_coords(k=("freq_r", mean["freq_r"].data * 2 * np.pi))
