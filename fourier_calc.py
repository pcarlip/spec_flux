from collections.abc import Collection, Iterable
from typing import Literal

import cupy_xarray
import numpy as np
import xarray as xr
import xrft
from xrscipy.integrate import cumulative_simpson

from .advection import advection, advection_xr
from .utils import (
    Axis,
    GradMethod,
    SimData,
    SimDataLite,
    axis_name,
    ndarray,
    spacings_krange,
    xp_fft,
)


def pi_int_dir(
    data: SimData | SimDataLite,
    axis: Axis,
    spacings: tuple[float, ...],
    k_ranges: tuple[ndarray, ndarray, ndarray],
    method: GradMethod,
    edge_order: Literal[1, 2],
) -> ndarray:
    """Generate a component of  array to integrate: Re[FT(u)* • FT((u•∇)u)],
    with optional cupy acceleration

    Parameters
    ----------
    data : SimData | SimDataLite
        Object containing realspace velocity components
    axis : Axis
        Axis along which to generate the component
    spacings : tuple[float, float, float]
        Grid spacings
    k_ranges : tuple[ndarray, ndarray, ndarray]
        Range of m-values (z), l-values (y), k-values (x) associated with the grid size
    edge_order : Literal[1, 2]
        Order of finite difference gradients


    Returns
    -------
    ndarray
        Directional component of Re[FT(u)* • FT((u•∇)u)]
    """
    vel = (data.w, data.v, data.u)[axis.value]
    xp, genfft = xp_fft(vel)
    vel_hat = genfft.fftshift(genfft.fftn(vel))
    adv_realspace = advection(data, axis, spacings, k_ranges, method, edge_order)
    dxdydz = spacings[0] * spacings[1] * spacings[2]
    adv_spec = genfft.fftshift(genfft.fftn(adv_realspace) * dxdydz / (2 * xp.pi))
    vel_conj = xp.conj(vel_hat) * dxdydz / (2 * xp.pi)
    return xp.real(vel_conj * adv_spec)


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


def fourier_prep(
    data: SimData | SimDataLite,
    grad_method: GradMethod = GradMethod.numpy,
    edge_order: Literal[1, 2] = 2,
) -> tuple[ndarray, ndarray]:
    """Calculate integrand for spectral flux via fourier methods, with optional cupy
    acceleration

    Parameters
    ----------
    data : SimData | SimDataLite
        Dataclass with velocity and (optionally) advection data
    grad_method : GradMethod, optional
        Method for calculating gradients, by default GradMethod.numpy
    edge_order : Literal[1, 2], optional
        Order of finite difference gradients at domain edges, by default 2

    Returns
    -------
    ndarray
        Array of integrand values, to be appropriately summed to get the spectral flux
    ndarray
        Array of (k^2 + l^2 + m^2) for each integrand value
    """
    xp, _ = xp_fft(data.u)

    spacings, ranges = spacings_krange(data)
    # dx^3 converts DFT to analog to FT, fftshift moves k = 0 to the middle
    # not sure about the factors of 2π, those come from
    # https://github.com/BrodiePearson/Paper_Bessel_SF_Method/blob/main/analysis/Calculate_Spectral_Fluxes_2D.m

    pi_int = sum(
        pi_int_dir(data, axis, spacings, ranges, grad_method, edge_order) for axis in Axis
    )
    # you can get Π by integrating Re[FT(u)* • FT((u•∇)u)]

    k_mesh = xp.meshgrid(*ranges, indexing="ij")
    k_grid = k_mesh[0] ** 2 + k_mesh[1] ** 2 + k_mesh[2] ** 2

    return (pi_int, k_grid)


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


def fourier_int(
    data: SimData | SimDataLite, pi_int: ndarray, klim: float, k_grid: ndarray
) -> float:
    """Calculate spectral flux of energy dissipation through a fourier transform

    Parameters
    ----------
    data : SimData | SimDataLite
        Dataclass with velocity and (optionally) advection data
    pi_int : ndarray
        integrand for spectral flux
    klim : float
        Wavenumber at which spectral flux is calculated
    k_grid : ndarray
        Grid of k magnitudes

    Returns
    -------
    float
        Spectral flux at wavenumber klim
    """
    xp, _ = xp_fft(data.u)
    N = len(data.u)
    dx = data.x[1] - data.x[0]
    L = N * dx
    dk = 2 * xp.pi / L

    masked_array = xp.where(k_grid <= klim**2, pi_int, xp.zeros_like(pi_int))

    # I think the L**3 is a normalization condition, I'm not sure why I need the 2π
    # but it doesn't match structure function methods without it
    return float(xp.sum(masked_array) * dk**3 / (2 * xp.pi * L**3))


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
    return mean - mean.isel(freq_r=-1)
