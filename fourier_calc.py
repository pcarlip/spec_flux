from typing import Literal

import cupy_xarray
import numpy as np
import xarray as xr
import xrft

from .advection import advection, advection_xr
from .utils import (
    Axis,
    GradMethod,
    SimData,
    SimDataLite,
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
    vel = (data.w, data.v, data.u)[axis.value]
    adv_realspace = advection_xr(data, axis, method, edge_order=edge_order)
    vel_hat = np.conj(xrft.fft(vel))
    adv_spec = xrft.fft(adv_realspace)
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
        Order of finite difference gradients, by default 2

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
    pi_int: xr.DataArray = sum(
        pi_int_dir_xr(data, axis, grad_method, edge_order) for axis in Axis
    ).rename("pi_int")  # type: ignore
    k = (pi_int.freq_x_caa**2 + pi_int.freq_y_aca**2 + pi_int.freq_z_aac**2).rename("k")
    return xr.merge([pi_int, k]).assign_attrs({"L": data.x_caa[-1] - data.x_caa[0]})


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


def fourier_int_xr(data: xr.Dataset, klim: float) -> float:
    masked = data["pi_int"].where(data["k"] <= (klim / (2 * np.pi)) ** 2, 0.0)
    return np.real(
        (
            masked.integrate(["freq_x_caa", "freq_y_aca", "freq_z_aac"])
            .as_numpy()
            .data.item()
        )
        / data.L**3
    )
