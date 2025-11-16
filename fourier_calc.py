from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import fft


@dataclass
class SimData:
    """A wrapper to store the relevant data from the NetCDF at a particular time step"""

    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    uadv: np.ndarray
    vadv: np.ndarray
    wadv: np.ndarray
    x: np.ndarray


class GradMethod(Enum):
    oceananigans = 1
    spectral = 2
    numpy = 3


def fourier_prep(
    data: SimData, grad_method: GradMethod = GradMethod.oceananigans
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate integrand for spectral flux via fourier methods

    Parameters
    ----------
    data : SimData
        Dataclass with velocity and advection data

    Returns
    -------
    np.ndarray
        Array of integrand values, to be appropriately summed to get the spectral flux
    np.ndarray
        Array of (k^2 + l^2 + m^2) for each integrand value
    """
    N = len(data.u)
    dx = data.x[1] - data.x[0]
    L = N * dx
    dk = 2 * np.pi / L
    k_range = fft.fftshift(fft.fftfreq(N) * N * dk)

    u_conj = fft.fftshift(np.conj(fft.fftn(data.u)) * dx**3 / (2 * np.pi))
    v_conj = fft.fftshift(np.conj(fft.fftn(data.v)) * dx**3 / (2 * np.pi))
    w_conj = fft.fftshift(np.conj(fft.fftn(data.w)) * dx**3 / (2 * np.pi))
    # dx^3 converts DFT to analog to FT, fftshift moves k = 0 to the middle
    # not sure about the factors of 2π, those come from
    # https://github.com/BrodiePearson/Paper_Bessel_SF_Method/blob/main/analysis/Calculate_Spectral_Fluxes_2D.m

    if grad_method == GradMethod.spectral:
        # see above link for calculations of gradients from FFT of u
        u_hat = fft.fftshift(fft.fftn(data.u))
        v_hat = fft.fftshift(fft.fftn(data.v))
        w_hat = fft.fftshift(fft.fftn(data.w))
        k_mesh = np.meshgrid(k_range, k_range, k_range)
        dudx = np.real(fft.ifftn(fft.ifftshift(1j * k_mesh[2] * u_hat)))
        dudy = np.real(fft.ifftn(fft.ifftshift(1j * k_mesh[0] * u_hat)))
        dudz = np.real(fft.ifftn(fft.ifftshift(1j * k_mesh[1] * u_hat)))
        dvdx = np.real(fft.ifftn(fft.ifftshift(1j * k_mesh[2] * v_hat)))
        dvdy = np.real(fft.ifftn(fft.ifftshift(1j * k_mesh[0] * v_hat)))
        dvdz = np.real(fft.ifftn(fft.ifftshift(1j * k_mesh[1] * v_hat)))
        dwdx = np.real(fft.ifftn(fft.ifftshift(1j * k_mesh[2] * w_hat)))
        dwdy = np.real(fft.ifftn(fft.ifftshift(1j * k_mesh[0] * w_hat)))
        dwdz = np.real(fft.ifftn(fft.ifftshift(1j * k_mesh[1] * w_hat)))
        u_adv_realspace = data.u * dudx + data.v * dudy + data.w * dudz
        v_adv_realspace = data.u * dvdx + data.v * dvdy + data.w * dvdz
        w_adv_realspace = data.u * dwdx + data.v * dwdy + data.w * dwdz
        u_adv = fft.fftshift(fft.fftn(u_adv_realspace) * dx**3 / (2 * np.pi))
        v_adv = fft.fftshift(fft.fftn(v_adv_realspace) * dx**3 / (2 * np.pi))
        w_adv = fft.fftshift(fft.fftn(w_adv_realspace) * dx**3 / (2 * np.pi))
    elif grad_method == GradMethod.numpy:
        dudz, dudy, dudx = np.gradient(data.u, dx)
        dvdz, dvdy, dvdx = np.gradient(data.v, dx)
        dwdz, dwdy, dwdx = np.gradient(data.w, dx)
        u_adv_realspace = data.u * dudx + data.v * dudy + data.w * dudz
        v_adv_realspace = data.u * dvdx + data.v * dvdy + data.w * dvdz
        w_adv_realspace = data.u * dwdx + data.v * dwdy + data.w * dwdz
        u_adv = fft.fftshift(fft.fftn(u_adv_realspace) * dx**3 / (2 * np.pi))
        v_adv = fft.fftshift(fft.fftn(v_adv_realspace) * dx**3 / (2 * np.pi))
        w_adv = fft.fftshift(fft.fftn(w_adv_realspace) * dx**3 / (2 * np.pi))
    else:
        u_adv = fft.fftshift(fft.fftn(data.uadv) * dx**3 / (2 * np.pi))
        v_adv = fft.fftshift(fft.fftn(data.vadv) * dx**3 / (2 * np.pi))
        w_adv = fft.fftshift(fft.fftn(data.wadv) * dx**3 / (2 * np.pi))

    pi_int = np.real(u_conj * u_adv + v_conj * v_adv + w_conj * w_adv)
    # you can get Π by integrating Re[FT(u)* • FT((u•∇)u)]

    k_grid = np.zeros((N, N, N))
    for i, j, k in np.ndindex(N, N, N):
        k_grid[i, j, k] = k_range[i] ** 2 + k_range[j] ** 2 + k_range[k] ** 2

    return (pi_int, k_grid)


def fourier_int(
    data: SimData, pi_int: np.ndarray, klim: float, k_grid: np.ndarray
) -> float:
    """Calculate spectral flux of energy dissipation through a fourier transform

    Parameters
    ----------
    data : SimData
        Dataclass with velocity and advection data
    pi_int: np.ndarray
        integrand for spectral flux
    klim : float
        Wavenumber at which spectral flux is calculated

    Returns
    -------
    float
        Spectral flux at wavenumber klim
    """
    N = len(data.u)
    dx = data.x[1] - data.x[0]
    L = N * dx
    dk = 2 * np.pi / L

    masked_array = np.where(k_grid <= klim**2, pi_int, np.zeros_like(pi_int))

    # I think the L**3 is a normalization condition, I'm not sure why I need the 2π
    # but it doesn't match structure function methods without it
    return np.sum(masked_array) * dk**3 / (2 * np.pi * L**3)
