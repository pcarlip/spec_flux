import cupy as cp
import numpy as np
from cupyx.scipy import fft as cufft
from scipy import fft


def spectral_der_gpu(vel_hat: cp.ndarray, k_grid: cp.ndarray) -> cp.ndarray:
    """Approximate dv/dx as IFT(ik FT(v)), with cupy acceleration

    Parameters
    ----------
    vel_hat : cp.ndarray
        Fourier transform of a velocity component
    k_grid : cp.ndarray
        meshgrid component of k values along an axis

    Returns
    -------
    cp.ndarray
        Realspace derivative of velocity with respect to the direction from the k-grid
    """
    return cp.real(cufft.ifftn(cufft.ifftshift(1j * k_grid * vel_hat)))


def spectral_der_cpu(vel_hat: np.ndarray, k_grid: np.ndarray) -> np.ndarray:
    """Approximate dv/dx as IFT(ik FT(v))

    Parameters
    ----------
    vel_hat : np.ndarray
        Fourier transform of a velocity component
    k_grid : np.ndarray
        meshgrid component of k values along an axis

    Returns
    -------
    np.ndarray
        Realspace derivative of velocity with respect to the direction from the k-grid
    """
    return np.real(fft.ifftn(fft.ifftshift(1j * k_grid * vel_hat)))
