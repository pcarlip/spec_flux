import cupy as cp
import numpy as np
from cupyx.scipy import fft as cufft
from scipy import fft

from .utils import ndarray


def spectral_der(vel_hat: ndarray, k_grid: ndarray) -> ndarray:
    """Approximate dv/dx as IFT(ik FT(v)), with cupy acceleration

    Parameters
    ----------
    vel_hat : np.ndarray | cp.ndarray
        Fourier transform of a velocity component
    k_grid : np.ndarray | cp.ndarray
        meshgrid component of k values along an axis

    Returns
    -------
    np.ndarray | cp.ndarray
        Realspace derivative of velocity with respect to the direction from the k-grid
    """
    xp = cp.get_array_module(vel_hat)
    genfft = fft if xp.__name__ == "np" else cufft
    return xp.real(genfft.ifftn(genfft.ifftshift(1j * k_grid * vel_hat)))
