from typing import Literal

from .utils import Axis, GradMethod, SimData, SimDataLite, ndarray, xp_fft


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
    xp, genfft = xp_fft(vel_hat)
    return xp.real(genfft.ifftn(genfft.ifftshift(1j * k_grid * vel_hat)))


def advection(
    data: SimData | SimDataLite,
    axis: Axis,
    spacings: tuple[float, ...],
    k_ranges: tuple[ndarray, ndarray, ndarray],
    method: GradMethod,
    edge_order: Literal[1, 2] = 1,
) -> ndarray:
    """Calculate (realspace) advection of a velocity array

    Parameters
    ----------
    axis : Axis
        Velocity direction to use
    data : SimData | SimDataLite
        Dataclass with velocity and (optionally) advection data
    spacings : tuple[float, ...]
        Grid spacings
    k_ranges : tuple[ndarray, ndarray, ndarray]
        Range of m-values (z), l-values (y), k-values (x) associated with the grid size
    method : GradMethod
        Method for calculating velocity gradients
    edge_order : Literal[1, 2], optional
        Order of finite difference gradients if using GradMethod.numpy, by default 1

    Returns
    -------
    ndarray
        Advection of selected velocity component

    Raises
    ------
    Exception
        Attempt to use GradMethod.oceananigans (precalculated advection) when
        input data doesn't include that information
    """
    vel_lst = (data.w, data.v, data.u)
    vel = vel_lst[axis.value]
    xp, genfft = xp_fft(vel)
    if method == GradMethod.oceananigans:
        if type(data) is SimData:
            adv = (data.wadv, data.vadv, data.uadv)[axis.value]
        else:
            raise Exception(
                "Include advection arrays to use Oceananigans or other precalculated gradients"
            )
    elif method == GradMethod.numpy:
        adv = sum(
            xp.gradient(vel, spacings[i], axis=i, edge_order=edge_order) * vel_lst[i]
            for i in range(3)
        )
    else:  # GradMethod == GradMethod.spectral
        k_mesh = xp.meshgrid(*k_ranges, indexing="ij")
        vel_hat = genfft.fftshift(genfft.fftn(vel))
        adv = sum(spectral_der(vel_hat, k_mesh[i]) * vel_lst[i] for i in range(3))
    return adv
