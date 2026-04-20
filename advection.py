from typing import Literal

import cupy_xarray
import xarray as xr

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
    data: SimData | SimDataLite | xr.Dataset,
    axis: Axis,
    spacings: tuple[float, ...],
    k_ranges: tuple[ndarray, ndarray, ndarray],
    method: GradMethod,
    edge_order: Literal[1, 2] = 1,
) -> ndarray:
    """Calculate (realspace) advection of a velocity array

    Parameters
    ----------
    data : SimData | SimDataLite | xr.Dataset
        Dataclass with velocity and (optionally) advection data
    axis : Axis
        Velocity direction to use
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
    if type(data) is xr.Dataset:
        vel_lst = (data.w.data, data.v.data, data.u.data)
    else:
        vel_lst = (data.w, data.v, data.u)
    vel = vel_lst[axis.value]
    xp, genfft = xp_fft(vel)
    if method == GradMethod.oceananigans:
        if type(data) is SimData:
            adv = (data.wadv, data.vadv, data.uadv)[axis.value]
        elif type(data) is xr.Dataset and "uadv" in data:
            adv = (data.wadv.data, data.vadv.data, data.uadv.data)[axis.value]
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


def advection_xr(
    data: xr.Dataset,
    axis: Axis,
    k_ranges: tuple[ndarray, ndarray, ndarray],
    method: GradMethod,
    ax_names: tuple[str, str, str] = ("z_aac", "y_aca", "x_caa"),
    edge_order: Literal[1, 2] = 1,
) -> xr.Variable:
    """Calculate (realspace) advection of a velocity array

    Parameters
    ----------
    data : xr.Dataset
        Dataset with velocity and (optionally) advection data
    axis : Axis
        Velocity direction to use
    k_ranges : tuple[ndarray, ndarray, ndarray]
        Range of m-values (z), l-values (y), k-values (x) associated with the grid size
    method : GradMethod
        Method for calculating velocity gradients
    ax_names : tuple[str, str, str], optional
        names of z, y, and x axes in dataset, by default ("z_aac", "y_aca", "x_caa")
    edge_order : Literal[1, 2], optional
        Order of finite difference gradients if using GradMethod.numpy, by default 1

    Returns
    -------
    xr.Variable
        xarray variable containing advection along the specified axis

    Raises
    ------
    Exception
        Attempt to use GradMethod.oceananigans (precalculated advection) when
        input data doesn't include that information
    """
    vel = (data.w, data.v, data.u)[axis.value]
    if method == GradMethod.oceananigans:
        if "uadv" in data and "vadv" in data and "wadv" in data:
            adv = (data.wadv, data.vadv, data.uadv)[axis.value]
        else:
            raise Exception(
                "Include advection arrays to use Oceananigans or other precalculated gradients"
            )
    elif method == GradMethod.numpy:
        adv = (
            data.u * vel.differentiate(ax_names[2], edge_order)
            + data.v * vel.differentiate(ax_names[1], edge_order)
            + data.w * vel.differentiate(ax_names[0], edge_order)
        )
    else:
        xp, genfft = xp_fft(vel.data)
        k_mesh = xp.meshgrid(*k_ranges, indexing="ij")
        vel_hat = genfft.fftshift(genfft.fftn(vel.data))
        adv = (
            spectral_der(vel_hat, k_mesh[0]) * data.w
            + spectral_der(vel_hat, k_mesh[1]) * data.v
            + spectral_der(vel_hat, k_mesh[2]) * data.u
        )
    return adv
