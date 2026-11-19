from typing import Literal

import cupy_xarray
import xarray as xr

from .utils import (
    Axis,
    GradMethod,
    _genfft,
    _krange_fft,
    _xp,
    ndarray,
)


def spectral_der(
    vel: xr.DataArray,
    ds: xr.Dataset,
    der_ax: Axis,
) -> ndarray:
    """Compute the derivative of a velocity field through its Fourier transform

    Parameters
    ----------
    vel : xr.DataArray
        Velocity field
    ds : xr.Dataset
        Simulation dataset, used for spacings
    der_ax : Axis
        Axis along which to take the derivative

    Returns
    -------
    ndarray
        Derivative of velocity field, as array
    """
    xp = _xp(vel.data)
    genfft = _genfft(vel.data)
    k_ranges = _krange_fft(ds)
    k_mesh: tuple[ndarray, ndarray, ndarray] = xp.meshgrid(*k_ranges, indexing="ij")
    vel_hat: ndarray = genfft.fftshift(genfft.fftn(vel.data))
    return xp.real(genfft.ifftn(genfft.ifftshift(1j * k_mesh[der_ax.value] * vel_hat)))


def advection_xr(
    data: xr.Dataset,
    axis: Axis,
    method: GradMethod,
    ax_names: tuple[str, str, str] = ("z_aac", "y_aca", "x_caa"),
    edge_order: Literal[1, 2] = 1,
) -> xr.DataArray:
    """Calculate (realspace) advection of a velocity array

    Parameters
    ----------
    data : xr.Dataset
        Dataset with velocity and (optionally) advection data
    axis : Axis
        Velocity direction to use
    method : GradMethod
        Method for calculating velocity gradients
    ax_names : tuple[str, str, str], optional
        names of z, y, and x axes in dataset, by default ("z_aac", "y_aca", "x_caa")
    edge_order : Literal[1, 2], optional
        Order of finite difference gradients if using GradMethod.numpy, by default 1

    Returns
    -------
    xr.DataArray
        xarray object containing advection along the specified axis

    Raises
    ------
    Exception
        Attempt to use GradMethod.oceananigans (precalculated advection) when
        input data doesn't include that information
    """
    vel = (data["w"], data["v"], data["u"])[axis.value]
    if method == GradMethod.precalc:
        if "uadv" in data and "vadv" in data and "wadv" in data:
            adv = (data["wadv"], data["vadv"], data["uadv"])[axis.value]
        else:
            raise Exception("Include advection arrays to use precalculated gradients")
    elif method == GradMethod.numpy:
        adv = (
            data["u"] * vel.differentiate(ax_names[2], edge_order)
            + data["v"] * vel.differentiate(ax_names[1], edge_order)
            + data["w"] * vel.differentiate(ax_names[0], edge_order)
        )
    else:
        adv = (
            data["w"] * spectral_der(vel, data, Axis.z)
            + data["v"] * spectral_der(vel, data, Axis.y)
            + data["u"] * spectral_der(vel, data, Axis.x)
        )
    return adv
