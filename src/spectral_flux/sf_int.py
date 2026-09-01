from collections.abc import Callable, Iterable

import numpy as np
import xarray as xr
from xrscipy.integrate import simpson as xrsimp

from .utils import ndarray


def conv_xr(
    k: float,
    sf: xr.DataArray,
    transformation: Callable[[float, xr.DataArray], np.typing.ArrayLike],
) -> xr.DataArray:
    """Integrate a 1d structure function against a transformation

    Parameters
    ----------
    k : float
        k-value of transformation
    sf : xr.DataArray
        structure function, with spacing dimension 'dr'
    transformation : Callable[[float, xr.DataArray], np.typing.ArrayLike]
        transformation of k, r

    Returns
    -------
    xr.DataArray
        Integrated structure function at the given k-value
    """
    integrand = (transformation(k, sf.dr) * sf).fillna(0.0)
    return xrsimp(integrand, coord="dr").assign_coords({"k": k})  # type: ignore


def conv_lst_xr(
    k_lst: Iterable[float],
    sf: xr.DataArray,
    transformation: Callable[[float, xr.DataArray], np.typing.ArrayLike],
) -> xr.DataArray:
    """Integrate a 1d structure function against a transformation at a range of k values

    Parameters
    ----------
    k_lst : Iterable[float]
        range of k-values for transformation
    sf : xr.DataArray
        structure function, with spacing dimension 'dr'
    transformation : Callable[[float, xr.DataArray], np.typing.ArrayLike]
        transformation of k, r

    Returns
    -------
    xr.DataArray
        Integrated structure function as a function of k
    """
    vals = [conv_xr(k, sf, transformation) for k in k_lst]
    return xr.concat(vals, "k").assign_coords({"k": k_lst})


def conv_full_xr(
    k: float,
    sf: xr.DataArray,
    transformation: Callable[[float, xr.DataArray], np.typing.ArrayLike],
    axes: Iterable[str] = ("dz_aac", "dy_aca", "dx_caa"),
) -> xr.DataArray:
    """Integrate a 3d structure function against a transformation

    Parameters
    ----------
    k : float
        k-value of transformation
    sf : xr.DataArray
        structure function
    transformation : Callable[[float, xr.DataArray], np.typing.ArrayLike]
        transformation of k, r
    axes: Iterable[str], optional
        names of axes along which to integrate the structure function
        by default ("dz_aac", "dy_aca", "dx_caa")

    Returns
    -------
    xr.DataArray
        Integrated structure function at the given k-value
    """
    dr: xr.DataArray = np.sqrt(sum([sf[ax] ** 2 for ax in axes])).cupy.as_cupy()  # type: ignore
    integrand = (transformation(k, dr) * sf.cupy.as_cupy()).fillna(0.0)
    out = integrand.as_numpy()
    for ax in axes:
        out = xrsimp(out, coord=ax)  # type: ignore
    return out.assign_coords({"k": k})


def conv_lst_full_xr(
    k_lst: Iterable,
    sf: xr.DataArray,
    transformation: Callable[[float, xr.DataArray], np.typing.ArrayLike],
    axes: Iterable[str] = ("dz_aac", "dy_aca", "dx_caa"),
) -> xr.DataArray:
    """Integrate a 3d structure function against a transformation at a range of k values

    Parameters
    ----------
    k_lst : Iterable[float]
        range of k-values for transformation
    sf : xr.DataArray
        structure function, with spacing dimension 'dr'
    transformation : Callable[[float, xr.DataArray], np.typing.ArrayLike]
        transformation of k, r
    axes: Iterable[str], optional
        names of axes along which to integrate the structure function
        by default ("dz_aac", "dy_aca", "dx_caa")

    Returns
    -------
    xr.DataArray
        Integrated structure function as a function of k
    """
    return xr.concat([conv_full_xr(k, sf, transformation, axes) for k in k_lst], "k")


def au_full_trans(k: float, r: ndarray | xr.DataArray) -> np.typing.ArrayLike:
    """Transformation for 3d advective structure function

    Parameters
    ----------
    k : float
        wavenumber
    r : ndarray | xr.DataArray
        range of separation distances

    Returns
    -------
    np.typing.ArrayLike
        transformation at each separation distance
    """
    return (np.sin(k * r) - k * r * np.cos(k * r)) / (4 * (np.pi**2) * r**3)


def lll_full_trans(k: float, r: ndarray | xr.DataArray) -> np.typing.ArrayLike:
    """Transformation for 3d 3rd order longitudinal structure function

    Parameters
    ----------
    k : float
        wavenumber
    r : ndarray | xr.DataArray
        range of separation distances

    Returns
    -------
    np.typing.ArrayLike
        transformation at each separation distance
    """
    return 5 * (np.sin(k * r) - k * r * np.cos(k * r)) / (8 * (np.pi**2) * r**4)


def au_ax_trans(k: float, r: ndarray | xr.DataArray) -> np.typing.ArrayLike:
    """Transformation for 1d advective structure function

    Parameters
    ----------
    k : float
        wavenumber
    r : ndarray | xr.DataArray
        range of separation distances

    Returns
    -------
    np.typing.ArrayLike
        transformation at each separation distance
    """
    return (np.sin(k * r) - k * r * np.cos(k * r)) / (np.pi * r)


def lll_ax_trans(k: float, r: ndarray | xr.DataArray) -> np.typing.ArrayLike:
    """Transformation for 1d 3rd order longitudinal structure function

    Parameters
    ----------
    k : float
        wavenumber
    r : ndarray | xr.DataArray
        range of separation distances

    Returns
    -------
    np.typing.ArrayLike
        transformation at each separation distance
    """
    return (np.sin(k * r) - k * r * np.cos(k * r)) * 5 / (2 * np.pi * r**2)
