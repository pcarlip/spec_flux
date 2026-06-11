from collections.abc import Callable, Iterable

import numpy as np
import xarray as xr
from scipy.integrate import simpson
from xrscipy.integrate import simpson as xrsimp

from .utils import Axis, IntMethod, StrAxis, ndarray, xp_fft


def conv_linear(
    k: float,
    sf: dict[str, np.ndarray],
    sf_name: str,
    transformation: Callable,
    axis: StrAxis | None,
) -> float:
    """Internal function, implements integration along a single, specified axis
    for a single k value"""
    x = sf["x-diffs"]

    if axis is not None:
        sf_val = sf[f"SF_{sf_name}_{axis.value}"]
    else:
        sf_val = np.mean(
            [
                sf[f"SF_{sf_name}_x"],
                sf[f"SF_{sf_name}_y"],
                sf[f"SF_{sf_name}_z"],
            ],
            axis=0,
        )

    integrand = np.empty_like(x)

    for i, r in enumerate(x):
        if r != 0:
            integrand[i] = transformation(k, r) * sf_val[i]
        else:
            integrand[i] = 0

    return float(simpson(integrand, x))


def conv_lst(
    k_lst: Iterable,
    sf: dict[str, np.ndarray],
    sf_name: str,
    transformation: Callable,
    axis: StrAxis | None = None,
) -> np.ndarray:
    """Integrate a structure function along an axis to estimate spectral flux

    Parameters
    ----------
    k_lst : Iterable
        List of wavenumbers at which spectral flux is calculated
    sf : dict[str, Any]
        Dict of structure function values, in the format produced by FluidSF
    sf_name : str
        Name of the particular structure function to use (e.g. LLL, advection_velocity)
    transformation : Callable
        Function of k and r to integrate against the structure function
    axis : Axis | None
        Calculate the integral along either only one axis, or averaged across all three
        Default value: None

    Returns
    -------
    np.ndarray
        Spectral flux at each wavenumber k
    """
    return np.array([conv_linear(k, sf, sf_name, transformation, axis) for k in k_lst])


def conv_xr(
    k: float,
    sf: xr.DataArray,
    transformation: Callable[[float, xr.DataArray], np.typing.ArrayLike],
) -> xr.DataArray:
    integrand = (transformation(k, sf.dr) * sf).fillna(0.0)
    return xrsimp(integrand, coord="dr").assign_coords({"k": k})  # type: ignore


def conv_lst_xr(
    k_lst: Iterable[float],
    sf: xr.DataArray,
    transformation: Callable[[float, xr.DataArray], np.typing.ArrayLike],
) -> xr.DataArray:
    vals = [conv_xr(k, sf, transformation) for k in k_lst]
    return xr.concat(vals, "k").assign_coords({"k": k_lst})


def conv_full(
    k: float,
    diffs: tuple[ndarray, ndarray, ndarray],
    sf: ndarray,
    transformation: Callable,
    taper: bool,
    int_method: IntMethod,
) -> float:
    """Internal function, implements integration for a single k value"""

    xp, _ = xp_fft(sf)
    z_lst, y_lst, x_lst = diffs
    dz = float(z_lst[1] - z_lst[0])
    dy = float(y_lst[1] - y_lst[0])
    dx = float(x_lst[1] - x_lst[0])

    integrand = xp.empty_like(sf)

    mesh = xp.meshgrid(z_lst, y_lst, x_lst)
    r = xp.sqrt(mesh[0] ** 2 + mesh[1] ** 2 + mesh[2] ** 2)
    if taper:
        rmax = xp.max(r)
        taper_arr = xp.sin((np.pi / 2) * (1 + r / rmax)) ** 2
        sf_var = sf * taper_arr
    else:
        sf_var = sf

    integrand = sf_var * transformation(k, r)
    integrand[0, 0, 0] = 0.0

    if int_method == IntMethod.addition:
        return float(xp.sum(integrand) * dx * dy * dz)
    else:
        assert int_method == IntMethod.simpson
        if xp.__name__ == "cupy":
            integrand = integrand.get()  # simpson doesn't seem to work for cupy arrays
        return simpson(simpson(simpson(integrand, dx=dx), dx=dy), dx=dz)  # type: ignore


def conv_lst_full(
    k_lst: Iterable,
    diffs: tuple[ndarray, ndarray, ndarray],
    sf: ndarray,
    transformation: Callable,
    taper: bool = False,
    int_method: IntMethod = IntMethod.simpson,
) -> np.ndarray:
    """Integrate a structure function across all combinations of separations
    Compatible with sf_au and sf_ln, accepts numpy or cupy arrays
    note: x is the third index in the velocity and output arrays, z is the first

    Parameters
    ----------
    k_lst : Iterable
        Wavenumbers at which spectral flux is calculated
    diffs : tuple[ndarray, ndarray, ndarray]
        tuple of 1d arrays of separation values
    sf : ndarray
        structure function values for each separation
    transformation : Callable
        Function of k and r to integrate against the structure function

    Returns
    -------
    float
        Spectral flux at each wavenumber k
    """
    return np.array(
        [conv_full(k, diffs, sf, transformation, taper, int_method) for k in k_lst]
    )
