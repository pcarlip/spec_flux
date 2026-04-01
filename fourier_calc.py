from collections.abc import Callable

from .advection import spectral_der
from .utils import Axis, GradMethod, SimData, SimDataLite, ndarray, xp_fft


def pi_int_dir_spectral(
    data: SimData | SimDataLite,
    axis: Axis,
    spacings: tuple[float, ...],
    k_ranges: tuple[ndarray, ndarray, ndarray],
) -> ndarray:
    """Generate a component of  array to integrate: Re[FT(u)* • FT((u•∇)u)],
    with optional cupy acceleration, using spectral estimate of gradients

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

    Returns
    -------
    ndarray
        Directional component of Re[FT(u)* • FT((u•∇)u)]
    """
    vel = (data.w, data.v, data.u)[axis.value]
    xp, genfft = xp_fft(vel)
    k_mesh = xp.meshgrid(*k_ranges, indexing="ij")
    vel_hat = genfft.fftshift(genfft.fftn(vel))
    adv_realspace = spectral_der(vel_hat, k_mesh[2]) * data.u  # u d(vel)/dx
    adv_realspace += spectral_der(vel_hat, k_mesh[1]) * data.v  # +v d(vel)/dy
    adv_realspace += spectral_der(vel_hat, k_mesh[0]) * data.w  # +w d(vel)/dz
    dxdydz = spacings[0] * spacings[1] * spacings[2]
    adv_spec = genfft.fftshift(genfft.fftn(adv_realspace) * dxdydz / (2 * xp.pi))
    vel_conj = xp.conj(vel_hat) * dxdydz / (2 * xp.pi)
    return xp.real(vel_conj * adv_spec)


def pi_int_dir_finite(
    data: SimData | SimDataLite,
    axis: Axis,
    spacings: tuple[float, ...],
    k_ranges: tuple[ndarray, ndarray, ndarray],
) -> ndarray:
    """Generate a component of  array to integrate: Re[FT(u)* • FT((u•∇)u)],
    with optional cupy acceleration, using numpy finite element estimate of gradients

    Parameters
    ----------
    data : SimData | SimDataLite
        Object containing realspace velocity components
    axis : Axis
        Axis along which to generate the component
    spacings : tuple[float, float, float]
        Grid spacings
    k_ranges : tuple[ndarray, ndarray, ndarray]
        Unused, for API compatability

    Returns
    -------
    ndarray
        Directional component of Re[FT(u)* • FT((u•∇)u)]
    """
    vel = (data.w, data.v, data.u)[axis.value]
    xp, genfft = xp_fft(vel)
    adv_realspace = xp.gradient(vel, spacings[2], axis=2) * data.u
    adv_realspace += xp.gradient(vel, spacings[1], axis=1) * data.v
    adv_realspace += xp.gradient(vel, spacings[0], axis=0) * data.w
    dxdydz = spacings[0] * spacings[1] * spacings[2]
    adv_spec = genfft.fftshift(genfft.fftn(adv_realspace) * dxdydz / (2 * xp.pi))
    vel_conj = genfft.fftshift(xp.conj(genfft.fftn(vel)) * dxdydz / (2 * xp.pi))
    return xp.real(vel_conj * adv_spec)


def pi_int_dir_ocean(
    data: SimData,
    axis: Axis,
    spacings: tuple[float, ...],
    k_ranges: tuple[ndarray, ndarray, ndarray],
) -> ndarray:
    """Generate a component of  array to integrate: Re[FT(u)* • FT((u•∇)u)],
    using provided gradients (e.g. calculated during the simulation), with optional
    cupy acceleration

    Parameters
    ----------
    data : SimData | SimDataLite
        Object containing realspace velocity and advection components
    axis : Axis
        Axis along which to generate the component
    spacings : tuple[float, float, float]
        Grid spacings
    k_ranges : tuple[ndarray, ndarray, ndarray]
        Unused, for API compatability

    Returns
    -------
    ndarray
        Directional component of Re[FT(u)* • FT((u•∇)u)]
    """
    vel = (data.w, data.v, data.u)[axis.value]
    adv = (data.wadv, data.vadv, data.uadv)[axis.value]
    xp, genfft = xp_fft(vel)
    dxdydz = spacings[0] * spacings[1] * spacings[2]
    adv_spec = genfft.fftshift(genfft.fftn(adv) * dxdydz / (2 * xp.pi))
    vel_conj = genfft.fftshift(xp.conj(genfft.fftn(vel)) * dxdydz / (2 * xp.pi))
    return xp.real(vel_conj * adv_spec)


def pi_int_func(
    data: SimData | SimDataLite,
    spacings: tuple[float, ...],
    k_ranges: tuple[ndarray, ndarray, ndarray],
    directional: Callable,
) -> ndarray:
    """Generate array of integrand values for fourier-derived spectral flux

    Parameters
    ----------
    data : SimData | SimDataLite
        Dataclass with velocity and (optionally) advection data
    spacings : tuple[float, ...]
        Grid spacings
    ranges : tuple[ndarray, ndarray, ndarray]
        Range of m-values (z), l-values (y), k-values (x) associated with the grid size
    directional : Callable
        Function to calculate directional component of integrand
        Must have signature (SimData | SimDataLite, Axis, spacings, ranges)

    Returns
    -------
    ndarray
        Array of integrand values
    """
    pi_int = directional(data, Axis.x, spacings, k_ranges)
    pi_int += directional(data, Axis.y, spacings, k_ranges)
    pi_int += directional(data, Axis.z, spacings, k_ranges)
    return pi_int


def fourier_prep(
    data: SimData | SimDataLite,
    grad_method: GradMethod = GradMethod.numpy,
) -> tuple[ndarray, ndarray]:
    """Calculate integrand for spectral flux via fourier methods, with optional cupy
    acceleration

    Parameters
    ----------
    data : SimData
        Dataclass with velocity and advection data

    Returns
    -------
    ndarray
        Array of integrand values, to be appropriately summed to get the spectral flux
    ndarray
        Array of (k^2 + l^2 + m^2) for each integrand value
    """
    xp, genfft = xp_fft(data.u)

    spacings = tuple(float(i[1] - i[0]) for i in [data.x, data.y, data.z])
    ranges = []
    for i in range(3):
        N = data.u.shape[i]
        L = N * spacings[i]
        dk = 2 * xp.pi / L
        k_range = genfft.fftshift(genfft.fftfreq(N) * N * dk)
        ranges.append(k_range)

    ranges = tuple(ranges)
    # dx^3 converts DFT to analog to FT, fftshift moves k = 0 to the middle
    # not sure about the factors of 2π, those come from
    # https://github.com/BrodiePearson/Paper_Bessel_SF_Method/blob/main/analysis/Calculate_Spectral_Fluxes_2D.m

    if grad_method == GradMethod.spectral:
        pi_int = pi_int_func(data, spacings, ranges, pi_int_dir_spectral)
    elif grad_method == GradMethod.numpy:
        pi_int = pi_int_func(data, spacings, ranges, pi_int_dir_finite)
    elif grad_method == GradMethod.oceananigans:
        if type(data) is SimData:
            pi_int = pi_int_func(data, spacings, ranges, pi_int_dir_ocean)
        else:
            raise Exception(
                "Include advection arrays to use Oceananigans or other precalculated gradients"
            )
    else:
        raise Exception("Something is wrong with your gradient method choice")

    # you can get Π by integrating Re[FT(u)* • FT((u•∇)u)]

    k_mesh = xp.meshgrid(*ranges, indexing="ij")
    k_grid = k_mesh[0] ** 2 + k_mesh[1] ** 2 + k_mesh[2] ** 2

    return (pi_int, k_grid)


def fourier_int(
    data: SimData | SimDataLite, pi_int: ndarray, klim: float, k_grid: ndarray
) -> float:
    """Calculate spectral flux of energy dissipation through a fourier transform

    Parameters
    ----------
    data : SimData
        Dataclass with velocity and advection data
    pi_int: ndarray
        integrand for spectral flux
    klim : float
        Wavenumber at which spectral flux is calculated

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
