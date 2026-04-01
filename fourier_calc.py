from collections.abc import Callable

from .advection import advection, spectral_der
from .utils import Axis, GradMethod, SimData, SimDataLite, ndarray, xp_fft


def pi_int_dir(
    method: GradMethod,
    data: SimData | SimDataLite,
    axis: Axis,
    spacings: tuple[float, ...],
    k_ranges: tuple[ndarray, ndarray, ndarray],
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

    Returns
    -------
    ndarray
        Directional component of Re[FT(u)* • FT((u•∇)u)]
    """
    vel = (data.w, data.v, data.u)[axis.value]
    xp, genfft = xp_fft(vel)
    vel_hat = genfft.fftshift(genfft.fftn(vel))
    adv_realspace = advection(data, axis, spacings, k_ranges, method)
    dxdydz = spacings[0] * spacings[1] * spacings[2]
    adv_spec = genfft.fftshift(genfft.fftn(adv_realspace) * dxdydz / (2 * xp.pi))
    vel_conj = xp.conj(vel_hat) * dxdydz / (2 * xp.pi)
    return xp.real(vel_conj * adv_spec)


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

    pi_int = sum(pi_int_dir(grad_method, data, axis, spacings, ranges) for axis in Axis)
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
