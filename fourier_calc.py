from .advection import spectral_der
from .utils import Axis, GradMethod, SimData, SimDataLite, meshgrid_sel, ndarray, xp_fft


def pi_int_dir_spectral(
    vel: ndarray, dx: float, data: SimData | SimDataLite, k_range: ndarray
) -> ndarray:
    """Generate a component of  array to integrate: Re[FT(u)* • FT((u•∇)u)],
    with cupy acceleration, using spectral estimate of gradients

    Parameters
    ----------
    vel : cp.ndarray
        Realspace velocity component
    dx : float
        grid spacing
    data : SimData | SimDataLite
        Object containing realspace velocity components
    k_range : cp.ndarray
        Range of k-values associated with the grid size

    Returns
    -------
    cp.ndarray
        Directional component of Re[FT(u)* • FT((u•∇)u)]
    """
    xp, genfft = xp_fft(vel)
    k_mesh = xp.meshgrid(k_range, k_range, k_range)
    vel_hat = genfft.fftshift(genfft.fftn(vel))
    adv_realspace = (
        spectral_der(vel_hat, meshgrid_sel(k_mesh, Axis.x)) * data.u
    )  # u d(vel)/dx
    adv_realspace += (
        spectral_der(vel_hat, meshgrid_sel(k_mesh, Axis.y)) * data.v
    )  # +v d(vel)/dy
    adv_realspace += (
        spectral_der(vel_hat, meshgrid_sel(k_mesh, Axis.z)) * data.w
    )  # +w d(vel)/dz
    adv_spec = genfft.fftshift(genfft.fftn(adv_realspace) * dx**3 / (2 * xp.pi))
    vel_conj = xp.conj(vel_hat) * dx**3 / (2 * xp.pi)
    return xp.real(vel_conj * adv_spec)


def pi_int_dir_finite(vel: ndarray, dx: float, data: SimData | SimDataLite) -> ndarray:
    """Generate a component of  array to integrate: Re[FT(u)* • FT((u•∇)u)],
    with cupy acceleration, using numpy finite element estimate of gradients

    Parameters
    ----------
    vel : cp.ndarray
        Realspace velocity component
    dx : float
        grid spacing
    data : SimData | SimDataLite
        Object containing realspace velocity components

    Returns
    -------
    cp.ndarray
        Directional component of Re[FT(u)* • FT((u•∇)u)]
    """
    xp, genfft = xp_fft(vel)
    adv_realspace = xp.gradient(vel, dx, axis=2) * data.u
    adv_realspace += xp.gradient(vel, dx, axis=1) * data.v
    adv_realspace += xp.gradient(vel, dx, axis=0) * data.w
    adv_spec = genfft.fftshift(genfft.fftn(adv_realspace) * dx**3 / (2 * xp.pi))
    vel_conj = genfft.fftshift(xp.conj(genfft.fftn(vel)) * dx**3 / (2 * xp.pi))
    return xp.real(vel_conj * adv_spec)


def pi_int_dir_ocean(vel: ndarray, dx: float, adv: ndarray) -> ndarray:
    """Generate a component of  array to integrate: Re[FT(u)* • FT((u•∇)u)],
    using provided gradients (e.g. calculated during the simulation)

    Parameters
    ----------
    vel : np.ndarray
        Realspace velocity component
    dx : float
        grid spacing
    adv : np.ndarray
        Realspace component of advection u•∇u

    Returns
    -------
    np.ndarray
        Directional component of Re[FT(u)* • FT((u•∇)u)]
    """
    xp, genfft = xp_fft(vel)
    adv_spec = genfft.fftshift(genfft.fftn(adv) * dx**3 / (2 * xp.pi))
    vel_conj = genfft.fftshift(xp.conj(genfft.fftn(vel)) * dx**3 / (2 * xp.pi))
    return xp.real(vel_conj * adv_spec)


def fourier_prep(
    data: SimData | SimDataLite,
    grad_method: GradMethod = GradMethod.oceananigans,
) -> tuple[ndarray, ndarray]:
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
    xp, genfft = xp_fft(data.u)

    N = len(data.u)
    dx = float(data.x[1] - data.x[0])
    L = N * dx
    dk = 2 * xp.pi / L
    k_range = genfft.fftshift(genfft.fftfreq(N) * N * dk)

    # dx^3 converts DFT to analog to FT, fftshift moves k = 0 to the middle
    # not sure about the factors of 2π, those come from
    # https://github.com/BrodiePearson/Paper_Bessel_SF_Method/blob/main/analysis/Calculate_Spectral_Fluxes_2D.m

    if grad_method == GradMethod.spectral:
        pi_int = pi_int_dir_spectral(data.u, dx, data, k_range)
        pi_int += pi_int_dir_spectral(data.v, dx, data, k_range)
        pi_int += pi_int_dir_spectral(data.w, dx, data, k_range)
    elif grad_method == GradMethod.numpy:
        pi_int = pi_int_dir_finite(data.u, dx, data)
        pi_int += pi_int_dir_finite(data.v, dx, data)
        pi_int += pi_int_dir_finite(data.w, dx, data)
    elif grad_method == GradMethod.oceananigans:
        if type(data) is SimData:
            pi_int = pi_int_dir_ocean(data.u, dx, data.uadv)
            pi_int += pi_int_dir_ocean(data.v, dx, data.vadv)
            pi_int += pi_int_dir_ocean(data.w, dx, data.wadv)
        else:
            raise Exception(
                "Include advection arrays to use Oceananigans or other precalculated gradients"
            )
    else:
        raise Exception("Something is wrong with your gradient method choice")

    # you can get Π by integrating Re[FT(u)* • FT((u•∇)u)]

    k_mesh = xp.meshgrid(k_range, k_range, k_range)
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
