import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter


def pi_cg_gauss(
    vel_arrs: tuple[cp.ndarray, cp.ndarray, cp.ndarray],
    dx: float,
    k: float,
) -> float:
    """Calculate spectral energy flux through coarse graining with a gaussian filter

    Parameters
    ----------
    vel_arrs : tuple[np.ndarray, np.ndarray, np.ndarray]
        u, v, w
    dx : float
        distance between gridpoints
    k : float
        wavenumber associated with the kernel

    Returns
    -------
    np.floating
        spectral energy flux
    """
    # <f(s)> = ∫dr G(r)f(s+r), for which I use "gaussian_filter"
    # τ_ij = <u_i u_j> - <u_i> <u_j>
    # Π = -(∂_i <u_j>) τ_ij
    running_sum = cp.zeros_like(vel_arrs[0])

    size = (1 / k) / dx

    for i in range(3):
        for j in range(3):
            tau_1 = gaussian_filter(vel_arrs[i] * vel_arrs[j], size, mode="wrap")
            tau_2 = gaussian_filter(vel_arrs[i], size, mode="wrap") * gaussian_filter(
                vel_arrs[j], size, mode="wrap"
            )
            tau = tau_1 - tau_2
            grad = cp.gradient(vel_arrs[i], dx, axis=j)
            running_sum -= tau * grad

    return cp.mean(running_sum).get()
