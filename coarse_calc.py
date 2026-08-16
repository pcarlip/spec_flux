from collections.abc import Iterable

import cupy as cp
import cupy_xarray
import xarray as xr
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

    smoothed_vels = [gaussian_filter(vel_arrs[i], size, mode="wrap") for i in range(3)]

    for i in range(3):
        for j in range(3):
            tau_1 = gaussian_filter(vel_arrs[i] * vel_arrs[j], size, mode="wrap")
            tau_2 = smoothed_vels[i] * smoothed_vels[j]  # type: ignore
            tau = tau_1 - tau_2
            grad = cp.gradient(vel_arrs[i], dx, axis=j)
            running_sum -= tau * grad

    return cp.mean(running_sum).get()


def pi_cg_gauss_xr(data: xr.Dataset, k: float) -> xr.DataArray:
    dx = float(data["x_caa"][1] - data["x_caa"][0])
    running_sum = xr.DataArray(0.0, {"time": data.time, "k": k})

    size = (1 / k) / dx
    vels = (data["u"], data["v"], data["w"])
    smoothed_vels = [
        xr.apply_ufunc(gaussian_filter, vels[i], kwargs={"sigma": size, "mode": "wrap"})
        for i in range(3)
    ]
    axes = ("x_caa", "y_aca", "z_aac")

    for i in range(3):
        for j in range(3):
            tau_1 = xr.apply_ufunc(
                gaussian_filter, vels[i] * vels[j], kwargs={"sigma": size, "mode": "wrap"}
            )
            tau_2 = smoothed_vels[i] * smoothed_vels[j]
            tau = tau_1 - tau_2
            grad = vels[i].differentiate(axes[j], 2)
            running_sum -= cp.mean(tau * grad).data.get()

    return running_sum


def pi_cg_lst_xr(data: xr.Dataset, k_cg: Iterable[float]) -> xr.DataArray:
    return xr.concat([pi_cg_gauss_xr(data, k) for k in k_cg], "k")


def pi_cg_gauss_nd(
    data: xr.Dataset, k: float, skip_dims: tuple[str, ...] = ("time",)
) -> xr.DataArray:
    use_dims = [i for i in data.dims if i not in skip_dims]
    axes = [data.u.dims.index(i) for i in use_dims]
    vels = (data["u"], data["v"], data["w"])
    dx = float(data["x_caa"][1] - data["x_caa"][0])
    size = (1 / k) / dx
    gauss_kwargs = {"sigma": size, "mode": "wrap", "axes": axes}
    smoothed_vels = [
        xr.apply_ufunc(gaussian_filter, vels[i], kwargs=gauss_kwargs) for i in range(3)
    ]
    running_sum = xr.DataArray(0.0, {"time": data.time, "k": k})
    for i in range(3):
        for j in range(3):
            tau_1 = xr.apply_ufunc(
                gaussian_filter, vels[i] * vels[j], kwargs=gauss_kwargs
            )
            tau_2 = smoothed_vels[i] * smoothed_vels[j]
            tau = tau_1 - tau_2
            grad = vels[i].differentiate(axes[j], 2)
            running_sum -= cp.mean(tau * grad).data.get()

    return running_sum


def pi_cg_lst_nd(
    data: xr.Dataset, k_cg: Iterable[float], skip_dims: tuple[str, ...] = ("time",)
) -> xr.DataArray:
    return xr.concat([pi_cg_gauss_nd(data, k, skip_dims) for k in k_cg], "k")
