import cupy as cp
import numpy as np
import xarray as xr

from spec_flux.advection import spectral_der
from spec_flux.utils import krange_fft, xp_fft


def test_spec_grad() -> None:
    L = 200
    M = 201
    N = 202
    x = np.arange(N)
    y = np.arange(M)
    z = np.arange(L)
    frac = 0.75  # note: this improves w/ larger grids
    xp, genfft = xp_fft(x)
    grid = xp.meshgrid(z, y, x, indexing="ij")
    u = (
        xp.sin(grid[0] * np.pi / L)
        * xp.sin(grid[1] * np.pi / M)
        * xp.sin(grid[2] * np.pi / N)
    )
    dims = ("z_aac", "y_aca", "x_caa")
    data = xr.Dataset(
        {"u": (dims, u)},
        coords={"z_aac": z, "y_aca": y, "x_caa": x},
    )

    k_ranges = krange_fft(data)
    k_mesh = np.meshgrid(*k_ranges, indexing="ij")
    u_hat = genfft.fftshift(genfft.fftn(u))

    dudx_spec = spectral_der(u_hat, k_mesh[2])
    dudy_spec = spectral_der(u_hat, k_mesh[1])
    dudz_spec = spectral_der(u_hat, k_mesh[0])
    dudx_np = data.u.differentiate("x_caa", 2)
    dudy_np = data.u.differentiate("y_aca", 2)
    dudz_np = data.u.differentiate("z_aac", 2)

    assert (
        np.sum(np.isclose(dudx_spec, dudx_np.data, atol=0, rtol=0.05)) / dudx_np.size
        > frac
    )
    assert (
        np.sum(np.isclose(dudy_spec, dudy_np.data, atol=0, rtol=0.05)) / dudx_np.size
        > frac
    )
    assert (
        np.sum(np.isclose(dudz_spec, dudz_np.data, atol=0, rtol=0.05)) / dudx_np.size
        > frac
    )


def test_spec_grad_cp() -> None:
    L = 500
    M = 501
    N = 502
    x = cp.arange(N)
    y = cp.arange(M)
    z = cp.arange(L)
    frac = 0.9
    xp, genfft = xp_fft(x)
    grid = xp.meshgrid(z, y, x, indexing="ij")  # type: ignore
    u = (
        xp.sin(grid[0] * np.pi / L)
        * xp.sin(grid[1] * np.pi / M)
        * xp.sin(grid[2] * np.pi / N)
    )
    dims = ("z_aac", "y_aca", "x_caa")
    data = xr.Dataset(
        {"u": (dims, u)},
        coords={"z_aac": z, "y_aca": y, "x_caa": x},
    ).cupy.as_cupy()

    k_ranges = krange_fft(data)
    k_mesh = xp.meshgrid(*k_ranges, indexing="ij")
    u_hat = genfft.fftshift(genfft.fftn(u))

    dudx_spec = spectral_der(u_hat, k_mesh[2])
    dudy_spec = spectral_der(u_hat, k_mesh[1])
    dudz_spec = spectral_der(u_hat, k_mesh[0])
    dudx_np = data.u.differentiate("x_caa", 2)
    dudy_np = data.u.differentiate("y_aca", 2)
    dudz_np = data.u.differentiate("z_aac", 2)

    assert (
        np.sum(np.isclose(dudx_spec, dudx_np.data, atol=0, rtol=0.05)) / dudx_np.size
        > frac
    )
    assert (
        np.sum(np.isclose(dudy_spec, dudy_np.data, atol=0, rtol=0.05)) / dudx_np.size
        > frac
    )
    assert (
        np.sum(np.isclose(dudz_spec, dudz_np.data, atol=0, rtol=0.05)) / dudx_np.size
        > frac
    )
