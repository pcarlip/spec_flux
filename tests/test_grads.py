import cupy as cp
import numpy as np


from spec_flux.advection import spectral_der
from spec_flux.utils import SimDataLite, spacings_krange, xp_fft


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
    v = u.copy()
    w = u.copy()
    data = SimDataLite(u, v, w, x, y, z)

    spacings, k_ranges = spacings_krange(data)
    k_mesh = np.meshgrid(*k_ranges, indexing="ij")
    u_hat = genfft.fftshift(genfft.fftn(u))

    dudx_spec = spectral_der(u_hat, k_mesh[2])
    dudy_spec = spectral_der(u_hat, k_mesh[1])
    dudz_spec = spectral_der(u_hat, k_mesh[0])
    dudx_np = xp.gradient(u, spacings[2], axis=2, edge_order=2)
    dudy_np = xp.gradient(u, spacings[1], axis=1, edge_order=2)
    dudz_np = xp.gradient(u, spacings[0], axis=0, edge_order=2)

    assert np.sum(np.isclose(dudx_spec, dudx_np, atol=0, rtol=0.05)) / dudx_np.size > frac
    assert np.sum(np.isclose(dudy_spec, dudy_np, atol=0, rtol=0.05)) / dudx_np.size > frac
    assert np.sum(np.isclose(dudz_spec, dudz_np, atol=0, rtol=0.05)) / dudx_np.size > frac


def test_spec_grad_cp() -> None:
    L = 500
    M = 501
    N = 502
    frac = 0.9
    x = cp.arange(N)
    y = cp.arange(M)
    z = cp.arange(L)
    xp, genfft = xp_fft(x)
    grid = xp.meshgrid(z, y, x, indexing="ij")
    u = (
        xp.sin(grid[0] * np.pi / L)
        * xp.sin(grid[1] * np.pi / M)
        * xp.sin(grid[2] * np.pi / N)
    )
    v = u.copy()
    w = u.copy()
    data = SimDataLite(u, v, w, x, y, z)

    spacings, k_ranges = spacings_krange(data)
    k_mesh = xp.meshgrid(*k_ranges, indexing="ij")
    u_hat = genfft.fftshift(genfft.fftn(u))

    dudx_spec = spectral_der(u_hat, k_mesh[2])
    dudy_spec = spectral_der(u_hat, k_mesh[1])
    dudz_spec = spectral_der(u_hat, k_mesh[0])
    dudx_np = xp.gradient(u, spacings[2], axis=2, edge_order=2)
    dudy_np = xp.gradient(u, spacings[1], axis=1, edge_order=2)
    dudz_np = xp.gradient(u, spacings[0], axis=0, edge_order=2)

    assert xp.sum(xp.isclose(dudx_spec, dudx_np, atol=0, rtol=0.05)) / dudx_np.size > frac
    assert xp.sum(xp.isclose(dudy_spec, dudy_np, atol=0, rtol=0.05)) / dudx_np.size > frac
    assert xp.sum(xp.isclose(dudz_spec, dudz_np, atol=0, rtol=0.05)) / dudx_np.size > frac
