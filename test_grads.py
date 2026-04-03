import sys

import cupy as cp
import numpy as np

sys.path.append("..")

from numerics.advection import spectral_der
from numerics.utils import SimDataLite, spacings_krange, xp_fft


def test_spec_grad() -> None:
    L = 100
    M = 101
    N = 102
    frac = 0.5  # note: this improves w/ larger grids
    u = np.ones((L, M, N))
    xp, genfft = xp_fft(u)
    it = np.nditer(u, flags=["multi_index"])
    for _ in it:
        i, j, k = it.multi_index
        u[i, j, k] = (
            np.sin(i * np.pi / L) * np.sin(j * 2 * np.pi / M) * np.sin(k * 3 * np.pi / N)
        )
    x = np.arange(N)
    y = np.arange(M)
    z = np.arange(L)
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
    u = cp.ones((L, M, N))
    xp, genfft = xp_fft(u)
    grid = xp.meshgrid(z, y, x, indexing="ij")
    u *= (
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
