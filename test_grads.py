import sys

import cupy as cp
import numpy as np

sys.path.append("..")

from numerics.advection import spectral_der
from numerics.utils import SimDataLite, spacings_krange, xp_fft


def test_spec_grad() -> None:
    u = np.ones((10, 11, 12))
    xp, genfft = xp_fft(u)
    it = np.nditer(u, flags=["multi_index"])
    for _ in it:
        i, j, k = it.multi_index
        u[i, j, k] = (
            np.sin(i * np.pi / 10)
            * np.sin(j * 2 * np.pi / 11)
            * np.sin(k * 3 * np.pi / 12)
        )
    x = np.arange(12)
    y = np.arange(11)
    z = np.arange(10)
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
    # print(np.sqrt(np.mean(dudx_np**2)))
    # print(np.sqrt(np.mean(dudx_spec**2)))
    # note: typical values are ~0.25

    assert np.sum(np.isclose(dudx_spec, dudx_np, rtol=0, atol=0.05)) > dudx_np.size * 0.6
    assert np.sum(np.isclose(dudy_spec, dudy_np, rtol=0, atol=0.05)) > dudx_np.size * 0.6
    assert np.sum(np.isclose(dudz_spec, dudz_np, rtol=0, atol=0.05)) > dudx_np.size * 0.6
    # spectral gradients should be v. good here, since it's sinusoids, but the
    # finite difference gradients can be off, hence the threshold 0.6
