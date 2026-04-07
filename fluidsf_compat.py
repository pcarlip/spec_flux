import cupy as cp

from .sf_au import sf_au_dir
from .sf_ln import sf_ln_dir
from .utils import Axis, ndarray


def fluidsf_compat(
    u: ndarray,
    v: ndarray,
    w: ndarray,
    x: ndarray,
    y: ndarray,
    z: ndarray,
    sf_names: list[str],
) -> dict[str, ndarray]:
    L = len(z) // 2
    M = len(y) // 2
    N = len(x) // 2

    dz = z[:L] - z[0]
    dy = y[:M] - y[0]
    dx = x[:N] - x[0]

    out = {"x-diffs": dx, "y-diffs": dy, "z-diffs": dz}
    if "ASF_V" in sf_names:
        out["SF_advection_velocity_x"] = sf_au_dir(u, v, w, x, y, z, Axis.x)[1]
        out["SF_advection_velocity_y"] = sf_au_dir(u, v, w, x, y, z, Axis.y)[1]
        out["SF_advection_velocity_z"] = sf_au_dir(u, v, w, x, y, z, Axis.z)[1]
    if "LLL" in sf_names:
        out["SF_LLL_x"] = sf_ln_dir(u, v, w, x, y, z, Axis.x, 3)[1]
        out["SF_LLL_y"] = sf_ln_dir(u, v, w, x, y, z, Axis.y, 3)[1]
        out["SF_LLL_z"] = sf_ln_dir(u, v, w, x, y, z, Axis.z, 3)[1]
    if "LL" in sf_names:
        out["SF_LL_x"] = sf_ln_dir(u, v, w, x, y, z, Axis.x, 2)[1]
        out["SF_LL_y"] = sf_ln_dir(u, v, w, x, y, z, Axis.y, 2)[1]
        out["SF_LL_z"] = sf_ln_dir(u, v, w, x, y, z, Axis.z, 2)[1]

    if cp.get_array_module(u).__name__ == "cupy":
        for key, val in out.items():
            out[key] = val.get()

    return out
