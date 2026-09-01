import cupy as cp
import numpy as np
import xarray as xr

from .sf_au import sf_au_dir_xr
from .sf_ln import sf_ln_dir_xr
from .utils import Axis, ndarray


def fluidsf_compat(
    u: ndarray,
    v: ndarray,
    w: ndarray,
    x: ndarray,
    y: ndarray,
    z: ndarray,
    sf_names: list[str],
) -> dict[str, np.ndarray]:

    dims = ("z_aac", "y_aca", "x_caa")
    ds = xr.Dataset(
        {"u": (dims, u), "v": (dims, v), "w": (dims, w)},
        coords={"z_aac": z, "y_aca": y, "x_caa": x},
    )

    L = len(z) // 2
    M = len(y) // 2
    N = len(x) // 2

    dz = z[:L] - z[0]
    dy = y[:M] - y[0]
    dx = x[:N] - x[0]

    out = {"x-diffs": dx, "y-diffs": dy, "z-diffs": dz}
    if "ASF_V" in sf_names:
        out["SF_advection_velocity_x"] = sf_au_dir_xr(ds, Axis.x).data
        out["SF_advection_velocity_y"] = sf_au_dir_xr(ds, Axis.y).data
        out["SF_advection_velocity_z"] = sf_au_dir_xr(ds, Axis.z).data
    if "LLL" in sf_names:
        out["SF_LLL_x"] = sf_ln_dir_xr(ds, Axis.x, 3).data
        out["SF_LLL_y"] = sf_ln_dir_xr(ds, Axis.y, 3).data
        out["SF_LLL_z"] = sf_ln_dir_xr(ds, Axis.z, 3).data
    if "LL" in sf_names:
        out["SF_LL_x"] = sf_ln_dir_xr(ds, Axis.x, 2).data
        out["SF_LL_y"] = sf_ln_dir_xr(ds, Axis.y, 2).data
        out["SF_LL_z"] = sf_ln_dir_xr(ds, Axis.z, 2).data

    if cp.get_array_module(u).__name__ == "cupy":
        for key, val in out.items():
            out[key] = val.get()

    return out
