import sys

import cupy as cp
import fluidsf
import numpy as np

sys.path.append("..")
from analysis.sf_au import sf_au
from analysis.sf_ln import sf_ln


def test_minimal() -> None:
    """The structure function of a uniform array must be 0"""
    u = np.ones(shape=(3, 3, 3))
    x = np.array([0, 1, 2])
    out = sf_au(x, x, x, u, u, u)
    assert np.all(out[1] == 0)


def test_rectangle() -> None:
    """Test that we can calculate the structure function of a non-cubic array"""
    u = np.ones(shape=(5, 4, 3))
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2, 3])
    z = np.array([0, 1, 2, 3, 4])
    out = sf_au(x, y, z, u, u, u)
    assert np.all(out[1] == 0)


def test_minimal_cp() -> None:
    """Repeat test_minimal with a cupy array"""
    u = cp.ones(shape=(3, 3, 3))
    x = cp.array([0, 1, 2])
    out = sf_au(x, x, x, u, u, u)
    assert np.all(out[1] == 0)


def test_rectangle_cp() -> None:
    """Repeat test_rectangle with a cupy array"""
    u = cp.ones(shape=(5, 4, 3))
    x = cp.array([0, 1, 2])
    y = cp.array([0, 1, 2, 3])
    z = cp.array([0, 1, 2, 3, 4])
    out = sf_au(x, y, z, u, u, u)
    assert np.all(out[1] == 0)
