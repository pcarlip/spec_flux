import sys

import cupy as cp
import fluidsf
import numpy as np

sys.path.append("..")
from analysis.sf_au import sf_au
from analysis.sf_ln import sf_ln


def test_minimal() -> None:
    u = np.ones(shape=(3, 3, 3))
    x = np.array([0, 1, 2])
    out = sf_au(x, x, x, u, u, u)
    assert np.all(out[1] == 0)
