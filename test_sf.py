import sys

import cupy as cp
import fluidsf
import numpy as np

sys.path.append("..")
from analysis.sf_au import sf_au
from analysis.sf_ln import sf_ln


def test_answer():
    a = np.ones(shape=(3, 3, 3))
    b = np.array([0, 1, 2])
    out = sf_au(b, b, b, a, a, a)
    print(out)
    assert np.all(out[1] == 0)
