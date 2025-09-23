import sys

import cupy as cp
import fluidsf
import numpy as np

sys.path.append("..")
from analysis.sf_au import sf_au
from analysis.sf_ln import sf_ln


def test_answer():
    a = np.ones(shape=(3, 3, 3))
    b = np.ones(shape=3)
    out = sf_au(b, b, b, a, a, a)
    assert np.all(out == 0)
