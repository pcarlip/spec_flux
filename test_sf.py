import sys

import cupy as cp
import fluidsf
import numpy as np

sys.path.append("..")
from analysis.sf_au import sf_au
from analysis.sf_ln import sf_ln


def func(x):
    return x + 1


def test_answer():
    assert func(3) == 5
