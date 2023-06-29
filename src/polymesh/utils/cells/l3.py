from numba import njit
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_L3(r: float) -> ndarray:
    return np.array([1, r, r**2], dtype=float)
