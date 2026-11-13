from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_W6_single(x: ndarray) -> ndarray:
    r, s, t = x
    return np.array([1, r, s, t, r * t, s * t], dtype=x.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def monoms_W6_multi(x: ndarray) -> ndarray:
    nP = x.shape[0]
    res = np.zeros((nP, 6), dtype=x.dtype)
    for i in prange(nP):
        res[i] = monoms_W6_single(x[i])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def monoms_W6_bulk_multi(x: ndarray) -> ndarray:
    nE, nP = x.shape[:2]
    res = np.zeros((nE, 6, 6), dtype=x.dtype)
    for i in prange(nE):
        for j in range(nP):
            res[i, j] = monoms_W6_single(x[i, j])
    return res


def monoms_W6(x: ndarray) -> ndarray:
    N = len(x.shape)
    if N == 1:
        return monoms_W6_single(x)
    elif N == 2:
        return monoms_W6_multi(x)
    elif N == 3:
        return monoms_W6_bulk_multi(x)
    else:
        raise NotImplementedError
