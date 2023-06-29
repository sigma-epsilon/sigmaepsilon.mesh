from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_TET10_single(x: ndarray) -> ndarray:
    r, s, t = x
    return np.array(
        [1, r, s, t, r * s, r * t, s * t, r**2, s**2, t**2], dtype=x.dtype
    )


@njit(nogil=True, parallel=True, cache=__cache)
def monoms_TET10_multi(x: ndarray) -> ndarray:
    nP = x.shape[0]
    res = np.zeros((nP, 10), dtype=x.dtype)
    for i in prange(nP):
        res[i] = monoms_TET10_single(x[i])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def monoms_TET10_bulk_multi(x: ndarray) -> ndarray:
    nE = x.shape[0]
    res = np.zeros((nE, 10, 10), dtype=x.dtype)
    for i in prange(nE):
        res[i] = monoms_TET10_multi(x[i])
    return res


def monoms_TET10(x: ndarray) -> ndarray:
    N = len(x.shape)
    if N == 1:
        return monoms_TET10_single(x)
    elif N == 2:
        return monoms_TET10_multi(x)
    elif N == 3:
        return monoms_TET10_bulk_multi(x)
    else:
        raise NotImplementedError
