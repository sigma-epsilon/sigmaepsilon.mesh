from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_W18_single(x: ndarray) -> ndarray:
    r, s, t = x
    res = np.array(
        [
            1,
            r,
            s,
            r**2,
            s**2,
            r * s,
            t,
            t * r,
            t * s,
            t * r**2,
            t * s**2,
            t * r * s,
            t**2,
            t**2 * r,
            t**2 * s,
            t**2 * r**2,
            t**2 * s**2,
            t**2 * r * s,
        ],
        dtype=x.dtype,
    )
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def monoms_W18_multi(x: ndarray) -> ndarray:
    nP = x.shape[0]
    res = np.zeros((nP, 18), dtype=x.dtype)
    for i in prange(nP):
        res[i] = monoms_W18_single(x[i])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def monoms_W18_bulk_multi(x: ndarray) -> ndarray:
    nE = x.shape[0]
    res = np.zeros((nE, 18, 18), dtype=x.dtype)
    for i in prange(nE):
        res[i] = monoms_W18_multi(x[i])
    return res


def monoms_W18(x: ndarray) -> ndarray:
    N = len(x.shape)
    if N == 1:
        return monoms_W18_single(x)
    elif N == 2:
        return monoms_W18_multi(x)
    elif N == 3:
        return monoms_W18_bulk_multi(x)
    else:
        raise NotImplementedError
