from numba import njit, prange
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def _interpolate_multi(N: ndarray, values_source: ndarray, out: ndarray) -> ndarray:
    nP_target, nP_source = N.shape
    nX = out.shape[0]
    for i in prange(nX):
        out[i] = N @ values_source[i]