from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_L2(r: float) -> ndarray:
    return np.array([1, r], dtype=float)


@njit(nogil=True, cache=__cache)
def shp_L2(r) -> ndarray:
    """
    Evaluates the shape functions at one location in the range [-1, 1].
    """
    return np.array([1 - r, 1 + r]) / 2


@njit(nogil=True, parallel=True, cache=__cache)
def shp_L2_multi(pcoords: ndarray) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_L2(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_L2_multi(pcoords: ndarray, ndof: int = 2) -> ndarray:
    nP = pcoords.shape[0]
    eye = np.eye(ndof, dtype=pcoords.dtype)
    res = np.zeros((nP, ndof, 2 * ndof), dtype=pcoords.dtype)
    for iP in prange(nP):
        shp = shp_L2(pcoords[iP])
        for iN in prange(2):
            res[iP, :, iN * ndof : (iN + 1) * ndof] = eye * shp[iN]
    return res


@njit(nogil=True, cache=__cache)
def dshp_L2(r) -> ndarray:
    return np.array([-1, 1]) / 2


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_L2_multi(pcoords: ndarray) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, 2, 1), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :, 0] = dshp_L2(pcoords[iP])
    return res
