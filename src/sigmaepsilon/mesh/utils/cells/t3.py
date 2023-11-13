from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_T3(x: ndarray) -> ndarray:
    r, s = x
    return np.array([1, r, s], dtype=float)


@njit(nogil=True, cache=__cache)
def shp_T3(pcoord: ndarray) -> ndarray:
    r, s = pcoord
    return np.array([1 / 3 - r - s, r + 1 / 3, s + 1 / 3], dtype=pcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def shp_T3_multi(pcoords: ndarray) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_T3(pcoords[iP])
    return res


@njit(nogil=True, parallel=False, cache=__cache)
def shape_function_matrix_T3(pcoord: ndarray, ndof: int = 2) -> ndarray:
    eye = np.eye(ndof, dtype=pcoord.dtype)
    shp = shp_T3(pcoord)
    res = np.zeros((ndof, ndof * 3), dtype=pcoord.dtype)
    for i in prange(3):
        res[:, i * ndof : (i + 1) * ndof] = eye * shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_T3_multi(pcoords: ndarray, ndof: int = 2) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, ndof, ndof * 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_T3(pcoords[iP], ndof)
    return res


@njit(nogil=True, cache=__cache)
def dshp_T3(x) -> ndarray:
    return np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_T3_multi(pcoords: ndarray) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_T3(pcoords[iP])
    return res
