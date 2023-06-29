from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_TET4_single(x: ndarray) -> ndarray:
    r, s, t = x
    return np.array(
        [1, r, s, t, r * s, r * t, s * t, r**2, s**2, t**2], dtype=x.dtype
    )


@njit(nogil=True, parallel=True, cache=__cache)
def monoms_TET4_multi(x: ndarray) -> ndarray:
    nP = x.shape[0]
    res = np.zeros((nP, 4), dtype=x.dtype)
    for i in prange(nP):
        res[i] = monoms_TET4_single(x[i])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def monoms_TET4_bulk_multi(x: ndarray) -> ndarray:
    nE = x.shape[0]
    res = np.zeros((nE, 4, 4), dtype=x.dtype)
    for i in prange(nE):
        res[i] = monoms_TET4_multi(x[i])
    return res


def monoms_TET4(x: ndarray) -> ndarray:
    N = len(x.shape)
    if N == 1:
        return monoms_TET4_single(x)
    elif N == 2:
        return monoms_TET4_multi(x)
    elif N == 3:
        return monoms_TET4_bulk_multi(x)
    else:
        raise NotImplementedError


@njit(nogil=True, cache=__cache)
def shp_TET4(pcoord: ndarray):
    r, s, t = pcoord
    return np.array([1 - r - s - t, r, s, t])


@njit(nogil=True, parallel=True, cache=__cache)
def shp_TET4_multi(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 4), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_TET4(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_TET4(pcoord: np.ndarray, ndof: int = 3):
    eye = np.eye(ndof, dtype=pcoord.dtype)
    shp = shp_TET4(pcoord)
    res = np.zeros((ndof, ndof * 4), dtype=pcoord.dtype)
    for i in prange(4):
        res[:, i * ndof : (i + 1) * ndof] = eye * shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_TET4_multi(pcoords: np.ndarray, ndof: int = 3):
    nP = pcoords.shape[0]
    res = np.zeros((nP, ndof, ndof * 4), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_TET4(pcoords[iP], ndof)
    return res


@njit(nogil=True, cache=__cache)
def dshp_TET4(x):
    return np.array(
        [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_TET4_multi(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 4, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_TET4(pcoords[iP])
    return res
