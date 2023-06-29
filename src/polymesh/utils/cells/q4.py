from numba import njit, prange
import numpy as np
from numpy import ndarray
from neumann import flatten2dC

from ..tri import area_tri_bulk


__cache = True


@njit(nogil=True, cache=__cache)
def monoms_Q4(x: ndarray) -> ndarray:
    r, s = x
    return np.array([1, r, s, r * s], dtype=float)


@njit(nogil=True, parallel=True, cache=__cache)
def area_Q4_bulk(ecoords: np.ndarray):
    nE = len(ecoords)
    res = np.zeros(nE, dtype=ecoords.dtype)
    res += area_tri_bulk(ecoords[:, :3, :])
    res += area_tri_bulk(ecoords[:, np.array([0, 2, 3]), :])
    return res


@njit(nogil=True, cache=__cache)
def shp_Q4(pcoord: np.ndarray):
    r, s = pcoord[:2]
    return np.array(
        [
            [0.25 * (1 - r) * (1 - s)],
            [0.25 * (1 + r) * (1 - s)],
            [0.25 * (1 + r) * (1 + s)],
            [0.25 * (1 - r) * (1 + s)],
        ],
        dtype=pcoord.dtype,
    )


@njit(nogil=True, parallel=True, cache=__cache)
def shp_Q4_multi(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 4), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = flatten2dC(shp_Q4(pcoords[iP]))
    return res


@njit(nogil=True, cache=__cache)
def dshp_Q4(pcoord: ndarray):
    r, s = pcoord[:2]
    return np.array(
        [
            [(s - 1) / 4, (r - 1) / 4],
            [(1 - s) / 4, (-r - 1) / 4],
            [(s + 1) / 4, (r + 1) / 4],
            [(-s - 1) / 4, (1 - r) / 4],
        ],
        dtype=pcoord.dtype,
    )


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_Q4_multi(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 4, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_Q4(pcoords[iP])
    return res


@njit(nogil=True, parallel=False, cache=__cache)
def shape_function_matrix_Q4(pcoord: ndarray, ndof: int = 2) -> ndarray:
    eye = np.eye(ndof, dtype=pcoord.dtype)
    shp = shp_Q4(pcoord)
    res = np.zeros((ndof, ndof * 4), dtype=pcoord.dtype)
    for i in prange(4):
        res[:, i * ndof : (i + 1) * ndof] = eye * shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_Q4_multi(pcoords: ndarray, ndof: int = 2) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, ndof, ndof * 4), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_Q4(pcoords[iP], ndof)
    return res
