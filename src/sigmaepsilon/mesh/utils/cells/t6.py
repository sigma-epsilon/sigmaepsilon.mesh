from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_T6(x: ndarray) -> ndarray:
    r, s = x
    return np.array([1, r, s, r**2, s**2, r * s], dtype=float)


@njit(nogil=True, cache=__cache)
def shp_T6(pcoord: ndarray):
    r, s = pcoord[0:2]
    return np.array(
        [
            2.0 * r**2 + 4.0 * r * s - 3.0 * r + 2.0 * s**2 - 3.0 * s + 1.0,
            2.0 * r**2 - 1.0 * r,
            2.0 * s**2 - 1.0 * s,
            -4.0 * r**2 - 4.0 * r * s + 4.0 * r,
            4.0 * r * s,
            -4.0 * r * s - 4.0 * s**2 + 4.0 * s,
        ],
        dtype=pcoord.dtype,
    )


@njit(nogil=True, parallel=True, cache=__cache)
def shp_T6_multi(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 6), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_T6(pcoords[iP])
    return res


@njit(nogil=True, parallel=False, cache=__cache)
def shape_function_matrix_T6(pcoord: ndarray, ndof: int = 2):
    eye = np.eye(ndof, dtype=pcoord.dtype)
    shp = shp_T6(pcoord)
    res = np.zeros((ndof, ndof * 6), dtype=pcoord.dtype)
    for i in prange(6):
        res[:, i * ndof : (i + 1) * ndof] = eye * shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_T6_multi(pcoords: np.ndarray, ndof: int = 2):
    nP = pcoords.shape[0]
    res = np.zeros((nP, ndof, ndof * 6), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_T6(pcoords[iP], ndof)
    return res


@njit(nogil=True, cache=__cache)
def dshp_T6(pcoord):
    r, s = pcoord[0:2]
    return np.array(
        [
            [4.0 * r + 4.0 * s - 3.0, 4.0 * r + 4.0 * s - 3.0],
            [4.0 * r - 1.0, 0],
            [0, 4.0 * s - 1.0],
            [-8.0 * r - 4.0 * s + 4.0, -4.0 * r],
            [4.0 * s, 4.0 * r],
            [-4.0 * s, -4.0 * r - 8.0 * s + 4.0],
        ]
    )


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_T6_multi(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 6, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_T6(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def areas_T6(ecoords: ndarray, qpos: ndarray, qweight: ndarray):
    nE = len(ecoords)
    res = np.zeros(nE, dtype=ecoords.dtype)
    nP = len(qweight)
    for i in range(nP):
        dshp = dshp_T6(qpos[i])
        for iE in prange(nE):
            jac = ecoords[iE].T @ dshp
            djac = np.linalg.det(jac)
            res[iE] += qweight[i] * djac
    return res
