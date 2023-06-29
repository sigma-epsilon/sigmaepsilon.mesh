from numba import njit, prange, guvectorize, float64
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_H8_single(x: ndarray) -> ndarray:
    r, s, t = x
    return np.array([1, r, s, t, r * s, r * t, s * t, r * s * t], dtype=x.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def monoms_H8_multi(x: ndarray) -> ndarray:
    nP = x.shape[0]
    res = np.zeros((nP, 8), dtype=x.dtype)
    for i in prange(nP):
        res[i] = monoms_H8_single(x[i])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def monoms_H8_bulk_multi(x: ndarray) -> ndarray:
    nE = x.shape[0]
    res = np.zeros((nE, 8, 8), dtype=x.dtype)
    for i in prange(nE):
        res[i] = monoms_H8_multi(x[i])
    return res


def monoms_H8(x: ndarray) -> ndarray:
    N = len(x.shape)
    if N == 1:
        return monoms_H8_single(x)
    elif N == 2:
        return monoms_H8_multi(x)
    elif N == 3:
        return monoms_H8_bulk_multi(x)
    else:
        raise NotImplementedError


@njit(nogil=True, parallel=True, cache=__cache)
def _pip_H8_bulk_(x: ndarray, tol: float = 1e-12) -> ndarray:
    nE, nP = x.shape[:2]
    res = np.zeros((nP, nE), dtype=np.bool_)
    for i in prange(nP):
        for j in prange(nE):
            c1 = np.all(x[j, i] > (-1 - tol))
            c2 = np.all(x[j, i] < (1 + tol))
            res[i, j] = c1 & c2
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _pip_H8_bulk_knn_(x: ndarray, knn: ndarray, tol: float = 1e-12) -> ndarray:
    nE, nP = x.shape[:2]
    nK = knn.shape[-1]
    res = np.zeros((nP, nE), dtype=np.bool_)
    for i in prange(nP):
        for k in prange(nK):
            c1 = np.all(x[k, i] > (-1 - tol))
            c2 = np.all(x[k, i] < (1 + tol))
            res[i, k] = c1 & c2
    return res


@njit(nogil=True, cache=__cache)
def shp_H8(pcoord: ndarray) -> ndarray:
    r, s, t = pcoord
    return np.array(
        [
            -0.125 * r * s * t
            + 0.125 * r * s
            + 0.125 * r * t
            - 0.125 * r
            + 0.125 * s * t
            - 0.125 * s
            - 0.125 * t
            + 0.125,
            0.125 * r * s * t
            - 0.125 * r * s
            - 0.125 * r * t
            + 0.125 * r
            + 0.125 * s * t
            - 0.125 * s
            - 0.125 * t
            + 0.125,
            -0.125 * r * s * t
            + 0.125 * r * s
            - 0.125 * r * t
            + 0.125 * r
            - 0.125 * s * t
            + 0.125 * s
            - 0.125 * t
            + 0.125,
            0.125 * r * s * t
            - 0.125 * r * s
            + 0.125 * r * t
            - 0.125 * r
            - 0.125 * s * t
            + 0.125 * s
            - 0.125 * t
            + 0.125,
            0.125 * r * s * t
            + 0.125 * r * s
            - 0.125 * r * t
            - 0.125 * r
            - 0.125 * s * t
            - 0.125 * s
            + 0.125 * t
            + 0.125,
            -0.125 * r * s * t
            - 0.125 * r * s
            + 0.125 * r * t
            + 0.125 * r
            - 0.125 * s * t
            - 0.125 * s
            + 0.125 * t
            + 0.125,
            0.125 * r * s * t
            + 0.125 * r * s
            + 0.125 * r * t
            + 0.125 * r
            + 0.125 * s * t
            + 0.125 * s
            + 0.125 * t
            + 0.125,
            -0.125 * r * s * t
            - 0.125 * r * s
            - 0.125 * r * t
            - 0.125 * r
            + 0.125 * s * t
            + 0.125 * s
            + 0.125 * t
            + 0.125,
        ]
    )


@njit(nogil=True, parallel=True, cache=__cache)
def shp_H8_multi(pcoords: ndarray) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, 8), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_H8(pcoords[iP])
    return res


@njit(nogil=True, parallel=False, cache=__cache)
def shape_function_matrix_H8(pcoord: ndarray, ndof: int = 3) -> ndarray:
    eye = np.eye(3, dtype=pcoord.dtype)
    shp = shp_H8(pcoord)
    res = np.zeros((ndof, ndof * 8), dtype=pcoord.dtype)
    for i in prange(8):
        res[:, i * ndof : (i + 1) * ndof] = eye * shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_H8_multi(pcoords: ndarray, ndof: int = 3) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, ndof, ndof * 8), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_H8(pcoords[iP], ndof)
    return res


@njit(nogil=True, cache=__cache)
def dshp_H8(pcoord: ndarray) -> ndarray:
    r, s, t = pcoord
    return np.array(
        [
            [
                -0.125 * s * t + 0.125 * s + 0.125 * t - 0.125,
                -0.125 * r * t + 0.125 * r + 0.125 * t - 0.125,
                -0.125 * r * s + 0.125 * r + 0.125 * s - 0.125,
            ],
            [
                0.125 * s * t - 0.125 * s - 0.125 * t + 0.125,
                0.125 * r * t - 0.125 * r + 0.125 * t - 0.125,
                0.125 * r * s - 0.125 * r + 0.125 * s - 0.125,
            ],
            [
                -0.125 * s * t + 0.125 * s - 0.125 * t + 0.125,
                -0.125 * r * t + 0.125 * r - 0.125 * t + 0.125,
                -0.125 * r * s - 0.125 * r - 0.125 * s - 0.125,
            ],
            [
                0.125 * s * t - 0.125 * s + 0.125 * t - 0.125,
                0.125 * r * t - 0.125 * r - 0.125 * t + 0.125,
                0.125 * r * s + 0.125 * r - 0.125 * s - 0.125,
            ],
            [
                0.125 * s * t + 0.125 * s - 0.125 * t - 0.125,
                0.125 * r * t + 0.125 * r - 0.125 * t - 0.125,
                0.125 * r * s - 0.125 * r - 0.125 * s + 0.125,
            ],
            [
                -0.125 * s * t - 0.125 * s + 0.125 * t + 0.125,
                -0.125 * r * t - 0.125 * r - 0.125 * t - 0.125,
                -0.125 * r * s + 0.125 * r - 0.125 * s + 0.125,
            ],
            [
                0.125 * s * t + 0.125 * s + 0.125 * t + 0.125,
                0.125 * r * t + 0.125 * r + 0.125 * t + 0.125,
                0.125 * r * s + 0.125 * r + 0.125 * s + 0.125,
            ],
            [
                -0.125 * s * t - 0.125 * s - 0.125 * t - 0.125,
                -0.125 * r * t - 0.125 * r + 0.125 * t + 0.125,
                -0.125 * r * s - 0.125 * r + 0.125 * s + 0.125,
            ],
        ]
    )


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_H8_multi(pcoords: ndarray) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, 8, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_H8(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def volumes_H8(ecoords: ndarray, qpos: ndarray, qweight: ndarray) -> ndarray:
    nE = ecoords.shape[0]
    volumes = np.zeros(nE, dtype=ecoords.dtype)
    nQ = len(qweight)
    for iQ in range(nQ):
        dshp = dshp_H8(qpos[iQ])
        for i in prange(nE):
            jac = ecoords[i].T @ dshp
            djac = np.linalg.det(jac)
            volumes[i] += qweight[iQ] * djac
    return volumes
