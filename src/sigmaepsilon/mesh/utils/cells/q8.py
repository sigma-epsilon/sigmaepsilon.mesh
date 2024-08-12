from numba import njit, prange
import numpy as np
from numpy import ndarray
from sigmaepsilon.math import flatten2dC

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_Q8(x: ndarray) -> ndarray:
    r, s = x
    res = np.array(
        [
            1,
            r,
            s,
            r * s,
            r**2,
            s**2,
            r * s**2,
            s * r**2,
        ],
        dtype=float,
    )
    return res


@njit(nogil=True, cache=__cache)
def shp_Q8(pcoord: np.ndarray) -> ndarray:
    r, s = pcoord[:2]
    res = np.array(
        [
            [
                -0.25 * r**2 * s
                + 0.25 * r**2
                - 0.25 * r * s**2
                + 0.25 * r * s
                + 0.25 * s**2
                - 0.25
            ],
            [
                -0.25 * r**2 * s
                + 0.25 * r**2
                + 0.25 * r * s**2
                - 0.25 * r * s
                + 0.25 * s**2
                - 0.25
            ],
            [
                0.25 * r**2 * s
                + 0.25 * r**2
                + 0.25 * r * s**2
                + 0.25 * r * s
                + 0.25 * s**2
                - 0.25
            ],
            [
                0.25 * r**2 * s
                + 0.25 * r**2
                - 0.25 * r * s**2
                - 0.25 * r * s
                + 0.25 * s**2
                - 0.25
            ],
            [0.5 * r**2 * s - 0.5 * r**2 - 0.5 * s + 0.5],
            [-0.5 * r * s**2 + 0.5 * r - 0.5 * s**2 + 0.5],
            [-0.5 * r**2 * s - 0.5 * r**2 + 0.5 * s + 0.5],
            [0.5 * r * s**2 - 0.5 * r - 0.5 * s**2 + 0.5],
        ],
        dtype=pcoord.dtype,
    )
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shp_Q8_multi(pcoords: np.ndarray) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, 8), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = flatten2dC(shp_Q8(pcoords[iP]))
    return res


@njit(nogil=True, parallel=False, cache=__cache)
def shape_function_matrix_Q8(pcoord: np.ndarray, ndof: int = 2) -> ndarray:
    eye = np.eye(ndof, dtype=pcoord.dtype)
    shp = shp_Q8(pcoord)
    res = np.zeros((ndof, ndof * 8), dtype=pcoord.dtype)
    for i in range(8):
        res[:, i * ndof : (i + 1) * ndof] = eye * shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_Q8_multi(pcoords: np.ndarray, ndof: int = 2) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, ndof, ndof * 8), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_Q8(pcoords[iP], ndof)
    return res


@njit(nogil=True, cache=__cache)
def dshp_Q8(pcoord: np.ndarray) -> ndarray:
    r, s = pcoord[:2]
    res = np.array(
        [
            [
                -0.5 * r * s + 0.5 * r - 0.25 * s**2 + 0.25 * s,
                -0.25 * r**2 - 0.5 * r * s + 0.25 * r + 0.5 * s,
            ],
            [
                -0.5 * r * s + 0.5 * r + 0.25 * s**2 - 0.25 * s,
                -0.25 * r**2 + 0.5 * r * s - 0.25 * r + 0.5 * s,
            ],
            [
                0.5 * r * s + 0.5 * r + 0.25 * s**2 + 0.25 * s,
                0.25 * r**2 + 0.5 * r * s + 0.25 * r + 0.5 * s,
            ],
            [
                0.5 * r * s + 0.5 * r - 0.25 * s**2 - 0.25 * s,
                0.25 * r**2 - 0.5 * r * s - 0.25 * r + 0.5 * s,
            ],
            [
                1.0 * r * s - 1.0 * r,
                0.5 * r**2 - 0.5,
            ],
            [
                0.5 - 0.5 * s**2,
                -1.0 * r * s - 1.0 * s,
            ],
            [
                -1.0 * r * s - 1.0 * r,
                0.5 - 0.5 * r**2,
            ],
            [
                0.5 * s**2 - 0.5,
                1.0 * r * s - 1.0 * s,
            ],
        ],
        dtype=pcoord.dtype,
    )
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_Q8_multi(pcoords: ndarray) -> ndarray:
    """
    Returns the first orderderivatives of the shape functions,
    evaluated at multiple points, according to 'pcoords'.

    ---
    (nP, nNE, 2)
    """
    nP = pcoords.shape[0]
    res = np.zeros((nP, 8, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_Q8(pcoords[iP])
    return res
