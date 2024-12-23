from numba import njit, prange
import numpy as np
from numpy import ndarray


__cache = True


@njit(nogil=True, cache=__cache)
def monoms_H27_single(x: ndarray) -> ndarray:
    r, s, t = x
    res = np.array(
        [
            1,
            r,
            s,
            t,
            s * t,
            r * t,
            r * s,
            r * s * t,
            r**2,
            s**2,
            t**2,
            r**2 * s,
            r * s**2,
            r * t**2,
            r**2 * t,
            s**2 * t,
            s * t**2,
            r**2 * s * t,
            r * s**2 * t,
            r * s * t**2,
            r**2 * s**2,
            s**2 * t**2,
            r**2 * t**2,
            r**2 * s**2 * t**2,
            r**2 * s**2 * t,
            r**2 * s * t**2,
            r * s**2 * t**2,
        ],
        dtype=x.dtype,
    )
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def monoms_H27_multi(x: ndarray) -> ndarray:
    nP = x.shape[0]
    res = np.zeros((nP, 27), dtype=x.dtype)
    for i in prange(nP):
        res[i] = monoms_H27_single(x[i])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def monoms_H27_bulk_multi(x: ndarray) -> ndarray:
    nE = x.shape[0]
    res = np.zeros((nE, 27, 27), dtype=x.dtype)
    for i in prange(nE):
        res[i] = monoms_H27_multi(x[i])
    return res


def monoms_H27(x: ndarray) -> ndarray:
    N = len(x.shape)
    if N == 1:
        return monoms_H27_single(x)
    elif N == 2:
        return monoms_H27_multi(x)
    elif N == 3:
        return monoms_H27_bulk_multi(x)
    else:
        raise NotImplementedError


@njit(nogil=True, cache=__cache)
def shp_H27(pcoord):
    r, s, t = pcoord
    res = np.array(
        [
            0.125 * r**2 * s**2 * t**2
            - 0.125 * r**2 * s**2 * t
            - 0.125 * r**2 * s * t**2
            + 0.125 * r**2 * s * t
            - 0.125 * r * s**2 * t**2
            + 0.125 * r * s**2 * t
            + 0.125 * r * s * t**2
            - 0.125 * r * s * t,
            0.125 * r**2 * s**2 * t**2
            - 0.125 * r**2 * s**2 * t
            - 0.125 * r**2 * s * t**2
            + 0.125 * r**2 * s * t
            + 0.125 * r * s**2 * t**2
            - 0.125 * r * s**2 * t
            - 0.125 * r * s * t**2
            + 0.125 * r * s * t,
            0.125 * r**2 * s**2 * t**2
            - 0.125 * r**2 * s**2 * t
            + 0.125 * r**2 * s * t**2
            - 0.125 * r**2 * s * t
            + 0.125 * r * s**2 * t**2
            - 0.125 * r * s**2 * t
            + 0.125 * r * s * t**2
            - 0.125 * r * s * t,
            0.125 * r**2 * s**2 * t**2
            - 0.125 * r**2 * s**2 * t
            + 0.125 * r**2 * s * t**2
            - 0.125 * r**2 * s * t
            - 0.125 * r * s**2 * t**2
            + 0.125 * r * s**2 * t
            - 0.125 * r * s * t**2
            + 0.125 * r * s * t,
            0.125 * r**2 * s**2 * t**2
            + 0.125 * r**2 * s**2 * t
            - 0.125 * r**2 * s * t**2
            - 0.125 * r**2 * s * t
            - 0.125 * r * s**2 * t**2
            - 0.125 * r * s**2 * t
            + 0.125 * r * s * t**2
            + 0.125 * r * s * t,
            0.125 * r**2 * s**2 * t**2
            + 0.125 * r**2 * s**2 * t
            - 0.125 * r**2 * s * t**2
            - 0.125 * r**2 * s * t
            + 0.125 * r * s**2 * t**2
            + 0.125 * r * s**2 * t
            - 0.125 * r * s * t**2
            - 0.125 * r * s * t,
            0.125 * r**2 * s**2 * t**2
            + 0.125 * r**2 * s**2 * t
            + 0.125 * r**2 * s * t**2
            + 0.125 * r**2 * s * t
            + 0.125 * r * s**2 * t**2
            + 0.125 * r * s**2 * t
            + 0.125 * r * s * t**2
            + 0.125 * r * s * t,
            0.125 * r**2 * s**2 * t**2
            + 0.125 * r**2 * s**2 * t
            + 0.125 * r**2 * s * t**2
            + 0.125 * r**2 * s * t
            - 0.125 * r * s**2 * t**2
            - 0.125 * r * s**2 * t
            - 0.125 * r * s * t**2
            - 0.125 * r * s * t,
            -0.25 * r**2 * s**2 * t**2
            + 0.25 * r**2 * s**2 * t
            + 0.25 * r**2 * s * t**2
            - 0.25 * r**2 * s * t
            + 0.25 * s**2 * t**2
            - 0.25 * s**2 * t
            - 0.25 * s * t**2
            + 0.25 * s * t,
            -0.25 * r**2 * s**2 * t**2
            + 0.25 * r**2 * s**2 * t
            + 0.25 * r**2 * t**2
            - 0.25 * r**2 * t
            - 0.25 * r * s**2 * t**2
            + 0.25 * r * s**2 * t
            + 0.25 * r * t**2
            - 0.25 * r * t,
            -0.25 * r**2 * s**2 * t**2
            + 0.25 * r**2 * s**2 * t
            - 0.25 * r**2 * s * t**2
            + 0.25 * r**2 * s * t
            + 0.25 * s**2 * t**2
            - 0.25 * s**2 * t
            + 0.25 * s * t**2
            - 0.25 * s * t,
            -0.25 * r**2 * s**2 * t**2
            + 0.25 * r**2 * s**2 * t
            + 0.25 * r**2 * t**2
            - 0.25 * r**2 * t
            + 0.25 * r * s**2 * t**2
            - 0.25 * r * s**2 * t
            - 0.25 * r * t**2
            + 0.25 * r * t,
            -0.25 * r**2 * s**2 * t**2
            - 0.25 * r**2 * s**2 * t
            + 0.25 * r**2 * s * t**2
            + 0.25 * r**2 * s * t
            + 0.25 * s**2 * t**2
            + 0.25 * s**2 * t
            - 0.25 * s * t**2
            - 0.25 * s * t,
            -0.25 * r**2 * s**2 * t**2
            - 0.25 * r**2 * s**2 * t
            + 0.25 * r**2 * t**2
            + 0.25 * r**2 * t
            - 0.25 * r * s**2 * t**2
            - 0.25 * r * s**2 * t
            + 0.25 * r * t**2
            + 0.25 * r * t,
            -0.25 * r**2 * s**2 * t**2
            - 0.25 * r**2 * s**2 * t
            - 0.25 * r**2 * s * t**2
            - 0.25 * r**2 * s * t
            + 0.25 * s**2 * t**2
            + 0.25 * s**2 * t
            + 0.25 * s * t**2
            + 0.25 * s * t,
            -0.25 * r**2 * s**2 * t**2
            - 0.25 * r**2 * s**2 * t
            + 0.25 * r**2 * t**2
            + 0.25 * r**2 * t
            + 0.25 * r * s**2 * t**2
            + 0.25 * r * s**2 * t
            - 0.25 * r * t**2
            - 0.25 * r * t,
            -0.25 * r**2 * s**2 * t**2
            + 0.25 * r**2 * s**2
            + 0.25 * r**2 * s * t**2
            - 0.25 * r**2 * s
            + 0.25 * r * s**2 * t**2
            - 0.25 * r * s**2
            - 0.25 * r * s * t**2
            + 0.25 * r * s,
            -0.25 * r**2 * s**2 * t**2
            + 0.25 * r**2 * s**2
            + 0.25 * r**2 * s * t**2
            - 0.25 * r**2 * s
            - 0.25 * r * s**2 * t**2
            + 0.25 * r * s**2
            + 0.25 * r * s * t**2
            - 0.25 * r * s,
            -0.25 * r**2 * s**2 * t**2
            + 0.25 * r**2 * s**2
            - 0.25 * r**2 * s * t**2
            + 0.25 * r**2 * s
            - 0.25 * r * s**2 * t**2
            + 0.25 * r * s**2
            - 0.25 * r * s * t**2
            + 0.25 * r * s,
            -0.25 * r**2 * s**2 * t**2
            + 0.25 * r**2 * s**2
            - 0.25 * r**2 * s * t**2
            + 0.25 * r**2 * s
            + 0.25 * r * s**2 * t**2
            - 0.25 * r * s**2
            + 0.25 * r * s * t**2
            - 0.25 * r * s,
            0.5 * r**2 * s**2 * t**2
            - 0.5 * r**2 * s**2
            - 0.5 * r**2 * t**2
            + 0.5 * r**2
            - 0.5 * r * s**2 * t**2
            + 0.5 * r * s**2
            + 0.5 * r * t**2
            - 0.5 * r,
            0.5 * r**2 * s**2 * t**2
            - 0.5 * r**2 * s**2
            - 0.5 * r**2 * t**2
            + 0.5 * r**2
            + 0.5 * r * s**2 * t**2
            - 0.5 * r * s**2
            - 0.5 * r * t**2
            + 0.5 * r,
            0.5 * r**2 * s**2 * t**2
            - 0.5 * r**2 * s**2
            - 0.5 * r**2 * s * t**2
            + 0.5 * r**2 * s
            - 0.5 * s**2 * t**2
            + 0.5 * s**2
            + 0.5 * s * t**2
            - 0.5 * s,
            0.5 * r**2 * s**2 * t**2
            - 0.5 * r**2 * s**2
            + 0.5 * r**2 * s * t**2
            - 0.5 * r**2 * s
            - 0.5 * s**2 * t**2
            + 0.5 * s**2
            - 0.5 * s * t**2
            + 0.5 * s,
            0.5 * r**2 * s**2 * t**2
            - 0.5 * r**2 * s**2 * t
            - 0.5 * r**2 * t**2
            + 0.5 * r**2 * t
            - 0.5 * s**2 * t**2
            + 0.5 * s**2 * t
            + 0.5 * t**2
            - 0.5 * t,
            0.5 * r**2 * s**2 * t**2
            + 0.5 * r**2 * s**2 * t
            - 0.5 * r**2 * t**2
            - 0.5 * r**2 * t
            - 0.5 * s**2 * t**2
            - 0.5 * s**2 * t
            + 0.5 * t**2
            + 0.5 * t,
            -1.0 * r**2 * s**2 * t**2
            + 1.0 * r**2 * s**2
            + 1.0 * r**2 * t**2
            - 1.0 * r**2
            + 1.0 * s**2 * t**2
            - 1.0 * s**2
            - 1.0 * t**2
            + 1.0,
        ]
    )
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shp_H27_multi(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 27), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_H27(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_H27(pcoord: np.ndarray, ndof: int = 3):
    eye = np.eye(ndof, dtype=pcoord.dtype)
    shp = shp_H27(pcoord)
    res = np.zeros((ndof, ndof * 27), dtype=pcoord.dtype)
    for i in prange(27):
        res[:, i * ndof : (i + 1) * ndof] = eye * shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_H27_multi(pcoords: np.ndarray, ndof: int = 3):
    nP = pcoords.shape[0]
    res = np.zeros((nP, ndof, ndof * 27), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_H27(pcoords[iP], ndof)
    return res


@njit(nogil=True, cache=__cache)
def dshp_H27(pcoord):
    r, s, t = pcoord
    res = np.array(
        [
            [
                0.25 * r * s**2 * t**2
                - 0.25 * r * s**2 * t
                - 0.25 * r * s * t**2
                + 0.25 * r * s * t
                - 0.125 * s**2 * t**2
                + 0.125 * s**2 * t
                + 0.125 * s * t**2
                - 0.125 * s * t,
                0.25 * r**2 * s * t**2
                - 0.25 * r**2 * s * t
                - 0.125 * r**2 * t**2
                + 0.125 * r**2 * t
                - 0.25 * r * s * t**2
                + 0.25 * r * s * t
                + 0.125 * r * t**2
                - 0.125 * r * t,
                0.25 * r**2 * s**2 * t
                - 0.125 * r**2 * s**2
                - 0.25 * r**2 * s * t
                + 0.125 * r**2 * s
                - 0.25 * r * s**2 * t
                + 0.125 * r * s**2
                + 0.25 * r * s * t
                - 0.125 * r * s,
            ],
            [
                0.25 * r * s**2 * t**2
                - 0.25 * r * s**2 * t
                - 0.25 * r * s * t**2
                + 0.25 * r * s * t
                + 0.125 * s**2 * t**2
                - 0.125 * s**2 * t
                - 0.125 * s * t**2
                + 0.125 * s * t,
                0.25 * r**2 * s * t**2
                - 0.25 * r**2 * s * t
                - 0.125 * r**2 * t**2
                + 0.125 * r**2 * t
                + 0.25 * r * s * t**2
                - 0.25 * r * s * t
                - 0.125 * r * t**2
                + 0.125 * r * t,
                0.25 * r**2 * s**2 * t
                - 0.125 * r**2 * s**2
                - 0.25 * r**2 * s * t
                + 0.125 * r**2 * s
                + 0.25 * r * s**2 * t
                - 0.125 * r * s**2
                - 0.25 * r * s * t
                + 0.125 * r * s,
            ],
            [
                0.25 * r * s**2 * t**2
                - 0.25 * r * s**2 * t
                + 0.25 * r * s * t**2
                - 0.25 * r * s * t
                + 0.125 * s**2 * t**2
                - 0.125 * s**2 * t
                + 0.125 * s * t**2
                - 0.125 * s * t,
                0.25 * r**2 * s * t**2
                - 0.25 * r**2 * s * t
                + 0.125 * r**2 * t**2
                - 0.125 * r**2 * t
                + 0.25 * r * s * t**2
                - 0.25 * r * s * t
                + 0.125 * r * t**2
                - 0.125 * r * t,
                0.25 * r**2 * s**2 * t
                - 0.125 * r**2 * s**2
                + 0.25 * r**2 * s * t
                - 0.125 * r**2 * s
                + 0.25 * r * s**2 * t
                - 0.125 * r * s**2
                + 0.25 * r * s * t
                - 0.125 * r * s,
            ],
            [
                0.25 * r * s**2 * t**2
                - 0.25 * r * s**2 * t
                + 0.25 * r * s * t**2
                - 0.25 * r * s * t
                - 0.125 * s**2 * t**2
                + 0.125 * s**2 * t
                - 0.125 * s * t**2
                + 0.125 * s * t,
                0.25 * r**2 * s * t**2
                - 0.25 * r**2 * s * t
                + 0.125 * r**2 * t**2
                - 0.125 * r**2 * t
                - 0.25 * r * s * t**2
                + 0.25 * r * s * t
                - 0.125 * r * t**2
                + 0.125 * r * t,
                0.25 * r**2 * s**2 * t
                - 0.125 * r**2 * s**2
                + 0.25 * r**2 * s * t
                - 0.125 * r**2 * s
                - 0.25 * r * s**2 * t
                + 0.125 * r * s**2
                - 0.25 * r * s * t
                + 0.125 * r * s,
            ],
            [
                0.25 * r * s**2 * t**2
                + 0.25 * r * s**2 * t
                - 0.25 * r * s * t**2
                - 0.25 * r * s * t
                - 0.125 * s**2 * t**2
                - 0.125 * s**2 * t
                + 0.125 * s * t**2
                + 0.125 * s * t,
                0.25 * r**2 * s * t**2
                + 0.25 * r**2 * s * t
                - 0.125 * r**2 * t**2
                - 0.125 * r**2 * t
                - 0.25 * r * s * t**2
                - 0.25 * r * s * t
                + 0.125 * r * t**2
                + 0.125 * r * t,
                0.25 * r**2 * s**2 * t
                + 0.125 * r**2 * s**2
                - 0.25 * r**2 * s * t
                - 0.125 * r**2 * s
                - 0.25 * r * s**2 * t
                - 0.125 * r * s**2
                + 0.25 * r * s * t
                + 0.125 * r * s,
            ],
            [
                0.25 * r * s**2 * t**2
                + 0.25 * r * s**2 * t
                - 0.25 * r * s * t**2
                - 0.25 * r * s * t
                + 0.125 * s**2 * t**2
                + 0.125 * s**2 * t
                - 0.125 * s * t**2
                - 0.125 * s * t,
                0.25 * r**2 * s * t**2
                + 0.25 * r**2 * s * t
                - 0.125 * r**2 * t**2
                - 0.125 * r**2 * t
                + 0.25 * r * s * t**2
                + 0.25 * r * s * t
                - 0.125 * r * t**2
                - 0.125 * r * t,
                0.25 * r**2 * s**2 * t
                + 0.125 * r**2 * s**2
                - 0.25 * r**2 * s * t
                - 0.125 * r**2 * s
                + 0.25 * r * s**2 * t
                + 0.125 * r * s**2
                - 0.25 * r * s * t
                - 0.125 * r * s,
            ],
            [
                0.25 * r * s**2 * t**2
                + 0.25 * r * s**2 * t
                + 0.25 * r * s * t**2
                + 0.25 * r * s * t
                + 0.125 * s**2 * t**2
                + 0.125 * s**2 * t
                + 0.125 * s * t**2
                + 0.125 * s * t,
                0.25 * r**2 * s * t**2
                + 0.25 * r**2 * s * t
                + 0.125 * r**2 * t**2
                + 0.125 * r**2 * t
                + 0.25 * r * s * t**2
                + 0.25 * r * s * t
                + 0.125 * r * t**2
                + 0.125 * r * t,
                0.25 * r**2 * s**2 * t
                + 0.125 * r**2 * s**2
                + 0.25 * r**2 * s * t
                + 0.125 * r**2 * s
                + 0.25 * r * s**2 * t
                + 0.125 * r * s**2
                + 0.25 * r * s * t
                + 0.125 * r * s,
            ],
            [
                0.25 * r * s**2 * t**2
                + 0.25 * r * s**2 * t
                + 0.25 * r * s * t**2
                + 0.25 * r * s * t
                - 0.125 * s**2 * t**2
                - 0.125 * s**2 * t
                - 0.125 * s * t**2
                - 0.125 * s * t,
                0.25 * r**2 * s * t**2
                + 0.25 * r**2 * s * t
                + 0.125 * r**2 * t**2
                + 0.125 * r**2 * t
                - 0.25 * r * s * t**2
                - 0.25 * r * s * t
                - 0.125 * r * t**2
                - 0.125 * r * t,
                0.25 * r**2 * s**2 * t
                + 0.125 * r**2 * s**2
                + 0.25 * r**2 * s * t
                + 0.125 * r**2 * s
                - 0.25 * r * s**2 * t
                - 0.125 * r * s**2
                - 0.25 * r * s * t
                - 0.125 * r * s,
            ],
            [
                -0.5 * r * s**2 * t**2
                + 0.5 * r * s**2 * t
                + 0.5 * r * s * t**2
                - 0.5 * r * s * t,
                -0.5 * r**2 * s * t**2
                + 0.5 * r**2 * s * t
                + 0.25 * r**2 * t**2
                - 0.25 * r**2 * t
                + 0.5 * s * t**2
                - 0.5 * s * t
                - 0.25 * t**2
                + 0.25 * t,
                -0.5 * r**2 * s**2 * t
                + 0.25 * r**2 * s**2
                + 0.5 * r**2 * s * t
                - 0.25 * r**2 * s
                + 0.5 * s**2 * t
                - 0.25 * s**2
                - 0.5 * s * t
                + 0.25 * s,
            ],
            [
                -0.5 * r * s**2 * t**2
                + 0.5 * r * s**2 * t
                + 0.5 * r * t**2
                - 0.5 * r * t
                - 0.25 * s**2 * t**2
                + 0.25 * s**2 * t
                + 0.25 * t**2
                - 0.25 * t,
                -0.5 * r**2 * s * t**2
                + 0.5 * r**2 * s * t
                - 0.5 * r * s * t**2
                + 0.5 * r * s * t,
                -0.5 * r**2 * s**2 * t
                + 0.25 * r**2 * s**2
                + 0.5 * r**2 * t
                - 0.25 * r**2
                - 0.5 * r * s**2 * t
                + 0.25 * r * s**2
                + 0.5 * r * t
                - 0.25 * r,
            ],
            [
                -0.5 * r * s**2 * t**2
                + 0.5 * r * s**2 * t
                - 0.5 * r * s * t**2
                + 0.5 * r * s * t,
                -0.5 * r**2 * s * t**2
                + 0.5 * r**2 * s * t
                - 0.25 * r**2 * t**2
                + 0.25 * r**2 * t
                + 0.5 * s * t**2
                - 0.5 * s * t
                + 0.25 * t**2
                - 0.25 * t,
                -0.5 * r**2 * s**2 * t
                + 0.25 * r**2 * s**2
                - 0.5 * r**2 * s * t
                + 0.25 * r**2 * s
                + 0.5 * s**2 * t
                - 0.25 * s**2
                + 0.5 * s * t
                - 0.25 * s,
            ],
            [
                -0.5 * r * s**2 * t**2
                + 0.5 * r * s**2 * t
                + 0.5 * r * t**2
                - 0.5 * r * t
                + 0.25 * s**2 * t**2
                - 0.25 * s**2 * t
                - 0.25 * t**2
                + 0.25 * t,
                -0.5 * r**2 * s * t**2
                + 0.5 * r**2 * s * t
                + 0.5 * r * s * t**2
                - 0.5 * r * s * t,
                -0.5 * r**2 * s**2 * t
                + 0.25 * r**2 * s**2
                + 0.5 * r**2 * t
                - 0.25 * r**2
                + 0.5 * r * s**2 * t
                - 0.25 * r * s**2
                - 0.5 * r * t
                + 0.25 * r,
            ],
            [
                -0.5 * r * s**2 * t**2
                - 0.5 * r * s**2 * t
                + 0.5 * r * s * t**2
                + 0.5 * r * s * t,
                -0.5 * r**2 * s * t**2
                - 0.5 * r**2 * s * t
                + 0.25 * r**2 * t**2
                + 0.25 * r**2 * t
                + 0.5 * s * t**2
                + 0.5 * s * t
                - 0.25 * t**2
                - 0.25 * t,
                -0.5 * r**2 * s**2 * t
                - 0.25 * r**2 * s**2
                + 0.5 * r**2 * s * t
                + 0.25 * r**2 * s
                + 0.5 * s**2 * t
                + 0.25 * s**2
                - 0.5 * s * t
                - 0.25 * s,
            ],
            [
                -0.5 * r * s**2 * t**2
                - 0.5 * r * s**2 * t
                + 0.5 * r * t**2
                + 0.5 * r * t
                - 0.25 * s**2 * t**2
                - 0.25 * s**2 * t
                + 0.25 * t**2
                + 0.25 * t,
                -0.5 * r**2 * s * t**2
                - 0.5 * r**2 * s * t
                - 0.5 * r * s * t**2
                - 0.5 * r * s * t,
                -0.5 * r**2 * s**2 * t
                - 0.25 * r**2 * s**2
                + 0.5 * r**2 * t
                + 0.25 * r**2
                - 0.5 * r * s**2 * t
                - 0.25 * r * s**2
                + 0.5 * r * t
                + 0.25 * r,
            ],
            [
                -0.5 * r * s**2 * t**2
                - 0.5 * r * s**2 * t
                - 0.5 * r * s * t**2
                - 0.5 * r * s * t,
                -0.5 * r**2 * s * t**2
                - 0.5 * r**2 * s * t
                - 0.25 * r**2 * t**2
                - 0.25 * r**2 * t
                + 0.5 * s * t**2
                + 0.5 * s * t
                + 0.25 * t**2
                + 0.25 * t,
                -0.5 * r**2 * s**2 * t
                - 0.25 * r**2 * s**2
                - 0.5 * r**2 * s * t
                - 0.25 * r**2 * s
                + 0.5 * s**2 * t
                + 0.25 * s**2
                + 0.5 * s * t
                + 0.25 * s,
            ],
            [
                -0.5 * r * s**2 * t**2
                - 0.5 * r * s**2 * t
                + 0.5 * r * t**2
                + 0.5 * r * t
                + 0.25 * s**2 * t**2
                + 0.25 * s**2 * t
                - 0.25 * t**2
                - 0.25 * t,
                -0.5 * r**2 * s * t**2
                - 0.5 * r**2 * s * t
                + 0.5 * r * s * t**2
                + 0.5 * r * s * t,
                -0.5 * r**2 * s**2 * t
                - 0.25 * r**2 * s**2
                + 0.5 * r**2 * t
                + 0.25 * r**2
                + 0.5 * r * s**2 * t
                + 0.25 * r * s**2
                - 0.5 * r * t
                - 0.25 * r,
            ],
            [
                -0.5 * r * s**2 * t**2
                + 0.5 * r * s**2
                + 0.5 * r * s * t**2
                - 0.5 * r * s
                + 0.25 * s**2 * t**2
                - 0.25 * s**2
                - 0.25 * s * t**2
                + 0.25 * s,
                -0.5 * r**2 * s * t**2
                + 0.5 * r**2 * s
                + 0.25 * r**2 * t**2
                - 0.25 * r**2
                + 0.5 * r * s * t**2
                - 0.5 * r * s
                - 0.25 * r * t**2
                + 0.25 * r,
                -0.5 * r**2 * s**2 * t
                + 0.5 * r**2 * s * t
                + 0.5 * r * s**2 * t
                - 0.5 * r * s * t,
            ],
            [
                -0.5 * r * s**2 * t**2
                + 0.5 * r * s**2
                + 0.5 * r * s * t**2
                - 0.5 * r * s
                - 0.25 * s**2 * t**2
                + 0.25 * s**2
                + 0.25 * s * t**2
                - 0.25 * s,
                -0.5 * r**2 * s * t**2
                + 0.5 * r**2 * s
                + 0.25 * r**2 * t**2
                - 0.25 * r**2
                - 0.5 * r * s * t**2
                + 0.5 * r * s
                + 0.25 * r * t**2
                - 0.25 * r,
                -0.5 * r**2 * s**2 * t
                + 0.5 * r**2 * s * t
                - 0.5 * r * s**2 * t
                + 0.5 * r * s * t,
            ],
            [
                -0.5 * r * s**2 * t**2
                + 0.5 * r * s**2
                - 0.5 * r * s * t**2
                + 0.5 * r * s
                - 0.25 * s**2 * t**2
                + 0.25 * s**2
                - 0.25 * s * t**2
                + 0.25 * s,
                -0.5 * r**2 * s * t**2
                + 0.5 * r**2 * s
                - 0.25 * r**2 * t**2
                + 0.25 * r**2
                - 0.5 * r * s * t**2
                + 0.5 * r * s
                - 0.25 * r * t**2
                + 0.25 * r,
                -0.5 * r**2 * s**2 * t
                - 0.5 * r**2 * s * t
                - 0.5 * r * s**2 * t
                - 0.5 * r * s * t,
            ],
            [
                -0.5 * r * s**2 * t**2
                + 0.5 * r * s**2
                - 0.5 * r * s * t**2
                + 0.5 * r * s
                + 0.25 * s**2 * t**2
                - 0.25 * s**2
                + 0.25 * s * t**2
                - 0.25 * s,
                -0.5 * r**2 * s * t**2
                + 0.5 * r**2 * s
                - 0.25 * r**2 * t**2
                + 0.25 * r**2
                + 0.5 * r * s * t**2
                - 0.5 * r * s
                + 0.25 * r * t**2
                - 0.25 * r,
                -0.5 * r**2 * s**2 * t
                - 0.5 * r**2 * s * t
                + 0.5 * r * s**2 * t
                + 0.5 * r * s * t,
            ],
            [
                1.0 * r * s**2 * t**2
                - 1.0 * r * s**2
                - 1.0 * r * t**2
                + 1.0 * r
                - 0.5 * s**2 * t**2
                + 0.5 * s**2
                + 0.5 * t**2
                - 0.5,
                1.0 * r**2 * s * t**2
                - 1.0 * r**2 * s
                - 1.0 * r * s * t**2
                + 1.0 * r * s,
                1.0 * r**2 * s**2 * t
                - 1.0 * r**2 * t
                - 1.0 * r * s**2 * t
                + 1.0 * r * t,
            ],
            [
                1.0 * r * s**2 * t**2
                - 1.0 * r * s**2
                - 1.0 * r * t**2
                + 1.0 * r
                + 0.5 * s**2 * t**2
                - 0.5 * s**2
                - 0.5 * t**2
                + 0.5,
                1.0 * r**2 * s * t**2
                - 1.0 * r**2 * s
                + 1.0 * r * s * t**2
                - 1.0 * r * s,
                1.0 * r**2 * s**2 * t
                - 1.0 * r**2 * t
                + 1.0 * r * s**2 * t
                - 1.0 * r * t,
            ],
            [
                1.0 * r * s**2 * t**2
                - 1.0 * r * s**2
                - 1.0 * r * s * t**2
                + 1.0 * r * s,
                1.0 * r**2 * s * t**2
                - 1.0 * r**2 * s
                - 0.5 * r**2 * t**2
                + 0.5 * r**2
                - 1.0 * s * t**2
                + 1.0 * s
                + 0.5 * t**2
                - 0.5,
                1.0 * r**2 * s**2 * t
                - 1.0 * r**2 * s * t
                - 1.0 * s**2 * t
                + 1.0 * s * t,
            ],
            [
                1.0 * r * s**2 * t**2
                - 1.0 * r * s**2
                + 1.0 * r * s * t**2
                - 1.0 * r * s,
                1.0 * r**2 * s * t**2
                - 1.0 * r**2 * s
                + 0.5 * r**2 * t**2
                - 0.5 * r**2
                - 1.0 * s * t**2
                + 1.0 * s
                - 0.5 * t**2
                + 0.5,
                1.0 * r**2 * s**2 * t
                + 1.0 * r**2 * s * t
                - 1.0 * s**2 * t
                - 1.0 * s * t,
            ],
            [
                1.0 * r * s**2 * t**2
                - 1.0 * r * s**2 * t
                - 1.0 * r * t**2
                + 1.0 * r * t,
                1.0 * r**2 * s * t**2
                - 1.0 * r**2 * s * t
                - 1.0 * s * t**2
                + 1.0 * s * t,
                1.0 * r**2 * s**2 * t
                - 0.5 * r**2 * s**2
                - 1.0 * r**2 * t
                + 0.5 * r**2
                - 1.0 * s**2 * t
                + 0.5 * s**2
                + 1.0 * t
                - 0.5,
            ],
            [
                1.0 * r * s**2 * t**2
                + 1.0 * r * s**2 * t
                - 1.0 * r * t**2
                - 1.0 * r * t,
                1.0 * r**2 * s * t**2
                + 1.0 * r**2 * s * t
                - 1.0 * s * t**2
                - 1.0 * s * t,
                1.0 * r**2 * s**2 * t
                + 0.5 * r**2 * s**2
                - 1.0 * r**2 * t
                - 0.5 * r**2
                - 1.0 * s**2 * t
                - 0.5 * s**2
                + 1.0 * t
                + 0.5,
            ],
            [
                -2.0 * r * s**2 * t**2 + 2.0 * r * s**2 + 2.0 * r * t**2 - 2.0 * r,
                -2.0 * r**2 * s * t**2 + 2.0 * r**2 * s + 2.0 * s * t**2 - 2.0 * s,
                -2.0 * r**2 * s**2 * t + 2.0 * r**2 * t + 2.0 * s**2 * t - 2.0 * t,
            ],
        ]
    )
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_H27_multi(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 27, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_H27(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def volumes_H27(ecoords: np.ndarray, qpos: np.ndarray, qweight: np.ndarray):
    nE = ecoords.shape[0]
    volumes = np.zeros(nE, dtype=ecoords.dtype)
    nQ = len(qweight)
    for iQ in range(nQ):
        dshp = dshp_H27(qpos[iQ])
        for i in prange(nE):
            jac = ecoords[i].T @ dshp
            djac = np.linalg.det(jac)
            volumes[i] += qweight[iQ] * djac
    return volumes
