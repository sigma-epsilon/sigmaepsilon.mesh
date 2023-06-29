from numba import njit, prange
import numpy as np
from numpy import ndarray
from neumann import flatten2dC

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_Q9(x: ndarray) -> ndarray:
    r, s = x
    return np.array(
        [
            1,
            r,
            s,
            r * s,
            r**2,
            s**2,
            r * s**2,
            s * r**2,
            s**2 * r**2,
        ],
        dtype=float,
    )


@njit(nogil=True, cache=__cache)
def shp_Q9(pcoord: np.ndarray):
    r, s = pcoord[:2]
    return np.array(
        [
            [
                0.25 * r**2 * s**2
                - 0.25 * r**2 * s
                - 0.25 * r * s**2
                + 0.25 * r * s
            ],
            [
                0.25 * r**2 * s**2
                - 0.25 * r**2 * s
                + 0.25 * r * s**2
                - 0.25 * r * s
            ],
            [
                0.25 * r**2 * s**2
                + 0.25 * r**2 * s
                + 0.25 * r * s**2
                + 0.25 * r * s
            ],
            [
                0.25 * r**2 * s**2
                + 0.25 * r**2 * s
                - 0.25 * r * s**2
                - 0.25 * r * s
            ],
            [-0.5 * r**2 * s**2 + 0.5 * r**2 * s + 0.5 * s**2 - 0.5 * s],
            [-0.5 * r**2 * s**2 + 0.5 * r**2 - 0.5 * r * s**2 + 0.5 * r],
            [-0.5 * r**2 * s**2 - 0.5 * r**2 * s + 0.5 * s**2 + 0.5 * s],
            [-0.5 * r**2 * s**2 + 0.5 * r**2 + 0.5 * r * s**2 - 0.5 * r],
            [1.0 * r**2 * s**2 - 1.0 * r**2 - 1.0 * s**2 + 1.0],
        ],
        dtype=pcoord.dtype,
    )


@njit(nogil=True, parallel=True, cache=__cache)
def shp_Q9_multi(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 9), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = flatten2dC(shp_Q9(pcoords[iP]))
    return res


@njit(nogil=True, parallel=False, cache=__cache)
def shape_function_matrix_Q9(pcoord: np.ndarray, ndof: int = 2):
    eye = np.eye(ndof, dtype=pcoord.dtype)
    shp = shp_Q9(pcoord)
    res = np.zeros((ndof, ndof * 9), dtype=pcoord.dtype)
    for i in range(9):
        res[:, i * ndof : (i + 1) * ndof] = eye * shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_Q9_multi(pcoords: np.ndarray, ndof: int = 2):
    nP = pcoords.shape[0]
    res = np.zeros((nP, ndof, ndof * 9), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_Q9(pcoords[iP], ndof)
    return res


@njit(nogil=True, cache=__cache)
def dshp_Q9(pcoord: np.ndarray):
    r, s = pcoord[:2]
    return np.array(
        [
            [
                0.5 * r * s**2 - 0.5 * r * s - 0.25 * s**2 + 0.25 * s,
                0.5 * r**2 * s - 0.25 * r**2 - 0.5 * r * s + 0.25 * r,
            ],
            [
                0.5 * r * s**2 - 0.5 * r * s + 0.25 * s**2 - 0.25 * s,
                0.5 * r**2 * s - 0.25 * r**2 + 0.5 * r * s - 0.25 * r,
            ],
            [
                0.5 * r * s**2 + 0.5 * r * s + 0.25 * s**2 + 0.25 * s,
                0.5 * r**2 * s + 0.25 * r**2 + 0.5 * r * s + 0.25 * r,
            ],
            [
                0.5 * r * s**2 + 0.5 * r * s - 0.25 * s**2 - 0.25 * s,
                0.5 * r**2 * s + 0.25 * r**2 - 0.5 * r * s - 0.25 * r,
            ],
            [
                -1.0 * r * s**2 + 1.0 * r * s,
                -1.0 * r**2 * s + 0.5 * r**2 + 1.0 * s - 0.5,
            ],
            [
                -1.0 * r * s**2 + 1.0 * r - 0.5 * s**2 + 0.5,
                -1.0 * r**2 * s - 1.0 * r * s,
            ],
            [
                -1.0 * r * s**2 - 1.0 * r * s,
                -1.0 * r**2 * s - 0.5 * r**2 + 1.0 * s + 0.5,
            ],
            [
                -1.0 * r * s**2 + 1.0 * r + 0.5 * s**2 - 0.5,
                -1.0 * r**2 * s + 1.0 * r * s,
            ],
            [2.0 * r * s**2 - 2.0 * r, 2.0 * r**2 * s - 2.0 * s],
        ],
        dtype=pcoord.dtype,
    )


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_Q9_multi(pcoords: ndarray):
    """
    Returns the first orderderivatives of the shape functions,
    evaluated at multiple points, according to 'pcoords'.

    ---
    (nP, nNE, 2)
    """
    nP = pcoords.shape[0]
    res = np.zeros((nP, 9, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_Q9(pcoords[iP])
    return res


if __name__ == "__main__":
    import numpy as np

    shape_function_matrix_Q9(np.array([0, 0]))
    shape_function_matrix_Q9(np.array([0, 0]))
    shape_function_matrix_Q9(np.array([0, 0]))
