"""
Algorithms to evaluate cell-in-cell calculations. 
"""

import numpy as np
from numpy import ndarray
from numba import njit, prange

from .tet import pip_tet

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def H8_in_TET4_bulk_knn(
    ecoords_H8: ndarray,
    ecoords_TET4: ndarray,
    neighbours: ndarray,
    tol: float = 1e-12,
) -> ndarray:
    """
    Tells if H8 cells are inside TET4 cells. The function returns
    a boolean NumPy array with the same length as the number of H8 cells.
    If the i-th value is `True`, it means that the i-th H8 cell is inside
    at least one TET4 cell.

    The function goes through all nodes of all H8 cells and check if it
    is inside any of the TET4 cells that make up the neighbourhood of the
    cell the point belongs to.

    Parameters
    ----------
    ecoords_H8: numpy.ndarray
        3d NumPy array of the coordinates of the nodes of the H8 cells.
    ecoords_TET4: numpy.ndarray
        3d NumPy array of the coordinates of the nodes of the TET4 cells.
    neighbours: numpy.ndarray
        2d NumPy array of the indices of the neighbours of each H8 cell. These
        indices refer to the TET4 cells in the `ecoords_TET4` array.
    tol: float, Optional
        Tolerance to consider a point inside a cell. Default is 1e-12.
    """
    nN = neighbours.shape[1]
    res = np.zeros(ecoords_H8.shape[0], dtype=np.bool_)
    for i in prange(ecoords_H8.shape[0]):
        pip = False
        for j in range(8):
            p = ecoords_H8[i, j]
            for k in range(nN):
                pip = pip_tet(p, ecoords_TET4[neighbours[i, k]], tol)
                if pip:
                    res[i] = True
                    break
            if pip:
                break
    return res
