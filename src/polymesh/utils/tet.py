import numpy as np
from numpy import ndarray
from numba import njit, prange

__cache = True


@njit(nogil=True, cache=__cache)
def vol_tet(ecoords: ndarray) -> ndarray:
    """
    Calculates volumes of several tetrahedra.

    Parameters
    ----------
    ecoords: numpy.ndarray
        A 3d float array of shape (nE, nNE, 3) of
        nodal coordinates for several elements. Here nE
        is the number of nodes and nNE is the number of
        nodes per element.

    Returns
    -------
    numpy.ndarray
        1d float array of volumes.

    Note
    ----
    This only returns exact results for linear cells. For
    nonlinear cells, use objects that calculate the volumes
    using numerical integration.
    """
    v1 = ecoords[1] - ecoords[0]
    v2 = ecoords[2] - ecoords[0]
    v3 = ecoords[3] - ecoords[0]
    return np.dot(np.cross(v1, v2), v3) / 6


@njit(nogil=True, parallel=True, cache=__cache)
def vol_tet_bulk(ecoords: ndarray) -> ndarray:
    """
    Calculates volumes of several tetrahedra.

    Parameters
    ----------
    ecoords: numpy.ndarray
        A 3d float array of shape (nE, nNE, 3) of
        nodal coordinates for several elements. Here nE
        is the number of nodes and nNE is the number of
        nodes per element.

    Returns
    -------
    numpy.ndarray
        1d float array of volumes.

    Note
    ----
    This only returns exact results for linear cells. For
    nonlinear cells, use objects that calculate the volumes
    using numerical integration.
    """
    nE = len(ecoords)
    res = np.zeros(nE, dtype=ecoords.dtype)
    for i in prange(nE):
        res[i] = vol_tet(ecoords[i])
    return np.abs(res)


@njit(nogil=True, cache=__cache)
def glob_to_nat_tet(gcoord: ndarray, ecoords: ndarray) -> ndarray:
    """
    Transformation from global to natural coordinates within a tetrahedron.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    ecoords_ = np.zeros_like(ecoords)
    ecoords_[:, :] = ecoords[:, :]
    V = vol_tet(ecoords)

    ecoords_[0, :] = gcoord
    v1 = vol_tet(ecoords_) / V
    ecoords_[0, :] = ecoords[0, :]

    ecoords_[1, :] = gcoord
    v2 = vol_tet(ecoords_) / V
    ecoords_[1, :] = ecoords[1, :]

    ecoords_[2, :] = gcoord
    v3 = vol_tet(ecoords_) / V
    ecoords_[2, :] = ecoords[2, :]

    return np.array([v1, v2, v3, 1 - v1 - v2 - v3], dtype=gcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def _glob_to_nat_tet_bulk_(points: ndarray, ecoords: ndarray) -> ndarray:
    nE = ecoords.shape[0]
    nP = points.shape[0]
    res = np.zeros((nP, nE, 4), dtype=points.dtype)
    for i in prange(nP):
        for j in prange(nE):
            res[i, j, :] = glob_to_nat_tet(points[i], ecoords[j])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def __pip_tet_bulk__(nat: ndarray, tol: float = 1e-12) -> ndarray:
    nP, nE = nat.shape[:2]
    res = np.zeros((nP, nE), dtype=np.bool_)
    for i in prange(nP):
        for j in prange(nE):
            c1 = np.all(nat[i, j] > (-tol))
            c2 = np.all(nat[i, j] < (1 + tol))
            res[i, j] = c1 & c2
    return res


@njit(nogil=True, cache=__cache)
def _pip_tet_bulk_(points: ndarray, ecoords: ndarray, tol: float = 1e-12) -> ndarray:
    nat = _glob_to_nat_tet_bulk_(points, ecoords)
    return __pip_tet_bulk__(nat, tol)


@njit(nogil=True, cache=__cache)
def _pip_tet_bulk_knn_(
    points: ndarray, ecoords: ndarray, neighbours: ndarray, tol: float = 1e-12
) -> ndarray:
    nat = _glob_to_nat_tet_bulk_knn_(points, ecoords, neighbours)
    return __pip_tet_bulk__(nat, tol)


@njit(nogil=True, parallel=True, cache=__cache)
def _glob_to_nat_tet_bulk_knn_(
    points: ndarray, ecoords: ndarray, neighbours: ndarray
) -> ndarray:
    kE = neighbours.shape[1]
    nP = points.shape[0]
    res = np.zeros((nP, kE, 4), dtype=points.dtype)
    for i in prange(nP):
        for k in prange(kE):
            res[i, k, :] = glob_to_nat_tet(points[i], ecoords[neighbours[i, k]])
    return res


@njit(nogil=True, cache=__cache)
def lcoords_tet() -> ndarray:
    """
    Returns coordinates of the master element
    of a simplex in 3d.
    """
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )


@njit(nogil=True, cache=__cache)
def nat_to_loc_tet(acoord: np.ndarray) -> ndarray:
    """
    Transformation from natural to local coordinates
    within a tetrahedra.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    return acoord.T @ lcoords_tet()
