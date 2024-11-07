import numpy as np
from numpy import ndarray
from numba import njit, prange

__cache = True


@njit(nogil=True, cache=__cache)
def vol_tet(ecoords: ndarray) -> ndarray:
    """
    Calculates volume of a single tetrahedron.

    Parameters
    ----------
    ecoords: numpy.ndarray
        A 3d float array of shape (nNE=4, 3) of nodal coordinates.

    Returns
    -------
    float
        The volume of the tetrahedron.

    Notes
    -----
    1) This only returns exact results for linear cells. For
    nonlinear cells, use objects that calculate the volumes
    using numerical integration.
    2) This function is numba-jittable in 'nopython' mode.
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
        A 3d float array of shape (nE, nNE=4, 3) of
        nodal coordinates for several elements. Here nE
        is the number of nodes and nNE is the number of
        nodes per element.

    Returns
    -------
    numpy.ndarray
        1d float array of volumes.

    Notes
    -----
    1) This only returns exact results for linear cells. For
    nonlinear cells, use objects that calculate the volumes
    using numerical integration.
    2) This function is numba-jittable in 'nopython' mode.
    """
    nE = len(ecoords)
    res = np.zeros(nE, dtype=ecoords.dtype)
    for i in prange(nE):
        res[i] = vol_tet(ecoords[i])
    return np.abs(res)


@njit(nogil=True, cache=__cache)
def glob_to_nat_tet(gcoord: ndarray, ecoords: ndarray) -> ndarray:
    """
    Transformation from global to natural coordinates within a tetrahedron
    for a single point and tetrahedra.

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


@njit(nogil=True, cache=__cache)
def pip_tet(point: ndarray, ecoords: ndarray, tol: float = 1e-12) -> ndarray:
    """
    Tells if a point is inside a tetrahedron or not.

    Parameters
    ----------
    point: numpy.ndarray
        1d NumPy array of the global coordinates of the point.
    ecoords: numpy.ndarray
        2d NumPy array of the coordinates of the nodes of the tetrahedron.
    tol: float, Optional
        Tolerance to consider a point inside a cell. Default is 1e-12.

    Returns
    -------
    bool
        True if the point is inside the tetrahedron, False otherwise.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    nat = glob_to_nat_tet(point, ecoords)
    return np.all(nat > -tol) and np.all(nat < (1 + tol))


@njit(nogil=True, parallel=True, cache=__cache)
def _pip_tet_bulk_nat_(nat: ndarray, tol: float = 1e-12) -> ndarray:
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
    return _pip_tet_bulk_nat_(nat, tol)


@njit(nogil=True, cache=__cache)
def _pip_tet_bulk_knn_(
    points: ndarray, ecoords: ndarray, neighbours: ndarray, tol: float = 1e-12
) -> ndarray:
    nat = _glob_to_nat_tet_bulk_knn_(points, ecoords, neighbours)
    return _pip_tet_bulk_nat_(nat, tol)


def pip_tet_bulk(
    points: ndarray,
    ecoords: ndarray,
    tol: float = 1e-12,
    neighbours: ndarray | None = None,
) -> ndarray:
    """
    Tells if points are inside tetrahedra or not.

    Parameters
    ----------
    points: numpy.ndarray
        2d NumPy array of the global coordinates of the points.
    ecoords: numpy.ndarray
        3d NumPy array of the coordinates of the nodes of the tetrahedra.
    tol: float, Optional
        Tolerance to consider a point inside a cell. Default is 1e-12.
    neighbours: numpy.ndarray, Optional
        2d NumPy array of the indices of the neighbouring cells of each point.
        These indices refer to the tetrahedra in the `ecoords` array. Providing
        this is optional, but strogly recommended for large meshes.
    """
    if neighbours is None:
        return _pip_tet_bulk_(points, ecoords, tol)
    else:
        return _pip_tet_bulk_knn_(points, ecoords, neighbours, tol)


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
def lcoords_tet(center: ndarray = None) -> ndarray:
    """
    Returns coordinates of the master element
    of a simplex in 3d.

    Parameters
    ----------
    center: numpy.ndarray, Optional
        The coordinates of the center of the master
        element. Default is None.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    res = np.array(
        [
            [-1 / 3, -1 / 3, -1 / 3],
            [2 / 3, -1 / 3, -1 / 3],
            [-1 / 3, 2 / 3, -1 / 3],
            [-1 / 3, -1 / 3, 2 / 3],
        ]
    )
    if center is not None:
        res += center
    return res


@njit(nogil=True, cache=__cache)
def nat_to_loc_tet(
    acoord: ndarray, lcoords: ndarray | None = None, center: ndarray | None = None
) -> ndarray:
    """
    Transformation from natural to local coordinates
    within a tetrahedra.

    Parameters
    ----------
    acoord: numpy.ndarray
        1d NumPy array of natural coordinates of a point.
    lcoords: numpy.ndarray, Optional
        2d NumPy array of parametric coordinates (r, s, t) of the
        master cell of a tetrahedron.
    center: numpy.ndarray
        The local coordinates (r, s, t) of the geometric center
        of the master tetrahedron. If not provided it is assumed to
        be at (0, 0, 0).

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    if lcoords is None:
        lcoords = lcoords_tet(center)
    return acoord.T @ lcoords


@njit(nogil=True, cache=__cache)
def TET4_face_normals(
    coords: ndarray[float], topo: ndarray[int], normalize: bool = False
) -> ndarray[float]:
    """
    Returns face normals of a TET4 element.

    Parameters
    ----------
    coords: numpy.ndarray
        A 2d float array of shape (:, 3) of nodal coordinates.
    topo: numpy.ndarray
        A 1d int array of shape (4,) of the indices of the nodes.
    normalize: bool, Optional
        If True, the normals are normalized. Default is False.

    """
    normals = np.zeros((4, 3), dtype=np.float64)
    # face 1
    normals[0, :] = np.cross(
        coords[topo[1]] - coords[topo[0]], coords[topo[3]] - coords[topo[0]]
    )
    # face 2
    normals[1, :] = np.cross(
        coords[topo[2]] - coords[topo[0]], coords[topo[3]] - coords[topo[0]]
    )
    # face 3
    normals[2, :] = np.cross(
        coords[topo[2]] - coords[topo[1]], coords[topo[3]] - coords[topo[1]]
    )
    # face 4
    normals[3, :] = np.cross(
        coords[topo[2]] - coords[topo[0]], coords[topo[1]] - coords[topo[0]]
    )

    if normalize:
        for i in range(4):
            normals[i] /= np.linalg.norm(normals[i])

    return normals


@njit(nogil=True, cache=__cache)
def TET4_edge_vectors(
    coords: ndarray[float], topo: ndarray[int], normalize: bool = False
) -> ndarray[float]:
    """
    Returns edge vectors of a TET4 element.

    Parameters
    ----------
    coords: numpy.ndarray
        A 2d float array of shape (:, 3) of nodal coordinates.
    topo: numpy.ndarray
        A 1d int array of shape (4,) of the indices of the nodes.
    normalize: bool, Optional
        If True, the normals are normalized. Default is False.

    """
    edges = np.zeros((6, 3), dtype=np.float64)

    edges[0, :] = coords[topo[1]] - coords[topo[0]]
    edges[1, :] = coords[topo[3]] - coords[topo[0]]
    edges[2, :] = coords[topo[2]] - coords[topo[0]]
    edges[3, :] = coords[topo[2]] - coords[topo[1]]
    edges[4, :] = coords[topo[2]] - coords[topo[3]]
    edges[5, :] = coords[topo[3]] - coords[topo[1]]

    if normalize:
        for i in range(6):
            edges[i] /= np.linalg.norm(edges[i])

    return edges
