import numpy as np
from numpy import ndarray
from numba import njit, prange

from sigmaepsilon.math.knn import k_nearest_neighbours
from sigmaepsilon.math.utils import atleast2d

from ..tri import center_tri_bulk_3d
from ..tet import TET4_face_normals, TET4_edge_vectors
from .tr import H8_to_TET4

__all__ = [
    "T3_in_T3",
    "TET4_in_TET4",
    "H8_in_TET4",
    "TET4_in_H8",
    "TET4_in_T3",
    "H8_in_T3",
]

__cache = True


@njit(nogil=True, cache=__cache)
def _project_to_line(
    coords: ndarray[float], topo: ndarray[int], axis: ndarray[float]
) -> tuple[float, float]:
    """Projects the triangle to the axis and returns the start and end points."""
    start = np.dot(coords[topo[0]], axis)
    end = start
    n_nodes = len(topo)
    for i in range(1, n_nodes):
        X = np.dot(coords[topo[i]], axis)
        if X < start:
            start = X
        elif X > end:
            end = X
    return start, end


@njit(nogil=True, cache=__cache)
def _sat_single(
    coordsA: ndarray[float],
    topoA: ndarray[int],
    coordsB: ndarray[float],
    topoB: ndarray[int],
    axis: ndarray[float],
) -> bool:
    """Returns True if the two triangles are separated by the axis."""
    startA, endA = _project_to_line(coordsA, topoA, axis)
    startB, endB = _project_to_line(coordsB, topoB, axis)
    return endA < startB or endB < startA


@njit(nogil=True, cache=__cache)
def _T3_in_T3_single(
    coordsA: ndarray[float],
    topoA: ndarray[int],
    coordsB: ndarray[float],
    topoB: ndarray[int],
) -> ndarray[bool]:
    """Returns True if triangle A intersects triangle B."""
    axis = np.zeros(3, dtype=np.float64)
    edgesA = np.zeros((3, 3), dtype=np.float64)
    edgesB = np.zeros((3, 3), dtype=np.float64)

    # normal plane of triangle A
    edgesA[0, :] = coordsA[topoA[1]] - coordsA[topoA[0]]
    edgesA[1, :] = coordsA[topoA[2]] - coordsA[topoA[0]]
    normal_A = np.cross(edgesA[0, :], edgesA[1, :])
    axis[:] = normal_A
    if _sat_single(coordsA, topoA, coordsB, topoB, axis):
        return False

    # normal plane of triangle B
    edgesB[0, :] = coordsB[topoB[1]] - coordsB[topoB[0]]
    edgesB[1, :] = coordsB[topoB[2]] - coordsB[topoB[0]]
    normal_B = np.cross(edgesB[0, :], edgesB[1, :])
    axis[:] = normal_B
    if _sat_single(coordsA, topoA, coordsB, topoB, axis):
        return False

    # edge normal 1 of triangle A
    axis[:] = np.cross(normal_A, edgesA[0, :])
    if _sat_single(coordsA, topoA, coordsB, topoB, axis):
        return False

    # edge normal 2 of triangle A
    axis[:] = np.cross(normal_A, edgesA[1, :])
    if _sat_single(coordsA, topoA, coordsB, topoB, axis):
        return False

    # edge normal 3 of triangle A
    edgesA[2, :] = coordsA[topoA[2]] - coordsA[topoA[1]]
    axis[:] = np.cross(normal_A, edgesA[2, :])
    if _sat_single(coordsA, topoA, coordsB, topoB, axis):
        return False

    # edge normal 1 of triangle B
    axis[:] = np.cross(normal_B, edgesB[0, :])
    if _sat_single(coordsA, topoA, coordsB, topoB, axis):
        return False

    # edge normal 2 of triangle B
    axis[:] = np.cross(normal_B, edgesB[1, :])
    if _sat_single(coordsA, topoA, coordsB, topoB, axis):
        return False

    # edge normal 3 of triangle B
    edgesB[2, :] = coordsB[topoB[2]] - coordsB[topoB[1]]
    axis[:] = np.cross(normal_B, edgesB[2, :])
    if _sat_single(coordsA, topoA, coordsB, topoB, axis):
        return False

    normal_A /= np.linalg.norm(normal_A)
    normal_B /= np.linalg.norm(normal_B)
    parallels = np.dot(normal_A, normal_B) > 0.999999
    if parallels:
        return True

    # edge cross edge
    for i in range(3):
        for j in range(3):
            axis[:] = np.cross(edgesA[i, :], edgesB[j, :])
            if _sat_single(coordsA, topoA, coordsB, topoB, axis):
                return False

    return True


@njit(nogil=True, cache=__cache)
def _TET4_in_TET4_single(
    coordsA: ndarray[float],
    topoA: ndarray[int],
    coordsB: ndarray[float],
    topoB: ndarray[int],
) -> ndarray[bool]:
    axes = np.zeros((4, 3), dtype=np.float64)

    # check for face normals of A
    axes[:, :] = TET4_face_normals(coordsA, topoA)
    for i in range(4):
        if _sat_single(coordsA, topoA, coordsB, topoB, axes[i]):
            return False

    # check for face normals of B
    axes[:, :] = TET4_face_normals(coordsB, topoB)
    for i in range(4):
        if _sat_single(coordsA, topoA, coordsB, topoB, axes[i]):
            return False

    edgesA = TET4_edge_vectors(coordsA, topoA)
    edgesB = TET4_edge_vectors(coordsB, topoB)
    for i in range(6):
        for j in range(6):
            axis = np.cross(edgesA[i], edgesB[j])
            if _sat_single(coordsA, topoA, coordsB, topoB, axis):
                return False

    return True


@njit(nogil=True, cache=__cache)
def _TET4_in_T3_single(
    coords_TET4: ndarray[float],
    topo_TET4: ndarray[int],
    coords_T3: ndarray[float],
    topo_T3: ndarray[int],
) -> ndarray[bool]:
    """Returns True if the TET4 cell intersects the T3 cell."""
    axis = np.zeros(3, dtype=np.float64)
    edges_T3 = np.zeros((3, 3), dtype=np.float64)

    # normal plane of triangle
    edges_T3[0, :] = coords_T3[topo_T3[1]] - coords_T3[topo_T3[0]]
    edges_T3[1, :] = coords_T3[topo_T3[2]] - coords_T3[topo_T3[0]]
    normal_T3 = np.cross(edges_T3[0, :], edges_T3[1, :])
    axis[:] = normal_T3
    if _sat_single(coords_T3, topo_T3, coords_TET4, topo_TET4, axis):
        return False

    # edge normal 1 of triangle A
    axis[:] = np.cross(normal_T3, edges_T3[0, :])
    if _sat_single(coords_T3, topo_T3, coords_TET4, topo_TET4, axis):
        return False

    # edge normal 2 of triangle A
    axis[:] = np.cross(normal_T3, edges_T3[1, :])
    if _sat_single(coords_T3, topo_T3, coords_TET4, topo_TET4, axis):
        return False

    # edge normal 3 of triangle A
    edges_T3[2, :] = coords_T3[topo_T3[2]] - coords_T3[topo_T3[1]]
    axis[:] = np.cross(normal_T3, edges_T3[2, :])
    if _sat_single(coords_T3, topo_T3, coords_TET4, topo_TET4, axis):
        return False

    # check for face normals of TET4
    axes = TET4_face_normals(coords_TET4, topo_TET4)
    for i in range(4):
        if _sat_single(coords_T3, topo_T3, coords_TET4, topo_TET4, axes[i]):
            return False

    # cross products of edges
    edges_TET4 = TET4_edge_vectors(coords_TET4, topo_TET4)
    for i in range(3):
        for j in range(6):
            axis[:] = np.cross(edges_T3[i], edges_TET4[j])
            if _sat_single(coords_T3, topo_T3, coords_TET4, topo_TET4, axis):
                return False

    return True


@njit(nogil=True, parallel=True, cache=__cache)
def _T3_in_T3_bulk(
    coordsA: ndarray[float],
    topoA: ndarray[int],
    coordsB: ndarray[float],
    topoB: ndarray[int],
    neighbours: ndarray[int],
) -> ndarray[bool]:
    """Performs the separating axis theorem for all triangles in A and B."""
    n_cell_A = topoA.shape[0]
    n_neighbours = neighbours.shape[1]
    out = np.zeros((n_cell_A), dtype=np.bool_)
    for iA in prange(n_cell_A):
        for iN in range(n_neighbours):
            iB = neighbours[iA, iN]
            intersecting = _T3_in_T3_single(coordsA, topoA[iA], coordsB, topoB[iB])
            if intersecting:
                out[iA] = True
                break
    return out


@njit(nogil=True, parallel=True, cache=__cache)
def _TET4_in_TET4_bulk(
    coordsA: ndarray[float],
    topoA: ndarray[int],
    coordsB: ndarray[float],
    topoB: ndarray[int],
    neighbours: ndarray[int],
) -> ndarray[bool]:
    """Performs the separating axis theorem for all triangles in A and B."""
    n_cell_A = topoA.shape[0]
    n_neighbours = neighbours.shape[1]
    out = np.zeros((n_cell_A), dtype=np.bool_)
    for iA in prange(n_cell_A):
        for iN in range(n_neighbours):
            iB = neighbours[iA, iN]
            intersecting = _TET4_in_TET4_single(coordsA, topoA[iA], coordsB, topoB[iB])
            if intersecting:
                out[iA] = True
                break
    return out


@njit(nogil=True, parallel=True, cache=__cache)
def _TET4_in_T3_bulk(
    coords_TET4: ndarray[float],
    topo_TET4: ndarray[int],
    coords_T3: ndarray[float],
    topo_T3: ndarray[int],
    neighbours: ndarray[int],
) -> ndarray[bool]:
    """Performs the separating axis theorem for all TET4 cells in T3 cells."""
    n_cell_A = topo_TET4.shape[0]
    n_neighbours = neighbours.shape[1]
    out = np.zeros((n_cell_A), dtype=np.bool_)
    for iA in prange(n_cell_A):
        for iN in range(n_neighbours):
            iB = neighbours[iA, iN]
            intersecting = _TET4_in_T3_single(
                coords_TET4, topo_TET4[iA], coords_T3, topo_T3[iB]
            )
            if intersecting:
                out[iA] = True
                break
    return out


def T3_in_T3(
    coordsA: ndarray[float],
    trianglesA: ndarray[int],
    coordsB: ndarray[float],
    trianglesB: ndarray[int],
    k: int = 10,
) -> ndarray[bool]:
    """
    Calculates whether two sets of triangles are intersecting.

    The function returns a boolean array of length N, where N is the
    number of triangles in A. The i-th element of the array is True if
    the i-th triangle in A intersects any of the triangles in B. To narrow
    down the search, the function uses the k nearest neighbours of the
    centers of the triangles in B.

    .. versionadded:: 3.1.0

    Parameters
    ----------
    coordsA : ndarray
        The coordinates of the nodes of the mesh A.
    trianglesA : ndarray
        The topology of the mesh A.
    coordsB : ndarray
        The coordinates of the nodes of the mesh B.
    trianglesB : ndarray
        The topology of the mesh B.
    k : int, optional
        The number of nearest neighbours to consider. Default is 10.

    Example
    -------
    >>> import numpy as np
    >>> from sigmaepsilon.mesh.utils.topology import T3_in_T3
    >>> coordsA = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> trianglesA = np.array([[0, 1, 2]])
    >>> coordsB = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> trianglesB = np.array([[0, 1, 2]])
    >>> intersecting = T3_in_T3(coordsA, trianglesA, coordsB, trianglesB)
    >>> assert intersecting[0]

    """
    coordsA = np.asarray(coordsA, dtype=float)
    coordsB = np.asarray(coordsB, dtype=float)
    trianglesA = atleast2d(trianglesA, front=True)
    trianglesB = atleast2d(trianglesB, front=True)

    if not coordsA.shape[1] == 3:
        raise ValueError("Coordinates of A must have 3 dimensions.")

    if not coordsB.shape[1] == 3:
        raise ValueError("Coordinates of B must have 3 dimensions.")

    if not trianglesA.shape[1] == 3:
        raise ValueError("Topology of A must have 3 nodes.")

    if not trianglesB.shape[1] == 3:
        raise ValueError("Topology of B must have 3 nodes.")

    centersA = center_tri_bulk_3d(coordsA, trianglesA)
    centersB = center_tri_bulk_3d(coordsB, trianglesB)
    k = min(k, len(centersB))
    neighbours = k_nearest_neighbours(centersB, centersA, k=k)
    neighbours = atleast2d(neighbours, back=True)
    out = _T3_in_T3_bulk(coordsA, trianglesA, coordsB, trianglesB, neighbours)
    return out


def TET4_in_TET4(
    coordsA: ndarray[float],
    topologyA: ndarray[int],
    coordsB: ndarray[float],
    topologyB: ndarray[int],
    k: int = 10,
) -> ndarray[bool]:
    """
    Calculates whether two sets of tetrahedra are intersecting.

    The function returns a boolean array of length N, where N is the
    number of tetrahedra in A. The i-th element of the array is True if
    the i-th tetrahedra in A intersects any of the tetrahedra in B. To narrow
    down the search, the function uses the k nearest neighbours of the
    centers of the tetrahedra in B.

    .. versionadded:: 3.1.0

    Parameters
    ----------
    coordsA : ndarray
        The coordinates of the nodes of the mesh A.
    topologyA : ndarray
        The topology of the mesh A.
    coordsB : ndarray
        The coordinates of the nodes of the mesh B.
    topologyB : ndarray
        The topology of the mesh B.
    k : int, optional
        The number of nearest neighbours to consider. Default is 10.

    Example
    -------
    >>> import numpy as np
    >>> from sigmaepsilon.mesh.utils.topology import TET4_in_TET4
    >>> coordsA = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> tetraA = np.array([[0, 1, 2, 3]])
    >>> coordsB = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> tetraB = np.array([[0, 1, 2, 3]])
    >>> intersecting = TET4_in_TET4(coordsA, tetraA, coordsB, tetraB)
    >>> assert intersecting[0]

    """
    coordsA = np.asarray(coordsA, dtype=float)
    coordsB = np.asarray(coordsB, dtype=float)
    topologyA = atleast2d(topologyA, front=True)
    topologyB = atleast2d(topologyB, front=True)

    if not coordsA.shape[1] == 3:
        raise ValueError("Coordinates of A must have 3 dimensions.")

    if not coordsB.shape[1] == 3:
        raise ValueError("Coordinates of B must have 3 dimensions.")

    if not topologyA.shape[1] == 4:
        raise ValueError("Topology of A must have 4 nodes.")

    if not topologyB.shape[1] == 4:
        raise ValueError("Topology of B must have 4 nodes.")

    centersA = center_tri_bulk_3d(coordsA, topologyA)
    centersB = center_tri_bulk_3d(coordsB, topologyB)
    k = min(k, len(centersB))
    neighbours = k_nearest_neighbours(centersB, centersA, k=k)
    neighbours = atleast2d(neighbours, back=True)
    out = _TET4_in_TET4_bulk(coordsA, topologyA, coordsB, topologyB, neighbours)
    return out


def H8_in_TET4(
    coords_H8: ndarray[float],
    topology_H8: ndarray[int],
    coords_TET4: ndarray[float],
    topology_TET4: ndarray[int],
    k: int = 10,
) -> ndarray[bool]:
    """
    Performs intersection check between H8 and TET4 cells.

    The function returns a boolean array of length N, where N is the
    number of H8 cells. The i-th element of the array is True if
    the i-th H8 cell intersects any of the TET4 cells. To narrow
    down the search, the function uses a k nearest neighbours algorithm.

    .. versionadded:: 3.1.0

    Parameters
    ----------
    coords_H8 : ndarray
        The coordinates of the nodes of the H8 cells.
    topology_H8 : ndarray
        The topology of the H8 cells.
    coords_TET4 : ndarray
        The coordinates of the nodes of the TET4 cells.
    topology_TET4 : ndarray
        The topology of the TET4 cells.
    k : int, optional
        The number of nearest neighbours to consider. Default is 10.

    Example
    -------
    >>> import numpy as np
    >>> from sigmaepsilon.mesh.utils.topology import H8_in_TET4
    >>> coords_H8 = np.array([
    ...     [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    ...     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ... ])
    >>> topology_H8 = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    >>> coords_TET4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> topology_TET4 = np.array([[0, 1, 2, 3]])
    >>> intersecting = H8_in_TET4(coords_H8, topology_H8, coords_TET4, topology_TET4)
    >>> assert intersecting[0]

    """
    coords_H8 = np.asarray(coords_H8, dtype=float)
    coords_TET4 = np.asarray(coords_TET4, dtype=float)
    topology_H8 = atleast2d(topology_H8, front=True)
    topology_TET4 = atleast2d(topology_TET4, front=True)

    if not coords_H8.shape[1] == 3:
        raise ValueError("Coordinates of H8 cells must have 3 dimensions.")

    if not coords_TET4.shape[1] == 3:
        raise ValueError("Coordinates of TET4 cells must have 3 dimensions.")

    if not topology_H8.shape[1] == 8:
        raise ValueError("Topology of H8 cells must have 8 nodes.")

    if not topology_TET4.shape[1] == 4:
        raise ValueError("Topology of TET4 cells must have 4 nodes.")

    _coords_TET4, _topology_TET4 = H8_to_TET4(coords_H8, topology_H8)
    _out = TET4_in_TET4(_coords_TET4, _topology_TET4, coords_TET4, topology_TET4, k=k)
    out = np.any(_out.reshape((len(topology_H8), 5)), axis=1)
    return out


def TET4_in_H8(
    coords_TET4: ndarray[float],
    topology_TET4: ndarray[int],
    coords_H8: ndarray[float],
    topology_H8: ndarray[int],
    k: int = 10,
) -> ndarray[bool]:
    """
    Performs intersection check between TET4 and H8 cells.

    The function returns a boolean array of length N, where N is the
    number of TET4 cells. The i-th element of the array is True if
    the i-th TET4 cell intersects any of the H8 cells. To narrow
    down the search, the function uses a k nearest neighbours algorithm.

    .. versionadded:: 3.1.0

    Parameters
    ----------
    coords_TET4 : ndarray
        The coordinates of the nodes of the TET4 cells.
    topology_TET4 : ndarray
        The topology of the TET4 cells.
    coords_H8 : ndarray
        The coordinates of the nodes of the H8 cells.
    topology_H8 : ndarray
        The topology of the H8 cells.
    k : int, optional
        The number of nearest neighbours to consider. Default is 10.

    Example
    -------
    >>> import numpy as np
    >>> from sigmaepsilon.mesh.utils.topology import TET4_in_H8
    >>> coords_H8 = np.array([
    ...     [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    ...     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ... ])
    >>> topology_H8 = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    >>> coords_TET4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> topology_TET4 = np.array([[0, 1, 2, 3]])
    >>> intersecting = TET4_in_H8(coords_TET4, topology_TET4, coords_H8, topology_H8)
    >>> assert intersecting[0]

    """
    coords_H8 = np.asarray(coords_H8, dtype=float)
    coords_TET4 = np.asarray(coords_TET4, dtype=float)
    topology_H8 = atleast2d(topology_H8, front=True)
    topology_TET4 = atleast2d(topology_TET4, front=True)

    if not coords_H8.shape[1] == 3:
        raise ValueError("Coordinates of H8 cells must have 3 dimensions.")

    if not coords_TET4.shape[1] == 3:
        raise ValueError("Coordinates of TET4 cells must have 3 dimensions.")

    if not topology_H8.shape[1] == 8:
        raise ValueError("Topology of H8 cells must have 8 nodes.")

    if not topology_TET4.shape[1] == 4:
        raise ValueError("Topology of TET4 cells must have 4 nodes.")

    _coords_TET4, _topology_TET4 = H8_to_TET4(coords_H8, topology_H8)
    out = TET4_in_TET4(coords_TET4, topology_TET4, _coords_TET4, _topology_TET4, k=k)
    return out


def TET4_in_T3(
    coords_TET4: ndarray[float],
    topology_TET4: ndarray[int],
    coords_T3: ndarray[float],
    topology_T3: ndarray[int],
    k: int = 10,
) -> ndarray[bool]:
    """
    Performs intersection check between TET4 and T3 cells.

    The function returns a boolean array of length N, where N is the
    number of TET4 cells. The i-th element of the array is True if
    the i-th TET4 cell intersects any of the T3 cells. To narrow
    down the search, the function uses a k nearest neighbours algorithm.

    .. versionadded:: 3.1.0

    Parameters
    ----------
    coords_TET4 : ndarray
        The coordinates of the nodes of the H8 cells.
    topology_TET4 : ndarray
        The topology of the H8 cells.
    coords_T3 : ndarray
        The coordinates of the nodes of the TET4 cells.
    topology_T3 : ndarray
        The topology of the TET4 cells.
    k : int, optional
        The number of nearest neighbours to consider. Default is 10.

    """
    coords_TET4 = np.asarray(coords_TET4, dtype=float)
    coords_T3 = np.asarray(coords_T3, dtype=float)
    topology_TET4 = atleast2d(topology_TET4, front=True)
    topology_T3 = atleast2d(topology_T3, front=True)

    if not coords_T3.shape[1] == 3:
        raise ValueError("Coordinates of T3 cells must have 3 dimensions.")

    if not coords_TET4.shape[1] == 3:
        raise ValueError("Coordinates of TET4 cells must have 3 dimensions.")

    if not topology_TET4.shape[1] == 4:
        raise ValueError("Topology of TET4 cells must have 4 nodes.")

    if not topology_T3.shape[1] == 3:
        raise ValueError("Topology of T3 cells must have 3 nodes.")

    centers_TET4 = center_tri_bulk_3d(coords_TET4, topology_TET4)
    centers_T3 = center_tri_bulk_3d(coords_T3, topology_T3)
    k = min(k, len(centers_T3))
    neighbours = k_nearest_neighbours(centers_T3, centers_TET4, k=k)
    neighbours = atleast2d(neighbours, back=True)
    out = _TET4_in_T3_bulk(
        coords_TET4, topology_TET4, coords_T3, topology_T3, neighbours
    )
    return out


def H8_in_T3(
    coords_H8: ndarray[float],
    topology_H8: ndarray[int],
    coords_T3: ndarray[float],
    topology_T3: ndarray[int],
    k: int = 10,
) -> ndarray[bool]:
    """
    Performs intersection check between H8 and T3 cells.

    The function returns a boolean array of length N, where N is the
    number of H8 cells. The i-th element of the array is True if
    the i-th H8 cell intersects any of the T3 cells. To narrow
    down the search, the function uses a k nearest neighbours algorithm.

    .. versionadded:: 3.1.0

    Parameters
    ----------
    coords_H8 : ndarray
        The coordinates of the nodes of the H8 cells.
    topology_H8 : ndarray
        The topology of the H8 cells.
    coords_T3 : ndarray
        The coordinates of the nodes of the T3 cells.
    topology_T3 : ndarray
        The topology of the T3 cells.
    k : int, optional
        The number of nearest neighbours to consider. Default is 10.

    """
    coords_H8 = np.asarray(coords_H8, dtype=float)
    coords_T3 = np.asarray(coords_T3, dtype=float)
    topology_H8 = atleast2d(topology_H8, front=True)
    topology_T3 = atleast2d(topology_T3, front=True)

    if not coords_T3.shape[1] == 3:
        raise ValueError("Coordinates of T3 cells must have 3 dimensions.")

    if not coords_H8.shape[1] == 3:
        raise ValueError("Coordinates of H8 cells must have 3 dimensions.")

    if not topology_H8.shape[1] == 8:
        raise ValueError("Topology of H8 cells must have 8 nodes.")

    if not topology_T3.shape[1] == 3:
        raise ValueError("Topology of T3 cells must have 3 nodes.")

    _coords_TET4, _topology_TET4 = H8_to_TET4(coords_H8, topology_H8)
    _out = TET4_in_T3(_coords_TET4, _topology_TET4, coords_T3, topology_T3, k=k)
    out = np.any(_out.reshape((len(topology_H8), 5)), axis=1)
    return out
