import numpy as np
from numpy import ndarray
from numba import njit, prange

from sigmaepsilon.math.knn import k_nearest_neighbours
from sigmaepsilon.math.utils import atleast2d

from .tri import center_tri_bulk_3d


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
    >>> from sigmaepsilon.mesh.utils.topology.logical import T3_in_T3
    >>> coordsA = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> trianglesA = np.array([[0, 1, 2]])
    >>> coordsB = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> trianglesB = np.array([[0, 1, 2]])
    >>> intersecting = T3_in_T3(coordsA, trianglesA, coordsB, trianglesB)
    >>> assert intersecting[0]
        
    """
    coordsA = np.asarray(coordsA, dtype=np.float64)
    coordsB = np.asarray(coordsB, dtype=np.float64)
    trianglesA = atleast2d(trianglesA, front=True)
    trianglesB = atleast2d(trianglesB, front=True)
    centersA = center_tri_bulk_3d(coordsA, trianglesA)
    centersB = center_tri_bulk_3d(coordsB, trianglesB)
    k = min(k, len(centersB))
    neighbours = k_nearest_neighbours(centersB, centersA, k=k)
    neighbours = atleast2d(neighbours, back=True)
    out = _T3_in_T3_bulk(coordsA, trianglesA, coordsB, trianglesB, neighbours)
    return out
